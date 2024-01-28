import argparse
import csv

import tensorflow as tf
import numpy as np
import os
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from tqdm.keras import TqdmCallback

from utilities import Feature, Task, Options, load_data, Classifier

"""
While similar to test.py, this file is used to train a model and save the results to a file (while the test.py file will
only return the predictions made by the model). You can use the command line to create the model with different options. 
"""

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.utils.disable_interactive_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a model to detect generated text.'
    )

    parser.add_argument(
        "--features",
        nargs="+",
        choices=[feature.value for feature in Feature],
        help="The features that are used to train the model",
        required=True,
    )

    parser.add_argument(
        "--vectors-training-dir",
        default="vectors/SubtaskA/train_monolingual",
        help="The directory to load the vectorized training data from.",
    )

    parser.add_argument(
        "--vectors-test-dir",
        default="vectors/SubtaskA/dev_monolingual",
        help="The directory to load the vectorized test data from.",
    )

    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="Whether to normalize the features.",
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="The directory to load the data from.",
    )

    parser.add_argument(
        "--task",
        choices=["A", "B", "C"],
        help="The task to train the model for a specific task.",
        required=True,
    )

    parser.add_argument(
        "--model-dir",
        help="The directory to save the model to.",
    )

    parser.add_argument(
        "--results-file",
        help="The file to save the results to.",
        default="results.csv",
    )

    model_parser = parser.add_subparsers(
        dest="model",
        help="The model to train.",
    )

    nn_parser = model_parser.add_parser(
        "nn",
        help="Train a neural network.",
    )

    nn_parser.add_argument(
        "--model-number",
        type=int,
        default=1,
        help="The neural network from the neural networks file to train"
    )

    nn_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs to train the model.",
    )

    nn_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size to train the model.",
    )

    nn_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="The learning rate to train the model.",
    )

    classifier_parser = model_parser.add_parser(
        "traditional",
        help="Train a classifier.",
    )

    classifier_parser.add_argument(
        "--classifier",
        choices=["svm", "knn", "naive-bayes"],
        help="The classifier to train.",
        default="svm",
    )

    return parser.parse_args()


def nn_model(model_number: int, classes: int) -> Sequential:
    """Returns a neural network model."""
    match model_number:
        case 1:
            return Sequential([
                Dense(64, activation="softmax"),
                Dropout(0.5),
                Dense(256, activation="softmax"),
                Dropout(0.5),
                Dense(32, activation="softmax"),
                Dropout(0.5),
                Dense(classes, activation="sigmoid"),
            ])
        case 2:
            return Sequential([
                Dense(128, activation="relu"),
                Dropout(0.5),
                Dense(512, activation="relu"),
                Dropout(0.5),
                Dense(64, activation="relu"),
                Dropout(0.5),
                Dense(classes, activation="relu"),
            ])


def train_nn(
        options: Options,
        train_vectors: np.ndarray,
        train_labels: pd.Series,
) -> Sequential:
    """Trains a neural network."""
    classes = 2 if options.task == Task.A else 6
    model = nn_model(options.model_number, classes)

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=options.learning_rate),
        metrics=["accuracy"],
    )

    y_labels = to_categorical(train_labels, num_classes=classes)

    model.fit(
        train_vectors,
        y_labels,
        epochs=options.epochs,
        batch_size=options.batch_size,
        verbose=0,
        callbacks=[
            TqdmCallback(verbose=2),
        ]
    )

    return model


def train_classifier(
        options: Options,
        train_vectors: np.ndarray,
        train_labels: pd.Series,
) -> Classifier:
    """Trains a classifier."""

    classifier = None

    match options.classifier:
        case "svm":
            classifier = LinearSVC()
        case "knn":
            neighbors = 5 if options.task == Task.A else 15
            classifier = KNeighborsClassifier(n_neighbors=neighbors)
        case "naive-bayes":
            classifier = GaussianNB()

    classifier.fit(train_vectors, train_labels)

    return classifier


def predict_nn(model: Sequential, test_vectors: np.ndarray) -> list[int]:
    """Predicts the labels for the given vectors."""
    predictions = model.predict(test_vectors)
    predictions = np.argmax(predictions, axis=1)
    return predictions


def train_and_run_model(
        options: Options,
        train_vectors: np.ndarray,
        train_labels: pd.Series,
        test_vectors: np.ndarray,
) -> list[int]:
    """Trains the model."""
    match options.model:
        case "nn":
            model = train_nn(options, train_vectors, train_labels)
            return predict_nn(model, test_vectors)
        case "traditional":
            classifier = train_classifier(options, train_vectors, train_labels)
            return classifier.predict(test_vectors)


def save_results(options: Options, scores: list[float], accuracy: float) -> None:
    exists = os.path.isfile(options.results_file)
    features = ["+" if feature in options.features else "-" for feature in Feature]
    [precision, recall, f1, _] = scores

    normalize_features = "+" if options.normalize_features else "-"

    model_options = [
        option if option is not None else "-" for option in [
            options.classifier,
            options.model_number,
            options.epochs,
            options.batch_size,
            options.learning_rate
        ]
    ]

    with open(options.results_file, "a") as file:
        if not exists:
            csv_writer = csv.writer(file)
            header_features = [feature.value for feature in Feature]
            header = [
                "model", "classifier", "nn_number", "epochs", "batch-size", "learning-rate",
                "normalize-features", *header_features, "precision", "recall", "f1", "accuracy"
            ]
            csv_writer.writerow(header)

        csv_writer = csv.writer(file)

        row = [
            options.model,
            *model_options,
            normalize_features,
            *features,
            precision,
            recall,
            f1,
            accuracy
        ]

        csv_writer.writerow(row)


def run(options: Options):
    data = load_data(options)

    predictions = train_and_run_model(options, data.train_matrix, data.train_df["label"], data.test_matrix)
    scores = precision_recall_fscore_support(
        data.test_df["label"], predictions, average="binary" if options.task == Task.A else "macro"
    )
    accuracy = accuracy_score(data.test_df["label"], predictions)

    save_results(options, scores, accuracy)


def main():
    args = parse_args()

    options = Options(
        features=[Feature(feature) for feature in args.features],
        vectors_training_dir=args.vectors_training_dir,
        vectors_test_dir=args.vectors_test_dir,
        normalize_features=args.normalize_features,
        data_dir=args.data_dir,
        task=Task(args.task),
        model_dir=args.model_dir,
        results_file=args.results_file,
        model=args.model,
    )

    if options.model == "nn":
        options.model_number = args.model_number
        options.epochs = args.epochs
        options.batch_size = args.batch_size
        options.learning_rate = args.learning_rate
    else:
        options.classifier = args.classifier

    run(options)


if __name__ == '__main__':
    main()
