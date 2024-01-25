import argparse
import csv
from enum import Enum
from typing import Literal, Any

import sklearn.preprocessing
import tensorflow as tf
import numpy as np
import os
from dataclasses import dataclass
import pandas as pd
from keras.src.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from tqdm.keras import TqdmCallback
from neural_networks import nn_model

from utilities import Features

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.utils.disable_interactive_logging()


class Task(Enum):
    """The tasks that can be trained."""

    A = "A"
    B = "B"
    C = "C"


@dataclass
class Options:
    """Contains the options for training a model."""

    features: list[Features]
    vectors_training_dir: str
    vectors_test_dir: str
    normalize_features: bool
    data_dir: str
    task: Task
    model_dir: str
    results_file: str
    model: Literal["nn", "traditional"]
    model_number: int = None
    epochs: int = None
    batch_size: int = None
    learning_rate: float = None
    classifier: str = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a model to detect generated text.'
    )

    parser.add_argument(
        "--features",
        nargs="+",
        choices=[feature.value for feature in Features],
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
        choices=["svm", "logistic-regression", "knn", "naive-bayes"],
        help="The classifier to train.",
        default="svm",
    )

    return parser.parse_args()


@dataclass
class Score:
    """Contains the scores for a single generator/model."""

    precision: float
    recall: float
    accuracy: float
    f1: float


def load_dataframe(options: Options, split: Literal["train", "dev", "test"]) -> pd.DataFrame:
    """Loads the given dataframe from the given options."""
    path = os.path.join(
        options.data_dir,
        f"Subtask{options.task.value}",
        f"Subtask{options.task.value}_{split}_monolingual.jsonl"
    )

    return pd.read_json(path, lines=True)


def get_vectors(features: list[Features], dir: str) -> dict[Features, Any]:
    """Loads the vectors from the given dataframe."""
    result = {}

    for feature in features:
        filename = os.path.join(dir, feature.value, "vectors.npy")
        vector = np.load(filename)
        result[feature] = vector

    return result


def create_vector_matrix(vectors: dict[Features, Any]) -> np.ndarray:
    """Creates a matrix from the given vectors."""
    return np.concatenate(list(vectors.values()), axis=1)


def train_and_run_nn(
        options: Options,
        train_vectors: np.ndarray,
        train_labels: pd.Series,
        test_vectors: np.ndarray,
) -> list[str]:
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

    predictions = model.predict(test_vectors)
    predictions = np.argmax(predictions, axis=1)

    return predictions


def train_and_run_classifier(
        options: Options,
        train_vectors: np.ndarray,
        train_labels: pd.Series,
        test_vectors: np.ndarray,
) -> list[str]:
    """Trains a classifier."""

    classifier = None

    match options.classifier:
        case "svm":
            classifier = LinearSVC()
        case "logistic-regression":
            classifier = LogisticRegression()
        case "knn":
            classifier = KNeighborsClassifier()
        case "naive-bayes":
            classifier = GaussianNB()

    classifier.fit(train_vectors, train_labels)
    predictions = classifier.predict(test_vectors)
    return predictions


def train_and_run_model(
        options: Options,
        train_vectors: np.ndarray,
        train_labels: pd.Series,
        test_vectors: np.ndarray,
) -> list[str]:
    """Trains the model."""
    match options.model:
        case "nn":
            return train_and_run_nn(options, train_vectors, train_labels, test_vectors)
        case "traditional":
            return train_and_run_classifier(options, train_vectors, train_labels, test_vectors)


def save_results(options: Options, scores: list[float], accuracy: float) -> None:
    exists = os.path.isfile(options.results_file)
    features = ["+" if feature in options.features else "-" for feature in Features]
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
            header_features = [feature.value for feature in Features]
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
    train_df = load_dataframe(options, "train")
    test_df = load_dataframe(options, "dev")

    train_vectors = get_vectors(options.features, options.vectors_training_dir)
    test_vectors = get_vectors(options.features, options.vectors_test_dir)

    train_matrix = create_vector_matrix(train_vectors)
    test_matrix = create_vector_matrix(test_vectors)

    if options.normalize_features:
        train_matrix = sklearn.preprocessing.normalize(train_matrix, axis=1, norm="l1")
        test_matrix = sklearn.preprocessing.normalize(test_matrix, axis=1, norm="l1")

    predictions = train_and_run_model(options, train_matrix, train_df["label"], test_matrix)
    scores = precision_recall_fscore_support(
        test_df["label"], predictions, average="binary" if options.task == Task.A else "macro"
    )
    accuracy = accuracy_score(test_df["label"], predictions)

    save_results(options, scores, accuracy)


def main():
    args = parse_args()

    options = Options(
        features=[Features(feature) for feature in args.features],
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
