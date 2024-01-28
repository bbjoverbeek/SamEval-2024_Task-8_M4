import itertools
from typing import Literal
import requests

from model import run
from utilities import Feature, Task, Options

"""
To test the different set of features, classifiers and neural network options, we will run this file that will produce
scores for all different combinations. The results will be saved in a CSV file. Here you can easily filter the data 
based on features and filter them on scores.
"""

TOPIC = "gurzQZYSKDVkpMej1BR7IN6sKMPPzd36BDRzKBYJWtH4zP8Mpldt1I4AWWRHA"
TASK = Task.A
HEAD = None
NOTIFY = True
FEATURES = [
    Feature.TENSE,
    Feature.VOICE,
    Feature.PRONOUNS,
    Feature.NAMED_ENTITIES,
    Feature.SENTIMENT,
    Feature.POS_TAGS,
    Feature.DOMAIN
]


def create_vector_dir_name(task: Task, set_name: Literal["train", "dev"]) -> str:
    return f"vectors/Subtask{task.value}/{set_name}_monolingual"


def create_options(
        model: Literal["nn", "traditional"],
        task: Task,
        other_options: tuple[int, int, float, int, list[Feature]] | tuple[str, list[Feature]]
) -> Options:
    features = other_options[-1]

    options = Options(
        features=features,
        model=model,
        vectors_training_dir=create_vector_dir_name(task, "train"),
        vectors_test_dir=create_vector_dir_name(task, "dev"),
        normalize_features=False,
        data_dir="data",
        task=task,
        model_dir=f"models/{task}/{model}",
        results_file=f"results_{task.value}.csv",
    )

    match model:
        case "nn":
            epochs, batch_size, learning_rate, model_number = other_options[0:4]
            options.epochs = epochs
            options.batch_size = batch_size
            options.learning_rate = learning_rate
            options.model_number = model_number
        case "traditional":
            options.classifier = other_options[0]

    return options


def create_feature_combinations() -> list[list[Feature]]:
    status = [True, False]

    feature_combinations = list(map(
        lambda x: [FEATURES[index] for index, value in enumerate(x) if value],
        itertools.product(status, repeat=len(FEATURES))
    ))

    return feature_combinations


def create_classifier_combinations() -> list[tuple[str, list[Feature]]]:
    classifiers = ["svm", "knn", "naive-bayes"]
    feature_combinations = create_feature_combinations()

    classifier_combinations = list(itertools.product(classifiers, feature_combinations))

    return classifier_combinations


def create_nn_combinations() -> list[tuple[int, int, float, int, list[Feature]]]:
    epochs = [4, 8, 16, 32]
    batch_sizes = [8, 16, 32, 64]
    learning_rates = [0.0005, 0.001, 0.005]
    model_numbers = [1, 2]
    feature_combinations = create_feature_combinations()

    nn_combinations = list(itertools.product(epochs, batch_sizes, learning_rates, model_numbers, feature_combinations))

    return nn_combinations


def run_combinations(model: Literal["nn", "traditional"], combinations: list) -> None:
    if HEAD is not None:
        combinations = combinations[:HEAD]

    for index, combinations in enumerate(combinations):
        options = create_options("traditional", TASK, combinations)

        if index % 50 == 0 and NOTIFY:
            requests.post(
                f"https://ntfy.sh/{TOPIC}",
                data=f"Currently running {index + 1}/{len(combinations)} {model} combinations.".encode(
                    "utf-8"
                )
            )

        run(options)


def main():
    classifier_combinations = create_classifier_combinations()
    run_combinations("traditional", classifier_combinations)
    nn_combinations = create_nn_combinations()
    run_combinations("nn", nn_combinations)


if __name__ == "__main__":
    main()
