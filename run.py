import itertools
from typing import Literal
import requests
import warnings
from tqdm import tqdm

from model import Options, Task, run
from utilities import Features


warnings.filterwarnings('ignore')

def create_vector_dir_name(task: Task, set_name: Literal["train", "dev"]) -> str:
    return f"vectors/Subtask{task.value}/{set_name}_monolingual"


def create_options(
        model: Literal["nn", "traditional"],
        task: Task,
        other_options: tuple[int, int, float, int, list[Features]] | tuple[str, list[Features]]
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
        results_file=f"results.csv",
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


def create_feature_combinations() -> list[list[Features]]:
    features = [
        Features.TENSE,
        Features.VOICE,
        Features.PRONOUNS,
        Features.NAMED_ENTITIES,
        Features.SENTIMENT,
        Features.POS_TAGS,
        Features.DOMAIN
    ]
    status = [True, False]

    feature_combinations = list(map(
        lambda x: [features[index] for index, value in enumerate(x) if value],
        itertools.product(status, repeat=len(features))
    ))

    return feature_combinations


def create_classifier_combinations() -> list[tuple[str, list[Features]]]:
    classifiers = ["svm", "knn", "naive-bayes"]
    feature_combinations = create_feature_combinations()

    classifier_combinations = list(itertools.product(classifiers, feature_combinations))

    return classifier_combinations


def create_nn_combinations() -> list[tuple[int, int, float, int, list[Features]]]:
    epochs = [4, 8, 16, 32]
    batch_sizes = [8, 16, 32, 64]
    learning_rates = [0.0005, 0.001, 0.005]
    model_numbers = [1, 2]
    feature_combinations = create_feature_combinations()

    nn_combinations = list(itertools.product(epochs, batch_sizes, learning_rates, model_numbers, feature_combinations))

    return nn_combinations


def main():
    topic = "gurzQZYSKDVkpMej1BR7IN6sKMPPzd36BDRzKBYJWtH4zP8Mpldt1I4AWWRHA"

    classifier_combinations = create_classifier_combinations()
    for index, combinations in tqdm(enumerate(classifier_combinations), desc="Training Classifiers", total=len(classifier_combinations)):
        options = create_options("traditional", Task.A, combinations)

        if index % 10 == 0:
            requests.post(
                f"https://ntfy.sh/{topic}",
                data=f"Currently running {index + 1}/{len(classifier_combinations)} classifier combinations.".encode(
                    "utf-8"
                )
            )

        run(options)

    nn_combinations = create_nn_combinations()

    for index, combinations in tqdm(enumerate(nn_combinations), desc="Training Neural Networks", total=len(nn_combinations)):
        options = create_options("nn", Task.A, combinations)

        if index % 10 == 0:
            requests.post(
                f"https://ntfy.sh/{topic}",
                data=f"Currently running {index + 1}/{len(nn_combinations)} nn combinations.".encode(
                    "utf-8"
                )
            )

        run(options)

if __name__ == "__main__":
    main()
