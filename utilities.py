import os
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

"""
Classes and functions that are used by multiple files in the project are defined here. 
"""


class Feature(Enum):
    TENSE = "tense"
    VOICE = "voice"
    PRONOUNS = "pronouns"
    NAMED_ENTITIES = "named-entities"
    SENTIMENT = "sentiment"
    POS_TAGS = "pos-tags"
    DEP_TAGS = "dep-tags"
    SENTENCES = "sentences"
    DOMAIN = "domain"
    SENTENCE_SIMILARITY = "sentence-similarity"


class Task(Enum):
    """The tasks that can be trained for."""

    A = "A"
    B = "B"
    C = "C"


Classifier = LinearSVC | KNeighborsClassifier | GaussianNB


@dataclass
class Options:
    """Contains the options for training a model."""

    features: list[Feature]
    vectors_training_dir: str
    normalize_features: bool
    data_dir: str
    task: Task
    model: Literal["nn", "traditional"]
    vectors_test_dir: str = None
    model_dir: str = None
    results_file: str = None
    model_number: int = None
    epochs: int = None
    batch_size: int = None
    learning_rate: float = None
    classifier: str = None


@dataclass
class Score:
    """Contains the scores for a single generator/model."""

    precision: float
    recall: float
    accuracy: float
    f1: float


def load_dataframe(options: Options, split: Literal["train", "dev", "test"]) -> pd.DataFrame:
    """Loads the given dataframe from the given options."""

    additional = "_monolingual" if options.task == Task.A else ""

    path = os.path.join(
        options.data_dir,
        f"Subtask{options.task.value}",
        f"Subtask{options.task.value}_{split}{additional}.jsonl"
    )

    return pd.read_json(path, lines=True)


def get_vectors(features: list[Feature], dir_name: str) -> dict[Feature, Any]:
    """Loads the vectors from the given dataframe."""
    result = {}

    for feature in features:
        filename = os.path.join(dir_name, feature.value, "vectors.npy")
        vector = np.load(filename)
        result[feature] = vector

    return result


def create_vector_matrix(vectors: dict[Feature, Any]) -> np.ndarray:
    """Creates a matrix from the given vectors."""
    return np.concatenate(list(vectors.values()), axis=1)


@dataclass
class Data:
    train_df: pd.DataFrame
    train_matrix: np.ndarray
    test_df: pd.DataFrame = None
    test_matrix: np.ndarray = None


def load_data(options: Options, test_data=True, test_vectors=True) -> Data:
    train_df = load_dataframe(options, "train")
    train_vectors = get_vectors(options.features, options.vectors_training_dir)
    train_matrix = create_vector_matrix(train_vectors)

    test_df, test_matrix = None, None

    if options.normalize_features:
        train_matrix = sklearn.preprocessing.normalize(train_matrix, axis=1, norm="l1")

    if test_data:
        test_df = load_dataframe(options, "dev")

    if test_vectors:
        test_vectors = get_vectors(options.features, options.vectors_test_dir)
        test_matrix = create_vector_matrix(test_vectors)

        if options.normalize_features:
            test_matrix = sklearn.preprocessing.normalize(test_matrix, axis=1, norm="l1")

    return Data(train_df, train_matrix, test_df, test_matrix)
