import json
import argparse
import statistics

from keras.src.utils import pad_sequences
from tqdm import tqdm
from utilities import Feature
from typing import Any, Literal
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from dataclasses import dataclass
from enum import Enum
import numpy as np
import os
import pickle

"""
The features that are extracted from the data should be turned into vectors so that classifiers and neural networks can
be trained on them. This file contains the code to vectorize the data.
"""

TOKEN_BASED_FEATURES = {
    Feature.DEP_TAGS,
    Feature.POS_TAGS,
}

SENTENCE_BASED_FEATURES = {
    Feature.SENTIMENT,
    Feature.TENSE,
    Feature.VOICE,
}

FEATURES_WITH_VECTORIZER = {
    Feature.DEP_TAGS,
    Feature.POS_TAGS,
    Feature.SENTIMENT,
    Feature.TENSE,
    Feature.VOICE,
}

DOMAINS = {
    "wikipedia": 1,
    "wikihow": 2,
    "reddit": 3,
    "arxiv": 4,
    "peerread": 5,
}


class Vectorizer(Enum):
    COUNT = "count"
    TFIDF = "tfidf"


@dataclass
class VectorizeOptions:
    token_N_grams: int
    token_vectorizer: Vectorizer
    sentence_N_grams: int
    sentence_vectorizer: Vectorizer
    save_vectors: bool
    vectorizers: dict[Feature, CountVectorizer | TfidfVectorizer]


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--vectorizer",
        type=str,
        help="The directory in which you should look for the vectorizer of that option. When applied no new vectorizer is used.",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        choices=[feature.value for feature in Feature],
        help="The features to extract from the data. Each feature will get a seperate file, where \
        each item is connected to the original data by the id.",
    )

    parser.add_argument(
        "--token-N-grams",
        type=int,
        default=3,
        help="The N in N-grams for token related features. These include POS and DEP tags.",
    )

    parser.add_argument(
        "--token-vectorizer",
        type=str,
        default="count",
        choices=["count", "tfidf"],
        help="The type of vectorizer to use for token based features.",
    )

    parser.add_argument(
        "--sentence-N-grams",
        type=int,
        default=3,
        help="The N in N-grams for sentence related features. These include voice, tense and \
        sentiment.",
    )

    parser.add_argument(
        "--sentence-vectorizer",
        type=str,
        default="count",
        choices=["count", "tfidf"],
        help="The type of vectorizer to use for sentence based features.",
    )

    parser.add_argument(
        "--input", type=str, required=True, help="The name of the input directory"
    )

    parser.add_argument(
        "--output", type=str, required=True, help="The name of the output directory"
    )

    return parser


def vectorize_token_based_features(
        data: dict[int, Any], feature: Feature, options: VectorizeOptions
) -> dict[Literal["vectorizer", "vectors"], Any]:
    ngram_range = (options.token_N_grams, options.token_N_grams)
    min_df = 0.25 if (feature == Feature.POS_TAGS or feature == Feature.DEP_TAGS) else 5

    values = [" ".join(item) for item in data.values()]

    if options.vectorizers == {}:
        vectorizer = (
            CountVectorizer(ngram_range=ngram_range, min_df=min_df)
            if options.token_vectorizer == Vectorizer.COUNT
            else TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
        )
        vector = vectorizer.fit_transform(values).toarray()
    else:
        vectorizer = options.vectorizers[feature]
        vector = vectorizer.transform(values).toarray()

    return {"vectorizer": vectorizer, "vectors": vector}


def vectorize_sentence_based_features(
        data: dict[int, Any], feature: Feature, options: VectorizeOptions
) -> dict[Literal["vectorizer", "vectors"], Any]:
    ngram_range = (options.sentence_N_grams, options.sentence_N_grams)

    values = [" ".join(item) for item in data.values()]

    if options.vectorizers == {}:
        vectorizer = (
            CountVectorizer(ngram_range=ngram_range)
            if options.sentence_vectorizer == Vectorizer.COUNT
            else TfidfVectorizer(ngram_range=ngram_range)
        )
        vector = vectorizer.fit_transform(values).toarray()
    else:
        vectorizer = options.vectorizers[feature]
        vector = vectorizer.transform(values).toarray()

    return {"vectorizer": vectorizer, "vectors": vector}


# def vectorize_sentence_similarity(
#         data: dict[int, list[tuple[float, float]]], options: VectorizeOptions
# ) -> dict[Literal["vectors"], Any]:
#     values = []
#
#     for id, similarities in data.items():
#         min_prev_similarity = 1
#         max_prev_similarity = -1
#         min_next_similarity = 1
#         max_next_similarity = -1
#         mean_prev_similarity = []
#         mean_next_similarity = []
#         amount_next_similarity_higher = 0
#
#         for index, (prev_similarity, next_similarity) in enumerate(similarities):
#             min_prev_similarity = min(min_prev_similarity, prev_similarity)
#             max_prev_similarity = max(max_prev_similarity, prev_similarity)
#             min_next_similarity = min(min_next_similarity, next_similarity)
#             max_next_similarity = max(max_next_similarity, next_similarity)
#             mean_prev_similarity.append(prev_similarity)
#             mean_next_similarity.append(next_similarity)
#
#             if next_similarity > prev_similarity:
#                 amount_next_similarity_higher += 1
#
#         values.append([
#             min_prev_similarity,
#             max_prev_similarity,
#             min_next_similarity,
#             max_next_similarity,
#             statistics.mean(mean_prev_similarity),
#             statistics.mean(mean_next_similarity),
#             amount_next_similarity_higher / len(similarities)
#         ])
#
#     return {"vectors": np.array(values)}


def vectorize_sentence_similarity(
        data: dict[int, list[tuple[float, float]]], options: VectorizeOptions
) -> dict[Literal["vectors"], Any]:
    prev_values = []
    next_values = []

    for _id, similarities in data.items():
        prev_similarities = []
        next_similarities = []

        for index, (prev_similarity, next_similarity) in enumerate(similarities):
            prev_similarities.append(prev_similarity)
            next_similarities.append(next_similarity)

        prev_values.append(prev_similarities)
        next_values.append(next_similarities)

    prev_values = pad_sequences(prev_values, maxlen=20, dtype="float32", padding="post", truncating="post", value=0.0)
    next_values = pad_sequences(next_values, maxlen=20, dtype="float32", padding="post", truncating="post", value=0.0)

    return {"vectors": np.concatenate([prev_values, next_values], axis=1)}


def vectorize_data(
        data: dict[int, Any], feature: Feature, options: VectorizeOptions
) -> dict[Literal["vectorizer", "vectors"], Any]:
    match feature:
        case feature if feature in TOKEN_BASED_FEATURES:
            return vectorize_token_based_features(data, feature, options)
        case feature if feature in SENTENCE_BASED_FEATURES:
            return vectorize_sentence_based_features(data, feature, options)
        case Feature.PRONOUNS | Feature.NAMED_ENTITIES:
            return {"vectors": np.array([[item] for item in data.values()])}
        case Feature.DOMAIN:
            return {"vectors": np.array([[DOMAINS[item]] for item in data.values()])}
        case Feature.SENTENCE_SIMILARITY:
            return vectorize_sentence_similarity(data, options)
        case default:
            return dict()


def save_vector_data(
        result: dict[Literal["vectorizer", "vectors"], Any], output: str, feature: Feature
):
    dirname = os.path.join(output, feature.value)
    os.makedirs(dirname, exist_ok=True)

    vector_filename = os.path.join(dirname, "vectors.npy")
    np.save(vector_filename, result["vectors"])

    if "vectorizer" in result:
        vectorizer_filename = os.path.join(dirname, "vectorizer.pkl")
        with open(vectorizer_filename, "wb") as file:
            pickle.dump(result["vectorizer"], file)


def main():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    arguments = parser.parse_args()

    features = [Feature(feature) for feature in arguments.features]
    vectorizers = {}

    if arguments.vectorizer:
        for feature in FEATURES_WITH_VECTORIZER.intersection(set(features)):
            filename = os.path.join(
                arguments.vectorizer, feature.value, "vectorizer.pkl"
            )

            with open(filename, "rb") as file:
                vectorizer = pickle.load(file)
                vectorizers[feature] = vectorizer

    for feature in tqdm(features, desc="Vectorizing features"):
        with open(f"{arguments.input}/{feature.value}.json", "r") as file:
            data = json.load(file)

        result = vectorize_data(
            data,
            feature,
            VectorizeOptions(
                arguments.token_N_grams,
                Vectorizer(arguments.token_vectorizer),
                arguments.sentence_N_grams,
                Vectorizer(arguments.sentence_vectorizer),
                True,
                vectorizers,
            ),
        )

        save_vector_data(result, arguments.output, feature)


if __name__ == "__main__":
    main()
