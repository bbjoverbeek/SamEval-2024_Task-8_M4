import json
import argparse
from tqdm import tqdm
from utilities import Features
from typing import Any, Literal
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from dataclasses import dataclass
from enum import Enum
import numpy as np
import os
import pickle

TOKEN_BASED_FEATURES = {
    Features.DEP_TAGS,
    Features.POS_TAGS,
}

SENTENCE_BASED_FEATURES = {
    Features.SENTIMENT,
    Features.TENSE,
    Features.VOICE,
}

DOCUMENT_BASED_FEATURES = {
    Features.NAMED_ENTITIES,
    Features.PRONOUNS,
    Features.DOMAIN,
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
    vectorizers: dict[Features, CountVectorizer | TfidfVectorizer]


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--vectorizer",
        type=str,
        help="The directory in which you should look for the vectorizer of that option. When applied no new vectorizer is used.",
    )

    parser.add_argument(
        "--features",
        nargs="+",
        choices=[feature.value for feature in Features],
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
    data: dict[int, Any], feature: Features, options: VectorizeOptions
) -> dict[Literal["vectorizer", "vectors"], Any]:
    ngram_range = (options.token_N_grams, options.token_N_grams)
    min_df = 5

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
    data: dict[int, Any], feature: Features, options: VectorizeOptions
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


def vectorize_data(
    data: dict[int, Any], feature: Features, options: VectorizeOptions
) -> dict[Literal["vectorizer", "vectors"], Any]:
    match feature:
        case feature if feature in TOKEN_BASED_FEATURES:
            return vectorize_token_based_features(data, feature, options)
        case feature if feature in SENTENCE_BASED_FEATURES:
            return vectorize_sentence_based_features(data, feature, options)
        case Features.PRONOUNS | Features.NAMED_ENTITIES:
            return {"vectors": np.array([[item] for item in data.values()])}
        case Features.DOMAIN:
            return {"vectors": np.array([[DOMAINS[item]] for item in data.values()])}
        case default:
            return dict()


def save_vector_data(
    result: dict[Literal["vectorizer", "vectors"], Any], output: str, feature: Features
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

    features = [Features(feature) for feature in arguments.features]
    vectorizers = {}

    if arguments.vectorizer:
        for feature in features:
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
