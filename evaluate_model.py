"""
Train and evaluate a model on different splits of the data. 
The task we train on is task A (predict if generator is model or human).
"""

import argparse
from typing import Literal
from dataclasses import dataclass
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

Domain = Literal["wikipedia", "reddit", "wikihow", "peerread", "arxiv"]
Generator = Literal["human", "davinci", "chatGPT", "cohere", "dolly", "bloomz"]


@dataclass
class Score:
    """Contains the scores for a single generator/model."""

    precision: float
    recall: float
    accuracy: float
    f1: float


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments using an argparse parser."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a model on different splits of the data."
    )
    subparsers = parser.add_subparsers(dest="experiment_type")

    # cross domain experiment
    cross_domain_parser = subparsers.add_parser(
        "cross-domain",
        help="Train on one domain and evaluate on all domains.",
    )

    domain_group = cross_domain_parser.add_mutually_exclusive_group(required=True)

    domain_group.add_argument(
        "-d",
        "--train-domain",
        choices=["wikipedia", "reddit", "wikihow", "peerread", "arxiv"],
        help="Domain to train on. Will evaluate on the others.",
    )

    domain_group.add_argument(
        "-a",
        "--all-domains",
        action="store_true",
        help="Train on all domain.",
    )

    cross_domain_parser.add_argument(  # USE WITH CAUTION, DOES NOT WORK AS EXPECTED
        "--generator-filter",
        choices=["davinci", "chatGPT", "cohere", "dolly", "bloomz"],
        help="Generator data to run experiment on. If None, will run on all generators.",
    )

    # cross generator experiment
    cross_generator_parser = subparsers.add_parser(
        "cross-generator",
        help="Train on one generator and evaluate on all generators.",
    )

    generator_group = cross_generator_parser.add_mutually_exclusive_group(required=True)

    generator_group.add_argument(
        "-g",
        "--train-generator",
        choices=["davinci", "chatGPT", "cohere", "dolly", "bloomz"],
        help="Generator to train on. Will evaluate on the others.",
    )

    generator_group.add_argument(
        "-a",
        "--all-generators",
        action="store_true",
        help="Train on all generators.",
    )

    cross_generator_parser.add_argument(  # USE WITH CAUTION, DOES NOT WORK AS EXPECTED
        "--domain-filter",
        choices=["wikipedia", "reddit", "wikihow", "peerread", "arxiv"],
        help="Domain data to run experiment on. If None, will run on all domains.",
    )

    return parser.parse_args()


def load_dataframes(
    task: Literal["A", "B", "C"],
    filter_domain: Literal[Domain | None] = None,
    filter_generator: Literal[Generator | None] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Using the task, load the training and testing dataframes.
    Filters both dataframes on either domain or generator.
    """

    if filter_domain and filter_generator:
        raise ValueError("Cannot select both generator and domain.")

    paths = {
        "A": {
            "train": "./data/tokenized/SubtaskA/subtaskA_train_monolingual.jsonl",
            "test": "./data/tokenized/SubtaskA/subtaskA_dev_monolingual.jsonl",
        },
        "B": {
            "train": "./data/tokenized/SubtaskB/subtaskB_train.jsonl",
            "test": "./data/tokenized/SubtaskB/subtaskB_dev.jsonl",
        },
    }

    try:
        train_df = pd.read_json(paths[task]["train"], lines=True)
        test_df = pd.read_json(paths[task]["test"], lines=True)
    except KeyError as e:
        raise ValueError("Cannot find filepath. Run script from root directory") from e

    # transform the labels to to binary labels (machine vs human) if task B
    if task == "B":
        train_df["label"] = train_df["label"].apply(
            lambda label: 0 if label == 0 else 1
        )
        test_df["label"] = test_df["label"].apply(lambda label: 0 if label == 0 else 1)

    # filter on to contain only domain or only generator
    if filter_generator:
        train_df = train_df[
            (train_df["model"] == filter_generator) | (train_df["model"] == "human")
        ]
        test_df = test_df[
            (test_df["model"] == filter_generator) | (test_df["model"] == "human")
        ]
    elif filter_domain:
        train_df = train_df[train_df["source"] == filter_domain]
        test_df = test_df[test_df["source"] == filter_domain]

    return train_df, test_df


def create_model() -> Pipeline:
    """Create a model to train and evaluate."""
    model = make_pipeline(
        TfidfVectorizer(),
        SVC(),
    )
    return model


def evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_domain: Literal[Domain, None] = None,
    train_generator: Literal[Generator, None] = None,
) -> dict[Literal[Domain | Generator], Score]:
    """Train model on subsection on data and evaluate on the rest."""

    # validate the input arguments
    if train_domain and train_generator:
        raise ValueError("Cannot select both generator and domain.")
    if not train_domain and not train_generator:
        raise ValueError("Provide a value for either generator or domain.")

    # filter train df on selected domain or generator
    if train_domain:
        train_df = train_df[train_df["source"] == train_domain]
    elif train_generator:
        train_df = train_df[
            (train_df["model"] == train_generator) | (train_df["model"] == "human")
        ]

    # create model and fit on training data
    model = create_model()
    model.fit(train_df["text"], train_df["label"])

    # evaluate per generator/model, and remove human from test splits
    test_splits = (
        list(test_df["model"].unique())
        if train_generator
        else list(test_df["source"].unique())
    )
    if "human" in test_splits:
        test_splits.remove("human")

    scores = {}

    for split in test_splits:
        if train_generator:
            test_split_df = test_df[
                (test_df["model"] == split) | (test_df["model"] == "human")
            ]
        else:
            test_split_df = test_df[test_df["source"] == split]

        # make predictions
        predictions = model.predict(test_split_df["text"])

        # compute scores
        accuracy = float(accuracy_score(test_split_df["label"], predictions))
        precision = float(
            precision_score(test_split_df["label"], predictions, zero_division=0)
        )
        recall = float(
            recall_score(test_split_df["label"], predictions, zero_division=0)
        )
        f1 = float(f1_score(test_split_df["label"], predictions, zero_division=0))

        scores[split] = Score(precision, recall, accuracy, f1)

    return scores


def print_scores(
    train: Literal[Domain | Generator],
    scores: dict[Literal[Domain | Generator], Score],
    print_header: bool = False,
) -> None:
    """Prints the scores in a nice format."""
    if print_header:
        tqdm.write(f"{'Train↓':^10}{'Test→':5}|", end="")

        for key in sorted(scores.keys()):
            tqdm.write(f"{key:^40}|", end="")
        tqdm.write("\n", end="")

        tqdm.write(" " * 15 + "|", end="")
        for _ in scores.keys():
            tqdm.write(
                f"{'precision':^11}|{'recall':^8}|{'accuracy':^10}|{'f1':^8}|", end=""
            )
        tqdm.write("\n", end="")

    tqdm.write(f"{train:^15}|", end="")
    for key, score in sorted(scores.items(), key=lambda item: item[0]):
        tqdm.write(
            f"{score.precision*100:^11.2f}|{score.recall*100:^8.2f}|{score.accuracy*100:^10.2f}|{score.f1*100:^8.2f}|",
            end="",
        )

    tqdm.write("\n", end="")


def main():
    """Train and evaluate a model on different splits of the data."""

    args = parse_args()

    if args.experiment_type == "cross-domain":
        # select all domains or just one based on command line arguments
        domains: list[Domain] = (
            [args.train_domain]
            if args.train_domain
            else ["wikipedia", "reddit", "wikihow", "peerread", "arxiv"]
        )
        # load the dataframes
        train_df, test_df = load_dataframes("A", filter_generator=args.generator_filter)

        # run experiment for each domain
        for idx, domain in tqdm(enumerate(domains), total=len(domains), leave=False):
            scores = evaluate_model(
                train_df,
                test_df,
                train_domain=domain,
            )
            # print the scores in a table format (with header on first iteration)
            if idx == 0:
                print_scores(domain, scores, print_header=True)
            else:
                print_scores(domain, scores)
    elif args.experiment_type == "cross-generator":
        # select all generators or just one based on command line arguments
        generators: list[Generator] = (
            [args.train_generator]
            if args.train_generator
            else ["davinci", "chatGPT", "cohere", "dolly", "bloomz"]
        )
        # load the dataframes
        train_df, test_df = load_dataframes("B", filter_domain=args.domain_filter)

        # run the experiment for each generator
        for idx, generator in tqdm(
            enumerate(generators),
            total=len(generators),
            leave=False,
        ):
            scores = evaluate_model(
                train_df,
                test_df,
                train_generator=generator,
            )
            # print the scores in a table format (with header on first iteration)
            if idx == 0:
                print_scores(generator, scores, print_header=True)
            else:
                print_scores(generator, scores)

    else:
        raise ValueError("Invalid experiment type provided.")


if __name__ == "__main__":
    main()
