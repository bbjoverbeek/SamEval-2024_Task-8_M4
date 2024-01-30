"""
This script checks whether the results format for subtask A and subtask B is correct. 
It also provides some warnings about possible errors.

The submission of the result file should be in jsonl format. 
It should be a lines of objects:
{
  id     -> identifier of the test sample,
  labels -> labels (0 or 1 for subtask A and from 0 to 5 for subtask B),
}

"""

import os
import argparse
import logging
from typing import Literal
import pandas as pd

# pylint: disable=logging-fstring-interpolation
# pylint: disable=logging-not-lazy


logging.basicConfig(format="%(levelname)s\t%(message)s", level=logging.INFO)
COLUMNS = ["id", "label"]


def parse_args() -> argparse.Namespace:
    """Parses the command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pred_files_path",
        nargs="+",
        required=True,
        help="Path to the files you want to check.",
        type=str,
    )

    parser.add_argument(
        "-t",
        "--subtask",
        choices=["A", "B"],
        required=True,
        help="Subtask A or B",
    )

    return parser.parse_args()


def check_format(file_path: str, subtask: Literal["A", "B"]) -> bool:
    """Checks the format of a prediction file to make sure it conforms to the required format."""

    # check if file exists
    if not os.path.exists(file_path):
        logging.error(f"[{file_path}] ❌ File does not exist.")
        return False

    # check for jsonl format
    try:
        submission = pd.read_json(file_path, lines=True)[["id", "label"]]
    except ValueError:
        logging.error(f"[{file_path}] ❌ File is not a valid json file.")
        return False

    # check for na columns
    for column in COLUMNS:
        if submission[column].isna().any():
            logging.error(f"[{file_path}] ❌ NA value in column {column}")
            return False

    # check if all labels are valid and within range
    if subtask == "A" and not submission["label"].isin(range(0, 2)).all():
        logging.error(f"[{file_path}] ❌ Labels not in range 0-1")
        logging.error(
            f"[{file_path}] Unique Labels in the file are {submission['label'].unique()}"
        )
        return False

    if subtask == "B" and not submission["label"].isin(range(0, 6)).all():
        logging.error(f"[{file_path}] ❌ Labels not in range 0-5")
        logging.error(
            f"[{file_path}] Unique Labels in the file are {submission['label'].unique()}"
        )

        return False

    return True


def main() -> None:
    """Check format of all provided files."""
    args = parse_args()

    logging.info(
        "Checking the following files for formatting on "
        + f"subtask {args.subtask}: {args.pred_files_path}"
    )

    for pred_file_path in args.pred_files_path:
        if check_format(pred_file_path, args.subtask):
            logging.info(f"[{pred_file_path}] ✅ The format is correct.")


if __name__ == "__main__":
    main()
