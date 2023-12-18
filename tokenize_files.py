"""Tokenize the monolingual files in the corpus."""
import os
import pandas as pd
from tqdm import tqdm
import spacy


SPACY_MODEL = "en_core_web_sm"
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    print("Downloading language model for the spaCy tokenizer\n")
    from spacy.cli.download import download

    download(SPACY_MODEL)
    nlp = spacy.load(SPACY_MODEL)


def tokenize_text(df_item: pd.Series) -> pd.Series:
    """Tokenizes the text in a pandas series item."""

    final_text = []

    for token in nlp(df_item.text):
        if token.like_url:
            final_text.append("URL")
        # add more cases here
        else:
            final_text.append(token.text)

    df_item.text = " ".join(final_text)

    return df_item


def tokenize_files(filenames: list[str]) -> None:
    """tokenize the files and write them to the tokenized filepath"""

    tqdm.pandas()

    for filename in tqdm(filenames, desc="Tokenizing files"):
        raw_df = pd.DataFrame(pd.read_json(filename, lines=True))
        tokenized_df = raw_df.progress_apply(tokenize_text, axis=1)

        new_filename = filename.replace("raw", "tokenized")
        os.makedirs("/".join(new_filename.split("/")[0:-1]), exist_ok=True)

        tokenized_df.to_json(
            new_filename,
            orient="records",
            lines=True,
            # default_handler=str
        )


def main():
    """Tokenize all the English files in the raw folder and write them to the tokenized folder."""

    monolingual_filenames = [
        "./data/raw/SubtaskA/subtaskA_train_monolingual.jsonl",
        "./data/raw/SubtaskA/subtaskA_dev_monolingual.jsonl",
        "./data/raw/SubtaskB/subtaskB_train.jsonl",
        "./data/raw/SubtaskB/subtaskB_dev.jsonl",
        "./data/raw/SubtaskC/subtaskC_train.jsonl",
        "./data/raw/SubtaskC/subtaskC_dev.jsonl",
    ]

    tokenize_files(monolingual_filenames)


if __name__ == "__main__":
    main()
