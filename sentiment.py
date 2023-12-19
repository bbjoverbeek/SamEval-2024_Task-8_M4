"""Test sentiment as a feature for SubTask A"""
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
import pandas as pd


def load_dataframes(
    train_path: str, test_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads the training and development dataframes from the given paths"""

    # load train file
    if train_path.endswith(".jsonl"):
        train_df = pd.read_json(train_path, lines=True)
    elif train_path.endswith(".pickle"):
        with open(train_path, "rb") as inp:
            train_df = pickle.load(inp)
    else:
        raise ValueError("Invalid file type for train_path")

    # load dev file
    if test_path.endswith(".jsonl"):
        test_df = pd.read_json(test_path, lines=True)
    elif test_path.endswith(".pickle"):
        with open(test_path, "rb") as inp:
            test_df = pickle.load(inp)
    else:
        raise ValueError("Invalid file type for dev_path")

    return train_df, test_df


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the sentiment score to the dataframe"""
    tqdm.pandas()
    sia = SentimentIntensityAnalyzer()
    df["sentiment"] = df["text"].progress_apply(
        lambda text: sia.polarity_scores(text)["compound"]
    )
    return df


def create_model() -> Pipeline:
    """Create a model to train and evaluate."""
    model = make_pipeline(
        SVC(),
    )
    return model


def plot_sentiment(df: pd.DataFrame) -> None:
    """Plots the sentiment of the dataframes."""
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # plt.figure(figsize=(8, 6))
    # sns.boxplot(
    #     x="model", y="sentiment", data=df, showfliers=False
    # )  # ci=None removes error bars
    # plt.xlabel("Category")
    # plt.ylabel("Average Sentiment")
    # plt.title("Average Sentiment by Category")
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # sns.boxplot(
    #     x="source", y="sentiment", data=df, showfliers=False
    # )  # ci=None removes error bars
    # plt.xlabel("Category")
    # plt.ylabel("Average Sentiment")
    # plt.title("Average Sentiment by Source")
    # plt.show()
    pass


def main():
    """Loads the dataframes and trains the model, and prints the classification report."""

    # load files
    train_df, test_df = load_dataframes(
        train_path="./data/sentiment_tokenized_1000.pickle",
        test_path="./data/dev_sentiment_tokenized.pickle",
    )

    # add sentiment to dataframes if not present
    if "sentiment" not in train_df.columns:
        train_df = add_sentiment(train_df)
    if "sentiment" not in test_df.columns:
        test_df = add_sentiment(test_df)

    # create model (pipeline)
    model = create_model()

    # train model
    model.fit([[value] for value in train_df["sentiment"]], train_df["label"])

    y_pred = model.predict([[value] for value in test_df["sentiment"]])
    y_true = test_df["label"]

    # print classification report
    result = classification_report(y_true, y_pred, zero_division=0)
    print("Classification Report:\n", result)


if __name__ == "__main__":
    main()
