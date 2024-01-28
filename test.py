from model import train_and_run_model
from utilities import Feature, Task, Options, load_data
import pandas as pd

"""
Run a create model on the given data. This file is useful for creating the predictions that are submitted to the
shared task.
"""

OPTIONS = Options(
    features=[
        Feature.TENSE,
        Feature.VOICE,
        Feature.PRONOUNS,
        Feature.NAMED_ENTITIES,
        Feature.POS_TAGS,
        Feature.DOMAIN
    ],
    model="nn",
    vectors_training_dir="vectors/SubtaskA/train_monolingual",
    vectors_test_dir="vectors/SubtaskA/dev_monolingual",
    normalize_features=False,
    epochs=8,
    model_number=1,
    batch_size=8,
    learning_rate=0.0005,
    data_dir="data",
    task=Task.A,
)


def main():
    data = load_data(OPTIONS, test_data=False)

    predictions = train_and_run_model(OPTIONS, data.train_matrix, data.train_df["label"], data.test_matrix)

    df = pd.DataFrame([
        {"id": index, "label": prediction} for index, prediction in enumerate(predictions)
    ])
    df.to_json("output_nn.jsonl", lines=True, orient="records")


if __name__ == "__main__":
    main()
