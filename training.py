from enum import Enum

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

sources = {
    "arxiv": 0,
    "wikipedia": 1,
    "reddit": 2,
    "peerread": 3,
    "wikihow": 4,
}


class Feature(Enum):
    TENSES = "tenses"
    VOICES = "voices"
    SENTIMENTS = "sentiments"
    POS_TAGS = "pos-tags"
    DEP_TAGS = "dep-tags"
    TOKENS = "text"
    PRONOUNS = "pronouns"
    NAMED_ENTITIES = "named_entities"
    DOMAIN = "source"


def create_feature_union(features: list[Feature], ngram: tuple[int, int]) -> FeatureUnion:
    union_items = []

    for feature in features:
        match feature:
            case Feature.TENSES | Feature.VOICES | Feature.SENTIMENTS:
                union_items.append((feature.value, CountVectorizer(
                    ngram_range=ngram,
                    preprocessor=lambda x: " ".join(x[feature.value])
                )))
            case Feature.POS_TAGS | Feature.DEP_TAGS:
                union_items.append((feature.value, CountVectorizer(
                    ngram_range=ngram,
                    preprocessor=lambda x: " ".join(x[feature.value])
                )))
            case Feature.TOKENS:
                union_items.append((feature.value, CountVectorizer(
                    ngram_range=ngram,
                    preprocessor=lambda x: str(x[feature.value])
                )))
            case Feature.PRONOUNS | Feature.NAMED_ENTITIES:
                union_items.append((feature.value, FunctionTransformer(
                    lambda items: [[x[feature.value]] for x in items],
                    validate=False
                )))
            case Feature.DOMAIN:
                union_items.append((feature.value, FunctionTransformer(
                    lambda items: [[sources[x["source"]]] for x in items], validate=False
                )))

    return FeatureUnion(union_items)


def main():
    df = pd.read_json("data/SubtaskA/subtaskA_train_monolingual_annotations_random_all.json")

    classifier = SVC(kernel="linear", C=1, probability=True)
    ngram = (3, 3)
    features = [
        Feature.DOMAIN, Feature.SENTIMENTS, Feature.VOICES, Feature.POS_TAGS, Feature.DEP_TAGS
    ]
    union = create_feature_union(features, ngram)

    pipeline = Pipeline([
        ("features", union),
        ("classifier", classifier),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        df.to_dict(orient="records"), df["label"],
        test_size=0.2, shuffle=True, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_report_str = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:\n", classification_report_str)


if __name__ == '__main__':
    main()
