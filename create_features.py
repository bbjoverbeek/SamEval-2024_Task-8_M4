from enum import Enum
from typing import Any, Optional
import spacy
import argparse
import json

from sentence_transformers import SentenceTransformer
import sentence_transformers
from spacy.matcher import Matcher
from spacy.tokens import Doc
from transformers import pipeline, Pipeline
from tqdm import tqdm
import os
from utilities import Features

"""
From the data given by the shared task, we can extract multiple features: token, sentence and document based features.
This file contains the code to extract these features from the data. Also there are command line options to specify
which features you want to extract. To vectorize the features and with what options you want to vectorize them, you
should use the vectorize.py file.
"""

HUGGINGFACE_DEVICE = "mps"  # should be changed when running on a different device
HUGGINGFACE_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
HUGGINGFACE_SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SPACY_FEATURES = {
    Features.PRONOUNS,
    Features.NAMED_ENTITIES,
    Features.POS_TAGS,
    Features.DEP_TAGS,
    Features.SENTENCES,
    Features.TENSE,
    Features.VOICE,
}

HUGGINGFACE_FEATURES = {Features.SENTIMENT}

DATA_FEATURES = {Features.DOMAIN}

nlp = spacy.load("en_core_web_sm")

# These patterns come from GitHub copilot
TENSE_PATTERNS = {
    "past": [
        [{"TAG": "VBD"}],  # Simple past tense
        [{"TAG": "VBN"}, {"DEP": "auxpass"}],  # Past passive
        [{"LOWER": "had"}, {"TAG": "VBN"}],  # Past perfect
        [{"LOWER": "had"}, {"LOWER": "been"}, {"TAG": "VBG"}],  # Past perfect continuous
    ],
    "present": [
        [{"TAG": "VBZ"}],  # 3rd person singular present
        [{"TAG": "VBP"}],  # Non-3rd person singular present
        [{"TAG": "VBG"}, {"DEP": "aux"}],  # Present participle/gerund
        [{"LOWER": "has"}, {"TAG": "VBN"}],  # Present perfect
        [{"LOWER": "has"}, {"LOWER": "been"}, {"TAG": "VBG"}],  # Present perfect continuous
    ],
    "future": [
        [{"LOWER": "will"}, {"TAG": "VB"}],  # Future simple
        [{"LOWER": "is"}, {"LOWER": "going"}, {"LOWER": "to"}, {"TAG": "VB"}],
        # Future with "going to"
        [{"LOWER": "will"}, {"LOWER": "be"}, {"TAG": "VBG"}],  # Future continuous
        [{"LOWER": "will"}, {"LOWER": "have"}, {"TAG": "VBN"}],  # Future perfect
        [{"LOWER": "will"}, {"LOWER": "have"}, {"LOWER": "been"}, {"TAG": "VBG"}]
        # Future perfect continuous
    ],
}

# These patterns come from stackoverflow
VOICE_PATTERNS = {
    "passive": [
        [{"DEP": "nsubjpass"}, {"DEP": "aux", "OP": "*"}, {"DEP": "auxpass"}, {"TAG": "VBN"}],
        [{"DEP": "nsubjpass"}, {"DEP": "aux", "OP": "*"}, {"DEP": "auxpass"}, {"TAG": "VBZ"}],
        [
            {"DEP": "nsubjpass"},
            {"DEP": "aux", "OP": "*"},
            {"DEP": "auxpass"},
            {"TAG": "RB"},
            {"TAG": "VBN"},
        ],
    ],
    "active": [
        [{"DEP": "nsubj"}, {"TAG": "VBD", "DEP": "ROOT"}],
        [{"DEP": "nsubj"}, {"TAG": "VBP"}, {"TAG": "VBG", "OP": "!"}],
        [{"DEP": "nsubj"}, {"DEP": "aux", "OP": "*"}, {"TAG": "VB"}],
        [{"DEP": "nsubj"}, {"DEP": "aux", "OP": "*"}, {"TAG": "VBG"}],
        [{"DEP": "nsubj"}, {"TAG": "RB", "OP": "*"}, {"TAG": "VBG"}],
        [{"DEP": "nsubj"}, {"TAG": "RB", "OP": "*"}, {"TAG": "VBZ"}],
        [{"DEP": "nsubj"}, {"TAG": "RB", "OP": "+"}, {"TAG": "VBD"}],
    ],
}


def setup_matchers() -> tuple[Matcher, Matcher]:
    """
    Create the matchers for tense and voice.
    """
    tense_matcher = Matcher(nlp.vocab)
    voice_matcher = Matcher(nlp.vocab)

    for tense, patterns in TENSE_PATTERNS.items():
        tense_matcher.add(tense, patterns)

    for voice, patterns in VOICE_PATTERNS.items():
        voice_matcher.add(voice, patterns)

    return tense_matcher, voice_matcher


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--features",
        nargs="+",
        choices=[feature.value for feature in Features],
        help="The features to extract from the data. Each feature will get a seperate file, where \
        each item is connected to the original data by the id.",
    )

    parser.add_argument("--input", type=str, required=True, help="The name of the input file")

    parser.add_argument("--sentences", action="store_true", help="Whether to tokenize the text")

    parser.add_argument(
        "--output", type=str, required=True, help="The name of the output directory"
    )

    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="The number of items to process. Mainly for debugging purposes.",
    )

    return parser


def get_matches(doc: Doc, matcher: Matcher, patterns: dict[str, list]) -> list[str]:
    matches = matcher(doc)

    items = []
    for match_id, start, end in matches:
        if nlp.vocab.strings[match_id] in patterns.keys():
            items.append(nlp.vocab.strings[match_id])

    return items


def create_spacy_features(
        item: dict,
        features: list[Features],
        data_features: dict[Features, dict[int, Any]],
        tense_matcher: Matcher,
        voice_matcher: Matcher,
) -> None:
    """
    Given a list of features, extract the features from the text and add them to the data_features.
    All the features that are added are extracted using spacy.

    :param data_features: The dictionary that contains all the features. The dictionary has the
    following structure: { feature: { id: feature_value(s) }}.
    """
    doc = nlp(item["text"])
    id = item["id"]

    if Features.TENSE in features:
        data_features[Features.TENSE][id] = get_matches(doc, tense_matcher, TENSE_PATTERNS)

    if Features.VOICE in features:
        data_features[Features.VOICE][id] = get_matches(doc, voice_matcher, VOICE_PATTERNS)

    for feature in features:
        match feature:
            case Features.PRONOUNS:
                data_features[Features.PRONOUNS][id] = create_pronouns_feature(doc)
            case Features.NAMED_ENTITIES:
                data_features[Features.NAMED_ENTITIES][id] = create_named_entities_feature(doc)
            case Features.POS_TAGS:
                data_features[Features.POS_TAGS][id] = [token.pos_ for token in doc]
            case Features.DEP_TAGS:
                data_features[Features.DEP_TAGS][id] = [token.dep_ for token in doc]
            case Features.SENTENCES:
                data_features[Features.SENTENCES][id] = [str(sent) for sent in doc.sents]


def create_features(
        features: list[Features],
        data: list[dict],
        voice_matcher: Matcher,
        tense_matcher: Matcher,
        sentiment_pipeline: Optional[Pipeline],
        similarity_model: Optional[SentenceTransformer],
        tokenized_sentences: Optional[dict[int, Any]] = None,
) -> dict[Features, dict[int, Any]]:
    # the sentiment feature needs to have individual sentences, which are tokenized using spacy
    if (
            (Features.SENTIMENT in features or Features.SENTENCE_SIMILARITY)
            and Features.SENTENCES not in features
            and tokenized_sentences is None
    ):
        features.append(Features.SENTENCES)

    # for each feature create a dict with the id as key and the features as value
    data_features: dict[Features, dict[int, Any]] = {feature: {} for feature in features}

    if tokenized_sentences:
        data_features[Features.SENTENCES] = tokenized_sentences

    if len(SPACY_FEATURES.intersection(features)) != 0:
        for item in tqdm(data, desc="Extracting spacy features"):
            create_spacy_features(item, features, data_features, tense_matcher, voice_matcher)

    if Features.DOMAIN in features:
        data_features[Features.DOMAIN] = {item["id"]: item["source"] for item in data}

    if Features.SENTIMENT in features and sentiment_pipeline is not None:
        for id, sentences in tqdm(
                data_features[Features.SENTENCES].items(),
                desc="Extracting huggingface sentiment feature",
        ):
            data_features[Features.SENTIMENT][id] = [
                result["label"] for result in sentiment_pipeline(sentences)
            ]

    if Features.SENTENCE_SIMILARITY in features and similarity_model is not None:
        for (id, sentences) in tqdm(
                data_features[Features.SENTENCES].items(),
                desc="Extracting huggingface sentence similarity feature",
        ):
            similarities: list[tuple[float, float]] = []
            embeddings = []

            for sentence in sentences:
                similarity = similarity_model.encode(sentence, convert_to_tensor=True)
                embeddings.append(similarity)

            for index, embedding in enumerate(embeddings):
                similarity_previous = sentence_transformers.util.pytorch_cos_sim(
                    embedding, embeddings[index - 1]
                ).item() if index > 0 else 0

                similarity_next = sentence_transformers.util.pytorch_cos_sim(
                    embedding, embeddings[index + 1]
                ).item() if index < len(embeddings) - 1 else 0

                similarities.append((similarity_previous, similarity_next))

            data_features[Features.SENTENCE_SIMILARITY][id] = similarities

    return data_features


def create_pronouns_feature(doc: Doc) -> int:
    pronouns = 0

    for token in doc:
        if token.pos_ == "PRON":
            pronouns += 1

    return pronouns


def create_named_entities_feature(doc: Doc) -> int:
    named_entities = 0

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            named_entities += 1

    return named_entities


def main():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    arguments = parser.parse_args()

    features_args = [Features(feature) for feature in arguments.features]

    sentiment_pipeline = (
        pipeline(
            "sentiment-analysis",
            model=HUGGINGFACE_SENTIMENT_MODEL,
            device=HUGGINGFACE_DEVICE,
            truncation=True,
            max_length=512,
        )
        if Features.SENTIMENT in features_args
        else None
    )

    similarity_model = SentenceTransformer(
        HUGGINGFACE_SIMILARITY_MODEL, device=HUGGINGFACE_DEVICE
    ) if Features.SENTENCE_SIMILARITY in features_args \
        else None

    tense_matcher, voice_matcher = setup_matchers()

    with open(arguments.input, "r") as f:
        data = [json.loads(line) for line in f]
        if arguments.head is not None:
            data = data[: arguments.head]

    sentences = None
    if arguments.sentences:
        filename = arguments.output + "/sentences.json"
        with open(filename, "r") as f:
            sentences = json.load(f)
            if arguments.head is not None:
                sentences = dict(list(sentences.items())[: arguments.head])

    features = create_features(
        features_args, data, voice_matcher, tense_matcher, sentiment_pipeline, similarity_model, sentences
    )

    for feature, values in features.items():
        if feature == Features.SENTENCES and sentences is not None:
            continue
        filename = f"{arguments.output}/{feature.value}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(values, f)


if __name__ == "__main__":
    main()
