import json

import spacy
from spacy.matcher import Matcher
import pandas as pd
from spacy.tokens import Doc
from tqdm import tqdm
from transformers import pipeline

nlp = spacy.load('en_core_web_sm')

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device="mps",
    truncation=True,
    max_length=512,
)

# These patterns come from GitHub copilot
TENSE_PATTERNS = {
    "past": [
        [{'TAG': 'VBD'}],  # Simple past tense
        [{'TAG': 'VBN'}, {'DEP': 'auxpass'}],  # Past passive
        [{'LOWER': 'had'}, {'TAG': 'VBN'}],  # Past perfect
        [{'LOWER': 'had'}, {'LOWER': 'been'}, {'TAG': 'VBG'}]  # Past perfect continuous
    ],
    "present": [
        [{'TAG': 'VBZ'}],  # 3rd person singular present
        [{'TAG': 'VBP'}],  # Non-3rd person singular present
        [{'TAG': 'VBG'}, {'DEP': 'aux'}],  # Present participle/gerund
        [{'LOWER': 'has'}, {'TAG': 'VBN'}],  # Present perfect
        [{'LOWER': 'has'}, {'LOWER': 'been'}, {'TAG': 'VBG'}]  # Present perfect continuous
    ],
    "future": [
        [{'LOWER': 'will'}, {'TAG': 'VB'}],  # Future simple
        [{'LOWER': 'is'}, {'LOWER': 'going'}, {'LOWER': 'to'}, {'TAG': 'VB'}],
        # Future with "going to"
        [{'LOWER': 'will'}, {'LOWER': 'be'}, {'TAG': 'VBG'}],  # Future continuous
        [{'LOWER': 'will'}, {'LOWER': 'have'}, {'TAG': 'VBN'}],  # Future perfect
        [{'LOWER': 'will'}, {'LOWER': 'have'}, {'LOWER': 'been'}, {'TAG': 'VBG'}]
        # Future perfect continuous
    ]
}

# These patterns come from stackoverflow
VOICE_PATTERNS = {
    "passive": [
        [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'VBN'}],
        [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'VBZ'}],
        [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'RB'},
         {'TAG': 'VBN'}],
    ],
    "active": [
        [{'DEP': 'nsubj'}, {'TAG': 'VBD', 'DEP': 'ROOT'}],
        [{'DEP': 'nsubj'}, {'TAG': 'VBP'}, {'TAG': 'VBG', 'OP': '!'}],
        [{'DEP': 'nsubj'}, {'DEP': 'aux', 'OP': '*'}, {'TAG': 'VB'}],
        [{'DEP': 'nsubj'}, {'DEP': 'aux', 'OP': '*'}, {'TAG': 'VBG'}],
        [{'DEP': 'nsubj'}, {'TAG': 'RB', 'OP': '*'}, {'TAG': 'VBG'}],
        [{'DEP': 'nsubj'}, {'TAG': 'RB', 'OP': '*'}, {'TAG': 'VBZ'}],
        [{'DEP': 'nsubj'}, {'TAG': 'RB', 'OP': '+'}, {'TAG': 'VBD'}],
    ]
}


def setup() -> tuple[Matcher, Matcher]:
    tense_matcher = Matcher(nlp.vocab)
    voice_matcher = Matcher(nlp.vocab)

    for tense, patterns in TENSE_PATTERNS.items():
        tense_matcher.add(tense, patterns)

    for voice, patterns in VOICE_PATTERNS.items():
        voice_matcher.add(voice, patterns)

    return tense_matcher, voice_matcher


def get_tenses(doc: Doc, tense_matcher: Matcher) -> list[str]:
    matches = tense_matcher(doc)

    tenses = []
    for match_id, start, end in matches:
        if nlp.vocab.strings[match_id] in TENSE_PATTERNS.keys():
            tenses.append(nlp.vocab.strings[match_id])

    return tenses


def get_voices(doc: Doc, voice_matcher) -> list[str]:
    matches = voice_matcher(doc)

    voices = []
    for match_id, start, end in matches:
        if nlp.vocab.strings[match_id] in VOICE_PATTERNS.keys():
            voices.append(nlp.vocab.strings[match_id])

    return voices


def get_count_pronouns_named_entities(doc: Doc) -> tuple[int, int]:
    pronouns = 0
    named_entities = 0

    for token in doc:
        if token.pos_ == 'PRON':
            pronouns += 1

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            named_entities += 1

    return pronouns, named_entities


def get_sentiments(sentences: list[str]) -> list[str]:
    sentiments = []

    for sentence in sentences:
        sentiment = sentiment_pipeline(sentence)[0]
        sentiments.append(sentiment["label"])

    return sentiments


def main():
    df = pd.read_json("data/SubtaskA/subtaskA_train_monolingual.jsonl", lines=True)
    tense_matcher, voice_matcher = setup()

    df = df.sample(n=1600, random_state=42)
    df = df.reset_index(drop=True, inplace=False)

    new_columns = [
        "tenses", "voices", "sentences", "pronouns", "named_entities", "sentiments", "pos-tags",
        "dep-tags"
    ]

    for column in new_columns:
        df[column] = None

    print()

    for index, text in tqdm(enumerate(df["text"]), desc="Processing", total=len(df)):
        doc = nlp(text)
        df.at[index, "pos-tags"] = [token.pos_ for token in doc]
        df.at[index, "dep-tags"] = [token.dep_ for token in doc]
        sentences = [str(sent) for sent in doc.sents]
        df.at[index, "sentences"] = sentences
        df.at[index, "tenses"] = get_tenses(doc, tense_matcher)
        df.at[index, "voices"] = get_voices(doc, voice_matcher)
        df.at[index, "sentiments"] = get_sentiments(sentences)
        pronouns, named_entities = get_count_pronouns_named_entities(doc)
        df.at[index, "pronouns"] = pronouns
        df.at[index, "named_entities"] = named_entities

    df.to_json(
        "data/SubtaskA/subtaskA_train_monolingual_annotations_random_all.json",
        orient="records",
    )


if __name__ == '__main__':
    main()
