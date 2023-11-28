import sys
import json
import spacy
from spacy.matcher import Matcher
from nltk.tokenize import sent_tokenize


def initialize_matcher():
    """This function initializes the matcher that is used
    for finding sentences written in passive voice."""
    nlp = spacy.load('en_core_web_lg')
    matcher = Matcher(nlp.vocab)
    passive_rule = [{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'VBN'}]
    matcher.add('Passive', [passive_rule])

    return nlp, matcher


def detect_passive(nlp, matcher, sentence):
    """This function detects whether a sentence is written
    in passive voice or not."""
    doc = nlp(sentence)
    match = matcher(doc)

    if match:
        return True
    else:
        return False


def write_to_jsonl(data, sents_voices):
    """This function adds the enriched data to a jsonl file."""
    new_data = {"sentence_voices": sents_voices,
                "text": data['text'],
                "id": data['id'],
                "label": data['label'],
                "model": data['model'],
                "source": data['source']}

    with open('data/SubtaskA/subtaskA_train_monolingual_voice.jsonl', 'a', encoding='utf8') as f:
        print(f"Writing to json file - Current id: {new_data['id']}")
        f.write(json.dumps(new_data) + '\n')


def main(argv):
    nlp, matcher = initialize_matcher()

    with open(argv[1], 'r', encoding='utf8') as f:
        for line in f:
            data = json.loads(line)
            sents = sent_tokenize(data['text'])    # tokenize on sentence level
            sents_voices = dict()
            sent_id = 0

            for sent in sents:
                if detect_passive(nlp, matcher, sent):
                    sents_voices[sent_id] = [sent, 'passive']
                else:
                    sents_voices[sent_id] = [sent, 'active']
                sent_id += 1

            write_to_jsonl(data, sents_voices)


if __name__ == "__main__":
    main(sys.argv)
