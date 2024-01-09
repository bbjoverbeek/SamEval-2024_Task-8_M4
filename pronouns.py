import json
import spacy
import tqdm

nlp = spacy.load("en_core_web_sm")


def proper_names_pronouns_text(text: str) -> tuple[int, int]:
    """
    Count the number of proper named pronouns and named entities in a text.
    :param text: a piece of text
    :return: count of proper pronouns and named entities
    """
    doc = nlp(text.replace('\n', ' '))
    pronouns = []
    named_entities = []

    for token in doc:
        if token.pos_ == 'PRON':
            pronouns.append(token.text)

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            named_entities.append(ent.text)

    return len(pronouns), len(named_entities)


def add_additional_features(filename: str, data: list[dict]) -> None:
    """
    Add additional features to the data.
    :param filename: the name of the file to write the data to
    :param data: the data to add features to
    :return: None
    """
    filename = filename.replace('.jsonl', '_with_pronouns_features.jsonl')
    with open(filename, 'w') as f:
        for sample in tqdm.tqdm(data):
            text = sample['text']
            pronouns, named_entities = proper_names_pronouns_text(text)
            sample['pronouns'] = pronouns
            sample['named_entities'] = named_entities
            json.dump(sample, f)
            f.write('\n')
    return None


def main():
    filename = 'data/SubtaskA/subtaskA_train_monolingual.jsonl'
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]

    add_additional_features(filename, data)


if __name__ == '__main__':
    main()
