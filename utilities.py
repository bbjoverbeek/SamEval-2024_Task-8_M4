from enum import Enum


class Features(Enum):
    TENSE = "tense"
    VOICE = "voice"
    PRONOUNS = "pronouns"
    NAMED_ENTITIES = "named-entities"
    SENTIMENT = "sentiment"
    POS_TAGS = "pos-tags"
    DEP_TAGS = "dep-tags"
    SENTENCES = "sentences"
    DOMAIN = "domain"
