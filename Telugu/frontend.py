"""Telugu text -> phoneme sequence (used by inference.py)."""
import re

import numpy as np

from text import text_to_sequence
from text import cleaners
from text.telugu import word_to_phones

_PUNCT = ",.;:!?-"


def text_to_phones(text, preprocess_config):
    """Returns (phoneme string, numpy array of symbol ids)."""
    clean = cleaners.telugu_cleaners(text)
    tokens = re.split(r"([%s\s])" % re.escape(_PUNCT), clean)
    phones = []
    for tok in tokens:
        if not tok or tok.isspace():
            continue
        if tok in _PUNCT:
            phones.append("sp")          # punctuation = short pause
        else:
            phones += word_to_phones(tok)
    # drop a pause at the very end (the original repo strips final punctuation too)
    while phones and phones[-1] == "sp":
        phones.pop()
    phone_str = "{" + " ".join(phones) + "}"

    print("Raw text      :", text)
    print("Normalised    :", clean)
    print("Phoneme string:", phone_str)
    seq = text_to_sequence(phone_str, preprocess_config["preprocessing"]["text"]["text_cleaners"])
    return phone_str, np.array(seq)
