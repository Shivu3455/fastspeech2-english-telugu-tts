"""English text -> phoneme sequence (used by inference.py).

Changes compared with the original synthesize.py:
  * numbers, times and dates are expanded to words (E4)
  * simple context rule for the heteronym 'conduct' (noun vs verb, E2a / E2b)
  * only the CMU/LibriSpeech lexicon + g2p_en fallback (Mandarin removed)
"""
import re
from string import punctuation

import numpy as np
from g2p_en import G2p

from text import text_to_sequence

_g2p = None
_lexicon_cache = {}

MONTHS = ("January|February|March|April|May|June|July|August|September|"
          "October|November|December")

# 'conduct' is stressed on the first syllable as a noun, on the second as a verb
CONDUCT_NOUN = ["K", "AA1", "N", "D", "AH0", "K", "T"]
CONDUCT_VERB = ["K", "AH0", "N", "D", "AH1", "K", "T"]
VERB_TRIGGERS = {"will", "to", "can", "could", "would", "should", "may", "might",
                 "must", "shall", "they", "we", "you", "i", "who", "not", "also"}


def read_lexicon(lex_path):
    if lex_path in _lexicon_cache:
        return _lexicon_cache[lex_path]
    lexicon = {}
    with open(lex_path, encoding="utf-8") as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word, phones = temp[0].lower(), temp[1:]
            if word not in lexicon:
                lexicon[word] = phones
    _lexicon_cache[lex_path] = lexicon
    return lexicon


_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _words(n):
    """Integer -> English words (0 .. 999,999,999)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + _words(rest))
    for size, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= size:
            rest = n % size
            return _words(n // size) + " " + name + ("" if rest == 0 else " " + _words(rest))


def _year_words(y):
    if 2000 <= y <= 2009:
        return _words(y)
    hi, lo = divmod(y, 100)
    if lo == 0:
        return _words(hi) + " hundred"
    return _words(hi) + " " + ("oh " + _words(lo) if lo < 10 else _words(lo))


def normalize_english(text):
    """10:45 AM -> 'ten forty five ay em', 12 November 2026 -> 'twelve November twenty twenty six'."""
    def time_sub(m):
        h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
        out = _words(h)
        if mi == 0:
            out += " o'clock"
        elif mi < 10:
            out += " oh " + _words(mi)
        else:
            out += " " + _words(mi)
        if ap:
            out += " ay em" if ap.lower().startswith("a") else " pee em"
        return out

    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?", time_sub, text)
    text = re.sub(
        r"\b(\d{1,2})\s+(%s)\s+(\d{4})\b" % MONTHS,
        lambda m: "{} {} {}".format(_words(int(m.group(1))), m.group(2), _year_words(int(m.group(3)))),
        text,
    )
    text = re.sub(r"\b(\d+)\b", lambda m: _words(int(m.group(1))), text)
    text = re.sub(r"\bAM\b", "ay em", text)
    text = re.sub(r"\bPM\b", "pee em", text)
    return text


def text_to_phones(text, preprocess_config):
    """Returns (phoneme string, numpy array of symbol ids)."""
    global _g2p
    if _g2p is None:
        _g2p = G2p()
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    norm = normalize_english(text).rstrip(punctuation)
    tokens = re.split(r"([,;.\-\?\!\s+])", norm)
    use_rules = preprocess_config["preprocessing"]["text"].get("heteronym_rules", True)

    phones, prev_word = [], ""
    for w in tokens:
        lw = w.lower()
        if use_rules and lw == "conduct":
            phones += CONDUCT_VERB if prev_word in VERB_TRIGGERS else CONDUCT_NOUN
        elif lw in lexicon:
            phones += lexicon[lw]
        else:
            phones += [p for p in _g2p(w) if p != " "]
        if lw.strip() and re.match(r"[\w']+$", lw):
            prev_word = lw
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw text      :", text)
    print("Normalised    :", norm)
    print("Phoneme string:", phones)
    seq = text_to_sequence(phones, preprocess_config["preprocessing"]["text"]["text_cleaners"])
    return phones, np.array(seq)
