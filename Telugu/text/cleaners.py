"""Text cleaners for Telugu."""
import re
import unicodedata

from .telugu import expand_numbers

_whitespace_re = re.compile(r"\s+")


def telugu_cleaners(text):
    """NFC-normalise, drop zero-width joiners, expand numbers, collapse spaces."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = expand_numbers(text)
    return _whitespace_re.sub(" ", text).strip()
