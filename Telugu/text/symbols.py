"""Symbol set for the Telugu model (replaces the English/Mandarin set of the original repo)."""
from text import telugu

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_silences = ["@sp", "@spn", "@sil"]
_telugu_phones = ["@" + s for s in telugu.VALID_SYMBOLS]

symbols = [_pad] + list(_special) + list(_punctuation) + _telugu_phones + _silences
