"""Telugu grapheme-to-phoneme rules and number expansion.

Telugu spelling is almost phonetic, so a rule-based converter is enough:
  consonant            -> consonant phone (+ inherent 'a' unless a vowel sign or virama follows)
  vowel sign / vowel   -> vowel phone
  virama (్)           -> removes the inherent 'a'  (this builds conjuncts such as క్ష = k + ss)
  anusvara / candrabindu -> m,   visarga -> h
The phone names are plain ASCII so that they can also be used in the MFA dictionary.
"""
import re

VIRAMA = "\u0C4D"       # ్
ANUSVARA = "\u0C02"     # ం
CANDRABINDU = "\u0C01"  # ఁ
VISARGA = "\u0C03"      # ః

INDEPENDENT_VOWELS = {
    "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ii", "ఉ": "u", "ఊ": "uu",
    "ఋ": "ru", "ౠ": "ruu", "ఎ": "e", "ఏ": "ee", "ఐ": "ai",
    "ఒ": "o", "ఓ": "oo", "ఔ": "au",
}
VOWEL_SIGNS = {
    "ా": "aa", "ి": "i", "ీ": "ii", "ు": "u", "ూ": "uu", "ృ": "ru", "ౄ": "ruu",
    "ె": "e", "ే": "ee", "ై": "ai", "ొ": "o", "ో": "oo", "ౌ": "au",
}
# anusvara (ం) takes the nasal of the consonant that follows it
_NASAL_BEFORE = {}
for _chars, _nasal in (("కఖగఘఙ", "ng"), ("చఛజఝఞ", "ny"), ("టఠడఢణ", "nn"),
                       ("తథదధన", "n"), ("పఫబభమ", "m")):
    for _c in _chars:
        _NASAL_BEFORE[_c] = _nasal

CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "rr", "ల": "l", "ళ": "ll", "వ": "v",
    "శ": "sh", "ష": "ss", "స": "s", "హ": "h",
}

# every phone the model can see (silence tokens are added in symbols.py)
VALID_SYMBOLS = sorted(
    set(INDEPENDENT_VOWELS.values()) | set(VOWEL_SIGNS.values()) | set(CONSONANTS.values())
    | {"m", "h"}
)


def word_to_phones(word):
    """'విశ్లేషించారు' -> ['v', 'i', 'sh', 'l', 'ee', ...]"""
    phones = []
    chars = [c for c in word if c not in "\u200c\u200d"]
    i = 0
    while i < len(chars):
        c = chars[i]
        if c in CONSONANTS:
            phones.append(CONSONANTS[c])
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt == VIRAMA:
                i += 2
                continue
            if nxt in VOWEL_SIGNS:
                phones.append(VOWEL_SIGNS[nxt])
                i += 2
                continue
            phones.append("a")          # inherent vowel
        elif c in INDEPENDENT_VOWELS:
            phones.append(INDEPENDENT_VOWELS[c])
        elif c in (ANUSVARA, CANDRABINDU):
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            phones.append(_NASAL_BEFORE.get(nxt, "m"))
        elif c == VISARGA:
            phones.append("h")
        # anything else (stray signs, punctuation) is ignored
        i += 1
    return phones


# ----------------------------------------------------------------------------
# Numbers -> Telugu words (needed for the date and time in test sentence T4)
# ----------------------------------------------------------------------------
_ONES = {
    0: "సున్నా", 1: "ఒకటి", 2: "రెండు", 3: "మూడు", 4: "నాలుగు", 5: "ఐదు",
    6: "ఆరు", 7: "ఏడు", 8: "ఎనిమిది", 9: "తొమ్మిది", 10: "పది",
    11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు", 14: "పద్నాలుగు",
    15: "పదిహేను", 16: "పదహారు", 17: "పదిహేడు", 18: "పద్దెనిమిది", 19: "పంతొమ్మిది",
}
_TENS = {20: "ఇరవై", 30: "ముప్పై", 40: "నలభై", 50: "యాభై",
         60: "అరవై", 70: "డెబ్బై", 80: "ఎనభై", 90: "తొంభై"}


def number_to_words(n):
    """0 .. 99999 -> Telugu words (rule based, see README for limitations)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = (n // 10) * 10, n % 10
        return _TENS[t] + ("" if o == 0 else " " + _ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        if h == 1:
            return "వంద" if r == 0 else "నూట " + number_to_words(r)
        return _ONES[h] + (" వందలు" if r == 0 else " వందల " + number_to_words(r))
    th, r = divmod(n, 1000)
    if th == 1:
        head = "వెయ్యి"
    else:
        head = number_to_words(th) + (" వేలు" if r == 0 else " వేల")
    return head if r == 0 else head + " " + number_to_words(r)


def ordinal(n):
    """12 -> పన్నెండవ (12th)"""
    if n == 1:
        return "మొదటి"
    w = number_to_words(n)
    if w.endswith("ు"):
        return w[:-1] + "వ"
    if w.endswith("ై"):
        return w[:-1] + "య్యవ"
    if w.endswith("ి"):
        return w[:-1] + "వ"
    return w + "వ"


def expand_numbers(text):
    """10:45 -> 'పది నలభై ఐదు', 12వ -> 'పన్నెండవ', 2026 -> 'రెండు వేల ఇరవై ఆరు'"""
    text = re.sub(
        r"(\d{1,2}):(\d{2})",
        lambda m: number_to_words(int(m.group(1)))
        + ("" if int(m.group(2)) == 0 else " " + number_to_words(int(m.group(2)))),
        text,
    )
    text = re.sub(r"(\d+)\s*వ\b", lambda m: ordinal(int(m.group(1))), text)
    text = re.sub(r"\d+", lambda m: number_to_words(int(m.group(0))), text)
    return text
