"""Build the pronunciation dictionary that MFA needs for Telugu.

Every word in the corpus (.lab files) is converted with the rule-based Telugu G2P.
Usage:
    python make_mfa_dictionary.py raw_data telugu.dict
"""
import glob
import os
import sys

from text.telugu import word_to_phones

corpus, out = sys.argv[1], sys.argv[2]
words = set()
for f in glob.glob(os.path.join(corpus, "*", "*.lab")):
    with open(f, encoding="utf-8") as fh:
        words.update(fh.read().split())

n_empty = 0
with open(out, "w", encoding="utf-8") as fh:
    for w in sorted(words):
        ph = word_to_phones(w)
        if not ph:
            n_empty += 1
            continue
        fh.write("{}\t{}\n".format(w, " ".join(ph)))
print("{} words written to {} ({} skipped, no phones)".format(len(words) - n_empty, out, n_empty))
