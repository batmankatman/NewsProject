# preprocessing.py — Text tokenization and cleaning utilities
#
# Patterns from NaiveBayes.py (preprocess_tokens) and PortStemmer.py (tokenize_doc).

import re
from typing import Iterable, List

from nltk.corpus import stopwords

# Compiled regexes from NaiveBayes.py and PortStemmer.py
_WORD_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_+.-]*$")
_ALPHANUM = re.compile(r"[^a-zA-Z0-9]")

# Module-level stop-word set — loaded once, shared by all callers.
# Uses a frozenset for O(1) membership tests.
STOP_WORDS: frozenset = frozenset(w.lower() for w in stopwords.words("english"))


def preprocess_tokens(
    tokens: Iterable[str],
    use_stopwords: bool = True,
    min_len: int = 2,
    lower: bool = True,
    binarize: bool = False,
) -> List[str]:
    """
    Clean tokens: lowercase, filter stopwords and non-alphabetic/short tokens.
    If binarize=True each token appears at most once (presence/absence feature).
    Copied from NaiveBayes.py.
    """
    out: List[str] = []
    seen: set = set()
    for token in tokens:
        if use_stopwords and token.lower() in STOP_WORDS:
            continue
        if len(token) < min_len:
            continue
        if _WORD_RE.match(token) is None:
            continue
        token = token.lower() if lower else token
        if binarize and token in seen:
            continue
        out.append(token)
        seen.add(token)
    return out


def tokenize_doc(text: str) -> List[str]:
    """
    Tokenize a document the same way PortStemmer.py does:
        line.lower() -> split -> strip non-alphanum -> remove empty
    No stemming — keeps words human-readable for keyword display.
    """
    words: List[str] = []
    for line in text.splitlines():
        parts = [_ALPHANUM.sub("", tok.strip()) for tok in line.lower().split()]
        words.extend(p for p in parts if p)
    return words
