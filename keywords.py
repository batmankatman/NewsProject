# keywords.py — Keyword extraction: TF-IDF, POS tagging, NER, bigram collocations
#
# Bigram PMI is computed with stdlib (math + collections) instead of
# nltk.BigramCollocationFinder, removing that NLTK dependency.

import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk

from preprocessing import STOP_WORDS

# POS tags considered content-bearing (NLTK Book Ch. 5)
_KEYWORD_POS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}


def get_tfidf_keywords(
    doc_scores: Dict[str, float],
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """Return top-N keywords from a pre-computed {word: tfidf_score} dict."""
    filtered = {
        w: s for w, s in doc_scores.items()
        if w not in STOP_WORDS and len(w) > 2 and s > 0
    }
    return sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]


def get_pos_keywords(text: str, top_n: int = 10) -> List[Tuple[str, str, int]]:
    """
    Extract content keywords using POS tagging (NLTK Book Ch. 5).
    Keeps nouns and adjectives only. Returns (word, POS_tag, frequency).
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    candidates: Dict[str, int] = defaultdict(int)
    tag_map:    Dict[str, str] = {}
    for word, tag in tagged:
        w = word.lower()
        if tag in _KEYWORD_POS and w not in STOP_WORDS and len(w) > 2:
            candidates[w] += 1
            tag_map[w] = tag

    top = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(word, tag_map[word], count) for word, count in top]


def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities via nltk.ne_chunk (NLTK Book Ch. 7).
    Returns {entity_type: [entity_text, ...]}.
    """
    entities: Dict[str, List[str]] = defaultdict(list)
    for sentence in sent_tokenize(text):
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        tree   = ne_chunk(tagged)
        for subtree in tree:
            if hasattr(subtree, "label"):
                etype       = subtree.label()
                entity_text = " ".join(word for word, _ in subtree.leaves())
                if entity_text not in entities[etype]:
                    entities[etype].append(entity_text)
    return dict(entities)


def get_bigram_collocations(text: str, top_n: int = 5) -> List[Tuple[str, str]]:
    """
    Find significant bigram collocations by PMI score.
    Computed with stdlib (Counter + math.log2) — no nltk.BigramCollocationFinder needed.
    """
    tokens = [
        w.lower() for w in word_tokenize(text)
        if w.lower() not in STOP_WORDS and len(w) > 2 and w.isalpha()
    ]
    if len(tokens) < 5:
        return []

    unigram_counts = Counter(tokens)
    bigram_counts  = Counter(zip(tokens, tokens[1:]))
    N = len(tokens)

    # Keep only bigrams that appear twice or more
    candidates = {bg: cnt for bg, cnt in bigram_counts.items() if cnt >= 2}

    def pmi(w1: str, w2: str, cnt: int) -> float:
        p_w1w2 = cnt / N
        p_w1   = unigram_counts[w1] / N
        p_w2   = unigram_counts[w2] / N
        return math.log2(p_w1w2 / (p_w1 * p_w2)) if p_w1 and p_w2 else float("-inf")

    scored = sorted(candidates.items(), key=lambda x: pmi(x[0][0], x[0][1], x[1]), reverse=True)
    return [bg for bg, _ in scored[:top_n]]
