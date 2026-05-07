# tfidf.py — TF-IDF index and scoring

import math
from collections import defaultdict
from typing import Dict, List, Tuple


def build_index(docs: List[List[str]]) -> Dict[str, List[int]]:
    """Build an inverted index: word -> list of doc indices."""
    inv_index: Dict[str, List[int]] = defaultdict(list)
    for doc_idx, doc in enumerate(docs):
        for word in set(doc):
            inv_index[word].append(doc_idx)
    return inv_index


def compute_tfidf(
    docs: List[List[str]],
    inv_index: Dict[str, List[int]],
) -> Dict[Tuple[str, int], float]:
    """
    Compute TF-IDF for all (word, doc_idx) pairs.
    w_td = 1 + log10(tf),  idf = log10(N/df),  tfidf = w_td * idf
    """
    print("Calculating tf-idf...")
    tf: Dict[Tuple[str, int], int] = defaultdict(int)
    for doc_idx, doc in enumerate(docs):
        for word in doc:
            tf[(word, doc_idx)] += 1

    N = len(docs)
    tfidf: Dict[Tuple[str, int], float] = {}
    for (word, doc_idx), tf_td in tf.items():
        w_td  = 1 + math.log10(tf_td) if tf_td > 0 else 0
        idf_t = math.log10(N / len(inv_index[word]))
        tfidf[(word, doc_idx)] = w_td * idf_t
    return tfidf


def score_new_doc(
    doc_tokens: List[str],
    inv_index: Dict[str, List[int]],
    N_corpus: int,
) -> Dict[str, float]:
    """TF-IDF scores for a new document using corpus IDF values."""
    tf: Dict[str, int] = defaultdict(int)
    for w in doc_tokens:
        tf[w] += 1

    scores: Dict[str, float] = {}
    for w, count in tf.items():
        if w in inv_index and inv_index[w]:
            w_td  = 1 + math.log10(count) if count > 0 else 0
            idf_t = math.log10(N_corpus / len(inv_index[w]))
            scores[w] = w_td * idf_t
    return scores
