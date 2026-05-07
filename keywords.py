# keywords.py — Keyword extraction: TF-IDF, POS tagging, NER

from collections import defaultdict
from typing import Dict, List, Tuple

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk

from preprocessing import STOP_WORDS

_KEYWORD_POS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}  # nouns + adjectives


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
    """Extract top-N nouns/adjectives by frequency. Returns (word, POS_tag, count)."""
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
    """Extract named entities via nltk.ne_chunk. Returns {entity_type: [text, ...]}."""
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



