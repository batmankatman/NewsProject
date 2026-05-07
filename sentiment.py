# sentiment.py — Sentiment analysis: NB on BBC corpus (VADER pseudo-labels) + VADER

from typing import List, Tuple

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from preprocessing import preprocess_tokens
from config import MIN_TOKEN_LEN, BINARIZE


def load_bbc_sentiment_dataset(
    raw_texts: List[str],
    analyzer: SentimentIntensityAnalyzer,
    use_stopwords: bool = False,
    min_len: int = MIN_TOKEN_LEN,
    binarize: bool = BINARIZE,
) -> Tuple[List[List[str]], List[str]]:
    """
    Pseudo-label BBC articles with VADER compound scores, then return
    preprocessed bag-of-words for NB sentiment training.

    Labels: 'pos' (compound >= 0.05), 'neg' (compound <= -0.05).
    Neutral articles are excluded — they carry too little discriminative signal.
    """
    docs_tokens: List[List[str]] = []
    labels: List[str] = []
    for text in raw_texts:
        compound = analyzer.polarity_scores(text)["compound"]
        if compound >= 0.05:
            label = "pos"
        elif compound <= -0.05:
            label = "neg"
        else:
            continue  # skip neutral articles
        tokens = preprocess_tokens(
            text.split(),
            use_stopwords=use_stopwords,
            min_len=min_len,
            binarize=binarize,
        )
        if tokens:
            docs_tokens.append(tokens)
            labels.append(label)
    return docs_tokens, labels


def vader_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> Tuple[str, float]:
    """
    VADER rule-based sentiment. Returns (label, compound_score).
    Labels: 'pos' (compound >= 0.05), 'neg' (<= -0.05), 'neu' otherwise.
    """
    compound = analyzer.polarity_scores(text)["compound"]
    if compound >= 0.05:
        label = "pos"
    elif compound <= -0.05:
        label = "neg"
    else:
        label = "neu"
    return label, compound
