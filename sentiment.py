# sentiment.py — Sentiment analysis: NB on movie_reviews + VADER
#
# Mirrors the pattern in NaiveBayes.py (load_movie_reviews_dataset).

from typing import List, Tuple

from nltk.corpus import movie_reviews
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from preprocessing import preprocess_tokens
from config import MIN_TOKEN_LEN, BINARIZE


def load_movie_reviews_dataset(
    use_stopwords: bool = False,
    min_len: int = MIN_TOKEN_LEN,
    binarize: bool = BINARIZE,
) -> Tuple[List[List[str]], List[str], List[str]]:
    """
    Load NLTK movie_reviews and return preprocessed tokens, labels, and file IDs.
    Copied from NaiveBayes.py.
    """
    fileids = movie_reviews.fileids()
    labels  = [movie_reviews.categories(fid)[0] for fid in fileids]
    docs_tokens: List[List[str]] = [
        preprocess_tokens(
            movie_reviews.words(fid),
            use_stopwords=use_stopwords,
            min_len=min_len,
            lower=True,
            binarize=binarize,
        )
        for fid in fileids
    ]
    return docs_tokens, labels, list(fileids)


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
