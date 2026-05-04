# pipeline.py — Single-article analysis, display, RSS and interactive demos

import sys
from typing import Dict

import feedparser

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from config import BBC_RSS_URL, USE_STOPWORDS, MIN_TOKEN_LEN
from preprocessing import preprocess_tokens, tokenize_doc
from naive_bayes import NBModel
from tfidf import score_new_doc
from keywords import get_tfidf_keywords, get_pos_keywords, extract_named_entities, get_bigram_collocations
from sentiment import vader_sentiment


def analyze_article(
    text:        str,
    genre_model: NBModel,
    sent_model:  NBModel,
    analyzer:    SentimentIntensityAnalyzer,
    inv_index:   Dict,
    N_corpus:    int,
) -> Dict:
    """
    Full pipeline for one article:
      1. Genre classification (NB)
      2. TF-IDF keywords (corpus IDF)
      3. POS keywords + NER + bigram collocations
      4. Sentiment — NB (movie_reviews) + VADER
    """
    # Genre classification
    genre_tokens     = preprocess_tokens(text.split(), use_stopwords=USE_STOPWORDS, min_len=MIN_TOKEN_LEN)
    genre_log_scores = genre_model.predict_score(genre_tokens)
    predicted_genre  = max(genre_log_scores, key=genre_log_scores.get)
    max_score        = max(genre_log_scores.values())
    genre_rel        = {c: round(s - max_score, 2) for c, s in genre_log_scores.items()}

    # TF-IDF keywords
    doc_ir   = tokenize_doc(text)
    tfidf_sc = score_new_doc(doc_ir, inv_index, N_corpus)
    tfidf_kw = get_tfidf_keywords(tfidf_sc, top_n=10)

    # POS keywords, NER, collocations
    pos_kw = get_pos_keywords(text, top_n=10)
    ner    = extract_named_entities(text)
    colls  = get_bigram_collocations(text, top_n=5)

    # Sentiment
    sent_tokens     = preprocess_tokens(text.split(), use_stopwords=False)
    nb_label        = sent_model.predict_class(sent_tokens)
    v_label, v_score = vader_sentiment(text, analyzer)

    return {
        "genre":       predicted_genre,
        "genre_rel":   genre_rel,
        "tfidf_kw":    tfidf_kw,
        "pos_kw":      pos_kw,
        "ner":         ner,
        "colls":       colls,
        "nb_sent":     nb_label,
        "vader_label": v_label,
        "vader_score": v_score,
        "agree":       nb_label == v_label,
    }


def print_article(idx, text: str, result: Dict, true_label: str = ""):
    """Print one article's analysis results."""
    label_str = (
        f" | True: {true_label}  Predicted: {result['genre']}" if true_label
        else f" | Predicted: {result['genre']}"
    )
    print(f"\n--- Article {idx}{label_str} ---")
    print(f"Text:  {text[:90].replace(chr(10), ' ')}...")
    print(f"TF-IDF:    {', '.join(w for w, _ in result['tfidf_kw'][:8]) or '(none)'}")
    print(f"POS kw:    {', '.join(w for w, _, _ in result['pos_kw'][:8]) or '(none)'}")
    ner_parts = [f"{k}: {', '.join(v[:2])}" for k, v in result["ner"].items()]
    print(f"Entities:  {' | '.join(ner_parts) or '(none)'}")
    coll_strs = [f"{w1} {w2}" for w1, w2 in result["colls"]]
    print(f"Bigrams:   {', '.join(coll_strs) or '(none)'}")
    agree_str = "[agree]" if result["agree"] else "[disagree]"
    print(f"Sentiment: NB={result['nb_sent']}  VADER={result['vader_label']}  "
          f"compound={result['vader_score']:+.4f}  {agree_str}")
    sorted_genres = sorted(result["genre_rel"].items(), key=lambda x: x[1], reverse=True)
    print(f"Scores:    {'  '.join(f'{c}={s:+.1f}' for c, s in sorted_genres)}"
          f"  (0 = best fit, negative = less likely)")


def demo_rss(
    genre_model:  NBModel,
    sent_model:   NBModel,
    analyzer:     SentimentIntensityAnalyzer,
    inv_index:    Dict,
    N_corpus:     int,
    feed_url:     str = BBC_RSS_URL,
    max_articles: int = 5,
):
    """Fetch live BBC RSS articles and run the full pipeline on each."""
    print(f"\n=== Live RSS Demo | {feed_url} ===")
    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries[:max_articles]:
        title   = entry.get("title", "")
        summary = entry.get("summary", "")
        text    = f"{title}. {summary}" if summary else title
        if text.strip():
            articles.append(text)

    if not articles:
        print("No articles fetched. Check internet connection.")
        return

    print(f"Fetched {len(articles)} articles.")
    for i, text in enumerate(articles, start=1):
        result = analyze_article(text, genre_model, sent_model, analyzer, inv_index, N_corpus)
        print_article(i, text, result)


def demo_interactive(
    genre_model: NBModel,
    sent_model:  NBModel,
    analyzer:    SentimentIntensityAnalyzer,
    inv_index:   Dict,
    N_corpus:    int,
):
    """
    Interactive demo — reads text from stdin, blank line triggers analysis.
    Mirrors Regex.py's line-by-line stdin iteration pattern.
    """
    print("\n=== Interactive Demo ===")
    print("Paste article text (blank line to analyze, 'quit' to exit):\n")
    article_num = 1
    while True:
        lines = []
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line.strip().lower() == "quit":
                print("Exiting.")
                return
            if line == "" and lines:
                break
            lines.append(line)
        text = "\n".join(lines).strip()
        if not text:
            continue
        result = analyze_article(text, genre_model, sent_model, analyzer, inv_index, N_corpus)
        print_article(article_num, text, result)
        article_num += 1
        print("\nNext article (or 'quit'):\n")
