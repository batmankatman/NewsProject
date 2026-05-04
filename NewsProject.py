# ============================================================
# NewsProject.py — Entry point
#
# NLP Pipeline: BBC News Genre Classification,
# Keyword Extraction, and Sentiment Analysis + Live RSS Demo
#
# Modules:
#   config.py        — global parameters
#   preprocessing.py — tokenization and cleaning
#   naive_bayes.py   — Multinomial NB + K-Fold CV
#   tfidf.py         — TF-IDF index and scoring
#   keywords.py      — TF-IDF/POS keywords, NER, bigram collocations
#   sentiment.py     — movie_reviews NB + VADER
#   dataset.py       — BBC dataset loading and splitting
#   pipeline.py      — article analysis, display, RSS and interactive demos
#
# Usage:
#   python NewsProject.py
#   python NewsProject.py --skip-cv       (faster, skips cross-validation)
#   python NewsProject.py --interactive   (adds interactive text input demo)
#
# Requirements:
#   pip install nltk scikit-learn feedparser
#   BBC dataset: http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
# ============================================================
import random
import sys

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import (
    RANDOM_STATE, USE_STOPWORDS, MIN_TOKEN_LEN, BINARIZE, ALPHA, SAMPLE_N,
    BBC_DATA_DIR, BBC_CATEGORIES,
)
from preprocessing import preprocess_tokens, tokenize_doc
from naive_bayes import NaiveBayes, run_stratified_kfold_cv
from tfidf import build_index, compute_tfidf, score_new_doc
from keywords import get_tfidf_keywords, get_pos_keywords, extract_named_entities, get_bigram_collocations
from sentiment import load_movie_reviews_dataset, vader_sentiment
from dataset import load_bbc_raw, split_dataset
from pipeline import analyze_article, print_article, demo_rss, demo_interactive




def main():
    run_cv    = "--skip-cv" not in sys.argv
    run_inter = "--interactive" in sys.argv

    random.seed(RANDOM_STATE)

    # Download required NLTK data
    nltk.download("stopwords")
    nltk.download("movie_reviews")

    # PART 1: BBC Genre Classification
    print("\nLoading BBC News dataset...")
    print(f"Preprocessing: stopwords={USE_STOPWORDS} | min_len={MIN_TOKEN_LEN} | binarized={BINARIZE}")

    raw_texts, labels = load_bbc_raw(data_dir=BBC_DATA_DIR)
    docs_bow = [
        preprocess_tokens(text.split(), use_stopwords=USE_STOPWORDS,
                          min_len=MIN_TOKEN_LEN, binarize=BINARIZE)
        for text in raw_texts
    ]

    print(f"Loaded {len(docs_bow)} articles across {len(BBC_CATEGORIES)} categories")
    for cat in BBC_CATEGORIES:
        print(f"  {cat:15s}: {sum(1 for l in labels if l == cat)}")

    genre_clf = NaiveBayes(alpha=ALPHA)
    if run_cv:
        run_stratified_kfold_cv(genre_clf, docs_bow, labels, task_name="BBC Genre")

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(docs_bow, labels)
    genre_model = genre_clf.train(X_train, y_train)
    print(f"\nTrained on {len(X_train)} articles.")
    y_pred = genre_model.predict_bulk(X_test)
    print("\n=== Test Set Results (Genre) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=4, labels=genre_model.classes))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=genre_model.classes))

    # PART 2: Keyword Extraction (TF-IDF over full BBC corpus)
    print("\n=== Keyword Extraction + Sentiment on Sample Articles ===")
    docs_ir   = [tokenize_doc(text) for text in raw_texts]
    inv_index = build_index(docs_ir)
    tfidf     = compute_tfidf(docs_ir, inv_index)
    N_corpus  = len(docs_ir)

    # PART 3: Sentiment — NB on movie_reviews + VADER
    print(f"\nLoading movie_reviews corpus...")
    print(f"Preprocessing: stopwords=False | min_len={MIN_TOKEN_LEN} | binarized={BINARIZE}")
    sent_docs, sent_labels, _ = load_movie_reviews_dataset()

    sent_clf = NaiveBayes(alpha=ALPHA)
    if run_cv:
        run_stratified_kfold_cv(sent_clf, sent_docs, sent_labels, task_name="Sentiment")

    sent_model = sent_clf.train(sent_docs, sent_labels)
    analyzer   = SentimentIntensityAnalyzer()

    # Show SAMPLE_N articles with keyword and sentiment output
    print(f"\n=== Sample Article Analysis (n={SAMPLE_N}) ===")
    for i in range(SAMPLE_N):
        doc_tfidf_scores = {
            word: score
            for (word, d_idx), score in tfidf.items()
            if d_idx == i
        }
        result = analyze_article(
            raw_texts[i], genre_model, sent_model, analyzer, inv_index, N_corpus
        )
        # Use corpus-matrix scores for these training articles
        result["tfidf_kw"] = get_tfidf_keywords(doc_tfidf_scores, top_n=10)
        print_article(i, raw_texts[i], result, true_label=labels[i])

    # PART 4: Live RSS Demo
    demo_rss(genre_model, sent_model, analyzer, inv_index, N_corpus)

    # PART 5: Interactive Demo (opt-in)
    if run_inter:
        demo_interactive(genre_model, sent_model, analyzer, inv_index, N_corpus)
    else:
        print("\nRun with --interactive to try the text input demo.")


if __name__ == "__main__":
    main()
