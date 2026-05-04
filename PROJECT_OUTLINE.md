# NewsScope: Multi-Dimensional News Article Analysis

## Project Overview

A focused NLP pipeline that **classifies news articles by genre**, **extracts key topics/entities**, and **performs sentiment analysis** — trained on a public dataset and demonstrated live on RSS feeds.

**Class topics covered:** Text Classification, Sentiment Analysis, Lexical Analysis, Information Retrieval, Basic Text Processing

---

## Architecture

```
[Input: Article Text]
        │
        ▼
┌──────────────────────┐
│  Text Preprocessing   │  ← NLTK tokenize, stopwords, POS tag
│  preprocessing.py     │
└──────────┬───────────┘
           │
     ┌─────┴──────┬──────────────┐
     ▼            ▼              ▼
┌─────────┐ ┌──────────┐ ┌────────────┐
│  Genre  │ │ Keyword  │ │ Sentiment  │
│ Classif.│ │ Extract. │ │  Analysis  │
│naive_   │ │keywords/ │ │sentiment.py│
│bayes.py │ │tfidf.py  │ │+ VADER     │
└────┬────┘ └────┬─────┘ └─────┬──────┘
     │           │             │
     ▼           ▼             ▼
┌─────────────────────────────────────┐
│         pipeline.py                 │
│  analyze_article / print_article    │
└─────────────────────────────────────┘
           │
           ▼
  [Live RSS Feed Demo — feedparser]
```

---

## File Structure (v0.2)

```
NewsProject/
├── NewsProject.py      # Entry point — main() only
├── config.py           # Global parameters (paths, hyperparams)
├── preprocessing.py    # preprocess_tokens, tokenize_doc, STOP_WORDS
├── naive_bayes.py      # NBModel, NaiveBayes, run_stratified_kfold_cv
├── tfidf.py            # build_index, compute_tfidf, score_new_doc
├── keywords.py         # TF-IDF/POS keywords, NER, bigram PMI collocations
├── sentiment.py        # load_movie_reviews_dataset, vader_sentiment
├── dataset.py          # load_bbc_raw, split_dataset
├── pipeline.py         # analyze_article, print_article, demo_rss, demo_interactive
└── PROJECT_OUTLINE.md
```

**Requirements:** `pip install nltk scikit-learn feedparser`

---

## Module Breakdown

### `naive_bayes.py` — Genre Classification & Sentiment NB
**Goal:** Classify articles into: business, entertainment, politics, sport, tech.
Also trains a Naive Bayes sentiment model on `movie_reviews`.

**Dataset:** [BBC News Dataset](http://mlg.ucd.ie/datasets/bbc.html) — 2,225 articles, 5 categories.

**NLTK Components:**
- `nltk.corpus.stopwords` — stopword removal
- `nltk.corpus.movie_reviews` — sentiment training data

**Approach:**
1. Preprocessing: tokenize → lowercase → remove stopwords (via `preprocessing.py`)
2. **Multinomial Naive Bayes** with Laplace smoothing (custom implementation)
3. Stratified K-Fold cross-validation (`sklearn.model_selection.StratifiedKFold`)
4. 80/10/10 train/val/test split, final test-set evaluation

**Results:** ~98% accuracy on BBC test set.

---

### `tfidf.py` + `keywords.py` — Keyword & Entity Extraction
**Goal:** Extract the most important terms and named entities from each article.

**NLTK Components:**
- `nltk.pos_tag` — POS tagging (filter nouns NN*, adjectives JJ*)
- `nltk.ne_chunk` — Named Entity Recognition (PERSON, ORG, GPE, etc.)

**Bigram collocations:** Implemented with stdlib `collections.Counter` + `math.log2` PMI
(no `nltk.BigramCollocationFinder` dependency).

**Approach:**
1. POS-tag each article → filter content-bearing tags
2. TF-IDF ranking across the corpus (`tfidf.py` — formula from PortStemmer.py)
3. NER with `ne_chunk` → extract PERSON, ORG, GPE
4. PMI bigram collocations for multi-word keywords

---

### `sentiment.py` — Sentiment Analysis
**Goal:** Score articles as positive, negative, or neutral.

**NLTK Components:**
- `nltk.sentiment.vader.SentimentIntensityAnalyzer` (VADER) — rule-based
- `nltk.corpus.movie_reviews` — training data for custom NB sentiment model

**Approach:**
1. **VADER** (rule-based): compound score → pos (≥0.05) / neg (≤-0.05) / neu
2. **Custom NB**: trained on `movie_reviews`, applied to news articles
3. Both labels shown per article; agreement/disagreement flagged

---

### `pipeline.py` — Live Demo
**Goal:** Pull live articles from RSS and process through all modules.

**Implementation:**
- `feedparser` — parse BBC RSS feed
- Each article → `analyze_article()` → `print_article()`

**Example output:**
```
--- Article 1 | Predicted: business ---
Text:  Dollar gains on Greenspan speech...
TF-IDF:    greenspan, deficit, federal, reserve, dollar...
POS kw:    dollar, deficit, greenspan, federal, reserve...
Entities:  PERSON: Alan Greenspan | ORGANIZATION: Federal Reserve
Bigrams:   federal reserve, new york, current account
Sentiment: NB=pos  VADER=pos  compound=+0.9421  [agree]
Scores:    business=+0.0  politics=-200.5  tech=-228.7  ...
```

---

## Dataset Plan

| Phase | Source | Purpose |
|-------|--------|---------|
| Training | BBC News Dataset (2,225 articles, 5 classes) | Genre classifier training |
| Training | NLTK `movie_reviews` corpus | Sentiment NB baseline |
| Demo | Live RSS feed (BBC) | Real-time pipeline showcase |

---

## NLTK / Library Summary

| Component | Module | Purpose |
|-----------|--------|---------|
| `word_tokenize` | keywords, sentiment | Tokenization |
| `sent_tokenize` | keywords | Sentence splitting for NER |
| `stopwords` | preprocessing | Shared `STOP_WORDS` frozenset |
| `pos_tag` | keywords | Noun/adjective keyword filtering |
| `ne_chunk` | keywords | Named entity extraction |
| `SentimentIntensityAnalyzer` (VADER) | sentiment | Rule-based sentiment |
| `movie_reviews` corpus | sentiment | NB sentiment training data |
| `feedparser` | pipeline | RSS feed parsing |
| `scikit-learn` (StratifiedKFold, metrics) | naive_bayes | CV and evaluation |

---

## Written Outline

1. **Introduction** — Problem statement, motivation
2. **Related Work** — Brief overview of news classification approaches
3. **Methods**
   - Text preprocessing pipeline
   - Genre classification (NB + TF-IDF)
   - Keyword extraction (POS, NER, collocations)
   - Sentiment analysis (VADER vs NB)
4. **Experimental Setup** — Datasets, evaluation metrics, CV strategy
5. **Results** — Accuracy tables, confusion matrices, example outputs
6. **Live Demo Description** — RSS pipeline, screenshots
7. **Discussion & Conclusion** — What worked, limitations, future work

---

## Code Reuse from Existing Work

| Existing Code | Reused For |
|--------------|------------|
| `NaiveBayes.py` → `NaiveBayes` class | Genre classifier (extend to 5 classes — already supports multi-class) |
| `NaiveBayes.py` → `preprocess_tokens()` | Shared preprocessing pipeline |
| `NaiveBayes.py` → `run_stratified_kfold_cv()` | Evaluation framework |
| `PortStemmer.py` → TF-IDF computation | Keyword ranking + feature weighting |
| `PortStemmer.py` → `IRSystem.index()` | Inverted index for keyword lookup |
| `assignment2/` → Language models | Potential: perplexity-based genre signal |

---

## Dependencies

```
nltk
feedparser        # RSS feed parsing
scikit-learn      # metrics, CV (already used)
```
