# NewsScope — BBC News NLP Pipeline

Classifies BBC news articles by genre, extracts keywords and named entities, and performs sentiment analysis — demonstrated live on a BBC RSS feed.

**Techniques:** Multinomial Naive Bayes, TF-IDF, POS tagging, NER, VADER sentiment, K-Fold CV.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install nltk scikit-learn feedparser

# 3. Unzip the dataset (produces bbc/ inside this folder)
unzip bbc.zip
```

NLTK resources are downloaded automatically on first run.

---

## Usage

```bash
python NewsProject.py            # full run with 5-fold CV
python NewsProject.py --skip-cv  # skip cross-validation (faster)
```

---

## Dataset

**BBC News** — 2,225 articles across 5 categories. Included as `bbc.zip`; the extracted `bbc/` folder is gitignored.

| Category      | Articles |
|---------------|----------|
| business      | 510      |
| entertainment | 386      |
| politics      | 417      |
| sport         | 511      |
| tech          | 401      |

**Split:** stratified 80 / 10 / 10 (train / val / test) per category.

---

## Function

1. **Genre classification** — custom Multinomial NB with Laplace smoothing (~98% test accuracy).
2. **Keyword extraction** — TF-IDF ranking + POS-filtered keywords (nouns/adjectives) + NER.
3. **Sentiment** — NB trained on BBC articles pseudo-labeled by VADER, plus VADER for a second opinion.
4. **Live RSS demo** — fetches the latest BBC headlines and runs the full pipeline on each.

---

## File structure

```
NewsProject/
├── NewsProject.py      # entry point
├── config.py           # paths and hyperparameters
├── preprocessing.py    # tokenization and cleaning
├── naive_bayes.py      # NB model + K-Fold CV
├── tfidf.py            # TF-IDF index and scoring
├── keywords.py         # TF-IDF/POS keywords and NER
├── sentiment.py        # BBC-labeled NB sentiment + VADER
├── dataset.py          # dataset loading and splitting
├── pipeline.py         # per-article analysis and RSS demo
└── bbc.zip             # BBC News dataset (unzip before running)
```
