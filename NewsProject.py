# ============================================================
# NewsProject.py
# All-in-one NLP Pipeline: BBC News Genre Classification,
# Keyword Extraction, and Sentiment Analysis + Live RSS Demo
#
# Patterns copied from:
#   NaiveBayes.py  — NBModel, NaiveBayes, preprocess_tokens,
#                    load_movie_reviews_dataset, run_stratified_kfold_cv
#   PortStemmer.py — alphanum regex, tokenize_doc, index, compute_tfidf
#   Regex.py       — line-by-line stdin processing in demo_interactive
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
import math
import random
import re
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk, BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =======================
# Global parameters
# =======================
N_SPLITS      = 5
RANDOM_STATE  = 42
SHUFFLE       = True
USE_STOPWORDS = True       # genre + keyword preprocessing
MIN_TOKEN_LEN = 2
BINARIZE      = False
ALPHA         = 1.0
REPORT        = True
BBC_DATA_DIR  = "./bbc"
BBC_RSS_URL   = "https://feeds.bbci.co.uk/news/rss.xml"
SAMPLE_N      = 3          # articles shown in detail during demo

BBC_CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]

# =======================
# Text utilities
# (copied from NaiveBayes.py)
# =======================
_WORD_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_+.-]*$")

def preprocess_tokens(
    tokens: Iterable[str],
    use_stopwords: bool = True,
    min_len: int = 2,
    lower: bool = True,
    binarize: bool = False
) -> List[str]:
    """
    Cleans tokens: lowercase, filters stopwords and non-alphanumeric/short tokens.
    If `binarize=True`, returns each token only once (presence/absence), no repetitions.
    Copied from NaiveBayes.py.
    """
    stopwords_english: set = set(w.lower() for w in stopwords.words("english"))
    out: List[str] = []
    out_word_set: set = set()

    for token in tokens:
        if use_stopwords and token.lower() in stopwords_english:
            continue
        if len(token) < min_len:
            continue
        if _WORD_RE.match(token) is None:
            continue
        token = token.lower() if lower else token
        if binarize and token in out_word_set:
            continue
        out.append(token)
        out_word_set.add(token)

    return out

# =======================
# Multinomial NB model
# (copied from NaiveBayes.py)
# =======================
@dataclass
class NBModel:
    log_class_priors: Dict[str, float]               # log P(c)
    cond_log_probs:   Dict[str, Dict[str, float]]    # log P(w|c)
    vocab:            set                            # vocabulary set
    classes:          List[str]                      # sorted class labels

    def predict_score(self, tokens: List[str]) -> Dict[str, float]:
        """
        Returns the prediction score for a document using log-probability sums.
        Standard Multinomial NB (uses document frequencies).
        """
        ans: Dict[str, float] = defaultdict(float)
        for c, log_prior in self.log_class_priors.items():
            score = log_prior
            for w in tokens:
                if w not in self.vocab:
                    continue
                score += self.cond_log_probs[w][c]
            ans[c] = score
        return ans

    def predict_class(self, tokens: List[str]) -> str:
        """
        Predicts a label for a document using log-probability sums.
        Standard Multinomial NB (uses document frequencies).
        """
        ans = self.predict_score(tokens)
        # [(c, score), (c, score), ...]
        c, _ = max(ans.items(), key=lambda a: a[1])
        return c

    def predict_bulk(self, docs_tokens: List[List[str]]) -> List[str]:
        return [self.predict_class(toks) for toks in docs_tokens]

# =======================
# Naive Bayes with Laplace smoothing
# (copied from NaiveBayes.py)
# =======================
class NaiveBayes:
    """
    Multinomial Naive Bayes.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def train(self, docs: List[List[str]], labels: List[str]) -> NBModel:
        # each index of doc has the same label equal to labels[index]
        assert len(docs) == len(labels), "Documents and labels have to have the same size"

        N = len(docs)
        vocab = set(w for doc in docs for w in doc)
        V = len(vocab)

        docs_j_count = defaultdict(int)
        for label in labels:
            docs_j_count[label] += 1

        # Log class priors
        log_class_priors = {
            c: math.log(count / N) for c, count in docs_j_count.items()
        }

        num_toks_in_label: Dict[str, int] = defaultdict(int)
        # Token frequency for each (w, c)
        token_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for doc, label in zip(docs, labels):
            for w in doc:
                token_freq[w][label] += 1
                num_toks_in_label[label] += 1

        # Conditional Log-probabilities
        cond_log_probs: Dict[str, Dict[str, float]] = defaultdict(dict)
        for w in vocab:
            for c in docs_j_count.keys():
                numerator   = token_freq[w][c] + self.alpha
                denominator = num_toks_in_label[c] + self.alpha * V
                cond_log_probs[w][c] = math.log(numerator / denominator)

        return NBModel(
            log_class_priors=log_class_priors,
            cond_log_probs=cond_log_probs,
            vocab=vocab,
            classes=sorted(docs_j_count.keys())
        )

# =======================
# TF-IDF + Inverted Index
# (copied from PortStemmer.py — IRSystem.index() and compute_tfidf())
# =======================

# Same regex used in PortStemmer.py to strip non-alphanumeric characters
alphanum = re.compile('[^a-zA-Z0-9]')

def tokenize_doc(text: str) -> List[str]:
    """
    Tokenize a document the same way PortStemmer.py does in __read_raw_data:
        line.lower() -> split -> alphanum.sub -> remove empty
    No stemming — keeps words human-readable for keyword display.
    """
    words = []
    for line in text.splitlines():
        line = line.lower()
        line = [xx.strip() for xx in line.split()]
        line = [alphanum.sub('', xx) for xx in line]
        line = [xx for xx in line if xx != '']
        words.extend(line)
    return words


def index(docs: List[List[str]]) -> Dict[str, List[int]]:
    """
    Build an inverted index: word -> list of doc indices.
    Adapted from PortStemmer.py IRSystem.index().
    """
    # Mapping of Words -> documents where they occur
    inv_index = defaultdict(list)
    for doc_idx, doc in enumerate(docs):
        for word in set(doc):
            inv_index[word].append(doc_idx)
    return inv_index


def compute_tfidf(docs: List[List[str]], inv_index: Dict) -> Dict[Tuple, float]:
    """
    Compute TF-IDF for all (word, doc_idx) pairs.
    Formula copied exactly from PortStemmer.py compute_tfidf():
        w_td  = 1 + log10(tf)
        idf_t = log10(N / df)
        tfidf = w_td * idf_t
    """
    print("Calculating tf-idf...")
    # Same variable name and loop structure as PortStemmer.py
    tf = defaultdict(int)
    for doc_idx, doc in enumerate(docs):
        for word in doc:
            tf[(word, doc_idx)] += 1

    tfidf = {}
    for (word, doc_idx), tf_td in tf.items():
        w_td  = 1 + math.log10(tf_td) if tf_td > 0 else 0
        idf_t = math.log10(len(docs) / len(inv_index[word]))
        tfidf[(word, doc_idx)] = w_td * idf_t

    return tfidf


def tfidf_new_doc(doc_tokens: List[str], inv_index: Dict, N_corpus: int) -> Dict[str, float]:
    """
    Compute TF-IDF scores for a single new document (not in the training corpus)
    by applying the corpus IDF values to the document's term frequencies.
    This is the standard way to score new documents at query time.
    """
    tf = defaultdict(int)
    for w in doc_tokens:
        tf[w] += 1
    scores = {}
    for w, count in tf.items():
        if w in inv_index and len(inv_index[w]) > 0:
            w_td  = 1 + math.log10(count) if count > 0 else 0
            idf_t = math.log10(N_corpus / len(inv_index[w]))
            scores[w] = w_td * idf_t
    return scores

# =======================
# BBC News dataset
# =======================
def load_bbc_raw(data_dir: str = BBC_DATA_DIR) -> Tuple[List[str], List[str]]:
    """
    Load raw article texts and labels from the BBC News dataset.
    Expected structure: data_dir/business/001.txt, etc.
    Download: http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
    Returns: raw_texts, labels
    """
    root = data_dir
    if not os.path.isdir(os.path.join(root, "business")):
        root = os.path.join(data_dir, "bbc")
    if not os.path.isdir(os.path.join(root, "business")):
        raise FileNotFoundError(
            f"BBC dataset not found at {data_dir}.\n"
            "Download: http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
        )
    raw_texts: List[str] = []
    labels:    List[str] = []
    for category in BBC_CATEGORIES:
        cat_dir = os.path.join(root, category)
        for filename in sorted(os.listdir(cat_dir)):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(cat_dir, filename), "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
            if text:
                raw_texts.append(text)
                labels.append(category)
    return raw_texts, labels


def split_dataset(
    docs_tokens: List[List[str]],
    labels: List[str],
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed: int = RANDOM_STATE
) -> Tuple[List, List, List, List, List, List]:
    """Stratified train / val / test split."""
    rng = random.Random(seed)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in by_label.items():
        rng.shuffle(indices)
        n       = len(indices)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train_idx.extend(indices[:n_train])
        val_idx.extend(  indices[n_train: n_train + n_val])
        test_idx.extend( indices[n_train + n_val:])

    def gather(idx_list):
        return [docs_tokens[i] for i in idx_list], [labels[i] for i in idx_list]

    X_train, y_train = gather(train_idx)
    X_val,   y_val   = gather(val_idx)
    X_test,  y_test  = gather(test_idx)
    return X_train, y_train, X_val, y_val, X_test, y_test

# =======================
# Cross-validation
# (copied from NaiveBayes.py — run_stratified_kfold_cv)
# =======================
def run_stratified_kfold_cv(
    clf,
    docs_tokens: List[List[str]],
    labels: List[str],
    task_name: str = "",
    n_splits: int = N_SPLITS,
    shuffle: bool = SHUFFLE,
    random_state: int = RANDOM_STATE,
    print_report: bool = REPORT
):
    """
    Runs Stratified K-Fold CV. Takes a classifier and trains/evaluates on each fold.
    Copied from NaiveBayes.py.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    indices = list(range(len(labels)))
    accs: List[float] = []

    clf_name = clf.__class__.__name__
    label_str = f" ({task_name})" if task_name else ""

    print(f"\n=== Stratified {n_splits}-Fold CV | clf={clf_name}{label_str} ===")
    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels), start=1):
        X_train = [docs_tokens[i] for i in train_idx]
        y_train = [labels[i] for i in train_idx]
        X_test  = [docs_tokens[i] for i in test_idx]
        y_test  = [labels[i] for i in test_idx]

        model  = clf.train(X_train, y_train)
        y_pred = model.predict_bulk(X_test)
        acc    = accuracy_score(y_test, y_pred)
        accs.append(acc)

        print(f"\n[FOLD {fold}] Accuracy: {acc:.4f}")
        if print_report:
            print(classification_report(y_test, y_pred, digits=4))
            print("Confusion matrix:")
            print(confusion_matrix(y_test, y_pred, labels=model.classes))

    mean_acc = sum(accs) / len(accs)
    print(f"\n=== Mean Accuracy ({n_splits} folds): {mean_acc:.4f} ===")
    return accs, mean_acc

# =======================
# Keyword extraction
# =======================

# Keep nouns (NN*) and adjectives (JJ*) as content-bearing POS tags
_KEYWORD_POS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}


def get_tfidf_keywords(
    doc_scores: Dict[str, float],
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Return top-N keywords from a pre-computed {word: tfidf_score} dict,
    filtering stopwords.
    """
    stop_words = set(w.lower() for w in stopwords.words("english"))
    filtered = {
        w: s for w, s in doc_scores.items()
        if w not in stop_words and len(w) > 2 and s > 0
    }
    return sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]


def get_pos_keywords(text: str, top_n: int = 10) -> List[Tuple[str, str, int]]:
    """
    Extract content keywords using POS tagging (NLTK Book Ch. 5).
    Keeps nouns and adjectives only.
    Returns list of (word, POS_tag, frequency).
    """
    stop_words = set(w.lower() for w in stopwords.words("english"))
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    candidates: Dict[str, int] = defaultdict(int)
    tag_map:    Dict[str, str] = {}
    for word, tag in tagged:
        w = word.lower()
        if tag in _KEYWORD_POS and w not in stop_words and len(w) > 2:
            candidates[w] += 1
            tag_map[w] = tag

    top = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(word, tag_map.get(word, ""), count) for word, count in top]


def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities using nltk.ne_chunk (NLTK Book Ch. 7).
    Returns dict: { entity_type: [entity_text, ...] }
    """
    entities: Dict[str, List[str]] = defaultdict(list)
    for sentence in sent_tokenize(text):
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        tree   = ne_chunk(tagged)
        for subtree in tree:
            if hasattr(subtree, "label"):
                etype       = subtree.label()
                entity_text = " ".join(word for word, tag in subtree.leaves())
                if entity_text not in entities[etype]:
                    entities[etype].append(entity_text)
    return dict(entities)


def get_bigram_collocations(text: str, top_n: int = 5) -> List[Tuple[str, str]]:
    """Find significant bigram collocations using PMI score."""
    stop_words = set(w.lower() for w in stopwords.words("english"))
    tokens = [w.lower() for w in word_tokenize(text)
              if w.lower() not in stop_words and len(w) > 2]
    if len(tokens) < 5:
        return []
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(2)
    return finder.nbest(BigramAssocMeasures.pmi, top_n)

# =======================
# Sentiment analysis
# =======================
def load_movie_reviews_dataset(
    use_stopwords: bool = False,
    min_len: int = MIN_TOKEN_LEN,
    binarize: bool = BINARIZE
) -> Tuple[List[List[str]], List[str], List[str]]:
    """
    Returns:
      - docs_tokens: list of preprocessed tokens per document
      - labels: list of labels ("pos" / "neg")
      - fileids: file IDs (for traceability)
    Copied from NaiveBayes.py.
    """
    fileids = movie_reviews.fileids()
    labels  = [movie_reviews.categories(fid)[0] for fid in fileids]
    docs_tokens: List[List[str]] = []
    for fid in fileids:
        tokens = movie_reviews.words(fid)
        tokens = preprocess_tokens(
            tokens,
            use_stopwords=use_stopwords,
            min_len=min_len,
            lower=True,
            binarize=binarize
        )
        docs_tokens.append(tokens)
    return docs_tokens, labels, fileids


def vader_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> Tuple[str, float]:
    """VADER rule-based sentiment. Returns (label, compound_score)."""
    scores   = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "pos"
    elif compound <= -0.05:
        label = "neg"
    else:
        label = "neu"
    return label, compound

# =======================
# Article analysis (single article)
# =======================
def analyze_article(
    text:          str,
    genre_model:   NBModel,
    sent_model:    NBModel,
    analyzer:      SentimentIntensityAnalyzer,
    inv_index:     Dict,
    N_corpus:      int
) -> Dict:
    """
    Run the full pipeline on one text string:
      1. Genre classification (NB)
      2. TF-IDF keywords (using corpus IDF)
      3. POS keywords + NER + bigram collocations
      4. Sentiment — NB (movie_reviews) + VADER
    """
    # Genre
    genre_tokens  = preprocess_tokens(text.split(), use_stopwords=USE_STOPWORDS, min_len=MIN_TOKEN_LEN)
    genre_log_scores = genre_model.predict_score(genre_tokens)
    predicted_genre  = max(genre_log_scores, key=genre_log_scores.get)

    # Normalize genre scores: subtract max so the best = 0, others negative but readable
    max_score = max(genre_log_scores.values())
    genre_rel = {c: round(s - max_score, 2) for c, s in genre_log_scores.items()}

    # TF-IDF keywords (corpus IDF applied to this document)
    doc_ir   = tokenize_doc(text)
    tfidf_sc = tfidf_new_doc(doc_ir, inv_index, N_corpus)
    tfidf_kw = get_tfidf_keywords(tfidf_sc, top_n=10)

    # POS keywords, NER, collocations
    pos_kw = get_pos_keywords(text, top_n=10)
    ner    = extract_named_entities(text)
    colls  = get_bigram_collocations(text, top_n=5)

    # Sentiment
    sent_tokens = preprocess_tokens(text.split(), use_stopwords=False)
    nb_label    = sent_model.predict_class(sent_tokens)
    v_label, v_score = vader_sentiment(text, analyzer)

    return {
        "genre":        predicted_genre,
        "genre_rel":    genre_rel,
        "tfidf_kw":     tfidf_kw,
        "pos_kw":       pos_kw,
        "ner":          ner,
        "colls":        colls,
        "nb_sent":      nb_label,
        "vader_label":  v_label,
        "vader_score":  v_score,
        "agree":        nb_label == v_label,
    }


def print_article(idx, text, result, true_label=""):
    """
    Print one article's analysis results.
    Clean, simple output — similar to the fold summaries in NaiveBayes.py.
    """
    label_str = f" | True: {true_label}  Predicted: {result['genre']}" if true_label else \
                f" | Predicted: {result['genre']}"
    print(f"\n--- Article {idx}{label_str} ---")
    print(f"Text:  {text[:90].replace(chr(10), ' ')}...")

    # TF-IDF
    kw_words = [w for w, _ in result['tfidf_kw'][:8]]
    print(f"TF-IDF:    {', '.join(kw_words) if kw_words else '(none)'}")

    # POS keywords
    pos_words = [w for w, _, _ in result['pos_kw'][:8]]
    print(f"POS kw:    {', '.join(pos_words) if pos_words else '(none)'}")

    # NER
    ner_parts = []
    for etype, items in result['ner'].items():
        ner_parts.append(f"{etype}: {', '.join(items[:2])}")
    print(f"Entities:  {' | '.join(ner_parts) if ner_parts else '(none)'}")

    # Collocations
    coll_strs = [f"{w1} {w2}" for w1, w2 in result['colls']]
    print(f"Bigrams:   {', '.join(coll_strs) if coll_strs else '(none)'}")

    # Sentiment
    agree_str = "[agree]" if result['agree'] else "[disagree]"
    print(f"Sentiment: NB={result['nb_sent']}  VADER={result['vader_label']}  "
          f"compound={result['vader_score']:+.4f}  {agree_str}")

    # Genre relative scores (0 = best, negative = less likely)
    sorted_genres = sorted(result['genre_rel'].items(), key=lambda x: x[1], reverse=True)
    score_str = "  ".join(f"{c}={s:+.1f}" for c, s in sorted_genres)
    print(f"Scores:    {score_str}  (0 = best fit, negative = less likely)")

# =======================
# Demo functions
# =======================
def demo_rss(
    genre_model: NBModel,
    sent_model:  NBModel,
    analyzer:    SentimentIntensityAnalyzer,
    inv_index:   Dict,
    N_corpus:    int,
    feed_url:    str = BBC_RSS_URL,
    max_articles: int = 5
):
    """
    Fetch live BBC RSS articles and run analyze_article() on each.
    Analogous to Regex.py's process_dir() — iterates a collection and calls
    process on each item.
    """
    try:
        import feedparser
    except ImportError:
        print("feedparser not installed. Run: pip install feedparser")
        return

    print(f"\n=== Live RSS Demo | {feed_url} ===")
    feed     = feedparser.parse(feed_url)
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
    for i, text in enumerate(articles):
        result = analyze_article(text, genre_model, sent_model, analyzer, inv_index, N_corpus)
        print_article(i + 1, text, result)


def demo_interactive(
    genre_model: NBModel,
    sent_model:  NBModel,
    analyzer:    SentimentIntensityAnalyzer,
    inv_index:   Dict,
    N_corpus:    int
):
    """
    Interactive demo — reads text line by line from stdin, exactly like
    Regex.py's process_file(name, f) iterates over lines of a file.
    Blank line triggers analysis. 'quit' exits.
    """
    print(f"\n=== Interactive Demo ===")
    print("Paste article text (blank line to analyze, 'quit' to exit):\n")

    article_num = 1
    while True:
        lines = []
        # Iterate over stdin line by line — same pattern as Regex.py's `for line in f:`
        for line in sys.stdin:
            line = line.rstrip('\n')
            if line.strip().lower() == 'quit':
                print("Exiting.")
                return
            if line == '' and lines:
                break
            lines.append(line)

        text = '\n'.join(lines).strip()
        if not text:
            continue

        result = analyze_article(text, genre_model, sent_model, analyzer, inv_index, N_corpus)
        print_article(article_num, text, result)
        article_num += 1
        print("\nNext article (or 'quit'):\n")

# =======================
# Main
# =======================
def main():
    run_cv    = "--skip-cv" not in sys.argv
    run_inter = "--interactive" in sys.argv

    random.seed(RANDOM_STATE)

    # Download required NLTK data
    for resource in ['stopwords', 'movie_reviews', 'vader_lexicon', 'punkt', 'punkt_tab',
                     'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
                     'maxent_ne_chunker', 'maxent_ne_chunker_tab', 'words']:
        nltk.download(resource, quiet=True)

    # ------------------------------------------------------------------
    # PART 1: BBC News Genre Classification
    # ------------------------------------------------------------------
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

    # Cross-validation (same as NaiveBayes.py)
    genre_clf = NaiveBayes(alpha=ALPHA)
    if run_cv:
        run_stratified_kfold_cv(genre_clf, docs_bow, labels, task_name="BBC Genre")

    # 80/10/10 split + test set evaluation
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(docs_bow, labels)
    genre_model = genre_clf.train(X_train, y_train)
    print(f"\nTrained on {len(X_train)} articles.")
    y_pred = genre_model.predict_bulk(X_test)
    print(f"\n=== Test Set Results (Genre) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=4, labels=genre_model.classes))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=genre_model.classes))

    # ------------------------------------------------------------------
    # PART 2: Keyword Extraction
    # Build TF-IDF index over full BBC corpus (PortStemmer.py pattern)
    # ------------------------------------------------------------------
    print("\n=== Keyword Extraction (TF-IDF over BBC corpus) ===")
    docs_ir   = [tokenize_doc(text) for text in raw_texts]
    inv_index = index(docs_ir)
    tfidf     = compute_tfidf(docs_ir, inv_index)
    N_corpus  = len(docs_ir)

    for i in range(SAMPLE_N):
        # Get this article's TF-IDF scores from the pre-built corpus matrix
        doc_tfidf_scores = {
            word: score
            for (word, d_idx), score in tfidf.items()
            if d_idx == i
        }
        tfidf_kw = get_tfidf_keywords(doc_tfidf_scores, top_n=10)
        pos_kw   = get_pos_keywords(raw_texts[i], top_n=10)
        ner      = extract_named_entities(raw_texts[i])
        colls    = get_bigram_collocations(raw_texts[i], top_n=5)

        print(f"\n--- Article {i} | {labels[i]} ---")
        print(f"Text:  {raw_texts[i][:90].replace(chr(10), ' ')}...")
        print(f"TF-IDF:   {', '.join(w for w, _ in tfidf_kw[:8])}")
        print(f"POS kw:   {', '.join(w for w, _, _ in pos_kw[:8])}")
        ner_parts = [f"{k}: {', '.join(v[:2])}" for k, v in ner.items()]
        print(f"Entities: {' | '.join(ner_parts) if ner_parts else '(none)'}")
        coll_strs = [f"{w1} {w2}" for w1, w2 in colls]
        print(f"Bigrams:  {', '.join(coll_strs) if coll_strs else '(none)'}")

    # ------------------------------------------------------------------
    # PART 3: Sentiment Analysis
    # (from NaiveBayes.py — same load + CV + train pattern)
    # ------------------------------------------------------------------
    print(f"\n\nLoading movie_reviews corpus...")
    print(f"Preprocessing: stopwords=False | min_len={MIN_TOKEN_LEN} | binarized={BINARIZE}")
    sent_docs, sent_labels, _ = load_movie_reviews_dataset()

    sent_clf = NaiveBayes(alpha=ALPHA)
    if run_cv:
        run_stratified_kfold_cv(sent_clf, sent_docs, sent_labels, task_name="Sentiment")

    sent_model = sent_clf.train(sent_docs, sent_labels)
    analyzer   = SentimentIntensityAnalyzer()

    # Compare NB vs VADER on BBC sample articles
    print(f"\n=== NB (movie_reviews) vs VADER on BBC Articles ===")
    for i in range(SAMPLE_N):
        sent_tokens = preprocess_tokens(raw_texts[i].split(), use_stopwords=False)
        nb_label    = sent_model.predict_class(sent_tokens)
        v_label, v_score = vader_sentiment(raw_texts[i], analyzer)
        agree = "agree" if nb_label == v_label else "disagree"
        print(f"  Article {i} ({labels[i]:15s})  NB={nb_label}  VADER={v_label}  "
              f"compound={v_score:+.4f}  [{agree}]")

    # ------------------------------------------------------------------
    # PART 4: Live RSS Demo
    # ------------------------------------------------------------------
    demo_rss(genre_model, sent_model, analyzer, inv_index, N_corpus)

    # ------------------------------------------------------------------
    # PART 5: Interactive Demo (opt-in)
    # ------------------------------------------------------------------
    if run_inter:
        demo_interactive(genre_model, sent_model, analyzer, inv_index, N_corpus)
    else:
        print("\nRun with --interactive to try the text input demo.")


if __name__ == "__main__":
    # Make sure you have executed:
    #   pip install nltk scikit-learn feedparser
    #   Download BBC dataset: http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
    main()
