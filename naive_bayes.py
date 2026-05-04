# naive_bayes.py — Multinomial Naive Bayes with Laplace smoothing
#
# Copied from NaiveBayes.py (NBModel, NaiveBayes, run_stratified_kfold_cv).

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import N_SPLITS, SHUFFLE, RANDOM_STATE, REPORT


@dataclass
class NBModel:
    log_class_priors: Dict[str, float]            # log P(c)
    cond_log_probs:   Dict[str, Dict[str, float]] # log P(w|c)
    vocab:            set
    classes:          List[str]

    def predict_score(self, tokens: List[str]) -> Dict[str, float]:
        ans: Dict[str, float] = defaultdict(float)
        for c, log_prior in self.log_class_priors.items():
            score = log_prior
            for w in tokens:
                if w in self.vocab:
                    score += self.cond_log_probs[w][c]
            ans[c] = score
        return ans

    def predict_class(self, tokens: List[str]) -> str:
        scores = self.predict_score(tokens)
        return max(scores, key=scores.get)

    def predict_bulk(self, docs_tokens: List[List[str]]) -> List[str]:
        return [self.predict_class(toks) for toks in docs_tokens]


class NaiveBayes:
    """Multinomial Naive Bayes with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def train(self, docs: List[List[str]], labels: List[str]) -> NBModel:
        assert len(docs) == len(labels)
        N     = len(docs)
        vocab = set(w for doc in docs for w in doc)
        V     = len(vocab)

        class_counts: Dict[str, int] = defaultdict(int)
        for label in labels:
            class_counts[label] += 1

        log_class_priors = {c: math.log(cnt / N) for c, cnt in class_counts.items()}

        tok_counts:     Dict[str, int]            = defaultdict(int)
        token_freq:     Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for doc, label in zip(docs, labels):
            for w in doc:
                token_freq[w][label] += 1
                tok_counts[label]    += 1

        cond_log_probs: Dict[str, Dict[str, float]] = defaultdict(dict)
        for w in vocab:
            for c in class_counts:
                numerator   = token_freq[w][c] + self.alpha
                denominator = tok_counts[c]    + self.alpha * V
                cond_log_probs[w][c] = math.log(numerator / denominator)

        return NBModel(
            log_class_priors=log_class_priors,
            cond_log_probs=cond_log_probs,
            vocab=vocab,
            classes=sorted(class_counts.keys()),
        )


def run_stratified_kfold_cv(
    clf,
    docs_tokens: List[List[str]],
    labels: List[str],
    task_name: str = "",
    n_splits: int = N_SPLITS,
    shuffle: bool = SHUFFLE,
    random_state: int = RANDOM_STATE,
    print_report: bool = REPORT,
):
    """Stratified K-Fold CV. Copied from NaiveBayes.py."""
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    indices = list(range(len(labels)))
    accs: List[float] = []

    label_str = f" ({task_name})" if task_name else ""
    print(f"\n=== Stratified {n_splits}-Fold CV | clf={clf.__class__.__name__}{label_str} ===")

    for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels), start=1):
        X_train = [docs_tokens[i] for i in train_idx]
        y_train = [labels[i]      for i in train_idx]
        X_test  = [docs_tokens[i] for i in test_idx]
        y_test  = [labels[i]      for i in test_idx]

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
