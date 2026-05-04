# dataset.py — BBC News dataset loading and train/val/test splitting

import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from config import BBC_DATA_DIR, BBC_CATEGORIES, RANDOM_STATE
from preprocessing import preprocess_tokens, tokenize_doc


def load_bbc_raw(data_dir: str = BBC_DATA_DIR) -> Tuple[List[str], List[str]]:
    """
    Load raw article texts and labels from the BBC News dataset.
    Expected structure: data_dir/business/001.txt, etc.
    Download: http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
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
    seed: int = RANDOM_STATE,
) -> Tuple[List, List, List, List, List, List]:
    """Stratified train / val / test split."""
    rng = random.Random(seed)
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for indices in by_label.values():
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
