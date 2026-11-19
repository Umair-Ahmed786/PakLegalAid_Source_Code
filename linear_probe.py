"""
Linear-probe baseline for the transformer models in Table 3 (PLOS ONE revision).

A linear probe keeps the pre-trained encoder FROZEN and trains only a logistic-regression
classifier on its sentence embeddings. It replaces the "without fine-tuning" rows, which used
an untrained, randomly initialised head.

Same data, de-duplication, stratified 80/20 split (seed 42) and metric rule as
evaluate_classifiers.py:
    accuracy over all test queries; precision, recall and F1 for the legal class
    (average="binary", pos_label=1); confusion matrix for every model.

Pre-declared choices (fixed before looking at the results, no tuning on the test split):
    - embedding  : mean of the last hidden states over the real (non-padding) tokens  [primary]
                   the first-token ([CLS] / <s>) vector is also reported as a check
    - max length : 128 tokens (same as fine-tuning in evaluate_classifiers.py)
    - classifier : StandardScaler + LogisticRegression(C=1.0, max_iter=5000, random_state=42),
                   trained on all 5,813 training texts

Run on Google Colab (GPU runtime recommended; CPU also works, just slower):
    !pip -q install scikit-learn transformers
    !python linear_probe.py --csv /content/Classification_dataset.csv --out /content/probe

Outputs (in --out):
    linear_probe_results.csv   one row per model and embedding, all metrics + confusion matrix
    linear_probe_log.txt       split sizes, library versions, timings
"""
import argparse
import json
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd

SEED = 42
TEST_SIZE = 0.20
MAX_LEN = 128
BATCH = 64

TRANSFORMERS = {
    "DistilBERT": "distilbert-base-uncased",
    "RoBERTa": "FacebookAI/roberta-base",
    "Legal-BERT": "nlpaueb/legal-bert-base-uncased",
}


def set_seed(seed=SEED):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred):
    """Identical rule to evaluate_classifiers.py."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                 pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": 100 * accuracy_score(y_true, y_pred),
        "precision_legal": 100 * p, "recall_legal": 100 * r, "f1_legal": 100 * f,
        "TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
    }


def load_split(csv_path):
    """Same steps, in the same order, as main() in evaluate_classifiers.py."""
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("question") or cols.get("text")
    label_col = cols.get("label")
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(int)
    n_raw = len(df)
    df[text_col] = df[text_col].str.strip()
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].tolist(), df[label_col].tolist(), test_size=TEST_SIZE,
        random_state=SEED, stratify=df[label_col].tolist())
    return n_raw, len(df), X_train, X_test, np.array(y_train), np.array(y_test)


def embed(texts, tokenizer, model, device):
    """Return (mean-pooled, first-token) embeddings from the frozen encoder."""
    import torch
    mean_out, cls_out = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tokenizer(texts[i:i + BATCH], truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state                 # (b, t, h)
            mask = enc["attention_mask"].unsqueeze(-1).float()      # (b, t, 1)
            mean_out.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
            cls_out.append(hidden[:, 0, :].cpu().numpy())
    return np.vstack(mean_out), np.vstack(cls_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Classification_dataset.csv (columns: question, label)")
    ap.add_argument("--out", default="probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    import sklearn
    import transformers
    from transformers import AutoModel, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_raw, n_total, X_train, X_test, y_train, y_test = load_split(a.csv)
    log = {
        "python": sys.version, "platform": platform.platform(),
        "n_rows_in_csv": n_raw, "n_total": n_total,
        "n_train": len(X_train), "n_test": len(X_test),
        "test_legal": int(y_test.sum()), "test_nonlegal": int(len(y_test) - y_test.sum()),
        "seed": SEED, "max_len": MAX_LEN,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "sklearn": sklearn.__version__,
    }
    print(json.dumps(log, indent=2))
    # must be the same split as Tables 1-3
    assert (len(X_train), len(X_test), log["test_legal"]) == (5813, 1454, 798), \
        "Split differs from evaluate_classifiers.py - use the same classification CSV."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    rows = []
    for name, ckpt in TRANSFORMERS.items():
        t0 = time.time()
        set_seed()
        tok = AutoTokenizer.from_pretrained(ckpt)
        enc_model = AutoModel.from_pretrained(ckpt).to(device)   # frozen: no training, eval mode
        tr_mean, tr_cls = embed(X_train, tok, enc_model, device)
        te_mean, te_cls = embed(X_test, tok, enc_model, device)
        del enc_model
        if device == "cuda":
            torch.cuda.empty_cache()
        for emb, (Xtr, Xte) in (("mean-pooled", (tr_mean, te_mean)), ("first-token", (tr_cls, te_cls))):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=5000, random_state=SEED))
            clf.fit(Xtr, y_train)
            m = metrics(y_test, clf.predict(Xte))
            rows.append({"model": f"{name} (linear probe)", "embedding": emb,
                         "primary": emb == "mean-pooled", **m,
                         "seconds": round(time.time() - t0, 1), "checkpoint": ckpt})
            print(f"{name:11s} {emb:12s} acc {m['accuracy']:.2f}  P {m['precision_legal']:.2f}  "
                  f"R {m['recall_legal']:.2f}  F1 {m['f1_legal']:.2f}  "
                  f"TN {m['TN']} FP {m['FP']} FN {m['FN']} TP {m['TP']}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "linear_probe_results.csv"), index=False)
    hm = 2 * res.precision_legal * res.recall_legal / (res.precision_legal + res.recall_legal).replace(0, np.nan)
    print("Sanity check (F1 == harmonic mean of P and R):",
          "OK" if ((hm - res.f1_legal).abs().fillna(0) < 0.01).all() else "VIOLATION")
    with open(os.path.join(a.out, "linear_probe_log.txt"), "w") as f:
        f.write(json.dumps(log, indent=2))
    print("\nDone. Send back:", os.path.join(a.out, "linear_probe_results.csv"),
          "and linear_probe_log.txt")


if __name__ == "__main__":
    main()
