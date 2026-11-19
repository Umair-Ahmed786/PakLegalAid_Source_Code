"""
Unified legal vs. non-legal query classification evaluation for the PLOS ONE
revision (editor item A1).

One dataset (exact duplicate texts removed: 8,000 -> 7,267 rows), ONE stratified
80/20 split (seed 42; 5,813 train / 1,454 test), every model, ONE metric rule.
The paper reports precision, recall and F1 for the legal class (average="binary",
pos_label=1); accuracy, macro averages and the confusion matrix are also saved in
results_classification.csv for reference. Produces Tables 1-3 of the revised
manuscript, except the "without fine-tuning" rows of Table 3, which come from
linear_probe.py (frozen encoder + logistic regression head).

Run on Google Colab (GPU runtime recommended):

    !pip -q install scikit-learn xgboost transformers datasets accelerate tensorflow
    !python evaluate_classifiers.py --csv /content/Classification_dataset.csv --out /content/results

Outputs (in --out):
    results_classification.csv   one row per model, all metrics + confusion matrix
    predictions_<model>.csv      per-sample predictions for audit
    tables_latex.tex             ready-to-paste LaTeX for Tables 1, 2, 3
    run_log.txt                  library versions, split sizes, timings

Everything is wrapped so that one failing model does not stop the run.
"""

import argparse
import json
import os
import platform
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd

SEED = 42
TEST_SIZE = 0.20
MAX_LEN_TRANSFORMER = 128
TRANSFORMER_EPOCHS = 3
TRANSFORMER_LR = 2e-5
TRANSFORMER_BATCH = 16
DL_EPOCHS = 20
DL_BATCH = 32
VOCAB = 5000
DL_MAXLEN = 100

TRANSFORMERS = {
    "DistilBERT": "distilbert-base-uncased",
    "RoBERTa": "FacebookAI/roberta-base",
    "Legal-BERT": "nlpaueb/legal-bert-base-uncased",
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_b, r_b, f_b, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": 100 * acc,
        "precision_macro": 100 * p_m,
        "recall_macro": 100 * r_m,
        "f1_macro": 100 * f_m,
        "precision_legal": 100 * p_b,
        "recall_legal": 100 * r_b,
        "f1_legal": 100 * f_b,
        "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]), "TP": int(cm[1, 1]),
    }


class Recorder:
    def __init__(self, out_dir, test_texts, y_test, resume=False):
        self.out_dir = out_dir
        self.rows = []
        self.test_texts = list(test_texts)
        self.y_test = np.asarray(y_test).astype(int)
        self.path = os.path.join(out_dir, "results_classification.csv")
        if resume and os.path.exists(self.path):
            old = pd.read_csv(self.path)
            old = old[~old["note"].fillna("").str.startswith("FAILED")]
            old = old.drop_duplicates(subset=["model"], keep="first")
            self.rows = old.to_dict("records")
            pd.DataFrame(self.rows).to_csv(self.path, index=False)
            print(f"Resumed {len(self.rows)} finished rows from {self.path}")

    def done(self, model):
        return any(r.get("model") == model for r in self.rows)

    def add(self, family, model, y_pred, seconds, note=""):
        m = metrics(self.y_test, y_pred)
        m.update({"family": family, "model": model,
                  "train_seconds": round(seconds, 1), "note": note})
        self.rows.append(m)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)
        safe = model.replace(" ", "_").replace("(", "").replace(")", "")
        pd.DataFrame({"text": self.test_texts, "y_true": self.y_test,
                      "y_pred": np.asarray(y_pred).astype(int)}).to_csv(
            os.path.join(self.out_dir, f"predictions_{safe}.csv"), index=False)
        print(f"[{family}] {model:32s} acc={m['accuracy']:.2f} "
              f"legal-class P={m['precision_legal']:.2f} R={m['recall_legal']:.2f} "
              f"F1={m['f1_legal']:.2f} (macro F1={m['f1_macro']:.2f})  "
              f"cm=[[{m['TN']},{m['FP']}],[{m['FN']},{m['TP']}]]",
              flush=True)

    def fail(self, family, model, exc):
        print(f"[{family}] {model}: FAILED -> {exc!r}", flush=True)
        traceback.print_exc()
        self.rows.append({"family": family, "model": model,
                          "note": f"FAILED: {exc!r}"})
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


# ----------------------------------------------------------------------------
# 1. classical ML on TF-IDF
# ----------------------------------------------------------------------------
def run_classical(rec, X_train_txt, X_test_txt, y_train):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    X_train = vec.fit_transform(X_train_txt)
    X_test = vec.transform(X_test_txt)
    print(f"TF-IDF features: {X_train.shape[1]}")

    models = [
        ("Naive Bayes", MultinomialNB()),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ("Linear Discriminant Analysis", "LDA"),
        ("Decision Tree", DecisionTreeClassifier(random_state=SEED)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=SEED)),
        ("Support Vector Machine", SVC(kernel="linear", random_state=SEED)),
        ("Multi-Layer Perceptron", MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=SEED)),
        ("AdaBoost", AdaBoostClassifier(n_estimators=100, random_state=SEED)),
        ("XGBoost", "XGB"),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=SEED)),
    ]
    for name, clf in models:
        t0 = time.time()
        try:
            if clf == "LDA":
                # LDA needs a dense, lower-dimensional input
                from sklearn.decomposition import TruncatedSVD
                svd = TruncatedSVD(n_components=300, random_state=SEED)
                Xtr = svd.fit_transform(X_train)
                Xte = svd.transform(X_test)
                clf = LinearDiscriminantAnalysis()
                clf.fit(Xtr, y_train)
                pred = clf.predict(Xte)
                note = "TF-IDF -> TruncatedSVD(300) -> LDA"
            elif clf == "XGB":
                from xgboost import XGBClassifier
                clf = XGBClassifier(eval_metric="logloss", random_state=SEED)
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                note = ""
            else:
                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)
                note = ""
            rec.add("ML", name, pred, time.time() - t0, note)
        except Exception as e:
            rec.fail("ML", name, e)


# ----------------------------------------------------------------------------
# 2. Keras LSTM / GRU / CNN
# ----------------------------------------------------------------------------
def run_deep(rec, X_train_txt, X_test_txt, y_train):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Embedding, LSTM, GRU, Conv1D,
                                         GlobalMaxPooling1D, Dense, Dropout)
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split

    tok = Tokenizer(num_words=VOCAB, oov_token="<OOV>")
    tok.fit_on_texts(X_train_txt)
    Xtr = pad_sequences(tok.texts_to_sequences(X_train_txt), maxlen=DL_MAXLEN)
    Xte = pad_sequences(tok.texts_to_sequences(X_test_txt), maxlen=DL_MAXLEN)
    ytr = np.asarray(y_train)
    # carve a validation set out of TRAIN only (never touch the test split)
    Xtr2, Xval, ytr2, yval = train_test_split(
        Xtr, ytr, test_size=0.1, random_state=SEED, stratify=ytr)

    def build(kind):
        m = Sequential()
        m.add(Embedding(VOCAB, 64))
        if kind == "LSTM":
            m.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
        elif kind == "GRU":
            m.add(GRU(32, dropout=0.2, recurrent_dropout=0.2))
        else:
            m.add(Conv1D(64, 5, activation="relu"))
            m.add(GlobalMaxPooling1D())
            m.add(Dense(32, activation="relu"))
        m.add(Dropout(0.2))
        m.add(Dense(1, activation="sigmoid"))
        m.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return m

    for kind in ["LSTM", "GRU", "CNN"]:
        t0 = time.time()
        try:
            set_seed()
            m = build(kind)
            es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            m.fit(Xtr2, ytr2, validation_data=(Xval, yval), epochs=DL_EPOCHS,
                  batch_size=DL_BATCH, callbacks=[es], verbose=0)
            pred = (m.predict(Xte, verbose=0).ravel() >= 0.5).astype(int)
            rec.add("DL", kind, pred, time.time() - t0,
                    f"epochs_run={len(m.history.history['loss'])}")
        except Exception as e:
            rec.fail("DL", kind, e)


# ----------------------------------------------------------------------------
# 3. Transformers, without and with fine-tuning
# ----------------------------------------------------------------------------
def run_transformers(rec, X_train_txt, X_test_txt, y_train, out_dir):
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              Trainer, TrainingArguments, DataCollatorWithPadding)
    from sklearn.model_selection import train_test_split

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch device: {device}")

    import inspect
    _trainer_params = inspect.signature(Trainer.__init__).parameters
    _tok_kw = "processing_class" if "processing_class" in _trainer_params else "tokenizer"
    _ta_params = inspect.signature(TrainingArguments.__init__).parameters
    _eval_kw = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"

    def make_trainer(model, args, collator, tokenizer, **kw):
        return Trainer(model=model, args=args, data_collator=collator,
                       **{_tok_kw: tokenizer}, **kw)

    tr_txt, va_txt, tr_y, va_y = train_test_split(
        list(X_train_txt), list(y_train), test_size=0.1, random_state=SEED, stratify=list(y_train))

    for name, ckpt in TRANSFORMERS.items():
        if rec.done(f"{name} (with fine-tuning)"):
            print(f"[Transformer] {name}: already done, skipping")
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(ckpt)

            def enc(batch):
                return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN_TRANSFORMER)

            ds_tr = Dataset.from_dict({"text": tr_txt, "label": tr_y}).map(enc, batched=True)
            ds_va = Dataset.from_dict({"text": va_txt, "label": va_y}).map(enc, batched=True)
            ds_te = Dataset.from_dict({"text": list(X_test_txt),
                                       "label": [0] * len(X_test_txt)}).map(enc, batched=True)
            collator = DataCollatorWithPadding(tokenizer)

            # ---- without fine-tuning: pre-trained encoder + randomly initialised head
            if not rec.done(f"{name} (without fine-tuning)"):
                t0 = time.time()
                set_seed()
                model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
                args = TrainingArguments(output_dir=os.path.join(out_dir, "tmp_noft"),
                                         per_device_eval_batch_size=64, report_to=[], seed=SEED)
                trainer = make_trainer(model, args, collator, tokenizer)
                pred = np.argmax(trainer.predict(ds_te).predictions, axis=-1)
                rec.add("Transformer", f"{name} (without fine-tuning)", pred, time.time() - t0,
                        "pre-trained encoder with untrained classification head; no training")
                del model, trainer
                torch.cuda.empty_cache()

            # ---- with fine-tuning
            t0 = time.time()
            set_seed()
            model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
            args = TrainingArguments(
                output_dir=os.path.join(out_dir, f"tmp_ft_{name}"),
                num_train_epochs=TRANSFORMER_EPOCHS,
                learning_rate=TRANSFORMER_LR,
                per_device_train_batch_size=TRANSFORMER_BATCH,
                per_device_eval_batch_size=64,
                save_strategy="no",
                logging_steps=50,
                weight_decay=0.01,
                warmup_steps=int(0.06 * TRANSFORMER_EPOCHS * ((len(tr_txt) + TRANSFORMER_BATCH - 1) // TRANSFORMER_BATCH)),
                seed=SEED,
                fp16=(device == "cuda"),
                report_to=[],
                **{_eval_kw: "epoch"},
            )
            trainer = make_trainer(model, args, collator, tokenizer,
                                   train_dataset=ds_tr, eval_dataset=ds_va)
            trainer.train()
            pred = np.argmax(trainer.predict(ds_te).predictions, axis=-1)
            rec.add("Transformer", f"{name} (with fine-tuning)", pred, time.time() - t0,
                    f"full fine-tuning, {TRANSFORMER_EPOCHS} epochs, lr={TRANSFORMER_LR}, "
                    f"bs={TRANSFORMER_BATCH}, max_len={MAX_LEN_TRANSFORMER}")
            del model, trainer
            torch.cuda.empty_cache()
        except Exception as e:
            rec.fail("Transformer", name, e)


# ----------------------------------------------------------------------------
# 4. LaTeX tables
# ----------------------------------------------------------------------------
def write_latex(rec):
    df = pd.DataFrame(rec.rows)
    df = df[df["note"].fillna("").str.startswith("FAILED") == False]
    out = []

    n_test = int(len(rec.y_test))
    n_pos = int(rec.y_test.sum())
    split_txt = (f"on the held-out test split of {n_test:,} queries "
                 f"({n_pos:,} legal, {n_test - n_pos:,} non-legal). Precision, recall, and F1-score "
                 f"are reported for the legal (positive) class, so that F1 is the harmonic mean of the "
                 f"precision and recall in the same row; macro-averaged values are given in the released results file")

    def table(family, caption, label):
        caption = caption.replace("SPLIT", split_txt)
        sub = df[df["family"] == family]
        if sub.empty:
            return
        best = sub["accuracy"].idxmax()
        out.append("\\begin{table}[H]\n\\centering")
        out.append(f"\\caption{{{caption}}}\n\\label{{{label}}}")
        out.append("\\begin{tabular}{|l|c|c|c|c|}\n\\hline")
        out.append("\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Precision} & "
                   "\\textbf{Recall} & \\textbf{F1-Score} \\\\\n\\hline")
        for i, r in sub.iterrows():
            cells = [f"{r['accuracy']:.2f}", f"{r['precision_legal']:.2f}",
                     f"{r['recall_legal']:.2f}", f"{r['f1_legal']:.2f}"]
            if i == best:
                cells = [f"\\textbf{{{c}}}" for c in cells]
            out.append(f"{r['model']} & " + " & ".join(cells) + " \\\\")
        out.append("\\hline\n\\end{tabular}\n\\end{table}\n")

    table("ML", "Performance of traditional machine learning models SPLIT.", "tab:ml_results")
    table("DL", "Performance of deep learning models SPLIT.", "tab:dl_results")
    table("Transformer", "Performance of pre-trained transformer models with and without fine-tuning SPLIT.",
          "tab:transformer_results")
    path = os.path.join(rec.out_dir, "tables_latex.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"LaTeX tables written to {path}")

    # sanity check: legal-class F1 must equal the harmonic mean of legal-class P and R
    hm = 2 * df["precision_legal"] * df["recall_legal"] / (df["precision_legal"] + df["recall_legal"]).replace(0, np.nan)
    bad = df[(hm - df["f1_legal"]).abs() > 0.01]
    print("Sanity check (F1 == harmonic mean of P and R, per row):", "OK" if bad.empty else "VIOLATION\n" + str(bad))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Classification_dataset.csv (columns: question, label)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--skip", default="", help="comma list from {ml,dl,transformers}")
    ap.add_argument("--resume", action="store_true",
                    help="keep finished rows already in <out>/results_classification.csv and only run what is missing")
    ap.add_argument("--no-dedup", action="store_true",
                    help="keep exact duplicate questions (NOT recommended: duplicates leak between train and test)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    skip = set(s.strip() for s in a.skip.split(",") if s.strip())

    set_seed()
    df = pd.read_csv(a.csv)
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("question") or cols.get("text")
    label_col = cols.get("label")
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(int)
    n_raw = len(df)
    df[text_col] = df[text_col].str.strip()
    if not a.no_dedup:
        df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    n_dup_removed = n_raw - len(df)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].tolist(), df[label_col].tolist(), test_size=TEST_SIZE,
        random_state=SEED, stratify=df[label_col].tolist())

    log = {
        "python": sys.version, "platform": platform.platform(),
        "n_rows_in_csv": int(n_raw), "n_duplicates_removed": int(n_dup_removed),
        "n_total": int(len(df)), "n_train": len(X_train), "n_test": len(X_test),
        "train_class_counts": pd.Series(y_train).value_counts().to_dict(),
        "test_class_counts": pd.Series(y_test).value_counts().to_dict(),
        "seed": SEED, "test_size": TEST_SIZE,
    }
    for lib in ["sklearn", "xgboost", "tensorflow", "torch", "transformers"]:
        try:
            log[lib] = __import__(lib).__version__
        except Exception:
            log[lib] = "not installed"
    with open(os.path.join(a.out, "run_log.txt"), "w") as f:
        f.write(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))

    rec = Recorder(a.out, X_test, y_test, resume=a.resume)
    if "ml" not in skip and not rec.done("Logistic Regression"):
        run_classical(rec, X_train, X_test, y_train)
    if "dl" not in skip and not rec.done("CNN"):
        run_deep(rec, X_train, X_test, y_train)
    if "transformers" not in skip and not rec.done("Legal-BERT (with fine-tuning)"):
        run_transformers(rec, X_train, X_test, y_train, a.out)
    write_latex(rec)
    print("\nDone. Send back:", rec.path, "and tables_latex.tex and run_log.txt")


if __name__ == "__main__":
    main()
