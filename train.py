"""Train a text classifier for the Zindi Tech4MentalHealth challenge.

The dataset is not included. Download it from Zindi (if you are an authorised
participant), place ``Train.csv`` and ``Test.csv`` in ``data/``, then run:

    python train.py --model svm

The script prints cross-validation metrics and writes a submission CSV.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

DEFAULT_CLASSES = ("Depression", "Alcohol", "Suicide", "Drugs")
STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer()


def clean_text(value: object) -> str:
    """Lowercase, remove non-letters, drop stopwords, and stem."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    tokens = re.sub(r"[^A-Za-z]+", " ", str(value)).lower().split()
    return " ".join(STEMMER.stem(token) for token in tokens if token not in STOPWORDS)


def read_csv(path: str | Path, required: tuple[str, ...], name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} not found: {path}. Download the competition files and place "
            "Train.csv and Test.csv in data/."
        )
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} is missing column(s): {', '.join(missing)}. "
            f"Found: {', '.join(map(str, frame.columns))}."
        )
    return frame


def load_train(path, text_col: str, label_col: str, id_col: str, classes: tuple[str, ...]):
    frame = read_csv(path, (id_col, text_col, label_col), "Training data")
    if frame[[id_col, text_col, label_col]].isna().any().any():
        raise ValueError("Training data contains missing values.")
    if frame[id_col].duplicated().any():
        raise ValueError("Training data contains duplicate IDs.")

    frame = frame.copy()
    frame[text_col] = frame[text_col].astype(str)
    frame[label_col] = frame[label_col].astype(str).str.strip()

    unknown = sorted(set(frame[label_col]) - set(classes))
    if unknown:
        raise ValueError(
            f"Unknown label(s): {', '.join(unknown)}. "
            f"Expected one of: {', '.join(classes)}. Use --classes to change them."
        )
    missing = sorted(set(classes) - set(frame[label_col]))
    if missing:
        raise ValueError(f"Training data does not contain: {', '.join(missing)}.")
    return frame


def load_test(path, text_col: str, id_col: str):
    frame = read_csv(path, (id_col, text_col), "Test data")
    if frame[[id_col, text_col]].isna().any().any():
        raise ValueError("Test data contains missing values.")
    if frame[id_col].duplicated().any():
        raise ValueError("Test data contains duplicate IDs.")
    frame = frame.copy()
    frame[text_col] = frame[text_col].astype(str)
    return frame


def build_model(name: str, seed: int, calibration_cv: int = 5) -> Pipeline:
    """Build a TF-IDF text classifier."""
    if name == "logreg":
        features = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        classifier = LogisticRegression(
            C=16,
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        features = FeatureUnion(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(2, 5),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        classifier = CalibratedClassifierCV(
            LinearSVC(C=0.25, class_weight="balanced", max_iter=5000, random_state=seed),
            method="sigmoid",
            cv=calibration_cv,
            ensemble=False,
        )
    return Pipeline([("features", features), ("classifier", classifier)])


def evaluate(model: Pipeline, texts: list[str], labels: np.ndarray, cv: int, seed: int):
    counts = np.bincount(labels, minlength=labels.max() + 1)
    if counts.min() < cv:
        raise ValueError(
            f"Each class needs at least {cv} examples for {cv}-fold CV; "
            f"the smallest class has {counts.min()}."
        )

    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(model, texts, labels, cv=splitter, method="predict_proba")
    predictions = probabilities.argmax(axis=1)
    return {
        "log_loss": log_loss(labels, probabilities),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a text classifier for Tech4MentalHealth and write a submission."
    )
    parser.add_argument("--train", default="data/Train.csv", help="Training CSV path.")
    parser.add_argument("--test", default="data/Test.csv", help="Test CSV path.")
    parser.add_argument(
        "--output", default="submissions/submission.csv", help="Submission CSV path."
    )
    parser.add_argument("--text-col", default="text", help="Text column name.")
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--id-col", default="ID", help="ID column name.")
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class labels in submission order.",
    )
    parser.add_argument("--model", choices=("logreg", "svm"), default="logreg")
    parser.add_argument("--cv", type=int, default=5, help="CV folds; 0 skips evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    classes = tuple(item.strip() for item in args.classes.split(",") if item.strip())
    if len(classes) < 2:
        raise ValueError("--classes must contain at least two comma-separated labels.")
    if args.cv != 0 and args.cv < 2:
        raise ValueError("--cv must be 0 (skip) or at least 2.")

    train = load_train(args.train, args.text_col, args.label_col, args.id_col, classes)
    texts = [clean_text(value) for value in train[args.text_col]]
    labels = train[args.label_col].map({label: index for index, label in enumerate(classes)})
    labels = labels.to_numpy(dtype=int)

    class_counts = np.bincount(labels, minlength=len(classes))
    calibration_cv = max(1, min(5, int(class_counts.min())))
    if args.model == "svm" and calibration_cv < 2:
        raise ValueError("The SVM model needs at least two examples per class.")

    model = build_model(args.model, args.seed, calibration_cv)

    print(f"Training rows: {len(train)} | classes: {', '.join(classes)}")
    print(f"Model: {args.model}")
    if args.cv:
        metrics = evaluate(model, texts, labels, args.cv, args.seed)
        print(
            f"{args.cv}-fold CV: log loss {metrics['log_loss']:.4f} | "
            f"accuracy {metrics['accuracy']:.3f} | macro F1 {metrics['macro_f1']:.3f}"
        )

    model.fit(texts, labels)

    test = load_test(args.test, args.text_col, args.id_col)
    test_texts = [clean_text(value) for value in test[args.text_col]]
    probabilities = np.clip(model.predict_proba(test_texts), 1e-9, 1 - 1e-9)

    submission = pd.DataFrame({args.id_col: test[args.id_col]})
    for index, label in enumerate(classes):
        submission[label] = probabilities[:, index]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Wrote {len(submission)} rows to {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")
