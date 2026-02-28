# backend/model_train.py
"""
SmartEdge Copilot – Prompt Classification Pipeline
Trains a TF-IDF + Logistic Regression classifier that predicts the feature
category a prompt belongs to (e.g. "summarizer", "planner", "research").

The trained artefacts are persisted with joblib so the inference path
is a single file load – no re-training required at serving time.
"""

import os
import json
import warnings
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from typing import Any

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

MODELS_DIR   = Path("models")
MODEL_PATH   = MODELS_DIR / "prompt_classifier.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
META_PATH    = MODELS_DIR / "model_meta.json"


# ─────────────────────────────────────────────
# Synthetic Bootstrap Data
# ─────────────────────────────────────────────
# Used when no real logged prompts are available yet (cold-start).
# Extend or replace with your own corpus as the system accumulates data.

BOOTSTRAP_DATA: list[dict] = [
    # ── summarizer ──────────────────────────────────────────────────────
    {"text": "Summarize this article about climate change",               "label": "summarizer"},
    {"text": "Give me a brief summary of the quarterly earnings report",  "label": "summarizer"},
    {"text": "TL;DR this research paper on neural networks",              "label": "summarizer"},
    {"text": "Condense this 10-page document into key bullet points",     "label": "summarizer"},
    {"text": "What are the main takeaways from this blog post?",          "label": "summarizer"},
    {"text": "Summarize the meeting notes from last Tuesday",             "label": "summarizer"},
    {"text": "Provide an executive summary of this white paper",          "label": "summarizer"},
    {"text": "Distill this legal document into plain English",            "label": "summarizer"},
    {"text": "Create a short abstract for this academic paper",           "label": "summarizer"},
    {"text": "Summarize the key findings of this market analysis",        "label": "summarizer"},

    # ── planner ─────────────────────────────────────────────────────────
    {"text": "Create a 30-day learning plan for Python",                  "label": "planner"},
    {"text": "Build a project roadmap for a mobile app launch",           "label": "planner"},
    {"text": "Plan my week given these tasks and deadlines",              "label": "planner"},
    {"text": "Design a sprint plan for the next two weeks",              "label": "planner"},
    {"text": "Help me schedule these meetings without conflicts",         "label": "planner"},
    {"text": "Create a study schedule for the AWS certification exam",    "label": "planner"},
    {"text": "Outline the milestones for our product v2 release",        "label": "planner"},
    {"text": "Plan a content calendar for Q3",                           "label": "planner"},
    {"text": "Generate a timeline for onboarding new team members",      "label": "planner"},
    {"text": "Create an action plan to reduce customer churn by 20%",    "label": "planner"},

    # ── research ────────────────────────────────────────────────────────
    {"text": "What are the latest trends in large language models?",      "label": "research"},
    {"text": "Research the top 5 competitors in the CRM market",         "label": "research"},
    {"text": "Find information about the history of transformer models",  "label": "research"},
    {"text": "What does the literature say about retrieval-augmented generation?", "label": "research"},
    {"text": "Compare PostgreSQL vs MongoDB for a high-write workload",   "label": "research"},
    {"text": "What are the pros and cons of microservices architecture?", "label": "research"},
    {"text": "Investigate the causes of the 2008 financial crisis",      "label": "research"},
    {"text": "Research best practices for API rate limiting",             "label": "research"},
    {"text": "What are the current GDPR requirements for data retention?","label": "research"},
    {"text": "Explore state-of-the-art methods for time-series forecasting","label": "research"},

    # ── optimizer ───────────────────────────────────────────────────────
    {"text": "Optimize this SQL query for better performance",            "label": "optimizer"},
    {"text": "Refactor this Python function to reduce memory usage",      "label": "optimizer"},
    {"text": "How can I speed up this React component?",                  "label": "optimizer"},
    {"text": "Reduce the latency in this API endpoint",                   "label": "optimizer"},
    {"text": "Improve the efficiency of this data pipeline",              "label": "optimizer"},
    {"text": "Rewrite this loop to use vectorized numpy operations",      "label": "optimizer"},
    {"text": "Suggest caching strategies for this high-traffic service",  "label": "optimizer"},
    {"text": "How do I reduce the bundle size of my webpack build?",      "label": "optimizer"},
    {"text": "Optimize database indexes for these slow queries",          "label": "optimizer"},
    {"text": "Profile and fix the bottleneck in this ML training loop",   "label": "optimizer"},

    # ── general ─────────────────────────────────────────────────────────
    {"text": "What is the capital of France?",                            "label": "general"},
    {"text": "Explain how TCP/IP works",                                  "label": "general"},
    {"text": "Write a professional email declining a meeting",            "label": "general"},
    {"text": "Translate this paragraph to Spanish",                       "label": "general"},
    {"text": "What is the difference between async and threading?",       "label": "general"},
    {"text": "Generate a regex to match valid email addresses",           "label": "general"},
    {"text": "Give me 5 name ideas for my SaaS startup",                  "label": "general"},
    {"text": "What are Python type hints and why should I use them?",     "label": "general"},
    {"text": "How does garbage collection work in Java?",                 "label": "general"},
    {"text": "Write unit tests for this authentication module",           "label": "general"},
]


# ─────────────────────────────────────────────
# Pipeline Factory
# ─────────────────────────────────────────────

def build_pipeline(
    max_features: int   = 8_000,
    ngram_range:  tuple = (1, 2),
    C:            float = 1.0,
    max_iter:     int   = 1_000,
    class_weight: str   = "balanced",
) -> Any:
    """
    Construct a fresh (untrained) sklearn Pipeline.

    Architecture:
        TfidfVectorizer  →  LogisticRegression

    Args:
        max_features:  Vocabulary cap for the TF-IDF step.
        ngram_range:   Unigrams + bigrams by default.
        C:             Inverse regularisation strength for LR.
        max_iter:      Solver iteration limit.
        class_weight:  "balanced" corrects for class imbalance automatically.

    Returns:
        sklearn Pipeline (unfitted).
    """
    from importlib import import_module
    TfidfVectorizer = import_module("sklearn.feature_extraction.text").TfidfVectorizer
    LogisticRegression = import_module("sklearn.linear_model").LogisticRegression
    Pipeline = import_module("sklearn.pipeline").Pipeline
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,       # apply log(1+tf) – helps with long prompts
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]*\b",
        min_df=1,
    )

    lr = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([("tfidf", tfidf), ("clf", lr)], memory=None)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(
    texts:          list[str],
    labels:         list[str],
    test_size:      float = 0.20,
    cv_folds:       int   = 5,
    save:           bool  = True,
    model_path:     Path  = MODEL_PATH,
    encoder_path:   Path  = ENCODER_PATH,
    meta_path:      Path  = META_PATH,
    pipeline_kwargs: Optional[dict] = None,
) -> dict:
    """
    Train the TF-IDF + LogisticRegression pipeline on ``texts`` / ``labels``.

    Steps:
        1. Encode string labels to integers.
        2. Stratified train / test split.
        3. Fit the Pipeline on the training set.
        4. Evaluate on the hold-out test set.
        5. Run k-fold cross-validation for a more robust accuracy estimate.
        6. (Optional) Persist artefacts with joblib.

    Args:
        texts:           List of raw prompt strings.
        labels:          Corresponding feature-name labels (same length).
        test_size:       Fraction of data held out for evaluation.
        cv_folds:        Number of stratified CV folds.
        save:            Whether to write artefacts to disk.
        model_path:      Destination for the fitted Pipeline.
        encoder_path:    Destination for the fitted LabelEncoder.
        meta_path:       Destination for the JSON metadata file.
        pipeline_kwargs: Dict of keyword args forwarded to build_pipeline().

    Returns:
        Evaluation report dict with keys:
            accuracy, f1_weighted, cv_mean, cv_std,
            classification_report, confusion_matrix,
            classes, n_train, n_test, trained_at, model_path
    """
    if len(texts) != len(labels):
        raise ValueError(
            f"texts and labels must have the same length "
            f"({len(texts)} vs {len(labels)})."
        )
    if len(texts) < 10:
        raise ValueError(
            "Need at least 10 samples to train a meaningful classifier."
        )

    # ── Label encoding ────────────────────────────────────────────────
    from importlib import import_module
    LabelEncoder = import_module("sklearn.preprocessing").LabelEncoder
    classification_report = import_module("sklearn.metrics").classification_report
    confusion_matrix = import_module("sklearn.metrics").confusion_matrix
    accuracy_score = import_module("sklearn.metrics").accuracy_score
    f1_score = import_module("sklearn.metrics").f1_score
    train_test_split = import_module("sklearn.model_selection").train_test_split
    StratifiedKFold = import_module("sklearn.model_selection").StratifiedKFold
    cross_val_score = import_module("sklearn.model_selection").cross_val_score
    le = LabelEncoder()
    y  = le.fit_transform(labels)

    # ── Split ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y,
        test_size=test_size,
        stratify=y,
        random_state=42,
    )

    logger.info(
        "Training split: %d train / %d test  |  classes: %s",
        len(X_train), len(X_test), list(le.classes_),
    )

    # ── Build & fit ───────────────────────────────────────────────────
    pipe = build_pipeline(**(pipeline_kwargs or {}))
    pipe.fit(X_train, y_train)

    # ── Hold-out evaluation ───────────────────────────────────────────
    y_pred = pipe.predict(X_test)

    acc        = float(accuracy_score(y_test, y_pred))
    f1_w       = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    clf_report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    logger.info("Hold-out accuracy: %.4f   F1 (weighted): %.4f", acc, f1_w)

    # ── Cross-validation ──────────────────────────────────────────────
    cv_scores: np.ndarray = np.array([0.0])
    if cv_folds > 1 and len(texts) >= cv_folds * 2:
        skf       = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, texts, y, cv=skf, scoring="accuracy", n_jobs=-1)
        logger.info(
            "CV-%d accuracy: %.4f ± %.4f",
            cv_folds, cv_scores.mean(), cv_scores.std(),
        )

    # ── Persist artefacts ─────────────────────────────────────────────
    from importlib import import_module
    joblib = import_module("joblib")
    model_path_str = str(model_path)
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, model_path,   compress=3)
        joblib.dump(le,   encoder_path, compress=3)
        logger.info("Model saved  →  %s", model_path)
        logger.info("Encoder saved →  %s", encoder_path)

    # ── Metadata ──────────────────────────────────────────────────────
    meta = {
        "trained_at":            datetime.now(tz=timezone.utc).isoformat(),
        "n_samples":             len(texts),
        "n_train":               len(X_train),
        "n_test":                len(X_test),
        "classes":               list(le.classes_),
        "accuracy":              round(acc,  4),
        "f1_weighted":           round(f1_w, 4),
        "cv_mean":               round(float(cv_scores.mean()), 4),
        "cv_std":                round(float(cv_scores.std()),  4),
        "classification_report": clf_report,
        "confusion_matrix":      cm,
        "model_path":            model_path_str,
        "encoder_path":          str(encoder_path),
        "tfidf_vocab_size":      len(pipe.named_steps["tfidf"].vocabulary_),
    }

    if save:
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Metadata saved → %s", meta_path)

    return meta


# ─────────────────────────────────────────────
# Bootstrap Training Helper
# ─────────────────────────────────────────────

def train_on_bootstrap(save: bool = True) -> dict:
    """
    Train the classifier on the built-in bootstrap dataset.

    Useful for:
    - First-run initialisation before real data accumulates.
    - CI/CD smoke-test to verify the pipeline end-to-end.

    Returns:
        Evaluation report dict (same schema as train()).
    """
    texts  = [d["text"]  for d in BOOTSTRAP_DATA]
    labels = [d["label"] for d in BOOTSTRAP_DATA]
    logger.info("Training on %d bootstrap samples …", len(texts))
    return train(texts, labels, save=save)


def train_on_logged_data(
    db_path: str = "data/metrics.db",
    min_samples_per_class: int = 5,
    save: bool = True,
) -> dict:
    """
    Pull real prompts from the SQLite metrics DB and retrain the classifier.

    Falls back to bootstrap data if there are insufficient logged samples.

    Args:
        db_path:               Path to the SmartEdge SQLite database.
        min_samples_per_class: Minimum samples required per class to include it.
        save:                  Persist artefacts to disk.

    Returns:
        Evaluation report dict.
    """
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        df   = pd.read_sql_query(
            "SELECT feature, response_text FROM llm_metrics "
            "WHERE response_text IS NOT NULL AND response_text != ''",
            conn,
        )
        conn.close()
    except Exception as exc:
        logger.warning("Could not read DB (%s). Falling back to bootstrap.", exc)
        return train_on_bootstrap(save=save)

    if df.empty:
        logger.warning("No logged data found. Falling back to bootstrap.")
        return train_on_bootstrap(save=save)

    # Drop classes with too few samples
    counts = df["feature"].value_counts()
    valid  = counts[counts >= min_samples_per_class].index.tolist()
    df     = df[df["feature"].isin(valid)]

    if len(df) < 10:
        logger.warning(
            "Only %d usable logged samples. Falling back to bootstrap.", len(df)
        )
        return train_on_bootstrap(save=save)

    logger.info(
        "Training on %d logged samples across %d classes.",
        len(df), df["feature"].nunique(),
    )
    return train(
        df["response_text"].tolist(),
        df["feature"].tolist(),
        save=save,
    )


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────

def load_model(
    model_path:   Path = MODEL_PATH,
    encoder_path: Path = ENCODER_PATH,
) -> tuple[Any, Any]:
    """
    Load a previously saved Pipeline and LabelEncoder from disk.

    Auto-trains on bootstrap data if artefacts are not found.

    Args:
        model_path:   Path to the joblib Pipeline file.
        encoder_path: Path to the joblib LabelEncoder file.

    Returns:
        (fitted Pipeline, fitted LabelEncoder)

    Raises:
        RuntimeError: If loading fails even after auto-training.
    """
    if not Path(model_path).exists() or not Path(encoder_path).exists():
        logger.info("Model artefacts not found – running bootstrap training …")
        train_on_bootstrap(save=True)

    try:
        from importlib import import_module
        joblib = import_module("joblib")
        pipe    = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        logger.info("Model loaded from %s", model_path)
        return pipe, encoder
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model artefacts from {model_path} / {encoder_path}: {exc}"
        ) from exc


def load_meta(meta_path: Path = META_PATH) -> dict:
    """
    Load the JSON metadata file saved during training.

    Returns an empty dict if the file does not exist yet.
    """
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def predict(
    texts: list[str] | str,
    model_path:   Path = MODEL_PATH,
    encoder_path: Path = ENCODER_PATH,
) -> list[dict]:
    """
    Classify one or more prompt strings.

    Args:
        texts:        A single prompt string or list of prompts.
        model_path:   Path to the saved Pipeline.
        encoder_path: Path to the saved LabelEncoder.

    Returns:
        List of dicts, one per input:
            {
              "text":        original input,
              "predicted":   human-readable class label,
              "confidence":  probability of the predicted class (0–1),
              "probabilities": {class: probability, …}
            }
    """
    if isinstance(texts, str):
        texts = [texts]

    pipe, encoder = load_model(model_path, encoder_path)

    try:
        proba_matrix = pipe.predict_proba(texts)
    except Exception as exc:
        raise RuntimeError("Model pipeline is not fitted.") from exc

    results = []
    for text, proba_row in zip(texts, proba_matrix):
        top_idx   = int(np.argmax(proba_row))
        results.append({
           "text":          text,
           "predicted":     str(encoder.classes_[top_idx]),
           "confidence":    round(float(proba_row[top_idx]), 4),
           "probabilities": {
               str(cls): round(float(p), 4)
               for cls, p in zip(encoder.classes_, proba_row)
            },
        })
 
    return results


def predict_one(
    text: str,
    model_path:   Path = MODEL_PATH,
    encoder_path: Path = ENCODER_PATH,
) -> dict:
    """
    Convenience wrapper: classify a single prompt and return one dict.
    """
    return predict([text], model_path, encoder_path)[0]


# ─────────────────────────────────────────────
# Model Evaluation Report
# ─────────────────────────────────────────────

def evaluate(
    texts:        list[str],
    labels:       list[str],
    model_path:   Path = MODEL_PATH,
    encoder_path: Path = ENCODER_PATH,
) -> dict:
    """
    Evaluate the saved model on a labelled test set without retraining.

    Useful for regression testing after prompt dataset updates.

    Returns:
        Dict with accuracy, f1_weighted, classification_report, confusion_matrix.
    """
    pipe, encoder = load_model(model_path, encoder_path)

    y_true = encoder.transform(labels)
    y_pred = pipe.predict(texts)

    return {
        "accuracy":              round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_weighted":           round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix":      confusion_matrix(y_true, y_pred).tolist(),
        "classes":               list(encoder.classes_),
    }
