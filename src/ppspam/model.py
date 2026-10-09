"""MultinomialNB training, linear form (w, b), plain predictions (phase A3, A4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

from ppspam import config, features

MODEL_JOBLIB = config.ARTIFACTS_DIR / "model.joblib"
PLAIN_METRICS_JSON = config.RESULTS_DIR / "plain_metrics.json"
CV_RESULTS_CSV = config.RESULTS_DIR / "cv_results.csv"

PARAM_GRID: dict = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "tfidf__sublinear_tf": [False, True],
    "nb__alpha": [0.01, 0.03, 0.1, 0.3, 1.0],
}


def labels_to_y(labels: pd.Series) -> np.ndarray:
    """``spam`` → 1, ``ham`` → 0, so that ``classes_ == [0, 1]``."""
    return (labels == "spam").to_numpy(dtype=np.int64)


def build_pipeline(classifier=None, selector="passthrough", **tfidf_params) -> Pipeline:
    clf = MultinomialNB(alpha=config.NB_ALPHA) if classifier is None else classifier
    return Pipeline(
        [
            ("tfidf", features.make_vectorizer(**tfidf_params)),
            ("select", selector),
            ("nb", clf),
        ]
    )


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)


def tune(texts, y, param_grid: dict = PARAM_GRID) -> GridSearchCV:
    """Grid search on the training set only (5-fold stratified CV, refit on F1)."""
    search = GridSearchCV(
        build_pipeline(),
        param_grid,
        scoring={"accuracy": "accuracy", "f1": "f1"},
        refit="f1",
        cv=cv_splitter(),
        n_jobs=-1,
    )
    return search.fit(texts, y)


def train(texts, y, **pipeline_kwargs) -> Pipeline:
    return build_pipeline(**pipeline_kwargs).fit(texts, y)


def metrics(y_true, y_pred, y_score=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in out.items()}


def evaluate(model: Pipeline, texts, y) -> dict:
    return metrics(y, model.predict(texts), model.predict_proba(texts)[:, 1])


def baselines() -> dict:
    """Reference models sharing the tuned TF-IDF (context for the report)."""
    return {
        "always_ham": DummyClassifier(strategy="constant", constant=0),
        "complement_nb": ComplementNB(alpha=config.NB_ALPHA),
        "logistic_regression": LogisticRegression(
            C=10.0, max_iter=2000, random_state=config.SEED
        ),
    }


def evaluate_baselines(train_texts, y_train, test_texts, y_test) -> dict:
    out = {}
    for name, clf in baselines().items():
        model = train(train_texts, y_train, classifier=clone(clf))
        out[name] = evaluate(model, test_texts, y_test)
    return out
