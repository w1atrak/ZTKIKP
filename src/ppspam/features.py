"""Text preprocessing, TF-IDF, and feature selection (phase A2, A5)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ppspam import config

PREPROCESS_VERSION = 1
PUBLIC_MODEL_JSON = config.ARTIFACTS_DIR / "public_model.json"

_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b[\w.-]+\.(?:com|co\.uk|net|org|biz|info)\b\S*"
)
_MONEY_RE = re.compile(
    r"[£$€]\s?\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s?(?:p|pence|gbp|pounds?)\b"
)
_PHONE_RE = re.compile(r"\b\d{7,}\b")
_NUM_RE = re.compile(r"\d+")

# TfidfVectorizer arguments that define the feature space; shared by training and
# by the client-side reconstruction from public_model.json.
_VECTORIZER_KEYS = ("ngram_range", "min_df", "max_df", "max_features", "sublinear_tf")


def preprocess(text: str) -> str:
    """Normalize an SMS; the single preprocessing used by both client and server."""
    text = text.lower()
    text = _URL_RE.sub(" urltoken ", text)
    text = _MONEY_RE.sub(" moneytoken ", text)
    text = _PHONE_RE.sub(" phonetoken ", text)
    text = _NUM_RE.sub(" numtoken ", text)
    return text


def make_vectorizer(**params) -> TfidfVectorizer:
    """TF-IDF with the project preprocessing; ``params`` override config defaults."""
    kwargs = {**config.TFIDF_PARAMS, **params}
    return TfidfVectorizer(preprocessor=preprocess, norm="l2", **kwargs)


def export_public_model(
    vectorizer: TfidfVectorizer,
    selected: np.ndarray | None = None,
    path: Path | None = PUBLIC_MODEL_JSON,
    extra: dict | None = None,
) -> dict:
    """Write what the client needs to featurize SMS: vocabulary, IDF, parameters.

    ``selected`` are column indices kept after feature selection (``None`` = all);
    the vector sent to the server has exactly ``len(selected)`` entries.
    """
    terms = [None] * len(vectorizer.vocabulary_)
    for term, idx in vectorizer.vocabulary_.items():
        terms[idx] = term
    params = {k: getattr(vectorizer, k) for k in _VECTORIZER_KEYS}
    params["ngram_range"] = list(params["ngram_range"])
    dim = len(terms) if selected is None else len(selected)
    public = {
        "preprocess_version": PREPROCESS_VERSION,
        "vectorizer": params,
        "terms": terms,
        "idf": vectorizer.idf_.tolist(),
        "selected": None if selected is None else [int(i) for i in selected],
        "d": dim,
        **(extra or {}),
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(public, ensure_ascii=False))
    return public


class PublicFeaturizer:
    """Client-side TF-IDF rebuilt from the exported public model."""

    def __init__(self, public: dict):
        if public["preprocess_version"] != PREPROCESS_VERSION:
            raise ValueError("public model built with a different preprocessing")
        params = dict(public["vectorizer"])
        params["ngram_range"] = tuple(params["ngram_range"])
        vocabulary = {term: i for i, term in enumerate(public["terms"])}
        self.vectorizer = make_vectorizer(vocabulary=vocabulary, **params)
        self.vectorizer.idf_ = np.asarray(public["idf"], dtype=np.float64)
        sel = public["selected"]
        self.selected = None if sel is None else np.asarray(sel, dtype=np.intp)
        self.d = public["d"]

    @classmethod
    def load(cls, path: Path = PUBLIC_MODEL_JSON) -> PublicFeaturizer:
        return cls(json.loads(path.read_text()))

    def transform(self, texts) -> sparse.csr_matrix:
        X = self.vectorizer.transform(texts)
        return X if self.selected is None else X[:, self.selected]
