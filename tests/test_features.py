"""Preprocessing and public-model export/import (phase A2)."""

import numpy as np

from ppspam import features

TRAIN = [
    "WINNER!! Claim your £900 prize: call 09061701461 now",
    "Free entry, txt WIN to 87121, 150p/msg www.prize.co.uk",
    "Hey, are we still meeting at 6 tonight?",
    "Ok lar... I'll call you later",
    "URGENT! Visit http://win.example.com to claim 2000 pounds",
]
UNSEEN = ["claim your free prize at www.win.com", "see you at 7", "", "zzz qqq"]


def test_preprocess_tokens():
    out = features.preprocess("Call 09061701461 for £100 at www.x.com, 2 days")
    assert out.split() == [
        "call",
        "phonetoken",
        "for",
        "moneytoken",
        "at",
        "urltoken",
        "numtoken",
        "days",
    ]


def test_preprocess_is_deterministic_and_lowercase():
    assert features.preprocess("HeLLo") == features.preprocess("hello") == "hello"


def _round_trip(vec, selected=None, tmp_path=None):
    path = None if tmp_path is None else tmp_path / "public.json"
    public = features.export_public_model(vec, selected=selected, path=path)
    if path is not None:
        return features.PublicFeaturizer.load(path), public
    return features.PublicFeaturizer(public), public


def test_export_import_identical_vectors(tmp_path):
    vec = features.make_vectorizer(ngram_range=(1, 2)).fit(TRAIN)
    client, public = _round_trip(vec, tmp_path=tmp_path)
    texts = TRAIN + UNSEEN
    assert public["d"] == len(vec.vocabulary_)
    assert np.array_equal(
        client.transform(texts).toarray(), vec.transform(texts).toarray()
    )


def test_export_import_with_selection():
    vec = features.make_vectorizer().fit(TRAIN)
    selected = np.array([0, 3, 5, 7])
    client, public = _round_trip(vec, selected=selected)
    texts = TRAIN + UNSEEN
    expected = vec.transform(texts)[:, selected].toarray()
    out = client.transform(texts)
    assert public["d"] == out.shape[1] == 4
    assert np.array_equal(out.toarray(), expected)
