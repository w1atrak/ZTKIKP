"""Linear NB form vs sklearn (phase A4)."""

import numpy as np
import pandas as pd
import pytest

from ppspam import data, model


@pytest.fixture(scope="module")
def split():
    if not data.TEST_CSV.exists():
        pytest.skip("run scripts/download_data.py")
    return data.load_split()


@pytest.fixture(scope="module")
def trained(split):
    train, _ = split
    return model.train(train["text"], model.labels_to_y(train["label"]))


def test_spam_is_positive_class(trained):
    assert list(trained.named_steps["nb"].classes_) == [model.HAM, model.SPAM]


def test_labels_to_y():
    y = model.labels_to_y(pd.Series(["ham", "spam", "ham"]))
    assert y.tolist() == [0, 1, 0]


@pytest.mark.parametrize("part", [0, 1], ids=["train", "test"])
def test_predict_linear_matches_sklearn(trained, split, part):
    texts = split[part]["text"]
    w, b = model.linear_params(trained)
    scores = model.linear_score(model.featurize(trained, texts), w, b)
    np.testing.assert_array_equal(model.predict_linear(scores), trained.predict(texts))


def test_score_equals_joint_log_likelihood_difference(trained, split):
    texts = split[1]["text"]
    w, b = model.linear_params(trained)
    scores = model.linear_score(model.featurize(trained, texts), w, b)
    jll = trained.named_steps["nb"].predict_joint_log_proba(
        model.featurize(trained, texts)
    )
    np.testing.assert_allclose(scores, jll[:, 1] - jll[:, 0], rtol=0, atol=1e-9)


def test_tie_goes_to_ham():
    assert model.predict_linear(np.array([0.0, -1e-12, 1e-12])).tolist() == [0, 0, 1]
