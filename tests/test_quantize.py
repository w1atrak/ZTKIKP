"""Quantization correctness (phase A6)."""

import numpy as np
import pytest
from scipy import sparse

from ppspam import config, data, model, quantize


def test_quantize_returns_python_ints():
    X = np.array([[0.25, -1.6], [0.0, 0.49]])
    X_q = quantize.quantize_features(X, 10)
    w_q, b_q = quantize.quantize_weights([1.2, -0.4], 0.51, S_x=10, S_w=100)
    assert X_q == [[2, -16], [0, 5]]
    assert w_q == [120, -40]
    assert b_q == 510
    assert all(type(v) is int for row in X_q for v in row)
    assert all(type(v) is int for v in w_q)
    assert type(b_q) is int


def test_quantize_features_accepts_sparse():
    X = sparse.csr_matrix([[0.0, 0.4], [1.0, 0.0]])
    assert quantize.quantize_features(X, 100) == [[0, 40], [100, 0]]


def test_quantized_score_is_integer_dot_product():
    X_q = [[1, 2, 3], [0, -4, 5]]
    w_q = [10, -1, 2]
    s_q = quantize.quantized_score(X_q, w_q, b_q_prime=7)
    assert s_q == [7 + 10 - 2 + 6, 7 + 0 + 4 + 10]


def test_small_scale_can_flip_sign():
    X = np.array([[1.0]])
    w, b = np.array([0.04]), 0.0
    s_float = model.linear_score(X, w, b)
    assert (s_float > 0).all()
    X_q = quantize.quantize_features(X, S_x=10)
    w_q, b_q = quantize.quantize_weights(w, b, S_x=10, S_w=10)
    s_q = quantize.quantized_score(X_q, w_q, b_q)
    assert quantize.n_sign_mismatches(s_float, s_q) == 1


def test_search_picks_smallest_zero_mismatch_pair():
    rng = np.random.default_rng(0)
    X_train = rng.random((40, 5))
    X_test = rng.random((20, 5))
    w = rng.normal(size=5)
    b = 0.2
    S_x, S_w, rows = quantize.search_scales(
        X_train, X_test, w, b, candidates=(1e2, 1e3, 1e4)
    )
    zeros = [
        (r["S_x"], r["S_w"])
        for r in rows
        if r["n_mismatch_train"] == 0 and r["n_mismatch_test"] == 0
    ]
    assert zeros
    assert (S_x, S_w) == min(zeros, key=lambda p: (p[1], p[0]))


def test_score_range_assertion():
    quantize.assert_score_in_range([10, -3], n_length=1024)
    with pytest.raises(OverflowError):
        quantize.assert_score_in_range([1 << 1024], n_length=1024)


@pytest.mark.skipif(not data.TEST_CSV.exists(), reason="run scripts/download_data.py")
def test_configured_scales_match_sklearn_on_test():
    train, test = data.load_split()
    clf = model.train(train["text"], model.labels_to_y(train["label"]))
    w, b = model.linear_params(clf)
    X_test = model.featurize(clf, test["text"])
    s_float = model.linear_score(X_test, w, b)
    X_q = quantize.quantize_features(X_test, config.SCALE_X)
    w_q, b_q = quantize.quantize_weights(w, b, config.SCALE_X, config.SCALE_W)
    s_q = quantize.quantized_score(X_q, w_q, b_q)
    assert quantize.n_sign_mismatches(s_float, s_q) == 0
    pred_q = model.predict_linear(np.asarray(s_q, dtype=np.float64))
    np.testing.assert_array_equal(pred_q, clf.predict(test["text"]))
    quantize.assert_score_in_range(s_q)
