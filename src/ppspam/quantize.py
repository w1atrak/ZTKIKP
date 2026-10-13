"""Fixed-point quantization and consistency checks (phase A6)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

import numpy as np
from scipy import sparse

from ppspam import config, model

SERVER_WEIGHTS_Q_JSON = config.ARTIFACTS_DIR / "server_weights_q.json"
QUANTIZE_GRID_CSV = config.RESULTS_DIR / "quantize_grid.csv"

SCALE_CANDIDATES: tuple[float, ...] = (1e2, 1e3, 1e4, 1e5, 1e6)


def _as_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X, dtype=np.float64)


def _py_ints(values: Iterable) -> list[int]:
    """Python ``int`` (not numpy) so later Paillier ops cannot overflow int64."""
    return [int(np.rint(v)) for v in values]


def quantize_features(X, S_x: float) -> list[list[int]]:
    """``x_q = round(x · S_x)`` for each row; returns nested Python ints."""
    dense = _as_dense(X)
    return [_py_ints(row * S_x) for row in dense]


def quantize_weights(
    w: Sequence[float], b: float, S_x: float, S_w: float
) -> tuple[list[int], int]:
    """``w_q = round(w · S_w)``, ``b_q' = round(b · S_x · S_w)``."""
    w_q = _py_ints(np.asarray(w, dtype=np.float64) * S_w)
    b_q_prime = int(np.rint(b * S_x * S_w))
    return w_q, b_q_prime


def quantized_score(
    X_q: Sequence[Sequence[int]], w_q: Sequence[int], b_q_prime: int
) -> list[int]:
    """``s_q = b_q' + Σ x_q[i] · w_q[i]`` as Python ints.

    The matmul uses ``int64`` (our scales stay far below 2^63); each result is
    then converted to a Python ``int`` for Paillier.
    """
    Xa = np.asarray(X_q, dtype=np.int64)
    wa = np.asarray(w_q, dtype=np.int64)
    if Xa.ndim != 2 or Xa.shape[1] != wa.shape[0]:
        raise ValueError(f"shape mismatch: X_q {Xa.shape} vs w_q {wa.shape}")
    return [int(v) for v in Xa @ wa + np.int64(b_q_prime)]


def n_sign_mismatches(s_float: Sequence[float], s_q: Sequence[int]) -> int:
    """Disagreements of ``spam ⇔ s > 0`` (same tie-break as the linear form)."""
    pred_f = model.predict_linear(np.asarray(s_float, dtype=np.float64))
    pred_q = model.predict_linear(np.asarray(s_q, dtype=np.float64))
    return int((pred_f != pred_q).sum())


def paillier_max_int_lower_bound(n_length: int = config.KEY_SIZE_PROD) -> int:
    """``n/3`` for an ``n_length``-bit modulus is at least ``2^{n_length-1} / 3``."""
    return (1 << (n_length - 1)) // 3


def assert_score_in_range(
    s_q: Sequence[int], n_length: int = config.KEY_SIZE_PROD
) -> int:
    """Raise if any ``|s_q|`` could overflow ``public_key.max_int``."""
    peak = max((abs(s) for s in s_q), default=0)
    limit = paillier_max_int_lower_bound(n_length)
    if peak >= limit:
        raise OverflowError(
            f"max |s_q| = {peak} is not < Paillier max_int (≥ {limit}) "
            f"for a {n_length}-bit key"
        )
    return peak


def evaluate_scales(X, w: Sequence[float], b: float, S_x: float, S_w: float) -> dict:
    s_float = model.linear_score(X, np.asarray(w, dtype=np.float64), b)
    X_q = quantize_features(X, S_x)
    w_q, b_q_prime = quantize_weights(w, b, S_x, S_w)
    s_q = quantized_score(X_q, w_q, b_q_prime)
    return {
        "S_x": S_x,
        "S_w": S_w,
        "n_mismatch": n_sign_mismatches(s_float, s_q),
        "max_abs_s_q": max((abs(s) for s in s_q), default=0),
        "max_abs_w_q": max((abs(v) for v in w_q), default=0),
    }


def search_scales(
    X_train,
    X_test,
    w: Sequence[float],
    b: float,
    candidates: Sequence[float] = SCALE_CANDIDATES,
) -> tuple[float, float, list[dict]]:
    """Smallest ``(S_w, S_x)`` with zero train *and* test sign mismatches."""
    rows: list[dict] = []
    chosen: tuple[float, float] | None = None
    for S_w in candidates:
        for S_x in candidates:
            train = evaluate_scales(X_train, w, b, S_x, S_w)
            test = evaluate_scales(X_test, w, b, S_x, S_w)
            row = {
                "S_x": S_x,
                "S_w": S_w,
                "n_mismatch_train": train["n_mismatch"],
                "n_mismatch_test": test["n_mismatch"],
                "max_abs_s_q_test": test["max_abs_s_q"],
                "max_abs_w_q": test["max_abs_w_q"],
            }
            rows.append(row)
            if (
                chosen is None
                and row["n_mismatch_train"] == 0
                and row["n_mismatch_test"] == 0
            ):
                chosen = (S_x, S_w)
    if chosen is None:
        raise RuntimeError("no (S_x, S_w) pair in the grid has 0 sign mismatches")
    return chosen[0], chosen[1], rows


def server_weights_q(
    w: Sequence[float],
    b: float,
    S_x: float,
    S_w: float,
    extra: dict | None = None,
) -> dict:
    w_q, b_q_prime = quantize_weights(w, b, S_x, S_w)
    payload = {
        "w_q": w_q,
        "b_q_prime": b_q_prime,
        "S_x": S_x,
        "S_w": S_w,
        "d": len(w_q),
        "spam_iff": "s_q > 0",
        **(extra or {}),
    }
    return payload


def write_server_weights_q(payload: dict, path=SERVER_WEIGHTS_Q_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
