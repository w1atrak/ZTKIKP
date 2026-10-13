#!/usr/bin/env python3
"""Search S_x/S_w and write quantized server weights (A6).

Expects a model already trained by ``scripts/train.py``.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ppspam import config, data, model, quantize  # noqa: E402


def main() -> None:
    clf = joblib.load(model.MODEL_JOBLIB)
    train, test = data.load_split()
    w, b = model.linear_params(clf)
    X_train = model.featurize(clf, train["text"])
    X_test = model.featurize(clf, test["text"])

    S_x, S_w, rows = quantize.search_scales(X_train, X_test, w, b)
    df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(quantize.QUANTIZE_GRID_CSV, index=False)
    print(df.to_string(index=False))
    print(f"chosen: S_x={S_x:g}, S_w={S_w:g}")

    X_q = quantize.quantize_features(X_test, S_x)
    w_q, b_q_prime = quantize.quantize_weights(w, b, S_x, S_w)
    s_q = quantize.quantized_score(X_q, w_q, b_q_prime)
    peak = quantize.assert_score_in_range(s_q)
    pred_q = model.predict_linear(np.asarray(s_q, dtype=np.float64))
    n_vs_sklearn = int((pred_q != clf.predict(test["text"])).sum())
    payload = quantize.server_weights_q(
        w,
        b,
        S_x,
        S_w,
        extra={"max_abs_s_q_test": peak, "n_mismatch_vs_sklearn_test": n_vs_sklearn},
    )
    quantize.write_server_weights_q(payload)
    limit = quantize.paillier_max_int_lower_bound()
    print(f"max |s_q| on test = {peak} (limit ≥ {limit})")
    print(f"quantized vs sklearn mismatches on test: {n_vs_sklearn}")
    print(f"written: {quantize.SERVER_WEIGHTS_Q_JSON}, {quantize.QUANTIZE_GRID_CSV}")
    print(f"copy into config.py: SCALE_X = {S_x:g}, SCALE_W = {S_w:g}")
    if n_vs_sklearn:
        raise SystemExit("quantized scores disagree with sklearn on the test set")


if __name__ == "__main__":
    main()
