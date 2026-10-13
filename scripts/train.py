#!/usr/bin/env python3
"""Train the MultinomialNB model on the frozen train split and save artifacts (A3)."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from ppspam import config, data, features, model, quantize  # noqa: E402


def run_tuning(texts, y) -> None:
    search = model.tune(texts, y)
    cv = pd.DataFrame(search.cv_results_)
    cols = [c for c in cv.columns if c.startswith(("param_", "mean_test", "std_test"))]
    cv = cv[cols].sort_values("mean_test_f1", ascending=False)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv.to_csv(model.CV_RESULTS_CSV, index=False)
    print(cv.head(10).to_string(index=False))
    print(f"best (refit=f1): {search.best_params_}")
    print(f"written: {model.CV_RESULTS_CSV} — copy chosen params into config.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tune", action="store_true", help="grid search on train (5-fold CV) first"
    )
    args = parser.parse_args()

    train, test = data.load_split()
    texts, y = train["text"], model.labels_to_y(train["label"])
    if args.tune:
        run_tuning(texts, y)

    clf = model.train(texts, y)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model.MODEL_JOBLIB)
    public = features.export_public_model(
        clf.named_steps["tfidf"],
        selected=model.selected_columns(clf),
        extra={"S_x": config.SCALE_X, "S_w": config.SCALE_W},
    )
    model.SERVER_WEIGHTS_JSON.write_text(json.dumps(model.server_weights(clf)))
    print(f"trained on {len(texts)} SMS, d = {public['d']}")

    w, b = model.linear_params(clf)
    scores = model.linear_score(model.featurize(clf, test["text"]), w, b)
    n_diff = int((model.predict_linear(scores) != clf.predict(test["text"])).sum())
    stats = {
        "train": model.linear_stats(clf, texts),
        "test": model.linear_stats(clf, test["text"]),
        "test_linear_vs_sklearn_mismatches": n_diff,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.LINEAR_STATS_JSON.write_text(json.dumps(stats, indent=2) + "\n")
    ws, ss = stats["train"]["w"], stats["test"]["abs_score"]
    print(f"w: [{ws['min']:.3f}, {ws['max']:.3f}], b = {b:.4f}")
    print(
        f"test |s|: min {ss['q00']:.4g}, median {ss['q50']:.4g}, "
        f"below 0.1: {ss['n_below_0.1']}"
    )
    print(f"linear vs sklearn mismatches on test: {n_diff}")
    print(
        f"written: {model.MODEL_JOBLIB}, {features.PUBLIC_MODEL_JSON}, "
        f"{model.SERVER_WEIGHTS_JSON}, {model.LINEAR_STATS_JSON}"
    )
    if n_diff:
        raise SystemExit("linear form disagrees with sklearn predictions")

    X_test = model.featurize(clf, test["text"])
    X_q = quantize.quantize_features(X_test, config.SCALE_X)
    w_q, b_q_prime = quantize.quantize_weights(w, b, config.SCALE_X, config.SCALE_W)
    s_q = quantize.quantized_score(X_q, w_q, b_q_prime)
    peak = quantize.assert_score_in_range(s_q)
    n_q = quantize.n_sign_mismatches(scores, s_q)
    quantize.write_server_weights_q(
        quantize.server_weights_q(
            w,
            b,
            config.SCALE_X,
            config.SCALE_W,
            extra={"max_abs_s_q_test": peak, "n_mismatch_test": n_q},
        )
    )
    print(
        f"quantized S_x={config.SCALE_X:g} S_w={config.SCALE_W:g}: "
        f"{n_q} sign mismatches on test, max |s_q|={peak}"
    )
    print(f"written: {quantize.SERVER_WEIGHTS_Q_JSON}")
    if n_q:
        raise SystemExit("quantized scores disagree with the float linear form")


if __name__ == "__main__":
    main()
