#!/usr/bin/env python3
"""Evaluate the plain-text model and reference baselines on the test set (A3)."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib  # noqa: E402

from ppspam import config, data, model  # noqa: E402


def main() -> None:
    train, test = data.load_split()
    y_train = model.labels_to_y(train["label"])
    y_test = model.labels_to_y(test["label"])

    clf = joblib.load(model.MODEL_JOBLIB)
    result = {
        "model": {
            "tfidf": clf.named_steps["tfidf"].get_params(deep=False)
            | {"preprocessor": "ppspam.features.preprocess"},
            "nb_alpha": clf.named_steps["nb"].alpha,
            "d": len(clf.named_steps["nb"].feature_log_prob_[0]),
        },
        "n_test": len(test),
        "multinomial_nb": model.evaluate(clf, test["text"], y_test),
        "baselines": model.evaluate_baselines(
            train["text"], y_train, test["text"], y_test
        ),
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.PLAIN_METRICS_JSON.write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )

    rows = {"multinomial_nb": result["multinomial_nb"], **result["baselines"]}
    print(f"{'model':22s} {'acc':>7s} {'prec':>7s} {'rec':>7s} {'f1':>7s} {'auc':>7s}")
    for name, m in rows.items():
        print(
            f"{name:22s} {m['accuracy']:7.4f} {m['precision']:7.4f} "
            f"{m['recall']:7.4f} {m['f1']:7.4f} {m.get('roc_auc', float('nan')):7.4f}"
        )
    cm = rows["multinomial_nb"]["confusion_matrix"]
    print(f"confusion matrix (NB, rows = true ham/spam): {cm}")
    print(f"written: {model.PLAIN_METRICS_JSON}")
    if result["multinomial_nb"]["accuracy"] <= 0.94:
        raise SystemExit("K1 not met: test accuracy <= 94%")


if __name__ == "__main__":
    main()
