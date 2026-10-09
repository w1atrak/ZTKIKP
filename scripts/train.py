#!/usr/bin/env python3
"""Train the MultinomialNB model on the frozen train split and save artifacts (A3)."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from ppspam import config, data, features, model  # noqa: E402


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

    train, _ = data.load_split()
    texts, y = train["text"], model.labels_to_y(train["label"])
    if args.tune:
        run_tuning(texts, y)

    clf = model.train(texts, y)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model.MODEL_JOBLIB)
    public = features.export_public_model(clf.named_steps["tfidf"])
    print(f"trained on {len(texts)} SMS, d = {public['d']}")
    print(f"written: {model.MODEL_JOBLIB}, {features.PUBLIC_MODEL_JSON}")


if __name__ == "__main__":
    main()
