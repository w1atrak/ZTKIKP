#!/usr/bin/env python3
"""Feature-dimension experiment: CV quality vs d for each selection method (A5).

Selection runs inside the CV pipeline (fit on training folds only). The
encryption-cost axis uses a quick median of 2048-bit Paillier encryptions; the
full primitive benchmark is experiment E1 (phase B1).
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from phe import paillier  # noqa: E402
from sklearn.model_selection import cross_validate  # noqa: E402

from ppspam import config, data, features, model  # noqa: E402

DIMS = (100, 250, 500, 1000, 2000, 5000, None)
RESULTS_CSV = config.RESULTS_DIR / "feature_dim.csv"
RESULTS_PNG = config.RESULTS_DIR / "feature_dim.png"
CIPHERTEXT_BYTES = 2 * config.KEY_SIZE_PROD // 8


def encrypt_seconds(n_ops: int, key_bits: int = config.KEY_SIZE_PROD) -> float:
    pk, _ = paillier.generate_paillier_keypair(n_length=key_bits)
    times = []
    for i in range(n_ops):
        start = time.perf_counter()
        pk.encrypt(i * 7919)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def run_cv(texts, y) -> pd.DataFrame:
    rows = []
    for method in features.SELECTION_METHODS:
        for k in DIMS:
            if k is None and method != features.SELECTION_METHODS[0]:
                continue
            pipe = model.build_pipeline(method=method, k=k)
            cv = cross_validate(
                pipe, texts, y, cv=model.cv_splitter(), scoring=("accuracy", "f1")
            )
            d = k or len(model.build_pipeline(k=None).fit(texts, y)[0].vocabulary_)
            rows.append(
                {
                    "method": "full" if k is None else method,
                    "d": d,
                    "cv_accuracy": cv["test_accuracy"].mean(),
                    "cv_accuracy_std": cv["test_accuracy"].std(),
                    "cv_f1": cv["test_f1"].mean(),
                    "cv_f1_std": cv["test_f1"].std(),
                }
            )
            print(
                f"{rows[-1]['method']:13s} d={d:6d} acc={rows[-1]['cv_accuracy']:.4f} "
                f"f1={rows[-1]['cv_f1']:.4f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, t_enc: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    full = df[df["method"] == "full"].iloc[0]
    for method in features.SELECTION_METHODS:
        sub = df[df["method"] == method].sort_values("d")
        ax.errorbar(
            sub["d"],
            sub["cv_f1"],
            yerr=sub["cv_f1_std"],
            marker="o",
            capsize=3,
            label=f"F1 ({method})",
        )
    ax.axhline(full["cv_f1"], color="grey", ls=":", label=f"F1 full (d={full['d']})")
    ax.set_xscale("log")
    ax.set_xlabel("d (feature vector length)")
    ax.set_ylabel("5-fold CV F1 (spam), train split")
    ax.grid(True, which="both", alpha=0.3)

    ax2 = ax.twinx()
    dims = sorted(df["d"].unique())
    ax2.plot(dims, [d * t_enc for d in dims], "k--", label="est. encryption time")
    ax2.set_ylabel(
        f"est. client encryption time per SMS [s] ({config.KEY_SIZE_PROD} bit)"
    )

    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="lower right", fontsize=8)
    ax.set_title("Feature selection: quality vs d")
    fig.tight_layout()
    fig.savefig(RESULTS_PNG, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enc-ops", type=int, default=50)
    args = parser.parse_args()

    train, test = data.load_split()
    texts, y = train["text"], model.labels_to_y(train["label"])

    t_enc = encrypt_seconds(args.enc_ops)
    print(f"median encrypt @ {config.KEY_SIZE_PROD} bit: {t_enc * 1e3:.2f} ms")

    df = run_cv(texts, y)
    n_test = len(test)
    df["est_encrypt_s_per_sms"] = df["d"] * t_enc
    df["est_encrypt_h_test_set_1core"] = df["d"] * t_enc * n_test / 3600
    df["payload_kib_per_sms"] = df["d"] * CIPHERTEXT_BYTES / 1024
    df["t_enc_ms"] = t_enc * 1e3

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    plot(df, t_enc)
    with pd.option_context("display.width", 160, "display.precision", 4):
        print(df.drop(columns=["t_enc_ms"]).to_string(index=False))
    print(f"written: {RESULTS_CSV}, {RESULTS_PNG}")


if __name__ == "__main__":
    main()
