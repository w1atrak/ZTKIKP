#!/usr/bin/env python3
"""Full encrypted evaluation on the frozen test set (A7: K2, K3, K4).

Uses the trained artifacts, a 2048-bit Paillier key, and ProcessPoolExecutor.
Writes ``results/encrypted_eval.csv`` (+ checkpoint resumes) and
``results/encrypted_metrics.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
# Workers spawned by ProcessPoolExecutor inherit this.
os.environ.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from ppspam import config, crypto, data, model, quantize  # noqa: E402
from ppspam.server import Server  # noqa: E402

EVAL_CSV = config.RESULTS_DIR / "encrypted_eval.csv"
METRICS_JSON = config.RESULTS_DIR / "encrypted_metrics.json"
FIELDNAMES = [
    "id",
    "y_true",
    "pred_sklearn",
    "s_float",
    "s_q",
    "s_dec",
    "pred_enc",
    "encrypt_s",
    "server_s",
    "decrypt_s",
]

_WORKER: dict = {}


def _init_worker(n: int, p: int, q: int, w_q: list[int], b_q_prime: int, d: int) -> None:
    from phe import paillier

    public_key = paillier.PaillierPublicKey(n)
    private_key = paillier.PaillierPrivateKey(public_key, p, q)
    _WORKER["public_key"] = public_key
    _WORKER["private_key"] = private_key
    _WORKER["server"] = Server(
        weights={"w_q": w_q, "b_q_prime": b_q_prime, "d": d}
    )


def _score_one(item: tuple[int, list[int]]) -> dict:
    idx, x_q = item
    public_key = _WORKER["public_key"]
    private_key = _WORKER["private_key"]
    server: Server = _WORKER["server"]

    t0 = time.perf_counter()
    encrypted = crypto.encrypt_vector(public_key, x_q)
    t1 = time.perf_counter()
    result = server.score(public_key, encrypted)
    t2 = time.perf_counter()
    s_dec = crypto.decrypt_scalar(private_key, result)
    t3 = time.perf_counter()
    return {
        "id": idx,
        "s_dec": s_dec,
        "encrypt_s": t1 - t0,
        "server_s": t2 - t1,
        "decrypt_s": t3 - t2,
    }


def _load_done_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            done.add(int(row["id"]))
    return done


def _append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in FIELDNAMES})


def _summarize(
    df: pd.DataFrame, plain_metrics: dict | None, key_size: int
) -> dict:
    y_true = df["y_true"].to_numpy()
    pred_sk = df["pred_sklearn"].to_numpy()
    pred_enc = df["pred_enc"].to_numpy()
    s_q = df["s_q"].to_numpy()
    s_dec = df["s_dec"].to_numpy()

    n = len(df)
    k4_ok = int((s_dec == s_q).sum())
    k3_ok = int((pred_enc == pred_sk).sum())
    enc_metrics = model.metrics(y_true, pred_enc)
    sk_metrics = model.metrics(y_true, pred_sk)

    summary = {
        "n": n,
        "key_size": key_size,
        "d": int(config.FEATURE_DIM or 0),
        "S_x": config.SCALE_X,
        "S_w": config.SCALE_W,
        "K4_s_dec_eq_s_q": k4_ok == n,
        "K4_n_match": k4_ok,
        "K3_pred_enc_eq_pred_sklearn": k3_ok == n,
        "K3_n_match": k3_ok,
        "K2_accuracy_encrypted": enc_metrics["accuracy"],
        "K2_pass": enc_metrics["accuracy"] > 0.94,
        "metrics_encrypted": enc_metrics,
        "metrics_sklearn": sk_metrics,
        "metrics_plain_file": plain_metrics,
        "timings_s": {
            "encrypt_median": float(df["encrypt_s"].median()),
            "server_median": float(df["server_s"].median()),
            "decrypt_median": float(df["decrypt_s"].median()),
            "encrypt_sum": float(df["encrypt_s"].sum()),
            "server_sum": float(df["server_s"].sum()),
        },
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--key-size",
        type=int,
        default=config.KEY_SIZE_PROD,
        help=f"Paillier bits (default {config.KEY_SIZE_PROD})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
    )
    parser.add_argument("--limit", type=int, default=None, help="Only first N test SMS")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=EVAL_CSV,
        help="CSV path (also used as checkpoint)",
    )
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoint")
    args = parser.parse_args(argv)

    if not data.TEST_CSV.exists():
        raise SystemExit("missing data/processed/test.csv — run make data")
    if not model.MODEL_JOBLIB.exists():
        raise SystemExit("missing artifacts/model.joblib — run make train")
    if not quantize.SERVER_WEIGHTS_Q_JSON.exists():
        raise SystemExit("missing artifacts/server_weights_q.json — run make quantize")

    train, test = data.load_split()
    if args.limit is not None:
        test = test.iloc[: args.limit].copy()
    test = test.reset_index(drop=True)

    clf = joblib.load(model.MODEL_JOBLIB)
    weights = json.loads(quantize.SERVER_WEIGHTS_Q_JSON.read_text())
    w_q = [int(v) for v in weights["w_q"]]
    b_q_prime = int(weights["b_q_prime"])
    d = int(weights["d"])

    X = model.featurize(clf, test["text"])
    w, b = model.linear_params(clf)
    s_float = model.linear_score(X, w, b)
    pred_sklearn = clf.predict(test["text"])
    y_true = model.labels_to_y(test["label"])
    X_q = quantize.quantize_features(X, float(weights["S_x"]))
    s_q = quantize.quantized_score(X_q, w_q, b_q_prime)

    meta = {
        i: {
            "id": i,
            "y_true": int(y_true[i]),
            "pred_sklearn": int(pred_sklearn[i]),
            "s_float": float(s_float[i]),
            "s_q": int(s_q[i]),
        }
        for i in range(len(test))
    }

    out_path: Path = args.out
    if args.fresh and out_path.exists():
        out_path.unlink()
    done = _load_done_ids(out_path)
    pending = [(i, X_q[i]) for i in range(len(test)) if i not in done]
    print(
        f"encrypted eval: n={len(test)} pending={len(pending)} "
        f"key={args.key_size} workers={args.workers} d={d}",
        flush=True,
    )
    if not pending:
        print("nothing to do — checkpoint already complete", flush=True)
    else:
        print("generating keypair…", flush=True)
        public_key, private_key = crypto.generate_keypair(args.key_size)
        initargs = (
            int(public_key.n),
            int(private_key.p),
            int(private_key.q),
            w_q,
            b_q_prime,
            d,
        )
        buffer: list[dict] = []
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=initargs,
        ) as pool:
            futures = {pool.submit(_score_one, item): item[0] for item in pending}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="encrypted"
            ):
                result = future.result()
                idx = result["id"]
                row = {
                    **meta[idx],
                    "s_dec": int(result["s_dec"]),
                    "pred_enc": int(
                        model.predict_linear(
                            np.asarray([result["s_dec"]], dtype=np.float64)
                        )[0]
                    ),
                    "encrypt_s": result["encrypt_s"],
                    "server_s": result["server_s"],
                    "decrypt_s": result["decrypt_s"],
                }
                buffer.append(row)
                if len(buffer) >= args.checkpoint_every:
                    _append_rows(out_path, buffer)
                    buffer.clear()
        if buffer:
            _append_rows(out_path, buffer)

    df = pd.read_csv(out_path).sort_values("id")
    if len(df) != len(test):
        raise SystemExit(
            f"checkpoint incomplete: {len(df)}/{len(test)} rows in {out_path}"
        )

    plain_metrics = None
    if model.PLAIN_METRICS_JSON.exists():
        plain_metrics = json.loads(model.PLAIN_METRICS_JSON.read_text())

    summary = _summarize(df, plain_metrics, args.key_size)
    metrics_path = (
        METRICS_JSON
        if out_path.resolve() == EVAL_CSV.resolve()
        else out_path.with_suffix(".metrics.json")
    )
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"written: {out_path}")
    print(f"written: {metrics_path}")

    if not summary["K4_s_dec_eq_s_q"]:
        raise SystemExit("K4 failed: s_dec != s_q on some samples")
    if not summary["K3_pred_enc_eq_pred_sklearn"]:
        raise SystemExit("K3 failed: pred_enc != pred_sklearn on some samples")
    if args.limit is None and not summary["K2_pass"]:
        raise SystemExit(
            f"K2 failed: encrypted accuracy {summary['K2_accuracy_encrypted']} <= 0.94"
        )
    status = "K3, K4: PASS" if args.limit is not None else "K2, K3, K4: PASS"
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
