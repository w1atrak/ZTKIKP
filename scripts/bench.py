#!/usr/bin/env python3
"""Performance experiments E2–E7 (W2) and the B6 before/after table.

Every timing is the median and IQR of ``--repeats`` runs (default 10) after one
warm-up. HTTP runs go through a real uvicorn server on loopback.

Writes ``results/bench_e{2..7}.csv`` + ``.png``, ``results/bench_env.json`` and
the summary ``results/bench.md`` (built from whatever CSVs exist).
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import platform
import random
import socket
import sys
import tempfile
import threading
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
# Workers spawned by ProcessPoolExecutor inherit this.
os.environ.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))

import gmpy2  # noqa: E402
import httpx  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import phe  # noqa: E402
import uvicorn  # noqa: E402
from phe import util  # noqa: E402

from ppspam import (  # noqa: E402
    config,
    crypto,
    data,
    features,
    model,
    protocol,
    quantize,
)  # noqa: E402
from ppspam.client import Client  # noqa: E402
from ppspam.server import Server, create_app  # noqa: E402

RESULTS = config.RESULTS_DIR
ENV_JSON = RESULTS / "bench_env.json"
SUMMARY_MD = RESULTS / "bench.md"
FEATURE_DIM_CSV = RESULTS / "feature_dim.csv"
EXPERIMENTS = ("E2", "E3", "E4", "E5", "E6", "E7")

E2_DIMS = (100, 250, 500, 1000, 2000)
E3_KEYS = (config.KEY_SIZE_DEV, config.KEY_SIZE_PROD, config.KEY_SIZE_BENCH)
E4_SCALES_W = (1e2, 1e3, 1e4, 1e5, 1e6)
E5_DIMS = (250, 500, 1000, 2000)
NO_GMP_SLICE = 100  # coordinates encrypted without gmpy2 (then scaled to d)
MAX_ABS_W_Q = 664  # |w_q| at the chosen S_w (A6), for synthetic weights


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #


def measure(fn, repeats: int, setup=None, warmup: int = 1) -> dict:
    """Median and quartiles of ``fn()`` (or ``fn(setup())``) wall time."""
    samples = []
    for i in range(warmup + repeats):
        arg = setup() if setup is not None else None
        started = time.perf_counter()
        fn() if setup is None else fn(arg)
        elapsed = time.perf_counter() - started
        if i >= warmup:
            samples.append(elapsed)
    return stats(samples)


def stats(samples) -> dict:
    values = np.asarray(samples, dtype=np.float64)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"median": float(median), "q1": float(q1), "q3": float(q3), "n": len(values)}


def flat(prefix: str, st: dict) -> dict:
    return {
        f"{prefix}_median_s": st["median"],
        f"{prefix}_q1_s": st["q1"],
        f"{prefix}_q3_s": st["q3"],
    }


def json_bytes(obj) -> int:
    return len(json.dumps(obj, separators=(",", ":")).encode())


def yerr(df: pd.DataFrame, prefix: str) -> np.ndarray:
    med = df[f"{prefix}_median_s"].to_numpy()
    return np.vstack(
        [med - df[f"{prefix}_q1_s"].to_numpy(), df[f"{prefix}_q3_s"].to_numpy() - med]
    )


def write_csv(name: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path = RESULTS / f"bench_{name}.csv"
    df.to_csv(path, index=False)
    print(f"written: {path}", flush=True)
    return df


def save_fig(fig, name: str) -> None:
    path = RESULTS / f"bench_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"written: {path}", flush=True)


@contextlib.contextmanager
def serve(app):
    """Run ``app`` with uvicorn in a thread on a free loopback port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join()
        sock.close()


def post_score(http: httpx.Client, url: str, raw: bytes) -> tuple[dict, float, float]:
    """``(response json, round trip s, server compute s)`` for a prepared body."""
    started = time.perf_counter()
    response = http.post(
        f"{url}/score", content=raw, headers={"content-type": "application/json"}
    )
    round_trip = time.perf_counter() - started
    response.raise_for_status()
    timing = response.headers["server-timing"]
    server_s = float(timing.split("dur=")[1]) / 1000
    return response.json(), round_trip, server_s


def environment(args) -> dict:
    cpu = "unknown"
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    mem_kib = None
    with contextlib.suppress(OSError, ValueError):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                mem_kib = int(line.split()[1])
    release = platform.release()
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "kernel": release,
        "wsl2": "microsoft" in release.lower(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "ram_gib": None if mem_kib is None else round(mem_kib / 2**20, 2),
        "python": platform.python_version(),
        "phe": phe.__version__,
        "gmpy2": gmpy2.version(),
        "numpy": np.__version__,
        "repeats": args.repeats,
        "workers": args.workers,
        "d": config.FEATURE_DIM,
        "S_x": config.SCALE_X,
        "S_w": config.SCALE_W,
    }


# --------------------------------------------------------------------------- #
# Shared state
# --------------------------------------------------------------------------- #


class Bench:
    def __init__(self, args):
        self.repeats = args.repeats
        self.workers = args.workers
        self.quick = args.quick
        self.executor = ProcessPoolExecutor(max_workers=args.workers)
        self.rng = random.Random(config.SEED)
        self._keys: dict[int, tuple] = {}
        self._cache: dict[tuple, dict] = {}
        self.public = json.loads(features.PUBLIC_MODEL_JSON.read_text())
        self.weights_q = json.loads(quantize.SERVER_WEIGHTS_Q_JSON.read_text())
        self.weights_f = json.loads(model.SERVER_WEIGHTS_JSON.read_text())
        self.featurizer = features.PublicFeaturizer(self.public)
        self.texts = list(data.load_split()[1]["text"])
        self.d = int(self.weights_q["d"])
        self.x_q = self.quantized(self.texts[0])

    def close(self) -> None:
        self.executor.shutdown()

    def keypair(self, bits: int):
        if bits not in self._keys:
            self._keys[bits] = crypto.generate_keypair(bits)
        return self._keys[bits]

    def cached(self, key: tuple, fn) -> dict:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def quantized(self, text: str) -> list[int]:
        X = self.featurizer.transform([text])
        return quantize.quantize_features(X, float(self.weights_q["S_x"]))[0]

    def server(self, w_q=None, b_q_prime=None, **kwargs) -> Server:
        w_q = self.weights_q["w_q"] if w_q is None else w_q
        b_q_prime = self.weights_q["b_q_prime"] if b_q_prime is None else b_q_prime
        return Server(
            weights={"w_q": w_q, "b_q_prime": b_q_prime, "d": len(w_q)}, **kwargs
        )

    def synthetic(self, d: int) -> tuple[list[int], list[int]]:
        """Sparse x_q in TF-IDF range and weights of the real bit size."""
        x_q = [0] * d
        for i in self.rng.sample(range(d), min(d, 15)):
            x_q[i] = self.rng.randint(1, int(config.SCALE_X))
        w_q = [self.rng.randint(-MAX_ABS_W_Q, MAX_ABS_W_Q) for _ in range(d)]
        return x_q, w_q

    def serial_encrypt(self, bits: int, d: int) -> dict:
        """Serial gmpy2 encryption of a length-``d`` vector (shared by E2/E3/E7)."""
        public_key, _ = self.keypair(bits)
        vector = (self.x_q if d == self.d else self.synthetic(d)[0])[:d]
        return self.cached(
            ("encrypt_serial", bits, d),
            lambda: measure(
                lambda: crypto.encrypt_vector(public_key, vector), self.repeats
            ),
        )


# --------------------------------------------------------------------------- #
# E2 — d vs quality and cost
# --------------------------------------------------------------------------- #


def run_e2(b: Bench) -> pd.DataFrame:
    dims = E2_DIMS[:2] if b.quick else E2_DIMS
    quality = pd.read_csv(FEATURE_DIM_CSV) if FEATURE_DIM_CSV.exists() else None
    public_key, _ = b.keypair(config.KEY_SIZE_PROD)
    rows = []
    for d in dims:
        print(f"E2 d={d}", flush=True)
        x_q, w_q = b.synthetic(d)
        server = b.server(w_q, 0)
        encrypted = crypto.encrypt_vector_parallel(
            public_key, x_q, b.executor, b.workers
        )
        cts = [c.ciphertext() for c in encrypted]
        row = {"d": d, "key_bits": config.KEY_SIZE_PROD, "n_repeats": b.repeats}
        if quality is not None:
            match = quality[
                (quality.method == config.FEATURE_SELECTION) & (quality.d == d)
            ]
            if len(match):
                row["cv_accuracy"] = float(match.cv_accuracy.iloc[0])
                row["cv_f1"] = float(match.cv_f1.iloc[0])
        row |= flat("encrypt", b.serial_encrypt(config.KEY_SIZE_PROD, d))
        row |= flat(
            "server", measure(lambda: server.score(public_key, encrypted), b.repeats)
        )
        for enc in protocol.WIRE_ENCODINGS:
            body = protocol.score_request(str(public_key.n), cts, "1", encoding=enc)
            row[f"request_bytes_{enc}"] = json_bytes(body)
        rows.append(row)
    df = write_csv("e2", rows)

    fig, (ax_q, ax_c) = plt.subplots(1, 2, figsize=(11, 4))
    if "cv_f1" in df:
        ax_q.plot(df.d, df.cv_accuracy, "o-", label="accuracy (CV)")
        ax_q.plot(df.d, df.cv_f1, "s-", label="F1 spam (CV)")
    ax_q.set_xscale("log")
    ax_q.set_xlabel("d (liczba cech)")
    ax_q.set_ylabel("metryka na train (5-fold CV)")
    ax_q.set_title("E2: jakość vs d")
    ax_q.grid(alpha=0.3)
    ax_q.legend()
    ax_c.errorbar(
        df.d,
        df.encrypt_median_s,
        yerr=yerr(df, "encrypt"),
        fmt="o-",
        capsize=3,
        label="szyfrowanie klienta",
    )
    ax_c.errorbar(
        df.d,
        df.server_median_s,
        yerr=yerr(df, "server"),
        fmt="s-",
        capsize=3,
        label="obliczenia serwera",
    )
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel("d (liczba cech)")
    ax_c.set_ylabel("czas / SMS [s]")
    ax_c.set_title("E2: koszt vs d (klucz 2048)")
    ax_c.grid(alpha=0.3, which="both")
    ax_r = ax_c.twinx()
    ax_r.plot(
        df.d, df.request_bytes_dec / 2**20, "k--", alpha=0.6, label="żądanie [MiB]"
    )
    ax_r.set_ylabel("rozmiar żądania (JSON dziesiętny) [MiB]")
    lines = ax_c.get_legend_handles_labels()
    extra = ax_r.get_legend_handles_labels()
    ax_c.legend(lines[0] + extra[0], lines[1] + extra[1], loc="upper left")
    save_fig(fig, "e2")
    return df


# --------------------------------------------------------------------------- #
# E3 — key size
# --------------------------------------------------------------------------- #


def run_e3(b: Bench) -> pd.DataFrame:
    keys = E3_KEYS[:2] if b.quick else E3_KEYS
    server = b.server()
    rows = []
    for bits in keys:
        print(f"E3 key={bits}", flush=True)
        keygen = measure(lambda bits=bits: crypto.generate_keypair(bits), b.repeats)
        public_key, private_key = b.keypair(bits)
        encrypted = crypto.encrypt_vector_parallel(
            public_key, b.x_q, b.executor, b.workers
        )
        result = server.score(public_key, encrypted)
        cts = [c.ciphertext() for c in encrypted]
        row = {"key_bits": bits, "d": b.d, "n_repeats": b.repeats}
        row |= flat("keygen", keygen)
        row |= flat("encrypt", b.serial_encrypt(bits, b.d))
        row |= flat(
            "server", measure(lambda: server.score(public_key, encrypted), b.repeats)
        )
        row |= flat(
            "decrypt",
            measure(lambda: crypto.decrypt_scalar(private_key, result), b.repeats),
        )
        row["ciphertext_bytes"] = protocol.ciphertext_width(public_key.n)
        row["request_bytes_dec"] = json_bytes(
            protocol.score_request(str(public_key.n), cts, "1")
        )
        row["request_bytes_b64"] = json_bytes(
            protocol.score_request(str(public_key.n), cts, "1", encoding="b64")
        )
        rows.append(row)
    df = write_csv("e3", rows)

    steps = ("keygen", "encrypt", "server", "decrypt")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(df)
    x = np.arange(len(steps))
    for i, (_, row) in enumerate(df.iterrows()):
        med = [row[f"{s}_median_s"] for s in steps]
        err = [
            [row[f"{s}_median_s"] - row[f"{s}_q1_s"] for s in steps],
            [row[f"{s}_q3_s"] - row[f"{s}_median_s"] for s in steps],
        ]
        ax.bar(
            x + i * width,
            med,
            width,
            yerr=err,
            capsize=3,
            label=f"{int(row.key_bits)} bit ({int(row.ciphertext_bytes)} B/szyfrogram)",
        )
    ax.set_xticks(x + width * (len(df) - 1) / 2)
    ax.set_xticklabels(["keygen", f"szyfrowanie\n(d={b.d})", "serwer", "odszyfrowanie"])
    ax.set_yscale("log")
    ax.set_ylabel("czas [s]")
    ax.set_title("E3: koszt vs rozmiar klucza")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend()
    save_fig(fig, "e3")
    return df


# --------------------------------------------------------------------------- #
# E4 — quantization scale vs agreement and server cost
# --------------------------------------------------------------------------- #


def run_e4(b: Bench) -> pd.DataFrame:
    scales = E4_SCALES_W[::2] if b.quick else E4_SCALES_W
    grid = (
        pd.read_csv(quantize.QUANTIZE_GRID_CSV)
        if quantize.QUANTIZE_GRID_CSV.exists()
        else None
    )
    public_key, _ = b.keypair(config.KEY_SIZE_PROD)
    encrypted = crypto.encrypt_vector_parallel(public_key, b.x_q, b.executor, b.workers)
    rows = []
    for S_w in scales:
        print(f"E4 S_w={S_w:g}", flush=True)
        w_q, b_q = quantize.quantize_weights(
            b.weights_f["w"], b.weights_f["b"], config.SCALE_X, S_w
        )
        server = b.server(w_q, b_q)
        row = {
            "S_x": config.SCALE_X,
            "S_w": S_w,
            "max_abs_w_q": max(abs(w) for w in w_q),
            "w_q_bits": max(abs(w) for w in w_q).bit_length(),
            "n_repeats": b.repeats,
        }
        if grid is not None:
            match = grid[(grid.S_x == config.SCALE_X) & (grid.S_w == S_w)]
            if len(match):
                row["n_mismatch_train"] = int(match.n_mismatch_train.iloc[0])
                row["n_mismatch_test"] = int(match.n_mismatch_test.iloc[0])
        row |= flat(
            "server", measure(lambda: server.score(public_key, encrypted), b.repeats)
        )
        rows.append(row)
    df = write_csv("e4", rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        df.S_w,
        df.server_median_s * 1000,
        yerr=yerr(df, "server") * 1000,
        fmt="o-",
        capsize=3,
    )
    for _, row in df.iterrows():
        note = f"{int(row.w_q_bits)} bit"
        if "n_mismatch_test" in row and not pd.isna(row.n_mismatch_test):
            note += f", rozb. test={int(row.n_mismatch_test)}"
        ax.annotate(
            note,
            (row.S_w, row.server_median_s * 1000),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("S_w (skala wag)")
    ax.set_ylabel("obliczenia serwera / SMS [ms]")
    ax.set_title(f"E4: S_w vs czas serwera (S_x={config.SCALE_X:g}, d={b.d}, 2048 bit)")
    ax.grid(alpha=0.3, which="both")
    save_fig(fig, "e4")
    return df


# --------------------------------------------------------------------------- #
# E5 — communication overhead
# --------------------------------------------------------------------------- #


def _synthetic_app(b: Bench, d: int, w_q: list[int], tmp: Path, **kwargs):
    weights = {"w_q": w_q, "b_q_prime": 0, "d": d, "model_version": "bench"}
    public = {
        "preprocess_version": b.public["preprocess_version"],
        "vectorizer": b.public["vectorizer"],
        "terms": [f"t{i}" for i in range(d)],
        "idf": [1.0] * d,
        "selected": None,
        "d": d,
        "S_x": config.SCALE_X,
        "model_version": "bench",
    }
    weights_path = tmp / f"weights_{d}.json"
    public_path = tmp / f"public_{d}.json"
    weights_path.write_text(json.dumps(weights))
    public_path.write_text(json.dumps(public))
    return create_app(
        weights_path=weights_path, public_model_path=public_path, **kwargs
    )


def run_e5(b: Bench) -> pd.DataFrame:
    dims = E5_DIMS[:2] if b.quick else E5_DIMS
    public_key, _ = b.keypair(config.KEY_SIZE_PROD)
    n = str(public_key.n)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for d in dims:
            x_q, w_q = b.synthetic(d)
            cts = [
                c.ciphertext()
                for c in crypto.encrypt_vector_parallel(
                    public_key, x_q, b.executor, b.workers
                )
            ]
            app = _synthetic_app(b, d, w_q, Path(tmp))
            with serve(app) as url, httpx.Client(timeout=600) as http:
                for enc in protocol.WIRE_ENCODINGS:
                    print(f"E5 d={d} encoding={enc}", flush=True)

                    def serialize(enc=enc):
                        body = protocol.score_request(n, cts, "bench", encoding=enc)
                        return json.dumps(body, separators=(",", ":")).encode()

                    raw = serialize()
                    body = json.loads(raw)
                    transfer, server_s, response_bytes = [], [], 0
                    for i in range(b.repeats + 1):
                        payload, round_trip, srv = post_score(http, url, raw)
                        if i:
                            transfer.append(round_trip - srv)
                            server_s.append(srv)
                        response_bytes = json_bytes(payload)
                    row = {
                        "d": d,
                        "encoding": enc,
                        "key_bits": config.KEY_SIZE_PROD,
                        "n_repeats": b.repeats,
                        "request_bytes": len(raw),
                        "bytes_per_ciphertext": len(raw) / d,
                        "response_bytes": response_bytes,
                    }
                    row |= flat("serialize", measure(serialize, b.repeats))
                    row |= flat(
                        "parse",
                        measure(
                            lambda body=body: protocol.parse_score_request(body),
                            b.repeats,
                        ),
                    )
                    row |= flat("transfer", stats(transfer))
                    row |= flat("server_http", stats(server_s))
                    rows.append(row)
    df = write_csv("e5", rows)

    fig, (ax_b, ax_t) = plt.subplots(1, 2, figsize=(11, 4))
    for enc, label in (("dec", "JSON dziesiętny"), ("b64", "JSON base64")):
        part = df[df.encoding == enc]
        ax_b.plot(part.d, part.request_bytes / 2**20, "o-", label=label)
        ax_t.errorbar(
            part.d,
            part.transfer_median_s * 1000,
            yerr=yerr(part, "transfer") * 1000,
            fmt="o-",
            capsize=3,
            label=f"transfer, {label}",
        )
        ax_t.errorbar(
            part.d,
            (part.serialize_median_s + part.parse_median_s) * 1000,
            fmt="s--",
            capsize=3,
            label=f"serializacja+parsowanie, {label}",
        )
    ax_b.set_xlabel("d (liczba szyfrogramów)")
    ax_b.set_ylabel("rozmiar żądania [MiB]")
    ax_b.set_title("E5: bajty / SMS (klucz 2048)")
    ax_b.grid(alpha=0.3)
    ax_b.legend()
    ax_t.set_xlabel("d (liczba szyfrogramów)")
    ax_t.set_ylabel("czas [ms]")
    ax_t.set_title("E5: transfer HTTP (loopback) i serializacja")
    ax_t.grid(alpha=0.3)
    ax_t.legend(fontsize=8)
    save_fig(fig, "e5")
    return df


# --------------------------------------------------------------------------- #
# E6 — total overhead vs plaintext
# --------------------------------------------------------------------------- #

E6_STEPS = (
    "featurize",
    "quantize",
    "encrypt",
    "serialize",
    "transfer",
    "server",
    "decrypt",
)


def _encrypted_run(client: Client, http, url: str, text: str) -> dict:
    payload = client.prepare(text)
    t = dict(client.timings)
    started = time.perf_counter()
    body = protocol.score_request(
        payload["public_key"]["n"],
        [pair[0] for pair in payload["ciphertexts"]],
        client.model_version,
        encoding=client.encoding,
    )
    raw = json.dumps(body, separators=(",", ":")).encode()
    t["serialize_s"] += time.perf_counter() - started
    response, round_trip, server_s = post_score(http, url, raw)
    t["transfer_s"] = round_trip - server_s
    t["server_s"] = server_s
    started = time.perf_counter()
    client.decide_response(response)
    t["decrypt_s"] = time.perf_counter() - started
    t["total_s"] = sum(t[f"{s}_s"] for s in E6_STEPS)
    t["request_bytes"] = len(raw)
    return t


def _plain_run(b: Bench, w: np.ndarray, bias: float, text: str) -> None:
    X = b.featurizer.transform([text])
    model.predict_linear(model.linear_score(X, w, bias))


def _peak_bytes(fn) -> int:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def run_e6(b: Bench) -> pd.DataFrame:
    keypair = b.keypair(config.KEY_SIZE_PROD)
    w = np.asarray(b.weights_f["w"], dtype=np.float64)
    bias = float(b.weights_f["b"])
    texts = b.texts
    rows = []

    print("E6 plain", flush=True)
    plain_repeats = b.repeats * 20
    counter = itertools.count()
    plain = measure(
        lambda: _plain_run(b, w, bias, texts[next(counter) % len(texts)]),
        plain_repeats,
    )
    rows.append(
        {
            "variant": "jawny",
            "n_repeats": plain_repeats,
            **flat("total", plain),
            "throughput_sms_s": 1 / plain["median"],
            "peak_mem_bytes": _peak_bytes(lambda: _plain_run(b, w, bias, texts[0])),
        }
    )

    variants = (
        ("zaszyfrowany: bazowy", {}, {}, False),
        (
            "zaszyfrowany: równoległy + b64",
            {"executor": b.executor, "chunks": b.workers, "encoding": "b64"},
            {"executor": b.executor, "chunks": b.workers},
            False,
        ),
        (
            "zaszyfrowany: prekomputacja + b64",
            {"executor": b.executor, "chunks": b.workers, "encoding": "b64"},
            {"executor": b.executor, "chunks": b.workers},
            True,
        ),
    )
    for name, client_kw, server_kw, precompute in variants:
        print(f"E6 {name}", flush=True)
        client = Client(public_model=b.public, keypair=keypair, **client_kw)
        app = create_app(**server_kw)
        samples: dict[str, list[float]] = {f"{s}_s": [] for s in (*E6_STEPS, "total")}
        offline = []
        request_bytes = 0
        with serve(app) as url, httpx.Client(timeout=600) as http:
            for i in range(b.repeats + 1):
                if precompute:
                    offline.append(client.precompute(messages=1))
                t = _encrypted_run(client, http, url, texts[i % len(texts)])
                if precompute and client.encrypt_mode != "pool":
                    raise RuntimeError("precomputed run did not use the pool")
                request_bytes = t["request_bytes"]
                if i:
                    for key in samples:
                        samples[key].append(t[key])
            if precompute:
                client.precompute(messages=1)
            bodies = []
            peak_client = _peak_bytes(
                lambda: bodies.append(client.score_body(texts[0]))
            )
        server = b.server(**server_kw)
        peak_server = _peak_bytes(lambda: server.score_protocol(bodies[0]))
        total = stats(samples["total_s"])
        row = {"variant": name, "n_repeats": b.repeats, "request_bytes": request_bytes}
        for step in E6_STEPS:
            row |= flat(step, stats(samples[f"{step}_s"]))
        row |= flat("total", total)
        if offline:
            row |= flat("precompute_offline", stats(offline[1:]))
        row["throughput_sms_s"] = 1 / total["median"]
        row["slowdown_vs_plain"] = total["median"] / plain["median"]
        row["peak_mem_bytes"] = peak_client
        row["peak_mem_server_bytes"] = peak_server
        rows.append(row)
    df = write_csv("e6", rows)

    enc = df[df.variant != "jawny"]
    fig, ax = plt.subplots(figsize=(10, 4))
    left = np.zeros(len(enc))
    labels = {
        "featurize": "preprocess + TF-IDF",
        "quantize": "kwantyzacja",
        "encrypt": "szyfrowanie",
        "serialize": "serializacja",
        "transfer": "transfer",
        "server": "serwer",
        "decrypt": "odszyfrowanie + decyzja",
    }
    for step in E6_STEPS:
        values = enc[f"{step}_median_s"].to_numpy()
        ax.barh(enc.variant, values, left=left, label=labels[step])
        left += values
    for y, (_, row) in enumerate(enc.iterrows()):
        ax.text(
            left[y],
            y,
            f" {row.total_median_s:.2f} s (×{row.slowdown_vs_plain:,.0f})",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("czas end-to-end / SMS [s] (mediana kroków)")
    ax.set_title(
        f"E6: rozbicie czasu (d={b.d}, 2048 bit); "
        f"jawny: {plain['median'] * 1000:.2f} ms"
    )
    ax.set_xlim(0, left.max() * 1.35)
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()
    save_fig(fig, "e6")
    return df


# --------------------------------------------------------------------------- #
# E7 — effect of each optimization (B6 before/after)
# --------------------------------------------------------------------------- #


def run_e7(b: Bench) -> pd.DataFrame:
    bits = config.KEY_SIZE_PROD
    public_key, _ = b.keypair(bits)
    x_q = b.x_q
    server = b.server()
    encrypted = crypto.encrypt_vector_parallel(public_key, x_q, b.executor, b.workers)
    rows = []

    def add(opt, variant, st, unit="s", note="", scale=1.0):
        rows.append(
            {
                "optimization": opt,
                "variant": variant,
                "unit": unit,
                "median": st["median"] * scale,
                "q1": st["q1"] * scale,
                "q3": st["q3"] * scale,
                "n_repeats": st["n"],
                "note": note,
            }
        )

    def value(opt, variant, v, unit):
        add(opt, variant, {"median": v, "q1": v, "q3": v, "n": 1}, unit=unit)

    print("E7 gmpy2", flush=True)
    part = x_q[:NO_GMP_SLICE]
    scaled = (
        f"zmierzone na {NO_GMP_SLICE} współrzędnych, "
        f"przeskalowane ×{b.d // NO_GMP_SLICE}"
    )
    try:
        util.HAVE_GMP = False
        no_gmp_enc = measure(lambda: crypto.encrypt_vector(public_key, part), b.repeats)
        no_gmp_srv = measure(lambda: server.score(public_key, encrypted), b.repeats)
    finally:
        util.HAVE_GMP = True
    gmp_enc = measure(lambda: crypto.encrypt_vector(public_key, part), b.repeats)
    add(
        "gmpy2: szyfrowanie klienta",
        "przed (czysty Python)",
        no_gmp_enc,
        note=scaled,
        scale=b.d / NO_GMP_SLICE,
    )
    add(
        "gmpy2: szyfrowanie klienta",
        "po (gmpy2)",
        gmp_enc,
        note=scaled,
        scale=b.d / NO_GMP_SLICE,
    )
    serial_srv = measure(lambda: server.score(public_key, encrypted), b.repeats)
    add("gmpy2: serwer", "przed (czysty Python)", no_gmp_srv)
    add("gmpy2: serwer", "po (gmpy2)", serial_srv)

    print("E7 parallel client", flush=True)
    serial_enc = b.serial_encrypt(bits, b.d)
    add("równoległe szyfrowanie klienta", "przed (1 proces)", serial_enc)
    add(
        "równoległe szyfrowanie klienta",
        f"po ({b.workers} procesów)",
        measure(
            lambda: crypto.encrypt_vector_parallel(
                public_key, x_q, b.executor, b.workers
            ),
            b.repeats,
        ),
    )

    print("E7 precomputed obfuscators", flush=True)

    def fresh_pool():
        pool = crypto.ObfuscatorPool(public_key)
        pool.fill(b.d, executor=b.executor, chunks=b.workers)
        return pool

    add("prekomputacja r^n (online)", "przed (r^n w trakcie)", serial_enc)
    add(
        "prekomputacja r^n (online)",
        "po (pula offline)",
        measure(
            lambda pool: crypto.encrypt_vector(public_key, x_q, pool=pool),
            b.repeats,
            setup=fresh_pool,
        ),
        note="r^n liczone wcześniej (koszt offline w wierszu niżej)",
    )

    def fill_parallel():
        crypto.ObfuscatorPool(public_key).fill(
            b.d, executor=b.executor, chunks=b.workers
        )

    add(
        "prekomputacja r^n (offline, informacyjnie)",
        f"napełnienie puli, {b.workers} procesów",
        measure(fill_parallel, b.repeats),
    )

    print("E7 parallel server", flush=True)
    add("równoległy serwer", "przed (1 proces)", serial_srv)
    add(
        "równoległy serwer",
        f"po ({b.workers} procesów)",
        measure(
            lambda: server.score(
                public_key, encrypted, executor=b.executor, chunks=b.workers
            ),
            b.repeats,
        ),
    )

    print("E7 bias", flush=True)
    b_q = int(b.weights_q["b_q_prime"])
    add(
        "bias bez obfuskacji",
        "przed (Enc(b) z r^n)",
        measure(lambda: public_key.encrypt(b_q), b.repeats),
    )
    add(
        "bias bez obfuskacji",
        "po (Enc(b) z r=1)",
        measure(lambda: public_key.encrypt(b_q, r_value=1), b.repeats),
        note="oszczędza jedno r^n na żądanie (wynik i tak jest obfuskowany)",
    )

    print("E7 smaller scales", flush=True)
    w_big, b_big = quantize.quantize_weights(
        b.weights_f["w"], b.weights_f["b"], config.SCALE_X, 1e6
    )
    big = b.server(w_big, b_big)
    add(
        "mniejsza skala S_w",
        "przed (S_w=1e6)",
        measure(lambda: big.score(public_key, encrypted), b.repeats),
    )
    add("mniejsza skala S_w", f"po (S_w={config.SCALE_W:g})", serial_srv)

    print("E7 compact serialization", flush=True)
    cts = [c.ciphertext() for c in encrypted]
    for enc, variant in (
        ("dec", "przed (JSON dziesiętny)"),
        ("b64", "po (JSON base64)"),
    ):
        body = protocol.score_request(str(public_key.n), cts, "1", encoding=enc)
        value("kompaktowa serializacja", variant, json_bytes(body), "B")
    df = write_csv("e7", rows)

    timed = df[(df.unit == "s") & df.variant.str.match(r"(przed|po) ")]
    opts = list(dict.fromkeys(timed.optimization))
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(opts))
    for i, label in enumerate(("przed", "po")):
        part = timed[timed.variant.str.startswith(label)]
        part = part.set_index("optimization").reindex(opts)
        med = part["median"].to_numpy()
        err = np.vstack([med - part["q1"].to_numpy(), part["q3"].to_numpy() - med])
        ax.barh(y + (i - 0.5) * 0.4, med, 0.4, xerr=err, capsize=3, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(opts, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("czas / SMS [s]")
    ax.set_title(f"E7: efekt optymalizacji B6 (d={b.d}, 2048 bit)")
    ax.grid(alpha=0.3, axis="x", which="both")
    ax.legend()
    ax.invert_yaxis()
    save_fig(fig, "e7")
    return df


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #


def _read(name: str) -> pd.DataFrame | None:
    path = RESULTS / f"bench_{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def _ms(v: float) -> str:
    if v < 1e-3:
        return f"{v * 1e6:.0f} µs"
    return f"{v * 1000:.1f} ms" if v < 1 else f"{v:.2f} s"


def _size(n_bytes: float) -> str:
    if n_bytes < 2**20:
        return f"{n_bytes / 2**10:.1f} KiB"
    return f"{n_bytes / 2**20:.2f} MiB"


def _iqr(row, prefix: str) -> str:
    return (
        f"{_ms(row[f'{prefix}_median_s'])} "
        f"[{_ms(row[f'{prefix}_q1_s'])}–{_ms(row[f'{prefix}_q3_s'])}]"
    )


def _header(*cols: str, left: int = 0) -> list[str]:
    """Markdown header; the first ``left`` columns are text (left-aligned)."""
    align = " --- |" * left + " ---: |" * (len(cols) - left)
    return ["| " + " | ".join(cols) + " |", "|" + align]


def write_summary() -> None:
    env = json.loads(ENV_JSON.read_text()) if ENV_JSON.exists() else {}
    lines = [
        "# W2 — pomiary wydajności (E2–E7) i optymalizacje B6",
        "",
        "Wszystkie czasy: mediana [Q1–Q3] z powtórzeń po jednej rozgrzewce "
        "(wersja jawna w E6: 20× więcej powtórzeń, bo trwa poniżej 1 ms). "
        "Pamięć to szczyt `tracemalloc` w procesie głównym (bez procesów roboczych). "
        "Wygenerowane przez `make bench` (`scripts/bench.py`).",
        "",
        "## Środowisko",
        "",
        f"- CPU: {env.get('cpu')} ({env.get('logical_cpus')} wątków logicznych), "
        f"RAM: {env.get('ram_gib')} GiB",
        f"- System: {env.get('platform')} (WSL2: {env.get('wsl2')})",
        f"- Python {env.get('python')}, phe {env.get('phe')}, gmpy2 {env.get('gmpy2')}",
        f"- Powtórzenia: {env.get('repeats')}, procesy robocze: {env.get('workers')}, "
        f"d={env.get('d')}, S_x={env.get('S_x')}, S_w={env.get('S_w')}",
        f"- Data: {env.get('timestamp')}",
        "",
        "[!] WSL2 dzieli CPU z Windows; pomiary robione bez innych obciążających "
        "procesów, ale rozrzut (IQR) jest większy niż na natywnym Linuksie.",
        "",
    ]

    df = _read("e2")
    if df is not None:
        lines += [
            "## E2 — wpływ d na jakość i koszt (klucz 2048)",
            "",
            *_header(
                "d",
                "CV acc",
                "CV F1",
                "szyfrowanie / SMS",
                "serwer / SMS",
                "żądanie dec",
                "żądanie b64",
            ),
        ]
        for _, r in df.iterrows():
            acc = (
                f"{r.cv_accuracy:.4f}"
                if "cv_accuracy" in r and not pd.isna(r.cv_accuracy)
                else "–"
            )
            f1 = f"{r.cv_f1:.4f}" if "cv_f1" in r and not pd.isna(r.cv_f1) else "–"
            lines.append(
                f"| {int(r.d)} | {acc} | {f1} | {_iqr(r, 'encrypt')} | "
                f"{_iqr(r, 'server')} | {r.request_bytes_dec / 2**20:.2f} MiB | "
                f"{r.request_bytes_b64 / 2**20:.2f} MiB |"
            )
        lines += ["", "![E2](bench_e2.png)", ""]

    df = _read("e3")
    if df is not None:
        lines += [
            "## E3 — wpływ rozmiaru klucza (d=1000)",
            "",
            *_header(
                "klucz",
                "keygen",
                "szyfrowanie / SMS",
                "serwer",
                "odszyfrowanie",
                "szyfrogram",
                "żądanie dec",
            ),
        ]
        for _, r in df.iterrows():
            lines.append(
                f"| {int(r.key_bits)} | {_iqr(r, 'keygen')} | {_iqr(r, 'encrypt')} | "
                f"{_iqr(r, 'server')} | {_iqr(r, 'decrypt')} | "
                f"{int(r.ciphertext_bytes)} B | {r.request_bytes_dec / 2**20:.2f} MiB |"
            )
        lines += ["", "![E3](bench_e3.png)", ""]

    df = _read("e4")
    if df is not None:
        lines += [
            "## E4 — skala kwantyzacji wag S_w (S_x=1e3, klucz 2048)",
            "",
            *_header(
                "S_w",
                "max abs(w_q)",
                "bity",
                "rozbieżności train",
                "rozbieżności test",
                "serwer / SMS",
            ),
        ]
        for _, r in df.iterrows():
            mtr = (
                int(r.n_mismatch_train)
                if "n_mismatch_train" in r and not pd.isna(r.n_mismatch_train)
                else "–"
            )
            mte = (
                int(r.n_mismatch_test)
                if "n_mismatch_test" in r and not pd.isna(r.n_mismatch_test)
                else "–"
            )
            lines.append(
                f"| {r.S_w:g} | {int(r.max_abs_w_q)} | {int(r.w_q_bits)} | {mtr} | "
                f"{mte} | {_iqr(r, 'server')} |"
            )
        lines += ["", "![E4](bench_e4.png)", ""]

    df = _read("e5")
    if df is not None:
        lines += [
            "## E5 — narzut komunikacyjny (HTTP na loopbacku, klucz 2048)",
            "",
            *_header(
                "d",
                "kodowanie",
                "żądanie",
                "B / szyfrogram",
                "odpowiedź",
                "serializacja",
                "parsowanie",
                "transfer",
                left=2,
            ),
        ]
        for _, r in df.iterrows():
            lines.append(
                f"| {int(r.d)} | {r.encoding} | {r.request_bytes / 2**20:.2f} MiB | "
                f"{r.bytes_per_ciphertext:.0f} | {int(r.response_bytes)} B | "
                f"{_iqr(r, 'serialize')} | {_iqr(r, 'parse')} | {_iqr(r, 'transfer')} |"
            )
        lines += [
            "",
            "Transfer = czas żądania HTTP minus czas obliczeń z nagłówka "
            "`Server-Timing`; na loopbacku to głównie koszt stosu HTTP i kopiowania "
            "bajtów, w prawdziwej sieci rośnie liniowo z rozmiarem żądania.",
            "",
            "![E5](bench_e5.png)",
            "",
        ]

    df = _read("e6")
    if df is not None:
        lines += [
            "## E6 — narzut całkowity względem wersji jawnej (d=1000, klucz 2048)",
            "",
            *_header(
                "wariant",
                "end-to-end / SMS",
                "SMS/s",
                "narzut",
                "pamięć klient (peak)",
                "pamięć serwer (peak)",
                left=1,
            ),
        ]
        for _, r in df.iterrows():
            slow = (
                "1×"
                if pd.isna(r.get("slowdown_vs_plain"))
                else f"{r.slowdown_vs_plain:,.0f}×"
            )
            srv = (
                "–"
                if pd.isna(r.get("peak_mem_server_bytes"))
                else _size(r.peak_mem_server_bytes)
            )
            lines.append(
                f"| {r.variant} | {_iqr(r, 'total')} | {r.throughput_sms_s:,.2f} | "
                f"{slow} | {_size(r.peak_mem_bytes)} | {srv} |"
            )
        enc = df[df.variant != "jawny"]
        lines += [
            "",
            "Rozbicie (mediany):",
            "",
            "| wariant | " + " | ".join(E6_STEPS) + " |",
            "| --- |" + " ---: |" * len(E6_STEPS),
        ]
        for _, r in enc.iterrows():
            cells = " | ".join(_ms(r[f"{s}_median_s"]) for s in E6_STEPS)
            lines.append(f"| {r.variant} | {cells} |")
        pre = enc[enc.variant.str.contains("prekomputacja")]
        if len(pre) and "precompute_offline_median_s" in pre:
            lines += [
                "",
                "Wariant z prekomputacją dodatkowo liczy offline pulę `r^n` "
                f"({_iqr(pre.iloc[0], 'precompute_offline')} na SMS), zanim "
                "wiadomość istnieje.",
            ]
        lines += ["", "![E6](bench_e6.png)", ""]

    df = _read("e7")
    if df is not None:
        lines += [
            "## E7 / B6 — efekt optymalizacji (przed / po, d=1000, klucz 2048)",
            "",
            *_header("optymalizacja", "wariant", "mediana [Q1–Q3]", "uwagi", left=2),
        ]
        for _, r in df.iterrows():
            if r.unit == "B":
                cell = f"{r['median'] / 2**20:.3f} MiB"
            else:
                cell = f"{_ms(r['median'])} [{_ms(r.q1)}–{_ms(r.q3)}]"
            note = "" if pd.isna(r.note) else r.note
            lines.append(f"| {r.optimization} | {r.variant} | {cell} | {note} |")
        lines += ["", "Przyspieszenie (mediana przed / po):", ""]
        for opt, part in df.groupby("optimization", sort=False):
            before = part[part.variant.str.startswith("przed")]
            after = part[part.variant.str.startswith("po")]
            if len(before) and len(after):
                ratio = before["median"].iloc[0] / after["median"].iloc[0]
                lines.append(f"- {opt}: {ratio:,.1f}×")
        lines += ["", "![E7](bench_e7.png)", ""]

    SUMMARY_MD.write_text("\n".join(lines))
    print(f"written: {SUMMARY_MD}", flush=True)


# --------------------------------------------------------------------------- #


RUNNERS = {
    "E2": run_e2,
    "E3": run_e3,
    "E4": run_e4,
    "E5": run_e5,
    "E6": run_e6,
    "E7": run_e7,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--only",
        default=",".join(EXPERIMENTS),
        help="comma-separated subset, e.g. E3,E7",
    )
    parser.add_argument(
        "--quick", action="store_true", help="smaller grids (smoke run, not for report)"
    )
    args = parser.parse_args(argv)
    selected = [name.strip().upper() for name in args.only.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(EXPERIMENTS))
    if unknown:
        raise SystemExit(f"unknown experiments: {', '.join(unknown)}")
    if args.repeats < 10 and not args.quick:
        raise SystemExit("W2 needs >= 10 repeats (use --quick for a smoke run)")
    for path in (
        features.PUBLIC_MODEL_JSON,
        quantize.SERVER_WEIGHTS_Q_JSON,
        model.SERVER_WEIGHTS_JSON,
        data.TEST_CSV,
    ):
        if not path.exists():
            raise SystemExit(f"missing {path} — run make data train quantize")
    if not util.HAVE_GMP:
        raise SystemExit("gmpy2 is not active in phe — run make verify-gmpy2")

    RESULTS.mkdir(parents=True, exist_ok=True)
    ENV_JSON.write_text(json.dumps(environment(args), indent=2))
    bench = Bench(args)
    try:
        for name in EXPERIMENTS:
            if name in selected:
                started = time.perf_counter()
                RUNNERS[name](bench)
                print(
                    f"{name} done in {time.perf_counter() - started:.0f} s", flush=True
                )
    finally:
        bench.close()
    write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
