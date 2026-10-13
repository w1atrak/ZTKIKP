#!/usr/bin/env python3
"""Estimate encrypted-evaluation cost from the B1 timings (E1).

Reads ``results/bench_primitives.csv`` and writes ``results/budget.md``.
One SMS costs ``d`` encryptions; the server then does one plaintext
multiplication per coordinate (weights here fit in 16 bits).
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from ppspam import config  # noqa: E402

DIMS = (500, 1000, 2000, 5000, 9000)
NOTE_PATH = config.RESULTS_DIR / "budget.md"
CHOSEN_D = config.FEATURE_DIM


def _median(df: pd.DataFrame, operation: str, k_bits: int | None = None) -> float:
    mask = (
        (df.n_length == config.KEY_SIZE_PROD)
        & df.gmpy2.eq(True)
        & (df.operation == operation)
    )
    if k_bits is not None:
        mask &= df.k_bits == k_bits
    hits = df.loc[mask, "median_s"]
    if hits.empty:
        raise SystemExit(f"missing B1 row: {operation} k_bits={k_bits}")
    return float(hits.iloc[0])


def _fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 90:
        return f"{seconds:.1f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.1f} min"
    return f"{minutes / 60:.1f} h"


def main() -> None:
    df = pd.read_csv(config.RESULTS_DIR / "bench_primitives.csv")
    t_enc = _median(df, "encrypt")
    t_mul = _median(df, "mul", k_bits=16)
    ct_row = df.loc[
        (df.n_length == config.KEY_SIZE_PROD)
        & df.gmpy2.eq(True)
        & (df.operation == "encrypt"),
        "ciphertext_bytes",
    ]
    ct_bytes = int(ct_row.iloc[0])
    stats = json.loads((config.RESULTS_DIR / "data_stats.json").read_text())
    n_test = int(stats["test"]["total"])

    rows = []
    for d in DIMS:
        rows.append(
            {
                "d": d,
                "t_sms_enc_s": d * t_enc,
                "t_test_enc_s": n_test * d * t_enc,
                "t_sms_server_s": d * t_mul,
                "bytes_per_sms": d * ct_bytes,
            }
        )

    lines = [
        "# E1 — budżet obliczeniowy (klucz 2048, gmpy2)",
        "",
        f"Koszt jednego szyfrogramu: mediana `encrypt` = {t_enc:.6f} s "
        f"(B1, obfuskacja włączona). Rozmiar szyfrogramu: {ct_bytes} B. "
        f"Zbiór testowy: {n_test} SMS-ów.",
        "",
        "Czas szyfrowania jednego SMS-a to `d × t_enc` "
        "(każda współrzędna, także zera). "
        "Po stronie serwera dominuje `enc * w_q`; wagi po A6 mają ≤ 10 bitów, "
        f"więc bierzemy medianę mnożenia przez 16-bitową skalę ({t_mul:.6f} s).",
        "",
        "| d | 1 SMS (encrypt) | cały test (encrypt) "
        "| 1 SMS (serwer) | rozmiar / SMS |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['d']} | {_fmt_duration(row['t_sms_enc_s'])} | "
            f"{_fmt_duration(row['t_test_enc_s'])} | "
            f"{_fmt_duration(row['t_sms_server_s'])} | "
            f"{row['bytes_per_sms'] / 1e6:.2f} MB |"
        )
    chosen = next(r for r in rows if r["d"] == CHOSEN_D)
    lines += [
        "",
        "## Decyzja D5",
        "",
        f"`d = {CHOSEN_D}` (już wybrane w A5, w zakresie D5 {{500, 1000, 2000}}). "
        "Zaszyfrowanie całego testu trwa około "
        f"{_fmt_duration(chosen['t_test_enc_s'])}, "
        f"a jeden SMS to {chosen['bytes_per_sms'] / 1e6:.2f} MB. "
        "Przy `d = 5000` i `9000` sam klient spędza na szyfrowaniu wiele godzin, "
        "a A5 pokazało, że zejście do 1000 cech nie psuje accuracy modelu jawnego. "
        "Zostawiamy `FEATURE_DIM = 1000`.",
        "",
    ]
    NOTE_PATH.write_text("\n".join(lines))
    print(NOTE_PATH.read_text())


if __name__ == "__main__":
    main()
