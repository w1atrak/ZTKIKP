#!/usr/bin/env python3
"""Benchmark elementary Paillier operations and check homomorphisms (B1).

Writes ``results/bench_primitives.csv`` (median of 100 repeats, with and
without gmpy2) and a short note ``results/bench_primitives.md``.
"""

import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
from phe import paillier, util  # noqa: E402

from ppspam import config  # noqa: E402

KEY_SIZES = (config.KEY_SIZE_DEV, config.KEY_SIZE_PROD, config.KEY_SIZE_BENCH)
N_REPEATS = 100
N_KEYGEN = 5
K_BITS = (8, 16, 24, 32)
WARMUP = 3

CSV_PATH = config.RESULTS_DIR / "bench_primitives.csv"
NOTE_PATH = config.RESULTS_DIR / "bench_primitives.md"


def _median(samples: list[float]) -> float:
    return statistics.median(samples)


def _time_calls(fn, n: int = N_REPEATS) -> list[float]:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


def _ciphertext_bytes(public_key, n: int = 20) -> int:
    lengths = []
    for i in range(n):
        ct = public_key.encrypt(i).ciphertext()
        lengths.append((ct.bit_length() + 7) // 8)
    return int(statistics.median(lengths))


def _check_values(public_key, private_key) -> list[str]:
    """Encrypt/decrypt integers, a negative, zero and a float; check homs."""
    lines = []
    cases = [("int", 42, 42), ("negative", -7, -7), ("zero", 0, 0), ("float", 1.5, 1.5)]
    for name, plain, expected in cases:
        got = private_key.decrypt(public_key.encrypt(plain))
        if isinstance(expected, float):
            ok = abs(got - expected) < 1e-9
        else:
            ok = got == expected
        if not ok:
            raise SystemExit(f"decrypt({name}) = {got!r}, expected {expected!r}")
        lines.append(f"- Dec(Enc({plain})) = {got}")

    a, b, k = 12, -5, 9
    enc_sum = public_key.encrypt(a) + public_key.encrypt(b)
    if private_key.decrypt(enc_sum) != a + b:
        raise SystemExit("additive homomorphism failed")
    lines.append(f"- Dec(Enc({a})+Enc({b})) = {a + b}")

    for scalar in (k, -k):
        got = private_key.decrypt(public_key.encrypt(a) * scalar)
        if got != a * scalar:
            raise SystemExit(f"scalar homomorphism failed for k={scalar}: {got}")
        lines.append(f"- Dec(Enc({a})*{scalar}) = {a * scalar}")
    return lines


def _row(n_length, gmpy2, operation, samples, **extra) -> dict:
    return {
        "n_length": n_length,
        "gmpy2": gmpy2,
        "operation": operation,
        "n_repeats": len(samples),
        "median_s": _median(samples),
        "p95_s": sorted(samples)[max(0, int(len(samples) * 0.95) - 1)],
        **extra,
    }


def _bench_key(n_length: int) -> tuple[list[dict], list[str]]:
    rows = []
    notes = []
    for use_gmp in (True, False):
        util.HAVE_GMP = use_gmp
        samples = _time_calls(
            lambda: paillier.generate_paillier_keypair(n_length=n_length),
            n=N_KEYGEN,
        )
        rows.append(
            _row(n_length, use_gmp, "keygen", samples, k_bits="", ciphertext_bytes="")
        )

    util.HAVE_GMP = True
    public_key, private_key = paillier.generate_paillier_keypair(n_length=n_length)
    ct_bytes = _ciphertext_bytes(public_key)
    expected = (2 * n_length) // 8
    notes.append(
        f"- klucz {n_length}: mediana rozmiaru szyfrogramu {ct_bytes} B "
        f"(oczekiwane ~{expected} B = 2·n_length bitów)"
    )
    if abs(ct_bytes - expected) > expected * 0.05:
        raise SystemExit(
            f"ciphertext size {ct_bytes} B is far from ~{expected} B for {n_length}"
        )

    left = public_key.encrypt(3)
    right = public_key.encrypt(5)
    fresh = public_key.encrypt(11)
    for use_gmp in (True, False):
        util.HAVE_GMP = use_gmp
        enc_samples = _time_calls(lambda: public_key.encrypt(42))
        dec_samples = _time_calls(lambda: private_key.decrypt(fresh))
        add_samples = _time_calls(lambda: left + right)
        rows.append(
            _row(
                n_length,
                use_gmp,
                "encrypt",
                enc_samples,
                k_bits="",
                ciphertext_bytes=ct_bytes,
            )
        )
        rows.append(
            _row(
                n_length,
                use_gmp,
                "decrypt",
                dec_samples,
                k_bits="",
                ciphertext_bytes=ct_bytes,
            )
        )
        rows.append(
            _row(
                n_length,
                use_gmp,
                "add",
                add_samples,
                k_bits="",
                ciphertext_bytes=ct_bytes,
            )
        )
        for bits in K_BITS:
            scalar = (1 << bits) - 1
            mul_samples = _time_calls(lambda s=scalar: fresh * s)
            rows.append(
                _row(
                    n_length,
                    use_gmp,
                    "mul",
                    mul_samples,
                    k_bits=bits,
                    ciphertext_bytes=ct_bytes,
                )
            )
    if n_length == config.KEY_SIZE_DEV:
        util.HAVE_GMP = True
        notes = _check_values(public_key, private_key) + notes
    return rows, notes


def _write_note(df: pd.DataFrame, checks: list[str]) -> None:
    def med(n_length, operation, gmpy2, k_bits="") -> float:
        mask = (
            (df.n_length == n_length)
            & (df.operation == operation)
            & (df.gmpy2 == gmpy2)
        )
        if k_bits != "":
            mask &= df.k_bits == k_bits
        return float(df.loc[mask, "median_s"].iloc[0])

    lines = [
        "# B1 — operacje elementarne Paillier",
        "",
        "Mediana z "
        f"{N_REPEATS} powtórzeń (keygen: {N_KEYGEN}). "
        "`encrypt` używa domyślnej obfuskacji `r^n mod n²`. "
        "`add` i `mul` to surowe operacje homomorficzne (bez ponownej obfuskacji) — "
        "w `phe` właśnie obfuskacja, a nie `(1+n)^m`, dominuje koszt szyfrowania.",
        "",
        "## Poprawność (klucz 1024)",
        "",
        *checks,
        "",
        "## Mediana encrypt / decrypt / add (sekundy)",
        "",
        "| n | gmpy2 | keygen | encrypt | decrypt | add |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for n_length in KEY_SIZES:
        for gmpy2 in (True, False):
            lines.append(
                f"| {n_length} | {gmpy2} | {med(n_length, 'keygen', gmpy2):.6f} | "
                f"{med(n_length, 'encrypt', gmpy2):.6f} | "
                f"{med(n_length, 'decrypt', gmpy2):.6f} | "
                f"{med(n_length, 'add', gmpy2):.6f} |"
            )
    lines += [
        "",
        "## Mediana `enc * k` (sekundy)",
        "",
        "| n | gmpy2 | 8 bit | 16 bit | 24 bit | 32 bit |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for n_length in KEY_SIZES:
        for gmpy2 in (True, False):
            cells = " | ".join(
                f"{med(n_length, 'mul', gmpy2, bits):.6f}" for bits in K_BITS
            )
            lines.append(f"| {n_length} | {gmpy2} | {cells} |")

    enc_gmp = med(2048, "encrypt", True)
    enc_py = med(2048, "encrypt", False)
    lines += [
        "",
        f"Przy 2048 bitach gmpy2 przyspiesza `encrypt` około {enc_py / enc_gmp:.1f}×. "
        f"Mediana z gmpy2 ({enc_gmp:.4f} s) jest kosztem jednego szyfrogramu w E1.",
        "",
        f"Surowe liczby: `{CSV_PATH.name}`.",
        "",
    ]
    NOTE_PATH.write_text("\n".join(lines))


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    checks: list[str] = []
    try:
        for n_length in KEY_SIZES:
            print(f"benchmarking {n_length}-bit key...", flush=True)
            part, notes = _bench_key(n_length)
            rows.extend(part)
            checks.extend(notes)
    finally:
        util.HAVE_GMP = True

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    _write_note(df, checks)
    print(df.to_string(index=False))
    print(f"written: {CSV_PATH}")
    print(f"written: {NOTE_PATH}")


if __name__ == "__main__":
    main()
