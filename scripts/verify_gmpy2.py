#!/usr/bin/env python3
"""Verify phe uses gmpy2 (HAVE_GMP) and measure speedup on 2048-bit encrypt."""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from phe import paillier, util  # noqa: E402


def _bench_encrypt(
    public_key, n_ops: int = 20, use_gmp: bool = True
) -> float:
    util.HAVE_GMP = use_gmp
    start = time.perf_counter()
    for i in range(n_ops):
        public_key.encrypt(i % 1000)
    return time.perf_counter() - start


def main() -> None:
    try:
        import gmpy2  # noqa: F401
    except ImportError:
        print("gmpy2 is NOT installed — required for acceptable Paillier performance.")
        sys.exit(1)

    import gmpy2 as gmpy2_mod

    if not util.HAVE_GMP:
        print("phe.util.HAVE_GMP is False although gmpy2 is installed — check imports.")
        sys.exit(1)

    print("gmpy2:", gmpy2_mod.__version__)
    print("phe.util.HAVE_GMP:", util.HAVE_GMP)

    public_key, _ = paillier.generate_paillier_keypair(n_length=2048)
    n_ops = 20
    with_gmp = _bench_encrypt(public_key, n_ops=n_ops, use_gmp=True)
    without_gmp = _bench_encrypt(public_key, n_ops=n_ops, use_gmp=False)
    speedup = without_gmp / with_gmp if with_gmp > 0 else 0.0

    print(f"{n_ops} encrypts @ 2048-bit — with gmpy2:    {with_gmp:.3f}s")
    print(f"{n_ops} encrypts @ 2048-bit — simulated w/o:  {without_gmp:.3f}s")
    print(f"Speedup (simulated no gmpy2 / with gmpy2): {speedup:.2f}x")

    if speedup < 2.0:
        print("Warning: expected clear speedup at 2048-bit key size.")
        sys.exit(1)


if __name__ == "__main__":
    main()
