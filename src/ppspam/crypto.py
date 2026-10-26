"""Paillier vectors and ciphertext serialization (phase B2).

Plaintexts are integers (D3), so every ciphertext we send has ``exponent == 0``.
The private key is never serialized: payloads are checked for ``p`` / ``q``.
"""

from __future__ import annotations

import base64
import json
import random
from collections.abc import Sequence
from concurrent.futures import Executor

from phe import paillier
from phe.util import mulmod, powmod

from ppspam import config

_FORBIDDEN_KEYS = frozenset({"p", "q", "private_key", "secret_key"})


def generate_keypair(n_length: int = config.KEY_SIZE_DEV):
    """``(public_key, private_key)``; the secret stays on the client."""
    return paillier.generate_paillier_keypair(n_length=n_length)


class ObfuscatorPool:
    """Offline ``r^n mod n²`` values for one public key (B6).

    ``r`` does not depend on the message, so the expensive power can be done
    before the SMS exists. :meth:`take` removes the value: no ``r`` is used twice.
    """

    def __init__(self, public_key, size: int = 0):
        self.public_key = public_key
        self._values: list[int] = []
        if size:
            self.fill(size)

    def __len__(self) -> int:
        return len(self._values)

    def fill(
        self, count: int, executor: Executor | None = None, chunks: int = 1
    ) -> None:
        if executor is None:
            self._values.extend(make_obfuscators(self.public_key.n, count))
            return
        sizes = [len(part) for part in split_chunks([None] * count, chunks)]
        futures = [
            executor.submit(make_obfuscators, self.public_key.n, size) for size in sizes
        ]
        for future in futures:
            self._values.extend(future.result())

    def take(self) -> int:
        if not self._values:
            raise RuntimeError("obfuscator pool is empty; refill it before encrypting")
        return self._values.pop()


def make_obfuscators(n: int, count: int) -> list[int]:
    """``count`` fresh ``r^n mod n²`` with ``r`` from ``SystemRandom``."""
    rng = random.SystemRandom()
    nsquare = n * n
    return [powmod(rng.randrange(1, n), n, nsquare) for _ in range(count)]


def encrypt_vector(
    public_key, x_q: Sequence[int], pool: ObfuscatorPool | None = None
) -> list:
    """Encrypt each integer coordinate. Zeros are encrypted too (D4).

    With ``pool`` the online step is ``(1 + n·m) · r^n mod n²`` with a
    precomputed ``r^n``; otherwise ``phe`` draws and raises ``r`` per value.
    """
    _check_ints(x_q)
    if pool is not None:
        if pool.public_key.n != public_key.n:
            raise ValueError("obfuscator pool belongs to a different public key")
        if len(pool) < len(x_q):
            raise RuntimeError(
                f"obfuscator pool has {len(pool)} values, need {len(x_q)}"
            )
    encrypted = []
    for value in x_q:
        if pool is None:
            enc = public_key.encrypt(int(value))
        else:
            nude = public_key.encrypt(int(value), r_value=1)
            nude_ct = nude.ciphertext(be_secure=False)
            ct = mulmod(nude_ct, pool.take(), public_key.nsquare)
            enc = _obfuscated(public_key, ct, nude.exponent)
        if enc.exponent != 0:
            raise ValueError("integer Paillier ciphertext must have exponent 0 (D3)")
        encrypted.append(enc)
    return encrypted


def encrypt_vector_parallel(
    public_key, x_q: Sequence[int], executor: Executor, chunks: int
) -> list:
    """:func:`encrypt_vector` split into ``chunks`` contiguous parts on ``executor``.

    Workers only get ``n`` and plaintext integers; they return raw ciphertexts.
    """
    _check_ints(x_q)
    futures = [
        executor.submit(_encrypt_chunk, public_key.n, part)
        for part in split_chunks(list(x_q), chunks)
    ]
    encrypted = []
    for future in futures:
        encrypted.extend(_obfuscated(public_key, ct, 0) for ct in future.result())
    return encrypted


def split_chunks(values: list, chunks: int) -> list[list]:
    """Contiguous, near-equal, non-empty parts (order preserved)."""
    if chunks < 1:
        raise ValueError("chunks must be >= 1")
    size, extra = divmod(len(values), chunks)
    parts, start = [], 0
    for i in range(chunks):
        stop = start + size + (1 if i < extra else 0)
        if stop > start:
            parts.append(values[start:stop])
        start = stop
    return parts


def _encrypt_chunk(n: int, values: list[int]) -> list[int]:
    public_key = paillier.PaillierPublicKey(n=n)
    return [
        enc.ciphertext(be_secure=False) for enc in encrypt_vector(public_key, values)
    ]


def _check_ints(values: Sequence[int]) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"expected Python int, got {type(value).__name__}")


def _obfuscated(public_key, ciphertext: int, exponent: int) -> paillier.EncryptedNumber:
    enc = paillier.EncryptedNumber(public_key, ciphertext, exponent)
    # phe has no public setter; without it ciphertext() would obfuscate again.
    enc._EncryptedNumber__is_obfuscated = True
    return enc


def decrypt_scalar(private_key, encrypted) -> int:
    """Decrypt one integer ciphertext (the homomorphic score)."""
    value = private_key.decrypt(encrypted)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected integer plaintext, got {type(value).__name__}")
    return int(value)


def dump_public_key(public_key) -> dict:
    """Public key as JSON-ready ``{"n": ...}``. ``p`` and ``q`` are not here."""
    return {"n": str(public_key.n)}


def load_public_key(payload: dict):
    if set(payload) != {"n"}:
        raise ValueError("public key payload must be exactly {n}")
    return paillier.PaillierPublicKey(n=int(payload["n"]))


def ciphertext_to_pair(encrypted) -> tuple[str, int]:
    """``(decimal ciphertext, exponent)``. ``ciphertext()`` obfuscates if needed."""
    if encrypted.exponent != 0:
        raise ValueError("refusing to send a ciphertext with exponent != 0 (D3)")
    return (str(encrypted.ciphertext()), encrypted.exponent)


def ciphertext_from_pair(public_key, pair: Sequence) -> paillier.EncryptedNumber:
    if len(pair) != 2:
        raise ValueError("ciphertext pair must be (ct, exponent)")
    ct, exp = int(pair[0]), int(pair[1])
    _check_ciphertext_range(public_key, ct)
    return paillier.EncryptedNumber(public_key, ct, exp)


def ciphertext_to_compact(encrypted) -> dict:
    """Shorter form: ``int.to_bytes`` + base64, for the E5 size comparison."""
    ct_text, exp = ciphertext_to_pair(encrypted)
    ct = int(ct_text)
    width = max(1, (ct.bit_length() + 7) // 8)
    raw = ct.to_bytes(width, "big")
    return {
        "ct": base64.b64encode(raw).decode("ascii"),
        "exp": exp,
        "encoding": "b64",
    }


def ciphertext_from_compact(public_key, payload: dict) -> paillier.EncryptedNumber:
    if payload.get("encoding") != "b64":
        raise ValueError("compact ciphertext must declare encoding=b64")
    ct = int.from_bytes(base64.b64decode(payload["ct"]), "big")
    exp = int(payload["exp"])
    _check_ciphertext_range(public_key, ct)
    return paillier.EncryptedNumber(public_key, ct, exp)


def encoding_sizes(encrypted) -> dict:
    """Byte lengths of the decimal pair vs the compact form (one obfuscation)."""
    pair = ciphertext_to_pair(encrypted)
    ct = int(pair[0])
    width = max(1, (ct.bit_length() + 7) // 8)
    compact = {
        "ct": base64.b64encode(ct.to_bytes(width, "big")).decode("ascii"),
        "exp": pair[1],
        "encoding": "b64",
    }
    return {
        "raw_bytes": width,
        "decimal_json": len(json.dumps(list(pair), separators=(",", ":")).encode()),
        "b64_json": len(json.dumps(compact, separators=(",", ":")).encode()),
    }


def assert_public_payload(payload) -> None:
    """Reject anything that could carry a Paillier private key."""

    def walk(obj) -> None:
        if isinstance(obj, dict):
            leaked = _FORBIDDEN_KEYS & obj.keys()
            if leaked:
                raise ValueError(
                    "payload must not contain private-key fields: "
                    + ", ".join(sorted(leaked))
                )
            for value in obj.values():
                walk(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                walk(value)

    walk(payload)


def _check_ciphertext_range(public_key, ciphertext: int) -> None:
    if not 0 <= ciphertext < public_key.nsquare:
        raise ValueError("ciphertext is outside 0 .. n²-1")
