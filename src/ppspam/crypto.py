"""Paillier vectors and ciphertext serialization (phase B2).

Plaintexts are integers (D3), so every ciphertext we send has ``exponent == 0``.
The private key is never serialized: payloads are checked for ``p`` / ``q``.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence

from phe import paillier

from ppspam import config

_FORBIDDEN_KEYS = frozenset({"p", "q", "private_key", "secret_key"})


def generate_keypair(n_length: int = config.KEY_SIZE_DEV):
    """``(public_key, private_key)``; the secret stays on the client."""
    return paillier.generate_paillier_keypair(n_length=n_length)


def encrypt_vector(public_key, x_q: Sequence[int]) -> list:
    """Encrypt each integer coordinate. Zeros are encrypted too (D4)."""
    encrypted = []
    for value in x_q:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"expected Python int, got {type(value).__name__}")
        enc = public_key.encrypt(int(value))
        if enc.exponent != 0:
            raise ValueError("integer Paillier ciphertext must have exponent 0 (D3)")
        encrypted.append(enc)
    return encrypted


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
