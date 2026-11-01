"""JSON messages between the client and the honest-but-curious server (B5).

The score request carries the public modulus and one ciphertext per feature.
The private key is not a field of any message.
"""

from __future__ import annotations

import base64
import binascii
import math

from ppspam import config, crypto

# 2: optional "encoding": "b64" on score requests (B6).
PROTOCOL_VERSION = 2

# "dec": decimal strings; "b64": fixed-width big-endian bytes of each ciphertext.
WIRE_ENCODINGS = ("dec", "b64")

# Decimal JSON of d ciphertexts, each < n². At d=2000 and a 2048-bit key that
# is about 2.5 MB; 8 MiB still covers a 3072-bit key at the same d.
MAX_SCORE_BODY_BYTES = 8 * 1024 * 1024

_PUBLIC_FIELDS = (
    "preprocess_version",
    "vectorizer",
    "terms",
    "idf",
    "selected",
    "d",
    "S_x",
    "S_w",
)


class ProtocolError(ValueError):
    """A score request that does not match the published protocol."""


class VersionMismatch(ProtocolError):
    """``model_version`` is not the one this server is scoring with."""


def estimate_score_json_bytes(d: int, n_length: int) -> int:
    """Upper bound on a decimal JSON score body (ciphertext < n²)."""
    digits = math.floor(2 * n_length * math.log10(2)) + 1
    return d * (digits + 3) + 64


def public_model_message(public: dict, model_version: str | None = None) -> dict:
    """Whitelist of what ``GET /public-model`` may return. Weights stay off it."""
    required = ("preprocess_version", "vectorizer", "terms", "idf", "d", "S_x")
    missing = [key for key in required if key not in public]
    if missing:
        raise ValueError("public model is missing: " + ", ".join(missing))
    message = {key: public[key] for key in _PUBLIC_FIELDS if key in public}
    message["d"] = int(message["d"])
    message["model_version"] = str(
        public.get("model_version", model_version or config.MODEL_VERSION)
    )
    message["protocol_version"] = PROTOCOL_VERSION
    crypto.assert_public_payload(message)
    return message


def score_request(
    n: str,
    ciphertexts: list[str] | list[int],
    model_version: str,
    encoding: str = "dec",
) -> dict:
    """``POST /score`` body. Every ciphertext is an integer plaintext (exponent 0).

    ``ciphertexts`` are ints or decimal strings; ``encoding`` picks the wire form.
    """
    if encoding not in WIRE_ENCODINGS:
        raise ValueError(f"encoding must be one of {WIRE_ENCODINGS}")
    width = ciphertext_width(int(n))
    body = {
        "n": str(n),
        "ciphertexts": [
            encode_ciphertext(int(ct), encoding, width) for ct in ciphertexts
        ],
        "model_version": str(model_version),
    }
    if encoding != "dec":
        body["encoding"] = encoding
    parse_score_request(body)
    return body


def parse_score_request(body: dict) -> tuple[str, list[int], str]:
    """Return ``(n, ciphertexts, model_version)`` or raise :class:`ProtocolError`.

    Ciphertexts come back as ints whatever the wire ``encoding`` was.
    """
    crypto.assert_public_payload(body)
    if set(body) - {"encoding"} != {"n", "ciphertexts", "model_version"}:
        raise ProtocolError(
            "score request must be {n, ciphertexts, model_version[, encoding]}"
        )
    encoding = body.get("encoding", "dec")
    if encoding not in WIRE_ENCODINGS:
        raise ProtocolError(f"encoding must be one of {WIRE_ENCODINGS}")
    n = body["n"]
    if not isinstance(n, str) or not n.isdecimal():
        raise ProtocolError("n must be a decimal string")
    ciphertexts = body["ciphertexts"]
    if not isinstance(ciphertexts, list) or not ciphertexts:
        raise ProtocolError("ciphertexts must be a non-empty list")
    width = ciphertext_width(int(n))
    parsed = [decode_ciphertext(ct, encoding, width) for ct in ciphertexts]
    version = body["model_version"]
    if not isinstance(version, str) or not version:
        raise ProtocolError("model_version must be a non-empty string")
    return n, parsed, version


def ciphertext_width(n: int) -> int:
    """Bytes of one ciphertext below n² (b64 form is always this wide)."""
    return max(1, ((n * n).bit_length() + 7) // 8)


def encode_ciphertext(ciphertext: int, encoding: str, width: int) -> str:
    if encoding == "dec":
        return str(ciphertext)
    return base64.b64encode(ciphertext.to_bytes(width, "big")).decode("ascii")


def decode_ciphertext(text, encoding: str, width: int) -> int:
    if isinstance(text, bool) or not isinstance(text, str):
        raise ProtocolError("each ciphertext must be a string")
    if encoding == "dec":
        if not text.isdecimal():
            raise ProtocolError("each ciphertext must be a decimal string")
        return int(text)
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError("each ciphertext must be base64") from exc
    if len(raw) != width:
        raise ProtocolError(f"each b64 ciphertext must be {width} bytes")
    return int.from_bytes(raw, "big")


def score_response(result: str, exponent: int) -> dict:
    """``{"result": "<ciphertext>", "exponent": 0}``."""
    if int(exponent) != 0:
        raise ValueError("integer score ciphertext must have exponent 0 (D3)")
    if not str(result).isdecimal():
        raise ValueError("score ciphertext must be a decimal string")
    return {"result": str(result), "exponent": 0}
