"""Homomorphic linear score. Plaintext weights stay here; the secret key does not."""

from __future__ import annotations

import json
from pathlib import Path

from phe.paillier import EncryptedNumber

from ppspam import config, crypto, quantize


class Server:
    """Computes ``Enc(b_q' + x_q · w_q)`` from a dense encrypted vector."""

    def __init__(self, weights: dict | None = None, weights_path: Path | None = None):
        if weights is None:
            path = (
                quantize.SERVER_WEIGHTS_Q_JSON if weights_path is None else weights_path
            )
            weights = json.loads(Path(path).read_text())
        self.w_q = [int(value) for value in weights["w_q"]]
        self.b_q_prime = int(weights["b_q_prime"])
        self.d = int(weights.get("d", len(self.w_q)))
        if self.d != len(self.w_q):
            raise ValueError(f"d={self.d} does not match {len(self.w_q)} weights")

    def score(self, public_key, enc_vector) -> EncryptedNumber:
        """``Enc(b_q') + Σ enc_i * w_q[i]``. A zero weight skips the multiplication."""
        self._validate(public_key, enc_vector)
        acc = public_key.encrypt(self.b_q_prime)
        for cipher, weight in zip(enc_vector, self.w_q, strict=True):
            if weight == 0:
                continue
            acc = acc + cipher * int(weight)
        return acc

    def score_request(self, payload: dict) -> EncryptedNumber:
        """Deserialize a client payload and return the encrypted score."""
        crypto.assert_public_payload(payload)
        if "public_key" not in payload or "ciphertexts" not in payload:
            raise ValueError("payload needs public_key and ciphertexts")
        public_key = crypto.load_public_key(payload["public_key"])
        enc_vector = [
            crypto.ciphertext_from_pair(public_key, pair)
            for pair in payload["ciphertexts"]
        ]
        return self.score(public_key, enc_vector)

    def _validate(self, public_key, enc_vector) -> None:
        # Two n_length/2 primes can multiply to n_length-1 bits.
        if public_key.n.bit_length() < config.KEY_SIZE_DEV - 1:
            raise ValueError(
                f"modulus n has {public_key.n.bit_length()} bits; "
                f"need at least {config.KEY_SIZE_DEV - 1}"
            )
        if len(enc_vector) != self.d:
            raise ValueError(
                f"encrypted vector has length {len(enc_vector)}, expected {self.d}"
            )
        for item in enc_vector:
            if not isinstance(item, EncryptedNumber):
                raise TypeError("encrypted vector entries must be EncryptedNumber")
            if item.public_key != public_key:
                raise ValueError("ciphertext was encrypted under a different key")
            raw = item.ciphertext(be_secure=False)
            if not 0 <= raw < public_key.nsquare:
                raise ValueError("ciphertext is outside 0 .. n²-1")
