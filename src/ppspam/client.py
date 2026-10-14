"""Client pipeline: SMS → dense TF-IDF → encrypt → decrypt the score (B3)."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
from phe.paillier import EncryptedNumber

from ppspam import config, crypto, features, model, quantize


class Client:
    """Holds the Paillier secret and the public featurizer. The secret never leaves."""

    def __init__(
        self,
        public_model: dict | None = None,
        public_model_path=None,
        n_length: int = config.KEY_SIZE_DEV,
        keypair=None,
        scale_x: float = config.SCALE_X,
    ):
        if keypair is None:
            self.public_key, self.private_key = crypto.generate_keypair(n_length)
        else:
            self.public_key, self.private_key = keypair
        if public_model is None:
            path = (
                features.PUBLIC_MODEL_JSON
                if public_model_path is None
                else public_model_path
            )
            self.featurizer = features.PublicFeaturizer.load(path)
        else:
            self.featurizer = features.PublicFeaturizer(public_model)
        self.scale_x = scale_x
        self.timings: dict[str, float] = {}
        self.timing_log: list[dict[str, float]] = []

    def prepare(self, text: str) -> dict:
        """Dense encrypted vector of length ``d``, plus the public key."""
        started = time.perf_counter()
        features_row = self.featurizer.transform([text])
        featurized = time.perf_counter()
        x_q = quantize.quantize_features(features_row, self.scale_x)[0]
        quantized = time.perf_counter()
        if len(x_q) != self.featurizer.d:
            raise ValueError(
                f"quantized vector has length {len(x_q)}, expected {self.featurizer.d}"
            )
        encrypted = crypto.encrypt_vector(self.public_key, x_q)
        encrypted_at = time.perf_counter()
        payload = {
            "public_key": crypto.dump_public_key(self.public_key),
            "ciphertexts": [list(crypto.ciphertext_to_pair(c)) for c in encrypted],
            "d": self.featurizer.d,
        }
        crypto.assert_public_payload(payload)
        serialized = time.perf_counter()
        timings = {
            "featurize_s": featurized - started,
            "quantize_s": quantized - featurized,
            "encrypt_s": encrypted_at - quantized,
            "serialize_s": serialized - encrypted_at,
        }
        self.timings = timings
        self.timing_log.append(timings)
        return payload

    def decide(self, encrypted_result) -> tuple[int, int]:
        """``(label, s)``; spam iff ``s > 0``, same tie-break as the linear form."""
        started = time.perf_counter()
        encrypted = _as_encrypted(self.public_key, encrypted_result)
        score = crypto.decrypt_scalar(self.private_key, encrypted)
        self.timings = {**self.timings, "decrypt_s": time.perf_counter() - started}
        label = int(model.predict_linear(np.asarray([score], dtype=np.float64))[0])
        return label, score


def _as_encrypted(public_key, encrypted_result):
    if isinstance(encrypted_result, EncryptedNumber):
        return encrypted_result
    if isinstance(encrypted_result, Sequence) and not isinstance(encrypted_result, str):
        return crypto.ciphertext_from_pair(public_key, encrypted_result)
    raise TypeError("encrypted result must be an EncryptedNumber or a (ct, exp) pair")
