"""Client pipeline: SMS → dense TF-IDF → encrypt → decrypt the score (B3/B5).

In-process scoring uses :meth:`prepare` + a local ``Server``. Network scoring
uses :meth:`score_remote` against ``POST /score`` (httpx). The secret key never
leaves this process.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import httpx
import numpy as np
from phe.paillier import EncryptedNumber

from ppspam import config, crypto, features, model, protocol, quantize


class Client:
    """Holds the Paillier secret and the public featurizer. The secret never leaves."""

    def __init__(
        self,
        public_model: dict | None = None,
        public_model_path=None,
        n_length: int = config.KEY_SIZE_DEV,
        keypair=None,
        scale_x: float | None = None,
        model_version: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
    ):
        if keypair is None:
            self.public_key, self.private_key = crypto.generate_keypair(n_length)
        else:
            self.public_key, self.private_key = keypair

        if public_model is None:
            path = Path(
                features.PUBLIC_MODEL_JSON
                if public_model_path is None
                else public_model_path
            )
            public_model = json.loads(path.read_text())

        self.featurizer = features.PublicFeaturizer(public_model)
        # Prefer an explicit scale_x; otherwise take S_x from the public model.
        if scale_x is not None:
            self.scale_x = float(scale_x)
        elif "S_x" in public_model:
            self.scale_x = float(public_model["S_x"])
        else:
            self.scale_x = float(config.SCALE_X)

        self.model_version = str(
            model_version
            if model_version is not None
            else public_model.get("model_version", config.MODEL_VERSION)
        )
        self.base_url = None if base_url is None else base_url.rstrip("/")
        self.timeout = timeout
        self.timings: dict[str, float] = {}
        self.timing_log: list[dict[str, float]] = []

    @classmethod
    def from_server(
        cls,
        base_url: str,
        n_length: int = config.KEY_SIZE_DEV,
        keypair=None,
        timeout: float = 300.0,
    ) -> Client:
        """Fetch ``GET /public-model`` and build a networked client."""
        url = base_url.rstrip("/")
        with httpx.Client(timeout=timeout) as http:
            response = http.get(f"{url}/public-model")
            response.raise_for_status()
            public = response.json()
        crypto.assert_public_payload(public)
        return cls(
            public_model=public,
            n_length=n_length,
            keypair=keypair,
            base_url=url,
            timeout=timeout,
        )

    def prepare(self, text: str) -> dict:
        """Dense encrypted vector of length ``d``, plus the public key (in-process)."""
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

    def score_body(self, text: str) -> dict:
        """Wire-format ``POST /score`` body for ``text``."""
        payload = self.prepare(text)
        return protocol.score_request(
            payload["public_key"]["n"],
            [pair[0] for pair in payload["ciphertexts"]],
            self.model_version,
        )

    def decide(self, encrypted_result) -> tuple[int, int]:
        """``(label, s)``; spam iff ``s > 0``, same tie-break as the linear form."""
        started = time.perf_counter()
        encrypted = _as_encrypted(self.public_key, encrypted_result)
        score = crypto.decrypt_scalar(self.private_key, encrypted)
        self.timings = {**self.timings, "decrypt_s": time.perf_counter() - started}
        label = int(model.predict_linear(np.asarray([score], dtype=np.float64))[0])
        return label, score

    def decide_response(self, response: dict) -> tuple[int, int]:
        """Decrypt a ``{"result", "exponent"}`` score response."""
        if set(response) != {"result", "exponent"}:
            raise ValueError("score response must be {result, exponent}")
        return self.decide((response["result"], response["exponent"]))

    def score_remote(self, text: str, base_url: str | None = None) -> tuple[int, int]:
        """Encrypt ``text``, ``POST /score``, decrypt. Returns ``(label, s)``."""
        url = (base_url or self.base_url or "").rstrip("/")
        if not url:
            raise ValueError("base_url is required for remote scoring")
        body = self.score_body(text)
        started = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as http:
            response = http.post(f"{url}/score", json=body)
        self.timings = {**self.timings, "transfer_s": time.perf_counter() - started}
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"score failed: {response.status_code} {response.text}",
                request=response.request,
                response=response,
            )
        label, score = self.decide_response(response.json())
        self.timing_log[-1] = {**self.timing_log[-1], **self.timings}
        return label, score


def _as_encrypted(public_key, encrypted_result):
    if isinstance(encrypted_result, EncryptedNumber):
        return encrypted_result
    if isinstance(encrypted_result, Sequence) and not isinstance(encrypted_result, str):
        return crypto.ciphertext_from_pair(public_key, encrypted_result)
    raise TypeError("encrypted result must be an EncryptedNumber or a (ct, exp) pair")
