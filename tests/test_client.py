"""Client prepare/decide against a plaintext stub server (phase B3/B5)."""

import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from ppspam import crypto, features, model, quantize
from ppspam.client import Client
from ppspam.server import create_app

_TEXTS = [
    "WINNER!! Claim your prize now",
    "Free entry txt WIN",
    "Hey, are we still meeting tonight?",
    "",
]


def _public_model() -> dict:
    vectorizer = features.make_vectorizer(min_df=1).fit(_TEXTS)
    return features.export_public_model(
        vectorizer, path=None, extra={"S_x": 1000.0, "S_w": 100.0, "model_version": "1"}
    )


def test_prepare_is_dense_and_stub_score_round_trips():
    public = _public_model()
    client = Client(public_model=public, n_length=1024, scale_x=1000)
    d = public["d"]
    w_q = [((i * 3) % 7) - 3 for i in range(d)]
    b_q = -2

    for text in ("free prize", "hello there", ""):
        payload = client.prepare(text)
        crypto.assert_public_payload(payload)
        assert payload["d"] == d
        assert len(payload["ciphertexts"]) == d
        assert set(client.timings) >= {
            "featurize_s",
            "quantize_s",
            "encrypt_s",
            "serialize_s",
        }

        x_q = quantize.quantize_features(
            client.featurizer.transform([text]), client.scale_x
        )[0]
        assert len(x_q) == d
        s_q = quantize.quantized_score([x_q], w_q, b_q)[0]
        encrypted_score = client.public_key.encrypt(int(s_q))
        label, score = client.decide(crypto.ciphertext_to_pair(encrypted_score))
        assert score == s_q
        assert label == int(
            model.predict_linear(np.asarray([s_q], dtype=np.float64))[0]
        )
        assert "decrypt_s" in client.timings

    assert len(client.timing_log) == 3


def test_score_body_and_http_round_trip(tmp_path: Path, monkeypatch):
    public = _public_model()
    d = public["d"]
    w_q = [((i * 3) % 7) - 3 for i in range(d)]
    weights = {"w_q": w_q, "b_q_prime": -2, "d": d, "model_version": "1"}
    weights_path = tmp_path / "weights.json"
    public_path = tmp_path / "public.json"
    weights_path.write_text(json.dumps(weights))
    public_path.write_text(json.dumps(public))

    app = create_app(weights_path=weights_path, public_model_path=public_path)
    api = TestClient(app)
    public_msg = api.get("/public-model").json()
    crypto.assert_public_payload(public_msg)

    client = Client(public_model=public_msg, n_length=1024, base_url="http://test")
    body = client.score_body("free prize")
    assert set(body) == {"n", "ciphertexts", "model_version"}
    assert len(body["ciphertexts"]) == d

    class _FakeResponse:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload
            self.text = json.dumps(payload)
            self.request = None

        def json(self):
            return self._payload

    class _FakeHttp:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json):  # noqa: A002
            assert url.endswith("/score")
            return _FakeResponse(api.post("/score", json=json).json())

    monkeypatch.setattr("ppspam.client.httpx.Client", _FakeHttp)
    label, score = client.score_remote("hello there")
    x_q = quantize.quantize_features(
        client.featurizer.transform(["hello there"]), client.scale_x
    )[0]
    expected = quantize.quantized_score([x_q], w_q, -2)[0]
    assert score == expected
    assert label == int(model.predict_linear(np.asarray([expected], dtype=np.float64))[0])
    assert "transfer_s" in client.timings
