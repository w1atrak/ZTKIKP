"""What crosses the client–server boundary (K5, W3).

The server may see the public modulus and exactly ``d`` ciphertexts.
It must not see the SMS, ``p``, ``q``, or the private key.
"""

import inspect
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from phe.paillier import PaillierPublicKey

from ppspam import crypto, features
from ppspam.client import Client
from ppspam.server import create_app

_SMS = "rendezvous-at-platform-nine tonight"
_TEXTS = ["free prize call now", "hello there", "", "x" * 400, _SMS]


def _public_model() -> dict:
    vectorizer = features.make_vectorizer(min_df=1).fit(
        ["free prize", "hello there", "call now", ""]
    )
    return features.export_public_model(
        vectorizer, path=None, extra={"S_x": 1000.0, "S_w": 100.0, "model_version": "1"}
    )


@pytest.fixture(scope="module")
def public_model() -> dict:
    return _public_model()


@pytest.fixture(scope="module")
def client(public_model: dict) -> Client:
    return Client(public_model=public_model, n_length=1024, scale_x=1000)


def test_prepare_payload_has_no_plaintext_or_secret(client: Client):
    payload = client.prepare(_SMS)
    crypto.assert_public_payload(payload)
    assert set(payload) == {"public_key", "ciphertexts", "d"}
    assert set(payload["public_key"]) == {"n"}
    encoded = json.dumps(payload)
    assert _SMS not in encoded
    assert "rendezvous" not in encoded
    assert "platform" not in encoded
    wire = client.score_body(_SMS)
    crypto.assert_public_payload(wire)
    assert set(wire) <= {"n", "ciphertexts", "model_version", "encoding"}
    wire_text = json.dumps(wire)
    assert "rendezvous" not in wire_text
    assert "platform" not in wire_text


def test_ciphertext_count_equals_d_for_every_message(client: Client):
    d = client.featurizer.d
    assert d > 1
    for text in _TEXTS:
        payload = client.prepare(text)
        assert len(payload["ciphertexts"]) == d
        assert payload["d"] == d


def test_same_message_encrypts_twice_to_different_ciphertexts(client: Client):
    first = client.prepare(_SMS)["ciphertexts"]
    second = client.prepare(_SMS)["ciphertexts"]
    assert first != second
    assert {pair[1] for pair in first} == {0}


def test_obfuscator_pool_refuses_to_reuse_r():
    public_key, _private_key = crypto.generate_keypair(256)
    pool = crypto.ObfuscatorPool(public_key, size=2)
    first, second = pool.take(), pool.take()
    assert first != second
    assert len(pool) == 0
    with pytest.raises(RuntimeError, match="empty"):
        pool.take()


def test_phe_samples_r_with_system_random():
    source = inspect.getsource(PaillierPublicKey.get_random_lt_n)
    assert "SystemRandom" in source


def test_score_log_and_public_model_omit_message_and_weights(
    client: Client, public_model: dict, tmp_path: Path, caplog
):
    public = public_model
    d = client.featurizer.d
    weights = {"w_q": [1] * d, "b_q_prime": 1, "d": d, "model_version": "1"}
    weights_path = tmp_path / "w.json"
    public_path = tmp_path / "p.json"
    weights_path.write_text(json.dumps(weights))
    public_path.write_text(json.dumps(public))
    http = TestClient(
        create_app(weights_path=weights_path, public_model_path=public_path)
    )

    published = http.get("/public-model").json()
    crypto.assert_public_payload(published)
    assert "w_q" not in published and "b_q_prime" not in published

    body = client.score_body(_SMS)
    assert len(body["ciphertexts"]) == d
    caplog.set_level("INFO", logger="ppspam.server")
    response = http.post("/score", json=body)
    assert response.status_code == 200
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "rendezvous" not in logged
    assert "platform" not in logged
    # A decimal ciphertext is hundreds of digits; the log must not copy it.
    assert body["ciphertexts"][0][:32] not in logged
    assert "score ok" in logged
    crypto.assert_public_payload(response.json())
