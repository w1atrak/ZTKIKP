"""HTTP API: public model + encrypted score (phase B5)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ppspam import crypto, protocol, quantize
from ppspam.server import Server, create_app


@pytest.fixture()
def tiny_artifacts(tmp_path: Path):
    weights = {"w_q": [2, -3, 0], "b_q_prime": 4, "d": 3, "model_version": "t"}
    public = {
        "preprocess_version": 1,
        "vectorizer": {
            "ngram_range": [1, 1],
            "min_df": 1,
            "max_df": 1.0,
            "max_features": None,
            "sublinear_tf": False,
        },
        "terms": ["a", "b", "c"],
        "idf": [1.0, 1.0, 1.0],
        "selected": None,
        "d": 3,
        "S_x": 1000.0,
        "S_w": 100.0,
        "model_version": "t",
    }
    weights_path = tmp_path / "weights.json"
    public_path = tmp_path / "public.json"
    weights_path.write_text(json.dumps(weights))
    public_path.write_text(json.dumps(public))
    return weights_path, public_path, weights


@pytest.fixture()
def client(tiny_artifacts):
    weights_path, public_path, _ = tiny_artifacts
    app = create_app(weights_path=weights_path, public_model_path=public_path)
    return TestClient(app)


def test_public_model_has_no_weights(client):
    response = client.get("/public-model")
    assert response.status_code == 200
    body = response.json()
    crypto.assert_public_payload(body)
    assert body["d"] == 3
    assert body["model_version"] == "t"
    assert "w_q" not in body and "w" not in body


def test_score_round_trip(client, tiny_artifacts):
    _, _, weights = tiny_artifacts
    public_key, private_key = crypto.generate_keypair(1024)
    x_q = [5, 1, 9]
    encrypted = crypto.encrypt_vector(public_key, x_q)
    body = protocol.score_request(
        str(public_key.n),
        [crypto.ciphertext_to_pair(c)[0] for c in encrypted],
        "t",
    )
    response = client.post("/score", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["exponent"] == 0
    got = crypto.decrypt_scalar(
        private_key,
        crypto.ciphertext_from_pair(
            public_key, (payload["result"], payload["exponent"])
        ),
    )
    assert (
        got == quantize.quantized_score([x_q], weights["w_q"], weights["b_q_prime"])[0]
    )


def test_score_response_is_freshly_obfuscated(tiny_artifacts):
    weights_path, _, _ = tiny_artifacts
    server = Server(weights_path=weights_path)
    public_key, private_key = crypto.generate_keypair(1024)
    encrypted = crypto.encrypt_vector(public_key, [1, 2, 3])
    body = protocol.score_request(
        str(public_key.n),
        [crypto.ciphertext_to_pair(c)[0] for c in encrypted],
        server.model_version,
    )
    first = server.score_protocol(body)["result"]
    second = server.score_protocol(body)["result"]
    assert first != second
    scores = {
        crypto.decrypt_scalar(
            private_key, crypto.ciphertext_from_pair(public_key, (ct, 0))
        )
        for ct in (first, second)
    }
    assert scores == {4 + 2 - 6}


def test_score_rejects_bad_version_and_length(client):
    public_key, _ = crypto.generate_keypair(1024)
    encrypted = crypto.encrypt_vector(public_key, [1, 1, 1])
    cts = [crypto.ciphertext_to_pair(c)[0] for c in encrypted]
    bad_version = protocol.score_request(str(public_key.n), cts, "other")
    assert client.post("/score", json=bad_version).status_code == 409
    short = protocol.score_request(str(public_key.n), cts[:2], "t")
    assert client.post("/score", json=short).status_code == 400


def test_score_protocol_matches_in_process(tiny_artifacts):
    weights_path, _, weights = tiny_artifacts
    server = Server(weights_path=weights_path)
    public_key, private_key = crypto.generate_keypair(1024)
    x_q = [2, 0, -1]
    encrypted = crypto.encrypt_vector(public_key, x_q)
    body = protocol.score_request(
        str(public_key.n),
        [crypto.ciphertext_to_pair(c)[0] for c in encrypted],
        server.model_version,
    )
    response = server.score_protocol(body)
    got = crypto.decrypt_scalar(
        private_key,
        crypto.ciphertext_from_pair(public_key, (response["result"], response["exponent"])),
    )
    assert got == 4 + 2 * 2 + 0 * (-3) + (-1) * 0
