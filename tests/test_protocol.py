"""Wire format for the public model and the encrypted score (B5)."""

import pytest

from ppspam import crypto, protocol


def test_public_model_message_drops_weights():
    public = {
        "preprocess_version": 1,
        "vectorizer": {"ngram_range": [1, 1]},
        "terms": ["free"],
        "idf": [1.0],
        "selected": None,
        "d": 1,
        "S_x": 1000.0,
        "S_w": 100.0,
        "w_q": [5],
        "w": [0.1],
        "p": 3,
        "q": 5,
    }
    message = protocol.public_model_message(public)
    crypto.assert_public_payload(message)
    assert message["d"] == 1
    assert message["S_x"] == 1000.0
    assert message["model_version"] == "1"
    assert message["protocol_version"] == protocol.PROTOCOL_VERSION
    assert "w_q" not in message and "w" not in message


def test_score_request_round_trip_and_rejects_secrets():
    body = protocol.score_request("17", ["4", "9"], "1")
    assert protocol.parse_score_request(body) == ("17", [4, 9], "1")
    leaked = {**body, "p": "11"}
    with pytest.raises(ValueError, match="private-key"):
        protocol.parse_score_request(leaked)
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_score_request({**body, "ciphertexts": ["-1"]})
    with pytest.raises(protocol.ProtocolError):
        protocol.parse_score_request({**body, "ciphertexts": [4]})


def test_b64_request_round_trip_and_fixed_width():
    n = 2**61 - 1
    cts = [0, 5, n * n - 1]
    body = protocol.score_request(str(n), cts, "1", encoding="b64")
    assert body["encoding"] == "b64"
    assert len({len(ct) for ct in body["ciphertexts"]}) == 1
    assert protocol.parse_score_request(body) == (str(n), cts, "1")
    assert "encoding" not in protocol.score_request(str(n), cts, "1")
    with pytest.raises(protocol.ProtocolError, match="base64"):
        protocol.parse_score_request({**body, "ciphertexts": ["not base64!"]})
    with pytest.raises(protocol.ProtocolError, match="bytes"):
        protocol.parse_score_request({**body, "ciphertexts": ["AAAA"]})
    with pytest.raises(protocol.ProtocolError, match="encoding"):
        protocol.parse_score_request({**body, "encoding": "hex"})


def test_score_response_keeps_exponent_zero():
    assert protocol.score_response("99", 0) == {"result": "99", "exponent": 0}
    with pytest.raises(ValueError, match="exponent"):
        protocol.score_response("99", 1)


def test_body_estimate_fits_the_documented_limit():
    estimated = protocol.estimate_score_json_bytes(2000, 2048)
    assert 1_000_000 <= estimated <= 2_600_000
    assert estimated < protocol.MAX_SCORE_BODY_BYTES
    wider = protocol.estimate_score_json_bytes(2000, 3072)
    assert wider < protocol.MAX_SCORE_BODY_BYTES
