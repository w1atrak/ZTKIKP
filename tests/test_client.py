"""Client prepare/decide against a plaintext stub server (phase B3)."""

import numpy as np

from ppspam import crypto, features, model, quantize
from ppspam.client import Client

_TEXTS = [
    "WINNER!! Claim your prize now",
    "Free entry txt WIN",
    "Hey, are we still meeting tonight?",
    "",
]


def _public_model() -> dict:
    vectorizer = features.make_vectorizer(min_df=1).fit(_TEXTS)
    return features.export_public_model(vectorizer, path=None)


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
