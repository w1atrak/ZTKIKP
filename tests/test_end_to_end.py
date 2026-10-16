"""Walking skeleton: text → client → server → label, in one process (W1)."""

import numpy as np
import pytest

from ppspam import crypto, data, features, model, quantize
from ppspam.client import Client
from ppspam.server import Server

D = 64
N_SMS = 30


@pytest.mark.skipif(not data.TEST_CSV.exists(), reason="run scripts/download_data.py")
def test_encrypted_score_matches_plain_quantized_score():
    train, test = data.load_split()
    clf = model.train(
        train["text"], model.labels_to_y(train["label"]), k=D, method="max_features"
    )
    public = features.export_public_model(
        clf.named_steps["tfidf"],
        selected=model.selected_columns(clf),
        path=None,
    )
    assert public["d"] == D

    w, b = model.linear_params(clf)
    texts = test["text"].iloc[:N_SMS]
    X_train = model.featurize(clf, train["text"])
    X_test = model.featurize(clf, texts)
    S_x, S_w, _rows = quantize.search_scales(X_train, X_test, w, b)
    weights = quantize.server_weights_q(w, b, S_x, S_w)

    X_q = quantize.quantize_features(X_test, S_x)
    s_q = quantize.quantized_score(X_q, weights["w_q"], weights["b_q_prime"])
    plain = clf.predict(texts)

    client = Client(public_model=public, n_length=1024, scale_x=S_x)
    server = Server(weights=weights)

    for i, text in enumerate(texts):
        payload = client.prepare(text)
        assert len(payload["ciphertexts"]) == D
        encrypted_score = server.score_request(payload)
        label, score = client.decide(crypto.ciphertext_to_pair(encrypted_score))
        assert score == s_q[i]
        expected = int(model.predict_linear(np.asarray([s_q[i]], dtype=np.float64))[0])
        assert label == expected == int(plain[i])
