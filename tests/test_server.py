"""Homomorphic dot product on the server (phase B4)."""

import ast
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from phe.paillier import EncryptedNumber, PaillierPublicKey

from ppspam import crypto, quantize
from ppspam.server import Server


def test_server_source_does_not_import_private_key():
    source = Path(quantize.__file__).with_name("server.py").read_text()
    tree = ast.parse(source)
    banned = {"PaillierPrivateKey", "private_key", "decrypt"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name not in banned
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in banned
        elif isinstance(node, ast.Name):
            assert node.id not in {"PaillierPrivateKey"}


def test_score_matches_integer_dot_product():
    rng = random.Random(0)
    d = 24
    x_q = [rng.randint(-6, 6) for _ in range(d)]
    w_q = [rng.randint(-6, 6) for _ in range(d)]
    w_q[3] = 0
    b_q = rng.randint(-20, 20)
    public_key, private_key = crypto.generate_keypair(1024)
    server = Server(weights={"w_q": w_q, "b_q_prime": b_q, "d": d})
    encrypted = crypto.encrypt_vector(public_key, x_q)
    got = crypto.decrypt_scalar(private_key, server.score(public_key, encrypted))
    assert got == quantize.quantized_score([x_q], w_q, b_q)[0]


def test_parallel_score_matches_serial():
    rng = random.Random(1)
    d = 40
    x_q = [rng.randint(0, 900) for _ in range(d)]
    w_q = [rng.randint(-600, 600) for _ in range(d)]
    w_q[0] = w_q[7] = 0
    b_q = -12345
    public_key, private_key = crypto.generate_keypair(1024)
    server = Server(weights={"w_q": w_q, "b_q_prime": b_q, "d": d})
    encrypted = crypto.encrypt_vector(public_key, x_q)
    expected = quantize.quantized_score([x_q], w_q, b_q)[0]
    with ProcessPoolExecutor(max_workers=2) as executor:
        for chunks in (1, 3, 64):
            result = server.score(
                public_key, encrypted, executor=executor, chunks=chunks
            )
            assert crypto.decrypt_scalar(private_key, result) == expected
        pooled = Server(
            weights={"w_q": w_q, "b_q_prime": b_q, "d": d},
            executor=executor,
            chunks=4,
        )
        result = pooled.score(public_key, encrypted)
        assert crypto.decrypt_scalar(private_key, result) == expected


def test_score_request_round_trip():
    public_key, private_key = crypto.generate_keypair(1024)
    server = Server(weights={"w_q": [2, -3, 0], "b_q_prime": 4, "d": 3})
    payload = {
        "public_key": crypto.dump_public_key(public_key),
        "ciphertexts": [
            list(pair)
            for pair in (
                crypto.ciphertext_to_pair(c)
                for c in crypto.encrypt_vector(public_key, [5, 1, 9])
            )
        ],
    }
    got = crypto.decrypt_scalar(private_key, server.score_request(payload))
    assert got == 4 + 5 * 2 + 1 * (-3)


def test_rejects_bad_inputs():
    public_key, _ = crypto.generate_keypair(1024)
    server = Server(weights={"w_q": [1, 2], "b_q_prime": 0, "d": 2})
    encrypted = crypto.encrypt_vector(public_key, [1, 1])
    with pytest.raises(ValueError, match="length"):
        server.score(public_key, encrypted[:1])
    with pytest.raises(TypeError):
        server.score(public_key, [1, 2])
    tiny = PaillierPublicKey(n=65537)
    with pytest.raises(ValueError, match="modulus"):
        server.score(tiny, [EncryptedNumber(tiny, 1, 0), EncryptedNumber(tiny, 1, 0)])
    other, _ = crypto.generate_keypair(1024)
    foreign = crypto.encrypt_vector(other, [1, 1])
    with pytest.raises(ValueError, match="different key"):
        server.score(public_key, foreign)
