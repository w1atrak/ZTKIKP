"""Paillier vectors, homomorphisms and serialization (phase B2)."""

import pytest

from ppspam import crypto

_X = [3, -2, 0, 5]
_W = [4, -7, 9, 1]
_B = -3


@pytest.fixture(scope="module", params=[1024, 2048])
def keypair(request):
    return crypto.generate_keypair(request.param)


def test_round_trip_and_exponent(keypair):
    public_key, private_key = keypair
    encrypted = crypto.encrypt_vector(public_key, _X)
    assert [crypto.decrypt_scalar(private_key, c) for c in encrypted] == _X
    assert all(c.exponent == 0 for c in encrypted)


def test_encrypt_vector_rejects_non_ints(keypair):
    public_key, _ = keypair
    with pytest.raises(TypeError):
        crypto.encrypt_vector(public_key, [1.5])
    with pytest.raises(TypeError):
        crypto.encrypt_vector(public_key, [True])


def test_negative_weights_homomorphism(keypair):
    public_key, private_key = keypair
    encrypted = crypto.encrypt_vector(public_key, _X)
    acc = public_key.encrypt(_B)
    for cipher, weight in zip(encrypted, _W, strict=True):
        acc += cipher * weight
    expected = _B + sum(x * w for x, w in zip(_X, _W, strict=True))
    assert crypto.decrypt_scalar(private_key, acc) == expected


def test_decimal_and_compact_round_trip(keypair):
    public_key, private_key = keypair
    encrypted = crypto.encrypt_vector(public_key, [-11, 0])
    for cipher, plain in zip(encrypted, [-11, 0], strict=True):
        restored = crypto.ciphertext_from_pair(
            public_key, crypto.ciphertext_to_pair(cipher)
        )
        assert crypto.decrypt_scalar(private_key, restored) == plain
        compact = crypto.ciphertext_to_compact(cipher)
        restored_c = crypto.ciphertext_from_compact(public_key, compact)
        assert crypto.decrypt_scalar(private_key, restored_c) == plain


def test_compact_json_is_smaller(keypair):
    public_key, _ = keypair
    sizes = crypto.encoding_sizes(public_key.encrypt(42))
    assert sizes["raw_bytes"] == public_key.n.bit_length() * 2 // 8
    assert sizes["b64_json"] < sizes["decimal_json"]


def test_public_key_round_trip_has_no_secret(keypair):
    public_key, private_key = keypair
    payload = crypto.dump_public_key(public_key)
    crypto.assert_public_payload(payload)
    assert crypto.load_public_key(payload).n == public_key.n
    assert not hasattr(payload, "p")
    assert "p" not in payload and "q" not in payload
    assert private_key.p != public_key.n

    vector = {
        "public_key": payload,
        "ciphertexts": [
            list(crypto.ciphertext_to_pair(c))
            for c in crypto.encrypt_vector(public_key, [0, 1])
        ],
    }
    crypto.assert_public_payload(vector)
    with pytest.raises(ValueError):
        crypto.assert_public_payload({"n": "1", "p": "2", "q": "3"})
    with pytest.raises(ValueError):
        crypto.load_public_key({"n": str(public_key.n), "p": "5"})


def test_pool_encryption_round_trip_and_consumes_once(keypair):
    public_key, private_key = keypair
    pool = crypto.ObfuscatorPool(public_key, size=len(_X) + 2)
    encrypted = crypto.encrypt_vector(public_key, _X, pool=pool)
    assert len(pool) == 2
    assert [crypto.decrypt_scalar(private_key, c) for c in encrypted] == _X
    assert all(c.exponent == 0 for c in encrypted)
    raw = [c.ciphertext(be_secure=False) for c in encrypted]
    assert [c.ciphertext() for c in encrypted] == raw  # already obfuscated


def test_pool_gives_distinct_ciphertexts_for_equal_values(keypair):
    public_key, _ = keypair
    pool = crypto.ObfuscatorPool(public_key, size=4)
    encrypted = crypto.encrypt_vector(public_key, [0, 0, 0, 0], pool=pool)
    assert len({c.ciphertext() for c in encrypted}) == 4


def test_pool_refuses_short_or_foreign_pool(keypair):
    public_key, _ = keypair
    short = crypto.ObfuscatorPool(public_key, size=1)
    with pytest.raises(RuntimeError, match="pool has 1"):
        crypto.encrypt_vector(public_key, [1, 2], pool=short)
    assert len(short) == 1
    short.take()
    with pytest.raises(RuntimeError, match="empty"):
        short.take()
    other, _ = crypto.generate_keypair(1024)
    with pytest.raises(ValueError, match="different public key"):
        crypto.encrypt_vector(
            public_key, [1], pool=crypto.ObfuscatorPool(other, size=1)
        )


def test_rejects_ciphertext_outside_modulus(keypair):
    public_key, _ = keypair
    with pytest.raises(ValueError):
        crypto.ciphertext_from_pair(public_key, (str(public_key.nsquare), 0))
