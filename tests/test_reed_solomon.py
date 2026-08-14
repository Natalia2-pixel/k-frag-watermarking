import random

import pytest

from kfrag.crypto.reed_solomon import encode, reconstruct


def test_encode_produces_16_symbols() -> None:
    assert len(encode(bytes(range(12)))) == 16


def test_random_sets_of_12_reconstruct() -> None:
    data = bytes(range(12))
    symbols = encode(data)
    rng = random.Random(2026)
    for _ in range(100):
        indices = rng.sample(range(16), 12)
        assert reconstruct((index, symbols[index]) for index in indices) == data


def test_invalid_symbol_sets_are_rejected() -> None:
    symbols = encode(bytes(range(12)))
    with pytest.raises(ValueError, match="duplicate"):
        reconstruct([(0, symbols[0])] * 12)
    with pytest.raises(ValueError, match="index"):
        reconstruct([(index, 0) for index in range(11)] + [(16, 0)])
    with pytest.raises(ValueError, match="at least 12"):
        reconstruct((index, symbols[index]) for index in range(11))
