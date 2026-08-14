from dataclasses import replace
import random

import pytest

from kfrag.crypto.packets import create_packets, verify_and_recover_token
from kfrag.crypto.token import ProvenanceToken


def test_end_to_end_recovery_from_random_fragments() -> None:
    key = b"unit-test-secret"
    token = ProvenanceToken.generate(issuer_id=42, version=1)
    packets = create_packets(token, key)
    assert len(packets) == 16
    rng = random.Random(77)
    for _ in range(50):
        selection = rng.sample(packets, 12)
        assert verify_and_recover_token(selection, key) == token


def test_changed_symbol_or_tag_is_rejected() -> None:
    key = b"unit-test-secret"
    packets = list(create_packets(ProvenanceToken(1, 2, 1), key))
    changed_symbol = replace(packets[0], coded_symbol=packets[0].coded_symbol ^ 1)
    with pytest.raises(ValueError):
        verify_and_recover_token([changed_symbol, *packets[1:12]], key)
    changed_tag = replace(packets[0], authentication_tag=bytes(4))
    with pytest.raises(ValueError, match="authentication"):
        verify_and_recover_token([changed_tag, *packets[1:12]], key)


def test_wrong_key_duplicates_and_too_few_are_rejected() -> None:
    packets = create_packets(ProvenanceToken(1, 2, 1), b"right-key")
    with pytest.raises(ValueError, match="authentication"):
        verify_and_recover_token(packets[:12], b"wrong-key")
    with pytest.raises(ValueError, match="duplicate"):
        verify_and_recover_token([packets[0], *packets[:11]], b"right-key")
    with pytest.raises(ValueError, match="at least 12"):
        verify_and_recover_token(packets[:11], b"right-key")
