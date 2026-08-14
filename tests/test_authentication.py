from kfrag.crypto.authentication import generate_tag, verify_tag


def test_tag_is_four_bytes_and_verifies() -> None:
    key = b"test-only-key"
    token = bytes(range(12))
    tag = generate_tag(key, token, 3, 99)
    assert len(tag) == 4
    assert verify_tag(key, token, 3, 99, tag)


def test_changed_inputs_and_wrong_key_fail() -> None:
    token = bytes(range(12))
    tag = generate_tag(b"right-key", token, 3, 99)
    assert not verify_tag(b"wrong-key", token, 3, 99, tag)
    assert not verify_tag(b"right-key", token, 3, 98, tag)
    assert not verify_tag(b"right-key", token, 3, 99, bytes([tag[0] ^ 1]) + tag[1:])
