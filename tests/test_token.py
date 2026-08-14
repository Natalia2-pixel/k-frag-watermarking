import pytest

from kfrag.crypto.token import ProvenanceToken


def test_pack_unpack_round_trip() -> None:
    token = ProvenanceToken(0xABCDEF, 0x0123456789ABCDEF, 1)
    packed = token.pack()
    assert len(packed) == 12
    assert packed == bytes.fromhex("abcdef0123456789abcdef01")
    assert ProvenanceToken.unpack(packed) == token


@pytest.mark.parametrize(
    ("field", "values"),
    [("issuer_id", (-1, 1 << 24)), ("asset_id", (-1, 1 << 64)), ("version", (-1, 256))],
)
def test_integer_ranges(field: str, values: tuple[int, int]) -> None:
    arguments = {"issuer_id": 1, "asset_id": 2, "version": 1}
    for value in values:
        arguments[field] = value
        with pytest.raises(ValueError):
            ProvenanceToken(**arguments)


def test_generate_produces_valid_asset_id() -> None:
    token = ProvenanceToken.generate(issuer_id=7)
    assert 0 <= token.asset_id < (1 << 64)
