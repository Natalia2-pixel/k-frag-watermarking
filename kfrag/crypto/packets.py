"""Authenticated regional packets and protocol orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from kfrag.crypto.authentication import TAG_SIZE, generate_tag, verify_tag
from kfrag.crypto.reed_solomon import encode, reconstruct
from kfrag.crypto.token import ProvenanceToken


@dataclass(frozen=True)
class RegionalPacket:
    """One indexed coded byte and its four-byte authentication tag."""

    region_index: int
    coded_symbol: int
    authentication_tag: bytes

    def __post_init__(self) -> None:
        if isinstance(self.region_index, bool) or not isinstance(self.region_index, int) or not 0 <= self.region_index <= 15:
            raise ValueError("region_index must be in the range 0..15")
        if isinstance(self.coded_symbol, bool) or not isinstance(self.coded_symbol, int) or not 0 <= self.coded_symbol <= 255:
            raise ValueError("coded_symbol must be an integer in the range 0..255")
        if not isinstance(self.authentication_tag, bytes) or len(self.authentication_tag) != TAG_SIZE:
            raise ValueError(f"authentication_tag must be exactly {TAG_SIZE} bytes")


def create_packets(token: ProvenanceToken, secret_key: bytes) -> tuple[RegionalPacket, ...]:
    """Encode a token into all 16 authenticated regional packets."""
    token_bytes = token.pack()
    return tuple(
        RegionalPacket(index, symbol, generate_tag(secret_key, token_bytes, index, symbol))
        for index, symbol in enumerate(encode(token_bytes))
    )


def verify_and_recover_token(
    packets: Iterable[RegionalPacket], secret_key: bytes
) -> ProvenanceToken:
    """Reconstruct a token and authenticate every supplied regional packet."""
    supplied = list(packets)
    token_bytes = reconstruct(
        (packet.region_index, packet.coded_symbol) for packet in supplied
    )
    for packet in supplied:
        if not verify_tag(
            secret_key,
            token_bytes,
            packet.region_index,
            packet.coded_symbol,
            packet.authentication_tag,
        ):
            raise ValueError(f"authentication failed for region {packet.region_index}")
    return ProvenanceToken.unpack(token_bytes)
