"""The fixed-width K-FRAG provenance token."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

TOKEN_SIZE = 12


def _validate_uint(name: str, value: int, bits: int) -> None:
    """Validate that *value* is an unsigned integer of the requested width."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value < (1 << bits):
        raise ValueError(f"{name} must be in the range 0..{(1 << bits) - 1}")


@dataclass(frozen=True)
class ProvenanceToken:
    """A protocol-v1 token containing exactly 96 bits of provenance data."""

    issuer_id: int
    asset_id: int
    version: int

    def __post_init__(self) -> None:
        _validate_uint("issuer_id", self.issuer_id, 24)
        _validate_uint("asset_id", self.asset_id, 64)
        _validate_uint("version", self.version, 8)

    @classmethod
    def generate(cls, issuer_id: int, version: int = 1) -> ProvenanceToken:
        """Create a token with a cryptographically random 64-bit asset ID."""
        return cls(issuer_id=issuer_id, asset_id=secrets.randbits(64), version=version)

    def pack(self) -> bytes:
        """Serialize the token deterministically in network byte order."""
        return (
            self.issuer_id.to_bytes(3, "big")
            + self.asset_id.to_bytes(8, "big")
            + self.version.to_bytes(1, "big")
        )

    @classmethod
    def unpack(cls, data: bytes) -> ProvenanceToken:
        """Deserialize exactly 12 bytes into a provenance token."""
        if not isinstance(data, bytes):
            raise TypeError("token data must be bytes")
        if len(data) != TOKEN_SIZE:
            raise ValueError(f"token data must be exactly {TOKEN_SIZE} bytes")
        return cls(
            issuer_id=int.from_bytes(data[0:3], "big"),
            asset_id=int.from_bytes(data[3:11], "big"),
            version=data[11],
        )
