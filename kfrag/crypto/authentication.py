"""Packet authentication for K-FRAG protocol v1."""

from __future__ import annotations

import hashlib
import hmac

TAG_SIZE = 4


def _message(token_bytes: bytes, region_index: int, coded_symbol: int) -> bytes:
    if not isinstance(token_bytes, bytes) or len(token_bytes) != 12:
        raise ValueError("token_bytes must be exactly 12 bytes")
    if isinstance(region_index, bool) or not isinstance(region_index, int) or not 0 <= region_index <= 15:
        raise ValueError("region_index must be in the range 0..15")
    if isinstance(coded_symbol, bool) or not isinstance(coded_symbol, int) or not 0 <= coded_symbol <= 255:
        raise ValueError("coded_symbol must be an integer in the range 0..255")
    return token_bytes + bytes((region_index, coded_symbol))


def generate_tag(secret_key: bytes, token_bytes: bytes, region_index: int, coded_symbol: int) -> bytes:
    """Generate the four-byte truncated HMAC-SHA256 packet tag."""
    if not isinstance(secret_key, bytes) or not secret_key:
        raise ValueError("secret_key must be non-empty bytes")
    digest = hmac.new(secret_key, _message(token_bytes, region_index, coded_symbol), hashlib.sha256)
    return digest.digest()[:TAG_SIZE]


def verify_tag(
    secret_key: bytes,
    token_bytes: bytes,
    region_index: int,
    coded_symbol: int,
    tag: bytes,
) -> bool:
    """Compare a packet tag in constant time."""
    if not isinstance(tag, bytes) or len(tag) != TAG_SIZE:
        return False
    expected = generate_tag(secret_key, token_bytes, region_index, coded_symbol)
    return hmac.compare_digest(expected, tag)
