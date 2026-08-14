"""Reed-Solomon RS(16, 12) coding over one-byte symbols."""

from __future__ import annotations

from collections.abc import Iterable

from reedsolo import RSCodec, ReedSolomonError

DATA_SYMBOLS = 12
PARITY_SYMBOLS = 4
CODED_SYMBOLS = 16
_CODEC = RSCodec(PARITY_SYMBOLS)


def encode(data: bytes) -> tuple[int, ...]:
    """Encode 12 bytes into 16 integer-valued byte symbols."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if len(data) != DATA_SYMBOLS:
        raise ValueError(f"data must be exactly {DATA_SYMBOLS} bytes")
    encoded = bytes(_CODEC.encode(data))
    if len(encoded) != CODED_SYMBOLS:  # defensive check against codec changes
        raise RuntimeError("unexpected Reed-Solomon codeword length")
    return tuple(encoded)


def reconstruct(symbols: Iterable[tuple[int, int]]) -> bytes:
    """Recover data from indexed symbols, supplying absent indices as erasures."""
    received = list(symbols)
    indices = [index for index, _ in received]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate symbol indices are not allowed")
    if len(received) < DATA_SYMBOLS:
        raise ValueError(f"at least {DATA_SYMBOLS} symbols are required")

    codeword = bytearray(CODED_SYMBOLS)
    present: set[int] = set()
    for index, symbol in received:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < CODED_SYMBOLS:
            raise ValueError(f"symbol index must be in the range 0..{CODED_SYMBOLS - 1}")
        if isinstance(symbol, bool) or not isinstance(symbol, int) or not 0 <= symbol <= 255:
            raise ValueError("symbol must be an integer in the range 0..255")
        codeword[index] = symbol
        present.add(index)

    erasures = [index for index in range(CODED_SYMBOLS) if index not in present]
    try:
        decoded = _CODEC.decode(bytes(codeword), erase_pos=erasures)[0]
    except ReedSolomonError as exc:
        raise ValueError("symbols could not be reconstructed") from exc
    return bytes(decoded)
