"""Convert authenticated regional packets to and from binary tensors.

The 44-bit packet layout is, in order::

    region index (4 bits) | coded symbol (8 bits) | authentication tag (32 bits)

All integers and individual tag bytes are encoded most-significant-bit first.
Tag byte order is unchanged.  These conversions operate only on already-created
packets and therefore require no secret key.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.nn import functional as F

from kfrag.crypto.packets import RegionalPacket

REGION_INDEX_BITS = 4
CODED_SYMBOL_BITS = 8
AUTHENTICATION_TAG_BITS = 32
PACKET_BITS = REGION_INDEX_BITS + CODED_SYMBOL_BITS + AUTHENTICATION_TAG_BITS
NUM_REGIONS = 16
GRID_SIZE = 4


def _integer_bits(value: int, width: int) -> list[int]:
    """Return an unsigned integer's fixed-width, MSB-first bits."""
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def _validated_binary_tensor(value: torch.Tensor, shape: tuple[int, ...], name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {list(shape)}, got {list(value.shape)}")
    if not torch.all((value == 0) | (value == 1)).item():
        raise ValueError(f"{name} must contain only binary values 0 or 1")
    return value


def packet_to_bits(packet: RegionalPacket) -> torch.Tensor:
    """Encode one regional packet as a float32 binary tensor of shape ``[44]``."""
    if not isinstance(packet, RegionalPacket):
        raise TypeError("packet must be a RegionalPacket")
    bits = _integer_bits(packet.region_index, REGION_INDEX_BITS)
    bits.extend(_integer_bits(packet.coded_symbol, CODED_SYMBOL_BITS))
    for byte in packet.authentication_tag:
        bits.extend(_integer_bits(byte, 8))
    return torch.tensor(bits, dtype=torch.float32)


def bits_to_packet(bits: torch.Tensor) -> RegionalPacket:
    """Decode a ``[44]`` binary tensor into its exact packet fields."""
    bits = _validated_binary_tensor(bits, (PACKET_BITS,), "bits")
    values = bits.to(dtype=torch.uint8, device="cpu").tolist()

    def unsigned(start: int, width: int) -> int:
        result = 0
        for bit in values[start : start + width]:
            result = (result << 1) | bit
        return result

    region_index = unsigned(0, REGION_INDEX_BITS)
    coded_symbol = unsigned(REGION_INDEX_BITS, CODED_SYMBOL_BITS)
    tag_start = REGION_INDEX_BITS + CODED_SYMBOL_BITS
    authentication_tag = bytes(unsigned(tag_start + offset, 8) for offset in range(0, 32, 8))
    return RegionalPacket(region_index, coded_symbol, authentication_tag)


def packets_to_grid(
    packets: Iterable[RegionalPacket], grid_size: int = GRID_SIZE
) -> torch.Tensor:
    """Place all 16 packets in a row-major ``[44, 4, 4]`` payload grid."""
    if grid_size != GRID_SIZE:
        raise ValueError("grid_size must be 4 for the 16-region protocol")
    packet_list = list(packets)
    if len(packet_list) != NUM_REGIONS:
        raise ValueError(f"exactly {NUM_REGIONS} packets are required")

    grid = torch.empty((PACKET_BITS, GRID_SIZE, GRID_SIZE), dtype=torch.float32)
    indices: set[int] = set()
    for packet in packet_list:
        if not isinstance(packet, RegionalPacket):
            raise TypeError("all packets must be RegionalPacket instances")
        index = packet.region_index
        if not 0 <= index < NUM_REGIONS:
            raise ValueError("region indices must be in the range 0..15")
        if index in indices:
            raise ValueError(f"duplicate region index: {index}")
        indices.add(index)
        row, column = divmod(index, GRID_SIZE)
        grid[:, row, column] = packet_to_bits(packet)

    missing = set(range(NUM_REGIONS)) - indices
    if missing:
        raise ValueError(f"missing region indices: {sorted(missing)}")
    return grid


def grid_to_packets(grid: torch.Tensor) -> list[RegionalPacket]:
    """Reconstruct the 16 row-major packets from a ``[44, 4, 4]`` grid."""
    grid = _validated_binary_tensor(grid, (PACKET_BITS, GRID_SIZE, GRID_SIZE), "grid")
    packets: list[RegionalPacket] = []
    for index in range(NUM_REGIONS):
        row, column = divmod(index, GRID_SIZE)
        packet = bits_to_packet(grid[:, row, column])
        if packet.region_index != index:
            raise ValueError(
                f"packet at grid position ({row}, {column}) encodes region index "
                f"{packet.region_index}, expected {index}"
            )
        packets.append(packet)
    return packets


def batch_packets_to_grid(
    batch_packets: Iterable[Iterable[RegionalPacket]],
) -> torch.Tensor:
    """Convert packet collections into a ``[B, 44, 4, 4]`` tensor."""
    grids = [packets_to_grid(packets) for packets in batch_packets]
    if not grids:
        raise ValueError("batch_packets must contain at least one packet collection")
    return torch.stack(grids)


def expand_payload_grid(grid: torch.Tensor, image_size: int = 256) -> torch.Tensor:
    """Nearest-neighbour expand a payload grid to image spatial resolution."""
    if not isinstance(grid, torch.Tensor):
        raise TypeError("grid must be a torch.Tensor")
    if image_size <= 0 or image_size % GRID_SIZE != 0:
        raise ValueError("image_size must be a positive multiple of 4")
    if grid.ndim == 3:
        _validated_binary_tensor(grid, (PACKET_BITS, GRID_SIZE, GRID_SIZE), "grid")
        expanded = F.interpolate(
            grid.to(dtype=torch.float32).unsqueeze(0),
            size=(image_size, image_size),
            mode="nearest",
        ).squeeze(0)
    elif grid.ndim == 4:
        if grid.shape[0] < 1 or tuple(grid.shape[1:]) != (PACKET_BITS, GRID_SIZE, GRID_SIZE):
            raise ValueError(f"grid must have shape [B, {PACKET_BITS}, 4, 4]")
        if not torch.all((grid == 0) | (grid == 1)).item():
            raise ValueError("grid must contain only binary values 0 or 1")
        expanded = F.interpolate(
            grid.to(dtype=torch.float32), size=(image_size, image_size), mode="nearest"
        )
    else:
        raise ValueError("grid must have shape [44, 4, 4] or [B, 44, 4, 4]")
    return expanded
