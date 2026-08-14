from dataclasses import replace

import pytest
import torch

from kfrag.crypto import ProvenanceToken, RegionalPacket, create_packets
from kfrag.payload.regional_tensor import (
    batch_packets_to_grid,
    bits_to_packet,
    expand_payload_grid,
    grid_to_packets,
    packet_to_bits,
    packets_to_grid,
)


@pytest.fixture
def packets() -> tuple[RegionalPacket, ...]:
    return create_packets(ProvenanceToken(7, 123456789, 1), b"payload-test-key")


def assert_binary_float32(tensor: torch.Tensor) -> None:
    assert tensor.dtype == torch.float32
    assert torch.all((tensor == 0) | (tensor == 1))


def test_packet_is_exactly_44_binary_bits_and_round_trips(
    packets: tuple[RegionalPacket, ...],
) -> None:
    bits = packet_to_bits(packets[5])
    assert bits.shape == (44,)
    assert_binary_float32(bits)
    assert bits_to_packet(bits) == packets[5]


def test_bit_order_is_msb_first_and_tag_byte_order_is_preserved() -> None:
    packet = RegionalPacket(0b1010, 0b11000001, bytes([0x80, 0x01, 0xA5, 0x5A]))
    bits = packet_to_bits(packet)
    assert bits[:4].tolist() == [1, 0, 1, 0]
    assert bits[4:12].tolist() == [1, 1, 0, 0, 0, 0, 0, 1]
    assert bits[12:20].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]
    assert bits[20:28].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    assert bits_to_packet(bits) == packet


def test_packets_map_to_row_major_grid_and_reconstruct(
    packets: tuple[RegionalPacket, ...],
) -> None:
    grid = packets_to_grid(reversed(packets))
    assert grid.shape == (44, 4, 4)
    assert_binary_float32(grid)
    assert torch.equal(grid[:, 0, 0], packet_to_bits(packets[0]))
    assert torch.equal(grid[:, 3, 3], packet_to_bits(packets[15]))
    assert grid_to_packets(grid) == list(packets)


def test_batch_and_nearest_expansion(packets: tuple[RegionalPacket, ...]) -> None:
    batch = batch_packets_to_grid([packets] * 4)
    assert batch.shape == (4, 44, 4, 4)
    assert_binary_float32(batch)
    expanded = expand_payload_grid(batch)
    assert expanded.shape == (4, 44, 256, 256)
    assert_binary_float32(expanded)
    for row in range(4):
        for column in range(4):
            expected = batch[:, :, row, column].unsqueeze(-1).unsqueeze(-1)
            assert torch.equal(
                expanded[:, :, row * 64 : (row + 1) * 64, column * 64 : (column + 1) * 64],
                expected.expand(-1, -1, 64, 64),
            )


def test_single_grid_expansion(packets: tuple[RegionalPacket, ...]) -> None:
    expanded = expand_payload_grid(packets_to_grid(packets))
    assert expanded.shape == (44, 256, 256)
    assert_binary_float32(expanded)


def test_duplicate_and_missing_packets_are_rejected(
    packets: tuple[RegionalPacket, ...],
) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        packets_to_grid([packets[0], *packets[:-1]])
    with pytest.raises(ValueError, match="exactly 16"):
        packets_to_grid(packets[:-1])


@pytest.mark.parametrize(
    "operation,value",
    [
        (bits_to_packet, torch.zeros(43)),
        (bits_to_packet, torch.zeros(45)),
        (grid_to_packets, torch.zeros(44, 4, 3)),
        (grid_to_packets, torch.zeros(45, 4, 4)),
        (expand_payload_grid, torch.zeros(44, 5, 4)),
        (expand_payload_grid, torch.zeros(2, 44, 4, 3)),
    ],
)
def test_malformed_tensor_shapes_are_rejected(operation, value: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="shape"):
        operation(value)


def test_nonbinary_tensors_are_rejected(packets: tuple[RegionalPacket, ...]) -> None:
    bits = packet_to_bits(packets[0])
    bits[20] = 0.5
    with pytest.raises(ValueError, match="binary"):
        bits_to_packet(bits)


def test_changing_one_grid_bit_changes_reconstructed_packet(
    packets: tuple[RegionalPacket, ...],
) -> None:
    grid = packets_to_grid(packets)
    changed = grid.clone()
    changed[4, 2, 1] = 1 - changed[4, 2, 1]  # coded-symbol bit in region 9
    reconstructed = grid_to_packets(changed)
    assert reconstructed[9] == replace(
        packets[9], coded_symbol=packets[9].coded_symbol ^ 0x80
    )
    assert reconstructed[:9] + reconstructed[10:] == list(packets[:9] + packets[10:])
