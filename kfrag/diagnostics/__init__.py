"""Controlled diagnostics for the learned watermark channel."""

from .channel_sanity import (
    CAPACITY_RANGES,
    capacity_mask,
    circular_payload_shuffle,
    generate_payload_bank,
    masked_bce_with_logits,
    payload_splits_are_disjoint,
    run_channel_sanity,
    should_advance_capacity,
    validate_config,
)

__all__ = [
    "CAPACITY_RANGES", "capacity_mask", "circular_payload_shuffle",
    "generate_payload_bank", "masked_bce_with_logits", "payload_splits_are_disjoint",
    "run_channel_sanity", "should_advance_capacity", "validate_config",
]
