"""Evaluation utilities for clean K-FRAG watermark models."""

from .anti_memorization import (
    FIELD_SLICES,
    circular_shift_targets,
    evaluate_anti_memorization,
    evaluate_checkpoint,
    field_metrics,
)

__all__ = ["FIELD_SLICES", "circular_shift_targets", "evaluate_anti_memorization",
           "evaluate_checkpoint", "field_metrics"]
