"""Versioned Stage-A checkpoints and strict, atomic Stage-B restoration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from kfrag.models import StructuredChannelSystem


CHECKPOINT_SCHEMA_VERSION = "1.0"
CARRIER_ARCHITECTURE = "RegionalCarrierBank.v1"
DECODER_ARCHITECTURE = "StructuredRegionalDecoder.v1"
GRID_SIZE = [4, 4]
ACTIVE_CHANNELS = list(range(4, 12))
RESIDUAL_PARAMETERIZATION = "alpha_times_tanh_normalized_orthogonal_carrier_sum"


class CheckpointCompatibilityError(ValueError):
    """Raised before mutation when a Stage-A checkpoint cannot be transferred."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_checkpoint_path(path: str | Path, repository_root: str | Path | None = None) -> str:
    path = Path(path)
    root = Path.cwd() if repository_root is None else Path(repository_root)
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("checkpoint path must be inside the repository") from error
    return relative.as_posix()


def _shapes(state: Mapping[str, torch.Tensor]) -> dict[str, list[int]]:
    return {name: list(value.shape) for name, value in state.items()}


def make_stage_a_checkpoint(model: StructuredChannelSystem, *, optimizer_state_dict: Mapping[str, Any] | None,
                            completed_training_step: int, config: Mapping[str, Any], seed: int,
                            gate_results: Mapping[str, Any], passed: bool) -> dict[str, Any]:
    if model.carrier_bank.mode != "learnable":
        raise ValueError("Stage-A transfer checkpoints require a genuinely learnable carrier")
    carrier = model.carrier_bank.state_dict()
    decoder = model.decoder.state_dict()
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "carrier_state_dict": carrier,
        "regional_decoder_state_dict": decoder,
        "optimizer_state_dict": optimizer_state_dict,
        "completed_training_step": int(completed_training_step),
        "stage_a_configuration": dict(config),
        "active_channels": ACTIVE_CHANNELS,
        "grid_size": GRID_SIZE,
        "residual_alpha": float(model.carrier_bank.alpha),
        "residual_parameterization": RESIDUAL_PARAMETERIZATION,
        "initialization_seed": int(seed),
        "carrier_architecture_identifier": CARRIER_ARCHITECTURE,
        "decoder_architecture_identifier": DECODER_ARCHITECTURE,
        "tensor_dimensions": {"carrier": _shapes(carrier), "decoder": _shapes(decoder)},
        "stage_a_gate_results": dict(gate_results),
        "stage_a_passed": bool(passed),
        "genuinely_learned_carrier": True,
    }


def save_stage_a_checkpoints(output: str | Path, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Always save last; replace best only for a passing learned checkpoint."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(dict(checkpoint), output / "last.pt")
    if checkpoint.get("stage_a_passed") and checkpoint.get("genuinely_learned_carrier"):
        torch.save(dict(checkpoint), output / "best.pt")
    return {"last_saved": True, "best_saved": bool(
        checkpoint.get("stage_a_passed") and checkpoint.get("genuinely_learned_carrier"))}


def _require(checkpoint: Mapping[str, Any], name: str) -> Any:
    if name not in checkpoint:
        raise CheckpointCompatibilityError(f"Stage-A checkpoint is missing required field: {name}")
    return checkpoint[name]


def validate_stage_a_checkpoint(checkpoint: Mapping[str, Any], model: StructuredChannelSystem,
                                *, expected_alpha: float) -> dict[str, Any]:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "carrier_architecture_identifier": CARRIER_ARCHITECTURE,
        "decoder_architecture_identifier": DECODER_ARCHITECTURE,
        "grid_size": GRID_SIZE,
        "active_channels": ACTIVE_CHANNELS,
        "residual_parameterization": RESIDUAL_PARAMETERIZATION,
    }
    for field, value in expected.items():
        actual = _require(checkpoint, field)
        if actual != value:
            raise CheckpointCompatibilityError(
                f"incompatible {field}: expected {value!r}, found {actual!r}")
    alpha = float(_require(checkpoint, "residual_alpha"))
    if alpha != float(expected_alpha):
        raise CheckpointCompatibilityError(
            f"incompatible residual_alpha: expected {expected_alpha!r}, found {alpha!r}")
    for field in ("completed_training_step", "stage_a_configuration", "initialization_seed",
                  "stage_a_gate_results", "stage_a_passed", "genuinely_learned_carrier",
                  "tensor_dimensions"):
        _require(checkpoint, field)
    if not checkpoint["stage_a_passed"]:
        raise CheckpointCompatibilityError("Stage-A checkpoint did not pass every Stage-A gate")
    if not checkpoint["genuinely_learned_carrier"]:
        raise CheckpointCompatibilityError("Stage-A checkpoint carrier was not genuinely learned")
    states = {
        "carrier": (_require(checkpoint, "carrier_state_dict"), model.carrier_bank.state_dict()),
        "decoder": (_require(checkpoint, "regional_decoder_state_dict"), model.decoder.state_dict()),
    }
    for label, (saved, current) in states.items():
        if set(saved) != set(current):
            missing, extra = sorted(set(current) - set(saved)), sorted(set(saved) - set(current))
            raise CheckpointCompatibilityError(
                f"incompatible {label} state-dictionary keys; missing={missing}, unexpected={extra}")
        for name in current:
            if not isinstance(saved[name], torch.Tensor) or saved[name].shape != current[name].shape:
                found = None if not isinstance(saved[name], torch.Tensor) else list(saved[name].shape)
                raise CheckpointCompatibilityError(
                    f"incompatible {label} parameter shape for {name}: "
                    f"expected {list(current[name].shape)}, found {found}")
        recorded = checkpoint["tensor_dimensions"].get(label)
        if recorded != _shapes(saved):
            raise CheckpointCompatibilityError(f"incompatible recorded tensor dimensions for {label}")
    return {"passed": True, "checks": [*expected, "residual_alpha", "required_metadata",
                                        "state_dictionary_keys", "parameter_tensor_dimensions"]}


def load_stage_a_checkpoint(path: str | Path, model: StructuredChannelSystem,
                            *, expected_alpha: float) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required Stage-A checkpoint does not exist: {path.as_posix()}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise CheckpointCompatibilityError("Stage-A checkpoint root must be a mapping")
    result = validate_stage_a_checkpoint(checkpoint, model, expected_alpha=expected_alpha)
    # Validation above is complete, so neither model component can be partially loaded.
    model.carrier_bank.load_state_dict(checkpoint["carrier_state_dict"], strict=True)
    model.decoder.load_state_dict(checkpoint["regional_decoder_state_dict"], strict=True)
    return dict(checkpoint), result


def state_dict_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)
