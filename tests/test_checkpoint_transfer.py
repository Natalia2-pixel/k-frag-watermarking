from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from kfrag.diagnostics.checkpoint_transfer import (ACTIVE_CHANNELS,
    CHECKPOINT_SCHEMA_VERSION, CheckpointCompatibilityError, load_stage_a_checkpoint,
    make_stage_a_checkpoint, relative_checkpoint_path, save_stage_a_checkpoints,
    sha256_file, state_dict_equal)
from kfrag.diagnostics.stage_b_natural import decoder_accepts_only_questioned_image, run_stage_b
from kfrag.models import StructuredChannelSystem


def checkpoint_fixture(*, passed: bool = True):
    torch.manual_seed(17)
    model = StructuredChannelSystem(mode="learnable", image_size=16, alpha=.05)
    with torch.no_grad():
        model.carrier_bank.carriers.add_(.001)
        model.decoder.log_gain.add_(.01)
    model.sync_decoder_carriers()
    checkpoint = make_stage_a_checkpoint(
        model, optimizer_state_dict={"state": {}, "param_groups": []},
        completed_training_step=4, config={"alpha": .05}, seed=17,
        gate_results={"all_gates_passed": passed}, passed=passed)
    return model, checkpoint


def test_successful_checkpoint_contains_carrier_decoder_and_required_metadata():
    _, checkpoint = checkpoint_fixture()
    assert checkpoint["carrier_state_dict"]
    assert checkpoint["regional_decoder_state_dict"]
    assert checkpoint["optimizer_state_dict"] is not None
    assert checkpoint["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["active_channels"] == ACTIVE_CHANNELS


def test_best_is_created_only_after_all_gates_pass_and_failure_cannot_overwrite(tmp_path):
    _, failed = checkpoint_fixture(passed=False)
    result = save_stage_a_checkpoints(tmp_path, failed)
    assert result == {"last_saved": True, "best_saved": False}
    assert (tmp_path / "last.pt").is_file()
    assert not (tmp_path / "best.pt").exists()
    _, passed = checkpoint_fixture(passed=True)
    save_stage_a_checkpoints(tmp_path, passed)
    original = (tmp_path / "best.pt").read_bytes()
    save_stage_a_checkpoints(tmp_path, failed)
    assert (tmp_path / "best.pt").read_bytes() == original
    assert not torch.load(tmp_path / "last.pt", weights_only=False)["stage_a_passed"]


def _rejects(tmp_path, mutate, match):
    _, checkpoint = checkpoint_fixture()
    mutate(checkpoint)
    path = tmp_path / "checkpoint.pt"; torch.save(checkpoint, path)
    target = StructuredChannelSystem(mode="learnable", image_size=16, alpha=.05)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(CheckpointCompatibilityError, match=match):
        load_stage_a_checkpoint(path, target, expected_alpha=.05)
    assert state_dict_equal(before, target.state_dict())


def test_missing_required_checkpoint_raises_clear_error(tmp_path):
    target = StructuredChannelSystem(mode="learnable", image_size=16)
    with pytest.raises(FileNotFoundError, match="required Stage-A checkpoint"):
        load_stage_a_checkpoint(tmp_path / "missing.pt", target, expected_alpha=.05)


@pytest.mark.parametrize(("mutate", "match"), [
    (lambda c: c.__setitem__("checkpoint_schema_version", "999"), "schema"),
    (lambda c: c.__setitem__("active_channels", list(range(8))), "active_channels"),
    (lambda c: c.__setitem__("grid_size", [2, 8]), "grid_size"),
    (lambda c: c.__setitem__("residual_alpha", .1), "residual_alpha"),
    (lambda c: c.__setitem__("residual_parameterization", "unbounded"), "residual_parameterization"),
])
def test_incompatible_checkpoint_metadata_is_rejected(tmp_path, mutate, match):
    _rejects(tmp_path, mutate, match)


def test_incompatible_parameter_shape_is_rejected_without_partial_loading(tmp_path):
    def mutate(checkpoint):
        checkpoint["regional_decoder_state_dict"]["active_bias"] = torch.zeros(7)
        checkpoint["tensor_dimensions"]["decoder"]["active_bias"] = [7]
    _rejects(tmp_path, mutate, "parameter shape")


def test_strict_loading_restores_both_models_and_identical_outputs(tmp_path):
    source, checkpoint = checkpoint_fixture()
    path = tmp_path / "best.pt"; torch.save(checkpoint, path)
    target = StructuredChannelSystem(mode="learnable", image_size=16, alpha=.05)
    loaded, result = load_stage_a_checkpoint(path, target, expected_alpha=.05)
    assert result["passed"]
    assert state_dict_equal(source.carrier_bank.state_dict(), target.carrier_bank.state_dict())
    assert state_dict_equal(source.decoder.state_dict(), target.decoder.state_dict())
    image = torch.rand(2, 3, 16, 16)
    payload = torch.zeros(2, 44, 4, 4); payload[:, 4:12] = torch.randint(0, 2, (2, 8, 4, 4))
    for name, value in source(image, payload).items():
        assert torch.equal(value, target(image, payload)[name])
    assert loaded["completed_training_step"] == 4


def test_hash_and_report_path_are_portable(tmp_path):
    _, checkpoint = checkpoint_fixture()
    path = tmp_path / "best.pt"; torch.save(checkpoint, path)
    assert sha256_file(path) == __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    relative = relative_checkpoint_path(path, repository_root=tmp_path)
    assert not Path(relative).is_absolute()
    assert "C:\\Users\\" not in relative


def test_checkpoint_transfer_directory_is_separate_and_decoder_remains_blind(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_transfer"):
        run_stage_b({"output_directory": str(tmp_path / "stage_b_natural")})
    assert decoder_accepts_only_questioned_image()
    baseline = Path("outputs/stage_b_natural/analytical_baseline/summary.json")
    assert baseline.is_file()
