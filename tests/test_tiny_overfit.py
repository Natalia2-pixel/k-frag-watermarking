import csv
import json

import torch

from kfrag.training import build_fixed_payloads, run_tiny_overfit


def test_fixed_authenticated_payloads_are_deterministic_and_unique() -> None:
    first = build_fixed_payloads(8, seed=2026)
    second = build_fixed_payloads(8, seed=2026)
    assert first.shape == (8, 44, 4, 4)
    assert torch.equal(first, second)
    assert len({payload.numpy().tobytes() for payload in first}) == 8


def test_short_synthetic_cpu_run_writes_safe_artifacts(tmp_path) -> None:
    generator = torch.Generator().manual_seed(10)
    images = torch.rand((8, 3, 256, 256), generator=generator)
    payloads = build_fixed_payloads(8, seed=11)
    config = {
        "num_images": 8,
        "image_size": 256,
        "steps": 1,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "log_every": 1,
        "seed": 12,
        "num_workers": 0,
        "device": "cpu",
        "base_channels": 1,
        "message_channels": 1,
        "residual_alpha": 0.02,
    }

    summary = run_tiny_overfit(config, images, payloads, tmp_path)

    assert summary["steps_completed"] == 1
    assert set(summary["final_metrics"]) == {
        "total_loss", "payload_bce_loss", "image_fidelity_loss", "bit_accuracy",
        "regional_packet_accuracy", "image_payload_accuracy", "psnr",
        "max_absolute_residual",
    }
    assert {path.name for path in tmp_path.iterdir()} == {
        "best.pt", "last.pt", "history.csv", "summary.json"
    }
    checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert set(checkpoint) == {"model_state", "optimizer_state", "step", "metrics", "configuration",
                               "evaluation_bundle"}
    assert tuple(checkpoint["evaluation_bundle"]["payloads"].shape) == (8, 44, 4, 4)
    assert "secret" not in repr(checkpoint).lower()
    with (tmp_path / "history.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    assert json.loads((tmp_path / "summary.json").read_text())["device"] == "cpu"
