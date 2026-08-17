import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_kfrag.py"


def invoke(*arguments):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, arguments)], cwd=ROOT,
        text=True, capture_output=True, timeout=120)


def config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": "1.0", "experiment_name": "cli_test",
        "data_root": str(tmp_path / "absent-data"), "image_size": 32,
        "steps": 9, "batch_size": 2, "width": 4, "seed": 19,
    }), encoding="utf-8")
    return path


def test_help_lists_new_options():
    result = invoke("--help")
    assert result.returncode == 0
    for option in ("--dry-run", "--device", "--seed", "--output-directory", "--resume"):
        assert option in result.stdout


def test_dry_run_validates_and_writes_nothing(tmp_path):
    cfg = config(tmp_path); output = tmp_path / "planned"
    result = invoke("--config", cfg, "--smoke", "--dry-run", "--device", "cpu", "--output-directory", output)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["dry_run"] is True and plan["device"] == "cpu"
    assert plan["output_directory"] == output.as_posix()
    assert plan["scientific_stage"] == "natural_image_communication"
    assert not output.exists()
    assert not list(tmp_path.rglob("*.pt"))


def test_seeded_cpu_smoke_is_deterministic_and_disclaims_scientific_success(tmp_path):
    cfg = config(tmp_path); first = tmp_path / "first"; second = tmp_path / "second"
    a = invoke("--config", cfg, "--smoke", "--device", "cpu", "--seed", "7", "--output-directory", first)
    b = invoke("--config", cfg, "--smoke", "--device", "cpu", "--seed", "7", "--output-directory", second)
    assert a.returncode == b.returncode == 0, a.stderr + b.stderr
    assert "not scientific gate success" in a.stdout
    assert (first / "history.csv").read_text() == (second / "history.csv").read_text()
    one = json.loads((first / "summary.json").read_text()); two = json.loads((second / "summary.json").read_text())
    assert one["metrics"] == two["metrics"] and one["gates"] == two["gates"]


def test_invalid_device_is_rejected_before_output_creation(tmp_path):
    output = tmp_path / "output"
    result = invoke("--config", config(tmp_path), "--smoke", "--device", "quantum:0", "--output-directory", output)
    assert result.returncode != 0 and "invalid device" in result.stderr
    assert not output.exists()


def test_existing_output_directory_is_protected(tmp_path):
    output = tmp_path / "existing"; output.mkdir(); marker = output / "keep.txt"; marker.write_text("immutable")
    result = invoke("--config", config(tmp_path), "--smoke", "--output-directory", output)
    assert result.returncode != 0 and "immutable run already exists" in result.stderr
    assert marker.read_text() == "immutable" and list(output.iterdir()) == [marker]
