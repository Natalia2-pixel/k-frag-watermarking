"""Packaging regression tests for modules required by command-line entry points."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_clean_install_exposes_checkpoint_api(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(REPOSITORY_ROOT / "pyproject.toml", source)
    shutil.copy2(REPOSITORY_ROOT / "README.md", source)
    shutil.copytree(REPOSITORY_ROOT / "kfrag", source / "kfrag")
    installation = tmp_path / "installation"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(installation),
            str(source),
        ],
        check=True,
        cwd=tmp_path,
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installation)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import kfrag.checkpoints; "
                "from kfrag.checkpoints.provenance import "
                "load_checkpoint, save_checkpoint, sha256_file; "
                "assert all(callable(symbol) for symbol in "
                "(load_checkpoint, save_checkpoint, sha256_file))"
            ),
        ],
        check=True,
        cwd=tmp_path,
        env=environment,
    )
