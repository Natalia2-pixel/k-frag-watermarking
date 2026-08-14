"""Command-line entry point for the clean eight-image memorization experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from kfrag.training import run_tiny_overfit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    run_tiny_overfit(config)


if __name__ == "__main__":
    main()
