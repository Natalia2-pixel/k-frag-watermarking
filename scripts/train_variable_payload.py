"""Train the variable-payload clean communication experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from kfrag.training import run_variable_payload  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    run_variable_payload(config)


if __name__ == "__main__":
    main()
