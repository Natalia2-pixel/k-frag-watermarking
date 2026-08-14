"""CLI for the clean-model anti-memorization evaluation."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kfrag.evaluation import evaluate_checkpoint  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    if not isinstance(config, dict): raise ValueError("configuration must be a YAML mapping")
    results = evaluate_checkpoint(config, args.checkpoint)
    for condition, metrics in results.items(): print(condition + " " + " ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))

if __name__ == "__main__": main()
