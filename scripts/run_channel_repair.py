"""Run the fail-fast structured Stage-A channel repair."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kfrag.diagnostics.channel_repair import run_channel_repair

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True, type=Path)
args = parser.parse_args()
with args.config.open(encoding="utf-8") as handle:
    result = run_channel_repair(yaml.safe_load(handle))
if result.get("first_failure"):
    raise SystemExit("FAIL: " + result["first_failure"])
print("PASS: fixed and learnable structured Stage-A channels meet all gates")
