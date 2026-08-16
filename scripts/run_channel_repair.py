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
fresh = result["learnable_carrier"]["evaluations"]["fresh_on_the_fly_payloads"]
display = lambda value: "N/A" if value is None else value
print(f"exact active 8-bit regional-symbol accuracy: {fresh['exact_active_region_accuracy']:.6f}")
print(f"authentication tag accuracy: {display(fresh['authentication_tag_accuracy'])}")
print(f"exact 44-bit regional-packet accuracy: {display(fresh['exact_regional_packet_accuracy'])}")
print(f"exact image-payload accuracy: {display(fresh['exact_image_payload_accuracy'])}")
