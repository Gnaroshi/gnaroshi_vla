#!/usr/bin/env python3
"""Freeze V1 weights from train-split raw-loss medians only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


TARGET_CONTRIBUTIONS = {
    "direct_latent": 0.25,
    "composed_latent": 0.25,
    "direct_action": 0.20,
    "composed_action": 0.20,
    "composition": 0.09,
    "smooth": 0.01,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-losses", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = json.loads(args.raw_losses.read_text(encoding="utf-8"))
    if data.get("split_role") != "train_raw_loss_calibration" or data.get("uses_online_sr", True):
        raise RuntimeError("raw-loss calibration must use the training calibration split and no online SR")
    split_sha256 = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    if data.get("split_manifest_sha256") != split_sha256:
        raise RuntimeError("raw-loss calibration split-manifest provenance mismatch")
    medians = data["raw_loss_medians"]
    if set(medians) != set(TARGET_CONTRIBUTIONS):
        raise RuntimeError("raw-loss metric set is incomplete")
    if any(not math.isfinite(float(value)) or float(value) <= 0 for value in medians.values()):
        raise RuntimeError("raw-loss medians must be finite and positive")
    unscaled = {name: TARGET_CONTRIBUTIONS[name] / float(medians[name]) for name in medians}
    scale = 0.5 / unscaled["direct_latent"]
    weights = {name: value * scale for name, value in unscaled.items()}
    payload = {
        "schema_version": 1,
        "status": "V1_RAW_LOSS_WEIGHTS_LOCKED",
        "uses_online_sr": False,
        "split_manifest_sha256": split_sha256,
        "raw_loss_medians": medians,
        "target_contributions": TARGET_CONTRIBUTIONS,
        "weights": weights,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["status"])


if __name__ == "__main__":
    main()
