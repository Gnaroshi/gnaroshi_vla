#!/usr/bin/env python3
"""Freeze explicitly user-approved V0 loss weights after raw-scale inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from source_lock_v2 import verify_lock


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "loss_weight_lock_id"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-loss-calibration", required=True)
    parser.add_argument("--cache-gate", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-weight", type=float, required=True)
    parser.add_argument("--chunk-weight", type=float, required=True)
    parser.add_argument("--executed-weight", type=float, required=True)
    parser.add_argument("--gripper-weight", type=float, required=True)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    if not args.approve:
        raise RuntimeError("loss weights require explicit human approval via --approve")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite loss-weight lock: {output}")
    lock = verify_lock(args.source_lock)
    raw_path = Path(args.raw_loss_calibration).resolve()
    cache_gate_path = Path(args.cache_gate).resolve()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    cache_gate = json.loads(cache_gate_path.read_text(encoding="utf-8"))
    if (
        raw.get("V0_RAW_LOSS_CALIBRATION_COMPLETE") is not True
        or raw.get("variant") != "v0"
        or raw.get("source_lock_id") != lock["source_lock_id"]
    ):
        raise RuntimeError("raw-loss calibration is absent, stale, or not V0")
    if (
        cache_gate.get("FULL_CACHE_SCHEMA_V2_PASS") is not True
        or cache_gate.get("source_lock_id") != lock["source_lock_id"]
        or cache_gate.get("cache_manifest_sha256") != raw.get("cache_manifest_sha256")
    ):
        raise RuntimeError("raw-loss calibration and independently accepted full cache do not match")
    weights = {
        "state": args.state_weight,
        "chunk": args.chunk_weight,
        "executed": args.executed_weight,
        "gripper": args.gripper_weight,
        "composition": 0.0,
    }
    if any(not 0.0 <= value < float("inf") for value in weights.values()):
        raise ValueError("all V0 loss weights must be finite and nonnegative")
    if sum(weights.values()) <= 0:
        raise ValueError("at least one V0 loss weight must be positive")
    payload = {
        "schema_version": 2,
        "frozen": True,
        "V0_LOSS_WEIGHTS_APPROVED": True,
        "markers": ["V0_LOSS_WEIGHTS_APPROVED"],
        "source_lock_id": lock["source_lock_id"],
        "cache_manifest_sha256": raw["cache_manifest_sha256"],
        "cache_gate": str(cache_gate_path),
        "cache_gate_sha256": _sha256(cache_gate_path),
        "raw_loss_calibration": str(raw_path),
        "raw_loss_calibration_sha256": _sha256(raw_path),
        "raw_loss_calibration_id": raw["raw_loss_calibration_id"],
        "weights": weights,
        "approval": "explicit_user_cli_--approve",
    }
    payload["loss_weight_lock_id"] = _identity(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
