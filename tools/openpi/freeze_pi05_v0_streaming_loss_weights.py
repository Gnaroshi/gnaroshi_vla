#!/usr/bin/env python3
"""Freeze human-approved V0 weights for the cacheless streaming source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from source_lock_v2 import verify_lock

from architectures.openpi.adapters.latentloop.cache_contract_v2 import load_final_evaluation_manifest
from architectures.openpi.adapters.latentloop.cache_contract_v2 import load_split_contract
from architectures.openpi.adapters.latentloop.streaming_teacher import StreamingTeacherConfig
from architectures.openpi.adapters.latentloop.streaming_teacher import build_streaming_provenance


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "loss_weight_lock_id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-loss-calibration", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--final-evaluation-manifest", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-weight", type=float, required=True)
    parser.add_argument("--chunk-weight", type=float, required=True)
    parser.add_argument("--executed-weight", type=float, required=True)
    parser.add_argument("--gripper-weight", type=float, required=True)
    parser.add_argument("--noise-seed-base", type=int, default=20260820)
    parser.add_argument("--episode-order-seed", type=int, default=42)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    if not args.approve:
        raise RuntimeError("streaming loss weights require explicit human approval via --approve")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite streaming loss-weight lock: {output}")

    verified = verify_lock(args.source_lock)
    source_lock = json.loads(Path(args.source_lock).read_text(encoding="utf-8"))
    final_manifest = load_final_evaluation_manifest(args.final_evaluation_manifest)
    _, split_contract = load_split_contract(args.split_contract, final_manifest)
    provenance = build_streaming_provenance(
        source_lock=source_lock,
        checkpoint=args.checkpoint,
        final_manifest=final_manifest,
        final_manifest_path=args.final_evaluation_manifest,
        split_contract=split_contract,
        split_contract_path=args.split_contract,
        config=StreamingTeacherConfig(
            noise_seed_base=args.noise_seed_base,
            episode_order_seed=args.episode_order_seed,
        ),
    )
    raw_path = Path(args.raw_loss_calibration).resolve()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if (
        raw.get("V0_RAW_LOSS_CALIBRATION_COMPLETE") is not True
        or raw.get("variant") != "v0"
        or raw.get("source_lock_id") != verified["source_lock_id"]
        or raw.get("training_source_id") != provenance["training_source_id"]
    ):
        raise RuntimeError("raw-loss calibration is absent, stale, or for another streaming source")
    weights = {
        "state": args.state_weight,
        "chunk": args.chunk_weight,
        "executed": args.executed_weight,
        "gripper": args.gripper_weight,
        "composition": 0.0,
    }
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("all streaming V0 loss weights must be finite and nonnegative")
    if sum(weights.values()) <= 0:
        raise ValueError("at least one streaming V0 loss weight must be positive")
    payload = {
        "schema_version": 1,
        "frozen": True,
        "V0_STREAMING_LOSS_WEIGHTS_APPROVED": True,
        "markers": ["V0_STREAMING_LOSS_WEIGHTS_APPROVED"],
        "source_lock_id": verified["source_lock_id"],
        "training_source_mode": provenance["training_source_mode"],
        "training_source_id": provenance["training_source_id"],
        "action_execution_mode": raw.get("action_execution_mode", "A"),
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
