#!/usr/bin/env python3
"""Validate reusable sync data and host-memory capacity for efficient training."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import h5py

from architectures.seer.adapters.latent_bridge.dataset import BridgeTransitionDataset
from architectures.seer.adapters.latent_bridge.provenance import sha256_file
from methods.latent_bridge import ComputeMatchedTrainingContract


TENSOR_FIELDS = (
    "previous_condition",
    "target_condition",
    "stable_context",
    "current_state",
    "previous_executed_action",
)


def _available_memory_bytes(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("MemAvailable:"):
                continue
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                break
            available = int(fields[1]) * 1024
            if available > 0:
                return available
            break
    except (OSError, ValueError):
        pass

    # MemAvailable includes reclaimable caches. Keep sysconf as a fallback for
    # platforms without Linux's /proc/meminfo interface.
    page_size = os.sysconf("SC_PAGE_SIZE")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    return int(page_size * available_pages)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-stage", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--per-rank-batch", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--r0-optimizer-steps", type=int, default=50_200)
    parser.add_argument("--r1-optimizer-steps", type=int, default=44_000)
    args = parser.parse_args()

    sync_stage = Path(args.sync_stage).resolve()
    result_root = Path(args.result_root).resolve()
    if sync_stage == result_root or sync_stage in result_root.parents:
        raise RuntimeError("efficient result root must not overlap the read-only sync source")
    if not (sync_stage / "COMPLETE").is_file():
        raise RuntimeError(f"sync source is incomplete: {sync_stage}")
    decision_path = sync_stage / "stable_context.txt"
    decision = decision_path.read_text(encoding="utf-8").splitlines()
    if len(decision) != 3 or not decision[0] or not decision[1] or int(decision[2]) <= 0:
        raise RuntimeError(f"invalid measured stable-context decision: {decision_path}")

    shard_root = sync_stage / "sync_300" / "shards"
    shards = sorted(shard_root.glob("sync_transitions_rank*.h5"))
    if len(shards) != args.world_size:
        raise RuntimeError(f"expected {args.world_size} sync shards, found {len(shards)}")

    transitions = 0
    tensor_bytes_per_copy = 0
    shard_records = []
    for shard in shards:
        manifest_path = shard.with_suffix(shard.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_hash = sha256_file(shard)
        if actual_hash != manifest.get("dataset_sha256"):
            raise RuntimeError(f"sync shard hash mismatch: {shard}")
        with h5py.File(shard, "r") as handle:
            transitions += int(handle["previous_condition"].shape[0])
            tensor_bytes_per_copy += sum(
                int(handle[key].size * handle[key].dtype.itemsize) for key in TENSOR_FIELDS
            )
        shard_records.append(
            {
                "path": str(shard),
                "sha256": actual_hash,
                "transitions": int(manifest["num_transitions"]),
            }
        )

    contract_r0 = ComputeMatchedTrainingContract(
        stage="R0",
        optimizer_steps=args.r0_optimizer_steps,
        learning_rate=3e-4,
        per_rank_batch=args.per_rank_batch,
        world_size=args.world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    contract_r1 = ComputeMatchedTrainingContract(
        stage="R1",
        optimizer_steps=args.r1_optimizer_steps,
        learning_rate=3e-5,
        per_rank_batch=args.per_rank_batch,
        world_size=args.world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    contract_r0.validate()
    contract_r1.validate()

    train_samples = 0
    validation_samples = 0
    for shard in shards:
        train_samples += len(BridgeTransitionDataset(shard, split="train"))
        validation_samples += len(BridgeTransitionDataset(shard, split="validation"))
    rank_samples = math.ceil(train_samples / args.world_size)
    microbatches = rank_samples // args.per_rank_batch
    usable_microbatches = (
        microbatches // args.gradient_accumulation_steps
    ) * args.gradient_accumulation_steps
    updates_per_data_epoch = usable_microbatches // args.gradient_accumulation_steps

    available = _available_memory_bytes()
    preload_all_ranks = tensor_bytes_per_copy * args.world_size
    # R1 holds sync and DAgger tensors. Reserve a second sync-sized corpus plus
    # 8 GiB for models, filesystem cache, and the parent shell.
    required_available = int(preload_all_ranks * 2.0 + 8 * 1024**3)
    if available < required_available:
        raise RuntimeError(
            "insufficient host memory for fail-closed in-memory R1 training: "
            f"available={available}, required={required_available}"
        )

    payload = {
        "status": "SEER_LATENT_BRIDGE_EFFICIENT_PREFLIGHT_PASS",
        "sync_source": str(sync_stage),
        "sync_source_mode": "read_only_reuse",
        "stable_context": {
            "layer": decision[0],
            "token_group": decision[1],
            "sequence_length": int(decision[2]),
        },
        "sync_transitions": transitions,
        "training_samples": train_samples,
        "validation_samples": validation_samples,
        "shards": shard_records,
        "batch_contract": {
            "per_rank_batch": args.per_rank_batch,
            "world_size": args.world_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch": contract_r0.effective_batch,
        },
        "optimizer_step_budget": {
            "R0": contract_r0.optimizer_steps,
            "R1": contract_r1.optimizer_steps,
        },
        "updates_per_sync_data_epoch": updates_per_data_epoch,
        "estimated_r0_data_epochs": math.ceil(
            contract_r0.optimizer_steps / updates_per_data_epoch
        ),
        "host_memory": {
            "available_bytes": available,
            "sync_preload_bytes_all_ranks": preload_all_ranks,
            "fail_closed_required_bytes": required_available,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
