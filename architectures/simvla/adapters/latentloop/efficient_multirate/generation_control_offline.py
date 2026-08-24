"""Bounded same-query action-tail audit for the faithful naive NFE=3 control."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from architectures.simvla.adapters.latentloop.action_adapter import (
    ActionNoiseKey,
    explicit_action_noise,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    ExactTeacherStore,
    _drop_unused_vlm,
    validate_exact_cache,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FROZEN_GENERATION_SOURCE_SHA256,
    FROZEN_TEACHER_CACHE_SOURCE_SHA256,
    atomic_write_json,
    load_json,
    native_nfe_time_grid,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    load_frozen_simvla,
)


def _quantiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _noise(query: dict[str, Any], device: torch.device) -> torch.Tensor:
    fields = query["metadata"]["noise_key"]
    key = ActionNoiseKey(
        checkpoint=str(fields["checkpoint"]),
        task_id=int(fields["task_id"]),
        episode_id=str(fields["episode_id"]),
        policy_query_index=int(fields["policy_query_index"]),
        seed_base=int(fields["seed_base"]),
    )
    if key.seed() != int(query["noise_seed"]):
        raise RuntimeError("exact-cache action-noise key changed")
    return explicit_action_noise(
        key,
        batch_size=1,
        action_horizon=10,
        action_dim=7,
        device=device,
        dtype=torch.float32,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        raise RuntimeError("one explicit CUDA_VISIBLE_DEVICES value is required")
    if "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("offline naive audit uses exactly one GPU")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing offline output: {output}")
    source_lock = load_json(args.source_lock)
    if source_lock.get("combined_sha256") != FROZEN_GENERATION_SOURCE_SHA256:
        raise RuntimeError("source lock differs from frozen Generation checkpoint")
    validation = validate_exact_cache(args.cache, verify_checksums=False)
    if validation.get("verdict") != "EXACT_TEACHER_CACHE_VALID":
        raise RuntimeError(f"exact teacher cache invalid: {validation}")
    store = ExactTeacherStore(args.cache)
    if store.manifest.get("source_combined_sha256") != FROZEN_TEACHER_CACHE_SOURCE_SHA256:
        raise RuntimeError("exact teacher cache source identity changed")

    configure_strict_torch_determinism(int(args.seed))
    device = torch.device("cuda:0")
    model, _, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    _drop_unused_vlm(model)
    query_ids = [str(item["query_id"]) for item in store.manifest["query_index"]]
    selected = query_ids[: min(int(args.queries), len(query_ids))]
    first5: list[float] = []
    full_chunk: list[float] = []
    translation: list[float] = []
    rotation: list[float] = []
    gripper: list[float] = []
    latency: list[float] = []
    rows: list[dict[str, Any]] = []
    for query_id in selected:
        query = store.query(query_id)
        condition = query["condition"].unsqueeze(0).to(device)
        proprio = query["proprio"].unsqueeze(0).to(device)
        reference = query["teacher_action"].unsqueeze(0).to(device)
        noise = _noise(query, device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        decoded = action_adapter.decode_action_from_condition(
            condition,
            proprio,
            steps=3,
            initial_noise=noise,
            requires_grad=False,
            return_debug=True,
        )
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if int(decoded.debug["iterations"]) != 3:
            raise RuntimeError("source-native NFE=3 did not execute exactly three updates")
        difference = (decoded.action.float() - reference.float()).abs()
        metrics = {
            "first5_action_l1": float(difference[:, :5].mean().item()),
            "full_chunk_action_l1": float(difference.mean().item()),
            "translation_l1": float(difference[:, :5, :3].mean().item()),
            "rotation_l1": float(difference[:, :5, 3:6].mean().item()),
            "continuous_gripper_l1": float(difference[:, :5, 6:].mean().item()),
            "latency_ms": elapsed_ms,
        }
        first5.append(metrics["first5_action_l1"])
        full_chunk.append(metrics["full_chunk_action_l1"])
        translation.append(metrics["translation_l1"])
        rotation.append(metrics["rotation_l1"])
        gripper.append(metrics["continuous_gripper_l1"])
        latency.append(metrics["latency_ms"])
        rows.append({"query_id": query_id, "nfe": 3, **metrics})

    output.mkdir(parents=True)
    with (output / "naive_nfe3_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "verdict": "NAIVE_NFE3_OFFLINE_TAIL_PASS",
        "source_combined_sha256": FROZEN_GENERATION_SOURCE_SHA256,
        "queries": len(rows),
        "full_conditions_only": True,
        "same_query_same_initial_noise": True,
        "no_generation_updater_loaded": True,
        "time_grid": list(native_nfe_time_grid(3)),
        "summaries": {
            "3": {
                "first5_action_l1": _quantiles(first5),
                "full_chunk_action_l1": _quantiles(full_chunk),
                "translation_l1": _quantiles(translation),
                "rotation_l1": _quantiles(rotation),
                "continuous_gripper_l1": _quantiles(gripper),
                "latency_ms": _quantiles(latency),
            }
        },
    }
    atomic_write_json(output / "naive_nfe_audit.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
