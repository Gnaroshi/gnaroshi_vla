#!/usr/bin/env python3
"""Bounded real-batch kernel timing for the repaired operation vocabulary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from _common import DEFAULT_CHECKPOINT, load_local_policy, require_run
from audit_pi05_k1_equivalence import _real_dataset_observation
from architectures.openpi.adapters.latentloop.policy_io import explicit_policy_noise
from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixKVHook
from architectures.openpi.adapters.latentloop.transition_core import OpenPIKVLatentLoop
from methods.variable_time_latentloop.operation_counters_v2 import (
    full_hook_query,
    latent_query,
    native_full_query,
)
from pi05_stage_gate_v2 import verify_stage


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(operation: Callable[[], object], repeats: int, device: torch.device) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        _sync(device)
        started = time.perf_counter()
        operation()
        _sync(device)
        values.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(values, dtype=np.float64)
    return {
        "repeats": int(repeats),
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.quantile(array, 0.50)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "minimum_ms": float(array.min()),
        "maximum_ms": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--k1-gate", required=True)
    parser.add_argument("--freeze-gate", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require_run(args.run, "OPENPI_LATENTLOOP_MICROBENCH_RUN")
    if not 1 <= args.repeats <= 20:
        raise ValueError("bounded microbenchmark repeats must be in [1,20]")
    output = Path(args.output).resolve()
    gate = verify_stage(
        "stage2_cache_smoke",
        args.source_lock,
        [args.k1_gate, args.freeze_gate],
        output_candidate=output,
    )
    lock_payload = json.loads(Path(args.source_lock).read_text(encoding="utf-8"))
    if Path(lock_payload["checkpoint"]["directory"]).resolve() != Path(args.checkpoint).resolve():
        raise RuntimeError("checkpoint mismatch: microbenchmark checkpoint differs from source lock")

    device = torch.device(args.device)
    policy = load_local_policy(args.checkpoint, args.device, flow_steps=args.flow_steps)
    model = policy._model  # noqa: SLF001
    observation = _real_dataset_observation(policy, args.dataset_index)
    noise = explicit_policy_noise(
        (1, model.config.action_horizon, model.config.action_dim),
        seed=20260820,
        device=device,
    )
    hook = PrefixKVHook(model)
    adapter = OpenPIKVLatentLoop().to(device).eval()
    sampler = getattr(model.sample_actions, "_torchdynamo_orig_callable", model.sample_actions)

    with torch.no_grad():
        extraction = hook.extract(observation)
        current_prefix, robot_state, _ = hook.embed(observation)
        executed = torch.zeros((1, 5, 7), device=device)

        def native_full() -> None:
            sampler(device, observation, noise=noise, num_steps=args.flow_steps)

        def full_hook() -> None:
            value = hook.extract(observation)
            hook.sample_actions_from_state(
                value.state, value.robot_state, noise, num_steps=args.flow_steps
            )

        def prefix_embedding() -> None:
            hook.embed(observation)

        def prefix_transformer() -> None:
            hook.extract_from_embedding(current_prefix, robot_state)

        def sequential_transition() -> None:
            adapter(
                extraction.state,
                current_prefix,
                extraction.state.embeddings,
                executed,
                robot_state,
                delta_q=1,
                delta_a=5,
                full_refresh_age=1,
                executed_action_lengths=torch.full((1,), 5, device=device, dtype=torch.long),
            )

        def direct_transition() -> None:
            adapter(
                extraction.state,
                current_prefix,
                extraction.state.embeddings,
                executed,
                robot_state,
                delta_q=1,
                delta_a=5,
                full_refresh_age=1,
                executed_action_lengths=torch.full((1,), 5, device=device, dtype=torch.long),
                intermediate_prefix_embeddings=current_prefix.embeddings[:, None],
                robot_state_history=robot_state[:, None],
            )

        def action_expert() -> None:
            hook.sample_actions_from_state(
                extraction.state, robot_state, noise, num_steps=args.flow_steps
            )

        operations = {
            "native_full_query": native_full,
            "full_hook_query": full_hook,
            "prefix_embedding": prefix_embedding,
            "prefix_transformer": prefix_transformer,
            "latentloop_sequential": sequential_transition,
            "latentloop_direct": direct_transition,
            "action_expert_with_cache_rebuild": action_expert,
        }
        for operation in operations.values():
            operation()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        timings = {
            name: _measure(operation, args.repeats, device)
            for name, operation in operations.items()
        }

    payload = {
        "schema_version": 2,
        "OPERATION_COST_MICROBENCHMARK_PASS": True,
        "markers": ["OPERATION_COST_MICROBENCHMARK_PASS"],
        "scientific_use": "bounded kernel diagnostic; not an online episode latency result",
        "adapter_state": "randomly initialized architecture-only timing",
        "source_lock_id": gate["source_lock_id"],
        "checkpoint_sha256": lock_payload["checkpoint"]["model_sha256"],
        "config_sha256": lock_payload["checkpoint"]["config_sha256"],
        "norm_stats_sha256": lock_payload["normalization"]["sha256"],
        "dataset_index": args.dataset_index,
        "flow_steps": args.flow_steps,
        "repeats": args.repeats,
        "timings": timings,
        "operation_templates": {
            "native_full_query": native_full_query(args.flow_steps).to_dict(),
            "full_hook_query": full_hook_query(args.flow_steps).to_dict(),
            "latent_level0": latent_query(args.flow_steps, direct=True).to_dict(),
        },
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "adapter_trainable_parameters": adapter.trainable_parameters,
    }
    output.mkdir(parents=True)
    (output / "operation_cost_microbenchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output / "operation_cost_microbenchmark.json")


if __name__ == "__main__":
    main()
