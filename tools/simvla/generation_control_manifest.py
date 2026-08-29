#!/usr/bin/env python3
"""Create immutable EGL manifests for predeclared SimVLA control axes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FROZEN_GENERATION_SOURCE_SHA256,
    SEED_CONTRACTS,
    atomic_write_json,
    canonical_sha256,
    load_json,
)


def create(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing manifest: {output}")
    base = load_json(args.base_manifest)
    seed = SEED_CONTRACTS[args.inference_seed]
    payload = {key: value for key, value in base.items() if key != "manifest_sha256"}
    payload.update(
        {
            "source_combined_sha256": FROZEN_GENERATION_SOURCE_SHA256,
            "suite": args.suite,
            "evaluation_axis": args.evaluation_axis,
            "determinism_seed": seed["determinism_seed"],
            "action_noise_seed_base": seed["action_noise_seed_base"],
            "environment_seed": seed["environment_seed"],
            "inference_seed_replica": args.inference_seed,
            "training_seed_replica": "fixed_generation_step_030000",
            "same_trained_checkpoint_across_replicas": True,
            "renderer": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "PYTHONHASHSEED": str(seed["determinism_seed"]),
                "SIMVLA_RENDER_AXIS": args.evaluation_axis,
            },
        }
    )
    episodes = []
    for task_id in range(10):
        for trial_id in range(50):
            episodes.append(
                {
                    "suite": args.suite,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "init_state_index": trial_id,
                    "environment_seed": int(seed["environment_seed"]),
                    "physical_gpu_id": int(args.manifest_gpu_id),
                }
            )
    payload["episodes"] = episodes
    payload["episodes_per_row"] = 500
    payload["trials_per_task"] = 50
    payload["selected_physical_gpu_ids"] = [int(args.manifest_gpu_id)]
    payload["task_partition"] = {"rank0": list(range(10))}
    payload["task_iteration_order"] = {"rank0": list(reversed(range(10)))}
    payload["manifest_sha256"] = canonical_sha256(payload)
    atomic_write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--suite",
        choices=("libero_10", "libero_spatial", "libero_object", "libero_goal"),
        required=True,
    )
    parser.add_argument("--inference-seed", choices=tuple(SEED_CONTRACTS), required=True)
    parser.add_argument("--evaluation-axis", required=True)
    parser.add_argument("--manifest-gpu-id", type=int, default=0)
    args = parser.parse_args()
    payload = create(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
