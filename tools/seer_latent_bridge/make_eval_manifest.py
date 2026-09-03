#!/usr/bin/env python3
"""Create the exact task/trial/init-state manifest shared by every eval row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from architectures.seer.adapters.latent_bridge.provenance import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--renderer", choices=("osmesa", "egl"), default="osmesa")
    parser.add_argument("--reuse-if-matching", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["PYOPENGL_PLATFORM"] = args.renderer
    os.environ["MUJOCO_GL"] = args.renderer
    sys.path.insert(0, args.libero_path)
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    episodes = []
    for task_id in range(args.num_tasks):
        task = suite.get_task(task_id)
        bddl = Path(args.libero_path) / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
        init_path = Path(args.libero_path) / "libero/libero/init_files" / task.problem_folder / task.init_states_file
        states = torch.load(init_path, map_location="cpu")
        if len(states) < args.episodes_per_task:
            raise RuntimeError(f"task {task_id} has only {len(states)} initial states")
        for trial_id in range(args.episodes_per_task):
            state = np.ascontiguousarray(np.asarray(states[trial_id]))
            episodes.append(
                {
                    "global_id": task_id * args.episodes_per_task + trial_id,
                    "task_id": task_id,
                    "task_name": task.name,
                    "language": task.language,
                    "trial_id": trial_id,
                    "bddl_file": str(bddl),
                    "bddl_sha256": sha256_file(bddl),
                    "init_states_file": str(init_path),
                    "init_states_file_sha256": sha256_file(init_path),
                    "init_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                }
            )
    payload = {
        "schema_version": 1,
        "suite": "libero_10",
        "renderer": args.renderer,
        "seed": args.seed,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "num_tasks": args.num_tasks,
        "episodes_per_task": args.episodes_per_task,
        "num_episodes": len(episodes),
        "policy_contract": {
            "sequence_length": 7,
            "action_pred_steps": 3,
            "temporal_ensembling": True,
            "ensembling_temperature": 0.01,
            "control_frequency_hz": 20,
            "max_policy_steps": 600,
            "observation_preprocessing": "Seer upstream CLIP/MAE evaluation path",
        },
        "episodes": episodes,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    if output.exists():
        if not args.reuse_if_matching:
            raise FileExistsError(output)
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(
                f"existing manifest differs from the requested evaluation contract: {output}"
            )
        print(
            json.dumps(
                {"status": "REUSED", "episodes": len(episodes), "output": str(output)},
                indent=2,
            )
        )
        return
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "episodes": len(episodes), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
