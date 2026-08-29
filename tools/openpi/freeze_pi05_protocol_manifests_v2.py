#!/usr/bin/env python3
"""Freeze paired final-eval and episode-disjoint teacher split manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from _common import require_run, require_source_lock_v2
from architectures.openpi.adapters.latentloop.cache_contract_v2 import (
    SPLIT_ROLES,
    SUITES,
    canonical_instruction,
    canonical_payload_hash,
    resolve_task_identity,
    tree_hash,
)


MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def _initial_state_hash(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()


def _image_uint8(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--noise-seed-base", type=int, default=7)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require_run(args.run, "OPENPI_LATENTLOOP_MANIFEST_RUN")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite protocol manifest root: {output}")
    lock = require_source_lock_v2(args.source_lock)

    from libero.libero import benchmark
    from openpi.training import config as config_api
    from openpi.training import data_loader

    train_config = config_api.get_config("pi05_libero_lora_pytorch")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = data_loader.create_torch_dataset(
        data_config, train_config.model.action_horizon, train_config.model
    )
    base_dataset = getattr(dataset, "_dataset", dataset)
    starts = [int(value) for value in base_dataset.episode_data_index["from"].tolist()]
    stops = [int(value) for value in base_dataset.episode_data_index["to"].tolist()]
    dataset_tasks: dict[str, set[int]] = {}
    for start in starts:
        sample = dataset[start]
        dataset_tasks.setdefault(canonical_instruction(str(sample["prompt"])), set()).add(
            int(sample["task_index"])
        )
    if any(len(indices) != 1 for indices in dataset_tasks.values()):
        raise RuntimeError("dataset instruction does not identify exactly one dataset task index")

    tasks: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite_name in SUITES:
        suite = benchmark_dict[suite_name]()
        if suite.n_tasks != 10:
            raise RuntimeError(f"{suite_name} no longer has ten benchmark tasks")
        for task_index in range(10):
            task = suite.get_task(task_index)
            instruction = canonical_instruction(str(task.language))
            dataset_indices = dataset_tasks.get(instruction)
            if dataset_indices is None or len(dataset_indices) != 1:
                raise RuntimeError(
                    f"benchmark instruction does not map to one dataset task: {suite_name}/{task_index}"
                )
            tasks.append(
                {
                    "suite": suite_name,
                    "benchmark_task_index": task_index,
                    "dataset_task_index": next(iter(dataset_indices)),
                    "canonical_task_name": f"{task.problem_folder}/{task.bddl_file}",
                    "canonical_instruction": instruction,
                }
            )
            initial_states = suite.get_task_init_states(task_index)
            if len(initial_states) < 50:
                raise RuntimeError(f"{suite_name}/task{task_index} has fewer than 50 initial states")
            for trial in range(50):
                episodes.append(
                    {
                        "suite": suite_name,
                        "benchmark_task_index": task_index,
                        "episode_namespace": "final_scientific_evaluation",
                        "trial": trial,
                        "environment_seed": args.environment_seed,
                        "initial_state_identifier": _initial_state_hash(initial_states[trial]),
                        "query_noise_key_prefix": (
                            f"{args.noise_seed_base}:{suite_name}:{task_index}:{trial}:"
                        ),
                        "max_episode_steps": MAX_STEPS[suite_name],
                    }
                )
    final_manifest = {
        "schema_version": 2,
        "frozen": True,
        "source_lock_id": lock["source_lock_id"],
        "protocol": {
            "action_horizon_h": 10,
            "execution_horizon_r": 5,
            "wait_steps": 10,
            "resize_size": 224,
            "renderer": "egl",
            "trials_per_task": 50,
            "policy_noise": "explicit_query_keyed_sha256_v2",
            "noise_seed_base": args.noise_seed_base,
        },
        "tasks": tasks,
        "episodes": episodes,
    }
    final_manifest["manifest_id"] = canonical_payload_hash(final_manifest, "manifest_id")

    task_manifest = {"tasks": tasks}
    demonstration_rows: list[dict[str, Any]] = []
    for episode_id, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        sample = dataset[start]
        task = resolve_task_identity(
            int(sample["task_index"]), str(sample["prompt"]), task_manifest
        )
        initial_identity = tree_hash(
            {
                "image": _image_uint8(sample["image"]),
                "wrist_image": _image_uint8(sample["wrist_image"]),
                "state": np.asarray(sample["state"]),
            }
        )
        demonstration_rows.append(
            {
                "suite": task["suite"],
                "benchmark_task_index": task["benchmark_task_index"],
                "episode_namespace": "teacher_demonstration",
                "episode_id": str(episode_id),
                "dataset_frame_start": start,
                "dataset_frame_stop": stop,
                "environment_seed": "not_recorded_in_source_lerobot_dataset",
                "initial_state_identifier": initial_identity,
            }
        )
    ordered = sorted(
        demonstration_rows,
        key=lambda row: hashlib.sha256(
            f"20260820:{row['suite']}:{row['benchmark_task_index']}:{row['episode_id']}".encode()
        ).hexdigest(),
    )
    count = len(ordered)
    boundaries = {
        "train": round(count * 0.60),
        "checkpoint_validation": round(count * 0.70),
        "defect_fit": round(count * 0.80),
        "defect_validity": round(count * 0.88),
    }
    assignments: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        if index < boundaries["train"]:
            role = "train"
        elif index < boundaries["checkpoint_validation"]:
            role = "checkpoint_validation"
        elif index < boundaries["defect_fit"]:
            role = "defect_fit"
        elif index < boundaries["defect_validity"]:
            role = "defect_validity"
        else:
            role = "scheduler_calibration"
        assignments.append({**row, "role": role})
    split_contract = {
        "schema_version": 2,
        "frozen": True,
        "source_lock_id": lock["source_lock_id"],
        "final_manifest_id": final_manifest["manifest_id"],
        "required_cache_roles": list(SPLIT_ROLES),
        "assignments": assignments,
    }
    split_contract["split_contract_id"] = canonical_payload_hash(
        split_contract, "split_contract_id"
    )

    output.mkdir(parents=True)
    final_path = output / "pi05_final_evaluation_manifest_v2.json"
    split_path = output / "pi05_split_contract_v2.json"
    final_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split_path.write_text(json.dumps(split_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"final_manifest": str(final_path), "split_contract": str(split_path)}, indent=2))


if __name__ == "__main__":
    main()
