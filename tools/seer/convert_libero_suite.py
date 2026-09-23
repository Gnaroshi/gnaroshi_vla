#!/usr/bin/env python3
"""Convert one raw LIBERO suite to Seer's per-step on-disk format."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


SUPPORTED_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


def _demo_index(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


def _task_language(path: Path) -> str:
    suffix = "_demo.hdf5"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected LIBERO demonstration filename: {path.name}")
    task = path.name[: -len(suffix)]
    if "_SCENE" in task:
        match = re.match(r"^.+_SCENE[0-9]+_(.+)$", task)
        if match is None:
            raise ValueError(f"Cannot extract scene task language from {path.name}")
        task = match.group(1)
    return task.lower().replace("_", " ")


def _inspect_task(path: Path, episode_offset: int, debug_episode_limit: int) -> dict:
    with h5py.File(path, "r") as source:
        if "data" not in source:
            raise KeyError(f"Missing data group: {path}")
        demo_names = sorted(source["data"].keys(), key=_demo_index)
        if debug_episode_limit > 0:
            demo_names = demo_names[:debug_episode_limit]
        lengths = []
        required_obs = {
            "agentview_rgb",
            "eye_in_hand_rgb",
            "joint_states",
            "ee_states",
            "gripper_states",
        }
        for name in demo_names:
            demo = source["data"][name]
            missing = sorted(required_obs.difference(demo["obs"].keys()))
            if missing or "actions" not in demo:
                raise KeyError(f"Invalid demo {path}:{name}; missing={missing}")
            lengths.append(int(demo["actions"].shape[0]))
    return {
        "source_path": str(path),
        "language": _task_language(path),
        "demo_names": demo_names,
        "episode_offset": int(episode_offset),
        "lengths": lengths,
    }


def _convert_task(job: dict, staging_dir: str) -> list[list[object]]:
    episodes_dir = Path(staging_dir) / "episodes"
    rows = []
    with h5py.File(job["source_path"], "r") as source:
        for local_index, demo_name in enumerate(job["demo_names"]):
            episode_index = int(job["episode_offset"]) + local_index
            demo = source["data"][demo_name]
            obs = demo["obs"]
            primary = np.asarray(obs["agentview_rgb"])
            wrist = np.asarray(obs["eye_in_hand_rgb"])
            actions = np.asarray(demo["actions"])
            joint_states = np.asarray(obs["joint_states"])
            ee_states = np.asarray(obs["ee_states"])
            gripper_positions = np.asarray(obs["gripper_states"])
            length = int(actions.shape[0])

            arrays = {
                "agentview_rgb": primary,
                "eye_in_hand_rgb": wrist,
                "joint_states": joint_states,
                "ee_states": ee_states,
                "gripper_states": gripper_positions,
            }
            bad_lengths = {name: int(value.shape[0]) for name, value in arrays.items() if value.shape[0] != length}
            if bad_lengths:
                raise ValueError(
                    f"Observation/action length mismatch in {job['source_path']}:{demo_name}: "
                    f"actions={length}, observations={bad_lengths}"
                )

            gripper_state = np.empty(length, dtype=actions.dtype)
            gripper_state[0] = actions[0, -1]
            gripper_state[1:] = actions[:-1, -1]

            episode_name = f"{episode_index:06d}"
            episode_dir = episodes_dir / episode_name
            steps_dir = episode_dir / "steps"
            steps_dir.mkdir(parents=True, exist_ok=False)
            with h5py.File(episode_dir / "meta_info.h5", "w") as output:
                output.create_dataset("length", data=length)

            for step_index in range(length):
                step_dir = steps_dir / f"{step_index:04d}"
                step_dir.mkdir()
                Image.fromarray(primary[step_index]).save(step_dir / "image_primary.jpg")
                Image.fromarray(wrist[step_index]).save(step_dir / "image_wrist.jpg")
                with h5py.File(step_dir / "other.h5", "w") as output:
                    output.create_dataset(
                        "language_instruction",
                        data=np.asarray(
                            job["language"],
                            dtype=h5py.string_dtype(encoding="utf-8"),
                        ),
                    )
                    output.create_dataset("episode_length", data=length)
                    output.create_dataset("action", data=actions[step_index])
                    observation = output.create_group("observation")
                    observation.create_dataset("proprio", data=joint_states[step_index])
                    observation.create_dataset("tcp_pose", data=ee_states[step_index])
                    observation.create_dataset("gripper_state", data=gripper_state[step_index])
                    observation.create_dataset(
                        "gripper_position", data=gripper_positions[step_index]
                    )
            rows.append([episode_name, length])
    return rows


def _load_complete_manifest(target_dir: Path, suite: str) -> dict | None:
    manifest_path = target_dir / "conversion_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("suite") != suite:
        return None
    if not (target_dir / "meta_info.h5").is_file() or not (target_dir / "data_info.json").is_file():
        return None
    return manifest


def convert(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_root).resolve() / args.suite
    output_root = Path(args.output_root).resolve()
    dataset_name = f"{args.suite}_converted"
    target_dir = output_root / dataset_name

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing raw LIBERO suite: {source_dir}")
    if target_dir.exists():
        manifest = _load_complete_manifest(target_dir, args.suite)
        if manifest is None:
            raise FileExistsError(
                f"Refusing to overwrite incomplete or foreign conversion: {target_dir}"
            )
        print(
            f"[CONVERT][SKIP] {args.suite} already complete: "
            f"episodes={manifest['num_episodes']} steps={manifest['num_steps']}"
        )
        return

    task_paths = sorted(source_dir.glob("*_demo.hdf5"))
    if args.debug_max_tasks > 0:
        task_paths = task_paths[: args.debug_max_tasks]
    if not task_paths:
        raise FileNotFoundError(f"No demonstration HDF5 files under {source_dir}")

    jobs = []
    episode_offset = 0
    for path in task_paths:
        job = _inspect_task(path, episode_offset, args.debug_max_episodes_per_task)
        jobs.append(job)
        episode_offset += len(job["demo_names"])

    output_root.mkdir(parents=True, exist_ok=True)
    staging_dir = output_root / f".{dataset_name}.tmp-{os.getpid()}"
    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")
    (staging_dir / "episodes").mkdir(parents=True)

    started = time.time()
    rows = []
    completed_tasks = 0
    workers = min(max(1, args.workers), len(jobs))
    print(
        f"[CONVERT] suite={args.suite} tasks={len(jobs)} episodes={episode_offset} "
        f"workers={workers} target={target_dir}",
        flush=True,
    )
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_convert_task, job, str(staging_dir)) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                converted_rows = future.result()
                rows.extend(converted_rows)
                completed_tasks += 1
                elapsed = time.time() - started
                rate = completed_tasks / elapsed if elapsed > 0.0 else 0.0
                eta = (len(jobs) - completed_tasks) / rate if rate > 0.0 else 0.0
                print(
                    f"[CONVERT PROGRESS] tasks={completed_tasks}/{len(jobs)} "
                    f"episodes={len(rows)}/{episode_offset} elapsed={elapsed / 60.0:.1f}m "
                    f"eta={eta / 60.0:.1f}m",
                    flush=True,
                )

        rows.sort(key=lambda item: item[0])
        if len(rows) != episode_offset:
            raise RuntimeError(f"Converted {len(rows)} episodes, expected {episode_offset}")
        total_steps = sum(int(item[1]) for item in rows)
        with h5py.File(staging_dir / "meta_info.h5", "w") as output:
            output.create_dataset("num_episodes", data=len(rows))
        (staging_dir / "data_info.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "suite": args.suite,
            "dataset_name": dataset_name,
            "source_dir": str(source_dir),
            "source_files": [str(path) for path in task_paths],
            "num_tasks": len(task_paths),
            "num_episodes": len(rows),
            "num_steps": total_steps,
            "workers": workers,
            "debug_max_tasks": args.debug_max_tasks,
            "debug_max_episodes_per_task": args.debug_max_episodes_per_task,
            "elapsed_s": time.time() - started,
        }
        (staging_dir / "conversion_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging_dir.rename(target_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(
        f"[CONVERT][DONE] suite={args.suite} episodes={len(rows)} steps={total_steps} "
        f"elapsed={(time.time() - started) / 60.0:.1f}m target={target_dir}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, choices=SUPPORTED_SUITES)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--debug-max-tasks", type=int, default=0)
    parser.add_argument("--debug-max-episodes-per-task", type=int, default=0)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.debug_max_tasks < 0 or args.debug_max_episodes_per_task < 0:
        parser.error("debug limits cannot be negative")
    return args


if __name__ == "__main__":
    convert(parse_args())
