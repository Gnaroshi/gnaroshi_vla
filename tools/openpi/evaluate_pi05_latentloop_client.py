#!/usr/bin/env python3
"""Instrumented LIBERO client preserving the local OpenPI evaluation semantics."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import time

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import mujoco
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import tqdm

from architectures.openpi.adapters.latentloop.cache_contract_v2 import load_final_evaluation_manifest


DUMMY_ACTION = [0.0] * 6 + [-1.0]
ENV_RESOLUTION = 256
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def quat2axisangle(quat):
    quat = np.asarray(quat).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def make_env(task, seed):
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=ENV_RESOLUTION,
        camera_widths=ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def observation_request(obs, prompt, resize_size):
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": str(prompt),
    }, image


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def episode_action_metrics(actions):
    values = np.asarray(actions, dtype=np.float64)
    if len(values) < 1:
        return {
            "translation_second_difference_mean": None,
            "rotation_second_difference_mean": None,
            "gripper_switches": 0,
            "gripper_short_reversals": 0,
        }
    second = values[2:] - 2.0 * values[1:-1] + values[:-2]
    gripper = np.where(values[:, 6] >= 0, 1, -1)
    switches = int(np.sum(gripper[1:] != gripper[:-1]))
    reversals = int(np.sum((gripper[2:] == gripper[:-2]) & (gripper[1:-1] != gripper[:-2])))
    return {
        "translation_second_difference_mean": (
            float(np.linalg.norm(second[:, :3], axis=1).mean()) if len(second) else 0.0
        ),
        "rotation_second_difference_mean": (
            float(np.linalg.norm(second[:, 3:6], axis=1).mean()) if len(second) else 0.0
        ),
        "gripper_switches": switches,
        "gripper_short_reversals": reversals,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", choices=tuple(MAX_STEPS), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-seed-base", type=int, default=7)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument(
        "--policy-path",
        choices=("original", "k1", "hold", "latent_bridge", "v0", "v1", "v2"),
        default="original",
    )
    parser.add_argument("--paired-k1-smoke", action="store_true")
    parser.add_argument("--final-evaluation-manifest")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args()
    if args.replan_steps != 5 or args.resize_size != 224 or args.wait_steps != 10:
        raise ValueError("pinned OpenPI protocol requires R=5, resize=224, wait=10")
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.paired_k1_smoke:
        if args.max_tasks != 1 or args.num_trials != 2:
            raise ValueError("K1 parity smoke is exactly one task x two episodes")
        selected_manifest = None
    else:
        if not args.final_evaluation_manifest:
            raise ValueError("scientific evaluation requires --final-evaluation-manifest")
        if args.max_tasks != 10 or args.num_trials != 50:
            raise ValueError("scientific suite shard is exactly 10 tasks x 50 trials")
        final_manifest_path = Path(args.final_evaluation_manifest).resolve()
        final_manifest = load_final_evaluation_manifest(final_manifest_path)
        manifest_hash = hashlib.sha256(final_manifest_path.read_bytes()).hexdigest()
        if server_metadata.get("final_evaluation_manifest_sha256") != manifest_hash:
            raise RuntimeError("server/client final evaluation manifest hashes differ")
        selected_manifest = [row for row in final_manifest["episodes"] if row["suite"] == args.suite]
        if len(selected_manifest) != 500:
            raise ValueError("scientific suite manifest must contain exactly 500 episodes")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite output: {}".format(output))
    output.mkdir(parents=True)
    if args.save_video:
        (output / "videos").mkdir()
    np.random.seed(args.seed)
    outcomes = []
    query_rows = []
    manifest = []
    paths = ("original", "k1") if args.paired_k1_smoke else (args.policy_path,)
    run_started = time.time()

    manifest_by_task = {
        task_id: (
            [row for row in selected_manifest if int(row["benchmark_task_index"]) == task_id]
            if selected_manifest is not None
            else [
                {
                    "suite": args.suite,
                    "benchmark_task_index": task_id,
                    "trial": trial,
                    "environment_seed": args.seed,
                    "initial_state_identifier": None,
                    "query_noise_key_prefix": f"{args.noise_seed_base}:{args.suite}:{task_id}:{trial}:",
                    "max_episode_steps": MAX_STEPS[args.suite],
                }
                for trial in range(args.num_trials)
            ]
        )
        for task_id in range(args.max_tasks)
    }
    for task_id in tqdm.tqdm(range(args.max_tasks), desc=args.suite):
        task = suite.get_task(task_id)
        if selected_manifest is not None:
            task_row = next(
                row
                for row in final_manifest["tasks"]
                if row["suite"] == args.suite and int(row["benchmark_task_index"]) == task_id
            )
            canonical_runtime = " ".join(str(task.language).strip().lower().split())
            canonical_frozen = " ".join(str(task_row["canonical_instruction"]).strip().lower().split())
            if canonical_runtime != canonical_frozen:
                raise RuntimeError(f"benchmark task instruction mismatch for {args.suite}/task{task_id}")
        initial_states = suite.get_task_init_states(task_id)
        env = make_env(task, args.seed)
        try:
            for manifest_row in sorted(manifest_by_task[task_id], key=lambda row: int(row["trial"])):
                trial = int(manifest_row["trial"])
                environment_seed = int(manifest_row["environment_seed"])
                initial_hash = hashlib.sha256(
                    np.ascontiguousarray(np.asarray(initial_states[trial])).tobytes()
                ).hexdigest()
                if manifest_row.get("initial_state_identifier") not in (None, initial_hash):
                    raise RuntimeError(
                        f"initial-state mismatch for {args.suite}/task{task_id}/trial{trial}"
                    )
                expected_noise_prefix = f"{args.noise_seed_base}:{args.suite}:{task_id}:{trial}:"
                if manifest_row["query_noise_key_prefix"] != expected_noise_prefix:
                    raise RuntimeError("query-keyed policy-noise prefix differs from the frozen manifest")
                episode_limit = int(manifest_row["max_episode_steps"])
                if episode_limit != MAX_STEPS[args.suite]:
                    raise RuntimeError("episode limit differs from the pinned OpenPI protocol")
                manifest.append({**manifest_row, "initial_state_identifier": initial_hash})
                for policy_path in paths:
                    env.seed(environment_seed)
                    env.reset()
                    obs = env.set_init_state(initial_states[trial])
                    action_plan = collections.deque()
                    replay_images = []
                    episode_actions = []
                    query_index = 0
                    active_query_row = None
                    done = False
                    steps = 0
                    episode_started = time.time()
                    while steps < episode_limit + args.wait_steps:
                        if steps < args.wait_steps:
                            obs, _, done, _ = env.step(DUMMY_ACTION)
                            steps += 1
                            continue
                        request, image = observation_request(obs, task.language, args.resize_size)
                        replay_images.append(image)
                        if not action_plan:
                            request["latentloop"] = {
                                "suite": args.suite,
                                "task_id": task_id,
                                "episode_id": trial,
                                "query_index": query_index,
                                "reset": query_index == 0,
                                "policy_path": policy_path,
                            }
                            request_started = time.perf_counter()
                            response = client.infer(request)
                            client_roundtrip_ms = (time.perf_counter() - request_started) * 1000.0
                            chunk = np.asarray(response["actions"])
                            if len(chunk) < args.replan_steps:
                                raise RuntimeError("policy returned a shorter-than-native action chunk")
                            metrics = dict(response.get("latentloop_metrics", {}))
                            metrics.update(
                                {
                                    "policy_path": policy_path,
                                    "client_query_index": query_index,
                                    "client_wall_time": time.time(),
                                    "client_roundtrip_ms": client_roundtrip_ms,
                                    "executed_actions_actual": 0,
                                }
                            )
                            query_rows.append(metrics)
                            active_query_row = metrics
                            action_plan.extend(chunk[: args.replan_steps])
                            query_index += 1
                        action = action_plan.popleft()
                        episode_actions.append(np.asarray(action, dtype=np.float64))
                        obs, _, done, _ = env.step(action.tolist())
                        if active_query_row is not None:
                            active_query_row["executed_actions_actual"] += 1
                        if done:
                            break
                        steps += 1
                    row = {
                        "suite": args.suite,
                        "task_id": task_id,
                        "task": str(task.language),
                        "trial": trial,
                        "policy_path": policy_path,
                        "success": bool(done),
                        "episode_steps": steps,
                        "policy_queries": query_index,
                        "episode_wall_seconds": time.time() - episode_started,
                        **episode_action_metrics(episode_actions),
                    }
                    outcomes.append(row)
                    if args.save_video:
                        suffix = "success" if done else "failure"
                        target = output / "videos" / (
                            "task{:02d}_trial{:03d}_{}_{}.mp4".format(task_id, trial, policy_path, suffix)
                        )
                        imageio.mimwrite(target, replay_images, fps=args.video_fps)
                    print(json.dumps(row, sort_keys=True), flush=True)
        finally:
            env.close()

    outcomes_path = output / "episode_outcomes.csv"
    query_metrics_path = output / "query_metrics.jsonl"
    episode_manifest_path = output / "episode_manifest.json"
    environment_metadata_path = output / "environment_metadata.json"
    write_csv(outcomes_path, outcomes)
    with query_metrics_path.open("w", encoding="utf-8") as handle:
        for row in query_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    episode_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    successes = sum(int(row["success"]) for row in outcomes)
    operation_names = (
        "vision_encoder_calls",
        "prefix_embedding_calls",
        "prefix_transformer_calls",
        "latentloop_sequential_calls",
        "latentloop_direct_calls",
        "direct_reanchor_events",
        "full_prefix_refreshes",
        "action_expert_calls",
        "flow_iterations",
        "cache_rebuild_calls",
    )
    operation_totals = {
        name: sum(int(row.get(name, 0)) for row in query_rows) for name in operation_names
    }
    full_queries = operation_totals["full_prefix_refreshes"]
    executed_actions = sum(int(row.get("executed_actions_actual", 0)) for row in query_rows)
    full_intervals = []
    grouped_queries = {}
    for row in query_rows:
        key = (row["policy_path"], row["task_id"], row["episode_id"])
        grouped_queries.setdefault(key, []).append(row)
    for rows in grouped_queries.values():
        indices = [
            int(row["query_index"]) for row in rows if int(row.get("full_prefix_refreshes", 0))
        ]
        full_intervals.extend(b - a for a, b in zip(indices, indices[1:]))

    def latency_summary(key):
        values = np.asarray([float(row[key]) for row in query_rows if key in row], dtype=np.float64)
        if not len(values):
            return {"count": 0, "mean": None, "p50": None, "p95": None}
        return {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
        }

    summary = {
        "complete": True,
        "suite": args.suite,
        "tasks": min(args.max_tasks, suite.n_tasks),
        "trials_per_task": args.num_trials,
        "rollouts": len(outcomes),
        "successes": successes,
        "success_rate": successes / len(outcomes) if outcomes else 0.0,
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "resize_size": args.resize_size,
        "wait_steps": args.wait_steps,
        "elapsed_seconds": time.time() - run_started,
        "source_lock_id": server_metadata.get("source_lock_id"),
        "final_evaluation_manifest_sha256": server_metadata.get(
            "final_evaluation_manifest_sha256"
        ),
        "final_evaluation_manifest_id": server_metadata.get(
            "final_evaluation_manifest_id"
        ),
        "method_label": args.policy_path,
        "efficiency": {
            "policy_queries": len(query_rows),
            "operation_call_matrix": operation_totals,
            "actually_executed_actions": executed_actions,
            "full_prefix_call_ratio": full_queries / len(query_rows) if query_rows else 0.0,
            "actual_mean_k_q": len(query_rows) / full_queries if full_queries else None,
            "actual_mean_k_a": executed_actions / full_queries if full_queries else None,
            "observed_inter_refresh_k_q": {
                "count": len(full_intervals),
                "p10": float(np.quantile(full_intervals, 0.10)) if full_intervals else None,
                "p50": float(np.quantile(full_intervals, 0.50)) if full_intervals else None,
                "p90": float(np.quantile(full_intervals, 0.90)) if full_intervals else None,
            },
            "amortized_server_infer_ms_per_query": (
                sum(float(row.get("infer_ms", 0.0)) for row in query_rows) / len(query_rows)
                if query_rows else None
            ),
            "amortized_server_infer_ms_per_executed_action": (
                sum(float(row.get("infer_ms", 0.0)) for row in query_rows) / executed_actions
                if executed_actions else None
            ),
            "executed_action_throughput_per_second": (
                executed_actions / (time.time() - run_started) if executed_actions else 0.0
            ),
            "peak_vram_bytes": max([int(row.get("peak_vram_bytes", 0)) for row in query_rows] or [0]),
            "adapter_trainable_parameters": server_metadata.get("adapter_trainable_parameters", 0),
            "latency_ms": {
                key: latency_summary(key)
                for key in (
                    "infer_ms",
                    "client_roundtrip_ms",
                    "prefix_embedding_ms",
                    "full_prefix_ms",
                    "updater_ms",
                    "cache_rebuild_ms",
                    "action_expert_ms",
                )
            },
        },
    }
    metadata = {
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "robosuite": importlib.metadata.version("robosuite"),
        "render_backend": __import__("os").environ.get("MUJOCO_GL"),
        "success_check_timing": "immediately after each env.step, matching examples/libero/main.py",
        "suite_max_steps": MAX_STEPS[args.suite],
        "policy_server_metadata": server_metadata,
    }
    environment_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["protocol_artifacts"] = {
        "episode_outcomes": str(outcomes_path.resolve()),
        "episode_outcomes_sha256": hashlib.sha256(outcomes_path.read_bytes()).hexdigest(),
        "query_metrics": str(query_metrics_path.resolve()),
        "query_metrics_sha256": hashlib.sha256(query_metrics_path.read_bytes()).hexdigest(),
        "episode_manifest": str(episode_manifest_path.resolve()),
        "episode_manifest_sha256": hashlib.sha256(episode_manifest_path.read_bytes()).hexdigest(),
        "environment_metadata": str(environment_metadata_path.resolve()),
        "environment_metadata_sha256": hashlib.sha256(
            environment_metadata_path.read_bytes()
        ).hexdigest(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.paired_k1_smoke:
        grouped = {}
        for row in outcomes:
            grouped.setdefault((row["task_id"], row["trial"]), {})[row["policy_path"]] = row
        paired_outcomes = all(
            paths_by_key["original"]["success"] == paths_by_key["k1"]["success"]
            for paths_by_key in grouped.values()
        )
        k1_queries = [row for row in query_rows if row["policy_path"] == "k1"]
        smoke = {
            "complete": len(grouped) == args.max_tasks * args.num_trials,
            "tasks": args.max_tasks,
            "episodes": args.max_tasks * args.num_trials,
            "rollouts": len(outcomes),
            "all_query_actions_exact": bool(
                k1_queries and all(row.get("k1_reference_exact") is True for row in k1_queries)
            ),
            "paired_outcomes_identical": paired_outcomes,
            "updater_calls": max([int(row.get("updater_calls", 0)) for row in k1_queries] or [0]),
            "queries": len(k1_queries),
        }
        (output / "k1_episode_smoke.json").write_text(
            json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(smoke, indent=2, sort_keys=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
