#!/usr/bin/env python3
"""Resumable paired LIBERO-Long evaluation, compatible with the Python 3.8 client."""

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from tqdm import tqdm
from libero.libero import benchmark
from openpi_client import websocket_client_policy

from evaluate_pi05_latentloop_client import make_env, observation_request, DUMMY_ACTION


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def summarize(output, config_id, row, smoke):
    episodes = [json.loads(p.read_text()) for p in sorted((output / "episodes").glob("task*_trial*.json"))]
    rows = [e["outcome"] for e in episodes]
    queries = [q for e in episodes for q in e["queries"]]
    warm = [q for q in queries if q["query_index"] >= 2]
    metric_names = sorted({k for q in warm for k in q if k.endswith("_ms")})
    latency = {k: {"mean": float(np.mean([q.get(k, 0.0) for q in warm])),
                   "p50": float(np.median([q.get(k, 0.0) for q in warm])),
                   "p95": float(np.percentile([q.get(k, 0.0) for q in warm], 95))}
               for k in metric_names} if warm else {}
    counters = {key: sum(q.get(key, 0) for q in queries) for key in (
        "full_prefix_calls", "condition_updater_calls", "action_expert_calls", "generation_updater_calls", "flow_iterations")}
    summary = {"config_id": config_id, "row": row, "episodes": len(rows),
               "successes": sum(r["success"] for r in rows), "smoke_only": smoke,
               "success_rate": sum(r["success"] for r in rows) / max(1, len(rows)),
               "complete": not smoke and len(rows) == 500,
               "latency_ms_per_query_excluding_first_two_queries_per_episode": latency,
               "operation_counts": counters,
               "actual_executed_actions": sum(q["executed_actions_actual"] for q in queries),
               "policy_ms_per_actual_action": sum(q["infer_ms"] for q in queries) / max(1, sum(q["executed_actions_actual"] for q in queries)),
               "task_successes": {str(i): sum(r["success"] for r in rows if r["task_id"] == i) for i in range(10)}}
    atomic_json(output / "summary.json", summary)
    if rows:
        with (output / "episode_outcomes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def run(args):
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    (output / "episodes").mkdir(parents=True, exist_ok=True)
    contract = output / "contract.json"
    identity = {"config_id": config["config_id"], "row": args.row, "smoke": args.smoke}
    if contract.exists() and json.loads(contract.read_text()) != identity:
        raise ValueError("output directory contains a different evaluation contract")
    atomic_json(contract, identity)
    client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=args.port)
    metadata = client.get_server_metadata()
    if metadata.get("config_id") != config["config_id"]:
        raise ValueError("server/client provenance mismatch")
    atomic_json(output / "environment_metadata.json", {
        "python": platform.python_version(), "mujoco": importlib.metadata.version("mujoco"),
        "numpy": np.__version__, "server": metadata,
        "protocol": {"H": 10, "R": 5, "wait_steps": 10, "max_policy_steps": 520,
                     "renderer": "egl", "camera_resolution": 256, "client_resize": 224}})
    manifest = json.loads(Path(config["final_manifest"]).read_text())
    selected = [r for r in manifest["episodes"] if r["suite"] == "libero_10"]
    if len(selected) != 500 or len({(r["benchmark_task_index"], r["trial"]) for r in selected}) != 500:
        raise ValueError("expected 10 tasks x 50 unique trials")
    selected.sort(key=lambda r: (r["benchmark_task_index"], r["trial"]))
    if args.smoke:
        selected = selected[:1]
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    progress = tqdm(selected, desc=args.row, unit="episode", mininterval=2.0)
    env = None
    previous_task = None
    successes = 0
    try:
        for frozen in progress:
            task_id, trial = int(frozen["benchmark_task_index"]), int(frozen["trial"])
            episode_path = output / "episodes" / ("task%02d_trial%03d.json" % (task_id, trial))
            if episode_path.exists():
                prior = json.loads(episode_path.read_text())
                if prior["config_id"] != config["config_id"]:
                    raise ValueError("episode provenance mismatch")
                successes += int(prior["outcome"]["success"])
                continue
            task = suite.get_task(task_id)
            instruction = next(t["canonical_instruction"] for t in manifest["tasks"]
                               if t["suite"] == "libero_10" and t["benchmark_task_index"] == task_id)
            if " ".join(task.language.lower().split()) != " ".join(instruction.lower().split()):
                raise ValueError("LIBERO task order/language changed")
            initial_states = suite.get_task_init_states(task_id)
            init_hash = hashlib.sha256(np.ascontiguousarray(np.asarray(initial_states[trial])).tobytes()).hexdigest()
            if init_hash != frozen["initial_state_identifier"]:
                raise ValueError("LIBERO initial state changed")
            if frozen["query_noise_key_prefix"] != "%d:libero_10:%d:%d:" % (config["noise_seed"], task_id, trial):
                raise ValueError("paired policy noise differs from manifest")
            if env is None or previous_task != task_id:
                if env is not None:
                    env.close()
                env = make_env(task, int(frozen["environment_seed"]))
                previous_task = task_id
            env.seed(int(frozen["environment_seed"]))
            env.reset()
            obs = env.set_init_state(initial_states[trial])
            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            started = time.perf_counter()
            plan = collections.deque()
            queries = []
            done = False
            actions = []
            images = []
            limit = 10 if args.smoke else int(frozen["max_episode_steps"])
            if not args.smoke and limit != 520:
                raise ValueError("wrong pi0.5 LIBERO-Long episode horizon")
            for step in range(limit):
                if not plan:
                    request, image = observation_request(obs, task.language, 224)
                    request["latentloop"] = {"suite": "libero_10", "task_id": task_id, "episode_id": trial,
                                             "query_index": len(queries), "reset": not queries, "policy_path": args.row}
                    call_start = time.perf_counter()
                    reply = client.infer(request)
                    chunk = np.asarray(reply["actions"])
                    if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                        raise ValueError("invalid returned H=10 chunk")
                    metric = dict(reply["latentloop_metrics"])
                    metric.update(client_roundtrip_ms=(time.perf_counter() - call_start) * 1000,
                                  executed_actions_actual=0)
                    expected_expert = 10 if args.row in ("baseline", "condition_k2") else 3
                    expected_grid = 3 if args.row == "naive_nfe3" else 10
                    if metric["action_expert_calls"] != expected_expert or metric["flow_iterations"] != expected_grid:
                        raise ValueError("runtime action expert/grid counters differ from requested row")
                    queries.append(metric)
                    plan.extend(chunk[:5])
                    if trial == 0:
                        images.append(image)
                action = plan.popleft()
                actions.append(action)
                obs, _, done, _ = env.step(action.tolist())
                queries[-1]["executed_actions_actual"] += 1
                if done:
                    break
            outcome = {"task_id": task_id, "trial": trial, "success": bool(done), "policy_steps": len(actions),
                       "queries": len(queries), "wall_seconds": time.perf_counter() - started,
                       "row": args.row, "video_warning": ""}
            np.save(output / "episodes" / ("task%02d_trial%03d_actions.npy" % (task_id, trial)), np.asarray(actions))
            if images:
                try:
                    import imageio
                    imageio.mimwrite(str(episode_path.with_suffix(".mp4")), images, fps=4)
                except (OSError, RuntimeError) as error:
                    outcome["video_warning"] = str(error)
            atomic_json(episode_path, {"config_id": config["config_id"], "outcome": outcome, "queries": queries,
                                       "initial_state_sha256": init_hash})
            successes += int(done)
            finished = len(list((output / "episodes").glob("task*_trial*.json")))
            progress.set_postfix(success="%d/%d" % (successes, finished))
            tqdm.write("%s task=%d trial=%d success=%s total=%d/%d" % (args.row, task_id, trial, done, successes, finished))
            summarize(output, config["config_id"], args.row, args.smoke)
    finally:
        if env is not None:
            env.close()
        summarize(output, config["config_id"], args.row, args.smoke)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--row", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())
