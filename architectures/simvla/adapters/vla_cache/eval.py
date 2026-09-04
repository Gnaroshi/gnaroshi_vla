"""Manifest-paired LIBERO evaluation for the SimVLA VLA-Cache adaptation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    freeze_module,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    build_env_obs,
    get_libero_env,
    save_episode_video,
    video_frame_from_obs,
)

from .policy import VLACacheSimVLAPolicy
from .recipe import evaluation_row, scientific_contract


def _configure_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[4]
    upstream = Path(
        os.environ.get("SIMVLA_UPSTREAM_ROOT", root / "architectures/simvla/upstream")
    ).expanduser().resolve()
    libero = Path(
        os.environ.get("LIBERO_ROOT", upstream / "evaluation/libero/LIBERO")
    ).expanduser().resolve()
    for path in (root, upstream, libero):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if not (upstream / "models/modeling_smolvlm_vla.py").is_file():
        raise FileNotFoundError(f"SimVLA upstream not found: {upstream}")
    if not (libero / "libero").is_dir():
        raise FileNotFoundError(f"LIBERO root not found: {libero}")
    return root, upstream, libero


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_snapshot(root: Path) -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "root": str(root),
        "head": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_short": command("status", "--short"),
    }


def _load_manifest(path: Path, *, row: str, max_episodes: int | None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "checkpoint",
        "checkpoint_revision",
        "norm_stats",
        "suite",
        "episodes",
        "task_iteration_order",
        "determinism_seed",
        "action_noise_seed_base",
        "action_horizon",
        "execution_horizon",
        "flow_steps",
        "num_wait_steps",
        "max_policy_actions",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"episode manifest is missing fields: {missing}")
    expected = {
        "suite": "libero_10",
        "action_horizon": 10,
        "execution_horizon": 5,
        "flow_steps": 10,
        "num_wait_steps": 10,
        "max_policy_actions": 900,
    }
    mismatches = {
        key: {"expected": value, "observed": data.get(key)}
        for key, value in expected.items()
        if data.get(key) != value
    }
    if mismatches:
        raise ValueError(f"manifest violates SimVLA paper evaluation contract: {mismatches}")
    order = [int(value) for value in data["task_iteration_order"]["rank0"]]
    order_index = {task_id: index for index, task_id in enumerate(order)}
    episodes = sorted(
        data["episodes"],
        key=lambda item: (order_index[int(item["task_id"])], int(item["trial_id"])),
    )
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError("--max-episodes must be positive")
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError("episode manifest selected no episodes")
    data["selected_episodes"] = episodes
    data["selected_episode_count"] = len(episodes)
    data["row"] = row
    data["manifest_file_sha256"] = _sha256(path)
    return data


def _load_model(args: argparse.Namespace, manifest: dict[str, Any], device: torch.device):
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    model = SmolVLMVLA.from_pretrained(
        manifest["checkpoint"], revision=manifest["checkpoint_revision"]
    ).to(device).eval()
    model.action_space.load_norm_stats(args.norm_stats)
    freeze_module(model)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model)
    return model, processor


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def _episode_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["task_id"]), int(row["trial_id"])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    row_contract = evaluation_row(args.row)
    root, upstream, libero = _configure_paths()
    manifest_path = Path(args.episode_manifest).expanduser().resolve()
    manifest = _load_manifest(
        manifest_path, row=args.row, max_episodes=args.max_episodes
    )
    norm_stats = Path(args.norm_stats).expanduser().resolve()
    if not norm_stats.is_file():
        raise FileNotFoundError(f"norm stats not found: {norm_stats}")
    manifest_norm = Path(manifest["norm_stats"]).expanduser()
    if manifest_norm.is_file() and _sha256(manifest_norm) != _sha256(norm_stats):
        raise RuntimeError("provided norm stats differ from the paired manifest")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.jsonl"
    trace_path = output / "cache_query_trace.jsonl"
    contract = {
        "row": args.row,
        "row_contract": row_contract.to_dict(),
        "manifest": str(manifest_path),
        "manifest_file_sha256": manifest["manifest_file_sha256"],
        "selected_episode_count": manifest["selected_episode_count"],
        "norm_stats": str(norm_stats),
        "norm_stats_sha256": _sha256(norm_stats),
        "smolvlm_model": args.smolvlm_model,
        "scientific_contract": scientific_contract(),
    }
    contract_path = output / "run_contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError("existing output has a different immutable run contract")
    else:
        _write_json(contract_path, contract)

    completed_rows = _read_jsonl(progress_path)
    completed = {_episode_key(item) for item in completed_rows}
    selected_keys = {_episode_key(item) for item in manifest["selected_episodes"]}
    if not completed.issubset(selected_keys):
        raise RuntimeError("existing progress contains episodes outside this manifest")

    configure_strict_torch_determinism(int(manifest["determinism_seed"]))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, processor = _load_model(args, manifest, device)
    policy = VLACacheSimVLAPolicy(
        enable_reuse=row_contract.enable_reuse,
        model=model,
        processor=processor,
        device=device,
        suite=manifest["suite"],
        task_id=-1,
        trial_id=-1,
        action_noise_seed_base=int(manifest["action_noise_seed_base"]),
        log_action_chunks=False,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    metadata = {
        **contract,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "simvla_upstream_root": str(upstream),
        "libero_root": str(libero),
        "renderer": {
            key: os.environ.get(key)
            for key in (
                "MUJOCO_GL",
                "PYOPENGL_PLATFORM",
                "MUJOCO_EGL_DEVICE_ID",
                "CUBLAS_WORKSPACE_CONFIG",
                "CUDA_DEVICE_MAX_CONNECTIONS",
                "PYTHONHASHSEED",
            )
        },
        "git": _git_snapshot(root),
        "resume_completed_at_start": len(completed),
    }
    _write_json(output / "environment_metadata.json", metadata)

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[manifest["suite"]]()
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for episode in manifest["selected_episodes"]:
        by_task[int(episode["task_id"])].append(episode)
    progress = tqdm(
        total=len(selected_keys),
        initial=len(completed),
        desc=f"{args.row} LIBERO-Long",
        dynamic_ncols=True,
        mininterval=args.tqdm_mininterval,
    )
    run_started = time.time()
    for task_id in [int(value) for value in manifest["task_iteration_order"]["rank0"]]:
        pending = [item for item in by_task.get(task_id, []) if _episode_key(item) not in completed]
        if not pending:
            continue
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        environment_seed = int(pending[0]["environment_seed"])
        env, prompt = get_libero_env(task, int(manifest["environment_resolution"]), environment_seed)
        try:
            for episode in pending:
                trial_id = int(episode["trial_id"])
                init_state_index = int(episode["init_state_index"])
                policy.task_id = task_id
                policy.trial_id = trial_id
                policy.reset()
                env.reset()
                obs = env.set_init_state(init_states[init_state_index])
                for _ in range(int(manifest["num_wait_steps"])):
                    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

                success = False
                policy_step_ms: list[float] = []
                query_step_ms: list[float] = []
                frames: list[np.ndarray] = []
                for action_index in range(int(manifest["max_policy_actions"])):
                    if args.save_failure_videos and action_index % args.video_stride == 0:
                        frames.append(video_frame_from_obs(obs))
                    image0, image1, proprio = build_env_obs(obs)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    action = policy.act(image0, image1, proprio, prompt)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    policy_step_ms.append(elapsed_ms)
                    if action.info["refreshed"]:
                        query_step_ms.append(elapsed_ms)
                    obs, _, done, _ = env.step(action.action.tolist())
                    if done:
                        success = True
                        break

                counters = dict(policy.metrics.counters)
                decoder_reports = [item["vla_cache"]["decoder"] for item in policy.query_trace]
                selected_counts = [
                    len(item["vla_cache"]["selection"]["reusable_positions"])
                    for item in policy.query_trace
                ]
                computed = int(counters.get("computed_text_token_layers", 0))
                skipped = int(counters.get("skipped_text_token_layers", 0))
                row = {
                    "row": args.row,
                    "suite": manifest["suite"],
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "init_state_index": init_state_index,
                    "environment_seed": environment_seed,
                    "success": success,
                    "episode_length": len(policy_step_ms),
                    "num_policy_queries": int(counters.get("num_policy_queries", 0)),
                    "num_action_transformer_calls": int(counters.get("num_action_transformer_calls", 0)),
                    "num_vla_cache_anchor_queries": int(counters.get("num_vla_cache_anchor_queries", 0)),
                    "num_vla_cache_nonanchor_queries": int(counters.get("num_vla_cache_nonanchor_queries", 0)),
                    "num_actual_kv_reuse_queries": int(counters.get("num_actual_kv_reuse_queries", 0)),
                    "computed_text_token_layers": computed,
                    "skipped_text_token_layers": skipped,
                    "text_token_layer_reduction": skipped / max(computed + skipped, 1),
                    "reusable_visual_positions_mean": _mean(selected_counts),
                    "policy_latency_mean_ms": _mean(policy_step_ms),
                    "policy_latency_p50_ms": _quantile(policy_step_ms, 0.50),
                    "policy_latency_p95_ms": _quantile(policy_step_ms, 0.95),
                    "query_latency_mean_ms": _mean(query_step_ms),
                    "vlm_latency_total_ms": float(sum(policy.metrics.latencies.get("VLM_encoder_ms", []))),
                    "action_latency_total_ms": float(sum(policy.metrics.latencies.get("action_transformer_ms", []))),
                    "actual_reuse_observed": any(item["actual_kv_reuse"] for item in decoder_reports),
                }
                _write_json(
                    output / "query_traces" / f"task_{task_id:02d}_trial_{trial_id:03d}.json",
                    [
                        {"task_id": task_id, "trial_id": trial_id, **query}
                        for query in policy.query_trace
                    ],
                )
                _append_jsonl(progress_path, row)
                completed_rows.append(row)
                completed.add((task_id, trial_id))
                if args.save_failure_videos and not success:
                    save_episode_video(
                        frames,
                        output / "videos" / f"task_{task_id:02d}_trial_{trial_id:03d}_failure.mp4",
                        10,
                    )
                successes = sum(bool(item["success"]) for item in completed_rows)
                progress.update(1)
                progress.set_postfix(success=f"{successes}/{len(completed_rows)}", sr=f"{100 * successes / len(completed_rows):.1f}%")
                _write_json(
                    output / "live_summary.json",
                    {
                        "status": "running",
                        "completed_episodes": len(completed_rows),
                        "total_episodes": len(selected_keys),
                        "successes": successes,
                        "success_rate": successes / len(completed_rows),
                        "elapsed_seconds": time.time() - run_started,
                    },
                )
        finally:
            env.close()
    progress.close()

    completed_rows.sort(key=_episode_key)
    with trace_path.open("w", encoding="utf-8") as trace_handle:
        for item in completed_rows:
            episode_trace = output / "query_traces" / (
                f"task_{int(item['task_id']):02d}_trial_{int(item['trial_id']):03d}.json"
            )
            if not episode_trace.is_file():
                raise RuntimeError(f"missing completed episode query trace: {episode_trace}")
            for query in json.loads(episode_trace.read_text(encoding="utf-8")):
                trace_handle.write(json.dumps(query, sort_keys=True) + "\n")
    with (output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(completed_rows[0]))
        writer.writeheader()
        writer.writerows(completed_rows)
    taskwise = {}
    for task_id in sorted({int(item["task_id"]) for item in completed_rows}):
        task_rows = [item for item in completed_rows if int(item["task_id"]) == task_id]
        taskwise[str(task_id)] = {
            "episodes": len(task_rows),
            "successes": sum(bool(item["success"]) for item in task_rows),
            "success_rate": float(np.mean([item["success"] for item in task_rows])),
        }
    computed = sum(int(item["computed_text_token_layers"]) for item in completed_rows)
    skipped = sum(int(item["skipped_text_token_layers"]) for item in completed_rows)
    summary = {
        "verdict": "SIMVLA_VLA_CACHE_LIBERO_EVAL_COMPLETE",
        "row": args.row,
        "episodes": len(completed_rows),
        "successes": sum(bool(item["success"]) for item in completed_rows),
        "success_rate": float(np.mean([item["success"] for item in completed_rows])),
        "taskwise": taskwise,
        "latency_per_executed_action_ms": _mean([float(item["policy_latency_mean_ms"]) for item in completed_rows]),
        "query_latency_mean_ms": _mean([float(item["query_latency_mean_ms"]) for item in completed_rows]),
        "computed_text_token_layers": computed,
        "skipped_text_token_layers": skipped,
        "text_token_layer_reduction": skipped / max(computed + skipped, 1),
        "actual_kv_reuse_queries": sum(int(item["num_actual_kv_reuse_queries"]) for item in completed_rows),
        "peak_cuda_memory_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
        ),
        "manifest_file_sha256": manifest["manifest_file_sha256"],
        "manifest_declared_sha256": manifest.get("manifest_sha256"),
        "elapsed_seconds_this_invocation": time.time() - run_started,
    }
    if row_contract.enable_reuse and len(completed_rows) > 1:
        if summary["actual_kv_reuse_queries"] <= 0 or summary["skipped_text_token_layers"] <= 0:
            raise RuntimeError("VLA-Cache completed without actual KV reuse/token skipping")
    if not row_contract.enable_reuse:
        if summary["actual_kv_reuse_queries"] or summary["skipped_text_token_layers"]:
            raise RuntimeError("matched full control unexpectedly reused cached computation")
    _write_json(output / "summary.json", summary)
    _write_json(output / "live_summary.json", {"status": "complete", **summary})
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", required=True)
    value.add_argument("--episode-manifest", required=True)
    value.add_argument("--row", choices=("vla_cache_full", "vla_cache"), required=True)
    value.add_argument("--norm-stats", required=True)
    value.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    value.add_argument("--max-episodes", type=int)
    value.add_argument("--save-failure-videos", action="store_true")
    value.add_argument("--video-stride", type=int, default=2)
    value.add_argument("--tqdm-mininterval", type=float, default=1.0)
    value.add_argument("--device", default="cuda")
    return value


def main() -> None:
    print(json.dumps(evaluate(parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
