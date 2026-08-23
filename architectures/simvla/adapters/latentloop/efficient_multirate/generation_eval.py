"""Paired LIBERO-Long screening for SimVLA Generation Loop policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    load_generation_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_policy import (
    RealSimVLAGenerationPolicy,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_source_lock import (
    generation_source_lock,
)
from architectures.simvla.adapters.latentloop.native_v0_long_eval import (
    _SynchronizedFullPolicy,
    _trajectory_metrics,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    append_jsonl,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
    write_json,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    build_env_obs,
    get_libero_env,
    save_episode_video,
    video_frame_from_obs,
)
from architectures.simvla.wrappers.simvla_generation_gpu_guard import (
    parse_generation_gpu_ids,
)


MANIFEST_SCHEMA = "simvla_generation_libero_long_v1"


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    selected = parse_generation_gpu_ids(os.environ.get("SIMVLA_GPU_IDS"))
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing existing manifest: {destination}")
    source = generation_source_lock(
        checkpoint=args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        norm_stats=args.norm_stats,
        exact_cache=args.cache,
    )
    task_partitions = (
        (tuple(range(10)),)
        if len(selected) == 1
        else (tuple(range(5)), tuple(range(5, 10)))
    )
    task_to_rank = {
        task_id: rank
        for rank, task_ids in enumerate(task_partitions)
        for task_id in task_ids
    }
    episodes = [
        {
            "suite": "libero_10",
            "task_id": task_id,
            "trial_id": trial_id,
            "init_state_index": trial_id,
            "environment_seed": int(args.environment_seed),
            "physical_gpu_id": selected[task_to_rank[task_id]],
        }
        for task_id in range(10)
        for trial_id in range(args.trials_per_task)
    ]
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "source_combined_sha256": source["combined_sha256"],
        "selected_physical_gpu_ids": list(selected),
        "checkpoint": args.checkpoint,
        "checkpoint_revision": args.checkpoint_revision,
        "norm_stats": str(Path(args.norm_stats).expanduser().resolve()),
        "exact_cache": str(Path(args.cache).expanduser().resolve()),
        "suite": "libero_10",
        "tasks": 10,
        "trials_per_task": int(args.trials_per_task),
        "episodes_per_row": len(episodes),
        "task_partition": {
            f"rank{rank}": list(task_ids)
            for rank, task_ids in enumerate(task_partitions)
        },
        "task_iteration_order": {
            f"rank{rank}": list(reversed(task_ids))
            for rank, task_ids in enumerate(task_partitions)
        },
        "action_horizon": 10,
        "execution_horizon": 5,
        "flow_steps": 10,
        "num_wait_steps": 10,
        "max_policy_actions": 900,
        "client_resize_size": 224,
        "model_image_size": 384,
        "environment_resolution": 256,
        "action_noise_seed_base": int(args.action_noise_seed_base),
        "determinism_seed": int(args.seed),
        "environment_seed": int(args.environment_seed),
        "renderer": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "GALLIUM_DRIVER": "llvmpipe",
            "LIBGL_ALWAYS_SOFTWARE": "true",
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "PYTHONHASHSEED": str(args.seed),
        },
        "episodes": episodes,
    }
    payload["manifest_sha256"] = _canonical_hash(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, payload)
    return payload


def _validate_manifest(manifest: dict[str, Any], selected: tuple[int, ...]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported Generation Loop evaluation manifest")
    if manifest.get("selected_physical_gpu_ids") != list(selected):
        raise RuntimeError("manifest physical GPUs differ from SIMVLA_GPU_IDS")
    copied = dict(manifest)
    digest = copied.pop("manifest_sha256")
    if _canonical_hash(copied) != digest:
        raise RuntimeError("manifest hash mismatch")
    mismatches = {
        key: (os.environ.get(key), value)
        for key, value in manifest["renderer"].items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"renderer contract mismatch: {mismatches}")


def _make_policy(
    *,
    row: str,
    model: Any,
    processor: Any,
    updater: Any,
    device: torch.device,
    task_id: int,
    trial_id: int,
    action_noise_seed_base: int,
) -> Any:
    if row == "baseline_k1":
        return _SynchronizedFullPolicy(
            model=model,
            processor=processor,
            dcld_core=None,
            mode="full",
            refresh_every=1,
            flow_steps=10,
            image_size=384,
            replan_steps=5,
            client_resize_size=224,
            device=device,
            suite="libero_10",
            row_name=row,
            task_id=task_id,
            trial_id=trial_id,
            paired_action_noise=True,
            action_noise_seed_base=action_noise_seed_base,
            log_action_chunks=False,
        )
    n_g = int(row.rsplit("ng", 1)[1])
    return RealSimVLAGenerationPolicy(
        model=model,
        processor=processor,
        updater=updater,
        n_g=n_g,
        device=device,
        suite="libero_10",
        task_id=task_id,
        trial_id=trial_id,
        action_noise_seed_base=action_noise_seed_base,
        log_action_chunks=False,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    selected = parse_generation_gpu_ids(os.environ.get("SIMVLA_GPU_IDS"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, selected)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must exactly match SIMVLA_GPU_IDS")
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size != len(selected):
        raise RuntimeError("Generation Loop evaluation WORLD_SIZE must match SIMVLA_GPU_IDS")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(manifest, selected)
    configure_strict_torch_determinism(int(manifest["determinism_seed"]))
    source = generation_source_lock(
        checkpoint=args.checkpoint,
        checkpoint_revision=args.checkpoint_revision,
        norm_stats=args.norm_stats,
        exact_cache=args.cache,
    )
    if source["combined_sha256"] != manifest["source_combined_sha256"]:
        raise RuntimeError("current source differs from evaluation manifest")

    updater = None
    checkpoint_payload = None
    if args.row != "baseline_k1":
        updater, checkpoint_payload = load_generation_checkpoint(
            args.generation_checkpoint, device=device
        )
        if checkpoint_payload["source_lock"]["combined_sha256"] != source["combined_sha256"]:
            raise RuntimeError("Generation checkpoint source differs")
        freeze_module(updater)

    output = Path(args.output).expanduser().resolve()
    exists = torch.tensor(int(output.exists()), device=device)
    dist.all_reduce(exists, op=dist.ReduceOp.MAX)
    if int(exists.item()):
        raise FileExistsError(f"refusing existing output: {output}")
    if rank == 0:
        output.mkdir(parents=True)
    dist.barrier()
    task_ids = [int(value) for value in manifest["task_iteration_order"][f"rank{rank}"]]
    shard = output / f"shard_rank{rank}_tasks_{min(task_ids)}_{max(task_ids)}"
    shard.mkdir()
    write_json(shard / "source_lock.json", source)

    model, processor, _ = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    episode_rows: list[dict[str, Any]] = []
    assigned_total = len(task_ids) * int(manifest["trials_per_task"])
    progress = tqdm(
        total=assigned_total,
        desc=f"{args.row} rank{rank}",
        dynamic_ncols=True,
    )
    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        specs = [item for item in manifest["episodes"] if item["task_id"] == task_id]
        env, prompt = get_libero_env(task, 256, int(manifest["environment_seed"]))
        try:
            for spec in specs:
                trial_id = int(spec["trial_id"])
                env.reset()
                obs = env.set_init_state(
                    initial_states[int(spec["init_state_index"]) % len(initial_states)]
                )
                for _ in range(10):
                    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                policy = _make_policy(
                    row=args.row,
                    model=model,
                    processor=processor,
                    updater=updater,
                    device=device,
                    task_id=task_id,
                    trial_id=trial_id,
                    action_noise_seed_base=int(manifest["action_noise_seed_base"]),
                )
                actions: list[np.ndarray] = []
                policy_ms: list[float] = []
                frames: list[np.ndarray] = []
                success = False
                for action_index in range(900):
                    if args.save_video and action_index % args.video_stride == 0:
                        frames.append(video_frame_from_obs(obs))
                    image0, image1, proprio = build_env_obs(obs)
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    step = policy.act(image0, image1, proprio, prompt)
                    torch.cuda.synchronize(device)
                    policy_ms.append((time.perf_counter() - started) * 1000.0)
                    obs, _, done, _ = env.step(step.action.tolist())
                    actions.append(step.action.copy())
                    if done:
                        success = True
                        break
                counters = {key: int(value) for key, value in policy.metrics.counters.items()}
                latencies = {
                    key: [float(value) for value in values]
                    for key, values in policy.metrics.latencies.items()
                }
                row = {
                    "row": args.row,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "success": success,
                    "episode_length": len(actions),
                    "num_policy_queries": counters.get("num_policy_queries", 0),
                    "num_full_vlm_calls": counters.get("num_full_vlm_calls", 0),
                    "num_action_transformer_flow_iterations": counters.get(
                        "num_action_transformer_calls", 0
                    ),
                    "num_generation_decoder_only_steps": counters.get(
                        "num_generation_decoder_only_steps", 0
                    ),
                    "latency_per_executed_action_ms": float(np.mean(policy_ms)),
                    "synchronized_policy_ms_p50": float(np.quantile(policy_ms, 0.50)),
                    "synchronized_policy_ms_p95": float(np.quantile(policy_ms, 0.95)),
                    **_trajectory_metrics(actions),
                }
                episode_rows.append(row)
                append_jsonl(shard / "progress.jsonl", {"completed": len(episode_rows), "total": assigned_total, **row})
                append_jsonl(
                    shard / "latency_records.jsonl",
                    {
                        "row": args.row,
                        "task_id": task_id,
                        "trial_id": trial_id,
                        "synchronized_policy_ms_per_executed_action": policy_ms,
                        "VLM_encoder_ms": latencies.get("VLM_encoder_ms", []),
                        "action_transformer_ms": latencies.get("action_transformer_ms", []),
                        "generation_loop_ms": latencies.get("generation_loop_ms", []),
                    },
                )
                if args.save_video and (not args.video_failures_only or not success):
                    video_root = shard / "videos" / f"task_{task_id:02d}"
                    existing = len(list(video_root.glob("*.mp4")))
                    if existing < args.video_max_per_task:
                        suffix = "success" if success else "failure"
                        save_episode_video(
                            frames,
                            video_root / f"trial_{trial_id:03d}_{suffix}.mp4",
                            10,
                        )
                progress.update(1)
                progress.set_postfix(
                    successes=sum(int(item["success"]) for item in episode_rows),
                    sr=f"{np.mean([item['success'] for item in episode_rows]) * 100:.1f}%",
                )
        finally:
            env.close()
    progress.close()
    with (shard / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    shard_summary = {
        "row": args.row,
        "rank": rank,
        "episodes": len(episode_rows),
        "successes": sum(int(row["success"]) for row in episode_rows),
        "success_rate": float(np.mean([row["success"] for row in episode_rows])),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_combined_sha256": source["combined_sha256"],
        "optimizer_step": (
            int(checkpoint_payload["optimizer_step"])
            if checkpoint_payload is not None
            else None
        ),
    }
    write_json(shard / "shard_summary.json", shard_summary)
    dist.barrier()
    result: dict[str, Any] = shard_summary
    if rank == 0:
        shard_dirs = sorted(output.glob("shard_rank*_tasks_*"))
        summaries = [
            json.loads((directory / "shard_summary.json").read_text(encoding="utf-8"))
            for directory in shard_dirs
        ]
        successes = sum(int(item["successes"]) for item in summaries)
        episodes = sum(int(item["episodes"]) for item in summaries)
        result = {
            "verdict": "GENERATION_LIBERO_LONG_SCREEN_COMPLETE",
            "paper_result": bool(int(manifest["trials_per_task"]) == 50),
            "row": args.row,
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "manifest_sha256": manifest["manifest_sha256"],
            "source_combined_sha256": source["combined_sha256"],
            "optimizer_step": shard_summary["optimizer_step"],
            "shards": summaries,
        }
        write_json(output / "row_summary.json", result)
    dist.barrier()
    dist.destroy_process_group()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--cache", required=True)
    manifest.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    manifest.add_argument("--checkpoint-revision", required=True)
    manifest.add_argument("--norm-stats", required=True)
    manifest.add_argument("--trials-per-task", type=int, choices=(10, 50), default=10)
    manifest.add_argument("--seed", type=int, default=20260815)
    manifest.add_argument("--environment-seed", type=int, default=7)
    manifest.add_argument("--action-noise-seed-base", type=int, default=6828326409295398833)
    manifest.set_defaults(handler=create_manifest)
    run = commands.add_parser("evaluate")
    run.add_argument("--row", choices=("baseline_k1", "generation_ng3", "generation_ng2"), required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--cache", required=True)
    run.add_argument("--generation-checkpoint", default="")
    run.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    run.add_argument("--checkpoint-revision", required=True)
    run.add_argument("--norm-stats", required=True)
    run.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    run.add_argument("--save-video", action="store_true")
    run.add_argument("--video-failures-only", action="store_true")
    run.add_argument("--video-stride", type=int, default=2)
    run.add_argument("--video-max-per-task", type=int, default=2)
    run.set_defaults(handler=evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.handler(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
