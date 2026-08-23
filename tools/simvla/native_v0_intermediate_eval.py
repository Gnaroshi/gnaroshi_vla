"""Intermediate-checkpoint LIBERO-Long diagnostics for native SimVLA V0.

This runner deliberately does not satisfy or bypass the final-150K scientific
gate. It evaluates an immutable intermediate checkpoint on a paired 10x50
manifest and labels every artifact as diagnostic-only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    CHECKPOINT_FORMAT,
    load_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.native_v0_long_eval import (
    _canonical_hash,
    _policy,
    _trajectory_metrics,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    append_jsonl,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
    native_v0_source_manifest,
    require_gate,
    write_json,
)
from architectures.simvla.adapters.latentloop.source_lock import sha256_file
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    build_env_obs,
    get_libero_env,
    save_episode_video,
    video_frame_from_obs,
)
from architectures.simvla.wrappers.simvla_two_gpu_guard import parse_selected_gpu_ids


MANIFEST_SCHEMA = "simvla_native_v0_intermediate_libero_long_500_v1"
DIAGNOSTIC_CLASS = "INTERMEDIATE_CHECKPOINT_DIAGNOSTIC_ONLY"


def _source_without_gpu_slots(source: dict[str, Any]) -> dict[str, Any]:
    """Remove invocation metadata while retaining every scientific input."""

    normalized = copy.deepcopy(source)
    normalized.pop("combined_sha256", None)
    normalized.pop("selected_physical_gpu_ids", None)
    complete = normalized.get("complete_source_lock")
    if isinstance(complete, dict):
        # These fields describe how/where collection was invoked. Model/data
        # identity remains covered by critical_file_sha256, explicit commits,
        # checkpoint identity, norm hash, package versions, and Python/CUDA.
        for key in (
            "command",
            "conda_env",
            "cuda_visible_devices",
            "root_branch",
            "root_commit",
            "root_status_short",
            "simvla_upstream_status_short",
        ):
            complete.pop(key, None)
    return normalized


def require_source_compatible(
    *,
    checkpoint_source: dict[str, Any],
    runtime_source: dict[str, Any],
) -> None:
    expected = _source_without_gpu_slots(checkpoint_source)
    observed = _source_without_gpu_slots(runtime_source)
    if expected == observed:
        return
    changed = sorted(
        key
        for key in set(expected) | set(observed)
        if expected.get(key) != observed.get(key)
    )
    raise RuntimeError(
        "runtime source differs from checkpoint source beyond physical GPU ordinals: "
        + ", ".join(changed)
    )


def _load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path).expanduser().resolve(), map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported native V0 checkpoint: {payload.get('checkpoint_format')}")
    step = int(payload.get("global_optimizer_step", -1))
    if not 0 < step < 150_000:
        raise ValueError(f"intermediate diagnostic requires 0 < step < 150000, got {step}")
    if bool(payload.get("scientific_primary_checkpoint")):
        raise ValueError("final scientific checkpoint must use the strict final evaluator")
    return payload


def create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    selected = parse_selected_gpu_ids(os.environ.get("SIMVLA_GPU_IDS"))
    path = Path(args.output).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing existing manifest: {path}")
    checkpoint_path = Path(args.v0_checkpoint).expanduser().resolve()
    checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
    checkpoint_source = checkpoint_payload["source_lock"]
    runtime_source = native_v0_source_manifest(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        cache=args.cache,
    )
    require_source_compatible(
        checkpoint_source=checkpoint_source,
        runtime_source=runtime_source,
    )
    require_gate(
        args.parity_gate,
        verdicts=("K1_HOOK_PARITY_PASS",),
        source_combined_sha256=checkpoint_source["combined_sha256"],
    )
    episodes = []
    for task_id in range(10):
        physical_gpu = selected[0] if task_id <= 4 else selected[1]
        for trial_id in range(50):
            episodes.append(
                {
                    "suite": "libero_10",
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "init_state_index": trial_id,
                    "environment_seed": int(args.environment_seed),
                    "flow_noise_key": {
                        "seed_base": args.action_noise_seed_base,
                        "suite": "libero_10",
                        "task_id": task_id,
                        "trial_id": trial_id,
                        "policy_query_index": "runtime_integer",
                        "flow_steps": 10,
                        "action_horizon": 10,
                        "action_dim": 7,
                    },
                    "physical_gpu_id": physical_gpu,
                    "shard": 0 if task_id <= 4 else 1,
                }
            )
    renderer = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "GALLIUM_DRIVER": "llvmpipe",
        "HF_HUB_OFFLINE": "1",
        "LIBGL_ALWAYS_SOFTWARE": "true",
        "LP_NUM_THREADS": "0",
        "MKL_NUM_THREADS": "1",
        "MUJOCO_GL": "osmesa",
        "NUMEXPR_NUM_THREADS": "1",
        "NVIDIA_TF32_OVERRIDE": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYOPENGL_PLATFORM": "osmesa",
        "PYTHONHASHSEED": "20260815",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    training_config = checkpoint_payload.get("training_config", {})
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "evaluation_class": DIAGNOSTIC_CLASS,
        "scientific_claim_allowed": False,
        "final_150k_required_for_scientific_evaluation": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint_payload["global_optimizer_step"]),
        "checkpoint_scientific_primary": False,
        "source_combined_sha256": checkpoint_source["combined_sha256"],
        "runtime_source_combined_sha256": runtime_source["combined_sha256"],
        "gpu_ordinal_only_source_difference_allowed": True,
        "checkpoint_source_lock": checkpoint_source,
        "runtime_source_lock": runtime_source,
        "selected_physical_gpu_ids": list(selected),
        "suite": "libero_10",
        "tasks": 10,
        "episodes_per_task": 50,
        "episodes_per_row": 500,
        "task_partition": {"rank0": [0, 1, 2, 3, 4], "rank1": [5, 6, 7, 8, 9]},
        "task_iteration_order": {"rank0": [4, 3, 2, 1, 0], "rank1": [9, 8, 7, 6, 5]},
        "action_horizon": 10,
        "execution_horizon": 5,
        "flow_steps": 10,
        "num_wait_steps": 10,
        "max_policy_actions": 900,
        "client_resize_size": 224,
        "model_image_size": 384,
        "environment_resolution": 256,
        "renderer": renderer,
        "checkpoint": args.checkpoint,
        "norm_stats": str(Path(args.norm_stats).expanduser().resolve()),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "action_noise_seed_base": int(args.action_noise_seed_base),
        "seed_base": int(args.seed_base),
        "environment_seed": int(args.environment_seed),
        "environment_seed_semantics": "upstream-compatible fixed seed once per task environment",
        "training_dataset_splits": training_config.get("dataset_splits"),
        "heldout_split_sha256": training_config.get("heldout_split_sha256"),
        "episodes": episodes,
    }
    payload["manifest_sha256"] = _canonical_hash(payload)
    write_json(path, payload)
    return payload


def _validate_manifest(manifest: dict[str, Any], selected: tuple[int, int]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported intermediate manifest")
    if manifest.get("evaluation_class") != DIAGNOSTIC_CLASS:
        raise ValueError("manifest is not diagnostic-only")
    if manifest.get("scientific_claim_allowed") is not False:
        raise ValueError("intermediate manifest cannot permit a scientific claim")
    if manifest.get("selected_physical_gpu_ids") != list(selected):
        raise RuntimeError("manifest and SIMVLA_GPU_IDS differ")
    mismatches = {
        name: (os.environ.get(name), value)
        for name, value in manifest["renderer"].items()
        if os.environ.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"renderer contract mismatch: {mismatches}")
    copied = dict(manifest)
    digest = copied.pop("manifest_sha256")
    if _canonical_hash(copied) != digest:
        raise RuntimeError("manifest content hash mismatch")


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    selected = parse_selected_gpu_ids(os.environ.get("SIMVLA_GPU_IDS"))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(map(str, selected)):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must exactly match SIMVLA_GPU_IDS")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("LIBERO-Long evaluation requires exactly two processes")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank:
        raise RuntimeError("single-node task partition requires global rank == local rank")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configure_strict_torch_determinism(int(manifest["seed_base"]))
    _validate_manifest(manifest, selected)
    runtime_source = native_v0_source_manifest(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        cache=args.cache,
    )
    require_source_compatible(
        checkpoint_source=manifest["checkpoint_source_lock"],
        runtime_source=runtime_source,
    )
    require_gate(
        args.parity_gate,
        verdicts=("K1_HOOK_PARITY_PASS",),
        source_combined_sha256=manifest["source_combined_sha256"],
    )
    checkpoint_payload = _load_checkpoint_payload(args.v0_checkpoint)
    if int(checkpoint_payload["global_optimizer_step"]) != int(manifest["checkpoint_step"]):
        raise RuntimeError("checkpoint step differs from frozen diagnostic manifest")
    require_source_compatible(
        checkpoint_source=checkpoint_payload["source_lock"],
        runtime_source=runtime_source,
    )
    adapter = None
    if args.row == "native_v0_k4":
        adapter, loaded = load_native_v0_checkpoint(
            args.v0_checkpoint,
            device=device,
            require_final_150k=False,
        )
        if int(loaded["global_optimizer_step"]) != int(manifest["checkpoint_step"]):
            raise RuntimeError("loaded adapter step differs from manifest")
        freeze_module(adapter)

    output = Path(args.output).expanduser().resolve()
    exists = torch.tensor([int(output.exists())], device=device)
    dist.all_reduce(exists, op=dist.ReduceOp.MAX)
    if int(exists.item()):
        raise FileExistsError(f"refusing existing output: {output}")
    if rank == 0:
        output.mkdir(parents=True)
    dist.barrier()
    shard = output / f"shard_rank{rank}_tasks_{0 if rank == 0 else 5}_{4 if rank == 0 else 9}"
    shard.mkdir()
    write_json(
        shard / "source_lock.json",
        {
            "evaluation_class": DIAGNOSTIC_CLASS,
            "checkpoint_source_lock": checkpoint_payload["source_lock"],
            "runtime_source_lock": runtime_source,
        },
    )
    write_json(
        shard / "eval_contract.json",
        {
            "evaluation_class": DIAGNOSTIC_CLASS,
            "scientific_claim_allowed": False,
            "row": args.row,
            "rank": rank,
            "physical_gpu_id": selected[rank],
            "checkpoint_step": int(manifest["checkpoint_step"]),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
        },
    )
    model, processor, _ = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task_ids = [4, 3, 2, 1, 0] if rank == 0 else [9, 8, 7, 6, 5]
    episode_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    progress = tqdm(total=250, desc=f"{args.row} step{manifest['checkpoint_step']} rank{rank}", dynamic_ncols=True)
    peak_vram = 0
    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        task_manifest = [item for item in manifest["episodes"] if item["task_id"] == task_id]
        env, prompt = get_libero_env(task, 256, int(task_manifest[0]["environment_seed"]))
        try:
            for episode_spec in task_manifest:
                trial_id = int(episode_spec["trial_id"])
                env.reset()
                obs = env.set_init_state(initial_states[int(episode_spec["init_state_index"]) % len(initial_states)])
                for _ in range(10):
                    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                policy = _policy(
                    row=args.row,
                    model=model,
                    processor=processor,
                    adapter=adapter,
                    checkpoint=args.checkpoint,
                    device=device,
                    task_id=task_id,
                    trial_id=trial_id,
                    action_noise_seed_base=int(manifest["action_noise_seed_base"]),
                )
                actions: list[np.ndarray] = []
                synchronized_policy_ms: list[float] = []
                frames: list[np.ndarray] = []
                success = False
                for env_action_index in range(900):
                    if args.save_video and env_action_index % args.video_stride == 0:
                        frames.append(video_frame_from_obs(obs))
                    image0, image1, proprio = build_env_obs(obs)
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    step_output = policy.act(image0, image1, proprio, prompt)
                    torch.cuda.synchronize(device)
                    synchronized_policy_ms.append((time.perf_counter() - started) * 1000.0)
                    obs, _, done, _ = env.step(step_output.action.tolist())
                    actions.append(step_output.action.copy())
                    if done:
                        success = True
                        break
                diagnostics = _trajectory_metrics(actions)
                counters = {key: int(value) for key, value in policy.metrics.counters.items()}
                latencies = {
                    key: [float(value) for value in values]
                    for key, values in policy.metrics.latencies.items()
                }
                append_jsonl(
                    shard / "latency_records.jsonl",
                    {
                        "row": args.row,
                        "task_id": task_id,
                        "trial_id": trial_id,
                        "synchronized_policy_ms_per_executed_action": synchronized_policy_ms,
                        "VLM_encoder_ms": latencies.get("VLM_encoder_ms", []),
                        "condition_updater_ms": latencies.get("condition_updater_ms", []),
                        "action_transformer_ms": latencies.get("action_transformer_ms", []),
                    },
                )
                row = {
                    "row": args.row,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "environment_seed": int(episode_spec["environment_seed"]),
                    "init_state_index": int(episode_spec["init_state_index"]),
                    "success": success,
                    "episode_length": len(actions),
                    "num_policy_queries": counters.get("num_policy_queries", 0),
                    "num_full_vlm_calls": counters.get("num_full_vlm_calls", 0),
                    "num_condition_updater_calls": counters.get("num_condition_updater_calls", 0),
                    "num_action_transformer_flow_iterations": counters.get("num_action_transformer_calls", 0),
                    "num_action_transformer_decodes": counters.get("num_action_transformer_decodes", counters.get("num_policy_queries", 0)),
                    "effective_k": counters.get("num_policy_queries", 0) / max(counters.get("num_full_vlm_calls", 0), 1),
                    "synchronized_policy_ms_p50": float(np.quantile(synchronized_policy_ms, 0.50)),
                    "synchronized_policy_ms_p95": float(np.quantile(synchronized_policy_ms, 0.95)),
                    "latency_per_executed_action_ms": float(np.mean(synchronized_policy_ms)),
                    "vlm_latency_ms_total": float(sum(latencies.get("VLM_encoder_ms", []))),
                    "updater_latency_ms_total": float(sum(latencies.get("condition_updater_ms", []))),
                    "action_transformer_latency_ms_total": float(sum(latencies.get("action_transformer_ms", []))),
                    "normalized_second_difference": diagnostics["normalized_second_difference"],
                    "short_reversal": diagnostics["short_reversal"],
                    "switch_disagreement": diagnostics["switch_disagreement"],
                    "fallback_full_calls": 0,
                }
                episode_rows.append(row)
                traces = getattr(policy, "query_trace", getattr(policy, "action_chunk_records", []))
                for trace in traces:
                    cleaned = {key: value for key, value in trace.items() if not torch.is_tensor(value)}
                    query_rows.append({"row": args.row, "task_id": task_id, "trial_id": trial_id, **cleaned})
                peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated(device)))
                if args.save_video and (not args.video_failures_only or not success):
                    task_video_dir = shard / "videos" / f"task_{task_id:02d}"
                    existing = len(list(task_video_dir.glob("*.mp4")))
                    if existing < args.video_max_per_task:
                        suffix = "success" if success else "failure"
                        save_episode_video(
                            frames,
                            task_video_dir / f"trial_{trial_id:03d}_{suffix}.mp4",
                            10,
                        )
                append_jsonl(shard / "progress.jsonl", {"completed": len(episode_rows), "total": 250, **row})
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
    with (shard / "query_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in query_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "evaluation_class": DIAGNOSTIC_CLASS,
        "scientific_claim_allowed": False,
        "checkpoint_step": int(manifest["checkpoint_step"]),
        "row": args.row,
        "rank": rank,
        "physical_gpu_id": selected[rank],
        "v0_module_parameters": int(adapter.parameter_audit()["total"]) if adapter is not None else 0,
        "tasks": task_ids,
        "episodes": len(episode_rows),
        "successes": sum(int(row["success"]) for row in episode_rows),
        "success_rate": float(np.mean([row["success"] for row in episode_rows])),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_combined_sha256": manifest["source_combined_sha256"],
        "runtime_source_combined_sha256": runtime_source["combined_sha256"],
        "peak_vram_bytes": peak_vram,
        "full_vlm_calls": sum(row["num_full_vlm_calls"] for row in episode_rows),
        "condition_updater_calls": sum(row["num_condition_updater_calls"] for row in episode_rows),
        "fallback_full_calls": 0,
        "query_metrics_jsonl": str(shard / "query_metrics.jsonl"),
    }
    write_json(shard / "shard_summary.json", summary)
    dist.barrier()
    dist.destroy_process_group()
    return summary


def _read_csv(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(int(row["task_id"]), int(row["trial_id"])): row for row in rows}


def compare_rows(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing comparison output: {output}")
    output.mkdir(parents=True)
    baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.v0_summary).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if baseline["row"] != "baseline_k1" or candidate["row"] != "native_v0_k4":
        raise ValueError("comparison requires baseline_k1 and native_v0_k4")
    if baseline["manifest_sha256"] != candidate["manifest_sha256"]:
        raise RuntimeError("rows do not share the same manifest")
    if candidate["manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("row summaries do not match the supplied manifest")
    baseline_rows = _read_csv(Path(baseline["episode_metrics_csv"]))
    candidate_rows = _read_csv(Path(candidate["episode_metrics_csv"]))
    if set(baseline_rows) != set(candidate_rows) or len(baseline_rows) != 500:
        raise RuntimeError("rows are not paired over exactly 500 episodes")
    flips = {
        "baseline_fail_v0_success": 0,
        "baseline_success_v0_fail": 0,
        "both_success": 0,
        "both_fail": 0,
    }
    for key in baseline_rows:
        baseline_success = baseline_rows[key]["success"].lower() in {"true", "1", "yes"}
        candidate_success = candidate_rows[key]["success"].lower() in {"true", "1", "yes"}
        if baseline_success and candidate_success:
            flips["both_success"] += 1
        elif baseline_success:
            flips["baseline_success_v0_fail"] += 1
        elif candidate_success:
            flips["baseline_fail_v0_success"] += 1
        else:
            flips["both_fail"] += 1
    result = {
        "verdict": "INTERMEDIATE_DIAGNOSTIC_COMPLETE",
        "evaluation_class": DIAGNOSTIC_CLASS,
        "scientific_claim_allowed": False,
        "paper_table_eligible": False,
        "final_150k_evaluation_still_required": True,
        "checkpoint_step": int(manifest["checkpoint_step"]),
        "checkpoint_path": manifest["checkpoint_path"],
        "norm_stats_sha256": manifest["norm_stats_sha256"],
        "manifest_sha256": candidate["manifest_sha256"],
        "baseline": {
            "successes": baseline["successes"],
            "episodes": baseline["episodes"],
            "success_rate": baseline["success_rate"],
            "full_vlm_calls": baseline["full_vlm_calls"],
            "latency": baseline["latency"],
        },
        "native_v0_k4": {
            "successes": candidate["successes"],
            "episodes": candidate["episodes"],
            "success_rate": candidate["success_rate"],
            "full_vlm_calls": candidate["full_vlm_calls"],
            "condition_updater_calls": candidate["condition_updater_calls"],
            "latency": candidate["latency"],
        },
        "success_rate_delta_percentage_points": 100.0 * (candidate["success_rate"] - baseline["success_rate"]),
        "full_vlm_call_reduction_fraction": 1.0 - candidate["full_vlm_calls"] / max(baseline["full_vlm_calls"], 1),
        "paired_flips": flips,
    }
    write_json(output / "intermediate_comparison.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--cache", required=True)
    manifest.add_argument("--v0-checkpoint", required=True)
    manifest.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    manifest.add_argument("--norm-stats", required=True)
    manifest.add_argument("--parity-gate", required=True)
    manifest.add_argument("--seed-base", type=int, default=20260815)
    manifest.add_argument("--environment-seed", type=int, default=7)
    manifest.add_argument("--action-noise-seed-base", type=int, default=6828326409295398833)
    manifest.set_defaults(handler=create_manifest)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--row", choices=("baseline_k1", "native_v0_k4"), required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--cache", required=True)
    evaluate.add_argument("--v0-checkpoint", required=True)
    evaluate.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    evaluate.add_argument("--norm-stats", required=True)
    evaluate.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    evaluate.add_argument("--parity-gate", required=True)
    evaluate.add_argument("--save-video", action="store_true")
    evaluate.add_argument("--video-failures-only", action="store_true")
    evaluate.add_argument("--video-stride", type=int, default=2)
    evaluate.add_argument("--video-max-per-task", type=int, default=2)
    evaluate.set_defaults(handler=run_eval)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--output", required=True)
    compare.add_argument("--manifest", required=True)
    compare.add_argument("--baseline-summary", required=True)
    compare.add_argument("--v0-summary", required=True)
    compare.set_defaults(handler=compare_rows)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.handler(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
