"""Manifest-locked 500-episode LIBERO-Long evaluation for SimVLA V0."""

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

from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
from architectures.simvla.adapters.latentloop.native_v0_policy import RealSimVLANativeV0Policy
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
    RealSimVLADCLDPolicy,
    build_env_obs,
    get_libero_env,
    save_episode_video,
    video_frame_from_obs,
)
from architectures.simvla.wrappers.simvla_two_gpu_guard import parse_selected_gpu_ids


MANIFEST_SCHEMA = "simvla_native_v0_libero_long_500_v1"


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _trajectory_metrics(actions: list[np.ndarray]) -> dict[str, float]:
    tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
    if tensor.shape[0] < 2:
        return {
            "normalized_second_difference": 0.0,
            "short_reversal": 0.0,
            "switch_disagreement": 0.0,
        }
    first = tensor[1:, :6] - tensor[:-1, :6]
    first_scale = torch.linalg.vector_norm(first, dim=-1).mean().clamp_min(1e-8)
    if tensor.shape[0] >= 3:
        second = tensor[2:, :6] - 2 * tensor[1:-1, :6] + tensor[:-2, :6]
        normalized_second = torch.linalg.vector_norm(second, dim=-1).mean() / first_scale
        consecutive = first[1:] * first[:-1]
        reversal = (consecutive.sum(dim=-1) < 0).float().mean()
    else:
        normalized_second = tensor.new_zeros(())
        reversal = tensor.new_zeros(())
    switches = ((tensor[1:, 6] >= 0) != (tensor[:-1, 6] >= 0)).float().mean()
    return {
        "normalized_second_difference": float(normalized_second.item()),
        "short_reversal": float(reversal.item()),
        "switch_disagreement": float(switches.item()),
    }


def create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    selected = parse_selected_gpu_ids(os.environ.get("SIMVLA_GPU_IDS"))
    path = Path(args.output).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing existing manifest: {path}")
    source = native_v0_source_manifest(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        cache=args.cache,
    )
    require_gate(
        args.parity_gate,
        verdicts=("K1_HOOK_PARITY_PASS",),
        source_combined_sha256=source["combined_sha256"],
    )
    offline_gate = require_gate(
        args.offline_gate,
        verdicts=("OFFLINE_K4_GATE_PASS",),
        source_combined_sha256=source["combined_sha256"],
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
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "source_combined_sha256": source["combined_sha256"],
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
        "renderer": {
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
        },
        "checkpoint": args.checkpoint,
        "norm_stats": str(Path(args.norm_stats).resolve()),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "action_noise_seed_base": args.action_noise_seed_base,
        "seed_base": args.seed_base,
        "environment_seed": int(args.environment_seed),
        "environment_seed_semantics": "upstream-compatible fixed seed once per task environment",
        "training_dataset_splits": offline_gate.get("dataset_splits"),
        "heldout_split_sha256": offline_gate.get("heldout_split_sha256"),
        "episodes": episodes,
        "previous_100_episode_rows_are_scientific": False,
    }
    payload["manifest_sha256"] = _canonical_hash(payload)
    write_json(path, payload)
    return payload


def audit_baseline_reuse(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    reusable = False
    reasons: list[str] = []
    if args.candidate_summary:
        candidate_path = Path(args.candidate_summary).expanduser().resolve()
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        reusable = bool(
            candidate.get("row") == "baseline_k1"
            and candidate.get("manifest_sha256") == manifest["manifest_sha256"]
            and candidate.get("episodes") == 500
            and candidate.get("successes") is not None
        )
        if not reusable:
            reasons.append("candidate summary does not match the immutable 500-episode manifest")
    else:
        reasons.append("no candidate baseline summary was supplied")
    result = {
        "verdict": "BASELINE_500_REUSABLE" if reusable else "BASELINE_500_RERUN_REQUIRED",
        "manifest_sha256": manifest["manifest_sha256"],
        "source_combined_sha256": manifest["source_combined_sha256"],
        "candidate_summary": str(Path(args.candidate_summary).resolve()) if args.candidate_summary else None,
        "reasons": reasons,
        "note": "The separate four-suite 2,000-episode released-checkpoint reproduction has no identical paired manifest and is not reused for this comparison.",
    }
    write_json(output / "baseline_reuse_audit.json", result)
    return result


class _SynchronizedFullPolicy(RealSimVLADCLDPolicy):
    """K1 baseline with explicit noise and synchronized latency boundaries."""

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _decode(
        self,
        condition: torch.Tensor,
        proprio: torch.Tensor,
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, int | None]:
        self._sync()
        started = time.perf_counter()
        initial_noise, seed = self._paired_initial_noise(condition, proprio, policy_query_index)
        decoded = self.action_adapter.decode_action_from_condition(
            condition,
            proprio,
            steps=10,
            initial_noise=initial_noise,
            return_debug=True,
        )
        self._sync()
        self.metrics.latencies["action_transformer_ms"].append((time.perf_counter() - started) * 1000.0)
        self.metrics.counters["num_action_transformer_calls"] += int(decoded.debug["iterations"])
        self.metrics.counters["num_action_transformer_decodes"] += 1
        return decoded.action, seed

    def _full_refresh(
        self,
        batch: dict[str, torch.Tensor],
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        self._sync()
        started = time.perf_counter()
        condition = self.condition_adapter.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
        self._sync()
        self.metrics.latencies["VLM_encoder_ms"].append((time.perf_counter() - started) * 1000.0)
        self.metrics.counters["num_full_vlm_calls"] += 1
        action, seed = self._decode(condition, batch["proprio"], policy_query_index=policy_query_index)
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action.detach()
        return condition, action, seed


def _validate_manifest_runtime(manifest: dict[str, Any], selected: tuple[int, int]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported LIBERO-Long manifest")
    if manifest.get("selected_physical_gpu_ids") != list(selected):
        raise RuntimeError("manifest and SIMVLA_GPU_IDS differ")
    expected_renderer = manifest["renderer"]
    mismatches = {
        name: (os.environ.get(name), value)
        for name, value in expected_renderer.items()
        if os.environ.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"renderer contract mismatch: {mismatches}")
    copied = dict(manifest)
    digest = copied.pop("manifest_sha256")
    if _canonical_hash(copied) != digest:
        raise RuntimeError("manifest content hash mismatch")


def _policy(
    *,
    row: str,
    model: Any,
    processor: Any,
    adapter: Any,
    checkpoint: str,
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
            log_action_chunks=True,
        )
    return RealSimVLANativeV0Policy(
        model=model,
        processor=processor,
        adapter=adapter,
        checkpoint_id=checkpoint,
        device=device,
        suite="libero_10",
        task_id=task_id,
        trial_id=trial_id,
        action_noise_seed_base=action_noise_seed_base,
        log_action_chunks=True,
    )


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
    _validate_manifest_runtime(manifest, selected)
    source = native_v0_source_manifest(checkpoint=args.checkpoint, norm_stats=args.norm_stats, cache=args.cache)
    if source["combined_sha256"] != manifest["source_combined_sha256"]:
        raise RuntimeError("current source lock differs from the evaluation manifest")
    require_gate(args.parity_gate, verdicts=("K1_HOOK_PARITY_PASS",), source_combined_sha256=source["combined_sha256"])
    adapter = None
    if args.row == "baseline_k1":
        baseline_audit = require_gate(
            args.baseline_reuse_audit,
            verdicts=("BASELINE_500_RERUN_REQUIRED", "BASELINE_500_REUSABLE"),
            source_combined_sha256=source["combined_sha256"],
        )
        if baseline_audit["verdict"] == "BASELINE_500_REUSABLE":
            raise RuntimeError("matching baseline already exists; reuse it instead of rerunning")
    if args.row == "native_v0_k4":
        offline_gate = require_gate(args.offline_gate, verdicts=("OFFLINE_K4_GATE_PASS",), source_combined_sha256=source["combined_sha256"])
        if offline_gate.get("dataset_splits") != manifest.get("training_dataset_splits"):
            raise RuntimeError("offline gate and Long manifest use different training splits")
        baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
        if baseline.get("row") != "baseline_k1" or baseline.get("episodes") != 500 or baseline.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RuntimeError("V0 evaluation requires the complete matching 500-episode baseline")
        adapter, checkpoint_payload = load_native_v0_checkpoint(args.v0_checkpoint, device=device, require_final_150k=True)
        if checkpoint_payload["source_lock"]["combined_sha256"] != source["combined_sha256"]:
            raise RuntimeError("V0 checkpoint source lock differs from manifest")
        if checkpoint_payload["training_config"].get("dataset_splits") != manifest.get("training_dataset_splits"):
            raise RuntimeError("V0 checkpoint and Long manifest use different training splits")
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
    write_json(shard / "source_lock.json", source)
    write_json(shard / "eval_contract.json", {"row": args.row, "rank": rank, "physical_gpu_id": selected[rank], "manifest": str(manifest_path), "manifest_sha256": manifest["manifest_sha256"]})

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
    total = 250
    progress = tqdm(total=total, desc=f"{args.row} rank{rank}", dynamic_ncols=True)
    peak_vram = 0
    for task_id in task_ids:
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        task_manifest = [item for item in manifest["episodes"] if item["task_id"] == task_id]
        env, prompt = get_libero_env(task, 256, int(task_manifest[0]["environment_seed"]))
        try:
            for episode_spec in task_manifest:
                trial_id = int(episode_spec["trial_id"])
                episode_seed = int(episode_spec["environment_seed"])
                env.reset()
                obs = env.set_init_state(initial_states[int(episode_spec["init_state_index"]) % len(initial_states)])
                for _ in range(10):
                    obs, _, done, _ = env.step([0.0] * 6 + [-1.0])
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
                latencies = {key: [float(value) for value in values] for key, values in policy.metrics.latencies.items()}
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
                    "environment_seed": episode_seed,
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
                    existing = len(list((shard / "videos" / f"task_{task_id:02d}").glob("*.mp4")))
                    if existing < args.video_max_per_task:
                        suffix = "success" if success else "failure"
                        save_episode_video(frames, shard / "videos" / f"task_{task_id:02d}" / f"trial_{trial_id:03d}_{suffix}.mp4", 10)
                append_jsonl(shard / "progress.jsonl", {"completed": len(episode_rows), "total": total, **row})
                progress.update(1)
                progress.set_postfix(successes=sum(int(item["success"]) for item in episode_rows), sr=f"{np.mean([item['success'] for item in episode_rows])*100:.1f}%")
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
        "row": args.row,
        "rank": rank,
        "physical_gpu_id": selected[rank],
        "v0_module_parameters": (
            int(adapter.parameter_audit()["total"]) if adapter is not None else 0
        ),
        "tasks": task_ids,
        "episodes": len(episode_rows),
        "successes": sum(int(row["success"]) for row in episode_rows),
        "success_rate": float(np.mean([row["success"] for row in episode_rows])),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_combined_sha256": source["combined_sha256"],
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--cache", required=True)
    manifest.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    manifest.add_argument("--norm-stats", required=True)
    manifest.add_argument("--parity-gate", required=True)
    manifest.add_argument("--offline-gate", required=True)
    manifest.add_argument("--seed-base", type=int, default=20260815)
    manifest.add_argument("--environment-seed", type=int, default=7)
    manifest.add_argument("--action-noise-seed-base", type=int, default=6828326409295398833)
    manifest.set_defaults(handler=create_manifest)
    audit = subparsers.add_parser("audit-baseline")
    audit.add_argument("--output", required=True)
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--candidate-summary", default="")
    audit.set_defaults(handler=audit_baseline_reuse)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--row", choices=("baseline_k1", "native_v0_k4"), required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--cache", required=True)
    evaluate.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    evaluate.add_argument("--norm-stats", required=True)
    evaluate.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    evaluate.add_argument("--parity-gate", required=True)
    evaluate.add_argument("--baseline-reuse-audit", default="")
    evaluate.add_argument("--offline-gate", default="")
    evaluate.add_argument("--v0-checkpoint", default="")
    evaluate.add_argument("--baseline-summary", default="")
    evaluate.add_argument("--save-video", action="store_true")
    evaluate.add_argument("--video-failures-only", action="store_true")
    evaluate.add_argument("--video-stride", type=int, default=2)
    evaluate.add_argument("--video-max-per-task", type=int, default=2)
    evaluate.set_defaults(handler=run_eval)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.handler(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
