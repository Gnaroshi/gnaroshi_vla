"""Paired baseline/Latent Bridge evaluation on the original SimVLA protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    RealSimVLADCLDPolicy,
    build_env_obs,
    get_libero_env,
    save_episode_video,
    video_frame_from_obs,
)

from .checkpoint import load_bridge_checkpoint
from .dataset import DAGGER_SCHEMA, sha256_file
from .policy import RealSimVLALatentBridgePolicy
from .provenance import (
    latent_bridge_source_manifest,
    simvla_latent_bridge_integration_manifest,
)
from .recipe import EVALUATION_ROWS, evaluation_row, scientific_contract


def _configure_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[4]
    upstream = Path(
        os.environ.get("SIMVLA_UPSTREAM_ROOT", root / "architectures/simvla/upstream")
    ).expanduser().resolve()
    libero = Path(
        os.environ.get(
            "LIBERO_ROOT", upstream / "evaluation/libero/LIBERO"
        )
    ).expanduser().resolve()
    for path in (root, upstream, libero):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if not (upstream / "models/modeling_smolvlm_vla.py").is_file():
        raise FileNotFoundError(f"SimVLA upstream not found: {upstream}")
    if not (libero / "libero").is_dir():
        raise FileNotFoundError(f"LIBERO root not found: {libero}")
    return upstream, libero


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class SynchronizedFullPolicy(RealSimVLADCLDPolicy):
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
        action, seed = super()._decode(
            condition, proprio, policy_query_index=policy_query_index
        )
        self._sync()
        # Replace the unsynchronized parent sample with the synchronized value.
        self.metrics.latencies["action_transformer_ms"][-1] = (
            time.perf_counter() - started
        ) * 1000.0
        self.metrics.counters["num_action_transformer_decodes"] += 1
        return action, seed

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
        self.metrics.latencies["VLM_encoder_ms"].append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_full_vlm_calls"] += 1
        action, seed = self._decode(
            condition, batch["proprio"], policy_query_index=policy_query_index
        )
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action.detach()
        return condition, action, seed


def _load_simvla(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device).eval()
    model.action_space.load_norm_stats(args.norm_stats)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model)
    return model, processor


def _make_policy(
    row: str,
    *,
    model: Any,
    processor: Any,
    bridge: Any,
    args: argparse.Namespace,
    device: torch.device,
    task_id: int,
    trial_id: int,
) -> Any:
    row_contract = evaluation_row(row)
    common = dict(
        model=model,
        processor=processor,
        flow_steps=10,
        image_size=384,
        replan_steps=5,
        client_resize_size=224,
        device=device,
        suite=args.suite,
        row_name=row,
        task_id=task_id,
        trial_id=trial_id,
        paired_action_noise=True,
        action_noise_seed_base=args.action_noise_seed_base,
        log_action_chunks=False,
    )
    if not row_contract.uses_bridge:
        return SynchronizedFullPolicy(
            dcld_core=None, mode="full", refresh_every=1, **common
        )
    if bridge is None:
        raise RuntimeError("Latent Bridge row requires a bridge checkpoint")
    return RealSimVLALatentBridgePolicy(
        bridge=bridge,
        refresh_every=row_contract.refresh_every,
        collect_dagger_teacher=args.collect_dagger_teacher,
        **common,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    supported_rows = set(EVALUATION_ROWS)
    unknown_rows = set(args.rows) - supported_rows
    if unknown_rows:
        raise ValueError(f"unsupported rows: {sorted(unknown_rows)}")
    if args.trial_offset < 0 or args.num_trials < 1:
        raise ValueError("trial_offset must be non-negative and num_trials positive")
    if args.collect_dagger_teacher and args.rows != ["latent_bridge_f3"]:
        raise ValueError(
            "paper feature-bridge DAgger collection must use only latent_bridge_f3"
        )
    if args.collect_dagger_teacher and args.compile_bridge:
        raise ValueError("DAgger collection uses the uncompiled bridge training path")
    if args.environment_seed is None:
        args.environment_seed = args.seed
    if args.action_noise_seed_base is None:
        args.action_noise_seed_base = args.seed
    upstream, libero = _configure_paths()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing evaluation output: {output}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    model, processor = _load_simvla(args, device)
    bridge = None
    bridge_payload = None
    bridge_parameter_audit = None
    if any(evaluation_row(row).uses_bridge for row in args.rows):
        if not args.bridge_checkpoint:
            raise ValueError("--bridge-checkpoint is required for Latent Bridge rows")
        bridge, bridge_payload = load_bridge_checkpoint(
            args.bridge_checkpoint, device=device
        )
        bridge_parameter_audit = bridge.parameter_audit()
        if args.bridge_precision == "bf16":
            if device.type != "cuda":
                raise ValueError("bf16 bridge evaluation requires CUDA")
            bridge = bridge.to(dtype=torch.bfloat16)
        bridge.eval()
        for parameter in bridge.parameters():
            parameter.requires_grad_(False)
        checkpoint_identity = bridge_payload["provenance"].get(
            "training_data_identity", {}
        )
        if checkpoint_identity.get("checkpoint") != args.checkpoint:
            raise RuntimeError("bridge training checkpoint and evaluation base checkpoint differ")
        current_norm_hash = sha256_file(args.norm_stats)
        if checkpoint_identity.get("norm_stats_sha256") != current_norm_hash:
            raise RuntimeError("bridge training and evaluation norm stats differ")
        if args.compile_bridge:
            bridge = torch.compile(bridge, mode="max-autotune")
            sequence_length = bridge.config.sequence_length
            feature_dim = bridge.config.feature_dim
            bridge_dtype = next(bridge.parameters()).dtype
            with torch.inference_mode():
                bridge.predict_next(
                    torch.zeros(
                        1, sequence_length, feature_dim, device=device, dtype=bridge_dtype
                    ),
                    torch.zeros(
                        1, sequence_length, feature_dim, device=device, dtype=bridge_dtype
                    ),
                    torch.zeros(1, bridge.config.state_dim, device=device, dtype=bridge_dtype),
                    torch.zeros(1, bridge.config.action_dim, device=device, dtype=bridge_dtype),
                )
            torch.cuda.synchronize(device)
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_count = min(
        int(suite.get_num_tasks()), args.max_tasks or int(suite.get_num_tasks())
    )
    task_ids = list(reversed(range(task_count)))
    metadata = {
        "checkpoint": args.checkpoint,
        "norm_stats": str(Path(args.norm_stats).expanduser().resolve()),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "bridge_checkpoint": (
            str(Path(args.bridge_checkpoint).resolve())
            if args.bridge_checkpoint
            else None
        ),
        "bridge_parameter_audit": bridge_parameter_audit,
        "latent_bridge_source": latent_bridge_source_manifest(),
        "simvla_latent_bridge_integration": simvla_latent_bridge_integration_manifest(),
        "simvla_upstream_root": str(upstream),
        "libero_root": str(libero),
        "suite": args.suite,
        "task_ids": task_ids,
        "trials_per_task": args.num_trials,
        "trial_offset": args.trial_offset,
        "rows": args.rows,
        "row_contracts": {
            row: evaluation_row(row).serializable() for row in args.rows
        },
        "action_horizon": 10,
        "execution_horizon": 5,
        "flow_steps": 10,
        "evaluation_seed": args.seed,
        "client_resize_size": 224,
        "num_wait_steps": 10,
        "max_policy_actions": args.max_policy_steps,
        "action_noise_seed_base": args.action_noise_seed_base,
        "environment_seed": args.environment_seed,
        "renderer": {
            key: os.environ.get(key)
            for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "EGL_DEVICE_ID")
        },
        "runtime_stack": {
            "bridge_precision": args.bridge_precision,
            "bridge_compile": bool(args.compile_bridge),
            "bridge_compile_mode": "max-autotune" if args.compile_bridge else None,
            "comparison_axis": (
                "official_optimized_secondary"
                if args.compile_bridge
                else "matched_eager_primary"
            ),
        },
        "comparison_label": (
            "official-algorithm SimVLA adaptation; not official Latent Bridge "
            "SimVLA code"
        ),
        "scientific_contract": scientific_contract(),
        "bridge_checkpoint_provenance": (
            bridge_payload.get("provenance") if bridge_payload is not None else None
        ),
    }
    _write_json(output / "environment_metadata.json", metadata)
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for row_name in args.rows:
        row_contract = evaluation_row(row_name)
        row_output = output / row_name
        row_output.mkdir()
        episode_rows: list[dict[str, Any]] = []
        dagger_entries: list[dict[str, Any]] = []
        total_episodes = task_count * args.num_trials
        progress = tqdm(total=total_episodes, desc=row_name, dynamic_ncols=True)
        for task_id in task_ids:
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            env, prompt = get_libero_env(task, 256, args.environment_seed)
            try:
                for local_trial_id in range(args.num_trials):
                    trial_id = args.trial_offset + local_trial_id
                    env.reset()
                    obs = env.set_init_state(init_states[trial_id % len(init_states)])
                    for _ in range(10):
                        obs, _, done, _ = env.step([0.0] * 6 + [-1.0])
                    policy = _make_policy(
                        row_name,
                        model=model,
                        processor=processor,
                        bridge=bridge,
                        args=args,
                        device=device,
                        task_id=task_id,
                        trial_id=trial_id,
                    )
                    frames: list[np.ndarray] = []
                    success = False
                    policy_ms: list[float] = []
                    try:
                        for action_index in range(args.max_policy_steps):
                            if args.save_video and action_index % args.video_stride == 0:
                                frames.append(video_frame_from_obs(obs))
                            image0, image1, proprio = build_env_obs(obs)
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            started = time.perf_counter()
                            action = policy.act(image0, image1, proprio, prompt)
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                            policy_ms.append(
                                (time.perf_counter() - started) * 1000.0
                            )
                            obs, _, done, _ = env.step(action.action.tolist())
                            if done:
                                success = True
                                break
                    finally:
                        if hasattr(policy, "close"):
                            policy.close()
                    counters = dict(policy.metrics.counters)
                    latencies = policy.metrics.latencies
                    episode = {
                        "row": row_name,
                        "task_id": task_id,
                        "trial_id": trial_id,
                        "init_state_index": trial_id % len(init_states),
                        "environment_seed": args.environment_seed,
                        "success": success,
                        "episode_length": len(policy_ms),
                        "num_policy_queries": counters.get(
                            "num_policy_queries", 0
                        ),
                        "num_full_vlm_calls": counters.get(
                            "num_full_vlm_calls", 0
                        ),
                        "num_condition_updater_calls": counters.get(
                            "num_condition_updater_calls", 0
                        ),
                        "num_action_transformer_calls": counters.get(
                            "num_action_transformer_calls", 0
                        ),
                        "latency_per_executed_action_ms": float(np.mean(policy_ms)),
                        "policy_latency_p50_ms": float(
                            np.quantile(policy_ms, 0.5)
                        ),
                        "policy_latency_p95_ms": float(
                            np.quantile(policy_ms, 0.95)
                        ),
                        "vlm_latency_total_ms": float(
                            sum(latencies.get("VLM_encoder_ms", []))
                        ),
                        "bridge_latency_total_ms": float(
                            sum(latencies.get("condition_updater_ms", []))
                        ),
                        "action_latency_total_ms": float(
                            sum(latencies.get("action_transformer_ms", []))
                        ),
                    }
                    episode_rows.append(episode)
                    all_rows.append(episode)
                    _append_jsonl(row_output / "progress.jsonl", episode)
                    if args.collect_dagger_teacher and row_contract.uses_bridge:
                        dagger_dir = row_output / "dagger"
                        dagger_dir.mkdir(exist_ok=True)
                        dagger_path = dagger_dir / f"task{task_id:02d}_trial{trial_id:03d}.pt"
                        torch.save(
                            {
                                "schema_version": DAGGER_SCHEMA,
                                "task_id": task_id,
                                "trial_id": trial_id,
                                "transitions": policy.dagger_transitions,
                            },
                            dagger_path,
                        )
                        dagger_entries.append(
                            {
                                "file": dagger_path.name,
                                "sha256": sha256_file(dagger_path),
                                "size_bytes": dagger_path.stat().st_size,
                                "task_id": task_id,
                                "trial_id": trial_id,
                                "transitions": len(policy.dagger_transitions),
                            }
                        )
                    if args.save_video and (
                        not args.video_failures_only or not success
                    ):
                        save_episode_video(
                            frames,
                            row_output
                            / "videos"
                            / (
                                f"task{task_id:02d}_trial{trial_id:03d}_"
                                f"{'success' if success else 'failure'}.mp4"
                            ),
                            10,
                        )
                    progress.update(1)
                    successes = sum(int(item["success"]) for item in episode_rows)
                    progress.set_postfix(sr=f"{100 * successes / len(episode_rows):.1f}%")
            finally:
                env.close()
        progress.close()
        with (row_output / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
            writer.writeheader()
            writer.writerows(episode_rows)
        summaries[row_name] = {
            "episodes": len(episode_rows),
            "successes": sum(int(item["success"]) for item in episode_rows),
            "success_rate": float(np.mean([item["success"] for item in episode_rows])),
            "latency_per_executed_action_ms": float(
                np.mean([item["latency_per_executed_action_ms"] for item in episode_rows])
            ),
            "full_vlm_calls": sum(item["num_full_vlm_calls"] for item in episode_rows),
            "bridge_calls": sum(item["num_condition_updater_calls"] for item in episode_rows),
            "refresh_every": row_contract.refresh_every,
            "expected_full_vlm_call_saving": row_contract.full_vlm_call_saving,
        }
        if dagger_entries:
            _write_json(
                row_output / "dagger" / "manifest.json",
                {
                    "schema_version": DAGGER_SCHEMA,
                    "row": row_name,
                    "base_checkpoint": args.checkpoint,
                    "norm_stats_sha256": sha256_file(args.norm_stats),
                    "bridge_checkpoint": str(Path(args.bridge_checkpoint).resolve()),
                    "bridge_checkpoint_sha256": sha256_file(args.bridge_checkpoint),
                    "bridge_config": bridge.config.serializable(),
                    "stable_layer_index": bridge.config.stable_layer_index,
                    "token_mode": bridge.config.token_mode,
                    "latent_bridge_upstream": latent_bridge_source_manifest(),
                    "simvla_latent_bridge_integration": (
                        simvla_latent_bridge_integration_manifest()
                    ),
                    "suite": args.suite,
                    "trial_offset": args.trial_offset,
                    "trials_per_task": args.num_trials,
                    "episodes": len(dagger_entries),
                    "refresh_every": row_contract.refresh_every,
                    "action_horizon": 10,
                    "execution_horizon": 5,
                    "flow_steps": 10,
                    "environment_seed": args.environment_seed,
                    "action_noise_seed_base": args.action_noise_seed_base,
                    "teacher_semantics": (
                        "full frozen SimVLA condition on the current observation; "
                        "rollout actions use the recursive bridge prediction"
                    ),
                    "shards": dagger_entries,
                    "total_transitions": sum(
                        int(entry["transitions"]) for entry in dagger_entries
                    ),
                },
            )
        _write_json(row_output / "summary.json", summaries[row_name])
    if "baseline_k1" in summaries:
        baseline_ms = summaries["baseline_k1"]["latency_per_executed_action_ms"]
        baseline_calls = summaries["baseline_k1"]["full_vlm_calls"]
        for row_name, summary in summaries.items():
            summary["speedup_vs_baseline"] = baseline_ms / summary["latency_per_executed_action_ms"]
            summary["observed_full_vlm_call_saving"] = (
                1.0 - summary["full_vlm_calls"] / baseline_calls
                if baseline_calls
                else None
            )
    result = {
        "verdict": "SIMVLA_LATENT_BRIDGE_EVAL_COMPLETE",
        "summaries": summaries,
        "paired_episode_identity": "suite+task_id+trial_id+init_state_index+environment_seed",
        "paired_action_noise": True,
    }
    _write_json(output / "comparison_summary.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", required=True)
    value.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    value.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    value.add_argument("--norm-stats", required=True)
    value.add_argument("--bridge-checkpoint")
    value.add_argument(
        "--rows",
        nargs="+",
        default=["baseline_k1", "latent_bridge_f3", "latent_bridge_f4"],
    )
    value.add_argument(
        "--suite",
        choices=("libero_10", "libero_spatial", "libero_object", "libero_goal"),
        default="libero_10",
    )
    value.add_argument("--num-trials", type=int, default=20)
    value.add_argument("--trial-offset", type=int, default=0)
    value.add_argument("--max-tasks", type=int)
    value.add_argument("--max-policy-steps", type=int, default=900)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--action-noise-seed-base", type=int)
    value.add_argument("--environment-seed", type=int)
    value.add_argument("--collect-dagger-teacher", action="store_true")
    value.add_argument("--bridge-precision", choices=("fp32", "bf16"), default="bf16")
    value.add_argument("--compile-bridge", action="store_true")
    value.add_argument("--save-video", action="store_true")
    value.add_argument("--video-stride", type=int, default=2)
    value.add_argument("--video-failures-only", action="store_true")
    value.add_argument("--device", default="cuda")
    return value


def main() -> None:
    result = evaluate(parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
