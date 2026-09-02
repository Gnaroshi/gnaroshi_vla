"""Paired original-protocol evaluation for frozen SimVLA and FastV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
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
    build_env_obs,
    get_libero_env,
    save_episode_video,
    video_frame_from_obs,
)

from .encoder import FastVForwardConfig
from .policy import FastVPolicy, SynchronizedBaselinePolicy
from .provenance import (
    fastv_source_manifest,
    sha256_file,
    simvla_fastv_integration_manifest,
)
from .recipe import EVALUATION_ROWS, evaluation_row, scientific_contract


ROOT = Path(__file__).resolve().parents[4]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _configure_paths() -> tuple[Path, Path]:
    upstream = Path(
        os.environ.get("SIMVLA_UPSTREAM_ROOT", ROOT / "architectures/simvla/upstream")
    ).expanduser().resolve()
    libero = Path(
        os.environ.get("LIBERO_ROOT", upstream / "evaluation/libero/LIBERO")
    ).expanduser().resolve()
    for path in (ROOT, upstream, libero):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if not (upstream / "models/modeling_smolvlm_vla.py").is_file():
        raise FileNotFoundError(f"SimVLA upstream not found: {upstream}")
    if not (libero / "libero").is_dir():
        raise FileNotFoundError(f"LIBERO root not found: {libero}")
    return upstream, libero


def _load_simvla(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device).eval()
    model.action_space.load_norm_stats(args.norm_stats)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model)
    return model, processor


def _policy(
    row_name: str,
    *,
    model: Any,
    processor: Any,
    args: argparse.Namespace,
    device: torch.device,
    task_id: int,
    trial_id: int,
) -> Any:
    row = evaluation_row(row_name)
    common = dict(
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
        suite=args.suite,
        row_name=row_name,
        task_id=task_id,
        trial_id=trial_id,
        paired_action_noise=True,
        action_noise_seed_base=args.action_noise_seed_base,
        log_action_chunks=False,
    )
    if not row.uses_fastv:
        return SynchronizedBaselinePolicy(**common)
    return FastVPolicy(
        fastv_config=FastVForwardConfig(
            prune_layer=int(row.prune_layer),
            prune_ratio=row.prune_ratio,
            score_mode=str(row.score_mode),
            restore_mode=str(row.restore_mode),
        ),
        **common,
    )


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def _fastv_episode_debug(policy: Any) -> dict[str, Any] | None:
    records = getattr(policy, "fastv_debug_records", None)
    if not records:
        return None
    return {
        "queries": len(records),
        "full_sequence_length": records[0]["full_sequence_length"],
        "compact_sequence_length": records[0]["compact_sequence_length"],
        "visual_tokens_before": records[0]["visual_tokens_before"],
        "visual_tokens_kept": records[0]["visual_tokens_kept"],
        "visual_tokens_pruned": records[0]["visual_tokens_pruned"],
        "nonvisual_tokens_kept": records[0]["nonvisual_tokens_kept"],
        "paper_equation_text_model_reduction": records[0][
            "paper_equation_text_model_reduction"
        ],
        "swiglu_aware_text_model_reduction": records[0][
            "swiglu_aware_text_model_reduction"
        ],
        "first_visual_keep_indices": records[0]["visual_keep_indices"],
        "last_visual_keep_indices": records[-1]["visual_keep_indices"],
        "unique_score_hashes": len(
            {record["visual_scores_sha256"] for record in records}
        ),
    }


def _environment_metadata(
    args: argparse.Namespace,
    *,
    upstream: Path,
    libero: Path,
) -> dict[str, Any]:
    try:
        import mujoco
    except Exception:
        mujoco = None
    try:
        import transformers
    except Exception:
        transformers = None
    return {
        "hostname": platform.node(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "transformers_version": getattr(transformers, "__version__", None),
        "mujoco_version": getattr(mujoco, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.cuda.is_available()
            else None
        ),
        "renderer": {
            key: os.environ.get(key)
            for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID")
        },
        "checkpoint": args.checkpoint,
        "norm_stats": str(Path(args.norm_stats).expanduser().resolve()),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "simvla_upstream_root": str(upstream),
        "libero_root": str(libero),
        "suite": args.suite,
        "rows": args.rows,
        "num_trials": args.num_trials,
        "trial_offset": args.trial_offset,
        "max_tasks": args.max_tasks,
        "environment_seed": args.environment_seed,
        "action_noise_seed_base": args.action_noise_seed_base,
        "action_horizon": 10,
        "execution_horizon": 5,
        "flow_steps": 10,
        "client_resize_size": 224,
        "num_wait_steps": 10,
        "max_policy_steps": args.max_policy_steps,
        "comparison_label": (
            "official FastV algorithm adapted to SimVLA; not official FastV SimVLA code"
        ),
        "fastv_source": fastv_source_manifest(),
        "simvla_fastv_integration": simvla_fastv_integration_manifest(),
        "scientific_contract": scientific_contract(),
    }


def _write_contract_only(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "verdict": "SIMVLA_FASTV_CONTRACT_PASS",
        "fastv_source": fastv_source_manifest(),
        "simvla_fastv_integration": simvla_fastv_integration_manifest(),
        "scientific_contract": scientific_contract(),
    }
    _write_json(output / "contract_preflight.json", result)
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    unknown = set(args.rows) - set(EVALUATION_ROWS)
    if unknown:
        raise ValueError(f"unsupported FastV rows: {sorted(unknown)}")
    if args.num_trials < 1 or args.trial_offset < 0:
        raise ValueError("num_trials must be positive and trial_offset non-negative")
    output = Path(args.output).expanduser().resolve()
    if args.contract_only:
        return _write_contract_only(output)
    if output.exists():
        raise FileExistsError(f"refusing existing evaluation output: {output}")
    upstream, libero = _configure_paths()
    output.mkdir(parents=True)
    device = torch.device(args.device)
    model, processor = _load_simvla(args, device)
    metadata = _environment_metadata(args, upstream=upstream, libero=libero)
    _write_json(output / "environment_metadata.json", metadata)

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_count = min(
        int(suite.get_num_tasks()), args.max_tasks or int(suite.get_num_tasks())
    )
    task_ids = list(reversed(range(task_count)))
    all_episode_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for row_name in args.rows:
        row_dir = output / row_name
        row_dir.mkdir()
        episodes: list[dict[str, Any]] = []
        all_policy_ms: list[float] = []
        all_vlm_ms: list[float] = []
        all_action_ms: list[float] = []
        progress = tqdm(
            total=task_count * args.num_trials,
            desc=row_name,
            dynamic_ncols=True,
        )
        for task_id in task_ids:
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            env, prompt = get_libero_env(task, 256, args.environment_seed)
            try:
                for local_trial in range(args.num_trials):
                    trial_id = args.trial_offset + local_trial
                    env.reset()
                    obs = env.set_init_state(init_states[trial_id % len(init_states)])
                    for _ in range(10):
                        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                    policy = _policy(
                        row_name,
                        model=model,
                        processor=processor,
                        args=args,
                        device=device,
                        task_id=task_id,
                        trial_id=trial_id,
                    )
                    frames: list[np.ndarray] = []
                    success = False
                    policy_ms: list[float] = []
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
                        policy_ms.append((time.perf_counter() - started) * 1000.0)
                        obs, _, done, _ = env.step(action.action.tolist())
                        if done:
                            success = True
                            break
                    counters = dict(policy.metrics.counters)
                    vlm_ms = list(policy.metrics.latencies.get("VLM_encoder_ms", []))
                    action_ms = list(
                        policy.metrics.latencies.get("action_transformer_ms", [])
                    )
                    fastv_debug = _fastv_episode_debug(policy)
                    episode = {
                        "row": row_name,
                        "task_id": task_id,
                        "trial_id": trial_id,
                        "init_state_index": trial_id % len(init_states),
                        "environment_seed": args.environment_seed,
                        "action_noise_seed_base": args.action_noise_seed_base,
                        "success": success,
                        "episode_length": len(policy_ms),
                        "num_policy_queries": counters.get("num_policy_queries", 0),
                        "num_full_vlm_calls": counters.get("num_full_vlm_calls", 0),
                        "num_fastv_calls": counters.get("num_fastv_calls", 0),
                        "num_action_transformer_calls": counters.get(
                            "num_action_transformer_calls", 0
                        ),
                        "latency_per_executed_action_ms": _mean(policy_ms),
                        "policy_latency_p50_ms": _percentile(policy_ms, 50),
                        "policy_latency_p95_ms": _percentile(policy_ms, 95),
                        "vlm_latency_per_query_ms": _mean(vlm_ms),
                        "action_latency_per_query_ms": _mean(action_ms),
                        "fastv_compact_sequence_length": (
                            fastv_debug["compact_sequence_length"]
                            if fastv_debug
                            else None
                        ),
                        "fastv_visual_tokens_pruned": (
                            fastv_debug["visual_tokens_pruned"] if fastv_debug else 0
                        ),
                    }
                    episodes.append(episode)
                    all_episode_rows.append(episode)
                    all_policy_ms.extend(policy_ms)
                    all_vlm_ms.extend(vlm_ms)
                    all_action_ms.extend(action_ms)
                    _append_jsonl(row_dir / "progress.jsonl", episode)
                    if fastv_debug:
                        _append_jsonl(
                            row_dir / "fastv_pruning_debug.jsonl",
                            {
                                "task_id": task_id,
                                "trial_id": trial_id,
                                **fastv_debug,
                            },
                        )
                    if args.save_video and (
                        not args.video_failures_only or not success
                    ):
                        save_episode_video(
                            frames,
                            row_dir
                            / "videos"
                            / (
                                f"task{task_id:02d}_trial{trial_id:03d}_"
                                f"{'success' if success else 'failure'}.mp4"
                            ),
                            10,
                        )
                    progress.update(1)
                    successes = sum(int(item["success"]) for item in episodes)
                    progress.set_postfix(
                        success=f"{successes}/{len(episodes)}",
                        sr=f"{100 * successes / len(episodes):.1f}%",
                    )
            finally:
                env.close()
        progress.close()
        with (row_dir / "episode_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
            writer.writeheader()
            writer.writerows(episodes)
        per_task = {
            str(task_id): float(
                np.mean([item["success"] for item in episodes if item["task_id"] == task_id])
            )
            for task_id in task_ids
        }
        latency_warmup = min(args.latency_warmup_queries, len(all_vlm_ms))
        summary = {
            "row_contract": evaluation_row(row_name).serializable(),
            "episodes": len(episodes),
            "successes": sum(int(item["success"]) for item in episodes),
            "success_rate": float(np.mean([item["success"] for item in episodes])),
            "per_task_success_rate": per_task,
            "executed_actions": len(all_policy_ms),
            "policy_latency_per_executed_action_ms": _mean(all_policy_ms),
            "policy_latency_p95_ms": _percentile(all_policy_ms, 95),
            "vlm_latency_per_query_ms": _mean(all_vlm_ms[latency_warmup:]),
            "action_latency_per_query_ms": _mean(all_action_ms[latency_warmup:]),
            "latency_warmup_queries_excluded": latency_warmup,
            "full_vlm_calls": sum(item["num_full_vlm_calls"] for item in episodes),
            "fastv_calls": sum(item["num_fastv_calls"] for item in episodes),
        }
        summaries[row_name] = summary
        _write_json(row_dir / "summary.json", summary)

    baseline = summaries.get("baseline_k1")
    if baseline:
        for summary in summaries.values():
            policy_ms = summary["policy_latency_per_executed_action_ms"]
            vlm_ms = summary["vlm_latency_per_query_ms"]
            summary["end_to_end_speedup_vs_baseline"] = (
                baseline["policy_latency_per_executed_action_ms"] / policy_ms
                if policy_ms
                else None
            )
            summary["vlm_speedup_vs_baseline"] = (
                baseline["vlm_latency_per_query_ms"] / vlm_ms if vlm_ms else None
            )
    result = {
        "verdict": "SIMVLA_FASTV_EVAL_COMPLETE",
        "summaries": summaries,
        "paired_episode_identity": (
            "suite+task_id+trial_id+init_state_index+environment_seed"
        ),
        "paired_action_noise": True,
        "control_protocol": "H=10,R=5,flow_steps=10,current_observation_each_query",
    }
    with (output / "episode_metrics_all.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_episode_rows[0]))
        writer.writeheader()
        writer.writerows(all_episode_rows)
    _write_json(output / "comparison_summary.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", required=True)
    value.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    value.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    value.add_argument("--norm-stats")
    value.add_argument(
        "--rows",
        nargs="+",
        default=["baseline_k1", "fastv_k2_r50"],
    )
    value.add_argument(
        "--suite",
        choices=("libero_10", "libero_spatial", "libero_object", "libero_goal"),
        default="libero_10",
    )
    value.add_argument("--num-trials", type=int, default=50)
    value.add_argument("--trial-offset", type=int, default=0)
    value.add_argument("--max-tasks", type=int)
    value.add_argument("--max-policy-steps", type=int, default=900)
    value.add_argument("--environment-seed", type=int, default=0)
    value.add_argument("--action-noise-seed-base", type=int, default=20260902)
    value.add_argument("--latency-warmup-queries", type=int, default=5)
    value.add_argument("--save-video", action="store_true")
    value.add_argument("--video-stride", type=int, default=2)
    value.add_argument("--video-failures-only", action="store_true")
    value.add_argument("--contract-only", action="store_true")
    value.add_argument("--device", default="cuda")
    return value


def main() -> None:
    args = parser().parse_args()
    if not args.contract_only and not args.norm_stats:
        raise ValueError("--norm-stats is required for evaluation")
    result = evaluate(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
