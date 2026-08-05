"""Paired official-queue LIBERO screening for SimVLA LatentLoop variants."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
LIBERO_ROOT = UPSTREAM / "evaluation" / "libero" / "LIBERO"
for path in (ROOT, UPSTREAM, LIBERO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.latentloop.checkpoint import (  # noqa: E402
    freeze_module,
    load_adapter_checkpoint,
)
from architectures.simvla.adapters.latentloop.simvla_policy import (  # noqa: E402
    RealSimVLALatentLoopPolicy,
)
from architectures.simvla.adapters.latentloop.source_lock import (  # noqa: E402
    collect_source_lock,
    require_empty_output,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (  # noqa: E402
    build_env_obs,
    get_libero_env,
    save_episode_video,
    video_frame_from_obs,
)
from methods.latentloop.eval import action_diagnostics, distribution_summary  # noqa: E402


@dataclass(frozen=True)
class EvalRow:
    """One predeclared paired online screening row."""

    name: str
    mode: str
    full_query_interval: int
    checkpoint_path: str | None


ADAPTER_MODES = {
    "chunk_aware_latentloop",
    "old_observation_only",
    "no_observation",
    "nonrecurrent_condition",
    "action_chunk_correction",
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _required_path(path: str, label: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {resolved}")
    return str(resolved)


def _load_gate(path: str) -> dict[str, Any]:
    if not path:
        raise ValueError("this matrix requires --gate-decision-json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _custom_rows(args: argparse.Namespace) -> list[EvalRow]:
    """Parse user-approved confirmation rows without guessing checkpoint paths."""

    rows: list[EvalRow] = []
    for spec in args.row:
        parts = spec.split("|", 3)
        if len(parts) != 4:
            raise ValueError("--row must be NAME|MODE|K|CHECKPOINT_OR_DASH")
        name, mode, raw_k, raw_checkpoint = parts
        if mode not in ADAPTER_MODES | {"full", "hold_condition"}:
            raise ValueError(f"unsupported custom row mode: {mode}")
        checkpoint = None
        if mode in ADAPTER_MODES:
            checkpoint = _required_path(raw_checkpoint, name)
        elif raw_checkpoint != "-":
            raise ValueError(f"row {name} must use '-' because mode {mode} has no adapter")
        rows.append(EvalRow(name, mode, int(raw_k), checkpoint))
    if not rows or rows[0] != EvalRow("full_k1", "full", 1, None):
        raise ValueError("custom matrices must start with 'full_k1|full|1|-'")
    if len({row.name for row in rows}) != len(rows):
        raise ValueError("custom row names must be unique")
    return rows


def planned_rows(args: argparse.Namespace) -> list[EvalRow]:
    """Build only matrices allowed before R=5 K2 passes."""

    if args.matrix == "k1_parity":
        chunk = _required_path(args.chunk_aware_checkpoint, "chunk-aware")
        return [
            EvalRow("full_k1", "full", 1, None),
            EvalRow("adapter_loaded_full_k1", "full", 1, chunk),
        ]
    if args.matrix == "confirmation":
        rows = _custom_rows(args)
        gate = _load_gate(args.gate_decision_json)
        approved = list(gate.get("CONFIRMATION_ROWS_APPROVED", []))
        if not bool(gate.get("CONFIRMATION_APPROVED", False)):
            raise RuntimeError("confirmation is blocked: CONFIRMATION_APPROVED is false")
        if approved != [row.name for row in rows]:
            raise RuntimeError("custom rows do not exactly match CONFIRMATION_ROWS_APPROVED")
        return rows
    required = {
        "chunk": _required_path(args.chunk_aware_checkpoint, "chunk-aware"),
        "old": _required_path(args.old_observation_checkpoint, "old-observation-only"),
        "no_obs": _required_path(args.no_observation_checkpoint, "no-observation"),
        "nonrec": _required_path(args.nonrecurrent_checkpoint, "nonrecurrent"),
        "action": _required_path(args.action_correction_checkpoint, "action-correction"),
    }
    if args.matrix == "protocol_a_screening":
        rows = [EvalRow("full_k1", "full", 1, None)]
        for k in (2, 4):
            rows.extend(
                (
                    EvalRow(f"old_observation_only_k{k}", "old_observation_only", k, required["old"]),
                    EvalRow(f"chunk_aware_latentloop_k{k}", "chunk_aware_latentloop", k, required["chunk"]),
                    EvalRow(f"nonrecurrent_condition_k{k}", "nonrecurrent_condition", k, required["nonrec"]),
                    EvalRow(f"action_chunk_correction_k{k}", "action_chunk_correction", k, required["action"]),
                    EvalRow(f"hold_condition_k{k}", "hold_condition", k, None),
                    EvalRow(f"no_observation_k{k}", "no_observation", k, required["no_obs"]),
                )
            )
        return rows
    if args.matrix == "protocol_b_k2_screening":
        return [
            EvalRow("full_k1", "full", 1, None),
            EvalRow("old_observation_only_k2", "old_observation_only", 2, required["old"]),
            EvalRow("chunk_aware_latentloop_k2", "chunk_aware_latentloop", 2, required["chunk"]),
            EvalRow("nonrecurrent_condition_k2", "nonrecurrent_condition", 2, required["nonrec"]),
            EvalRow("action_chunk_correction_k2", "action_chunk_correction", 2, required["action"]),
            EvalRow("hold_condition_k2", "hold_condition", 2, None),
            EvalRow("no_observation_k2", "no_observation", 2, required["no_obs"]),
        ]
    if args.matrix == "protocol_b_post_k":
        if args.post_k not in {3, 4}:
            raise ValueError("protocol_b_post_k requires --post-k 3 or 4")
        gate = _load_gate(args.gate_decision_json)
        required_gate = "PROCEED_TO_R5_K3" if args.post_k == 3 else "PROCEED_TO_R5_K4"
        if not bool(gate.get(required_gate, False)):
            raise RuntimeError(f"R=5 K={args.post_k} is blocked by {required_gate}=false")
        k = args.post_k
        return [
            EvalRow("full_k1", "full", 1, None),
            EvalRow(f"old_observation_only_k{k}", "old_observation_only", k, required["old"]),
            EvalRow(f"chunk_aware_latentloop_k{k}", "chunk_aware_latentloop", k, required["chunk"]),
            EvalRow(f"nonrecurrent_condition_k{k}", "nonrecurrent_condition", k, required["nonrec"]),
            EvalRow(f"action_chunk_correction_k{k}", "action_chunk_correction", k, required["action"]),
            EvalRow(f"hold_condition_k{k}", "hold_condition", k, None),
            EvalRow(f"no_observation_k{k}", "no_observation", k, required["no_obs"]),
        ]
    raise ValueError(f"Unsupported matrix: {args.matrix}")


def _hierarchical_paired_ci(
    baseline: dict[tuple[int, int], bool],
    candidate: dict[tuple[int, int], bool],
    *,
    seed: int,
    samples: int = 10_000,
) -> list[float | None]:
    tasks = sorted(
        {
            task_id
            for task_id, episode_id in baseline
            if (task_id, episode_id) in candidate
        }
    )
    if not tasks:
        return [None, None]
    values_by_task: dict[int, list[float]] = {}
    for task in tasks:
        episodes = sorted(
            episode
            for task_id, episode in baseline
            if task_id == task and (task, episode) in candidate
        )
        values_by_task[task] = [
            float(candidate[(task, episode)]) - float(baseline[(task, episode)])
            for episode in episodes
        ]
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        task_means: list[float] = []
        for task in sampled_tasks:
            values = values_by_task[int(task)]
            sampled_values = rng.choice(values, size=len(values), replace=True)
            task_means.append(float(np.mean(sampled_values)))
        estimates.append(100.0 * float(np.mean(task_means)))
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def _paired_summary(
    baseline: dict[tuple[int, int], bool],
    candidate: dict[tuple[int, int], bool],
    *,
    seed: int,
) -> dict[str, Any]:
    keys = sorted(set(baseline) & set(candidate))
    flips = collections.Counter()
    for key in keys:
        pair = (baseline[key], candidate[key])
        flips[
            {
                (False, False): "both_fail",
                (False, True): "fail_to_success",
                (True, False): "success_to_fail",
                (True, True): "both_success",
            }[pair]
        ] += 1
    difference = 100.0 * np.mean(
        [float(candidate[key]) - float(baseline[key]) for key in keys]
    ) if keys else None
    return {
        "pairs": len(keys),
        "candidate_minus_full_pp": float(difference) if difference is not None else None,
        "candidate_minus_baseline_pp": float(difference) if difference is not None else None,
        "task_hierarchical_paired_ci95_pp": _hierarchical_paired_ci(
            baseline,
            candidate,
            seed=seed,
        ),
        "paired_flips": dict(flips),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the selected paired screening matrix and preserve full traces."""

    from libero.libero import benchmark
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    output = require_empty_output(args.output)
    rows = planned_rows(args)
    expected_r = 1 if args.matrix == "protocol_a_screening" else 5
    if args.matrix == "k1_parity":
        expected_r = args.execution_horizon
    elif args.matrix == "confirmation":
        expected_r = args.execution_horizon
    if args.execution_horizon != expected_r:
        raise ValueError(
            f"matrix {args.matrix} requires R={expected_r}, got R={args.execution_horizon}"
        )
    source_lock = collect_source_lock(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    _write_json(output / "source_lock.json", source_lock)
    _write_json(
        output / "eval_config.json",
        {
            **vars(args),
            "rows": [row.__dict__ for row in rows],
            "environment_action_gap_by_row": {
                row.name: row.full_query_interval * args.execution_horizon for row in rows
            },
        },
    )
    os.environ.setdefault("LIBERO_ROOT", str(LIBERO_ROOT))
    device = torch.device(args.device)
    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device)
    model.action_space.load_norm_stats(args.norm_stats)
    freeze_module(model)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)
    loaded_adapters: dict[str, Any] = {}
    for row in rows:
        if row.checkpoint_path and row.checkpoint_path not in loaded_adapters:
            adapter, payload = load_adapter_checkpoint(row.checkpoint_path, device=device)
            freeze_module(adapter)
            loaded_adapters[row.checkpoint_path] = (adapter, payload)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = (
        list(range(suite.n_tasks - 1, -1, -1))
        if args.task_order == "official_reverse"
        else list(range(suite.n_tasks))
    )[: args.max_tasks]
    episode_csv = output / "episode_metrics.csv"
    query_trace_path = output / "query_trace.jsonl"
    progress_path = output / "eval_progress.jsonl"
    episode_rows: list[dict[str, Any]] = []
    outcomes: dict[str, dict[tuple[int, int], bool]] = {row.name: {} for row in rows}
    row_counters: dict[str, collections.Counter[str]] = {
        row.name: collections.Counter() for row in rows
    }
    row_latencies: dict[str, dict[str, list[float]]] = {
        row.name: collections.defaultdict(list) for row in rows
    }
    row_action_diagnostics: dict[str, dict[str, list[float]]] = {
        row.name: collections.defaultdict(list) for row in rows
    }
    row_tracking: dict[str, dict[int, dict[str, list[float]]]] = {
        row.name: collections.defaultdict(lambda: collections.defaultdict(list))
        for row in rows
    }
    k1_chunks: dict[str, dict[tuple[int, int, int], Tensor]] = {
        row.name: {} for row in rows if row.full_query_interval == 1
    }
    video_root = output / "videos"
    total_episodes = len(rows) * len(task_ids) * args.num_trials
    completed = 0
    started = time.time()
    for row in rows:
        adapter = loaded_adapters[row.checkpoint_path][0] if row.checkpoint_path else None
        for task_id in task_ids:
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            env, prompt = get_libero_env(task, args.resolution, args.seed)
            try:
                for episode in range(args.num_trials):
                    episode_id = f"task{task_id:02d}_trial{episode:03d}"
                    env.reset()
                    obs = env.set_init_state(init_states[episode % len(init_states)])
                    for _ in range(args.num_wait_steps):
                        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                    policy = RealSimVLALatentLoopPolicy(
                        model=model,
                        processor=processor,
                        adapter=adapter,
                        mode=row.mode,
                        full_query_interval=row.full_query_interval,
                        execution_horizon=args.execution_horizon,
                        checkpoint_id=args.checkpoint,
                        flow_steps=args.flow_steps,
                        image_size=args.image_size,
                        client_resize_size=args.client_resize_size,
                        device=device,
                        suite=args.suite,
                        row_name=row.name,
                        task_id=task_id,
                        episode_id=episode_id,
                        action_noise_seed_base=args.action_noise_seed_base,
                        log_action_chunks=row.full_query_interval == 1,
                        teacher_tracking=args.teacher_tracking,
                    )
                    done = False
                    actions: list[np.ndarray] = []
                    chunk_boundaries: list[bool] = []
                    env_step_latencies: list[float] = []
                    video_enabled = (
                        args.save_video
                        and task_id == args.video_task_id
                        and episode in args.video_episodes
                    )
                    frames: list[np.ndarray] = []
                    for env_action_index in range(args.max_env_actions):
                        if video_enabled and env_action_index % args.video_stride == 0:
                            frames.append(video_frame_from_obs(obs))
                        image0, image1, proprio = build_env_obs(obs)
                        query_before = policy.query_index
                        step_output = policy.act(image0, image1, proprio, prompt)
                        chunk_boundaries.append(policy.query_index != query_before)
                        started_env = time.perf_counter()
                        obs, _, done, _ = env.step(step_output.action.tolist())
                        env_step_latencies.append(1000.0 * (time.perf_counter() - started_env))
                        actions.append(step_output.action.copy())
                        if done:
                            break
                    success = bool(done)
                    outcomes[row.name][(task_id, episode)] = success
                    for name, value in policy.metrics.counters.items():
                        row_counters[row.name][name] += int(value)
                    for name, values in policy.metrics.latencies.items():
                        row_latencies[row.name][name].extend(values)
                    row_latencies[row.name]["env_step_ms"].extend(env_step_latencies)
                    action_tensor = (
                        torch.as_tensor(np.stack(actions), dtype=torch.float32)
                        if actions
                        else torch.empty((0, 7), dtype=torch.float32)
                    )
                    diagnostics = action_diagnostics(
                        action_tensor,
                        chunk_boundaries=torch.as_tensor(chunk_boundaries, dtype=torch.bool),
                    )
                    for name, value in diagnostics.items():
                        row_action_diagnostics[row.name][name].append(float(value))
                    for record in policy.latentloop_query_trace:
                        _append_jsonl(
                            query_trace_path,
                            {
                                **record,
                                "row": row.name,
                                "task_id": task_id,
                                "episode": episode,
                                "execution_horizon": args.execution_horizon,
                                "full_query_interval": row.full_query_interval,
                            },
                        )
                    for tracking in policy.latentloop_tracking_trace:
                        age = int(tracking["query_age"])
                        for name, value in tracking.items():
                            if isinstance(value, (int, float)) and name not in {
                                "policy_query_index",
                                "query_age",
                            }:
                                row_tracking[row.name][age][name].append(float(value))
                    for record in policy.latentloop_action_chunks:
                        k1_chunks[row.name][
                            (task_id, episode, int(record["policy_query_index"]))
                        ] = record["action_chunk"]
                    video_path = ""
                    if video_enabled:
                        suffix = "success" if success else "fail"
                        path = video_root / row.name / f"{row.name}_task{task_id:02d}_ep{episode:02d}_{suffix}.mp4"
                        video_path = save_episode_video(frames, path, args.video_fps) or ""
                    episode_row = {
                        "row": row.name,
                        "mode": row.mode,
                        "task_id": task_id,
                        "episode": episode,
                        "success": success,
                        "environment_actions": len(actions),
                        "policy_queries": int(policy.metrics.counters["num_policy_queries"]),
                        "full_condition_calls": int(policy.metrics.counters["num_full_vlm_calls"]),
                        "condition_updater_calls": int(policy.metrics.counters["num_condition_updater_calls"]),
                        "observation_encoder_calls": int(policy.metrics.counters["num_observation_encoder_calls"]),
                        "executed_action_encoder_calls": int(policy.metrics.counters["num_executed_action_encoder_calls"]),
                        "action_transformer_decodes": int(policy.metrics.counters["num_action_transformer_decodes"]),
                        "action_transformer_flow_iterations": int(policy.metrics.counters["num_action_transformer_calls"]),
                        "execution_horizon": args.execution_horizon,
                        "full_query_interval": row.full_query_interval,
                        "environment_action_gap": args.execution_horizon * row.full_query_interval,
                        "video_path": video_path,
                        **diagnostics,
                    }
                    episode_rows.append(episode_row)
                    completed += 1
                    _append_jsonl(
                        progress_path,
                        {
                            "event": "episode_done",
                            "completed": completed,
                            "total": total_episodes,
                            "row": row.name,
                            "task_id": task_id,
                            "episode": episode,
                            "success": success,
                            "elapsed_seconds": time.time() - started,
                        },
                    )
            finally:
                env.close()
    with episode_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    summaries: dict[str, Any] = {}
    full_outcomes = outcomes["full_k1"]
    paired: dict[str, Any] = {}
    for row in rows:
        counters = row_counters[row.name]
        policy_queries = int(counters["num_policy_queries"])
        full_calls = int(counters["num_full_vlm_calls"])
        env_actions = int(counters["num_env_steps"])
        successes = sum(outcomes[row.name].values())
        task_wise = {
            str(task_id): sum(
                success for (candidate_task, _), success in outcomes[row.name].items()
                if candidate_task == task_id
            ) / args.num_trials
            for task_id in task_ids
        }
        summaries[row.name] = {
            "successes": successes,
            "episodes": len(outcomes[row.name]),
            "success_rate": successes / max(len(outcomes[row.name]), 1),
            "task_wise_success": task_wise,
            "prediction_horizon": 10,
            "execution_horizon": args.execution_horizon,
            "full_condition_interval": row.full_query_interval,
            "full_condition_environment_action_gap": args.execution_horizon * row.full_query_interval,
            "mean_executed_chunk_length": env_actions / max(policy_queries, 1),
            "full_condition_reduction_per_policy_query": 1.0 - full_calls / max(policy_queries, 1),
            "full_condition_reduction_per_environment_step": 1.0
            - full_calls / max(env_actions / float(args.execution_horizon), 1.0),
            "full_condition_calls_per_environment_step": full_calls / max(env_actions, 1),
            "amortized_policy_ms_per_environment_action": sum(
                row_latencies[row.name].get("policy_total_ms", [])
            ) / max(env_actions, 1),
            "amortized_policy_ms_per_environment_action_distribution": distribution_summary(
                row_latencies[row.name].get("policy_total_ms", [])
            ),
            "counters": dict(counters),
            "latency_ms": {
                name: distribution_summary(values)
                for name, values in row_latencies[row.name].items()
            },
            "action_diagnostics": {
                name: distribution_summary(values)
                for name, values in row_action_diagnostics[row.name].items()
            },
            "condition_action_tracking_by_query_age": {
                str(age): {
                    name: distribution_summary(values)
                    for name, values in metrics.items()
                }
                for age, metrics in sorted(row_tracking[row.name].items())
            },
            "teacher_tracking_enabled": bool(args.teacher_tracking),
            "teacher_tracking_excluded_from_operational_latency": True,
        }
        if row.name != "full_k1":
            paired[row.name] = _paired_summary(
                full_outcomes,
                outcomes[row.name],
                seed=args.bootstrap_seed,
            )
    paired_between_rows: dict[str, dict[str, Any]] = {}
    paired_candidates = [
        row for row in rows if row.mode == "chunk_aware_latentloop"
    ]
    for candidate in paired_candidates:
        paired_between_rows[candidate.name] = {}
        for baseline in rows:
            if candidate.name == baseline.name:
                continue
            paired_between_rows[candidate.name][baseline.name] = _paired_summary(
                outcomes[baseline.name],
                outcomes[candidate.name],
                seed=args.bootstrap_seed,
            )
    k1_parity: dict[str, Any] | None = None
    if "adapter_loaded_full_k1" in outcomes:
        baseline_chunks = k1_chunks["full_k1"]
        adapter_chunks = k1_chunks["adapter_loaded_full_k1"]
        common = sorted(set(baseline_chunks) & set(adapter_chunks))
        max_diff = max(
            (
                float((baseline_chunks[key] - adapter_chunks[key]).abs().max().item())
                for key in common
            ),
            default=None,
        )
        adapter_counters = row_counters["adapter_loaded_full_k1"]
        k1_parity = {
            "paired_action_chunks": len(common),
            "missing_chunk_keys": len(set(baseline_chunks) ^ set(adapter_chunks)),
            "max_abs_action_chunk_diff": max_diff,
            "exact_action_chunk_equality": bool(
                common
                and len(set(baseline_chunks) ^ set(adapter_chunks)) == 0
                and max_diff == 0.0
            ),
            "identical_paired_outcomes": outcomes["full_k1"] == outcomes["adapter_loaded_full_k1"],
            "updater_calls": int(adapter_counters["num_condition_updater_calls"]),
            "observation_encoder_calls": int(adapter_counters["num_observation_encoder_calls"]),
            "action_encoder_calls": int(adapter_counters["num_executed_action_encoder_calls"]),
        }
        k1_parity["K1_PARITY_PASS"] = bool(
            k1_parity["exact_action_chunk_equality"]
            and k1_parity["identical_paired_outcomes"]
            and k1_parity["updater_calls"] == 0
            and k1_parity["observation_encoder_calls"] == 0
            and k1_parity["action_encoder_calls"] == 0
        )
    result = {
        "matrix": args.matrix,
        "suite": args.suite,
        "episodes_per_row": len(task_ids) * args.num_trials,
        "rows": summaries,
        "paired_vs_full": paired,
        "paired_between_rows": paired_between_rows,
        "k1_parity": k1_parity,
        "episode_metrics_csv": str(episode_csv),
        "query_trace_jsonl": str(query_trace_path),
    }
    _write_json(output / "online_summary.json", result)
    return result


def _parse_episode_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    """Parse predeclared matrices with explicit confirmation/post-K gates."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        choices=(
            "k1_parity",
            "protocol_a_screening",
            "protocol_b_k2_screening",
            "confirmation",
            "protocol_b_post_k",
        ),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--chunk-aware-checkpoint", default="")
    parser.add_argument("--old-observation-checkpoint", default="")
    parser.add_argument("--no-observation-checkpoint", default="")
    parser.add_argument("--nonrecurrent-checkpoint", default="")
    parser.add_argument("--action-correction-checkpoint", default="")
    parser.add_argument("--gate-decision-json", default="")
    parser.add_argument("--row", action="append", default=[])
    parser.add_argument("--post-k", type=int, choices=(3, 4), default=None)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--execution-horizon", type=int, choices=(1, 2, 5), required=True)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--max-env-actions", type=int, default=900)
    parser.add_argument("--num-wait-steps", type=int, default=10)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--client-resize-size", type=int, default=224)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-noise-seed-base", type=int, default=20260804)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--task-order", choices=("official_reverse", "ascending"), default="official_reverse")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--teacher-tracking", action="store_true")
    parser.add_argument("--video-task-id", type=int, default=9)
    parser.add_argument("--video-episodes", type=_parse_episode_list, default=(0, 1))
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
