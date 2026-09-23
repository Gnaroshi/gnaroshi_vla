"""Guarded paired LIBERO evaluator for Hierarchical Latent-Action Correction."""

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
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
LIBERO_ROOT = UPSTREAM / "evaluation" / "libero" / "LIBERO"
for path in (ROOT, UPSTREAM, LIBERO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.hierarchical_correction.simvla_hybrid_policy import (  # noqa: E402
    RealSimVLAHierarchicalCorrectionPolicy,
    RealSimVLAStaleActionChunkPolicy,
    hybrid_parameter_audit,
)
from architectures.simvla.adapters.hierarchical_correction.source_locked_loading import (  # noqa: E402
    load_source_locked_processor,
    load_source_locked_simvla,
)
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
    resolve_huggingface_checkpoint,
    sha256_file,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (  # noqa: E402
    build_env_obs,
    get_libero_env,
    save_episode_video,
    video_frame_from_obs,
)
from methods.hierarchical_correction.metrics import (  # noqa: E402
    correction_residuals_by_age,
    hierarchical_action_diagnostics,
    paired_outcome_summary,
    trace_metrics_by_age,
)
from methods.hierarchical_correction.policy_state import validate_trace_record  # noqa: E402
from methods.hierarchical_correction.provenance import (  # noqa: E402
    experiment_source_signature,
    hierarchical_source_manifest,
)
from methods.hierarchical_correction.schedules import HierarchicalSchedule  # noqa: E402
from methods.latentloop.eval import distribution_summary  # noqa: E402


@dataclass(frozen=True)
class EvalRow:
    name: str
    route: str
    full_refresh_interval: int
    action_regeneration_interval: int | None = None
    regeneration_candidate: str | None = None


PRIMARY_ROW = "hierarchical_hybrid_r1_kf4_kg2"
NATIVE_FULL_ROW = "native_full_simvla_k1"
NATIVE_ACTION_ROW = "native_action_correction_kf4"
NATIVE_CONDITION_ROW = "native_condition_regeneration_kf4"
NATIVE_HYBRID_ROW = "native_horizon_hybrid_kf4_kg2"
NATIVE_HOLD_ROW = "native_stale_action_chunk_kf4"
NATIVE_NONRECURRENT_ROW = "native_nonrecurrent_regeneration_kf4"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_json(path: str, label: str) -> dict[str, Any]:
    if not path:
        raise ValueError(f"--{label} is required")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return json.loads(resolved.read_text(encoding="utf-8"))


def _required_checkpoint(path: str, label: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {resolved}")
    return str(resolved)


def _require_source_signature(
    path: str,
    label: str,
    expected: dict[str, Any],
) -> None:
    artifact = _load_json(path, label)
    if artifact.get("source_signature") != expected:
        raise RuntimeError(f"{label} was produced from a different source signature")


def _parse_ids(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _resolve_task_ids(
    *,
    suite_tasks: int,
    task_order: str,
    max_tasks: int,
    explicit_task_ids: tuple[int, ...],
) -> list[int]:
    if explicit_task_ids:
        if len(set(explicit_task_ids)) != len(explicit_task_ids):
            raise ValueError("--task-ids must not contain duplicates")
        if any(task < 0 or task >= suite_tasks for task in explicit_task_ids):
            raise ValueError("--task-ids contains an out-of-range task")
        return list(explicit_task_ids)
    ordered = (
        list(range(suite_tasks - 1, -1, -1))
        if task_order == "official_reverse"
        else list(range(suite_tasks))
    )
    return ordered[:max_tasks]


def _validate_matrix_args(args: argparse.Namespace) -> list[EvalRow]:
    if args.matrix != "native_r5" and args.execution_horizon != 1:
        raise ValueError("new hierarchical diagnostics are locked to Protocol A, R=1")
    if args.matrix == "smoke":
        if args.num_trials != 2 or args.max_tasks != 1 or args.task_ids:
            raise ValueError("smoke is fixed to --max-tasks 1 --num-trials 2 without --task-ids")
        parity = _load_json(args.parity_summary, "parity-summary")
        if not parity.get("ENDPOINT_PARITY_PASS", False):
            raise RuntimeError("smoke blocked: endpoint parity has not passed")
        return [EvalRow(PRIMARY_ROW, "hybrid", 4, 2)]
    if args.matrix == "scientific_r1_k4":
        if args.num_trials != 20:
            raise ValueError("scientific_r1_k4 is fixed to 20 trials per selected task")
        if not args.task_ids and args.max_tasks != 10:
            raise ValueError("unsharded scientific_r1_k4 requires all 10 tasks")
        parity = _load_json(args.parity_summary, "parity-summary")
        smoke = _load_json(args.smoke_summary, "smoke-summary")
        offline = _load_json(args.offline_summary, "offline-summary")
        if not parity.get("ENDPOINT_PARITY_PASS", False):
            raise RuntimeError("scientific diagnostic blocked by endpoint parity")
        if not smoke.get("SMOKE_INVARIANTS_PASS", False):
            raise RuntimeError("scientific diagnostic blocked by smoke invariants")
        if not offline.get("ONLINE_EVALUATION_GATE_PASS", False):
            raise RuntimeError("scientific diagnostic blocked by offline replay gate")
        if not args.teacher_tracking:
            raise ValueError("scientific_r1_k4 requires --teacher-tracking for state diagnostics")
        rows = [
            EvalRow("full_k1", "full", 1),
            EvalRow("chunk_aware_latentloop_k4", "condition", 4),
            EvalRow("action_chunk_correction_k4", "action", 4),
            EvalRow(PRIMARY_ROW, "hybrid", 4, 2),
        ]
        if args.include_nonrecurrent:
            _required_checkpoint(args.nonrecurrent_checkpoint, "nonrecurrent")
            rows.append(EvalRow("nonrecurrent_condition_k4", "nonrecurrent", 4))
        return rows
    if args.matrix == "conditional_k8":
        if not args.enable_conditional_k8:
            raise RuntimeError("K8 is disabled; pass --enable-conditional-k8 after the K4 verdict")
        gate = _load_json(args.gate_decision_json, "gate-decision-json")
        if not gate.get("k8_diagnostic_allowed", False):
            raise RuntimeError("K8 is blocked by the predeclared K4 decision")
        if args.num_trials != 10 or args.max_tasks != 10:
            raise ValueError("conditional K8 diagnostic is fixed to 10 tasks x 10 trials")
        return [
            EvalRow("hierarchical_condition_endpoint_r1_kf8_kg1", "hybrid", 8, 1),
            EvalRow("hierarchical_hybrid_r1_kf8_kg2", "hybrid", 8, 2),
            EvalRow("hierarchical_hybrid_r1_kf8_kg4", "hybrid", 8, 4),
            EvalRow("hierarchical_action_endpoint_r1_kf8_kg8", "hybrid", 8, 8),
        ]
    if args.matrix == "native_r5":
        if not args.enable_native_r5:
            raise RuntimeError(
                "native R5 is default-off; pass --enable-native-r5 only after the exact-age gate"
            )
        gate = _load_json(args.r5_regeneration_gate_json, "r5-regeneration-gate-json")
        if not gate.get("ONLINE_R5_GATE_PASS", False):
            raise RuntimeError("native R5 is blocked: ONLINE_R5_GATE_PASS is false")
        if args.execution_horizon != 5:
            raise ValueError("native_r5 requires --execution-horizon 5")
        if int(gate.get("action_horizon", -1)) != 10 or int(
            gate.get("execution_horizon", -1)
        ) != 5:
            raise RuntimeError("native R5 gate was not produced for H=10,R=5")
        if list(gate.get("native_level_sequence", [])) != [2, 0, 1, 0, 2]:
            raise RuntimeError("native R5 gate does not contain the provenance-derived schedule")
        if args.suite != "libero_10" or args.num_trials != 20:
            raise ValueError("native_r5 is fixed to LIBERO-10 with 20 paired episodes per task")
        if args.task_ids or args.max_tasks != 10:
            raise ValueError("native_r5 requires one complete unsharded 10-task matrix")
        if not args.teacher_tracking:
            raise ValueError("native_r5 requires --teacher-tracking for post-exhaustion diagnostics")
        selected = str(gate.get("selected_candidate"))
        if selected not in {"recurrent_age2", "nonrecurrent_anchor_age2"}:
            raise RuntimeError(f"unsupported selected R5 candidate: {selected}")
        condition_route = "condition" if selected == "recurrent_age2" else "nonrecurrent"
        hybrid_route = "hybrid" if selected == "recurrent_age2" else "hybrid_nonrecurrent"
        rows = [
            EvalRow(NATIVE_FULL_ROW, "full", 1),
            EvalRow(NATIVE_ACTION_ROW, "action", 4),
            EvalRow(
                NATIVE_CONDITION_ROW,
                condition_route,
                4,
                regeneration_candidate=selected,
            ),
            EvalRow(
                NATIVE_HYBRID_ROW,
                hybrid_route,
                4,
                2,
                regeneration_candidate=selected,
            ),
            EvalRow(NATIVE_HOLD_ROW, "hold_action", 4),
        ]
        nonrecurrent_gate = gate.get("candidates", {}).get(
            "nonrecurrent_anchor_age2", {}
        )
        if selected == "recurrent_age2" and nonrecurrent_gate.get("pass", False):
            rows.append(
                EvalRow(
                    NATIVE_NONRECURRENT_ROW,
                    "nonrecurrent",
                    4,
                    regeneration_candidate="nonrecurrent_anchor_age2",
                )
            )
        return rows
    raise ValueError(f"unsupported matrix: {args.matrix}")


def _policy_for_row(
    row: EvalRow,
    *,
    model: Any,
    processor: Any,
    condition_adapter: Any,
    correction_adapter: Any,
    nonrecurrent_adapter: Any,
    args: argparse.Namespace,
    task_id: int,
    episode_id: str,
) -> RealSimVLALatentLoopPolicy:
    common = {
        "model": model,
        "processor": processor,
        "execution_horizon": args.execution_horizon,
        "checkpoint_id": args.checkpoint,
        "flow_steps": args.flow_steps,
        "image_size": args.image_size,
        "client_resize_size": args.client_resize_size,
        "device": torch.device(args.device),
        "suite": args.suite,
        "row_name": row.name,
        "task_id": task_id,
        "episode_id": episode_id,
        "action_noise_seed_base": args.action_noise_seed_base,
        "log_action_chunks": True,
        "teacher_tracking": args.teacher_tracking,
    }
    if row.route in {"hybrid", "hybrid_nonrecurrent"}:
        assert row.action_regeneration_interval is not None
        selected_adapter = (
            condition_adapter if row.route == "hybrid" else nonrecurrent_adapter
        )
        if selected_adapter is None:
            raise RuntimeError(f"row {row.name} requires a nonrecurrent checkpoint")
        return RealSimVLAHierarchicalCorrectionPolicy(
            condition_adapter=selected_adapter,
            action_correction_adapter=correction_adapter,
            full_refresh_interval=row.full_refresh_interval,
            action_regeneration_interval=row.action_regeneration_interval,
            **common,
        )
    if row.route == "hold_action":
        return RealSimVLAStaleActionChunkPolicy(
            full_query_interval=row.full_refresh_interval,
            **common,
        )
    route = {
        "full": ("full", None),
        "condition": ("chunk_aware_latentloop", condition_adapter),
        "action": ("action_chunk_correction", correction_adapter),
        "nonrecurrent": ("nonrecurrent_condition", nonrecurrent_adapter),
    }[row.route]
    return RealSimVLALatentLoopPolicy(
        adapter=route[1],
        mode=route[0],
        full_query_interval=row.full_refresh_interval,
        **common,
    )


def _smoke_invariants(
    traces: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_episode: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in traces:
        by_episode[(int(record["task_id"]), int(record["episode"]))].append(record)
    failures: list[str] = []
    for key, records in by_episode.items():
        records.sort(key=lambda record: int(record["policy_query_index"]))
        schedule = HierarchicalSchedule(4, 2, 1)
        for index, record in enumerate(records):
            failures.extend(
                f"{key} query {index}: {error}" for error in validate_trace_record(record)
            )
            expected_level = int(schedule.level(index))
            if int(record["execution_level"]) != expected_level:
                failures.append(f"{key} query {index}: level mismatch")
            if not record.get("condition_cache_advanced", False):
                failures.append(f"{key} query {index}: condition cache did not advance/reset")
            if not record.get("action_cache_replaced", False):
                failures.append(f"{key} query {index}: action cache was not replaced")
            if expected_level == 0:
                if record.get("action_transformer_called") or not record.get("action_correction_called"):
                    failures.append(f"{key} query {index}: invalid Level 0 calls")
                if not record.get("correction_consumes_latest_action_cache"):
                    failures.append(f"{key} query {index}: stale correction input")
            if expected_level == 1:
                if record.get("full_condition_called") or not record.get("action_transformer_called"):
                    failures.append(f"{key} query {index}: invalid Level 1 calls")
            if expected_level == 2:
                if not record.get("full_condition_called") or not record.get("action_transformer_called"):
                    failures.append(f"{key} query {index}: invalid Level 2 calls")
    finite = all(float(row["finite_action_fraction"]) == 1.0 for row in episode_rows)
    if not finite:
        failures.append("non-finite executed action observed")
    return {
        "SMOKE_INVARIANTS_PASS": not failures and len(by_episode) == 2,
        "episodes_checked": len(by_episode),
        "query_records_checked": len(traces),
        "failures": failures,
        "manual_review_required": [
            "inspect query_trace.jsonl for exact 2,0,1,0 sequence",
            "inspect query-age and cache-source columns at Level 1 -> Level 0 boundaries",
            "inspect gripper_positive_fraction per episode",
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run only a predeclared, explicitly gated matrix."""

    from libero.libero import benchmark
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    rows = _validate_matrix_args(args)
    output = require_empty_output(args.output)
    condition_path = _required_checkpoint(args.condition_checkpoint, "condition")
    correction_path = _required_checkpoint(args.action_correction_checkpoint, "action-correction")
    source_lock = collect_source_lock(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    source_lock["hierarchical_checkpoints"] = {
        "condition": {"path": condition_path, "sha256": sha256_file(condition_path)},
        "action_correction": {
            "path": correction_path,
            "sha256": sha256_file(correction_path),
        },
    }
    if args.matrix == "native_r5":
        nonrecurrent_path = _required_checkpoint(
            args.nonrecurrent_checkpoint, "nonrecurrent"
        )
        old_observation_path = _required_checkpoint(
            args.old_observation_checkpoint, "old-observation-only"
        )
        source_lock["hierarchical_checkpoints"] = {
            "condition": {"path": condition_path, "sha256": sha256_file(condition_path)},
            "nonrecurrent": {
                "path": nonrecurrent_path,
                "sha256": sha256_file(nonrecurrent_path),
            },
            "old_observation_only": {
                "path": old_observation_path,
                "sha256": sha256_file(old_observation_path),
            },
            "action_correction": {
                "path": correction_path,
                "sha256": sha256_file(correction_path),
            },
        }
    source_lock["processor_checkpoint"] = resolve_huggingface_checkpoint(
        args.smolvlm_model_path
    )
    source_lock["hierarchical_implementation"] = hierarchical_source_manifest(ROOT)
    source_signature = experiment_source_signature(source_lock)
    if args.matrix == "smoke":
        _require_source_signature(args.parity_summary, "parity-summary", source_signature)
    elif args.matrix == "scientific_r1_k4":
        _require_source_signature(args.parity_summary, "parity-summary", source_signature)
        _require_source_signature(args.smoke_summary, "smoke-summary", source_signature)
        _require_source_signature(args.offline_summary, "offline-summary", source_signature)
    elif args.matrix == "conditional_k8":
        _require_source_signature(
            args.gate_decision_json,
            "gate-decision-json",
            source_signature,
        )
    elif args.matrix == "native_r5":
        _require_source_signature(
            args.r5_regeneration_gate_json,
            "r5-regeneration-gate-json",
            source_signature,
        )
    _write_json(output / "source_lock.json", source_lock)
    os.environ.setdefault("LIBERO_ROOT", str(LIBERO_ROOT))
    device = torch.device(args.device)
    model = load_source_locked_simvla(SmolVLMVLA, source_lock, device=device)
    model.action_space.load_norm_stats(args.norm_stats)
    freeze_module(model)
    processor = load_source_locked_processor(SmolVLMVLAProcessor, source_lock)
    condition_adapter, condition_payload = load_adapter_checkpoint(condition_path, device=device)
    correction_adapter, correction_payload = load_adapter_checkpoint(correction_path, device=device)
    freeze_module(condition_adapter)
    freeze_module(correction_adapter)
    nonrecurrent_adapter = None
    if any(row.route in {"nonrecurrent", "hybrid_nonrecurrent"} for row in rows):
        nonrecurrent_adapter, _ = load_adapter_checkpoint(
            _required_checkpoint(args.nonrecurrent_checkpoint, "nonrecurrent"),
            device=device,
        )
        freeze_module(nonrecurrent_adapter)
    parameter_audit = hybrid_parameter_audit(condition_adapter, correction_adapter)
    parameter_audit.update(
        {
            "condition_checkpoint_step": int(condition_payload.get("step", -1)),
            "action_correction_checkpoint_step": int(correction_payload.get("step", -1)),
        }
    )
    if args.matrix == "native_r5":
        gate = _load_json(args.r5_regeneration_gate_json, "r5-regeneration-gate-json")
        selected = str(gate["selected_candidate"])
        nonrecurrent_parameters = (
            sum(parameter.numel() for parameter in nonrecurrent_adapter.parameters())
            if nonrecurrent_adapter is not None
            else 0
        )
        active_condition_parameters = (
            nonrecurrent_parameters
            if selected == "nonrecurrent_anchor_age2"
            else int(parameter_audit["condition_adapter_parameters"])
        )
        parameter_audit.update(
            {
                "selected_regeneration_candidate": selected,
                "nonrecurrent_adapter_parameters": nonrecurrent_parameters,
                "active_hybrid_adapter_parameters": active_condition_parameters
                + int(parameter_audit["action_correction_adapter_parameters"]),
            }
        )
    _write_json(output / "parameter_audit.json", parameter_audit)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = _resolve_task_ids(
        suite_tasks=suite.n_tasks,
        task_order=args.task_order,
        max_tasks=args.max_tasks,
        explicit_task_ids=args.task_ids,
    )
    _write_json(
        output / "eval_config.json",
        {
            **vars(args),
            "resolved_task_ids": task_ids,
            "rows": [row.__dict__ for row in rows],
            "parameter_audit": parameter_audit,
            "teacher_tracking_excluded_from_operational_counters_and_latency": True,
        },
    )

    episode_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    outcomes: dict[str, dict[tuple[int, int], bool]] = {row.name: {} for row in rows}
    row_counters = {row.name: collections.Counter() for row in rows}
    row_latencies: dict[str, dict[str, list[float]]] = {
        row.name: collections.defaultdict(list) for row in rows
    }
    row_diagnostics: dict[str, dict[str, list[float]]] = {
        row.name: collections.defaultdict(list) for row in rows
    }
    row_tracking: dict[str, dict[int, dict[str, list[float]]]] = {
        row.name: collections.defaultdict(lambda: collections.defaultdict(list)) for row in rows
    }
    trace_path = output / "query_trace.jsonl"
    progress_path = output / "eval_progress.jsonl"
    video_root = output / "videos"
    total = len(rows) * len(task_ids) * args.num_trials
    progress = tqdm(total=total, desc=f"Hierarchical correction {args.matrix}", mininterval=args.tqdm_mininterval)
    completed = 0
    wall_started = time.time()
    for row in rows:
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
                    policy = _policy_for_row(
                        row,
                        model=model,
                        processor=processor,
                        condition_adapter=condition_adapter,
                        correction_adapter=correction_adapter,
                        nonrecurrent_adapter=nonrecurrent_adapter,
                        args=args,
                        task_id=task_id,
                        episode_id=episode_id,
                    )
                    actions: list[np.ndarray] = []
                    boundaries: list[bool] = []
                    env_step_ms: list[float] = []
                    frames: list[np.ndarray] = []
                    done = False
                    video_enabled = args.save_video and task_id in args.video_task_ids and episode in args.video_episodes
                    for env_action_index in range(args.max_env_actions):
                        if video_enabled and env_action_index % args.video_stride == 0:
                            frames.append(video_frame_from_obs(obs))
                        image0, image1, proprio = build_env_obs(obs)
                        query_before = policy.query_index
                        output_step = policy.act(image0, image1, proprio, prompt)
                        boundaries.append(policy.query_index != query_before)
                        env_started = time.perf_counter()
                        obs, _, done, _ = env.step(output_step.action.tolist())
                        env_step_ms.append(1000.0 * (time.perf_counter() - env_started))
                        actions.append(output_step.action.copy())
                        if done:
                            break
                    success = bool(done)
                    outcomes[row.name][(task_id, episode)] = success
                    for name, value in policy.metrics.counters.items():
                        row_counters[row.name][name] += int(value)
                    for name, values in policy.metrics.latencies.items():
                        row_latencies[row.name][name].extend(values)
                    row_latencies[row.name]["env_step_ms"].extend(env_step_ms)
                    action_tensor = (
                        torch.as_tensor(np.stack(actions), dtype=torch.float32)
                        if actions
                        else torch.empty((0, 7), dtype=torch.float32)
                    )
                    diagnostics = hierarchical_action_diagnostics(
                        action_tensor,
                        chunk_boundaries=torch.as_tensor(boundaries, dtype=torch.bool),
                    )
                    for name, value in diagnostics.items():
                        row_diagnostics[row.name][name].append(float(value))
                    for record in policy.latentloop_query_trace:
                        enriched = {
                            **record,
                            "row": row.name,
                            "task_id": task_id,
                            "episode": episode,
                        }
                        traces.append(enriched)
                        _append_jsonl(trace_path, enriched)
                    for tracking in policy.latentloop_tracking_trace:
                        age = int(tracking["query_age"])
                        for name, value in tracking.items():
                            if isinstance(value, (int, float)) and name not in {"query_age", "policy_query_index"}:
                                row_tracking[row.name][age][name].append(float(value))
                    video_path = ""
                    if video_enabled:
                        suffix = "success" if success else "fail"
                        path = video_root / row.name / f"task{task_id:02d}_ep{episode:03d}_{suffix}.mp4"
                        video_path = save_episode_video(frames, path, args.video_fps) or ""
                    episode_row = {
                        "row": row.name,
                        "route": row.route,
                        "task_id": task_id,
                        "episode": episode,
                        "episode_key": episode_id,
                        "success": success,
                        "environment_actions": len(actions),
                        "policy_queries": int(policy.metrics.counters["num_policy_queries"]),
                        "full_condition_calls": int(policy.metrics.counters["num_full_vlm_calls"]),
                        "condition_updater_calls": int(policy.metrics.counters["num_condition_updater_calls"]),
                        "action_transformer_decodes": int(policy.metrics.counters["num_action_transformer_decodes"]),
                        "action_transformer_flow_iterations": int(policy.metrics.counters["num_action_transformer_calls"]),
                        "action_correction_calls": int(policy.metrics.counters["num_action_correction_calls"]),
                        "nonrecurrent_condition_calls": int(
                            policy.metrics.counters["num_nonrecurrent_condition_calls"]
                        ),
                        "stale_action_chunk_shifts": int(
                            policy.metrics.counters["num_stale_action_chunk_shifts"]
                        ),
                        "video_path": video_path,
                        **diagnostics,
                    }
                    episode_rows.append(episode_row)
                    completed += 1
                    progress.update(1)
                    progress.set_postfix(row=row.name, task=task_id, episode=episode, success=int(success))
                    _append_jsonl(
                        progress_path,
                        {
                            "event": "episode_done",
                            "completed": completed,
                            "total": total,
                            "row": row.name,
                            "task_id": task_id,
                            "episode": episode,
                            "success": success,
                            "elapsed_seconds": time.time() - wall_started,
                        },
                    )
            finally:
                env.close()
    progress.close()
    episode_csv = output / "episode_metrics.csv"
    with episode_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)

    metric_samples = {
        "schema_version": "simvla_hierarchical_metric_samples_v2",
        "latencies": {
            row: {name: list(values) for name, values in metrics.items()}
            for row, metrics in row_latencies.items()
        },
        "action_diagnostics": {
            row: {name: list(values) for name, values in metrics.items()}
            for row, metrics in row_diagnostics.items()
        },
        "condition_action_tracking_by_query_age": {
            row: {
                int(age): {name: list(values) for name, values in metrics.items()}
                for age, metrics in ages.items()
            }
            for row, ages in row_tracking.items()
        },
        "correction_residual_records": {
            row.name: [
                {
                    "query_age": int(record["query_age"]),
                    "action_correction_residual": record.get("action_correction_residual"),
                }
                for record in traces
                if record["row"] == row.name
                and record.get("action_correction_residual") is not None
            ]
            for row in rows
        },
        "condition_drift_records": {
            row.name: [
                {
                    "query_age": int(record["query_age"]),
                    "condition_cache_drift": record.get("condition_cache_drift"),
                }
                for record in traces
                if record["row"] == row.name
                and record.get("condition_cache_drift") is not None
            ]
            for row in rows
        },
    }
    metric_samples_path = output / "metric_samples.pt"
    torch.save(metric_samples, metric_samples_path)

    summaries: dict[str, Any] = {}
    for row in rows:
        row_outcomes = outcomes[row.name]
        counters = row_counters[row.name]
        env_actions = int(counters["num_env_steps"])
        summaries[row.name] = {
            "successes": sum(row_outcomes.values()),
            "episodes": len(row_outcomes),
            "success_rate": sum(row_outcomes.values()) / max(len(row_outcomes), 1),
            "task_wise_success": {
                str(task): sum(
                    success for (candidate_task, _), success in row_outcomes.items() if candidate_task == task
                )
                / args.num_trials
                for task in task_ids
            },
            "counters": dict(counters),
            "latency_ms": {
                name: distribution_summary(values) for name, values in row_latencies[row.name].items()
            },
            "amortized_policy_ms_per_environment_action": sum(
                row_latencies[row.name].get("policy_total_ms", [])
            )
            / max(env_actions, 1),
            "action_diagnostics": {
                name: distribution_summary(values) for name, values in row_diagnostics[row.name].items()
            },
            "correction_residual_by_query_age": correction_residuals_by_age(
                record for record in traces if record["row"] == row.name
            ),
            "condition_cache_drift_by_query_age": trace_metrics_by_age(
                (record for record in traces if record["row"] == row.name),
                field="condition_cache_drift",
            ),
            "condition_action_tracking_by_query_age": {
                str(age): {
                    name: distribution_summary(values) for name, values in metrics.items()
                }
                for age, metrics in sorted(row_tracking[row.name].items())
            },
            "teacher_tracking_enabled": bool(args.teacher_tracking),
            "teacher_tracking_excluded_from_operational_latency": True,
        }
    paired: dict[str, Any] = {}
    if "full_k1" in outcomes:
        for row in rows:
            if row.name != "full_k1":
                paired[f"{row.name}_minus_full_k1"] = paired_outcome_summary(
                    outcomes["full_k1"], outcomes[row.name], seed=args.bootstrap_seed
                )
    if "action_chunk_correction_k4" in outcomes:
        paired["hybrid_minus_action_correction"] = paired_outcome_summary(
            outcomes["action_chunk_correction_k4"],
            outcomes[PRIMARY_ROW],
            seed=args.bootstrap_seed,
        )
    if "chunk_aware_latentloop_k4" in outcomes:
        paired["hybrid_minus_pure_condition"] = paired_outcome_summary(
            outcomes["chunk_aware_latentloop_k4"],
            outcomes[PRIMARY_ROW],
            seed=args.bootstrap_seed,
        )
    if args.matrix == "native_r5":
        for row in rows:
            if row.name != NATIVE_FULL_ROW:
                paired[f"{row.name}_minus_{NATIVE_FULL_ROW}"] = paired_outcome_summary(
                    outcomes[NATIVE_FULL_ROW],
                    outcomes[row.name],
                    seed=args.bootstrap_seed,
                )
        paired["hybrid_minus_action_correction"] = paired_outcome_summary(
            outcomes[NATIVE_ACTION_ROW],
            outcomes[NATIVE_HYBRID_ROW],
            seed=args.bootstrap_seed,
        )
        paired["hybrid_minus_condition_regeneration"] = paired_outcome_summary(
            outcomes[NATIVE_CONDITION_ROW],
            outcomes[NATIVE_HYBRID_ROW],
            seed=args.bootstrap_seed,
        )
    result = {
        "matrix": args.matrix,
        "suite": args.suite,
        "task_ids": task_ids,
        "episodes_per_row": len(task_ids) * args.num_trials,
        "rows": summaries,
        "paired": paired,
        "parameter_audit": parameter_audit,
        "source_signature": source_signature,
        "episode_metrics_csv": str(episode_csv),
        "query_trace_jsonl": str(trace_path),
        "metric_samples_pt": str(metric_samples_path),
    }
    _write_json(output / "online_summary.json", result)
    if args.matrix == "smoke":
        smoke = _smoke_invariants(traces, episode_rows)
        _write_json(output / "smoke_invariants.json", smoke)
        result.update(smoke)
        _write_json(output / "smoke_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        choices=("smoke", "scientific_r1_k4", "conditional_k8", "native_r5"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--condition-checkpoint", required=True)
    parser.add_argument("--action-correction-checkpoint", required=True)
    parser.add_argument("--nonrecurrent-checkpoint", default="")
    parser.add_argument("--old-observation-checkpoint", default="")
    parser.add_argument("--include-nonrecurrent", action="store_true")
    parser.add_argument("--parity-summary", default="")
    parser.add_argument("--smoke-summary", default="")
    parser.add_argument("--offline-summary", default="")
    parser.add_argument("--gate-decision-json", default="")
    parser.add_argument("--enable-conditional-k8", action="store_true")
    parser.add_argument("--enable-native-r5", action="store_true")
    parser.add_argument("--r5-regeneration-gate-json", default="")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--execution-horizon", type=int, default=1)
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--max-tasks", type=int, required=True)
    parser.add_argument("--task-ids", type=_parse_ids, default=())
    parser.add_argument("--max-env-actions", type=int, default=900)
    parser.add_argument("--num-wait-steps", type=int, default=10)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--client-resize-size", type=int, default=224)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-noise-seed-base", type=int, default=20260804)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--task-order", choices=("official_reverse", "ascending"), default="official_reverse")
    parser.add_argument("--teacher-tracking", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-task-ids", type=_parse_ids, default=(9,))
    parser.add_argument("--video-episodes", type=_parse_ids, default=(0, 1))
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--tqdm-mininterval", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
