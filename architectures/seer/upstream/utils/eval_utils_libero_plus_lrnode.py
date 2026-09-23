"""LIBERO-Plus orchestration for the canonical Seer/LR-NODE wrapper.

Only Plus task enumeration and aggregation live here. Preprocessing, Seer and
LR-NODE inference, cache behavior, temporal ensembling, action conversion, and
latency counters remain in ``utils.eval_utils_libero``.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

from utils import eval_utils_libero as canonical
from utils.train_utils import get_cast_dtype


CATEGORY_ORDER = ["Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout"]
CATEGORY_ALIASES = {
    "camera": "Camera",
    "camera viewpoints": "Camera",
    "view": "Camera",
    "robot": "Robot",
    "robot initial states": "Robot",
    "language": "Language",
    "language instructions": "Language",
    "light": "Light",
    "light conditions": "Light",
    "background": "Background",
    "background textures": "Background",
    "noise": "Noise",
    "sensor noise": "Noise",
    "layout": "Layout",
    "objects layout": "Layout",
}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_category(value: Any) -> str | None:
    if value is None:
        return None
    return CATEGORY_ALIASES.get(str(value).strip().lower())


def _parse_categories(value: Any) -> list[str]:
    categories: set[str] = set()
    if isinstance(value, str):
        normalized = _normalize_category(value)
        if normalized is not None:
            categories.add(normalized)
    elif isinstance(value, list):
        for item in value:
            categories.update(_parse_categories(item))
    elif isinstance(value, dict):
        for key, inner in value.items():
            normalized_key = _normalize_category(key)
            if isinstance(inner, bool):
                if inner and normalized_key is not None:
                    categories.add(normalized_key)
            else:
                categories.update(_parse_categories(inner))
                if normalized_key is not None and inner:
                    categories.add(normalized_key)
    return sorted(categories)


def _classification_path(args) -> Path:
    return Path(args.libero_path) / "libero/libero/benchmark/task_classification.json"


def load_task_classification(args) -> dict[str, dict[str, dict[str, Any]]]:
    path = _classification_path(args)
    if not path.is_file():
        raise FileNotFoundError(f"LIBERO-Plus task classification is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a dict in {path}, got {type(raw)}")
    suites: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for suite, items in raw.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            suites[suite][str(item["name"])] = {
                "categories": _parse_categories(item.get("category")),
                "difficulty_level": item.get("difficulty_level"),
                "classification_id": item.get("id"),
            }
    return dict(suites)


def build_plus_result_tables(
    *,
    suite_name: str,
    task_records: Sequence[Mapping[str, Any]],
    classification: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build deterministic task/category tables from one outcome per Plus task."""
    task_rows: list[dict[str, Any]] = []
    category_values: dict[str, list[int]] = defaultdict(list)
    category_skips: dict[str, int] = defaultdict(int)
    for record in sorted(task_records, key=lambda item: int(item["task_id"])):
        result = int(record["result"])
        task_name = str(record["task_name"])
        metadata = classification.get(task_name, {})
        categories = list(metadata.get("categories", []))
        status = "success" if result == 1 else "fail" if result == 0 else "skip"
        task_rows.append({
            "suite": suite_name,
            "task_id": int(record["task_id"]),
            "task_name": task_name,
            "result": result,
            "status": status,
            "categories": "|".join(categories),
            "difficulty_level": metadata.get("difficulty_level"),
            "classification_id": metadata.get("classification_id"),
        })
        for category in categories:
            if category not in CATEGORY_ORDER:
                continue
            if result in (0, 1):
                category_values[category].append(result)
            else:
                category_skips[category] += 1

    category_rows: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        values = category_values.get(category, [])
        skipped = int(category_skips.get(category, 0))
        category_rows.append({
            "suite": suite_name,
            "category": category,
            "num_tasks": len(values) + skipped,
            "num_valid": len(values),
            "num_skipped": skipped,
            "successes": int(sum(values)),
            "avg_success": float(np.mean(values)) if values else None,
        })
    total_values = [int(row["result"]) for row in task_rows if int(row["result"]) in (0, 1)]
    total_skips = sum(int(row["result"]) not in (0, 1) for row in task_rows)
    category_rows.append({
        "suite": suite_name,
        "category": "Total",
        "num_tasks": len(task_rows),
        "num_valid": len(total_values),
        "num_skipped": total_skips,
        "successes": int(sum(total_values)),
        "avg_success": float(np.mean(total_values)) if total_values else None,
    })
    return task_rows, category_rows


def _analysis_dir() -> Path:
    log_dir = os.environ.get("LOG_DIR", "").strip()
    if not log_dir:
        raise RuntimeError("LIBERO-Plus LR-NODE evaluation requires LOG_DIR")
    path = Path(log_dir) / "analysis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_live_progress(
    *, args, local_pairs: Sequence[tuple[int, int]], local_assigned: int,
    total_tasks: int, last_task_id: int,
) -> None:
    rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    payload = {
        "schema_version": 1,
        "benchmark_variant": "LIBERO-Plus",
        "suite": "libero_10",
        "run_name": str(args.run_name),
        "rank": rank,
        "world_size": world_size,
        "total_tasks": total_tasks,
        "local_assigned": local_assigned,
        "local_completed": len(local_pairs),
        "local_successes": sum(result == 1 for result, _ in local_pairs),
        "local_failures": sum(result == 0 for result, _ in local_pairs),
        "local_skips": sum(result not in (0, 1) for result, _ in local_pairs),
        "last_task_id": int(last_task_id),
        "status": "rank_complete" if len(local_pairs) == local_assigned else "running",
        "updated_at_unix_s": time.time(),
    }
    analysis_dir = _analysis_dir()
    _atomic_write_json(analysis_dir / f"eval_progress_rank{rank}.json", payload)
    lock_path = analysis_dir / ".eval_progress.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        rank_payloads = []
        for rank_id in range(world_size):
            path = analysis_dir / f"eval_progress_rank{rank_id}.json"
            if not path.is_file():
                continue
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if int(item.get("total_tasks", -1)) == total_tasks:
                rank_payloads.append(item)
        completed = sum(int(item["local_completed"]) for item in rank_payloads)
        aggregate = {
            "schema_version": 1,
            "benchmark_variant": "LIBERO-Plus",
            "suite": "libero_10",
            "status": "complete" if completed == total_tasks else "running",
            "metric_scope": "completed_tasks_only",
            "total_tasks": total_tasks,
            "completed": completed,
            "successes": sum(int(item["local_successes"]) for item in rank_payloads),
            "failures": sum(int(item["local_failures"]) for item in rank_payloads),
            "skips": sum(int(item["local_skips"]) for item in rank_payloads),
            "ranks_reporting": len(rank_payloads),
            "world_size": world_size,
            "updated_at_unix_s": time.time(),
        }
        valid = aggregate["successes"] + aggregate["failures"]
        aggregate["completed_valid_success_rate"] = (
            float(aggregate["successes"]) / valid if valid else 0.0
        )
        _atomic_write_json(analysis_dir / "eval_progress.json", aggregate)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    print(
        "[LIBERO-PLUS PROGRESS] "
        f"completed={aggregate['completed']}/{total_tasks} success={aggregate['successes']} "
        f"fail={aggregate['failures']} skip={aggregate['skips']} "
        f"valid_sr={100.0 * aggregate['completed_valid_success_rate']:.2f}%",
        flush=True,
    )


def _build_wrapper(args, model, image_processor, tokenizer) -> canonical.ModelWrapper:
    return canonical.ModelWrapper(
        model, tokenizer, image_processor, get_cast_dtype(args.precision),
        history_len=args.sequence_length,
        use_ensembling=args.eval_libero_ensembling,
        ensembling_temp=args.ensembling_temp,
        libero_eval_max_steps=args.libero_eval_max_steps,
        action_pred_steps=args.action_pred_steps,
        gripper_width=args.gripper_width,
        use_lrnode_latent_update=args.use_lrnode_latent_update,
        lrnode_eval_skip_full_forward=args.lrnode_eval_skip_full_forward,
        lrnode_query_interval=args.lrnode_query_interval,
        lrnode_eval_step_log=args.lrnode_eval_step_log,
        lrnode_eval_shadow_full_forward=args.lrnode_eval_shadow_full_forward,
        lrnode_eval_profile_full_action_head=args.lrnode_eval_profile_full_action_head,
        lrnode_eval_refresh_policy=args.lrnode_eval_refresh_policy,
        lrnode_eval_max_full_forwards_per_episode=args.lrnode_eval_max_full_forwards_per_episode,
        lrnode_eval_ablation_mode=args.lrnode_eval_ablation_mode,
        lrnode_no_delta_mode=args.lrnode_no_delta_mode,
        lrnode_chunk_token_policy=args.lrnode_chunk_token_policy,
        lrnode_mechanism_trace=args.lrnode_mechanism_trace,
        lrnode_trace_save_latents=args.lrnode_trace_save_latents,
        lrnode_trace_episode_limit=args.lrnode_trace_episode_limit,
        lrnode_trace_output_dir=args.lrnode_trace_output_dir,
        lrnode_counterfactual_mode=args.lrnode_counterfactual_mode,
        lrnode_counterfactual_mix_stage=args.lrnode_counterfactual_mix_stage,
        lrnode_latent_fusion_alpha=args.lrnode_latent_fusion_alpha,
        lrnode_latent_fusion_mode=args.lrnode_latent_fusion_mode,
        lrnode_matched_random_seed=args.lrnode_matched_random_seed,
        lrnode_matched_random_norm_mode=args.lrnode_matched_random_norm_mode,
        lrnode_every_step_filter_mode=args.lrnode_every_step_filter_mode,
        lrnode_every_step_filter_alpha=args.lrnode_every_step_filter_alpha,
        lrnode_every_step_filter_beta=args.lrnode_every_step_filter_beta,
        lrnode_every_step_filter_diagnostics=args.lrnode_every_step_filter_diagnostics,
    )


def _annotate_results(
    *, args, suite_task_count: int, evaluated_task_count: int,
    task_rows: Sequence[Mapping[str, Any]], category_rows: Sequence[Mapping[str, Any]],
    skips: Sequence[Mapping[str, Any]],
) -> None:
    analysis_dir = _analysis_dir()
    metadata = {
        "protocol": "libero_plus_long_eval_only",
        "suite": "libero_10",
        "suite_task_count": suite_task_count,
        "evaluated_task_count": evaluated_task_count,
        "episodes_per_task": 1,
        "init_state_index": 0,
        "task_order": "LIBERO-Plus benchmark order",
        "policy_step_limit": int(args.libero_eval_max_steps),
        "settle_steps": int(canonical._settle_steps(canonical._eval_control_hz())),
        "valid_results": sum(int(row["result"]) in (0, 1) for row in task_rows),
        "skipped_results": len(skips),
        "classification_file": str(_classification_path(args)),
        "training_on_libero_plus": False,
    }
    for path in [analysis_dir / "eval_summary.json", *analysis_dir.glob("*_eval.json")]:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["benchmark_variant"] = "LIBERO-Plus"
        payload["libero_plus"] = metadata
        payload["libero_plus_category_results"] = list(category_rows)
        _atomic_write_json(path, payload)


def _gather(value, rank: int, world_size: int):
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    torch.distributed.gather_object(value, gathered, dst=0)
    return gathered


def _partition_task_ids(task_count: int, world_size: int, rank: int) -> list[int]:
    mode = os.environ.get("LIBERO_PLUS_SHARD_MODE", "round_robin").strip().lower()
    if mode == "round_robin":
        return list(range(rank, task_count, world_size))
    if mode == "contiguous":
        return np.array_split(list(range(task_count)), world_size)[rank].tolist()
    raise ValueError(
        f"LIBERO_PLUS_SHARD_MODE must be round_robin or contiguous, got {mode!r}"
    )


def _wait_for_rank_completion(
    *, args, total_tasks: int, world_size: int, rank: int,
) -> None:
    """Keep early ranks out of NCCL collectives while slower task shards finish."""
    timeout_s = float(os.environ.get("LIBERO_PLUS_COMPLETION_WAIT_SECONDS", "43200"))
    poll_s = float(os.environ.get("LIBERO_PLUS_COMPLETION_POLL_SECONDS", "10"))
    if timeout_s <= 0 or poll_s <= 0:
        raise ValueError("LIBERO-Plus completion wait and poll seconds must be positive")

    analysis_dir = _analysis_dir()
    deadline = time.monotonic() + timeout_s
    next_log = 0.0
    while True:
        complete_ranks: list[int] = []
        progress: dict[int, str] = {}
        for rank_id in range(world_size):
            path = analysis_dir / f"eval_progress_rank{rank_id}.json"
            if not path.is_file():
                progress[rank_id] = "missing"
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                progress[rank_id] = "unreadable"
                continue
            if (
                payload.get("run_name") != str(args.run_name)
                or int(payload.get("total_tasks", -1)) != total_tasks
                or int(payload.get("world_size", -1)) != world_size
            ):
                progress[rank_id] = "mismatched"
                continue
            local_completed = int(payload.get("local_completed", -1))
            local_assigned = int(payload.get("local_assigned", -1))
            progress[rank_id] = f"{local_completed}/{local_assigned}"
            if payload.get("status") == "rank_complete" and local_completed == local_assigned:
                complete_ranks.append(rank_id)

        if len(complete_ranks) == world_size:
            if rank == 0:
                print(
                    "[LIBERO-PLUS SYNC] all rank shards complete; entering final gather",
                    flush=True,
                )
            return

        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(
                "Timed out waiting for LIBERO-Plus rank completion before final gather: "
                f"timeout_s={timeout_s}, progress={progress}"
            )
        if rank == 0 and now >= next_log:
            print(
                "[LIBERO-PLUS SYNC] waiting for rank shards before final gather: "
                f"{progress}",
                flush=True,
            )
            next_log = now + 300.0
        time.sleep(poll_s)


def evaluate_policy_ddp(args, model: canonical.ModelWrapper) -> None:
    if args.finetune_type != "libero_10":
        raise ValueError("This protocol is intentionally restricted to LIBERO-Plus libero_10")
    task_suite = benchmark.get_benchmark_dict()["libero_10"]()
    suite_task_count = int(task_suite.n_tasks)
    max_tasks = int(os.environ.get("LIBERO_PLUS_MAX_TASKS", "0"))
    if max_tasks < 0:
        raise ValueError(f"LIBERO_PLUS_MAX_TASKS must be non-negative, got {max_tasks}")
    task_count = suite_task_count if max_tasks == 0 else min(max_tasks, suite_task_count)
    if task_count <= 0:
        raise ValueError("No LIBERO-Plus tasks selected")

    world_size = int(torch.distributed.get_world_size())
    rank = int(torch.distributed.get_rank())
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    local_task_ids = _partition_task_ids(task_count, world_size, rank)
    classification = load_task_classification(args).get("libero_10", {})
    if len(classification) < suite_task_count:
        raise RuntimeError(f"Classification covers {len(classification)}/{suite_task_count} tasks")

    control_hz = canonical._eval_control_hz()
    settle_steps = canonical._settle_steps(control_hz)
    horizon = canonical._env_horizon(args.libero_eval_max_steps, settle_steps)
    render_device = canonical._renderer_gpu_device_id(local_rank)
    progress_interval = max(1, int(os.environ.get("LIBERO_PLUS_PROGRESS_INTERVAL", "5")))
    if rank == 0:
        print(
            "[LIBERO-PLUS PROTOCOL] "
            f"suite_tasks={suite_task_count} selected_tasks={task_count} "
            f"episodes_per_task=1 init_state_index=0 world_size={world_size} "
            f"shard_mode={os.environ.get('LIBERO_PLUS_SHARD_MODE', 'round_robin')} "
            f"control_hz={control_hz:.2f} settle_steps={settle_steps} "
            f"policy_limit={args.libero_eval_max_steps} horizon={horizon}", flush=True,
        )

    local_pairs: list[tuple[int, int]] = []
    local_episode_metrics: list[dict[str, Any]] = []
    local_skips: list[dict[str, Any]] = []
    for local_index, task_id in enumerate(local_task_ids):
        task = task_suite.get_task(task_id)
        env = None
        result = -1
        try:
            bddl_path = os.path.join(
                args.libero_path, "libero", "libero", "bddl_files",
                task.problem_folder, task.bddl_file,
            )
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=args.libero_img_size,
                camera_widths=args.libero_img_size,
                render_gpu_device_id=render_device,
                control_freq=int(round(control_hz)),
                horizon=horizon,
            )
            env.exp_id = 0
            env.task_id = task_id
            env.task_name = task.name
            env.task_suite_name = "libero_10"
            env.reset()
            canonical.verify_renderer_backend(env, render_device)
            env.seed(args.seed)
            init_states = task_suite.get_task_init_states(task_id)
            if len(init_states) == 0:
                raise RuntimeError(f"Task {task_id} has no initial states")
            obs = env.set_init_state(init_states[0])
            for _ in range(settle_steps):
                env.step(np.zeros(7))
            result, episode_metrics = canonical.evaluate_libero_task(task, env, obs, args, model)
            env = None
            local_episode_metrics.append(episode_metrics)
        except Exception as exc:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            local_skips.append({
                "rank": rank,
                "task_id": task_id,
                "task_name": task.name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(
                f"[LIBERO-PLUS][SKIP] rank={rank} task_id={task_id} task={task.name} "
                f"error={type(exc).__name__}: {exc}", flush=True,
            )
        local_pairs.append((int(result), int(task_id)))
        if (local_index + 1) % progress_interval == 0 or local_index + 1 == len(local_task_ids):
            _write_live_progress(
                args=args, local_pairs=local_pairs, local_assigned=len(local_task_ids),
                total_tasks=task_count, last_task_id=task_id,
            )

    _wait_for_rank_completion(
        args=args, total_tasks=task_count, world_size=world_size, rank=rank,
    )
    all_pairs = _gather(local_pairs, rank, world_size)
    all_stats = _gather(model.get_lrnode_stats(), rank, world_size)
    all_renderer = _gather(canonical.get_renderer_backend_metadata(), rank, world_size)
    all_episode_metrics = _gather(local_episode_metrics, rank, world_size)
    all_skips = _gather(local_skips, rank, world_size)
    if rank != 0:
        return

    result_pairs = sorted(
        [item for rank_items in all_pairs for item in rank_items], key=lambda item: int(item[1])
    )
    actual_ids = [int(item[1]) for item in result_pairs]
    if actual_ids != list(range(task_count)):
        raise RuntimeError(f"LIBERO-Plus result coverage mismatch: got {actual_ids[:20]}...")
    task_records = [
        {"task_id": task_id, "task_name": task_suite.get_task(task_id).name, "result": result}
        for result, task_id in result_pairs
    ]
    task_rows, category_rows = build_plus_result_tables(
        suite_name="libero_10", task_records=task_records, classification=classification
    )
    analysis_dir = _analysis_dir()
    _write_csv(
        analysis_dir / "libero_plus_task_results.csv",
        ["suite", "task_id", "task_name", "result", "status", "categories",
         "difficulty_level", "classification_id"], task_rows,
    )
    _write_csv(
        analysis_dir / "libero_plus_category_results.csv",
        ["suite", "category", "num_tasks", "num_valid", "num_skipped", "successes",
         "avg_success"], category_rows,
    )
    flattened_metrics = [item for rank_items in all_episode_metrics for item in rank_items]
    flattened_skips = [item for rank_items in all_skips for item in rank_items]
    _atomic_write_json(analysis_dir / "libero_plus_skips.json", {"skips": flattened_skips})

    # The canonical writer predates Plus and reads these globals for task rows.
    canonical.task_num = task_count
    canonical.num_eval_episodes = 1
    canonical.save_eval_json(
        args, result_pairs, task_suite, all_stats, flattened_metrics, all_renderer
    )
    _annotate_results(
        args=args, suite_task_count=suite_task_count, evaluated_task_count=task_count,
        task_rows=task_rows, category_rows=category_rows, skips=flattened_skips,
    )
    _atomic_write_json(analysis_dir / "eval_progress.json", {
        "schema_version": 1,
        "benchmark_variant": "LIBERO-Plus",
        "suite": "libero_10",
        "status": "complete",
        "metric_scope": "completed_tasks_only",
        "total_tasks": task_count,
        "completed": task_count,
        "successes": sum(int(row["result"]) == 1 for row in task_rows),
        "failures": sum(int(row["result"]) == 0 for row in task_rows),
        "skips": len(flattened_skips),
        "ranks_reporting": world_size,
        "world_size": world_size,
        "updated_at_unix_s": time.time(),
    })
    print("\n[LIBERO-PLUS CATEGORY SUMMARY]")
    for row in category_rows:
        value = row["avg_success"]
        formatted = "NA" if value is None else f"{100.0 * float(value):.2f}%"
        print(
            f"{row['category']}: {formatted} "
            f"({row['successes']}/{row['num_valid']}, skipped={row['num_skipped']})"
        )


def eval_one_epoch_libero_plus_ddp(args, model, image_processor, tokenizer) -> None:
    control_hz = canonical._eval_control_hz()
    args.libero_eval_max_steps = canonical._scaled_step_count(
        int(args.libero_eval_max_steps), control_hz
    )
    evaluate_policy_ddp(args, _build_wrapper(args, model, image_processor, tokenizer))
