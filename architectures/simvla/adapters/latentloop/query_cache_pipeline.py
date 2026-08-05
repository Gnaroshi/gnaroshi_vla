"""Four-GPU smoke, pilot, production, and validation cache pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.source_lock import collect_source_lock
from methods.latentloop.training.query_cache_dataset import (
    load_manifest,
    merge_query_cache_parts,
    validate_query_cache,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_NVME_PREFIX = Path(
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla"
)
LOGGER = logging.getLogger("simvla_latentloop_cache_pipeline")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_logging(run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    terminal = logging.StreamHandler(sys.stdout)
    terminal.setFormatter(formatter)
    file_handler = logging.FileHandler(run_root / "cache_pipeline.log", mode="a")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(terminal)
    LOGGER.addHandler(file_handler)


def _parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain distinct physical GPU indices")
    return gpus


def _gpu_snapshot(requested: Sequence[str]) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    available: dict[str, dict[str, Any]] = {}
    for line in output.splitlines():
        index, name, memory_used, memory_total = [item.strip() for item in line.split(",", 3)]
        available[index] = {
            "index": index,
            "name": name,
            "memory_used_mib": int(memory_used),
            "memory_total_mib": int(memory_total),
        }
    missing = [index for index in requested if index not in available]
    if missing:
        raise RuntimeError(f"Requested GPUs are unavailable: {missing}")
    return [available[index] for index in requested]


def _tail(path: Path, lines: int = 60) -> str:
    if not path.exists():
        return "<worker log was not created>"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _episode_progress(worker_dirs: Sequence[Path]) -> dict[str, int]:
    summaries: dict[str, dict[str, Any]] = {}
    for worker_dir in worker_dirs:
        episodes_root = worker_dir / "episodes"
        if not episodes_root.exists():
            continue
        for path in episodes_root.glob("task*_trial*/episode_summary.json"):
            payload = _read_json(path)
            if payload.get("committed") and payload.get("validation", {}).get("passed"):
                summaries[str(payload["episode_id"])] = payload
    return {
        "episodes": len(summaries),
        "records": sum(int(item["records"]) for item in summaries.values()),
        "queries": sum(int(item["queries"]) for item in summaries.values()),
        "cache_bytes": sum(int(item["cache_bytes"]) for item in summaries.values()),
        "successes": sum(int(item["success"]) for item in summaries.values()),
    }


def _existing_phase_cache_bytes(phase_root: Path) -> int:
    parts_root = phase_root / "parts"
    if not parts_root.exists():
        return 0
    return _episode_progress(sorted(parts_root.glob("worker*")))["cache_bytes"]


def _terminate_processes(processes: Sequence[subprocess.Popen[str]]) -> None:
    running = [process for process in processes if process.poll() is None]
    for process in running:
        process.terminate()
    deadline = time.time() + 15.0
    while running and time.time() < deadline:
        running = [process for process in running if process.poll() is None]
        time.sleep(0.2)
    for process in running:
        process.kill()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _phase_config(
    args: argparse.Namespace,
    *,
    name: str,
    execution_horizon: int,
    num_trials: int,
    max_policy_queries: int,
    gpus: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": "simvla_parallel_query_cache_phase_v1",
        "name": name,
        "checkpoint": args.checkpoint,
        "norm_stats": str(Path(args.norm_stats).resolve()),
        "suite": args.suite,
        "execution_horizon": execution_horizon,
        "num_trials": num_trials,
        "max_tasks": args.max_tasks,
        "max_policy_queries": max_policy_queries,
        "num_wait_steps": args.num_wait_steps,
        "flow_steps": args.flow_steps,
        "client_resize_size": args.client_resize_size,
        "image_size": args.image_size,
        "resolution": args.resolution,
        "control_hz": args.control_hz,
        "seed": args.seed,
        "action_noise_seed_base": args.action_noise_seed_base,
        "task_order": args.task_order,
        "records_per_shard": args.records_per_shard,
        "gpus": list(gpus),
        "workers": len(gpus),
    }


def _worker_finished(worker_dir: Path) -> bool:
    path = worker_dir / "generation_summary.json"
    return path.exists() and bool(_read_json(path).get("validation", {}).get("passed"))


def _run_parallel_phase(
    args: argparse.Namespace,
    *,
    name: str,
    execution_horizon: int,
    num_trials: int,
    max_policy_queries: int,
    gpus: Sequence[str],
    max_worker_cache_gib: float = 0.0,
) -> dict[str, Any]:
    phase_started = time.time()
    phase_root = Path(args.cache_root) / name
    parts_root = phase_root / "parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    phase_config = _phase_config(
        args,
        name=name,
        execution_horizon=execution_horizon,
        num_trials=num_trials,
        max_policy_queries=max_policy_queries,
        gpus=gpus,
    )
    config_path = phase_root / "pipeline_phase_config.json"
    if config_path.exists() and _read_json(config_path) != phase_config:
        raise RuntimeError(f"Cannot resume phase with changed configuration: {config_path}")
    _write_json_atomic(config_path, phase_config)

    completed_path = phase_root / "generation_summary.json"
    if completed_path.exists():
        completed = _read_json(completed_path)
        if completed.get("validation", {}).get("passed"):
            LOGGER.info("Skipping completed phase %s", name)
            return completed

    total_episodes = args.max_tasks * num_trials
    if len(gpus) > total_episodes:
        raise RuntimeError(
            f"phase {name} has {total_episodes} episodes but {len(gpus)} workers"
        )
    worker_dirs = [parts_root / f"worker{index:02d}" for index in range(len(gpus))]
    logs_root = Path(args.run_root) / "worker_logs" / name
    logs_root.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[str]] = []
    process_logs: dict[subprocess.Popen[str], tuple[Path, Any]] = {}
    for worker_index, (gpu, worker_dir) in enumerate(zip(gpus, worker_dirs)):
        if _worker_finished(worker_dir):
            LOGGER.info("Phase %s worker %d already complete", name, worker_index)
            continue
        log_path = logs_root / f"worker{worker_index:02d}_gpu{gpu}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "architectures.simvla.adapters.latentloop.query_cache_generator",
            "--output",
            str(worker_dir),
            "--checkpoint",
            args.checkpoint,
            "--smolvlm-model-path",
            args.smolvlm_model_path,
            "--norm-stats",
            args.norm_stats,
            "--suite",
            args.suite,
            "--execution-horizon",
            str(execution_horizon),
            "--num-trials",
            str(num_trials),
            "--max-tasks",
            str(args.max_tasks),
            "--max-policy-queries",
            str(max_policy_queries),
            "--num-wait-steps",
            str(args.num_wait_steps),
            "--flow-steps",
            str(args.flow_steps),
            "--client-resize-size",
            str(args.client_resize_size),
            "--image-size",
            str(args.image_size),
            "--resolution",
            str(args.resolution),
            "--control-hz",
            str(args.control_hz),
            "--seed",
            str(args.seed),
            "--action-noise-seed-base",
            str(args.action_noise_seed_base),
            "--task-order",
            args.task_order,
            "--records-per-shard",
            str(args.records_per_shard),
            "--worker-index",
            str(worker_index),
            "--num-workers",
            str(len(gpus)),
            "--resume",
            "--disable-tqdm",
            "--device",
            "cuda",
        ]
        if max_worker_cache_gib > 0:
            command.extend(["--max-cache-gib", str(max_worker_cache_gib)])
        log_handle.write("COMMAND: " + " ".join(command) + "\n")
        log_handle.flush()
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        processes.append(process)
        process_logs[process] = (log_path, log_handle)
        LOGGER.info(
            "Started phase %s worker %d on physical GPU %s (pid=%d)",
            name,
            worker_index,
            gpu,
            process.pid,
        )

    progress = tqdm(
        total=total_episodes,
        desc=name,
        dynamic_ncols=True,
        mininterval=args.tqdm_mininterval,
        initial=_episode_progress(worker_dirs)["episodes"],
    )
    try:
        while any(process.poll() is None for process in processes):
            stats = _episode_progress(worker_dirs)
            progress.n = stats["episodes"]
            progress.set_postfix(
                records=stats["records"],
                queries=stats["queries"],
                success=stats["successes"],
                GiB=f"{stats['cache_bytes'] / 2**30:.1f}",
            )
            progress.refresh()
            failed = [process for process in processes if process.poll() not in (None, 0)]
            if failed:
                _terminate_processes(processes)
                process = failed[0]
                log_path, _ = process_logs[process]
                raise RuntimeError(
                    f"Cache worker failed with exit={process.returncode}: {log_path}\n"
                    + _tail(log_path)
                )
            time.sleep(args.poll_seconds)
        failed = [process for process in processes if process.returncode != 0]
        if failed:
            process = failed[0]
            log_path, _ = process_logs[process]
            raise RuntimeError(
                f"Cache worker failed with exit={process.returncode}: {log_path}\n"
                + _tail(log_path)
            )
    except BaseException:
        _terminate_processes(processes)
        raise
    finally:
        progress.close()
        for _, log_handle in process_logs.values():
            log_handle.close()

    missing = [str(path) for path in worker_dirs if not _worker_finished(path)]
    if missing:
        raise RuntimeError(f"Workers did not produce valid summaries: {missing}")
    worker_summaries = [_read_json(path / "generation_summary.json") for path in worker_dirs]
    source_lock = collect_source_lock(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    manifest = merge_query_cache_parts(
        phase_root,
        worker_dirs,
        metadata={
            "source_lock": source_lock,
            "parallel_phase_config": phase_config,
            "worker_count": len(worker_dirs),
        },
    )
    validation = validate_query_cache(phase_root)
    summary = {
        "cache_dir": str(phase_root),
        "records": int(manifest["total_records"]),
        "episodes": sum(int(item["episodes"]) for item in worker_summaries),
        "successes": sum(int(item["successes"]) for item in worker_summaries),
        "queries": sum(int(item["queries"]) for item in worker_summaries),
        "cache_bytes": sum(int(item["cache_bytes"]) for item in worker_summaries),
        "elapsed_seconds_this_invocation": time.time() - phase_started,
        "phase_config": phase_config,
        "worker_summaries": worker_summaries,
        "validation": validation,
    }
    _write_json_atomic(phase_root / "source_lock.json", source_lock)
    _write_json_atomic(completed_path, summary)
    if not validation["passed"]:
        raise RuntimeError(f"Merged cache validation failed for {name}: {validation['errors'][:20]}")
    LOGGER.info(
        "Completed %s: episodes=%d records=%d size=%.2f GiB",
        name,
        summary["episodes"],
        summary["records"],
        summary["cache_bytes"] / 2**30,
    )
    return summary


def _projection(
    summary: dict[str, Any],
    *,
    target_episodes: int,
    max_records_per_episode: int,
) -> dict[str, Any]:
    records = int(summary["records"])
    episodes = int(summary["episodes"])
    cache_bytes = int(summary["cache_bytes"])
    if records <= 0 or episodes <= 0 or cache_bytes <= 0:
        raise RuntimeError(f"Pilot cache is empty or invalid: {summary['cache_dir']}")
    bytes_per_record = cache_bytes / records
    records_per_episode = records / episodes
    projected_mean = bytes_per_record * records_per_episode * target_episodes
    projected_worst = bytes_per_record * max_records_per_episode * target_episodes
    return {
        "pilot_cache_dir": summary["cache_dir"],
        "pilot_episodes": episodes,
        "pilot_records": records,
        "pilot_cache_bytes": cache_bytes,
        "bytes_per_record": bytes_per_record,
        "records_per_episode": records_per_episode,
        "target_episodes": target_episodes,
        "projected_mean_bytes": int(projected_mean),
        "projected_mean_gib": projected_mean / 2**30,
        "projected_worst_bytes": int(projected_worst),
        "projected_worst_gib": projected_worst / 2**30,
    }


def _preflight_production(
    args: argparse.Namespace,
    *,
    pilot_r1: dict[str, Any],
    pilot_r5: dict[str, Any],
) -> dict[str, Any]:
    target_episodes = args.max_tasks * args.production_trials
    r1 = _projection(
        pilot_r1,
        target_episodes=target_episodes,
        max_records_per_episode=args.r1_max_policy_queries - 1,
    )
    r5 = _projection(
        pilot_r5,
        target_episodes=target_episodes,
        max_records_per_episode=args.r5_max_policy_queries - 1,
    )
    projected_mean = r1["projected_mean_bytes"] + r5["projected_mean_bytes"]
    projected_with_margin = int(projected_mean * args.projection_safety_factor)
    projected_worst = r1["projected_worst_bytes"] + r5["projected_worst_bytes"]
    disk = shutil.disk_usage(args.cache_root)
    budget_bytes = int(args.max_production_cache_gib * 2**30)
    reserve_bytes = int(args.min_free_after_gib * 2**30)
    gate_bytes = max(projected_with_margin, projected_worst)
    cache_root = Path(args.cache_root)
    existing_production_bytes = _existing_phase_cache_bytes(
        cache_root / f"query_v3_r1_full_10x{args.production_trials}"
    ) + _existing_phase_cache_bytes(
        cache_root / f"query_v3_r5_full_10x{args.production_trials}"
    )
    remaining_gate_bytes = max(0, gate_bytes - existing_production_bytes)
    errors: list[str] = []
    if gate_bytes > budget_bytes:
        errors.append(
            f"production projection gate={gate_bytes / 2**30:.2f} GiB exceeds "
            f"budget={args.max_production_cache_gib:.2f} GiB"
        )
    if disk.free - remaining_gate_bytes < reserve_bytes:
        errors.append(
            f"shared NVMe would retain less than {args.min_free_after_gib:.2f} GiB "
            "under the conservative projection"
        )
    result = {
        "passed": not errors,
        "errors": errors,
        "r1": r1,
        "r5": r5,
        "target_trials_per_task": args.production_trials,
        "target_episodes_per_cache": target_episodes,
        "projection_safety_factor": args.projection_safety_factor,
        "projected_mean_total_gib": projected_mean / 2**30,
        "projected_with_margin_total_gib": projected_with_margin / 2**30,
        "projected_worst_total_gib": projected_worst / 2**30,
        "max_production_cache_gib": args.max_production_cache_gib,
        "existing_production_cache_gib": existing_production_bytes / 2**30,
        "remaining_conservative_write_gib": remaining_gate_bytes / 2**30,
        "shared_disk_free_gib": disk.free / 2**30,
        "minimum_free_after_gib": args.min_free_after_gib,
        "projected_free_after_worst_gib": (disk.free - remaining_gate_bytes) / 2**30,
    }
    _write_json_atomic(Path(args.run_root) / "production_cache_preflight.json", result)
    if errors:
        raise RuntimeError("Production cache preflight failed: " + "; ".join(errors))
    LOGGER.info(
        "Production preflight passed: mean=%.1f GiB margin=%.1f GiB worst=%.1f GiB free=%.1f GiB",
        result["projected_mean_total_gib"],
        result["projected_with_margin_total_gib"],
        result["projected_worst_total_gib"],
        result["shared_disk_free_gib"],
    )
    return result


def _save_final_validation(
    args: argparse.Namespace,
    *,
    name: str,
    cache_dir: Path,
) -> dict[str, Any]:
    validation = validate_query_cache(cache_dir)
    output = Path(args.run_root) / "cache_validation" / name
    output.mkdir(parents=True, exist_ok=True)
    source_lock = collect_source_lock(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    cache_source_lock = load_manifest(cache_dir).get("metadata", {}).get("source_lock", {})
    _write_json_atomic(output / "cache_validation.json", validation)
    _write_json_atomic(output / "source_lock.json", source_lock)
    _write_json_atomic(output / "cache_source_lock.json", cache_source_lock)
    if not validation["passed"]:
        raise RuntimeError(f"Final validation failed for {name}: {validation['errors'][:20]}")
    return validation


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    cache_root = Path(args.cache_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    args.cache_root = str(cache_root)
    args.run_root = str(run_root)
    _configure_logging(run_root)
    if os.environ.get("CONDA_DEFAULT_ENV") != args.conda_env:
        raise RuntimeError(
            f"Expected conda env {args.conda_env}, got {os.environ.get('CONDA_DEFAULT_ENV')}"
        )
    required_prefix = Path(args.required_cache_prefix).expanduser().resolve()
    try:
        cache_root.relative_to(required_prefix)
    except ValueError as exc:
        raise RuntimeError(
            f"Cache root must stay on shared NVMe below {required_prefix}: {cache_root}"
        ) from exc
    Path(args.norm_stats).resolve(strict=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    gpus = _parse_gpus(args.gpus)
    gpu_snapshot = _gpu_snapshot(gpus)
    busy = [
        item
        for item in gpu_snapshot
        if int(item["memory_used_mib"]) > args.max_initial_gpu_memory_mib
    ]
    if busy:
        raise RuntimeError(f"Requested GPUs are already using too much memory: {busy}")
    disk = shutil.disk_usage(cache_root)
    if disk.free < args.min_free_after_gib * 2**30:
        raise RuntimeError(
            f"Shared NVMe free space is below the reserve: {disk.free / 2**30:.1f} GiB"
        )
    initial = {
        "cache_root": str(cache_root),
        "run_root": str(run_root),
        "gpus": gpu_snapshot,
        "shared_disk_free_gib": disk.free / 2**30,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    _write_json_atomic(run_root / "pipeline_preflight.json", initial)
    LOGGER.info("Cache root: %s", cache_root)
    LOGGER.info("Run metadata/log root: %s", run_root)
    LOGGER.info("Physical GPUs: %s", ",".join(gpus))
    if args.preflight_only:
        result = {"passed": True, "preflight_only": True, **initial}
        _write_json_atomic(run_root / "cache_pipeline_preflight_only.json", result)
        LOGGER.info("Preflight-only check complete; no cache rollout was started")
        return result

    smoke_r1 = _run_parallel_phase(
        args,
        name="query_v3_r1_smoke",
        execution_horizon=1,
        num_trials=1,
        max_policy_queries=3,
        gpus=gpus[:1],
    )
    smoke_r5 = _run_parallel_phase(
        args,
        name="query_v3_r5_smoke",
        execution_horizon=5,
        num_trials=1,
        max_policy_queries=3,
        gpus=gpus[:1],
    )
    pilot_r1 = _run_parallel_phase(
        args,
        name=f"query_v3_r1_pilot_10x{args.pilot_trials}",
        execution_horizon=1,
        num_trials=args.pilot_trials,
        max_policy_queries=args.r1_max_policy_queries,
        gpus=gpus,
    )
    pilot_r5 = _run_parallel_phase(
        args,
        name=f"query_v3_r5_pilot_10x{args.pilot_trials}",
        execution_horizon=5,
        num_trials=args.pilot_trials,
        max_policy_queries=args.r5_max_policy_queries,
        gpus=gpus,
    )
    preflight = _preflight_production(args, pilot_r1=pilot_r1, pilot_r5=pilot_r5)
    r1_worker_budget = args.max_production_cache_gib / len(gpus)
    production_r1_name = f"query_v3_r1_full_10x{args.production_trials}"
    production_r5_name = f"query_v3_r5_full_10x{args.production_trials}"
    production_r1 = _run_parallel_phase(
        args,
        name=production_r1_name,
        execution_horizon=1,
        num_trials=args.production_trials,
        max_policy_queries=args.r1_max_policy_queries,
        gpus=gpus,
        max_worker_cache_gib=r1_worker_budget,
    )
    remaining_production_gib = (
        args.max_production_cache_gib - production_r1["cache_bytes"] / 2**30
    )
    if remaining_production_gib <= 0:
        raise RuntimeError(
            "R=1 cache consumed the complete production cache budget before R=5"
        )
    r5_worker_budget = remaining_production_gib / len(gpus)
    production_r5 = _run_parallel_phase(
        args,
        name=production_r5_name,
        execution_horizon=5,
        num_trials=args.production_trials,
        max_policy_queries=args.r5_max_policy_queries,
        gpus=gpus,
        max_worker_cache_gib=r5_worker_budget,
    )
    validation_r1 = _save_final_validation(
        args,
        name="r1_full",
        cache_dir=cache_root / production_r1_name,
    )
    validation_r5 = _save_final_validation(
        args,
        name="r5_full",
        cache_dir=cache_root / production_r5_name,
    )
    final = {
        "passed": bool(validation_r1["passed"] and validation_r5["passed"]),
        "elapsed_seconds": time.time() - started,
        "cache_root": str(cache_root),
        "run_root": str(run_root),
        "smoke": {"r1": smoke_r1, "r5": smoke_r5},
        "pilot": {"r1": pilot_r1, "r5": pilot_r5},
        "production_preflight": preflight,
        "production": {"r1": production_r1, "r5": production_r5},
        "final_validation": {"r1": validation_r1, "r5": validation_r5},
    }
    _write_json_atomic(run_root / "cache_pipeline_summary.json", final)
    LOGGER.info(
        "Pipeline complete in %.2f hours. R1=%s R5=%s",
        final["elapsed_seconds"] / 3600,
        production_r1["cache_dir"],
        production_r5["cache_dir"],
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--required-cache-prefix", default=str(DEFAULT_NVME_PREFIX))
    parser.add_argument("--conda-env", default="simvla_libero")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument(
        "--norm-stats",
        default=str(ROOT / "architectures" / "simvla" / "upstream" / "norm_stats" / "libero_norm.json"),
    )
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--pilot-trials", type=int, default=2)
    parser.add_argument("--production-trials", type=int, default=20)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--r1-max-policy-queries", type=int, default=900)
    parser.add_argument("--r5-max-policy-queries", type=int, default=180)
    parser.add_argument("--num-wait-steps", type=int, default=10)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--client-resize-size", type=int, default=224)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-noise-seed-base", type=int, default=20260804)
    parser.add_argument("--task-order", choices=("official_reverse", "ascending"), default="official_reverse")
    parser.add_argument("--records-per-shard", type=int, default=128)
    parser.add_argument("--max-production-cache-gib", type=float, default=200.0)
    parser.add_argument("--min-free-after-gib", type=float, default=200.0)
    parser.add_argument("--projection-safety-factor", type=float, default=1.5)
    parser.add_argument("--max-initial-gpu-memory-mib", type=int, default=1024)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--tqdm-mininterval", type=float, default=1.0)
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required for this GPU/cache pipeline")
    result = run_pipeline(args)
    if result.get("preflight_only"):
        display = result
    else:
        display = {
            "passed": result["passed"],
            "elapsed_seconds": result["elapsed_seconds"],
            "cache_root": result["cache_root"],
            "run_root": result["run_root"],
            "r1_cache": result["production"]["r1"]["cache_dir"],
            "r5_cache": result["production"]["r5"]["cache_dir"],
            "summary": str(Path(result["run_root"]) / "cache_pipeline_summary.json"),
        }
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
