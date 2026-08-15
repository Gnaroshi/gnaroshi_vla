"""Rollout-generated, resumable SimVLA query-boundary teacher cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
LIBERO_ROOT = UPSTREAM / "evaluation" / "libero" / "LIBERO"
for path in (ROOT, UPSTREAM, LIBERO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.dcld import (  # noqa: E402
    SimVLAActionAdapter,
    SimVLAConditionAdapter,
)
from architectures.simvla.adapters.latentloop.action_adapter import (  # noqa: E402
    ActionNoiseKey,
    executed_subchunk,
    explicit_action_noise,
)
from architectures.simvla.adapters.latentloop.determinism import (  # noqa: E402
    configure_strict_determinism,
    episode_env_seed,
    evaluation_episode_seed,
    resolve_seed_plan,
    seed_all,
)
from architectures.simvla.adapters.latentloop.source_lock import (  # noqa: E402
    collect_source_lock,
    require_empty_output,
    write_source_lock,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (  # noqa: E402
    RealSimVLADCLDPolicy,
    build_env_obs,
    get_libero_env,
)
from methods.latentloop.training.query_cache_dataset import (  # noqa: E402
    QueryCacheShardWriter,
    load_manifest,
    merge_query_cache_parts,
    tensor_sha256,
    validate_query_cache,
)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def partition_episode_specs(
    task_ids: Sequence[int],
    num_trials: int,
    worker_index: int,
    num_workers: int,
) -> list[tuple[int, int, int]]:
    """Assign flattened task/trial episodes deterministically across workers."""

    if num_trials <= 0:
        raise ValueError("num_trials must be positive")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if not 0 <= worker_index < num_workers:
        raise ValueError("worker_index must be in [0, num_workers)")
    all_specs: list[tuple[int, int, int]] = []
    for task_id in task_ids:
        for trial_id in range(num_trials):
            all_specs.append((len(all_specs), int(task_id), trial_id))
    return [spec for spec in all_specs if spec[0] % num_workers == worker_index]


def _cache_bytes(cache_dir: Path) -> int:
    manifest = load_manifest(cache_dir)
    return sum((cache_dir / shard["file"]).stat().st_size for shard in manifest["shards"])


def query_snapshot(
    *,
    policy: RealSimVLADCLDPolicy,
    condition_adapter: SimVLAConditionAdapter,
    action_adapter: SimVLAActionAdapter,
    image0: np.ndarray,
    image1: np.ndarray,
    proprio: np.ndarray,
    prompt: str,
    checkpoint: str,
    task_id: int,
    episode_id: str,
    query_index: int,
    env_timestep: int,
    seed_base: int,
    flow_steps: int,
) -> dict[str, Any]:
    """Run the frozen teacher once at an official-preprocessed query boundary."""

    batch = policy.preprocess(image0, image1, proprio, prompt)
    with torch.no_grad():
        condition = condition_adapter.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
    noise_key = ActionNoiseKey(
        checkpoint=checkpoint,
        task_id=task_id,
        episode_id=episode_id,
        policy_query_index=query_index,
        seed_base=seed_base,
    )
    initial_noise = explicit_action_noise(
        noise_key,
        batch_size=1,
        action_horizon=action_adapter.num_actions,
        action_dim=action_adapter.dim_action,
        device=batch["proprio"].device,
        dtype=batch["proprio"].dtype,
    )
    with torch.no_grad():
        action_chunk = action_adapter.decode_action_from_condition(
            condition,
            batch["proprio"],
            steps=flow_steps,
            initial_noise=initial_noise,
        )
    return {
        "query_index": int(query_index),
        "absolute_env_timestep": int(env_timestep),
        "raw_rgb": (
            batch["raw_rgb"][0].detach().mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        ),
        "proprio": batch["proprio"][0].detach().cpu(),
        "full_condition": condition[0].detach().cpu(),
        "teacher_action_chunk": action_chunk[0].detach().cpu(),
        "initial_noise": initial_noise[0].detach().cpu(),
        "action_noise_hash": tensor_sha256(initial_noise[0]),
        "action_noise_seed": noise_key.seed(),
        "proprio_device": batch["proprio"],
        "condition_device": condition,
        "action_chunk_device": action_chunk,
    }


def transition_record(
    *,
    current: dict[str, Any],
    next_query: dict[str, Any],
    executed_actions: torch.Tensor,
    task_id: int,
    episode_id: str,
    task_identifier: str,
    language_instruction: str,
    execution_horizon: int,
    elapsed_time: float,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Build one serializable query-to-next-query teacher transition."""

    executed_cpu = executed_actions.detach().cpu()
    return {
        "task_id": int(task_id),
        "episode_id": episode_id,
        "query_index": int(current["query_index"]),
        "next_query_index": int(next_query["query_index"]),
        "absolute_env_timestep": int(current["absolute_env_timestep"]),
        "next_absolute_env_timestep": int(next_query["absolute_env_timestep"]),
        "language_instruction": language_instruction,
        "task_identifier": task_identifier,
        "raw_rgb": current["raw_rgb"],
        "proprio": current["proprio"],
        "full_condition": current["full_condition"],
        "teacher_action_chunk": current["teacher_action_chunk"],
        "initial_noise": current["initial_noise"],
        "action_noise_hash": current["action_noise_hash"],
        "executed_subchunk": executed_cpu,
        "executed_env_actions": executed_cpu.clone(),
        "execution_horizon": int(execution_horizon),
        "elapsed_time": float(elapsed_time),
        "next_raw_rgb": next_query["raw_rgb"],
        "next_proprio": next_query["proprio"],
        "next_full_condition": next_query["full_condition"],
        "next_teacher_action_chunk": next_query["teacher_action_chunk"],
        "next_initial_noise": next_query["initial_noise"],
        "next_action_noise_hash": next_query["action_noise_hash"],
        "provenance": provenance,
    }


def _worker_config(
    args: argparse.Namespace,
    *,
    task_ids: Sequence[int],
    protocol: dict[str, Any],
    source_lock: dict[str, Any],
    seed_plan: dict[str, Any],
    strict_runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "simvla_query_cache_worker_v1",
        "generator_sha256": _sha256_file(Path(__file__)),
        "checkpoint": args.checkpoint,
        "checkpoint_revision": source_lock["checkpoint"].get("revision"),
        "norm_stats_sha256": source_lock["norm_stats_sha256"],
        "root_commit": source_lock["root_commit"],
        "simvla_upstream_commit": source_lock["simvla_upstream_commit"],
        "suite": args.suite,
        "task_ids": [int(task_id) for task_id in task_ids],
        "num_trials": int(args.num_trials),
        "worker_index": int(args.worker_index),
        "num_workers": int(args.num_workers),
        "max_policy_queries": int(args.max_policy_queries),
        "seed": int(args.seed),
        "action_noise_seed_base": int(args.action_noise_seed_base),
        "experiment_seed": args.experiment_seed,
        "effective_seed_plan": seed_plan,
        "render_backend": args.render_backend,
        "strict_runtime": strict_runtime,
        "records_per_shard": int(args.records_per_shard),
        "protocol": protocol,
    }


def _prepare_worker_output(
    output: Path,
    *,
    resume: bool,
    config: dict[str, Any],
    source_lock: dict[str, Any],
) -> None:
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite nonempty cache worker: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "worker_config.json"
    if config_path.exists():
        observed = json.loads(config_path.read_text(encoding="utf-8"))
        if observed != config:
            raise RuntimeError(
                "Cannot resume cache worker with a different immutable configuration: "
                f"{config_path}"
            )
    elif any(output.iterdir()):
        raise RuntimeError(f"Nonempty cache worker has no worker_config.json: {output}")
    else:
        _write_json_atomic(config_path, config)
        _write_json_atomic(output / "source_lock.json", source_lock)
    episodes_root = output / "episodes"
    episodes_root.mkdir(exist_ok=True)
    for temporary in episodes_root.glob(".inprogress_*"):
        shutil.rmtree(temporary)


def _episode_summary(episode_dir: Path, config_fingerprint: str) -> dict[str, Any] | None:
    summary_path = episode_dir / "episode_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("committed") or not summary.get("validation", {}).get("passed"):
        return None
    if summary.get("worker_config_sha256") != config_fingerprint:
        raise RuntimeError(f"Episode cache configuration mismatch: {episode_dir}")
    return summary


def _commit_episode(
    output: Path,
    *,
    episode_id: str,
    records: list[dict[str, Any]],
    execution_horizon: int,
    records_per_shard: int,
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    episodes_root = output / "episodes"
    final_dir = episodes_root / episode_id
    if final_dir.exists():
        raise FileExistsError(f"Refusing to replace committed episode cache: {final_dir}")
    temporary = episodes_root / f".inprogress_{episode_id}_{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    writer = QueryCacheShardWriter(
        temporary,
        execution_horizon=execution_horizon,
        records_per_shard=records_per_shard,
        metadata=metadata,
    )
    for record in records:
        writer.add(record)
    writer.close()
    validation = validate_query_cache(temporary)
    if not validation["passed"]:
        raise RuntimeError(f"Episode cache validation failed: {validation['errors'][:10]}")
    validation["cache_dir"] = str(final_dir)
    committed_summary = {
        **summary,
        "records": len(records),
        "cache_bytes": _cache_bytes(temporary),
        "validation": validation,
        "committed": True,
    }
    _write_json_atomic(temporary / "episode_summary.json", committed_summary)
    temporary.replace(final_dir)
    return committed_summary


def _finalize_worker_cache(
    output: Path,
    *,
    assigned_specs: Sequence[tuple[int, int, int]],
    config: dict[str, Any],
    source_lock: dict[str, Any],
    protocol: dict[str, Any],
    config_fingerprint: str,
    started: float,
) -> dict[str, Any]:
    episode_dirs: list[Path] = []
    summaries: list[dict[str, Any]] = []
    for _, task_id, trial_id in assigned_specs:
        episode_dir = output / "episodes" / f"task{task_id:02d}_trial{trial_id:03d}"
        summary = _episode_summary(episode_dir, config_fingerprint)
        if summary is None:
            raise RuntimeError(f"Assigned episode was not committed: {episode_dir}")
        episode_dirs.append(episode_dir)
        summaries.append(summary)
    manifest = merge_query_cache_parts(
        output,
        episode_dirs,
        metadata={
            "source_lock": source_lock,
            "protocol": protocol,
            "worker_config": config,
            "worker_index": config["worker_index"],
            "num_workers": config["num_workers"],
        },
    )
    validation = validate_query_cache(output)
    summary = {
        "cache_dir": str(output),
        "records": int(manifest["total_records"]),
        "episodes": len(summaries),
        "successes": sum(int(item["success"]) for item in summaries),
        "success_rate": (
            sum(int(item["success"]) for item in summaries) / max(len(summaries), 1)
        ),
        "queries": sum(int(item["queries"]) for item in summaries),
        "cache_bytes": sum(int(item["cache_bytes"]) for item in summaries),
        "incomplete_terminal_transitions_discarded": sum(
            int(item["incomplete_terminal_transitions_discarded"]) for item in summaries
        ),
        "elapsed_seconds_this_invocation": time.time() - started,
        "protocol": protocol,
        "worker_index": config["worker_index"],
        "num_workers": config["num_workers"],
        "validation": validation,
    }
    _write_json_atomic(output / "generation_summary.json", summary)
    return summary


def generate_cache(args: argparse.Namespace) -> dict[str, Any]:
    """Run one resumable worker over its deterministic episode partition."""

    from libero.libero import benchmark

    started = time.time()
    seed_plan = resolve_seed_plan(
        experiment_seed=args.experiment_seed,
        environment_seed_base=args.seed,
        action_noise_seed_base=args.action_noise_seed_base,
        bootstrap_seed=args.seed,
    )
    if args.experiment_seed is not None:
        strict_runtime = configure_strict_determinism(
            seed_plan.process_seed,
            render_backend=args.render_backend,
        )
    else:
        seed_all(seed_plan.process_seed)
        strict_runtime = {
            "protocol": "legacy_cache_runtime_v1",
            "seed": seed_plan.process_seed,
            "render_backend": args.render_backend,
            "strict": False,
        }
    source_lock = collect_source_lock(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    protocol = {
        "execution_horizon_R": int(args.execution_horizon),
        "action_horizon_H_a": 10,
        "flow_steps": int(args.flow_steps),
        "client_resize_size": int(args.client_resize_size),
        "image_size": int(args.image_size),
        "num_wait_steps": int(args.num_wait_steps),
        "task_order": args.task_order,
        "control_hz": float(args.control_hz),
        "cache_semantics": "rollout_generated_query_to_next_query",
        "experiment_seed": args.experiment_seed,
        "effective_seed_plan": seed_plan.__dict__,
        "render_backend": args.render_backend,
        "environment_lifecycle": "fresh_environment_per_episode",
        "environment_seed_timing": "before_constructor_and_immediately_before_reset",
        "env_seed_semantics": (
            "evaluation_episode_seed(environment_seed_base,suite,task_id,trial_id)"
            if args.experiment_seed is not None
            else "sha256(base_seed,task_id,trial_id)"
        ),
    }
    os.environ.setdefault("LIBERO_ROOT", str(LIBERO_ROOT))
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = (
        list(range(suite.n_tasks - 1, -1, -1))
        if args.task_order == "official_reverse"
        else list(range(suite.n_tasks))
    )[: args.max_tasks]
    assigned_specs = partition_episode_specs(
        task_ids,
        args.num_trials,
        args.worker_index,
        args.num_workers,
    )
    if not assigned_specs:
        raise RuntimeError(
            f"worker {args.worker_index}/{args.num_workers} has no assigned episodes"
        )
    config = _worker_config(
        args,
        task_ids=task_ids,
        protocol=protocol,
        source_lock=source_lock,
        seed_plan=seed_plan.__dict__,
        strict_runtime=strict_runtime,
    )
    config_fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output = Path(args.output).expanduser().resolve()
    _prepare_worker_output(
        output,
        resume=args.resume,
        config=config,
        source_lock=source_lock,
    )

    existing_summary_path = output / "generation_summary.json"
    if existing_summary_path.exists():
        existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        if existing_summary.get("validation", {}).get("passed"):
            return existing_summary

    completed: dict[str, dict[str, Any]] = {}
    for _, task_id, trial_id in assigned_specs:
        episode_id = f"task{task_id:02d}_trial{trial_id:03d}"
        episode_dir = output / "episodes" / episode_id
        if episode_dir.exists():
            summary = _episode_summary(episode_dir, config_fingerprint)
            if summary is None:
                raise RuntimeError(f"Invalid committed episode directory: {episode_dir}")
            completed[episode_id] = summary

    if len(completed) == len(assigned_specs):
        return _finalize_worker_cache(
            output,
            assigned_specs=assigned_specs,
            config=config,
            source_lock=source_lock,
            protocol=protocol,
            config_fingerprint=config_fingerprint,
            started=started,
        )

    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    device = torch.device(args.device)
    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device).eval()
    model.action_space.load_norm_stats(args.norm_stats)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)
    condition_adapter = SimVLAConditionAdapter(model)
    action_adapter = SimVLAActionAdapter(model)
    progress_path = output / "generation_progress.jsonl"
    completed_cache_bytes = sum(int(item["cache_bytes"]) for item in completed.values())
    progress = tqdm(
        total=len(assigned_specs),
        initial=len(completed),
        desc=f"cache R={args.execution_horizon} worker={args.worker_index}",
        dynamic_ncols=True,
        mininterval=args.tqdm_mininterval,
        disable=args.disable_tqdm,
    )
    try:
        specs_by_task = {
            task_id: [spec for spec in assigned_specs if spec[1] == task_id]
            for task_id in task_ids
        }
        for task_id in task_ids:
            pending_specs = [
                spec
                for spec in specs_by_task[task_id]
                if f"task{task_id:02d}_trial{spec[2]:03d}" not in completed
            ]
            if not pending_specs:
                continue
            task = suite.get_task(task_id)
            init_states = suite.get_task_init_states(task_id)
            for _, _, trial_id in pending_specs:
                episode_started = time.time()
                episode_id = f"task{task_id:02d}_trial{trial_id:03d}"
                env_seed = (
                    evaluation_episode_seed(
                        seed_plan.environment_seed_base,
                        args.suite,
                        task_id,
                        trial_id,
                    )
                    if args.experiment_seed is not None
                    else episode_env_seed(args.seed, task_id, trial_id)
                )
                seed_all(env_seed)
                env, prompt = get_libero_env(task, args.resolution, env_seed)
                try:
                    seed_all(env_seed)
                    env.seed(env_seed)
                    env.reset()
                    obs = env.set_init_state(init_states[trial_id % len(init_states)])
                    env_timestep = 0
                    done = False
                    for _ in range(args.num_wait_steps):
                        obs, _, done, _ = env.step([0.0] * 6 + [-1.0])
                        env_timestep += 1
                    policy = RealSimVLADCLDPolicy(
                        model=model,
                        processor=processor,
                        dcld_core=None,
                        mode="full",
                        refresh_every=1,
                        flow_steps=args.flow_steps,
                        image_size=args.image_size,
                        replan_steps=args.execution_horizon,
                        client_resize_size=args.client_resize_size,
                        device=device,
                    )
                    pending: tuple[dict[str, Any], torch.Tensor] | None = None
                    query_index = 0
                    episode_records: list[dict[str, Any]] = []
                    incomplete_terminal_transitions = 0
                    while not done and query_index < args.max_policy_queries:
                        image0, image1, proprio = build_env_obs(obs)
                        snapshot = query_snapshot(
                            policy=policy,
                            condition_adapter=condition_adapter,
                            action_adapter=action_adapter,
                            image0=image0,
                            image1=image1,
                            proprio=proprio,
                            prompt=prompt,
                            checkpoint=args.checkpoint,
                            task_id=task_id,
                            episode_id=episode_id,
                            query_index=query_index,
                            env_timestep=env_timestep,
                            seed_base=seed_plan.action_noise_seed_base,
                            flow_steps=args.flow_steps,
                        )
                        if pending is not None:
                            previous, executed = pending
                            episode_records.append(
                                transition_record(
                                    current=previous,
                                    next_query=snapshot,
                                    executed_actions=executed,
                                    task_id=task_id,
                                    episode_id=episode_id,
                                    task_identifier=f"{task.problem_folder}/{task.bddl_file}",
                                    language_instruction=prompt,
                                    execution_horizon=args.execution_horizon,
                                    elapsed_time=args.execution_horizon / float(args.control_hz),
                                    provenance={
                                        "checkpoint": args.checkpoint,
                                        "norm_stats": args.norm_stats,
                                        "task_bddl": f"{task.problem_folder}/{task.bddl_file}",
                                        "init_state_index": trial_id % len(init_states),
                                        "env_seed": env_seed,
                                        "experiment_seed": args.experiment_seed,
                                        "render_backend": args.render_backend,
                                        "current_action_noise_seed": previous["action_noise_seed"],
                                        "next_action_noise_seed": snapshot["action_noise_seed"],
                                        "worker_index": args.worker_index,
                                        "num_workers": args.num_workers,
                                    },
                                )
                            )
                        planned = executed_subchunk(
                            snapshot["action_chunk_device"],
                            args.execution_horizon,
                        )[0]
                        actions_sent: list[torch.Tensor] = []
                        for action in planned:
                            obs, _, done, _ = env.step(action.detach().cpu().tolist())
                            actions_sent.append(action.detach().cpu())
                            env_timestep += 1
                            if done:
                                break
                        if len(actions_sent) == args.execution_horizon and not done:
                            pending = (snapshot, torch.stack(actions_sent, dim=0))
                        else:
                            pending = None
                            incomplete_terminal_transitions += 1
                        query_index += 1

                    episode_summary = _commit_episode(
                        output,
                        episode_id=episode_id,
                        records=episode_records,
                        execution_horizon=args.execution_horizon,
                        records_per_shard=args.records_per_shard,
                        metadata={
                            "protocol": protocol,
                            "worker_config_sha256": config_fingerprint,
                            "episode_id": episode_id,
                        },
                        summary={
                            "task_id": task_id,
                            "trial_id": trial_id,
                            "episode_id": episode_id,
                            "queries": query_index,
                            "success": bool(done),
                            "env_seed": env_seed,
                            "episode_elapsed_seconds": time.time() - episode_started,
                            "incomplete_terminal_transitions_discarded": (
                                incomplete_terminal_transitions
                            ),
                            "worker_config_sha256": config_fingerprint,
                        },
                    )
                    completed[episode_id] = episode_summary
                    completed_cache_bytes += int(episode_summary["cache_bytes"])
                    completed_count = len(completed)
                    projected_bytes = int(
                        completed_cache_bytes * len(assigned_specs) / completed_count
                    )
                    payload = {
                        "task_id": task_id,
                        "trial_id": trial_id,
                        "episode_id": episode_id,
                        "queries": query_index,
                        "success": bool(done),
                        "records": len(episode_records),
                        "episodes_completed": completed_count,
                        "episodes_assigned": len(assigned_specs),
                        "cache_bytes": completed_cache_bytes,
                        "projected_cache_bytes": projected_bytes,
                        "elapsed_seconds": time.time() - started,
                        "worker_index": args.worker_index,
                    }
                    _append_jsonl(progress_path, payload)
                    progress.update(1)
                    progress.set_postfix(
                        queries=query_index,
                        records=len(episode_records),
                        GiB=f"{completed_cache_bytes / 2**30:.1f}",
                        projected_GiB=f"{projected_bytes / 2**30:.1f}",
                    )
                    if args.max_cache_gib > 0 and completed_cache_bytes > args.max_cache_gib * 2**30:
                        raise RuntimeError(
                            f"worker cache exceeded --max-cache-gib={args.max_cache_gib}: "
                            f"{completed_cache_bytes / 2**30:.2f} GiB"
                        )
                finally:
                    env.close()
    finally:
        progress.close()

    return _finalize_worker_cache(
        output,
        assigned_specs=assigned_specs,
        config=config,
        source_lock=source_lock,
        protocol=protocol,
        config_fingerprint=config_fingerprint,
        started=started,
    )


def main() -> int:
    """Generate, resume, or validate a version-2 query-boundary cache."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-cache", default="")
    parser.add_argument("--validation-output", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--execution-horizon", type=int, choices=(1, 2, 5), default=5)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-policy-queries", type=int, default=3)
    parser.add_argument("--num-wait-steps", type=int, default=10)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--client-resize-size", type=int, default=224)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-noise-seed-base", type=int, default=20260804)
    parser.add_argument("--experiment-seed", type=int, default=None)
    parser.add_argument(
        "--render-backend",
        choices=("osmesa", "egl"),
        default="egl",
    )
    parser.add_argument("--task-order", choices=("official_reverse", "ascending"), default="official_reverse")
    parser.add_argument("--records-per-shard", type=int, default=128)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cache-gib", type=float, default=0.0)
    parser.add_argument("--tqdm-mininterval", type=float, default=1.0)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.validate_cache:
        result = validate_query_cache(args.validate_cache)
        if args.validation_output:
            validation_output = require_empty_output(args.validation_output)
            (validation_output / "cache_validation.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            write_source_lock(
                validation_output,
                checkpoint=args.checkpoint,
                norm_stats_path=args.norm_stats,
            )
            cache_source_lock = load_manifest(args.validate_cache).get(
                "metadata", {}
            ).get("source_lock", {})
            (validation_output / "cache_source_lock.json").write_text(
                json.dumps(cache_source_lock, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        if not args.output:
            parser.error("--output is required for generation")
        result = generate_cache(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("validation", result).get("passed", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
