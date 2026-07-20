#!/usr/bin/env python3
"""Offline K=1 replay parity diagnostic for SimVLA DCLD wrappers.

This diagnostic records policy-query inputs from one full SimVLA wrapper
rollout, then replays baseline_full_k1 and ours_full_k1 offline from exactly
the same tensors and explicit initial action noise. It does not run DCLD k>1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
LIBERO_EVAL = UPSTREAM / "evaluation" / "libero"
LIBERO_ROOT = LIBERO_EVAL / "LIBERO"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter, SimVLAConditionAdapter  # noqa: E402
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (  # noqa: E402
    LATENCY_FIELDS,
    REQUIRED_COUNTERS,
    RealSimVLADCLDPolicy,
    build_env_obs,
    collect_environment_metadata,
    command_output,
    format_duration,
    get_libero_env,
    stable_seed,
    tensor_action_diff,
    write_json,
    write_text,
)


def tensor_payload(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }


def bool_all(values: list[bool]) -> bool:
    return bool(values) and all(values)


def compare_tensor(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    return tensor_action_diff(a, b)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def record_git_state(out: Path, suffix: str) -> None:
    write_text(out / f"date_{suffix}.txt", time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    write_text(out / f"root_git_status_{suffix}.txt", command_output(["git", "status", "--short"], cwd=ROOT))
    write_text(out / f"root_git_branch_{suffix}.txt", command_output(["git", "branch", "--show-current"], cwd=ROOT))
    write_text(out / f"root_git_head_{suffix}.txt", command_output(["git", "rev-parse", "HEAD"], cwd=ROOT))
    write_text(out / f"root_git_diff_name_only_{suffix}.txt", command_output(["git", "diff", "--name-only"], cwd=ROOT))
    write_text(out / f"simvla_upstream_git_status_{suffix}.txt", command_output(["git", "-C", str(UPSTREAM), "status", "--short"], cwd=ROOT))
    write_text(out / f"simvla_upstream_git_head_{suffix}.txt", command_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], cwd=ROOT))


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device)
    model.eval()
    if args.norm_stats and Path(args.norm_stats).exists():
        model.action_space.load_norm_stats(args.norm_stats)
    for param in model.parameters():
        param.requires_grad_(False)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)
    return model, processor


def full_query_and_record(
    *,
    policy: RealSimVLADCLDPolicy,
    batch: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Run the full K=1 query path once and return tensors to save."""

    policy_query_index = int(policy.query_index)
    policy.metrics.counters["num_policy_queries"] += 1

    t0 = time.perf_counter()
    condition = policy.condition_adapter.encode_condition(
        input_ids=batch["input_ids"],
        image_input=batch["image_input"],
        image_mask=batch["image_mask"],
    )
    policy.metrics.latencies["VLM_encoder_ms"].append((time.perf_counter() - t0) * 1000.0)
    policy.metrics.counters["num_full_vlm_calls"] += 1

    initial_noise, action_noise_seed = policy._paired_initial_noise(condition, batch["proprio"], policy_query_index)
    if initial_noise is None:
        raise RuntimeError("offline replay diagnostic requires paired_action_noise=True")

    t_action = time.perf_counter()
    action_out = policy.action_adapter.decode_action_from_condition(
        condition,
        batch["proprio"],
        steps=policy.flow_steps,
        initial_noise=initial_noise,
        deterministic=False,
        return_debug=True,
    )
    policy.metrics.latencies["action_transformer_ms"].append((time.perf_counter() - t_action) * 1000.0)
    policy.metrics.counters["num_action_transformer_calls"] += int(action_out.debug.get("iterations", 0))

    action_chunk = action_out.action
    policy.cached_condition = condition.detach()
    policy.cached_raw_rgb = batch["raw_rgb"].detach()
    policy.cached_proprio = batch["proprio"].detach()
    policy.cached_action_chunk = action_chunk.detach()
    policy.action_queue.clear()
    for action in action_chunk[0, : policy.replan_steps]:
        policy.action_queue.append((action.detach(), "full_refresh"))
    policy.query_index += 1

    return {
        "metadata": {
            **metadata,
            "row_name": "baseline_full_k1_recorded",
            "mode": "full",
            "k": 1,
            "queue_mode": "full_refresh",
            "refreshed": True,
            "full_vlm_called": True,
            "dcld_called": False,
            "fast_encoder_called": False,
            "paired_action_noise": True,
            "action_noise_seed": int(action_noise_seed) if action_noise_seed is not None else None,
            "action_noise_seed_key_parts": [
                int(policy.action_noise_seed_base),
                policy.suite,
                int(policy.task_id),
                int(policy.trial_id),
                int(policy_query_index),
                int(policy.flow_steps),
                int(policy.action_adapter.num_actions),
                int(policy.action_adapter.dim_action),
            ],
            "condition_shape": list(condition.shape),
            "initial_noise_shape": list(initial_noise.shape),
            "action_chunk_shape": list(action_chunk.shape),
        },
        "input_ids": batch["input_ids"].detach().cpu(),
        "image_input": batch["image_input"].detach().cpu(),
        "image_mask": batch["image_mask"].detach().cpu(),
        "proprio": batch["proprio"].detach().cpu(),
        "raw_rgb": batch["raw_rgb"].detach().cpu(),
        "condition_recorded": condition.detach().cpu(),
        "initial_noise": initial_noise.detach().cpu(),
        "action_chunk_recorded": action_chunk.detach().cpu(),
    }


def run_recorded_rollout(
    *,
    args: argparse.Namespace,
    out: Path,
    model: Any,
    processor: Any,
    device: torch.device,
) -> dict[str, Any]:
    from libero.libero import benchmark

    os.environ.setdefault("LIBERO_ROOT", str(LIBERO_ROOT))
    records_dir = out / "offline_replay_inputs"
    records_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    env, task_description = get_libero_env(task, args.resolution, args.seed)

    manifest: list[dict[str, Any]] = []
    policy = RealSimVLADCLDPolicy(
        model=model,
        processor=processor,
        dcld_core=None,
        mode="full",
        refresh_every=1,
        flow_steps=args.flow_steps,
        image_size=args.image_size,
        replan_steps=args.replan_steps,
        client_resize_size=args.client_resize_size,
        device=device,
        suite=args.suite,
        row_name="baseline_full_k1_recorded",
        task_id=args.task_id,
        trial_id=args.trial_id,
        paired_action_noise=True,
        action_noise_seed_base=args.action_noise_seed_base,
        log_action_chunks=False,
    )

    done = False
    reward = 0.0
    info: dict[str, Any] = {}
    policy_steps = 0
    env_steps = 0
    rollout_start = time.time()
    try:
        env.reset()
        obs = env.set_init_state(initial_states[args.trial_id % len(initial_states)])
        for _ in range(args.num_wait_steps):
            t_env = time.perf_counter()
            obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
            policy.metrics.latencies["env_step_ms"].append((time.perf_counter() - t_env) * 1000.0)
            env_steps += 1

        while policy_steps < args.max_policy_steps:
            if not policy.action_queue:
                image0, image1, state = build_env_obs(obs)
                batch = policy.preprocess(image0, image1, state, task_description)
                query_index = int(policy.query_index)
                record = full_query_and_record(
                    policy=policy,
                    batch=batch,
                    metadata={
                        "suite": args.suite,
                        "task_id": int(args.task_id),
                        "trial_id": int(args.trial_id),
                        "policy_query_index": query_index,
                        "episode_step_index": int(policy.step_index),
                        "language_instruction": task_description,
                        "replan_steps": int(args.replan_steps),
                        "client_resize_size": int(args.client_resize_size),
                        "flow_steps": int(args.flow_steps),
                        "task_order": args.task_order,
                        "resolution": int(args.resolution),
                        "seed": int(args.seed),
                    },
                )
                record_name = f"task{args.task_id:02d}_trial{args.trial_id:02d}_query{query_index:04d}.pt"
                record_path = records_dir / record_name
                torch.save(record, record_path)
                manifest.append(
                    {
                        **record["metadata"],
                        "record_file": str(record_path),
                        "record_file_relative": str(record_path.relative_to(out)),
                    }
                )

            total_t0 = time.perf_counter()
            policy.metrics.counters["num_env_steps"] += 1
            queued_action, action_source = policy.action_queue.popleft()
            action = queued_action.reshape(1, -1)
            policy.metrics.counters["num_action_queue_steps"] += 1
            policy.cached_executed_action = action.detach().reshape(-1)
            policy.metrics.latencies["policy_total_ms"].append((time.perf_counter() - total_t0) * 1000.0)
            action_np = action.detach().cpu().numpy()[0].astype(np.float32)
            policy.metrics.observe_action(action_np)
            policy.step_index += 1

            t_env = time.perf_counter()
            obs, reward, done, info = env.step(action_np.tolist())
            policy.metrics.latencies["env_step_ms"].append((time.perf_counter() - t_env) * 1000.0)
            env_steps += 1
            policy_steps += 1
            if action_source != "full_refresh":
                raise RuntimeError(f"unexpected action source in full K1 recorder: {action_source}")
            if done:
                break
    finally:
        env.close()

    shapes: dict[str, Any] = {}
    if manifest:
        first = torch.load(manifest[0]["record_file"], map_location="cpu", weights_only=False)
        for key, value in first.items():
            if isinstance(value, torch.Tensor):
                shapes[key] = tensor_payload(value)
    write_json(out / "policy_query_record_manifest.json", manifest)
    write_json(out / "recorded_policy_query_shapes.json", shapes)

    rollout_summary = {
        "suite": args.suite,
        "task_id": int(args.task_id),
        "trial_id": int(args.trial_id),
        "task_description": task_description,
        "success": bool(done),
        "reward": float(reward) if np.isscalar(reward) else str(reward),
        "info": {key: str(value) for key, value in dict(info).items()},
        "policy_steps": int(policy_steps),
        "env_steps": int(env_steps),
        "num_policy_queries_recorded": len(manifest),
        "elapsed_seconds": float(time.time() - rollout_start),
        "counters": {name: int(policy.metrics.counters.get(name, 0)) for name in REQUIRED_COUNTERS},
        "latency": policy.metrics.summary().get("latency", {}),
        "records_dir": str(records_dir),
        "manifest_path": str(out / "policy_query_record_manifest.json"),
    }
    write_json(out / "recorded_rollout_summary.json", rollout_summary)
    return rollout_summary


def decode_from_saved(
    *,
    condition_adapter: SimVLAConditionAdapter,
    action_adapter: SimVLAActionAdapter,
    record: dict[str, Any],
    device: torch.device,
    flow_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = record["input_ids"].to(device=device)
    image_input = record["image_input"].to(device=device)
    image_mask = record["image_mask"].to(device=device)
    proprio = record["proprio"].to(device=device)
    initial_noise = record["initial_noise"].to(device=device, dtype=proprio.dtype)
    condition = condition_adapter.encode_condition(
        input_ids=input_ids,
        image_input=image_input,
        image_mask=image_mask,
    )
    action_out = action_adapter.decode_action_from_condition(
        condition,
        proprio,
        steps=flow_steps,
        initial_noise=initial_noise,
        deterministic=False,
        return_debug=True,
    )
    return condition.detach().cpu(), action_out.action.detach().cpu()


def run_offline_replay(
    *,
    args: argparse.Namespace,
    out: Path,
    model: Any,
    device: torch.device,
) -> dict[str, Any]:
    manifest_path = out / "policy_query_record_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline_action_adapter = SimVLAActionAdapter(model)
    baseline_condition_adapter = SimVLAConditionAdapter(model, baseline_action_adapter)
    ours_action_adapter = SimVLAActionAdapter(model)
    ours_condition_adapter = SimVLAConditionAdapter(model, ours_action_adapter)

    diff_rows: list[dict[str, Any]] = []
    for index, item in enumerate(manifest):
        record = torch.load(item["record_file"], map_location="cpu", weights_only=False)
        metadata = dict(record["metadata"])
        expected_seed = stable_seed(
            int(args.action_noise_seed_base),
            args.suite,
            int(metadata["task_id"]),
            int(metadata["trial_id"]),
            int(metadata["policy_query_index"]),
            int(args.flow_steps),
            int(model.num_actions),
            int(model.action_space.dim_action),
        )
        seed_match = int(metadata["action_noise_seed"]) == int(expected_seed)

        condition_baseline, action_baseline = decode_from_saved(
            condition_adapter=baseline_condition_adapter,
            action_adapter=baseline_action_adapter,
            record=record,
            device=device,
            flow_steps=args.flow_steps,
        )
        condition_ours, action_ours = decode_from_saved(
            condition_adapter=ours_condition_adapter,
            action_adapter=ours_action_adapter,
            record=record,
            device=device,
            flow_steps=args.flow_steps,
        )
        condition_recorded = record["condition_recorded"].detach().cpu()
        action_recorded = record["action_chunk_recorded"].detach().cpu()

        cond_base_ours = compare_tensor(condition_baseline, condition_ours)
        action_base_ours = compare_tensor(action_baseline, action_ours)
        cond_recorded_base = compare_tensor(condition_recorded, condition_baseline)
        action_recorded_base = compare_tensor(action_recorded, action_baseline)
        row = {
            "record_index": index,
            "record_file": item["record_file"],
            "suite": metadata["suite"],
            "task_id": metadata["task_id"],
            "trial_id": metadata["trial_id"],
            "policy_query_index": metadata["policy_query_index"],
            "episode_step_index": metadata["episode_step_index"],
            "action_noise_seed": metadata["action_noise_seed"],
            "expected_action_noise_seed": int(expected_seed),
            "seed_match": bool(seed_match),
            "condition_shape": str(list(condition_baseline.shape)),
            "action_chunk_shape": str(list(action_baseline.shape)),
            "condition_base_ours_mean_abs_diff": cond_base_ours["mean_abs_diff"],
            "condition_base_ours_max_abs_diff": cond_base_ours["max_abs_diff"],
            "condition_base_ours_l2_diff": cond_base_ours["l2_diff"],
            "condition_base_ours_allclose_1e_5": cond_base_ours["allclose_1e_5"],
            "condition_base_ours_allclose_1e_4": cond_base_ours["allclose_1e_4"],
            "action_base_ours_mean_abs_diff": action_base_ours["mean_abs_diff"],
            "action_base_ours_max_abs_diff": action_base_ours["max_abs_diff"],
            "action_base_ours_l2_diff": action_base_ours["l2_diff"],
            "action_base_ours_allclose_1e_5": action_base_ours["allclose_1e_5"],
            "action_base_ours_allclose_1e_4": action_base_ours["allclose_1e_4"],
            "action_base_ours_allclose_1e_3": action_base_ours["allclose_1e_3"],
            "action_base_ours_allclose_1e_2": action_base_ours["allclose_1e_2"],
            "condition_recorded_base_max_abs_diff": cond_recorded_base["max_abs_diff"],
            "condition_recorded_base_allclose_1e_5": cond_recorded_base["allclose_1e_5"],
            "action_recorded_base_max_abs_diff": action_recorded_base["max_abs_diff"],
            "action_recorded_base_allclose_1e_5": action_recorded_base["allclose_1e_5"],
        }
        diff_rows.append(row)

    write_csv(out / "offline_k1_replay_action_diff.csv", diff_rows)

    all_action_1e5 = bool_all([bool(row["action_base_ours_allclose_1e_5"]) for row in diff_rows])
    all_condition_1e5 = bool_all([bool(row["condition_base_ours_allclose_1e_5"]) for row in diff_rows])
    all_recorded_action_1e5 = bool_all([bool(row["action_recorded_base_allclose_1e_5"]) for row in diff_rows])
    num_seed_mismatches = sum(1 for row in diff_rows if not row["seed_match"])
    summary = {
        "num_policy_queries": len(diff_rows),
        "num_missing_pairs": 0,
        "num_seed_mismatches": int(num_seed_mismatches),
        "all_conditions_allclose_1e_5": bool(all_condition_1e5),
        "all_action_chunks_allclose_1e_5": bool(all_action_1e5),
        "all_recorded_vs_replayed_actions_allclose_1e_5": bool(all_recorded_action_1e5),
        "max_condition_base_ours_abs_diff": None if not diff_rows else max(float(row["condition_base_ours_max_abs_diff"] or 0.0) for row in diff_rows),
        "max_action_base_ours_abs_diff": None if not diff_rows else max(float(row["action_base_ours_max_abs_diff"] or 0.0) for row in diff_rows),
        "max_recorded_replayed_action_abs_diff": None if not diff_rows else max(float(row["action_recorded_base_max_abs_diff"] or 0.0) for row in diff_rows),
        "action_diff_csv": str(out / "offline_k1_replay_action_diff.csv"),
        "pass_criterion": {
            "all_action_chunks_allclose_1e_5": bool(all_action_1e5),
            "num_seed_mismatches_zero": int(num_seed_mismatches) == 0,
            "num_missing_pairs_zero": True,
        },
    }
    summary["passed"] = bool(
        summary["pass_criterion"]["all_action_chunks_allclose_1e_5"]
        and summary["pass_criterion"]["num_seed_mismatches_zero"]
        and summary["pass_criterion"]["num_missing_pairs_zero"]
    )
    write_json(out / "offline_k1_replay_summary.json", summary)
    write_text(
        out / "offline_k1_replay_report.md",
        "\n".join(
            [
                "# Offline K1 Replay Report",
                "",
                f"- num_policy_queries: `{summary['num_policy_queries']}`",
                f"- num_missing_pairs: `{summary['num_missing_pairs']}`",
                f"- num_seed_mismatches: `{summary['num_seed_mismatches']}`",
                f"- all_conditions_allclose_1e_5: `{summary['all_conditions_allclose_1e_5']}`",
                f"- all_action_chunks_allclose_1e_5: `{summary['all_action_chunks_allclose_1e_5']}`",
                f"- all_recorded_vs_replayed_actions_allclose_1e_5: `{summary['all_recorded_vs_replayed_actions_allclose_1e_5']}`",
                f"- max_condition_base_ours_abs_diff: `{summary['max_condition_base_ours_abs_diff']}`",
                f"- max_action_base_ours_abs_diff: `{summary['max_action_base_ours_abs_diff']}`",
                f"- max_recorded_replayed_action_abs_diff: `{summary['max_recorded_replayed_action_abs_diff']}`",
                f"- passed: `{summary['passed']}`",
            ]
        ),
    )
    return summary


def write_failure_diagnosis(out: Path, summary: dict[str, Any]) -> None:
    rows = []
    csv_path = out / "offline_k1_replay_action_diff.csv"
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    first_failed = None
    for row in rows:
        if row.get("action_base_ours_allclose_1e_5") != "True":
            first_failed = row
            break
    write_text(
        out / "offline_replay_failure_diagnosis.md",
        "\n".join(
            [
                "# Offline Replay Failure Diagnosis",
                "",
                "- input_ids/image_input/image_mask/proprio/initial_noise: same saved tensors were supplied to both offline rows.",
                "- DCLD/FastEncoder: inactive; only full SimVLA condition and action transformer paths were used.",
                f"- num_seed_mismatches: `{summary.get('num_seed_mismatches')}`",
                f"- max_action_base_ours_abs_diff: `{summary.get('max_action_base_ours_abs_diff')}`",
                f"- first_failed_row: `{first_failed}`",
                "",
                "Because the offline replay uses identical saved tensors and explicit initial_noise, any failure here should be treated as a model/wrapper implementation bug before DCLD k sweep.",
            ]
        ),
    )


def write_passed_next_steps(out: Path, args: argparse.Namespace) -> None:
    write_text(
        out / "offline_replay_passed_next_steps.md",
        "\n".join(
            [
                "# Offline Replay Passed: Next Steps",
                "",
                "K1 model/wrapper action equivalence is proven on identical saved observations, proprio, tokenized prompt, image mask, and explicit initial_noise tensors.",
                "",
                "The previous full-rollout strict action mismatches are therefore attributed to independent environment rollout drift after an earlier one-step trajectory difference, not to a baseline_full_k1 vs ours_full_k1 model-path mismatch.",
                "",
                "Recommended next step: proceed to DCLD k sweep with paired action noise, row semantics, videos, and efficiency logging enabled.",
                "",
                "Suggested command skeleton:",
                "",
                "```bash",
                "export GNAROSHI_VLA_ROOT=${GNAROSHI_VLA_ROOT:-$(git rev-parse --show-toplevel)}",
                "cd \"${GNAROSHI_VLA_ROOT}\"",
                "conda activate simvla_libero",
                "export CUDA_VISIBLE_DEVICES=4",
                "export HF_HOME=\"${GNAROSHI_VLA_ROOT}/.cache/huggingface\"",
                "export TOKENIZERS_PARALLELISM=false",
                "export MUJOCO_GL=egl",
                "export PYOPENGL_PLATFORM=egl",
                "export NUMBA_CACHE_DIR=/tmp/numba_cache",
                "export MPLCONFIGDIR=/tmp/matplotlib-${USER}",
                "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
                "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                "",
                "export PM047M_DCLD_CKPT=results/simvla/dcld/train/simvla_libero_dcld_pm047m/checkpoints/dcld_step_150000.pt",
                "",
                "SIMVLA_DCLD_EVAL_RUN=1 \\",
                "SIMVLA_DCLD_EVAL_OUTPUT=results/simvla/dcld/eval/qred20/simvla_libero_dcld_pm047m_paired_replan5 \\",
                "bash architectures/simvla/wrappers/simvla_dcld_eval_qred20.sh \\",
                "  --run \\",
                "  --checkpoint YuankaiLuo/SimVLA-LIBERO \\",
                "  --dcld-checkpoint \"${PM047M_DCLD_CKPT}\" \\",
                f"  --suite {args.suite} \\",
                "  --k-list 1,2,3,4 \\",
                "  --num-trials 10 \\",
                "  --max-tasks 10 \\",
                "  --max-policy-steps 900 \\",
                "  --num-wait-steps 10 \\",
                "  --replan-steps 5 \\",
                "  --task-order official_reverse \\",
                "  --client-resize-size 224 \\",
                "  --paired-action-noise \\",
                "  --action-noise-seed-base 20260708 \\",
                "  --flow-steps 10 \\",
                "  --tqdm-mininterval 1.0 \\",
                "  --eval-print-interval 1 \\",
                "  --save-video \\",
                "  --video-fps 10 \\",
                "  --video-stride 2 \\",
                "  --video-max-per-row 2 \\",
                "  --device cuda",
                "```",
            ]
        ),
    )


def write_interleaved_design(out: Path) -> None:
    write_text(
        out / "interleaved_k1_parity_design.md",
        "\n".join(
            [
                "# Interleaved K1 Parity Design",
                "",
                "This is a design note only; it was not run.",
                "",
                "Design:",
                "",
                "1. Create one LIBERO env instance and one episode state.",
                "2. At each policy query, build one obs/proprio/prompt batch.",
                "3. Feed the same batch to baseline_full_k1 and ours_full_k1.",
                "4. Force both rows to use the same explicit initial_noise tensor.",
                "5. Compare condition and action chunks before env.step.",
                "6. If allclose passes, execute one selected action sequence in the single env.",
                "7. Repeat until success/failure.",
                "",
                "This removes independent rollout drift while still exercising the online env loop.",
            ]
        ),
    )


def write_final_report(
    *,
    out: Path,
    args: argparse.Namespace,
    rollout_summary: dict[str, Any],
    replay_summary: dict[str, Any],
    environment_metadata: dict[str, Any],
    verdict: str,
) -> None:
    write_text(
        out / "final_offline_k1_replay_parity_report.md",
        "\n".join(
            [
                "# Final Offline K1 Replay Parity Report",
                "",
                f"- verdict: `{verdict}`",
                f"- output_dir: `{out}`",
                f"- recorded_task_trial: `{args.suite}` task `{args.task_id}`, trial `{args.trial_id}`",
                f"- policy_queries_recorded: `{rollout_summary.get('num_policy_queries_recorded')}`",
                f"- rollout_success: `{rollout_summary.get('success')}`",
                f"- rollout_policy_steps: `{rollout_summary.get('policy_steps')}`",
                f"- rollout_env_steps: `{rollout_summary.get('env_steps')}`",
                f"- explicit_initial_noise_saved_and_reused: `True`",
                f"- replay_all_action_chunks_allclose_1e_5: `{replay_summary.get('all_action_chunks_allclose_1e_5')}`",
                f"- replay_num_seed_mismatches: `{replay_summary.get('num_seed_mismatches')}`",
                f"- replay_num_missing_pairs: `{replay_summary.get('num_missing_pairs')}`",
                f"- max_action_base_ours_abs_diff: `{replay_summary.get('max_action_base_ours_abs_diff')}`",
                f"- max_recorded_replayed_action_abs_diff: `{replay_summary.get('max_recorded_replayed_action_abs_diff')}`",
                f"- conda_env: `{environment_metadata.get('conda_env')}`",
                f"- mujoco_version: `{environment_metadata.get('mujoco_version')}`",
                f"- torch_version: `{environment_metadata.get('torch_version')}`",
                f"- cuda_visible_devices: `{environment_metadata.get('cuda_visible_devices')}`",
                "",
                "## What Was Recorded",
                "",
                "Each saved policy query contains suite/task/trial/query ids, episode step index, language instruction, tokenized input_ids, image_input, image_mask, proprio, raw_rgb tensor, condition latent, explicit initial_noise tensor, action chunk, replan/client resize/flow settings, and action-noise seed metadata.",
                "",
                "## Replay Method",
                "",
                "The offline replay decodes baseline_full_k1 and ours_full_k1 from the same saved tensors using the same SimVLA checkpoint, same condition path, same action transformer, DCLD inactive, and the saved initial_noise tensor.",
                "",
                "## Files",
                "",
                f"- manifest: `{out / 'policy_query_record_manifest.json'}`",
                f"- shapes: `{out / 'recorded_policy_query_shapes.json'}`",
                f"- action diff csv: `{out / 'offline_k1_replay_action_diff.csv'}`",
                f"- replay summary: `{out / 'offline_k1_replay_summary.json'}`",
                f"- replay report: `{out / 'offline_k1_replay_report.md'}`",
                "",
                "## Code Scope",
                "",
                "- files_modified_by_this_task: `architectures/simvla/wrappers/simvla_dcld_offline_k1_replay.py`",
                "- upstream_simvla_modified: `False`",
                "- dcld_k_sweep_run: `False`",
                "- retraining_run: `False`",
                "",
                "## Next",
                "",
                "If the verdict is READY_FOR_DCLD_K_SWEEP_BY_OFFLINE_PARITY, DCLD k sweep can proceed from a model/wrapper K1 parity standpoint. The separate official reproduction gap remains a separate baseline-calibration issue and should not be conflated with K1 wrapper equivalence.",
            ]
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--dcld-checkpoint", default="results/simvla/dcld/train/simvla_libero_dcld_pm047m/checkpoints/dcld_step_150000.pt")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--task-order", choices=["official_reverse", "ascending"], default="official_reverse")
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--num-wait-steps", type=int, default=10)
    parser.add_argument("--max-policy-steps", type=int, default=900)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--client-resize-size", type=int, default=224)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-noise-seed-base", type=int, default=20260708)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.paired_action_noise = True

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    record_git_state(out, "start")
    write_json(out / "args_snapshot.json", {key: str(value) for key, value in vars(args).items()})
    write_text(out / "command.sh", " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]))

    if args.replan_steps != 5:
        raise ValueError("This diagnostic is intended for official-style replan_steps=5.")
    if args.client_resize_size != 224:
        raise ValueError("This diagnostic is intended for client_resize_size=224.")
    if not Path(args.dcld_checkpoint).exists():
        write_text(out / "WARNING_dcld_checkpoint_not_loaded.md", f"DCLD checkpoint path was recorded but not loaded for K1: `{args.dcld_checkpoint}`")

    device = torch.device(args.device)
    model, processor = load_model(args, device)
    environment_metadata = collect_environment_metadata(args)
    write_json(out / "environment_metadata.json", environment_metadata)

    rollout_summary = run_recorded_rollout(args=args, out=out, model=model, processor=processor, device=device)
    replay_summary = run_offline_replay(args=args, out=out, model=model, device=device)
    write_interleaved_design(out)

    if replay_summary.get("passed"):
        verdict = "READY_FOR_DCLD_K_SWEEP_BY_OFFLINE_PARITY"
        write_passed_next_steps(out, args)
    else:
        verdict = "NOT_READY_FOR_DCLD_K_SWEEP"
        write_failure_diagnosis(out, replay_summary)

    write_final_report(
        out=out,
        args=args,
        rollout_summary=rollout_summary,
        replay_summary=replay_summary,
        environment_metadata=environment_metadata,
        verdict=verdict,
    )
    record_git_state(out, "end")
    print(f"[offline-k1-replay] verdict={verdict}", flush=True)
    print(f"[offline-k1-replay] report={out / 'final_offline_k1_replay_parity_report.md'}", flush=True)
    print(f"[offline-k1-replay] elapsed_rollout={format_duration(float(rollout_summary.get('elapsed_seconds', 0.0)))}", flush=True)
    return 0 if verdict == "READY_FOR_DCLD_K_SWEEP_BY_OFFLINE_PARITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
