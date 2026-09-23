#!/usr/bin/env python3
"""One-shot Stage-A checkpoint decision on the frozen validation stream."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import os
import random
import sys
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from evaluate_joint_stage_a_action_space_v2 import (
    build_runtime,
    preflight as preflight_v2,
    read_json,
    resolved,
    sha256,
    strip_state,
)
from methods.joint_latent_action_surrogate.action_space_audit import (
    action_from_arm_and_logit,
    distribution,
    per_sample_l1,
)
from methods.joint_latent_action_surrogate.alignment import align_immutable_anchor
from methods.joint_latent_action_surrogate.checkpoint_sweep import (
    EXACT_ENSEMBLE_METRIC_UNAVAILABLE,
    build_fidelity_eligibility,
    select_eligible_checkpoint,
    utility_verdict,
)


PROTOCOL = "joint_stage_a_one_shot_checkpoint_sweep_v2"
AGES = (1, 2)
TOKEN_INDICES = (0, 1, 2)
ACTION_REPRESENTATION = "[arm_6d, sigmoid(gripper_logit)]"


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise RuntimeError(f"Expected one source function named {name}, got {len(matches)}")
    return matches[0]


def _attribute_name(call: ast.Call) -> str | None:
    return call.func.attr if isinstance(call.func, ast.Attribute) else None


def audit_shared_feature_call_path(
    repo_root: Path, contract: dict[str, Any]
) -> dict[str, Any]:
    """Prove that the surrogate reuses u_delta and adds no encoder invocation."""

    spec = contract["call_path_audit"]
    path = resolved(repo_root / spec["source_file"])
    if sha256(path) != spec["source_sha256"]:
        raise RuntimeError(f"Call-path source identity failed: {path}")
    source = path.read_text(encoding="utf-8")
    function = _find_function(ast.parse(source), spec["function"])
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    encoder_calls = sum(_attribute_name(call) == "lrnode_encode_delta" for call in calls)
    cache_update_calls = [
        call for call in calls if _attribute_name(call) == "_update_from_lrnode_cache"
    ]
    surrogate_calls = [
        call for call in calls if _attribute_name(call) == "surrogate_forward"
    ]
    if encoder_calls != 0 or len(cache_update_calls) != 1 or len(surrogate_calls) != 1:
        raise RuntimeError("Stage-A runtime no longer has the frozen shared-feature topology")
    decode_action_values = [
        keyword.value
        for keyword in cache_update_calls[0].keywords
        if keyword.arg == "decode_action"
    ]
    if not (
        len(decode_action_values) == 1
        and isinstance(decode_action_values[0], ast.Constant)
        and decode_action_values[0].value is False
    ):
        raise RuntimeError("Shared cache update must suppress the exact action-head decode")
    shared_values = [
        keyword.value
        for keyword in surrogate_calls[0].keywords
        if keyword.arg == "shared_feature"
    ]
    shared_debug_u_delta = False
    if len(shared_values) == 1 and isinstance(shared_values[0], ast.Subscript):
        value = shared_values[0]
        shared_debug_u_delta = (
            isinstance(value.value, ast.Name)
            and value.value.id == "debug"
            and isinstance(value.slice, ast.Constant)
            and value.slice.value == "u_delta"
        )
    if not shared_debug_u_delta:
        raise RuntimeError("Surrogate input is not the canonical shared debug['u_delta']")
    return {
        "status": "PASS",
        "source_file": str(path),
        "source_sha256": spec["source_sha256"],
        "function": spec["function"],
        "line_start": int(function.lineno),
        "line_end": int(function.end_lineno or function.lineno),
        "cache_update_calls": len(cache_update_calls),
        "cache_update_decode_action": False,
        "surrogate_forward_calls": len(surrogate_calls),
        "shared_feature": "debug['u_delta']",
        "additional_observation_encoder_calls": encoder_calls,
        "shared_observation_encoder_charged_to_surrogate": False,
    }


def validate_sweep_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract = read_json(args.contract)
    if contract.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Unexpected sweep protocol: {contract.get('protocol')}")
    if contract.get("status") != "PREDECLARED_BEFORE_USER_SWEEP":
        raise RuntimeError("One-shot checkpoint sweep contract is not frozen")
    if contract["evaluation"]["action_representation"] != (
        "arm_6d_plus_sigmoid_gripper_logit_1d"
    ):
        raise RuntimeError("Sweep does not use the executed action representation")
    if tuple(contract["evaluation"]["regeneration_ages"]) != AGES:
        raise RuntimeError("Sweep must evaluate regeneration ages 1 and 2")
    if tuple(contract["evaluation"]["token_indices"]) != TOKEN_INDICES:
        raise RuntimeError("Sweep must evaluate action tokens h=0,1,2")

    implementation_paths = {
        "evaluator_sha256": Path(__file__).resolve(),
        "v2_evaluator_sha256": (
            args.repo_root / "tools/seer/evaluate_joint_stage_a_action_space_v2.py"
        ),
        "metric_module_sha256": (
            args.repo_root
            / "methods/joint_latent_action_surrogate/action_space_audit.py"
        ),
        "ranking_module_sha256": (
            args.repo_root / "methods/joint_latent_action_surrogate/checkpoint_sweep.py"
        ),
    }
    for key, path in implementation_paths.items():
        if not path.is_file() or sha256(path) != contract["implementation"][key]:
            raise RuntimeError(f"Sweep implementation identity failed: {path}")

    completed = contract["completed_v2_audit"]
    for name in ("metrics", "gate", "provenance"):
        path = resolved(completed[f"{name}_path"])
        if not path.is_file() or sha256(path) != completed[f"{name}_sha256"]:
            raise RuntimeError(f"Completed v2 artifact identity failed: {path}")
    if read_json(resolved(completed["gate_path"])).get("pass") is not False:
        raise RuntimeError("One-shot sweep is only valid after the selected-v2 failure")

    historical = contract["historical_action_head_timing"]
    historical_path = resolved(historical["artifact"])
    if not historical_path.is_file() or sha256(historical_path) != historical["sha256"]:
        raise RuntimeError("Historical action-head timing artifact identity failed")
    return contract


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_sweep_contract(args)
    v2_args = Namespace(
        repo_root=args.repo_root,
        campaign_root=args.campaign_root,
        contract=args.v2_contract,
        output_dir=args.output_dir,
        workers=args.workers,
        device=args.device,
    )
    context = preflight_v2(v2_args)
    if sha256(context["source_lock_path"]) != contract["immutable_inputs"][
        "source_lock_sha256"
    ]:
        raise RuntimeError("Stage-A source-lock identity differs from the v2 sweep")
    selection_candidates = {
        row["checkpoint_sha256"]: row for row in context["selection"]["candidates"]
    }
    candidates: list[dict[str, Any]] = []
    for expected in contract["candidates"]:
        digest = expected["checkpoint_sha256"]
        selected_row = selection_candidates.get(digest)
        if selected_row is None:
            raise RuntimeError(f"Contract candidate is absent from selection: {digest}")
        path = resolved(expected["checkpoint"])
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Checkpoint identity failed: {path}")
        if int(selected_row["global_microbatches"]) != int(
            expected["global_microbatches"]
        ):
            raise RuntimeError(f"Checkpoint budget mismatch: {path}")
        payload = torch.load(path, map_location="cpu")
        if payload.get("joint_stage") != "stage_a" or payload.get("joint_mode") != "joint":
            raise RuntimeError(f"Checkpoint is not a Stage-A joint surrogate: {path}")
        candidates.append({**expected, "path": path})
    if len(candidates) != 5 or len(candidates) != len(selection_candidates):
        raise RuntimeError("Frozen sweep must contain exactly all five Stage-A checkpoints")
    context["sweep_contract"] = contract
    context["sweep_candidates"] = candidates
    context["call_path_audit"] = audit_shared_feature_call_path(
        args.repo_root, contract
    )
    return context


def build_candidate_modules(
    model: torch.nn.Module,
    candidates: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    base = model.joint_latent_action_surrogate
    expected_keys = set(base.state_dict())
    prefix = "joint_latent_action_surrogate."
    modules: dict[str, torch.nn.Module] = {}
    for candidate in candidates:
        checkpoint_id = str(candidate["checkpoint_id"])
        module = copy.deepcopy(base).cpu()
        state = {
            key[len(prefix) :]: value
            for key, value in strip_state(candidate["path"]).items()
            if key.startswith(prefix)
        }
        if set(state) != expected_keys:
            raise RuntimeError(
                f"Joint module identity failed for {checkpoint_id}: "
                f"missing={sorted(expected_keys - set(state))[:20]} "
                f"extra={sorted(set(state) - expected_keys)[:20]}"
            )
        module.load_state_dict(state, strict=True)
        module.to(device).eval().requires_grad_(False)
        modules[checkpoint_id] = module
    return modules


def _assert_action_shape(tensor: torch.Tensor, batch_size: int, label: str) -> None:
    expected = (batch_size, 3, 7)
    if tuple(tensor.shape) != expected:
        raise RuntimeError(f"{label} must be {expected}, got {tuple(tensor.shape)}")


def add_required_pair_metrics(
    metrics: dict[str, list[float]],
    prefix: str,
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Collect per-example full-horizon and h=0/1/2 distributions."""

    if predicted.shape != target.shape or predicted.shape[-2:] != (3, 7):
        raise ValueError(
            f"Pair shape mismatch for {prefix}: {tuple(predicted.shape)} "
            f"vs {tuple(target.shape)}"
        )
    metrics[f"{prefix}_horizon_l1"].extend(per_sample_l1(predicted, target))
    for token in TOKEN_INDICES:
        predicted_token = predicted[:, token : token + 1]
        target_token = target[:, token : token + 1]
        metrics[f"{prefix}_token{token}_l1"].extend(
            per_sample_l1(predicted_token, target_token)
        )
        metrics[f"{prefix}_token{token}_arm_l1"].extend(
            per_sample_l1(predicted_token[..., :6], target_token[..., :6])
        )
        metrics[f"{prefix}_token{token}_gripper_probability_l1"].extend(
            per_sample_l1(predicted_token[..., 6:], target_token[..., 6:])
        )


def _summarize(values: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    return {name: distribution(items) for name, items in sorted(values.items())}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_callable(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        _sync(device)
        timings: list[float] = []
        for _ in range(repeats):
            _sync(device)
            started = time.perf_counter_ns()
            fn()
            _sync(device)
            timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return distribution(timings)


def build_latency_audit(
    *,
    model: torch.nn.Module,
    modules: dict[str, torch.nn.Module],
    latency_inputs: dict[str, Any],
    device: torch.device,
    contract: dict[str, Any],
    call_path: dict[str, Any],
) -> dict[str, Any]:
    spec = contract["latency_microbenchmark"]
    warmup = int(spec["warmup_calls"])
    repeats = int(spec["measured_calls"])
    z_anchor = latency_inputs["z_anchor"]
    z_current = latency_inputs["z_current"]
    feature = latency_inputs["feature"]
    anchor = latency_inputs["anchor"]
    elapsed = int(latency_inputs["elapsed"])
    exact = benchmark_callable(
        lambda: model.decode_action_from_latent(z_current),
        device=device,
        warmup=warmup,
        repeats=repeats,
    )
    candidates: dict[str, Any] = {}
    for checkpoint_id, module in modules.items():
        surrogate = benchmark_callable(
            lambda module=module: module.surrogate_forward(
                anchor_arm=anchor["arm"],
                anchor_gripper_logit=anchor["gripper_logit"],
                anchor_latent=z_anchor,
                current_latent=z_current,
                shared_feature=feature,
                elapsed=elapsed,
            ),
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        if module.surrogate is None:
            raise RuntimeError("Stage-A checkpoint has no surrogate module")
        projection = benchmark_callable(
            lambda module=module: module.surrogate.latent_projection(
                z_current - z_anchor
            ),
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        projected_delta = float(surrogate["mean"]) - float(exact["mean"])
        candidates[checkpoint_id] = {
            "surrogate_total_after_shared_feature_ms": surrogate,
            "latent_projection_ms": projection,
            "additional_observation_encoder_calls": int(
                call_path["additional_observation_encoder_calls"]
            ),
            "additional_observation_encoder_latency_ms": 0.0,
            "total_level0_incremental_latency_ms": float(surrogate["mean"]),
            "projected_hybrid_latency_delta_ms": projected_delta,
            "projected_hybrid_strictly_cheaper": projected_delta < 0.0,
            "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        }
    return {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "device": str(device),
        "batch_size": int(z_current.shape[0]),
        "warmup_calls": warmup,
        "measured_calls": repeats,
        "timing_method": "perf_counter_ns_with_device_synchronize_before_and_after",
        "exact_frozen_seer_action_head_ms": exact,
        "exact_frozen_seer_action_head_parameter_count": sum(
            parameter.numel()
            for module in model.get_action_head_modules()
            for parameter in module.parameters()
        ),
        "shared_encoder_cost_in_both_paths": True,
        "shared_encoder_cost_excluded_from_incremental_comparison": True,
        "call_path_audit": call_path,
        "historical_action_head_timing": contract["historical_action_head_timing"],
        "candidates": candidates,
    }


def checkpoint_metric_schema() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "action_representation": ACTION_REPRESENTATION,
        "sample_unit": "one frozen validation example",
        "l1_definition": "mean(abs(predicted-target)) over the named action dimensions",
        "distribution_fields": ["count", "mean", "p50", "p90", "p95", "p99", "max"],
        "regeneration_ages": list(AGES),
        "token_indices": list(TOKEN_INDICES),
        "metric_prefixes": {
            "joint_to_exact": "surrogate versus A_psi(z_t^L), primary",
            "hold_to_exact": "immutable aligned anchor versus A_psi(z_t^L)",
            "joint_to_teacher": "surrogate versus shifted Full Seer, diagnostic only",
            "hold_to_teacher": "hold versus shifted Full Seer, diagnostic only",
            "exact_to_teacher": "canonical exact head versus shifted Full Seer, diagnostic",
        },
        "per_prefix_metrics": {
            "horizon_l1": "mean over P=3 and all seven action dimensions",
            "token{h}_l1": "mean over seven dimensions for token h",
            "token{h}_arm_l1": "mean over six arm dimensions for token h",
            "token{h}_gripper_probability_l1": "absolute probability error for token h",
        },
        "equal_age_average": "(age_1_metric + age_2_metric) / 2",
        "temporal_ensemble_metric": EXACT_ENSEMBLE_METRIC_UNAVAILABLE,
        "temporal_ensemble_reason": (
            "Frozen validation rows are independent windows and do not contain episode-ordered "
            "candidate history keyed by execution timestep."
        ),
        "latency_units": "milliseconds per batch-size-one call",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "checkpoint_id",
        "global_microbatches",
        "fidelity_eligible",
        "age1_joint_to_exact_token0_mean",
        "age1_hold_to_exact_token0_mean",
        "age2_joint_to_exact_token0_mean",
        "age2_hold_to_exact_token0_mean",
        "exact_first_token_l1_mean_age_average",
        "exact_arm_first_token_l1_p95_age_average",
        "exact_full_horizon_l1_mean_age_average",
        "age1_joint_to_teacher_token0_mean_diagnostic",
        "age2_joint_to_teacher_token0_mean_diagnostic",
        "exact_ensemble_status",
        "surrogate_incremental_latency_ms",
        "exact_action_head_latency_ms",
        "projected_hybrid_latency_delta_ms",
        "additional_observation_encoder_calls",
        "parameter_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_commands(
    path: Path,
    *,
    verdict: str,
    selected: dict[str, Any] | None,
    contract: dict[str, Any],
) -> None:
    lines = [
        "# Commands after the one-shot sweep",
        "",
        f"Frozen verdict: `{verdict}`.",
        "",
    ]
    if verdict != "SEER_DEPLOYMENT_CANDIDATE" or selected is None:
        lines.extend(
            [
                "No Seer online command is authorized by the frozen protocol.",
                "Do not run Stage B, wide training, retraining, or LIBERO for this track.",
                "",
            ]
        )
    else:
        online = contract["conditional_online_evaluation"]
        lines.extend(
            [
                "The following command is prepared but was not executed:",
                "",
                "```bash",
                "cd /home/mingyujung/private/gnaroshi_vla",
                "conda activate seer_libero",
                f"PYTHONPATH=$PWD python tools/seer/verify_joint_latent_action_surrogate_source_lock.py --manifest {online['source_lock_manifest']} && \\",
                "CUDA_VISIBLE_DEVICES=4,5,6,7 \\",
                f"BASELINE_CKPT={online['teacher_checkpoint']} \\",
                f"OURS_CKPT={selected['checkpoint']} \\",
                f"LRNODE_INIT_ADAPTER_CKPT={online['canonical_adapter_checkpoint']} \\",
                f"LATENTLOOP_HIERARCHICAL_ACTION_CHECKPOINT={online['action_correction_checkpoint']} \\",
                f"VIT_CHECKPOINT_PATH={online['vit_checkpoint']} \\",
                f"LIBERO_PATH={online['libero_path']} \\",
                f"RESULT_ROOT={online['new_result_root']} \\",
                "EXPERIMENT_NAME=stage_a_one_shot_kf8_kg3 EXPERIMENT_TAG=one_shot_v2 \\",
                "BASELINE_NAME=local_teacher33 OURS_NAME=stage_a_one_shot_selected \\",
                "RUN_BASELINE=0 RUN_OURS_FULL=0 LRNODE_QUERY_INTERVALS_STR=8 \\",
                "LRNODE_TRAIN_PROTOCOL=adapter JOINT_LATENT_ACTION_SURROGATE_MODE=joint \\",
                "JOINT_FORCE_EXACT_ACTION_HEAD=0 JOINT_ERROR_TRACE=1 \\",
                "LATENTLOOP_HIERARCHICAL_MODE=hybrid \\",
                "LATENTLOOP_HIERARCHICAL_FULL_INTERVAL=8 \\",
                "LATENTLOOP_HIERARCHICAL_REGENERATION_INTERVAL=3 \\",
                "LATENTLOOP_HIERARCHICAL_TRACE=1 LATENTLOOP_HIERARCHICAL_ASSERT_INVARIANTS=1 \\",
                "EVAL_NUM_EPISODES_PER_TASK=20 EVAL_NUM_TASKS=10 NODE_NUM=4 \\",
                "EVAL_LIBERO_ENSEMBLING=1 SAVE_VIDEO=0 MASTER_PORT=15100 \\",
                "bash architectures/seer/upstream/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh",
                "```",
                "",
                "Only the selected Stage-A row is new. Full Seer K1, canonical K8, pure "
                "action-correction K8, and naive K_F=8/K_G=3 endpoint rows must be reused by "
                "the exact hashes recorded in the frozen contract.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_transfer_report(
    path: Path,
    *,
    verdict: str,
    selected: dict[str, Any] | None,
    checkpoint_payload: dict[str, Any],
    latency: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    selected_id = "none" if selected is None else str(selected["checkpoint_id"])
    selected_metrics = None if selected is None else checkpoint_payload[selected_id]
    exact_ms = float(latency["exact_frozen_seer_action_head_ms"]["mean"])
    surrogate_ms = (
        None
        if selected is None
        else float(
            latency["candidates"][selected_id][
                "surrogate_total_after_shared_feature_ms"
            ]["mean"]
        )
    )
    lines = [
        "# Seer surrogate transfer implication",
        "",
        f"- Frozen Seer verdict: `{verdict}`",
        f"- Selected checkpoint: `{selected_id}`",
        "- Exact target contract: `A_hat_t ~= G_psi(z_t^L)` in the same executed "
        "action representation used by the expensive generator.",
        "- No SimVLA or pi0.5 success is claimed by this audit.",
        "",
        "## What Stage A established",
        "",
        "The immutable-anchor, nonrecursive surrogate predicts a complete P=3 horizon from "
        "the aligned exact anchor, shared canonical motion feature, latent displacement, and age.",
    ]
    if selected_metrics is not None:
        lines.extend(
            [
                "Its executed-action fidelity is determined by the age-1/age-2 canonical "
                f"eligibility gate: `{'PASS' if selected['fidelity_eligible'] else 'FAIL'}`.",
                f"The measured incremental surrogate mean is `{surrogate_ms:.6f} ms`; the "
                f"same-runtime exact Seer head mean is `{exact_ms:.6f} ms`.",
            ]
        )
    else:
        lines.append("No existing checkpoint passed the canonical executed-action gate.")
    lines.extend(
        [
            "",
            "## Objective mismatch learned from Seer",
            "",
            "Stage A optimized logit-space `surrogate`, `executed_token`, `tail`, `gripper`, "
            "and residual-regularization terms; latent and latent-action weights were zero. "
            "Checkpoint selection used their weighted validation total. The deployment audit "
            "instead measures `[arm_6d, sigmoid(gripper_logit)]` against the frozen exact "
            "canonical head independently at ages 1 and 2. Therefore teacher fidelity and the "
            "weighted training total cannot substitute for executed-action fidelity.",
            "",
            "## Transfer contract",
            "",
            "For an expensive SimVLA or pi0.5 action generator, retain the hierarchy only if "
            "the surrogate consumes already-available shared features and directly minimizes "
            "the generator's actually executed action distribution at every deployment age. "
            "Measure generator latency and surrogate incremental latency on the same device and "
            "call boundary. Seer can validate target alignment, but its approximately 0.5 ms "
            "head makes it a weak endpoint for an efficiency claim.",
            "",
            "## Evidence",
            "",
        ]
    )
    for item in contract["prior_evidence"]:
        lines.append(f"- `{item['path']}` (`sha256={item['sha256']}`)")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_final_report(
    path: Path,
    *,
    ranked: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    verdict: str,
    latency: dict[str, Any],
    correction: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    lines = [
        "# Final Stage-A one-shot decision",
        "",
        f"Primary verdict: `{verdict}`.",
        f"Selected checkpoint: `{selected['checkpoint_id'] if selected else 'none'}`.",
        "",
        "## Newly computed checkpoint sweep",
        "",
        "| Rank | Checkpoint | Age 1 exact | Age 1 hold | Age 2 exact | Age 2 hold | Arm p95 avg | Horizon avg | Eligible | Surrogate ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in ranked:
        lines.append(
            f"| {row['rank']} | {row['checkpoint_id']} | "
            f"{row['age1_joint_to_exact_token0_mean']:.9f} | "
            f"{row['age1_hold_to_exact_token0_mean']:.9f} | "
            f"{row['age2_joint_to_exact_token0_mean']:.9f} | "
            f"{row['age2_hold_to_exact_token0_mean']:.9f} | "
            f"{row['exact_arm_first_token_l1_p95_age_average']:.9f} | "
            f"{row['exact_full_horizon_l1_mean_age_average']:.9f} | "
            f"{'PASS' if row['fidelity_eligible'] else 'FAIL'} | "
            f"{row['surrogate_incremental_latency_ms']:.6f} |"
        )
    exact_ms = float(latency["exact_frozen_seer_action_head_ms"]["mean"])
    lines.extend(
        [
            "",
            "## Metric contract",
            "",
            "All action errors use `[arm_6d, sigmoid(gripper_logit)]`. For each example, "
            "L1 is averaged over the named dimensions; mean and p50/p90/p95/p99 are then "
            "computed over 1,920 examples. Raw h=0/1/2 metrics are stored separately at "
            "regeneration ages 1 and 2. Full P=3 horizon error is secondary.",
            "",
            f"Temporal ensemble: `{EXACT_ENSEMBLE_METRIC_UNAVAILABLE}`. The frozen rows are "
            "independent validation windows without episode-ordered candidate history, so no "
            "executed ensemble metric or proxy was fabricated.",
            "",
            "## Eligibility and ranking",
            "",
            "Eligibility requires candidate h=0 canonical error below hold overall and at "
            "each age, arm h=0 p95 within 1.05x hold at each age, finite/noncollapsed gripper "
            "probabilities, and all identity/count/hash/shape checks. Full-teacher metrics are "
            "diagnostic only. Eligible checkpoints are ranked by exact ensemble error when "
            "available, equal age-average canonical h=0 mean, arm h=0 p95, full-horizon mean, "
            "then earliest microbatch.",
            "",
            "## Latency and call path",
            "",
            f"The same-runtime exact frozen Seer head mean is `{exact_ms:.6f} ms`. The "
            "surrogate timing begins after the canonical `u_delta` feature exists. The static "
            "call-path audit found zero additional observation-encoder calls; the shared "
            "encoder is not charged twice. `total_level0_incremental_latency_ms` is therefore "
            "the complete surrogate call, including latent projection, trunk, arm/gripper "
            "heads, clipping, and sigmoid.",
            "",
            "## Correction-signal diagnostic",
            "",
            f"Best canonical surrogate error / hold correction magnitude: "
            f"`{correction['best_surrogate_to_target_ratio']:.6f}`. "
            f"Assessment: `{correction['scientific_assessment']}`.",
            "",
            "## Files changed and reused",
            "",
            "Modified analysis-only files are listed in `provenance.json`. Reused code: the "
            "existing v2 runtime builder, frozen validation split, exact action decoder, "
            "canonical delta encoder/updater, immutable-anchor alignment, and all five saved "
            "Stage-A checkpoints. Immutable v1/v2 artifacts were not rewritten.",
            "",
            "## Next allowed action",
            "",
        ]
    )
    if verdict == "SEER_DEPLOYMENT_CANDIDATE":
        lines.append(
            "Only the prepared source-locked K=8, K_G=3, 200-paired-episode command may be "
            "reviewed next; it was not executed by this task."
        )
    elif verdict == "TRANSFER_ONLY_CANDIDATE":
        lines.append(
            "Preserve the target/alignment contract for an expensive action generator; do not "
            "spend a Seer 200-episode budget on an efficiency claim."
        )
    elif verdict == "STOP_SEER_SURROGATE":
        lines.append(
            "Close the Seer surrogate track. Do not retrain Stage A, run Stage B, train wide "
            "LatentLoop, or launch a joint-surrogate K=8 environment evaluation."
        )
    else:
        lines.append("Repair only the missing/corrupt evidence before making a decision.")
    lines.extend(
        [
            "",
            "## Prohibited duplicate experiments",
            "",
            "No Stage-A training, v1/v2 audit, canonical LatentLoop training, matched action "
            "correction, nonrecurrent training, Q1/Q2, K sweep, K_g sweep, Full Seer K1, "
            "canonical K4/K8, action-correction K4/K8, naive hybrid, checkpoint generation, "
            "loss calibration, source-lock generation, Stage B, wide training, or LIBERO was "
            "run or authorized.",
            "",
            "## Execution integrity",
            "",
            "This evaluator performed one offline user-launched pass only. It does not train, "
            "invoke LIBERO, modify `seer_node3`, or run git add/commit/push. Static test status "
            "is recorded in the preparation report that supplied the command.",
            "",
            "## Prior evidence (referenced, not duplicated)",
            "",
        ]
    )
    for item in contract["prior_evidence"]:
        lines.append(f"- `{item['path']}` (`sha256={item['sha256']}`)")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    output_files = (
        "checkpoint_metrics.csv",
        "checkpoint_metrics.json",
        "checkpoint_metric_schema.json",
        "ranking.json",
        "sweep_decision.json",
        "latency_audit.json",
        "correction_signal_diagnostic.json",
        "per_sample_metrics.jsonl",
        "provenance.json",
        "final_stage_a_one_shot_decision_report.md",
        "seer_surrogate_transfer_implication.md",
        "commands_to_run.md",
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite checkpoint sweep: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    previous_cwd = Path.cwd()
    upstream = args.repo_root / "architectures/seer/upstream"
    os.chdir(upstream)
    try:
        parser_args, model, data, device, load_report = build_runtime(args, context)
        modules = build_candidate_modules(model, context["sweep_candidates"], device)
        split = data.split_manifest
        runtime_split_identity = {
            split["validation_episode_keys_sha256"],
            split["selected_indices_sha256"],
        }
        if runtime_split_identity != set(context["selection"]["validation_split_identity"]):
            raise RuntimeError("Runtime validation split differs from frozen selection")

        common: dict[int, dict[str, list[float]]] = {
            age: defaultdict(list) for age in AGES
        }
        candidate_values: dict[str, dict[int, dict[str, list[float]]]] = {
            checkpoint_id: {age: defaultdict(list) for age in AGES}
            for checkpoint_id in modules
        }
        max_batches = int(context["sweep_contract"]["evaluation"]["max_batches"])
        expected_examples = int(
            context["sweep_contract"]["evaluation"]["expected_examples"]
        )
        seed = int(parser_args.seed)
        data.set_epoch(0)
        batches = 0
        tensor_shapes_pass = True
        latency_inputs: dict[str, Any] | None = None
        for batch_index, batch in enumerate(data.dataloader):
            if batch_index >= max_batches:
                break
            images_primary = batch[0].to(device, dtype=torch.float32)
            images_wrist = batch[3].to(device, dtype=torch.float32)
            states = batch[4].to(device, dtype=torch.float32)
            actions = batch[2].to(device, dtype=torch.float32).clone()
            batch_size = int(images_primary.shape[0])
            if images_primary.shape[1] < 9:
                raise RuntimeError("Stage-A sweep requires at least nine frames")
            text = batch[1].to(device).unsqueeze(1).repeat(1, 10, 1)
            states = torch.cat((states[..., :6], states[..., -2:]), dim=-1)
            actions[..., 6:] = (actions[..., 6:] + 1) // 2
            selected_step = 6

            with torch.no_grad():
                torch.manual_seed(seed + batch_index * 31)
                anchor_output = model(
                    image_primary=images_primary[:, :7],
                    image_wrist=images_wrist[:, :7],
                    state=states[:, :7],
                    text_token=text[:, :7],
                    action=actions[:, :7],
                    return_action_latent=True,
                    lrnode_compute_loss=False,
                )
                z_anchor = anchor_output["action_latent"][:, selected_step]
                anchor = model.decode_action_diagnostics_from_latent(z_anchor)
                anchor_raw = torch.cat(
                    (anchor["arm"], anchor["gripper_logit"]), dim=-1
                )
                z_current = z_anchor
                for age in AGES:
                    torch.manual_seed(seed + batch_index * 31 + age)
                    teacher_output = model(
                        image_primary=images_primary[:, age : age + 7],
                        image_wrist=images_wrist[:, age : age + 7],
                        state=states[:, age : age + 7],
                        text_token=text[:, age : age + 7],
                        action=actions[:, age : age + 7],
                        return_action_latent=True,
                        lrnode_compute_loss=False,
                    )
                    z_teacher = teacher_output["action_latent"][:, selected_step]
                    teacher = model.decode_action_diagnostics_from_latent(z_teacher)
                    previous_frame = selected_step + age - 1
                    current_frame = selected_step + age
                    feature = model.lrnode_encode_delta(
                        key_image_primary=images_primary[:, previous_frame],
                        key_image_wrist=images_wrist[:, previous_frame],
                        cur_image_primary=images_primary[:, current_frame],
                        cur_image_wrist=images_wrist[:, current_frame],
                        q_key=states[:, previous_frame],
                        q_cur=states[:, current_frame],
                    )
                    z_current = model.lrnode_apply_dynamics(
                        z_prev=z_current,
                        u_delta=feature,
                        dt=1.0,
                        age=float(age),
                    )
                    exact = model.decode_action_diagnostics_from_latent(z_current)
                    exact_action = action_from_arm_and_logit(
                        exact["arm"], exact["gripper_logit"]
                    )
                    teacher_action = action_from_arm_and_logit(
                        teacher["arm"], teacher["gripper_logit"]
                    )
                    aligned = align_immutable_anchor(anchor_raw, age)
                    hold_action = action_from_arm_and_logit(
                        aligned.values[..., :6], aligned.values[..., 6:]
                    )
                    for label, tensor in (
                        ("exact_action", exact_action),
                        ("teacher_action", teacher_action),
                        ("hold_action", hold_action),
                    ):
                        _assert_action_shape(tensor, batch_size, f"age{age}.{label}")
                    add_required_pair_metrics(
                        common[age], "hold_to_exact", hold_action, exact_action
                    )
                    add_required_pair_metrics(
                        common[age], "hold_to_teacher", hold_action, teacher_action
                    )
                    add_required_pair_metrics(
                        common[age], "exact_to_teacher", exact_action, teacher_action
                    )

                    for checkpoint_id, module in modules.items():
                        surrogate = module.surrogate_forward(
                            anchor_arm=anchor["arm"],
                            anchor_gripper_logit=anchor["gripper_logit"],
                            anchor_latent=z_anchor,
                            current_latent=z_current,
                            shared_feature=feature,
                            elapsed=age,
                        )
                        joint_action = action_from_arm_and_logit(
                            surrogate.arm, surrogate.gripper_logit
                        )
                        _assert_action_shape(
                            joint_action, batch_size, f"age{age}.{checkpoint_id}"
                        )
                        values = candidate_values[checkpoint_id][age]
                        add_required_pair_metrics(
                            values, "joint_to_exact", joint_action, exact_action
                        )
                        add_required_pair_metrics(
                            values, "joint_to_teacher", joint_action, teacher_action
                        )
                        values["joint_gripper_probability_all_tokens"].extend(
                            surrogate.gripper_probability.detach()
                            .float()
                            .reshape(-1)
                            .cpu()
                            .tolist()
                        )
                        for token in TOKEN_INDICES:
                            values[
                                f"joint_gripper_probability_token{token}"
                            ].extend(
                                surrogate.gripper_probability[:, token]
                                .detach()
                                .float()
                                .reshape(-1)
                                .cpu()
                                .tolist()
                            )
                    if latency_inputs is None and age == 1:
                        latency_inputs = {
                            "z_anchor": z_anchor[:1].detach(),
                            "z_current": z_current[:1].detach(),
                            "feature": feature[:1].detach(),
                            "anchor": {
                                "arm": anchor["arm"][:1].detach(),
                                "gripper_logit": anchor["gripper_logit"][:1].detach(),
                            },
                            "elapsed": age,
                        }
            batches += 1
            if batches == 1 or batches % 10 == 0 or batches == max_batches:
                print(
                    f"[ONE-SHOT SWEEP] batches={batches}/{max_batches} "
                    f"examples={len(common[1]['hold_to_exact_token0_l1'])} "
                    f"candidates={len(modules)}"
                )

        if batches != max_batches:
            raise RuntimeError(f"Expected {max_batches} batches, evaluated {batches}")
        common_summary = {age: _summarize(common[age]) for age in AGES}
        for age in AGES:
            if int(common_summary[age]["hold_to_exact_token0_l1"]["count"]) != (
                expected_examples
            ):
                raise RuntimeError(f"Age-{age} checkpoint sweep count mismatch")
        if latency_inputs is None:
            raise RuntimeError("Latency inputs were not captured")
        latency = build_latency_audit(
            model=model,
            modules=modules,
            latency_inputs=latency_inputs,
            device=device,
            contract=context["sweep_contract"],
            call_path=context["call_path_audit"],
        )

        checkpoint_rows: list[dict[str, Any]] = []
        checkpoint_payload: dict[str, Any] = {}
        for candidate in context["sweep_candidates"]:
            checkpoint_id = str(candidate["checkpoint_id"])
            summaries = {
                age: _summarize(candidate_values[checkpoint_id][age]) for age in AGES
            }
            gripper_finite: dict[int, bool] = {}
            gripper_noncollapsed: dict[int, bool] = {}
            gripper_diagnostics: dict[int, Any] = {}
            for age in AGES:
                gripper = np.asarray(
                    candidate_values[checkpoint_id][age][
                        "joint_gripper_probability_token0"
                    ],
                    dtype=np.float64,
                )
                gripper_finite[age] = bool(np.isfinite(gripper).all())
                gripper_noncollapsed[age] = bool(
                    gripper_finite[age]
                    and float(gripper.std()) > 1e-4
                    and float(gripper.max() - gripper.min()) > 0.05
                )
                gripper_diagnostics[age] = {
                    "count": int(gripper.size),
                    "finite": gripper_finite[age],
                    "std": float(gripper.std()),
                    "min": float(gripper.min()),
                    "max": float(gripper.max()),
                    "range": float(gripper.max() - gripper.min()),
                    "noncollapsed_threshold": "std>1e-4 and range>0.05",
                }
            eligibility = build_fidelity_eligibility(
                exact_first_token_mean_by_age={
                    age: float(summaries[age]["joint_to_exact_token0_l1"]["mean"])
                    for age in AGES
                },
                hold_first_token_mean_by_age={
                    age: float(common_summary[age]["hold_to_exact_token0_l1"]["mean"])
                    for age in AGES
                },
                exact_arm_first_token_p95_by_age={
                    age: float(
                        summaries[age]["joint_to_exact_token0_arm_l1"]["p95"]
                    )
                    for age in AGES
                },
                hold_arm_first_token_p95_by_age={
                    age: float(
                        common_summary[age]["hold_to_exact_token0_arm_l1"]["p95"]
                    )
                    for age in AGES
                },
                gripper_finite_by_age=gripper_finite,
                gripper_noncollapsed_by_age=gripper_noncollapsed,
                identity_checks={
                    "checkpoint_identity": True,
                    "source_hash_identity": True,
                    "validation_split_identity": True,
                    "expected_example_count": True,
                    "tensor_shapes": tensor_shapes_pass,
                    "action_representation": True,
                },
                ensemble_status=EXACT_ENSEMBLE_METRIC_UNAVAILABLE,
            )
            exact_horizon_average = sum(
                float(summaries[age]["joint_to_exact_horizon_l1"]["mean"])
                for age in AGES
            ) / 2.0
            latency_row = latency["candidates"][checkpoint_id]
            row = {
                "checkpoint_id": checkpoint_id,
                "checkpoint": str(candidate["path"]),
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "global_microbatches": int(candidate["global_microbatches"]),
                "fidelity_eligible": bool(eligibility["eligible"]),
                "exact_ensemble_status": EXACT_ENSEMBLE_METRIC_UNAVAILABLE,
                "exact_ensemble_executed_action_l1_mean": None,
                "exact_first_token_l1_mean_age_average": float(
                    eligibility["summary"]["exact_first_token_l1_mean_age_average"]
                ),
                "exact_arm_first_token_l1_p95_age_average": float(
                    eligibility["summary"][
                        "exact_arm_first_token_l1_p95_age_average"
                    ]
                ),
                "exact_full_horizon_l1_mean_age_average": exact_horizon_average,
                "age1_joint_to_exact_token0_mean": float(
                    summaries[1]["joint_to_exact_token0_l1"]["mean"]
                ),
                "age1_hold_to_exact_token0_mean": float(
                    common_summary[1]["hold_to_exact_token0_l1"]["mean"]
                ),
                "age2_joint_to_exact_token0_mean": float(
                    summaries[2]["joint_to_exact_token0_l1"]["mean"]
                ),
                "age2_hold_to_exact_token0_mean": float(
                    common_summary[2]["hold_to_exact_token0_l1"]["mean"]
                ),
                "age1_joint_to_teacher_token0_mean_diagnostic": float(
                    summaries[1]["joint_to_teacher_token0_l1"]["mean"]
                ),
                "age2_joint_to_teacher_token0_mean_diagnostic": float(
                    summaries[2]["joint_to_teacher_token0_l1"]["mean"]
                ),
                "surrogate_incremental_latency_ms": float(
                    latency_row["surrogate_total_after_shared_feature_ms"]["mean"]
                ),
                "exact_action_head_latency_ms": float(
                    latency["exact_frozen_seer_action_head_ms"]["mean"]
                ),
                "projected_hybrid_latency_delta_ms": float(
                    latency_row["projected_hybrid_latency_delta_ms"]
                ),
                "additional_observation_encoder_calls": int(
                    latency_row["additional_observation_encoder_calls"]
                ),
                "parameter_count": int(latency_row["parameter_count"]),
            }
            checkpoint_rows.append(row)
            checkpoint_payload[checkpoint_id] = {
                "identity": {
                    "checkpoint": row["checkpoint"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "global_microbatches": row["global_microbatches"],
                },
                "eligibility": eligibility,
                "gripper_diagnostics": gripper_diagnostics,
                "metrics_by_age": summaries,
                "latency": latency_row,
            }

        ranked, selected = select_eligible_checkpoint(checkpoint_rows)
        selected_latency = None if selected is None else latency["candidates"][
            selected["checkpoint_id"]
        ]
        verdict = utility_verdict(
            selected,
            surrogate_incremental_latency_ms=(
                None
                if selected_latency is None
                else float(selected_latency["total_level0_incremental_latency_ms"])
            ),
            exact_action_head_latency_ms=(
                None
                if selected is None
                else float(latency["exact_frozen_seer_action_head_ms"]["mean"])
            ),
            additional_observation_encoder_calls=(
                0
                if selected_latency is None
                else int(selected_latency["additional_observation_encoder_calls"])
            ),
            projected_hybrid_latency_delta_ms=(
                None
                if selected_latency is None
                else float(selected_latency["projected_hybrid_latency_delta_ms"])
            ),
        )
        best_diagnostic = ranked[0]
        hold_age_average = sum(
            float(common_summary[age]["hold_to_exact_token0_l1"]["mean"])
            for age in AGES
        ) / 2.0
        best_error = float(best_diagnostic["exact_first_token_l1_mean_age_average"])
        correction = {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "definition": "Delta a* = A_psi(z_t^L) - aligned immutable anchor",
            "action_representation": ACTION_REPRESENTATION,
            "metrics_by_age": {
                str(age): {
                    "overall": common_summary[age]["hold_to_exact_token0_l1"],
                    "arm": common_summary[age]["hold_to_exact_token0_arm_l1"],
                    "gripper": common_summary[age][
                        "hold_to_exact_token0_gripper_probability_l1"
                    ],
                }
                for age in AGES
            },
            "target_correction_mean_age_average": hold_age_average,
            "best_checkpoint": best_diagnostic["checkpoint_id"],
            "best_surrogate_error_mean_age_average": best_error,
            "best_surrogate_to_target_ratio": best_error / hold_age_average,
            "scientific_assessment": (
                "CORRECTION_SIGNAL_NOT_LARGER_THAN_SURROGATE_ERROR;_FURTHER_SEER_"
                "SURROGATE_TRAINING_UNATTRACTIVE"
                if hold_age_average <= best_error
                else "SURROGATE_ERROR_BELOW_TARGET_CORRECTION_SCALE"
            ),
        }
        decision = {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "primary_verdict": verdict,
            "selected": selected,
            "diagnostic_best": best_diagnostic,
            "fidelity_eligible_count": sum(
                bool(row["fidelity_eligible"]) for row in ranked
            ),
            "candidates_evaluated": len(ranked),
            "expected_candidates": 5,
            "exact_ensemble_status": EXACT_ENSEMBLE_METRIC_UNAVAILABLE,
            "environment_success_rate_used": False,
            "training_run": False,
            "libero_run": False,
            "does_not_modify_v1_or_v2": True,
        }
        metrics_payload = {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "action_representation": ACTION_REPRESENTATION,
            "batches": batches,
            "examples_per_age": expected_examples,
            "regeneration_ages": list(AGES),
            "token_indices": list(TOKEN_INDICES),
            "common_metrics_by_age": common_summary,
            "historical_recursive_reference": {
                "status": "DIAGNOSTIC_ONLY_AGE_UNAVAILABLE",
                "target": "shifted_context_full_seer_teacher",
                "metrics": context["recursive_reference"]["metrics"],
            },
            "temporal_ensemble": {
                "status": EXACT_ENSEMBLE_METRIC_UNAVAILABLE,
                "reason": checkpoint_metric_schema()["temporal_ensemble_reason"],
                "proxy_computed": False,
            },
            "checkpoints": checkpoint_payload,
        }
        ranking_payload = {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "ranking_rule": context["sweep_contract"]["ranking_rule"],
            "validation_only": True,
            "environment_success_rate_used": False,
            "ranking": ranked,
        }

        common_sample_names = {
            age: [
                name for name, values in common[age].items()
                if len(values) == expected_examples
            ]
            for age in AGES
        }
        candidate_sample_names = {
            checkpoint_id: {
                age: [
                    name
                    for name, values in candidate_values[checkpoint_id][age].items()
                    if len(values) == expected_examples
                ]
                for age in AGES
            }
            for checkpoint_id in modules
        }
        with (args.output_dir / "per_sample_metrics.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for index in range(expected_examples):
                row = {
                    "sample_index": index,
                    "common_by_age": {
                        str(age): {
                            name: common[age][name][index]
                            for name in common_sample_names[age]
                        }
                        for age in AGES
                    },
                    "checkpoints": {
                        checkpoint_id: {
                            str(age): {
                                name: candidate_values[checkpoint_id][age][name][index]
                                for name in candidate_sample_names[checkpoint_id][age]
                            }
                            for age in AGES
                        }
                        for checkpoint_id in modules
                    },
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        provenance = {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "contract": str(args.contract),
            "contract_sha256": sha256(args.contract),
            "v2_contract": str(args.v2_contract),
            "v2_contract_sha256": sha256(args.v2_contract),
            "evaluator": str(Path(__file__).resolve()),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "source_lock": str(context["source_lock_path"]),
            "source_lock_sha256": sha256(context["source_lock_path"]),
            "selection": str(context["selection_path"]),
            "selection_sha256": sha256(context["selection_path"]),
            "runtime_validation_split": split,
            "load_report": load_report,
            "candidate_checkpoint_sha256": {
                item["checkpoint_id"]: item["checkpoint_sha256"]
                for item in context["sweep_candidates"]
            },
            "device": str(device),
            "workers": int(args.workers),
            "call_path_audit": context["call_path_audit"],
            "files_modified_for_v2_sweep": context["sweep_contract"][
                "files_modified"
            ],
            "no_training": True,
            "no_libero": True,
            "seer_node3_modified": False,
            "git_operations_run": False,
        }
        for name, payload in (
            ("checkpoint_metrics.json", metrics_payload),
            ("checkpoint_metric_schema.json", checkpoint_metric_schema()),
            ("ranking.json", ranking_payload),
            ("sweep_decision.json", decision),
            ("latency_audit.json", latency),
            ("correction_signal_diagnostic.json", correction),
            ("provenance.json", provenance),
        ):
            (args.output_dir / name).write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        _write_csv(args.output_dir / "checkpoint_metrics.csv", ranked)
        _write_transfer_report(
            args.output_dir / "seer_surrogate_transfer_implication.md",
            verdict=verdict,
            selected=selected,
            checkpoint_payload=checkpoint_payload,
            latency=latency,
            contract=context["sweep_contract"],
        )
        _write_commands(
            args.output_dir / "commands_to_run.md",
            verdict=verdict,
            selected=selected,
            contract=context["sweep_contract"],
        )
        _write_final_report(
            args.output_dir / "final_stage_a_one_shot_decision_report.md",
            ranked=ranked,
            selected=selected,
            verdict=verdict,
            latency=latency,
            correction=correction,
            contract=context["sweep_contract"],
        )
        return decision
    finally:
        os.chdir(previous_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--v2-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in ("repo_root", "campaign_root", "contract", "v2_contract", "output_dir"):
        setattr(args, name, resolved(getattr(args, name)))
    return args


def main() -> None:
    args = parse_args()
    context = preflight(args)
    print(
        "[PREFLIGHT][OK] five checkpoints; frozen 1,920-example split; ages 1/2; "
        "canonical exact-head target; shared u_delta call path"
    )
    print(f"[PREFLIGHT] temporal_ensemble={EXACT_ENSEMBLE_METRIC_UNAVAILABLE}")
    if args.preflight_only:
        return
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stage-A checkpoint sweep")
    decision = evaluate(args, context)
    print(f"[DONE] primary_verdict={decision['primary_verdict']}")
    print(args.output_dir / "final_stage_a_one_shot_decision_report.md")


if __name__ == "__main__":
    main()
