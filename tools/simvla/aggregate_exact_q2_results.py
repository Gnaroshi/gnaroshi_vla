#!/usr/bin/env python3
"""Loss-contract, selection, stop-gate, and final-gate utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.simvla_exact_q2.decisions import (  # noqa: E402
    evaluate_exact_q2_offline_gate,
    evaluate_short_budget_gate,
    online_native_r5_enabled,
)
from methods.simvla_exact_q2.losses import load_approved_loss_contract  # noqa: E402
from methods.simvla_exact_q2.validation import select_validation_checkpoint  # noqa: E402


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path: str | Path, payload: dict) -> Path:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loss_template(args: argparse.Namespace) -> dict:
    recurrent = _load(args.recurrent_scales)
    direct = _load(args.direct_scales)
    return {
        "schema_version": "simvla_exact_q2_loss_contract_v1",
        "experiment_identifier": "simvla_r5_exact_q2_regeneration",
        "approval_status": "REQUIRES_RAW_SCALE_REVIEW",
        "calibration": {
            "recurrent_path": str(Path(args.recurrent_scales).resolve()),
            "recurrent_sha256": _sha(args.recurrent_scales),
            "direct_path": str(Path(args.direct_scales).resolve()),
            "direct_sha256": _sha(args.direct_scales),
            "recurrent_raw_scales": recurrent,
            "direct_raw_scales": direct,
        },
        "weights": {
            "q2_prefix": None,
            "q2_chunk": None,
            "q2_condition": None,
            "q1_auxiliary": None,
            "update_regularization": None,
        },
        "intended_weighted_contributions": {
            "q2_prefix": None,
            "q2_chunk": None,
            "q2_condition": None,
            "q1_auxiliary": None,
            "update_regularization": None,
        },
        "review_rule": "Choose weights only after inspecting both raw-scale calibrations; q2_prefix must be the largest intended weighted contribution.",
    }


def validation_rows(directory: str | Path) -> list[dict]:
    paths = sorted(Path(directory).rglob("validation_step_*.json"))
    return [_load(path) for path in paths]


def aggregate_summary(args: argparse.Namespace) -> dict:
    short_gate = _load(args.short_gate)
    offline_gate = _load(args.offline_gate) if args.offline_gate else None
    selections = {}
    for candidate, path in (
        ("recurrent_exact_q2", args.recurrent_selection),
        ("direct_exact_q2", args.direct_selection),
    ):
        if path:
            selections[candidate] = _load(path)
    if offline_gate is not None and offline_gate.get("EXACT_Q2_OFFLINE_PASS"):
        overall = "EXACT_Q2_OFFLINE_PASS"
    elif offline_gate is not None:
        overall = "STOP_SIMVLA_CONDITION_REGENERATION"
    elif short_gate.get("verdict") == "EARLY_STOP_BOTH":
        overall = "STOP_SIMVLA_CONDITION_REGENERATION"
    else:
        overall = "FULL_OFFLINE_GATE_PENDING"
    return {
        "schema_version": "simvla_exact_q2_result_summary_v1",
        "experiment_identifier": "simvla_r5_exact_q2_regeneration",
        "overall_status": overall,
        "short_gate": short_gate,
        "selections": selections,
        "offline_gate": offline_gate,
        "online_native_r5_allowed": bool(
            offline_gate and offline_gate.get("ONLINE_NATIVE_R5_ALLOWED")
        ),
    }


def online_plan(args: argparse.Namespace) -> dict:
    gate = _load(args.offline_gate)
    if not online_native_r5_enabled(gate):
        raise RuntimeError("online native-R5 planning is blocked by the full offline gate")
    selections = {
        candidate: _load(path)
        for candidate, path in (
            ("recurrent_exact_q2", args.recurrent_selection),
            ("direct_exact_q2", args.direct_selection),
        )
        if path
    }
    passing = [
        candidate
        for candidate, row in gate["candidates"].items()
        if row.get("pass")
    ]
    missing = [candidate for candidate in passing if candidate not in selections]
    if missing:
        raise RuntimeError(f"passing candidates lack validation-only selections: {missing}")
    selected = min(
        passing,
        key=lambda candidate: (
            selections[candidate]["selected_metrics"]["q2_prefix_l1"]["mean"],
            selections[candidate]["selected_metrics"]["q2_prefix_l1"]["p95"],
            selections[candidate]["selected_step"],
        ),
    )
    rows = [
        "full_native_simvla_k1",
        "pure_action_correction",
        f"selected_exact_q2_regeneration:{selected}",
        f"q1_action_correction_q2_regeneration_hybrid:{selected}",
        "hold_stale_action_chunk",
    ]
    rows.extend(
        f"second_exact_q2_candidate:{candidate}"
        for candidate in passing
        if candidate != selected
    )
    return {
        "schema_version": "simvla_exact_q2_online_matrix_plan_v1",
        "experiment_identifier": "simvla_r5_exact_q2_regeneration",
        "offline_gate_sha256": _sha(args.offline_gate),
        "offline_gate_verdict": gate["verdict"],
        "selected_endpoint": selected,
        "candidate_checkpoints": {
            candidate: selections[candidate]["selected_checkpoint"]
            for candidate in passing
        },
        "protocol": {
            "suite": "libero_10",
            "tasks": 10,
            "paired_episodes_per_task": 20,
            "episodes_per_row": 200,
            "execution_horizon_R": 5,
            "identical_task_episode_noise_manifest_required": True,
        },
        "rows": rows,
        "execution_enabled": False,
        "activation_status": "POST_GATE_ONLINE_RUNNER_REVIEW_REQUIRED",
        "reason": "This task prepares the fixed q2 functional gate only; it does not silently reuse the incompatible historical age-1 online policy.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    command = sub.add_parser("loss-template")
    command.add_argument("--recurrent-scales", required=True)
    command.add_argument("--direct-scales", required=True)
    command.add_argument("--output", required=True)

    command = sub.add_parser("approve-loss-contract")
    command.add_argument("--reviewed-json", required=True)
    command.add_argument("--output", required=True)

    command = sub.add_parser("select")
    command.add_argument("--candidate", required=True, choices=("recurrent_exact_q2", "direct_exact_q2"))
    command.add_argument("--validation-dir", required=True)
    command.add_argument("--output", required=True)

    command = sub.add_parser("short-gate")
    command.add_argument("--recurrent-validation-dir", required=True)
    command.add_argument("--direct-validation-dir", required=True)
    command.add_argument("--output", required=True)

    command = sub.add_parser("offline-gate")
    command.add_argument("--recurrent-result", default="")
    command.add_argument("--direct-result", default="")
    command.add_argument("--output", required=True)

    command = sub.add_parser("summary")
    command.add_argument("--short-gate", required=True)
    command.add_argument("--recurrent-selection", default="")
    command.add_argument("--direct-selection", default="")
    command.add_argument("--offline-gate", default="")
    command.add_argument("--output", required=True)

    command = sub.add_parser("online-plan")
    command.add_argument("--offline-gate", required=True)
    command.add_argument("--recurrent-selection", default="")
    command.add_argument("--direct-selection", default="")
    command.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.mode == "loss-template":
        payload = loss_template(args)
    elif args.mode == "approve-loss-contract":
        payload = _load(args.reviewed_json)
        if payload.get("approval_status") != "APPROVED_AFTER_RAW_SCALE_REVIEW":
            raise ValueError("reviewed JSON must explicitly set APPROVED_AFTER_RAW_SCALE_REVIEW")
        load_approved_loss_contract(args.reviewed_json)
    elif args.mode == "select":
        payload = select_validation_checkpoint(
            validation_rows(args.validation_dir), candidate=args.candidate
        )
    elif args.mode == "short-gate":
        payload = evaluate_short_budget_gate(
            {
                "recurrent_exact_q2": validation_rows(args.recurrent_validation_dir),
                "direct_exact_q2": validation_rows(args.direct_validation_dir),
            }
        )
    elif args.mode == "offline-gate":
        rows = {}
        if args.recurrent_result:
            rows["recurrent_exact_q2"] = _load(args.recurrent_result)
        if args.direct_result:
            rows["direct_exact_q2"] = _load(args.direct_result)
        payload = evaluate_exact_q2_offline_gate(rows)
    elif args.mode == "summary":
        payload = aggregate_summary(args)
    else:
        payload = online_plan(args)
    output = _write_new(args.output, payload)
    print(json.dumps({"mode": args.mode, "output": str(output), "verdict": payload.get("verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
