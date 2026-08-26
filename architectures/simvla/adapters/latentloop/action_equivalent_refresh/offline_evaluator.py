"""Held-out matched-budget evaluation for action-equivalent refresh scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from architectures.simvla.adapters.latentloop.action_equivalent_refresh.checkpoint import (
    load_action_fidelity_checkpoint,
)
from architectures.simvla.adapters.latentloop.action_equivalent_refresh.training import (
    CompactActionFidelityDataset,
)
from methods.latentloop.modules.action_equivalent_refresh import (
    ExactCallBudgetCalibration,
    fit_sequential_exact_call_budget_calibration,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _predict(
    head: torch.nn.Module,
    dataset: CompactActionFidelityDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    arm: list[float] = []
    gripper: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(dataset), int(batch_size)):
            features = dataset.features[start : start + int(batch_size)].to(device)
            prediction = head(features)
            arm.extend(prediction.arm_q90.detach().cpu().tolist())
            gripper.extend(
                prediction.gripper_mismatch_probability.detach().cpu().tolist()
            )
    return arm, gripper


def _control_values(dataset: CompactActionFidelityDataset) -> dict[str, list[float]]:
    config = dataset.feature_config
    features = dataset.features
    group_base = config.delta_dim
    residual_rms = torch.stack(
        [
            features[:, group_base + group * 5 + 1]
            for group in range(config.num_token_groups)
        ],
        dim=1,
    ).amax(dim=1)
    gate_max = torch.stack(
        [
            features[:, group_base + group * 5 + 4]
            for group in range(config.num_token_groups)
        ],
        dim=1,
    ).amax(dim=1)
    age = [float(value["candidate_age"]) for value in dataset.routing_records]
    random_score = []
    for value in dataset.routing_records:
        payload = (
            f"20260827|{value['sequence_id']}|{value['anchor_offset']}|"
            f"{value['query_offset']}"
        ).encode("utf-8")
        integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        random_score.append(integer / float(2**64 - 1))
    return {
        "age_only": age,
        "applied_residual_rms": residual_rms.tolist(),
        "gate_max": gate_max.tolist(),
        "random_seed20260827": random_score,
    }


def _calibrate_scalar(
    values: Sequence[float],
    routing_records: Sequence[Mapping[str, Any]],
    *,
    target_exact_fraction: float,
    max_age: int,
) -> ExactCallBudgetCalibration:
    return fit_sequential_exact_call_budget_calibration(
        values,
        values,
        routing_records,
        target_exact_fraction=target_exact_fraction,
        max_approximate_age=max_age,
    )


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return float(torch.quantile(tensor, float(q)).item())


def _route_metrics(
    dataset: CompactActionFidelityDataset,
    scores: Sequence[float],
    *,
    threshold: float,
    max_age: int,
) -> dict[str, Any]:
    if len(scores) != len(dataset) or not dataset.routing_records:
        raise ValueError("routing scores require aligned all-anchor records")
    rows: dict[str, dict[tuple[int, int], int]] = {}
    for index, raw in enumerate(dataset.routing_records):
        sequence = str(raw["sequence_id"])
        key = (int(raw["anchor_offset"]), int(raw["query_offset"]))
        if key in rows.setdefault(sequence, {}):
            raise ValueError(f"duplicate routing row: {sequence} {key}")
        rows[sequence][key] = index

    arm = dataset.payload["arm_normalized_l1"].float()
    direction = dataset.payload["direction_cosine_error"].float()
    gripper = dataset.payload["gripper_mismatch"].float()
    exact_count = 0
    total_queries = 0
    approximate_indices: list[int] = []
    exact_indices: list[int] = []
    for sequence in sorted(rows):
        exact_count += 1
        total_queries += 1
        last_exact = 0
        for query in (1, 2, 3):
            key = (last_exact, query)
            if key not in rows[sequence]:
                raise ValueError(f"missing reachable row: {sequence} {key}")
            index = rows[sequence][key]
            age = query - last_exact
            choose_exact = age > int(max_age) or float(scores[index]) >= float(threshold)
            total_queries += 1
            if choose_exact:
                exact_count += 1
                exact_indices.append(index)
                last_exact = query
            else:
                approximate_indices.append(index)

    approximate_arm = [float(arm[index]) for index in approximate_indices]
    approximate_direction = [float(direction[index]) for index in approximate_indices]
    approximate_gripper = [float(gripper[index]) for index in approximate_indices]
    total_arm = sum(approximate_arm) / max(total_queries, 1)
    total_gripper = sum(approximate_gripper)
    return {
        "sequences": len(rows),
        "total_queries_including_q0": total_queries,
        "exact_calls": exact_count,
        "exact_fraction": exact_count / max(total_queries, 1),
        "approximate_calls": len(approximate_indices),
        "counterfactual_arm_error_mean_over_all_queries": total_arm,
        "counterfactual_gripper_mismatches_over_all_queries": int(total_gripper),
        "approximate_arm_error": {
            "mean": sum(approximate_arm) / max(len(approximate_arm), 1),
            "p90": _quantile(approximate_arm, 0.90),
            "p95": _quantile(approximate_arm, 0.95),
            "max": max(approximate_arm, default=0.0),
        },
        "approximate_direction_cosine_error": {
            "mean": sum(approximate_direction) / max(len(approximate_direction), 1),
            "p95": _quantile(approximate_direction, 0.95),
        },
        "approximate_gripper_mismatch": {
            "count": int(total_gripper),
            "rate": total_gripper / max(len(approximate_gripper), 1),
        },
        "selected_exact_candidate_rows": len(exact_indices),
    }


def _apply_calibration(
    calibration: ExactCallBudgetCalibration,
    left: Sequence[float],
    right: Sequence[float],
) -> list[float]:
    return [
        calibration.score_values(float(a), float(b))
        for a, b in zip(left, right)
    ]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    head, feature_config, learned_calibration, checkpoint = (
        load_action_fidelity_checkpoint(args.checkpoint, device=device)
    )
    validation = CompactActionFidelityDataset(
        args.validation_data, expected_split="checkpoint_validation"
    )
    final = CompactActionFidelityDataset(
        args.final_data, expected_split="final_offline"
    )
    if validation.feature_config != feature_config or final.feature_config != feature_config:
        raise ValueError("checkpoint and compact feature contracts differ")
    if not validation.routing_records or not final.routing_records:
        raise ValueError("offline evaluation requires all-anchor routing records")
    validation_arm, validation_gripper = _predict(
        head, validation, batch_size=args.batch_size, device=device
    )
    final_arm, final_gripper = _predict(
        head, final, batch_size=args.batch_size, device=device
    )
    learned_scores = _apply_calibration(
        learned_calibration, final_arm, final_gripper
    )
    selectors: dict[str, dict[str, Any]] = {
        "learned_action_fidelity": _route_metrics(
            final,
            learned_scores,
            threshold=learned_calibration.route_threshold,
            max_age=feature_config.max_age,
        )
    }
    validation_controls = _control_values(validation)
    final_controls = _control_values(final)
    calibrations: dict[str, Any] = {
        "learned_action_fidelity": learned_calibration.to_dict()
    }
    for name in sorted(validation_controls):
        calibration = _calibrate_scalar(
            validation_controls[name],
            validation.routing_records,
            target_exact_fraction=float(args.target_exact_fraction),
            max_age=feature_config.max_age,
        )
        scores = _apply_calibration(
            calibration, final_controls[name], final_controls[name]
        )
        selectors[name] = _route_metrics(
            final,
            scores,
            threshold=calibration.route_threshold,
            max_age=feature_config.max_age,
        )
        calibrations[name] = calibration.to_dict()

    oracle_validation_arm = validation.payload["arm_normalized_l1"].tolist()
    oracle_validation_gripper = validation.payload["gripper_mismatch"].tolist()
    oracle_calibration = fit_sequential_exact_call_budget_calibration(
        oracle_validation_arm,
        oracle_validation_gripper,
        validation.routing_records,
        target_exact_fraction=float(args.target_exact_fraction),
        max_approximate_age=feature_config.max_age,
    )
    oracle_scores = _apply_calibration(
        oracle_calibration,
        final.payload["arm_normalized_l1"].tolist(),
        final.payload["gripper_mismatch"].tolist(),
    )
    selectors["oracle_counterfactual_upper_bound"] = _route_metrics(
        final,
        oracle_scores,
        threshold=oracle_calibration.route_threshold,
        max_age=feature_config.max_age,
    )
    calibrations["oracle_counterfactual_upper_bound"] = oracle_calibration.to_dict()

    learned = selectors["learned_action_fidelity"]
    controls = [
        selectors[name]
        for name in selectors
        if name not in {"learned_action_fidelity", "oracle_counterfactual_upper_bound"}
    ]
    comparable = [
        value
        for value in controls
        if abs(value["exact_fraction"] - learned["exact_fraction"]) <= 0.03
    ]
    learned_beats_control_arm = bool(comparable) and all(
        learned["approximate_arm_error"]["p95"]
        <= value["approximate_arm_error"]["p95"]
        for value in comparable
    )
    learned_beats_control_gripper = bool(comparable) and all(
        learned["approximate_gripper_mismatch"]["count"]
        <= value["approximate_gripper_mismatch"]["count"]
        for value in comparable
    )
    result = {
        "verdict": "OFFLINE_ACTION_FIDELITY_COMPARISON_COMPLETE",
        "scientific_decision": "DESCRIPTIVE_ONLY_NO_AUTOMATIC_ONLINE_APPROVAL",
        "target_exact_fraction": float(args.target_exact_fraction),
        "feature_config": feature_config.to_dict(),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "validation_data": str(Path(args.validation_data).expanduser().resolve()),
        "final_data": str(Path(args.final_data).expanduser().resolve()),
        "selectors": selectors,
        "calibrations": calibrations,
        "comparison": {
            "matched_controls_within_3pp": len(comparable),
            "learned_no_worse_than_all_matched_controls_arm_p95": learned_beats_control_arm,
            "learned_no_worse_than_all_matched_controls_gripper_count": learned_beats_control_gripper,
            "online_candidate": learned_beats_control_arm and learned_beats_control_gripper,
            "note": "This is a transparent Pareto diagnostic, not a literature-derived pass threshold.",
        },
        "checkpoint_metadata": checkpoint.get("metadata", {}),
    }
    _atomic_json(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--final-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-exact-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
