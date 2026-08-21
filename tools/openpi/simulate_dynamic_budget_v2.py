#!/usr/bin/env python3
"""Episode-ordered V2 scheduler calibration on the frozen calibration role."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from architectures.openpi.adapters.latentloop.v2_schedule_state import V2ScheduleState
from defect_split_common import (
    episode_key,
    load_contract,
    sha256_file,
    verify_offline_summary,
)
from methods.variable_time_latentloop.budget_calibration import (
    BudgetCalibration,
    MonotonicBinnedCalibrator,
)
from methods.variable_time_latentloop.decisions import RefreshDecision
from methods.variable_time_latentloop.operation_counters_v2 import (
    OperationCountersV2,
    full_hook_query,
    latent_query,
)
from pi05_stage_gate_v2 import verify_stage


@dataclass(frozen=True)
class EpisodeSimulation:
    episode: tuple[str, int, str, str]
    policy_queries: int
    full_prefix_calls: int
    direct_reanchors: int
    executed_actions: int
    selected_first_r_errors: tuple[float, ...]
    decisions: tuple[str, ...]
    operation_counters: dict[str, int]

    @property
    def k_q_hat(self) -> float:
        return self.policy_queries / self.full_prefix_calls

    @property
    def k_a_hat(self) -> float:
        return self.executed_actions / self.full_prefix_calls


def _row_episode(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row["suite"]),
        int(row.get("benchmark_task_index", row.get("task_id"))),
        str(row.get("episode_namespace", "teacher_demonstration")),
        str(row.get("episode_id", row.get("trial"))),
    )


def _value(row: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return float(row[name])
    raise ValueError(f"scheduler row is missing one of {names}")


def simulate_episode(
    rows: Iterable[dict[str, Any]],
    *,
    predicted_error: Callable[[float], float],
    direct_threshold: float,
    full_threshold: float,
    m_seq: int,
    m_full: int,
    execution_horizon: int = 5,
    flow_steps: int = 10,
) -> EpisodeSimulation:
    ordered = sorted(rows, key=lambda row: int(row["query_index"]))
    if not ordered or int(ordered[0]["query_index"]) != 0:
        raise ValueError("each calibration episode must begin with mandatory query 0")
    indices = [int(row["query_index"]) for row in ordered]
    if indices != list(range(len(indices))):
        raise ValueError(f"calibration episode has noncontiguous policy queries: {indices[:8]}")
    identities = {_row_episode(row) for row in ordered}
    if len(identities) != 1:
        raise ValueError("scheduler simulation may not cross episode boundaries")
    if direct_threshold > full_threshold:
        raise ValueError("direct threshold must not exceed full-refresh threshold")

    schedule = V2ScheduleState(m_seq=m_seq, m_full=m_full)
    counters = OperationCountersV2()
    decisions: list[str] = []
    selected_errors: list[float] = []
    full_calls = 0
    direct_events = 0
    executed_total = 0
    for row in ordered:
        query = int(row["query_index"])
        actual_actions = int(row.get("executed_actions_actual", execution_horizon))
        if not 1 <= actual_actions <= execution_horizon:
            raise ValueError("executed_actions_actual must include terminal partial chunks in [1,R]")
        if query == 0:
            decision = RefreshDecision.FULL_PREFIX
            query_cost = full_hook_query(flow_steps)
        else:
            forced = schedule.forced_decision(query)
            full_age = query - int(schedule.last_full_query)
            max_age_safety_full = forced is RefreshDecision.FULL_PREFIX and full_age not in {1, 2, 3}
            query_cost = (
                full_hook_query(flow_steps)
                if max_age_safety_full
                else latent_query(flow_steps, direct=True)
            )
            if forced is not None:
                decision = forced
            else:
                error = predicted_error(_value(row, "latent_defect", "defect_score"))
                if error >= full_threshold:
                    decision = RefreshDecision.FULL_PREFIX
                elif error >= direct_threshold:
                    decision = RefreshDecision.DIRECT_REANCHOR
                else:
                    decision = RefreshDecision.SEQUENTIAL
            if decision is RefreshDecision.FULL_PREFIX and not max_age_safety_full:
                query_cost.prefix_transformer_calls += 1
                query_cost.full_prefix_refreshes += 1
            elif decision is RefreshDecision.DIRECT_REANCHOR:
                query_cost.direct_reanchor_events += 1
        if decision is RefreshDecision.FULL_PREFIX:
            selected = _value(row, "full_executed_mse", "teacher_full_executed_mse", "direct_executed_mse")
            full_calls += 1
        elif decision is RefreshDecision.DIRECT_REANCHOR:
            selected = _value(row, "direct_executed_mse")
            direct_events += 1
        else:
            selected = _value(row, "sequential_executed_mse")
        schedule.commit(decision, query, actual_actions)
        counters.add(query_cost)
        decisions.append(decision.name.lower())
        selected_errors.append(selected)
        executed_total += actual_actions
        if str(row.get("terminal", "false")).lower() in {"1", "true", "yes"} and query != indices[-1]:
            raise ValueError("rows after an explicitly terminal calibration query are forbidden")
    return EpisodeSimulation(
        episode=next(iter(identities)),
        policy_queries=len(ordered),
        full_prefix_calls=full_calls,
        direct_reanchors=direct_events,
        executed_actions=executed_total,
        selected_first_r_errors=tuple(selected_errors),
        decisions=tuple(decisions),
        operation_counters=counters.to_dict(),
    )


def aggregate_simulations(simulations: Iterable[EpisodeSimulation]) -> dict[str, Any]:
    values = list(simulations)
    if not values:
        raise ValueError("budget simulation requires at least one episode")
    queries = sum(item.policy_queries for item in values)
    full = sum(item.full_prefix_calls for item in values)
    actions = sum(item.executed_actions for item in values)
    errors = np.asarray(
        [error for item in values for error in item.selected_first_r_errors], dtype=np.float64
    )
    counters = OperationCountersV2()
    for item in values:
        counters.add(OperationCountersV2(**item.operation_counters))
    return {
        "episodes": len(values),
        "N_q": queries,
        "N_F": full,
        "actual_executed_actions": actions,
        "K_q_hat": queries / full,
        "K_a_hat": actions / full,
        "first_r_mean_error": float(errors.mean()),
        "first_r_p95_error": float(np.quantile(errors, 0.95)),
        "direct_reanchors": sum(item.direct_reanchors for item in values),
        "operation_counters": counters.to_dict(),
    }


def select_schedule(
    rows: list[dict[str, Any]],
    calibrator: MonotonicBinnedCalibrator,
    *,
    k_q_min: float = 3.8,
    k_q_max: float = 4.2,
) -> tuple[dict[str, Any], list[EpisodeSimulation]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_row_episode(row), []).append(row)
    predicted_values = np.asarray(
        [float(calibrator.predict(np.asarray([_value(row, "latent_defect", "defect_score")]))[0]) for row in rows],
        dtype=np.float64,
    )
    candidates = sorted(set(float(value) for value in predicted_values))
    candidates.append(float(np.nextafter(predicted_values.max(), np.inf)))
    feasible: list[tuple[tuple[float, float, float, int], dict[str, Any], list[EpisodeSimulation]]] = []
    def predict(score: float) -> float:
        return float(calibrator.predict(np.asarray([score]))[0])
    for m_seq in (1, 2, 3):
        for m_full in (2, 3, 4):
            for direct_threshold in candidates:
                for full_threshold in candidates:
                    if direct_threshold > full_threshold:
                        continue
                    simulations = [
                        simulate_episode(
                            episode_rows,
                            predicted_error=predict,
                            direct_threshold=direct_threshold,
                            full_threshold=full_threshold,
                            m_seq=m_seq,
                            m_full=m_full,
                        )
                        for episode_rows in grouped.values()
                    ]
                    aggregate = aggregate_simulations(simulations)
                    if not k_q_min <= aggregate["K_q_hat"] <= k_q_max:
                        continue
                    tie_break = (
                        aggregate["first_r_mean_error"],
                        aggregate["first_r_p95_error"],
                        abs(aggregate["K_q_hat"] - 4.0),
                        aggregate["direct_reanchors"],
                    )
                    selection = {
                        "direct_reanchor_threshold": direct_threshold,
                        "full_refresh_threshold": full_threshold,
                        "M_seq": m_seq,
                        "M_full": m_full,
                        **aggregate,
                    }
                    feasible.append((tie_break, selection, simulations))
    if not feasible:
        raise RuntimeError("no scheduler candidate satisfies 3.8 <= K_q_hat <= 4.2")
    _, selection, simulations = min(feasible, key=lambda item: item[0])
    return selection, simulations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-metrics", required=True)
    parser.add_argument("--scheduler-summary", required=True)
    parser.add_argument("--defect-fit", required=True)
    parser.add_argument("--defect-validity", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise RuntimeError("set --run only after scheduler-calibration inputs are frozen")
    output = Path(args.output).resolve()
    gate = verify_stage(
        "stage11_scheduler_calibration",
        args.source_lock,
        [args.defect_validity],
        output_candidate=output,
    )
    role_sets, split_payload = load_contract(args.split_contract)
    if split_payload.get("source_lock_id") != gate["source_lock_id"]:
        raise RuntimeError("defect split contract was frozen under another source lock")
    producer = verify_offline_summary(
        args.scheduler_summary,
        metrics_path=args.scheduler_metrics,
        contract_path=args.split_contract,
        role="scheduler_calibration",
        source_lock_id=gate["source_lock_id"],
    )
    rows = list(csv.DictReader(Path(args.scheduler_metrics).open(encoding="utf-8")))
    observed = {episode_key(row) for row in rows}
    expected = role_sets["scheduler_calibration"]
    if observed != expected:
        raise ValueError("scheduler metrics must exactly cover the frozen scheduler-calibration role")
    fit = json.loads(Path(args.defect_fit).read_text(encoding="utf-8"))
    validity = json.loads(Path(args.defect_validity).read_text(encoding="utf-8"))
    producer_keys = (
        "adapter_checkpoint",
        "adapter_checkpoint_sha256",
        "base_checkpoint",
        "base_checkpoint_sha256",
        "cache_manifest",
        "cache_manifest_id",
        "cache_manifest_sha256",
        "final_evaluation_manifest",
        "final_evaluation_manifest_sha256",
    )
    expected_producer = {key: producer[key] for key in producer_keys}
    if fit.get("DEFECT_FIT_PASS") is not True or validity.get("DEFECT_VALIDITY_PASS") is not True:
        raise RuntimeError("defect fit and independent validity gates must both pass")
    if fit.get("source_lock_id") != gate["source_lock_id"] or validity.get("source_lock_id") != gate["source_lock_id"]:
        raise RuntimeError("source mismatch: defect artifacts are stale")
    if (
        fit.get("role") != "defect_fit"
        or validity.get("role") != "defect_validity"
        or validity.get("fit_role") != "defect_fit"
        or fit.get("split_contract_sha256") != sha256_file(args.split_contract)
        or validity.get("split_contract_sha256") != sha256_file(args.split_contract)
        or fit.get("defect_split_contract_id") != split_payload["defect_split_contract_id"]
        or validity.get("defect_split_contract_id") != split_payload["defect_split_contract_id"]
        or validity.get("fit_sha256") != sha256_file(args.defect_fit)
        or fit.get("producer") != expected_producer
        or validity.get("producer") != expected_producer
    ):
        raise RuntimeError("defect producer provenance is incomplete or inconsistent")
    if (
        str(Path(args.model_checkpoint).resolve()) != producer["adapter_checkpoint"]
        or sha256_file(args.model_checkpoint) != producer["adapter_checkpoint_sha256"]
    ):
        raise RuntimeError("scheduler model checkpoint differs from offline metric producer")
    calibrator = MonotonicBinnedCalibrator.from_dict(fit["calibrator"])
    selection, simulations = select_schedule(rows, calibrator)
    calibration = BudgetCalibration(
        low_threshold=float(selection["direct_reanchor_threshold"]),
        high_threshold=float(selection["full_refresh_threshold"]),
        target_full_prefix_ratio=0.25,
        validation_full_prefix_ratio=1.0 / float(selection["K_q_hat"]),
        validation_direct_ratio=float(selection["direct_reanchors"]) / float(selection["N_q"]),
        validation_selected_error=float(selection["first_r_mean_error"]),
        calibrator=calibrator,
    )
    payload = {
        "schema_version": 2,
        "frozen": True,
        "DYNAMIC_BUDGET_LOCK_PASS": True,
        "source_lock_id": gate["source_lock_id"],
        "model_checkpoint": str(Path(args.model_checkpoint).resolve()),
        "model_checkpoint_sha256": sha256_file(args.model_checkpoint),
        "scheduler_calibration_manifest_sha256": sha256_file(args.split_contract),
        "scheduler_metrics_sha256": sha256_file(args.scheduler_metrics),
        "scheduler_metrics": str(Path(args.scheduler_metrics).resolve()),
        "scheduler_summary": str(Path(args.scheduler_summary).resolve()),
        "scheduler_summary_sha256": sha256_file(args.scheduler_summary),
        "split_contract": str(Path(args.split_contract).resolve()),
        "defect_fit_sha256": sha256_file(args.defect_fit),
        "defect_fit": str(Path(args.defect_fit).resolve()),
        "defect_validity_sha256": sha256_file(args.defect_validity),
        "defect_validity": str(Path(args.defect_validity).resolve()),
        "split_contract_id": split_payload["defect_split_contract_id"],
        "target": {"K_q": 4.0, "minimum": 3.8, "maximum": 4.2},
        "selection_tie_break": [
            "lower first-R mean error",
            "lower first-R p95",
            "K_q_hat closer to 4",
            "fewer direct re-anchors",
        ],
        "selected": selection,
        "calibration": calibration.to_dict(),
        "episode_count": len(simulations),
        "producer": expected_producer,
    }
    payload["dynamic_threshold_lock_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
