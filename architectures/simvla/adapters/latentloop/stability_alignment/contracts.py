"""Pure fail-closed contracts for the two-server stability campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "simvla_condition_stability_alignment_v2"
STATE_SCHEMA = "simvla_stability_pipeline_state_v2"
LOSS_SCHEMA = "simvla_stability_loss_weights_v2"
CHECKPOINT_SCHEMA = "simvla_stability_alignment_checkpoint_v2"
BUNDLE_SCHEMA = "simvla_stability_alignment_bundle_v2"

GENERATION_NG3_FULL_INDICES = (0, 4, 8)
NAIVE_NFE3_TIMES = (1.0, 2.0 / 3.0, 1.0 / 3.0)
ACTION_HORIZON = 10
EXECUTION_HORIZON = 5
CONDITION_AGES = (1, 2, 3)
SCHEDULER_HORIZON = 30_000
SCHEDULER_WARMUP = 1_500
SCHEDULER_FINAL_RATIO = 0.1
TWO_K_CONDITION_ONLY_SAFETY_CHECKS = (
    "finite_losses",
    "frozen_base_gradients_zero",
    "age1_first_r_within_5pct",
    "exact_ng3_within_5pct",
    "no_gripper_collapse",
    "no_p99_explosion",
)
GRAD_CLIP_NORM = 1.0

CONTRIBUTION_TARGETS = {
    "recursive_reference": 0.30,
    "teacher_forced_preservation": 0.15,
    "recursive_stability": 0.20,
    "end_to_end_execution": 0.20,
    "gripper_transition": 0.08,
    "tail_cvar": 0.04,
    "rotating_full_nfe_execution": 0.02,
    "parent_preservation": 0.01,
}

SD1_STAGES = tuple(f"S{index}" for index in range(15))
RB2_STAGES = tuple(f"R{index}" for index in range(13))


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def kc_schedule(k_c: int) -> tuple[int, ...]:
    value = int(k_c)
    if value not in {1, 2, 3, 4, 8}:
        raise ValueError("K_C must be one of 1,2,3,4,8")
    return tuple(range(value))


def rotating_condition_age(optimizer_step: int) -> int:
    """Cycle deterministically through ages 1, 2, and 3."""

    if int(optimizer_step) < 0:
        raise ValueError("optimizer_step must be non-negative")
    return CONDITION_AGES[int(optimizer_step) % len(CONDITION_AGES)]


def naive_nfe3_contract(times: Sequence[float], dt: float) -> dict[str, Any]:
    observed = tuple(float(value) for value in times)
    checks = {
        "exactly_three_transformer_evaluations": len(observed) == 3,
        "source_native_times": len(observed) == 3
        and all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
            for a, b in zip(observed, NAIVE_NFE3_TIMES)
        ),
        "source_native_dt": math.isclose(float(dt), -1.0 / 3.0, abs_tol=1e-12),
    }
    return {
        "verdict": "NAIVE_NFE3_CONTRACT_PASS" if all(checks.values()) else "NAIVE_NFE3_CONTRACT_FAIL",
        "checks": checks,
    }


def free_gpu_pairs(
    gpu_pool: Sequence[int],
    busy_gpu_ids: Sequence[int],
    *,
    running_pairs: Sequence[Sequence[int]] = (),
    max_simultaneous_pairs: int = 2,
) -> tuple[tuple[int, int], ...]:
    """Form stable adjacent pairs without assigning a GPU twice."""

    if int(max_simultaneous_pairs) < 1:
        raise ValueError("max_simultaneous_pairs must be positive")
    raw_pool = tuple(int(value) for value in gpu_pool)
    pool = tuple(dict.fromkeys(raw_pool))
    if len(pool) != len(raw_pool):
        raise ValueError("GPU pool contains duplicates")
    unavailable = {int(value) for value in busy_gpu_ids}
    for pair in running_pairs:
        if len(tuple(pair)) != 2:
            raise ValueError("running GPU allocation is not a pair")
        unavailable.update(int(value) for value in pair)
    capacity = max(0, int(max_simultaneous_pairs) - len(tuple(running_pairs)))
    free = [value for value in pool if value not in unavailable]
    result: list[tuple[int, int]] = []
    for offset in range(0, len(free) - 1, 2):
        if len(result) >= capacity:
            break
        result.append((free[offset], free[offset + 1]))
    return tuple(result)


def gpu_is_free(
    *,
    memory_used_mib: int,
    utilization_percent: int,
    compute_pids: Sequence[int],
    memory_threshold_mib: int = 512,
) -> bool:
    """Conservatively classify a GPU; any compute PID makes it busy."""

    return (
        not tuple(compute_pids)
        and int(memory_used_mib) <= int(memory_threshold_mib)
        and int(utilization_percent) == 0
    )


def scheduler_multiplier(step: int) -> float:
    value = min(max(int(step), 0), SCHEDULER_HORIZON)
    if value <= SCHEDULER_WARMUP:
        return float(value) / float(SCHEDULER_WARMUP)
    progress = (value - SCHEDULER_WARMUP) / (
        SCHEDULER_HORIZON - SCHEDULER_WARMUP
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return SCHEDULER_FINAL_RATIO + (1.0 - SCHEDULER_FINAL_RATIO) * cosine


@dataclass(frozen=True)
class GateResult:
    verdict: str
    passed: bool
    checks: dict[str, bool]
    measurements: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_metrics(metrics: Mapping[str, Any]) -> bool:
    values = [
        float(value)
        for value in metrics.values()
        if isinstance(value, (int, float))
    ]
    return bool(values) and all(math.isfinite(value) for value in values)


def evaluate_2k_gate(metrics: Mapping[str, Any]) -> GateResult:
    checks = {
        "finite_losses": _finite_metrics(metrics),
        "frozen_base_gradients_zero": bool(metrics["frozen_base_gradients_zero"]),
        "age1_first_r_within_5pct": float(metrics["age1_first_r_ratio_to_parent"]) <= 1.05,
        "exact_ng3_within_5pct": float(metrics["exact_ng3_ratio_to_parent"]) <= 1.05,
        "no_gripper_collapse": bool(metrics["no_gripper_collapse"]),
        "no_p99_explosion": float(metrics["p99_ratio_to_parent"]) <= 1.25,
        "stability_loss_decreasing": float(metrics["stability_slope"]) < 0.0,
    }
    passed = all(checks.values())
    return GateResult(
        verdict="STABILITY_2K_GATE_PASS" if passed else "STABILITY_2K_GATE_FAIL",
        passed=passed,
        checks=checks,
        measurements=dict(metrics),
    )


def condition_only_2k_continuation(gate_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the predeclared fallback when only the 2K trend check fails."""
    verdict = str(gate_payload.get("verdict"))
    if verdict not in {"STABILITY_2K_GATE_PASS", "STABILITY_2K_GATE_FAIL"}:
        raise ValueError(f"unexpected 2K gate verdict: {verdict}")
    gate = gate_payload.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("2K gate payload lacks gate details")
    checks = gate.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("2K gate payload lacks checks")
    required = (*TWO_K_CONDITION_ONLY_SAFETY_CHECKS, "stability_loss_decreasing")
    missing = [name for name in required if name not in checks]
    if missing:
        raise ValueError(f"2K gate lacks checks: {missing}")
    safety_checks = {
        name: bool(checks[name]) for name in TWO_K_CONDITION_ONLY_SAFETY_CHECKS
    }
    safety_passed = all(safety_checks.values())
    trend_passed = bool(checks["stability_loss_decreasing"])
    return {
        "verdict": (
            "STABILITY_2K_CONTINUE"
            if safety_passed
            else "STABILITY_2K_STOP_UNSAFE"
        ),
        "passed": safety_passed,
        "strict_gate_passed": verdict == "STABILITY_2K_GATE_PASS",
        "stability_trend_passed": trend_passed,
        "condition_only_fallback": safety_passed and not trend_passed,
        "safety_checks": safety_checks,
    }


def evaluate_10k_gate(metrics: Mapping[str, Any]) -> GateResult:
    checks = {
        "age2_recurrence_mean_improved_20pct": float(metrics["age2_recurrence_improvement"]) >= 0.20,
        "age3_recurrence_mean_improved_30pct": float(metrics["age3_recurrence_improvement"]) >= 0.30,
        "age3_gripper_sign_improved_25pct": float(metrics["age3_gripper_sign_improvement"]) >= 0.25,
        "age3_first_r_p95_improved": float(metrics["age3_first_r_p95_ratio"]) < 1.0,
        "teacher_forced_within_5pct": float(metrics["teacher_forced_first_r_ratio"]) <= 1.05,
        "age1_final_system_within_5pct": float(metrics["age1_final_system_ratio"]) <= 1.05,
        "exact_ng3_within_5pct": float(metrics["exact_ng3_ratio_to_parent"]) <= 1.05,
        "no_p99_or_gripper_collapse": bool(metrics["no_p99_or_gripper_collapse"]),
        "original_simvla_frozen": bool(metrics["original_simvla_frozen"]),
    }
    passed = all(checks.values())
    return GateResult(
        verdict="STABILITY_10K_GATE_PASS" if passed else "STABILITY_10K_GATE_FAIL",
        passed=passed,
        checks=checks,
        measurements=dict(metrics),
    )


def select_condition_only_parent(
    s50: Mapping[str, Any], s150: Mapping[str, Any]
) -> dict[str, Any]:
    if not bool(s50.get("gate_passed")) and not bool(s150.get("gate_passed")):
        return {"verdict": "NO_SHORT_SPAN_BRANCH_SELECTED", "selected": None}
    if bool(s50.get("gate_passed")) and bool(s150.get("gate_passed")):
        checks = {
            "age2_mean_within_3pct": float(s50["age2_recursive_first_r_mean"]) <= 1.03 * float(s150["age2_recursive_first_r_mean"]),
            "age3_mean_within_3pct": float(s50["age3_recursive_first_r_mean"]) <= 1.03 * float(s150["age3_recursive_first_r_mean"]),
            "age3_p95_within_5pct": float(s50["age3_recursive_first_r_p95"]) <= 1.05 * float(s150["age3_recursive_first_r_p95"]),
            "gripper_tail_no_worse": float(s50["age3_gripper_tail"]) <= float(s150["age3_gripper_tail"]),
            "exact_ng3_no_worse": float(s50["exact_ng3_error"]) <= float(s150["exact_ng3_error"]),
            "parameter_count_identical": int(s50["parameter_count"]) == int(s150["parameter_count"]),
        }
        if all(checks.values()):
            return {"verdict": "S50_SELECTED", "selected": "S50", "checks": checks}
    return {"verdict": "S150_SELECTED", "selected": "S150"}


def k_offline_readiness(k_c: int, age_pass: Mapping[int, bool]) -> dict[str, Any]:
    value = int(k_c)
    required = {3: (1, 2), 4: (1, 2, 3)}.get(value)
    if required is None:
        raise ValueError("offline short-span readiness is defined for K_C=3 or 4")
    checks = {f"age{age}": bool(age_pass.get(age, False)) for age in required}
    passed = all(checks.values())
    return {
        "verdict": f"KC{value}_OFFLINE_READY" if passed else f"KC{value}_OFFLINE_BLOCKED",
        "passed": passed,
        "checks": checks,
    }


class AtomicStageState:
    """Atomic stage ledger used by both launchers."""

    def __init__(self, path: str | Path, stages: Sequence[str]) -> None:
        self.path = Path(path).expanduser().resolve()
        self.stages = tuple(stages)
        if self.path.exists():
            payload = load_json(self.path)
            if payload.get("schema_version") != STATE_SCHEMA:
                raise ValueError("pipeline state schema changed")
            if tuple(payload.get("stage_order", ())) != self.stages:
                raise ValueError("pipeline stage graph changed")
            self.payload = payload
        else:
            self.payload = {
                "schema_version": STATE_SCHEMA,
                "stage_order": list(self.stages),
                "stages": {stage: {"state": "PENDING"} for stage in self.stages},
            }
            atomic_write_json(self.path, self.payload)

    def set(self, stage: str, state: str, **metadata: Any) -> None:
        if stage not in self.stages:
            raise KeyError(stage)
        allowed = {"PENDING", "RUNNING", "PASSED", "FAILED", "BLOCKED", "SKIPPED"}
        if state not in allowed:
            raise ValueError(f"invalid stage state: {state}")
        # The pipeline lock prevents concurrent owners.  A second RUNNING write
        # therefore means the previous owner exited and this invocation is
        # resuming the same stage, not that two launchers overlap.
        self.payload["stages"][stage] = {"state": state, **metadata}
        atomic_write_json(self.path, self.payload)

    def require_passed(self, *stages: str) -> None:
        failed = [
            stage
            for stage in stages
            if self.payload["stages"].get(stage, {}).get("state") != "PASSED"
        ]
        if failed:
            raise RuntimeError(f"prerequisite stages not passed: {failed}")
