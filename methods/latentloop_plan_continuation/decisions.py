"""Predeclared scientific decision logic for the plan-continuation claim."""

from __future__ import annotations

from typing import Any, Mapping


SUPPORTED = "PLAN_CONTINUATION_SUPPORTED"
INCONCLUSIVE = "PLAN_CONTINUATION_INCONCLUSIVE"
NOT_SUPPORTED = "PLAN_CONTINUATION_NOT_SUPPORTED"


def _bool(summary: Mapping[str, Any], key: str) -> bool:
    return bool(summary.get(key, False))


def apply_plan_continuation_decision(
    summary: Mapping[str, Any], rule: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen rule document without fitting thresholds to results."""

    required = tuple(rule.get("required_boolean_fields", ()))
    missing = [key for key in required if key not in summary]
    present = [key for key in required if key in summary]
    passed = [key for key in present if _bool(summary, key)]
    failed = [key for key in present if not _bool(summary, key)]
    explicit_not_supported = tuple(rule.get("not_supported_boolean_fields", ()))
    not_supported_reasons = [key for key in explicit_not_supported if _bool(summary, key)]
    if not_supported_reasons:
        verdict = NOT_SUPPORTED
        reason = "one or more predeclared refutation conditions passed"
    elif missing:
        verdict = INCONCLUSIVE
        reason = "required evidence is absent"
    elif not failed:
        verdict = SUPPORTED
        reason = "all predeclared support conditions passed"
    else:
        verdict = INCONCLUSIVE
        reason = "support conditions are incomplete or indistinguishable"
    return {
        "verdict": verdict,
        "passed": passed,
        "failed": failed,
        "missing": missing,
        "not_supported_reasons": not_supported_reasons,
        "reason": reason,
    }
