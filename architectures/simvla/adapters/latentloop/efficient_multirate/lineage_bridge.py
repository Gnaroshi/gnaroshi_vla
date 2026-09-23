"""Explicit child-source bridge for unchanged native V0 evaluators."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


INHERITABLE_PARENT_VERDICTS = {
    "K1_HOOK_PARITY_PASS",
    "PARAMETER_AUDIT_PASS",
}


def load_child_source_lock(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not payload.get("combined_sha256") or not payload.get(
        "parent_source_combined_sha256"
    ):
        raise ValueError("efficient child source lock is incomplete")
    return payload


def lineage_require_gate(
    path: str | Path,
    *,
    verdicts: Iterable[str],
    source_combined_sha256: str,
    child_source: dict[str, Any],
) -> dict[str, Any]:
    if source_combined_sha256 != child_source["combined_sha256"]:
        raise RuntimeError("evaluator requested a source outside the efficient child lock")
    gate_path = Path(path).expanduser().resolve()
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    allowed_verdicts = {str(value) for value in verdicts}
    observed_verdict = str(payload.get("verdict"))
    if observed_verdict not in allowed_verdicts:
        raise RuntimeError(
            f"gate {gate_path} verdict={observed_verdict!r}, expected {sorted(allowed_verdicts)}"
        )
    observed_source = payload.get("source_combined_sha256")
    if observed_source == child_source["combined_sha256"]:
        return payload
    inherited = (
        observed_source == child_source["parent_source_combined_sha256"]
        and observed_verdict in INHERITABLE_PARENT_VERDICTS
    )
    if not inherited:
        raise RuntimeError(f"gate {gate_path} is outside the child/parent source lineage")
    return payload


def install_native_evaluator_lineage_bridge(
    module: ModuleType,
    child_source: dict[str, Any],
) -> None:
    """Replace only source/gate lookup globals in an unchanged evaluator module."""

    module.native_v0_source_manifest = lambda **_kwargs: child_source

    def require_gate(
        path: str | Path,
        *,
        verdicts: Iterable[str],
        source_combined_sha256: str,
    ) -> dict[str, Any]:
        return lineage_require_gate(
            path,
            verdicts=verdicts,
            source_combined_sha256=source_combined_sha256,
            child_source=child_source,
        )

    module.require_gate = require_gate

