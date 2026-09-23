"""Validation-only e20/e40 LatentLoop V1 budget selection."""

from __future__ import annotations

import math
from collections.abc import Mapping


METRICS = (
    "direct_latent_mse",
    "composed_latent_mse",
    "direct_action_l1",
    "composed_action_l1",
    "composition_defect",
)
FINITE_METRICS = METRICS + ("hold_latent_mse", "hold_action_l1")


def _finite(row: Mapping[str, object]) -> bool:
    return all(math.isfinite(float(row[name])) for name in FINITE_METRICS)


def _validate_provenance(e20: Mapping[str, object], e40: Mapping[str, object]) -> tuple[str, str]:
    for name, epochs, row in (("e20", 20, e20), ("e40", 40, e40)):
        if row.get("split_role") != "checkpoint_validation":
            raise ValueError(f"{name} metrics are not from checkpoint validation")
        if bool(row.get("uses_online_sr", True)):
            raise ValueError(f"{name} metrics use online SR")
        expected = (epochs, 42, 4, 512, epochs, 0.05)
        actual = (
            int(row.get("budget_epochs", -1)),
            int(row.get("training_seed", -1)),
            int(row.get("world_size", -1)),
            int(row.get("effective_batch", -1)),
            int(row.get("cosine_horizon_epochs", -1)),
            float(row.get("warmup_fraction", -1.0)),
        )
        if actual != expected:
            raise ValueError(f"{name} training-budget provenance mismatch: {actual}")
    split_hashes = {str(row.get("split_manifest_sha256", "")) for row in (e20, e40)}
    source_hashes = {str(row.get("source_lock_sha256", "")) for row in (e20, e40)}
    if len(split_hashes) != 1 or len(next(iter(split_hashes))) != 64:
        raise ValueError("e20/e40 split-manifest provenance differs or is missing")
    if len(source_hashes) != 1 or len(next(iter(source_hashes))) != 64:
        raise ValueError("e20/e40 source-lock provenance differs or is missing")
    initialization_hashes = {str(row.get("initialization_sha256", "")) for row in (e20, e40)}
    run_ids = {str(row.get("independent_run_id", "")) for row in (e20, e40)}
    if len(initialization_hashes) != 1 or len(next(iter(initialization_hashes))) != 64:
        raise ValueError("e20/e40 initialization provenance differs or is missing")
    if len(run_ids) != 2 or "" in run_ids:
        raise ValueError("e20/e40 must be independent run identities")
    return next(iter(split_hashes)), next(iter(source_hashes))


def select_v1_budget(e20: Mapping[str, object], e40: Mapping[str, object]) -> dict[str, object]:
    split_hash, source_hash = _validate_provenance(e20, e40)
    if not _finite(e40) or bool(e40.get("gripper_collapse", True)):
        raise ValueError("e40 control is not a valid fallback candidate")
    checks = {
        "finite": _finite(e20) and _finite(e40),
        "direct_beats_hold_latent": float(e20["direct_latent_mse"]) < float(e20["hold_latent_mse"]),
        "composed_beats_hold_latent": float(e20["composed_latent_mse"]) < float(e20["hold_latent_mse"]),
        "direct_beats_hold_action": float(e20["direct_action_l1"]) < float(e20["hold_action_l1"]),
        "composed_beats_hold_action": float(e20["composed_action_l1"]) < float(e20["hold_action_l1"]),
        "no_gripper_collapse": not bool(e20.get("gripper_collapse", True)),
    }
    for metric in METRICS:
        checks[f"{metric}_within_1p05"] = float(e20[metric]) <= 1.05 * float(e40[metric])
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "selection_uses_online_sr": False,
        "split_manifest_sha256": split_hash,
        "source_lock_sha256": source_hash,
        "selected_budget_epochs": 20 if passed else 40,
        "verdict": "V1_E20_SELECTED" if passed else "V1_E40_SELECTED",
        "checks": checks,
    }
