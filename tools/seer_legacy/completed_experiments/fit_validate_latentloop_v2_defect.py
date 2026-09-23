#!/usr/bin/env python3
"""Fit and validate the predeclared V2 defect signal on disjoint episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    positive = int(label.sum())
    negative = len(label) - positive
    if not positive or not negative:
        return float("nan")
    ranks = rankdata(score) + 1.0
    return float((ranks[label].sum() - positive * (positive + 1) / 2) / (positive * negative))


def logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mean, std = float(x.mean()), max(float(x.std()), 1e-8)
    normalized = (x - mean) / std
    weight = 0.0
    bias = math.log((float(y.mean()) + 1e-4) / (1.0001 - float(y.mean())))
    for _ in range(2000):
        logits = np.clip(weight * normalized + bias, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        weight -= 0.05 * float(np.mean((probability - y) * normalized))
        bias -= 0.05 * float(np.mean(probability - y))
    return weight / std, bias - weight * mean / std


def ece(probability: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if index == bins - 1:
            mask |= probability == 1.0
        if mask.any():
            total += float(mask.mean()) * abs(float(probability[mask].mean() - label[mask].mean()))
    return total


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def latent_difference(row: dict[str, object]) -> np.ndarray:
    sequential = np.asarray(row["z_seq"], dtype=np.float64)
    direct = np.asarray(row["z_dir"], dtype=np.float64)
    if sequential.shape != direct.shape or not sequential.size:
        raise RuntimeError("z_seq and z_dir must have the same nonempty shape")
    return sequential - direct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--high-error-quantile", type=float, default=0.90)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = load_rows(args.trace)
    split_document = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split_sha256 = hashlib.sha256(args.split_manifest.read_bytes()).hexdigest()
    source_lock_sha256 = hashlib.sha256(args.source_lock.read_bytes()).hexdigest()
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    expected_source = {str(row.get("source_lock_sha256", "")) for row in rows}
    expected_checkpoint = {str(row.get("v1_checkpoint_sha256", "")) for row in rows}
    expected_split = {str(row.get("split_manifest_sha256", "")) for row in rows}
    if expected_source != {source_lock_sha256} or expected_checkpoint != {checkpoint_sha256} or expected_split != {split_sha256}:
        raise RuntimeError("defect trace provenance does not match source/split/selected V1 checkpoint")
    normalization = [row for row in rows if row["split"] == "transition_train"]
    fit = [row for row in rows if row["split"] == "defect_fit"]
    validation = [row for row in rows if row["split"] == "defect_validation"]
    normalization_keys = {str(row["episode_key"]) for row in normalization}
    fit_keys = {str(row["episode_key"]) for row in fit}
    validation_keys = {str(row["episode_key"]) for row in validation}
    if not normalization or not fit or not validation:
        raise RuntimeError("normalization, defect fit, and defect validation splits must be nonempty")
    if normalization_keys & fit_keys or normalization_keys & validation_keys or fit_keys & validation_keys:
        raise RuntimeError("normalization, defect fit, and validation must be episode-disjoint")
    locked_splits = split_document["splits"]
    for role, keys in (
        ("transition_train", normalization_keys),
        ("defect_fit", fit_keys),
        ("defect_validation", validation_keys),
    ):
        allowed = set(locked_splits[role]["episode_keys"])
        if not keys <= allowed:
            raise RuntimeError(f"defect trace contains episode keys outside locked {role}")
    train_differences = np.stack([latent_difference(row).reshape(-1) for row in normalization])
    scale = np.sqrt(np.mean(np.square(train_differences), axis=0))
    scale = np.maximum(scale, 1e-6)
    for row in fit + validation:
        row["eta"] = float(np.sqrt(np.mean(np.square(latent_difference(row).reshape(-1) / scale))))
    required_errors = (
        "sequential_action_error", "direct_action_error",
        "sequential_arm_error", "direct_arm_error",
        "sequential_gripper_error", "direct_gripper_error",
    )
    if any(name not in row for row in fit + validation for name in required_errors):
        raise RuntimeError("trace is missing sequential/direct frozen-action-generator errors")
    fit_error = np.asarray([row["sequential_action_error"] for row in fit], dtype=np.float64)
    threshold = float(np.quantile(fit_error, args.high_error_quantile))
    fit_eta = np.asarray([row["eta"] for row in fit], dtype=np.float64)
    fit_label = fit_error >= threshold
    slope, intercept = logistic_fit(fit_eta, fit_label.astype(np.float64))

    values = {
        name: np.asarray([row[name] for row in validation], dtype=np.float64)
        for name in ("eta", "sequential_action_error", "direct_action_error", "sequential_arm_error", "direct_arm_error", "sequential_gripper_error", "direct_gripper_error", "age", "observation_change_norm", "previous_action_magnitude")
    }
    label = values["sequential_action_error"] >= threshold
    probability = 1.0 / (1.0 + np.exp(-np.clip(slope * values["eta"] + intercept, -30.0, 30.0)))
    quantiles = np.quantile(values["eta"], np.linspace(0.0, 1.0, 6))
    bin_errors = []
    for index in range(5):
        mask = (values["eta"] >= quantiles[index]) & (values["eta"] <= quantiles[index + 1] if index == 4 else values["eta"] < quantiles[index + 1])
        if mask.any():
            bin_errors.append(float(values["sequential_action_error"][mask].mean()))
    monotonic = all(left <= right for left, right in zip(bin_errors, bin_errors[1:]))
    aucs = {
        "defect": auroc(values["eta"], label),
        "age": auroc(values["age"], label),
        "observation_change_norm": auroc(values["observation_change_norm"], label),
        "previous_action_magnitude": auroc(values["previous_action_magnitude"], label),
    }
    rho = spearman(values["eta"], values["sequential_action_error"])
    passed = (
        monotonic
        and math.isfinite(rho)
        and rho >= 0.10
        and aucs["defect"] >= 0.70
        and all(aucs["defect"] > aucs[name] for name in aucs if name != "defect")
    )
    payload = {
        "schema_version": 1,
        "verdict": "DEFECT_SIGNAL_PASS" if passed else "DEFECT_SIGNAL_FAIL",
        "source_lock_sha256": source_lock_sha256,
        "v1_checkpoint_sha256": checkpoint_sha256,
        "split_manifest_sha256": split_sha256,
        "fit_episode_count": len(fit_keys),
        "normalization_episode_count": len(normalization_keys),
        "validation_episode_count": len(validation_keys),
        "fit_validation_overlap": sorted(fit_keys & validation_keys),
        "normalization_overlap": sorted((normalization_keys & fit_keys) | (normalization_keys & validation_keys)),
        "normalization": {
            "source_split": "transition_train",
            "epsilon": 1e-6,
            "coordinate_count": int(scale.size),
            "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
        },
        "high_error_quantile": args.high_error_quantile,
        "high_error_threshold_from_fit": threshold,
        "calibration": {"slope": slope, "intercept": intercept, "ece": ece(probability, label)},
        "defect_bin_action_errors": bin_errors,
        "monotonic_over_defect_bins": monotonic,
        "spearman": rho,
        "auroc": aucs,
        "validation_error_means": {
            name: float(values[name].mean())
            for name in required_errors
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["verdict"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
