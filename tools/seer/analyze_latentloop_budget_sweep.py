#!/usr/bin/env python3
"""Select the smallest canonical LatentLoop post-training budget."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def metric(row: dict[str, Any], name: str) -> float:
    return float(row["metrics"][name]["mean"])


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not rows:
        raise ValueError("At least one validation result is required")
    for row, path in zip(rows, paths):
        if row.get("protocol") != "latentloop_posttrain_budget_validation_v1":
            raise ValueError(f"Unexpected validation protocol: {path}")
        row["validation_result_path"] = str(path.resolve())
    teacher_hashes = {row["teacher_sha256"] for row in rows}
    split_hashes = {row["validation_split_sha256"] for row in rows}
    vit_hashes = {row["vit_checkpoint_sha256"] for row in rows}
    if len(teacher_hashes) != 1 or len(split_hashes) != 1 or len(vit_hashes) != 1:
        raise RuntimeError(
            "Budget results do not share one teacher, validation split, and ViT checkpoint"
        )
    return sorted(rows, key=lambda row: (row["lane"], int(row["budget_epochs"])))


def select(rows: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    controls = [
        row
        for row in rows
        if row["lane"] == "standard_40" and int(row["budget_epochs"]) == 40
    ]
    if len(controls) != 1:
        raise RuntimeError(f"Expected exactly one standard 40-epoch control, got {len(controls)}")
    control = controls[0]
    candidates = sorted(
        (row for row in rows if row["lane"] == "compressed"),
        key=lambda row: int(row["budget_epochs"]),
    )
    if not candidates:
        raise RuntimeError("No compressed-budget candidates were supplied")

    control_latent = metric(control, "latent_mse")
    control_action = metric(control, "action_l1")
    eligible = []
    table = []
    for row in candidates:
        latent = metric(row, "latent_mse")
        action = metric(row, "action_l1")
        within = (
            bool(row["gates"]["pass"])
            and latent <= control_latent * (1.0 + tolerance)
            and action <= control_action * (1.0 + tolerance)
        )
        if within:
            eligible.append(row)
        table.append(
            {
                "lane": row["lane"],
                "budget_epochs": int(row["budget_epochs"]),
                "compute_ratio_vs_40": int(row["budget_epochs"]) / 40.0,
                "compute_reduction_pct_vs_40": 100.0
                * (1.0 - int(row["budget_epochs"]) / 40.0),
                "latent_mse": latent,
                "action_l1": action,
                "smooth_mse": metric(row, "smooth_mse"),
                "hold_latent_mse": metric(row, "hold_latent_mse"),
                "hold_action_l1": metric(row, "hold_action_l1"),
                "weighted_loss": float(row["selection_metric_value"]),
                "gate_pass": bool(row["gates"]["pass"]),
                "within_control_tolerance": within,
                "adapter": row["adapter"],
                "validation_result": row["validation_result_path"],
            }
        )
    table.append(
        {
            "lane": control["lane"],
            "budget_epochs": 40,
            "compute_ratio_vs_40": 1.0,
            "compute_reduction_pct_vs_40": 0.0,
            "latent_mse": control_latent,
            "action_l1": control_action,
            "smooth_mse": metric(control, "smooth_mse"),
            "hold_latent_mse": metric(control, "hold_latent_mse"),
            "hold_action_l1": metric(control, "hold_action_l1"),
            "weighted_loss": float(control["selection_metric_value"]),
            "gate_pass": bool(control["gates"]["pass"]),
            "within_control_tolerance": True,
            "adapter": control["adapter"],
            "validation_result": control["validation_result_path"],
        }
    )

    if eligible:
        selected = eligible[0]
        status = "MINIMUM_BUDGET_WITHIN_TOLERANCE"
    else:
        passing = [row for row in candidates if bool(row["gates"]["pass"])]
        pool = passing or candidates
        selected = min(pool, key=lambda row: float(row["selection_metric_value"]))
        status = "FALLBACK_BEST_COMPRESSED_NO_TOLERANCE_MATCH"

    return {
        "schema_version": 1,
        "protocol": "latentloop_posttrain_budget_selection_v1",
        "status": status,
        "relative_tolerance": tolerance,
        "selection_rule": (
            "Smallest compressed budget with finite/hold-improvement gates and both "
            "latent MSE and action L1 no more than (1+tolerance) times the independent "
            "standard 40-epoch control; otherwise lowest weighted compressed loss."
        ),
        "teacher": control["teacher"],
        "teacher_sha256": control["teacher_sha256"],
        "validation_split_sha256": control["validation_split_sha256"],
        "selected_budget_epochs": int(selected["budget_epochs"]),
        "selected_adapter": selected["adapter"],
        "selected_adapter_sha256": selected["adapter_sha256"],
        "control_budget_epochs": 40,
        "control_adapter": control["adapter"],
        "control_adapter_sha256": control["adapter_sha256"],
        "rows": table,
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "budget_selection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    columns = list(result["rows"][0])
    with (output_dir / "budget_validation_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result["rows"])

    lines = [
        "# LatentLoop Post-training Budget Selection",
        "",
        f"- Status: `{result['status']}`",
        f"- Selected budget: **{result['selected_budget_epochs']} epochs**",
        f"- Selected adapter: `{result['selected_adapter']}`",
        f"- Standard control: **40 epochs**, `{result['control_adapter']}`",
        f"- Relative tolerance: **{100.0 * result['relative_tolerance']:.1f}%**",
        "",
        "| Lane | Epochs | Compute vs. 40 | Latent MSE | Action L1 | Smooth MSE | Gate | Within tolerance |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in sorted(result["rows"], key=lambda item: int(item["budget_epochs"])):
        lines.append(
            "| {lane} | {budget_epochs} | {ratio:.1%} | {latent:.6g} | "
            "{action:.6g} | {smooth:.6g} | {gate} | {within} |".format(
                lane=row["lane"],
                budget_epochs=row["budget_epochs"],
                ratio=row["compute_ratio_vs_40"],
                latent=row["latent_mse"],
                action=row["action_l1"],
                smooth=row["smooth_mse"],
                gate="PASS" if row["gate_pass"] else "FAIL",
                within="yes" if row["within_control_tolerance"] else "no",
            )
        )
    lines += [
        "",
        "The epoch ratio is a training-update budget proxy. Measured wall time is stored "
        "inside each validation JSON and must be reported separately from this ratio.",
    ]
    (output_dir / "budget_selection_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    if not 0.0 <= args.relative_tolerance < 1.0:
        parser.error("--relative-tolerance must be in [0,1)")
    result = select(load_results(args.result), args.relative_tolerance)
    if not all(math.isfinite(float(row["weighted_loss"])) for row in result["rows"]):
        raise RuntimeError("Non-finite weighted loss in budget table")
    write_outputs(result, args.output_dir)


if __name__ == "__main__":
    main()
