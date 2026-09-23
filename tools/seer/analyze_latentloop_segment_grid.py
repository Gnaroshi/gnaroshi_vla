#!/usr/bin/env python3
"""Aggregate LatentLoop segment-length and feedback-density campaigns offline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from methods.latentloop_segment_grid.metrics import (
    distribution_profile,
    episode_key,
    paired_flip_counts,
    paired_hierarchical_bootstrap_interval,
    wilson_interval,
)
from methods.latentloop_segment_grid.serialization import atomic_write_json, read_json


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _mean(values: Iterable[object]) -> Optional[float]:
    present = [_float(value) for value in values if value not in (None, "")]
    return float(np.mean(present)) if present else None


def _discover_rows(input_roots: Sequence[Path]) -> List[Dict[str, object]]:
    records = []
    seen_paths = set()
    for root in input_roots:
        for path in sorted(root.resolve().rglob("segment_grid_row.json")):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            record = read_json(path)
            record["row_registry_path"] = str(path)
            records.append(record)
    if not records:
        raise FileNotFoundError("No segment_grid_row.json files were discovered")
    semantic_keys = defaultdict(list)
    for record in records:
        key = (
            str(record.get("checkpoint_profile", "legacy_unspecified")),
            str(record.get("baseline_checkpoint_sha256", "")),
            str(record.get("adapter_checkpoint_sha256", "")),
            int(record["checkpoint_id"]),
            int(record["segment_length"]),
            str(record["feedback_schedule"]),
            str(record["baseline_kind"]),
        )
        semantic_keys[key].append(record["row_registry_path"])
    duplicates = {key: paths for key, paths in semantic_keys.items() if len(paths) > 1}
    if duplicates:
        raise RuntimeError(
            "Duplicate semantic rows were provided. Aggregate one predeclared row "
            f"per condition: {duplicates}"
        )
    return records


def _load_step_rows(step_log_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not step_log_dir.is_dir():
        return rows
    for path in sorted(step_log_dir.glob("*.csv")):
        rows.extend(_read_csv(path))
    return rows


def _profile_selected(
    rows: Sequence[Mapping[str, str]],
    key: str,
    *,
    predicate=None,
) -> Dict[str, float]:
    values = []
    for row in rows:
        if predicate is not None and not predicate(row):
            continue
        if row.get(key, "") in ("", None):
            continue
        values.append(float(row[key]))
    return distribution_profile(values)


def _latency_row(record: Mapping[str, object], step_rows: Sequence[Mapping[str, str]]) -> Dict[str, object]:
    components = {
        "full_seer": _profile_selected(
            step_rows,
            "full_forward_ms",
            predicate=lambda row: _int(row.get("full_forward_called")) == 1,
        ),
        "full_action_head": _profile_selected(
            step_rows,
            "full_action_head_ms",
            predicate=lambda row: _int(row.get("full_forward_called")) == 1,
        ),
        "full_non_action_head": _profile_selected(
            step_rows,
            "full_non_action_head_ms",
            predicate=lambda row: _int(row.get("full_forward_called")) == 1,
        ),
        "observation_encoder": _profile_selected(
            step_rows,
            "fast_encoder_ms",
            predicate=lambda row: _int(row.get("fast_encoder_called")) == 1,
        ),
        "latent_updater": _profile_selected(
            step_rows,
            "node_update_ms",
            predicate=lambda row: _int(row.get("lrnode_update_called")) == 1,
        ),
        "skip_action_head": _profile_selected(
            step_rows,
            "action_head_ms",
            predicate=lambda row: _int(row.get("action_head_called")) == 1,
        ),
        "policy_total": _profile_selected(step_rows, "total_policy_ms"),
        "simulator_env_step": _profile_selected(step_rows, "env_step_ms"),
    }
    row: Dict[str, object] = {
        "row_id": record["row_id"],
        "checkpoint_profile": record.get("checkpoint_profile", "legacy_unspecified"),
        "baseline_checkpoint_sha256": record.get("baseline_checkpoint_sha256", ""),
        "adapter_checkpoint_sha256": record.get("adapter_checkpoint_sha256", ""),
        "checkpoint_id": int(record["checkpoint_id"]),
        "segment_length": int(record["segment_length"]),
        "feedback_schedule": record["feedback_schedule"],
        "baseline_kind": record["baseline_kind"],
        "latency_source": "per_step_log",
    }
    for component, profile in components.items():
        for statistic, value in profile.items():
            row[f"{component}_{statistic}_ms" if statistic != "count" else f"{component}_count"] = value
    return row


def _aggregate_offset_metrics(episodes: Sequence[Mapping[str, str]]) -> Dict[str, object]:
    weighted: Dict[str, Dict[str, List[Tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for episode in episodes:
        raw = episode.get("segment_offset_metrics_json", "")
        if not raw:
            continue
        for offset, metrics in json.loads(raw).items():
            count = int(metrics.get("count", 0))
            for key, value in metrics.items():
                if key == "count" or value is None:
                    continue
                weighted[offset][key].append((float(value), count))
            weighted[offset]["count"].append((float(count), 1))
    output: Dict[str, object] = {}
    for offset, metrics in weighted.items():
        output[offset] = {}
        for key, values in metrics.items():
            if key == "count":
                output[offset][key] = int(sum(value for value, _ in values))
                continue
            denominator = sum(weight for _, weight in values)
            output[offset][key] = (
                sum(value * weight for value, weight in values) / denominator
                if denominator else None
            )
    return output


def _row_summary(record: Mapping[str, object], episodes: Sequence[Mapping[str, str]], summary: Mapping[str, object]) -> Dict[str, object]:
    successes = sum(_int(row.get("success")) for row in episodes)
    total = len(episodes)
    ci_low, ci_high = wilson_interval(successes, total)
    query = summary.get("query_reduction", {})
    grid = summary.get("lrnode", {}).get("segment_grid", {})
    full_seer_calls = _int(
        summary.get("num_full_forward_calls"),
        default=_int(query.get("num_full_forward_calls")),
    )
    policy_steps = _int(
        summary.get("num_env_steps"),
        default=_int(query.get("num_env_steps")),
    )
    full_seer_call_ratio = (
        float(full_seer_calls) / float(policy_steps) if policy_steps else 0.0
    )
    actual_density = grid.get("actual_feedback_density")
    row = {
        "row_id": record["row_id"],
        "stage": record["stage"],
        "checkpoint_profile": record.get("checkpoint_profile", "legacy_unspecified"),
        "checkpoint_source": record.get("checkpoint_source", "legacy_unspecified"),
        "baseline_checkpoint_path": record.get("baseline_checkpoint_path", ""),
        "baseline_checkpoint_sha256": record.get("baseline_checkpoint_sha256", ""),
        "adapter_checkpoint_path": record.get("adapter_checkpoint_path", ""),
        "adapter_checkpoint_sha256": record.get("adapter_checkpoint_sha256", ""),
        "checkpoint_id": int(record["checkpoint_id"]),
        "adapter_id": int(record["adapter_id"]),
        "segment_length": int(record["segment_length"]),
        "feedback_schedule": record["feedback_schedule"],
        "planned_feedback_density": record.get("planned_feedback_density"),
        "actual_feedback_density": actual_density,
        "baseline_kind": record["baseline_kind"],
        "ablation_mode": record["ablation_mode"],
        "successes": successes,
        "episodes": total,
        "success_rate": successes / total if total else 0.0,
        "wilson_95_low": ci_low,
        "wilson_95_high": ci_high,
        "full_seer_calls": full_seer_calls,
        "latentloop_updater_calls": _int(query.get("num_lrnode_update_calls")),
        "observation_conditioned_updater_calls": _int(
            grid.get("observation_conditioned_updater_calls")
        ),
        "zero_feature_updater_calls": _int(grid.get("zero_feature_updater_calls")),
        "full_query_reduction_ratio": _float(
            summary.get("full_query_reduction_ratio"),
            default=1.0 - full_seer_call_ratio,
        ),
        "full_seer_call_ratio": full_seer_call_ratio,
        "avg_policy_step_latency_ms": _float(summary.get("avg_policy_step_latency_ms")),
        "renderer_backend": summary.get("renderer_backend", {}).get("effective_backend"),
        "renderer_all_ranks_verified": bool(
            summary.get("renderer_backend", {}).get("all_ranks_actual_context_verified", False)
        ),
        "summary_path": record["summary_path"],
    }
    return row


def _per_task_rows(record: Mapping[str, object], episodes: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for episode in episodes:
        grouped[(_int(episode["task_id"]), episode.get("task_name", ""))].append(
            _int(episode["success"])
        )
    return [
        {
            "row_id": record["row_id"],
            "checkpoint_profile": record.get("checkpoint_profile", "legacy_unspecified"),
            "checkpoint_id": int(record["checkpoint_id"]),
            "segment_length": int(record["segment_length"]),
            "feedback_schedule": record["feedback_schedule"],
            "baseline_kind": record["baseline_kind"],
            "task_id": task_id,
            "task_name": task_name,
            "successes": sum(values),
            "episodes": len(values),
            "success_rate": float(np.mean(values)),
        }
        for (task_id, task_name), values in sorted(grouped.items())
    ]


def _continuity_row(record: Mapping[str, object], episodes: Sequence[Mapping[str, str]], success_rate: float) -> Dict[str, object]:
    fields = (
        "translation_jerk_normalized_mean",
        "translation_jerk_normalized_p95",
        "rotation_jerk_normalized_mean",
        "rotation_jerk_normalized_p95",
        "gripper_switches_per_100_steps",
        "gripper_reverse_within_1_per_100_steps",
        "gripper_reverse_within_2_per_100_steps",
        "gripper_reverse_within_5_per_100_steps",
    )
    row = {
        "row_id": record["row_id"],
        "checkpoint_profile": record.get("checkpoint_profile", "legacy_unspecified"),
        "checkpoint_id": int(record["checkpoint_id"]),
        "segment_length": int(record["segment_length"]),
        "feedback_schedule": record["feedback_schedule"],
        "baseline_kind": record["baseline_kind"],
        "success_rate": success_rate,
    }
    row.update({field: _mean(episode.get(field) for episode in episodes) for field in fields})
    return row


def _paired_rows(
    records: Sequence[Mapping[str, object]],
    episode_rows: Mapping[str, Sequence[Mapping[str, str]]],
    bootstrap_iterations: int,
) -> List[Dict[str, object]]:
    def checkpoint_key(record: Mapping[str, object]) -> Tuple[str, str, str, int]:
        return (
            str(record.get("checkpoint_profile", "legacy_unspecified")),
            str(record.get("baseline_checkpoint_sha256", "")),
            str(record.get("adapter_checkpoint_sha256", "")),
            int(record["checkpoint_id"]),
        )

    baseline_by_checkpoint = {
        checkpoint_key(record): record
        for record in records
        if record["baseline_kind"] == "full_replanning"
        and int(record["segment_length"]) == 1
    }
    outputs = []
    for record in records:
        checkpoint = int(record["checkpoint_id"])
        baseline_record = baseline_by_checkpoint.get(checkpoint_key(record))
        if baseline_record is None or record["row_id"] == baseline_record["row_id"]:
            continue
        base_rows = episode_rows[baseline_record["row_id"]]
        candidate_rows = episode_rows[record["row_id"]]
        baseline_outcomes = {episode_key(row): _int(row["success"]) for row in base_rows}
        candidate_outcomes = {episode_key(row): _int(row["success"]) for row in candidate_rows}
        flips = paired_flip_counts(baseline_outcomes, candidate_outcomes)
        if set(baseline_outcomes) != set(candidate_outcomes):
            raise RuntimeError(
                "Paired episode keys differ between rows: "
                f"baseline={baseline_record['row_id']}, candidate={record['row_id']}, "
                f"missing_from_candidate={flips['missing_from_candidate']}, "
                f"missing_from_baseline={flips['missing_from_baseline']}"
            )
        candidate_by_key = {episode_key(row): row for row in candidate_rows}
        pairs = [
            {
                "task_id": row["task_id"],
                "baseline_success": _int(row["success"]),
                "candidate_success": _int(candidate_by_key[key]["success"]),
            }
            for row in base_rows
            for key in [episode_key(row)]
            if key in candidate_by_key
        ]
        bootstrap = paired_hierarchical_bootstrap_interval(
            pairs,
            iterations=bootstrap_iterations,
            seed=20260729 + checkpoint + int(record["segment_length"]),
        )
        outputs.append(
            {
                "checkpoint_profile": record.get(
                    "checkpoint_profile", "legacy_unspecified"
                ),
                "checkpoint_id": checkpoint,
                "baseline_row_id": baseline_record["row_id"],
                "candidate_row_id": record["row_id"],
                "segment_length": int(record["segment_length"]),
                "feedback_schedule": record["feedback_schedule"],
                "baseline_kind": record["baseline_kind"],
                **flips,
                "paired_sr_difference": bootstrap["mean_difference"],
                "paired_bootstrap_95_low": bootstrap["ci_low"],
                "paired_bootstrap_95_high": bootstrap["ci_high"],
                "bootstrap_iterations": bootstrap["iterations"],
            }
        )
    return outputs


def _load_parity(
    input_roots: Sequence[Path],
    parity_artifacts: Sequence[Path] = (),
) -> Dict[str, object]:
    parity: Dict[str, object] = {}
    paths = [
        path
        for root in input_roots
        for path in root.resolve().rglob("k1_parity.json")
    ]
    paths.extend(path.resolve() for path in parity_artifacts)
    for path in paths:
        item = read_json(path)
        key = str(int(item["checkpoint_id"]))
        if key in parity:
            raise RuntimeError(f"Duplicate K1 parity artifact for checkpoint {key}")
        item["path"] = str(path)
        parity[key] = item
    return parity


def _plot_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    per_task: Sequence[Mapping[str, object]],
    continuity: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dense = [
        row for row in rows
        if row["baseline_kind"] in {"full_replanning", "dense_latentloop"}
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for checkpoint in sorted({int(row["checkpoint_id"]) for row in dense}):
        points = sorted(
            [row for row in dense if int(row["checkpoint_id"]) == checkpoint],
            key=lambda row: int(row["segment_length"]),
        )
        ax.plot(
            [row["segment_length"] for row in points],
            [100.0 * float(row["success_rate"]) for row in points],
            marker="o",
            label=f"Seer ckpt {checkpoint}",
        )
        for row in points:
            ax.annotate(
                f"-{100.0 * float(row['full_query_reduction_ratio']):.0f}%",
                (row["segment_length"], 100.0 * float(row["success_rate"])),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    ax.set(xlabel="Segment length L", ylabel="Success rate (%)", title="Dense-feedback segment-length curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "segment_length_curve.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for row in rows:
        if row["baseline_kind"] == "adapter_k1_parity":
            continue
        ax.scatter(
            float(row["full_seer_call_ratio"]),
            100.0 * float(row["success_rate"]),
            label=f"{row['row_id']}",
            s=28,
        )
    ax.set(xlabel="Full-Seer calls / policy step", ylabel="Success rate (%)", title="Performance-compute frontier")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "performance_compute_pareto.png", dpi=220)
    plt.close(fig)

    schedules = ["dense", "alternate", "none"]
    lengths = sorted(
        {int(row["segment_length"]) for row in rows if row["feedback_schedule"] in schedules}
    )
    values = np.full((len(lengths), len(schedules)), np.nan)
    for i, length in enumerate(lengths):
        for j, schedule in enumerate(schedules):
            matching = [
                float(row["success_rate"])
                for row in rows
                if int(row["segment_length"]) == length
                and row["feedback_schedule"] == schedule
                and row["baseline_kind"] in {
                    "dense_latentloop",
                    "alternate_latentloop",
                    "no_observation_latent_dynamics",
                }
            ]
            if matching:
                values[i, j] = 100.0 * float(np.mean(matching))
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(schedules)), schedules)
    ax.set_yticks(range(len(lengths)), lengths)
    ax.set(xlabel="Feedback schedule", ylabel="Segment length L", title="Commitment-feedback grid")
    for i in range(len(lengths)):
        for j in range(len(schedules)):
            if np.isfinite(values[i, j]):
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="SR (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "commitment_feedback_heatmap.png", dpi=220)
    plt.close(fig)

    benefits = []
    for checkpoint in sorted({int(row["checkpoint_id"]) for row in rows}):
        for length in lengths:
            dense_row = next(
                (row for row in rows if int(row["checkpoint_id"]) == checkpoint and int(row["segment_length"]) == length and row["feedback_schedule"] == "dense"),
                None,
            )
            none_row = next(
                (row for row in rows if int(row["checkpoint_id"]) == checkpoint and int(row["segment_length"]) == length and row["feedback_schedule"] == "none"),
                None,
            )
            if dense_row and none_row:
                benefits.append((checkpoint, length, 100.0 * (float(dense_row["success_rate"]) - float(none_row["success_rate"]))))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for checkpoint in sorted({item[0] for item in benefits}):
        points = [item for item in benefits if item[0] == checkpoint]
        ax.plot([item[1] for item in points], [item[2] for item in points], marker="o", label=f"ckpt {checkpoint}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(xlabel="Segment length L", ylabel="Dense minus none SR (pp)", title="Current-observation feedback benefit")
    ax.grid(alpha=0.25)
    if benefits:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "feedback_benefit.png", dpi=220)
    plt.close(fig)

    baseline_task = {
        (int(row["checkpoint_id"]), int(row["task_id"])): float(row["success_rate"])
        for row in per_task
        if row["baseline_kind"] == "full_replanning"
    }
    candidate_rows = [row for row in per_task if row["baseline_kind"] != "full_replanning"]
    row_ids = sorted({str(row["row_id"]) for row in candidate_rows})
    task_ids = sorted({int(row["task_id"]) for row in candidate_rows})
    task_matrix = np.full((len(row_ids), len(task_ids)), np.nan)
    for i, row_id in enumerate(row_ids):
        for row in candidate_rows:
            if row["row_id"] != row_id:
                continue
            base = baseline_task.get((int(row["checkpoint_id"]), int(row["task_id"])))
            if base is not None:
                task_matrix[i, task_ids.index(int(row["task_id"]))] = 100.0 * (float(row["success_rate"]) - base)
    fig, ax = plt.subplots(figsize=(9.0, max(3.2, 0.28 * len(row_ids))))
    image = ax.imshow(task_matrix, aspect="auto", cmap="coolwarm", vmin=-100, vmax=100)
    ax.set_xticks(range(len(task_ids)), task_ids)
    ax.set_yticks(range(len(row_ids)), row_ids, fontsize=6)
    ax.set(xlabel="Task id", ylabel="Condition", title="Per-task SR difference from L=1 (pp)")
    fig.colorbar(image, ax=ax, label="SR difference (pp)")
    fig.tight_layout()
    fig.savefig(output_dir / "per_task_delta_heatmap.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharey=True)
    for row in continuity:
        axes[0].scatter(
            _float(row.get("translation_jerk_normalized_mean")),
            100.0 * float(row["success_rate"]),
            s=28,
        )
        axes[1].scatter(
            _float(row.get("rotation_jerk_normalized_mean")),
            100.0 * float(row["success_rate"]),
            s=28,
        )
    axes[0].set(
        xlabel="Normalized translation second difference",
        ylabel="Success rate (%)",
        title="Translation continuity",
    )
    axes[1].set(
        xlabel="Normalized rotation second difference",
        title="Rotation continuity",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Continuity-performance trade-off")
    fig.tight_layout()
    fig.savefig(output_dir / "continuity_tradeoff.png", dpi=220)
    plt.close(fig)


def _write_report(output_dir: Path, summary: Mapping[str, object]) -> None:
    rows = summary["rows"]
    lines = [
        "# LatentLoop segment-length / feedback-density report",
        "",
        "## Protocol",
        "",
        "- Observation: `o_t = (I_t^p, I_t^w, q_t)`",
        "- Context: `C_t = (o_{t-H+1}, ..., o_t, ell)`",
        "- Full Seer: `z_t^F = F_phi(C_t)`",
        "- Observation feature: `u_t = E_eta(o_{t-1}, o_t)`",
        "- Latent update: `z_t^L = U_theta(z_{t-1}^L, u_t, r_t)`",
        "- Shared action path: `a_t = A_psi(z_t)`",
        "- `L` includes the full-Seer anchor action at offset 0. Full-Seer call ratio is measured as calls divided by policy steps.",
        "- `rho` is the actually executed observation-conditioned updater calls divided by all intermediate updater calls; it is N/A for `L=1`.",
        "",
        "## Row summary",
        "",
        "| checkpoint | row | L | feedback | rho | SR | Wilson 95% | full-call reduction | policy ms |",
        "|---:|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (int(item["checkpoint_id"]), str(item["row_id"]))):
        density = row.get("actual_feedback_density")
        density_text = "N/A" if density is None else f"{float(density):.3f}"
        lines.append(
            f"| {row['checkpoint_id']} | {row['row_id']} | {row['segment_length']} | "
            f"{row['feedback_schedule']} | {density_text} | {100.0 * float(row['success_rate']):.1f}% | "
            f"[{100.0 * float(row['wilson_95_low']):.1f}, {100.0 * float(row['wilson_95_high']):.1f}]% | "
            f"{100.0 * float(row['full_query_reduction_ratio']):.1f}% | {float(row['avg_policy_step_latency_ms']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Statistical interpretation",
            "",
            "`segment_grid_paired_flips.csv` pairs rows by `(task_id, episode_id, seed)`. The paired bootstrap resamples tasks and then paired episodes within each sampled task. Wilson intervals are marginal Bernoulli intervals. No result is called statistically significant solely from a point estimate.",
            "",
            "## Compute and continuity",
            "",
            "Component p50/p95/p99 values are computed from per-step logs after filtering by the corresponding call flag. `env.step` is reported separately and is not included in model-component latency. Translation and rotation continuity are normalized second differences from the production evaluator; gripper metrics are switches and short-horizon reversals per 100 policy steps.",
            "",
            "## Latent diagnostics",
            "",
            "Offset-level update norm, gate, latent norm, and current-observation feature norm are stored in `segment_grid_summary.json`. Same-observation shadow drift is included only when a row explicitly enabled shadow diagnostics and is not treated as ground truth.",
            "",
            "## Completeness",
            "",
            f"- Rows: {len(rows)}",
            f"- K1 parity artifacts: {sorted(summary.get('k1_parity', {}))}",
            f"- All row contracts passed: {summary['completeness']['all_row_contracts_passed']}",
            f"- All rows OSMesa-verified: {summary['completeness']['all_rows_osmesa_verified']}",
        ]
    )
    (output_dir / "final_segment_grid_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--parity-artifact", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    records = _discover_rows(args.input_root)
    episode_rows: Dict[str, List[Dict[str, str]]] = {}
    main_rows = []
    task_rows = []
    latency_rows = []
    continuity_rows = []
    offset_metrics = {}
    for record in records:
        episodes = _read_csv(Path(str(record["episode_metrics_path"])))
        summary = read_json(Path(str(record["summary_path"])))
        steps = _load_step_rows(Path(str(record["step_log_dir"])))
        if not steps:
            raise RuntimeError(f"Missing per-step logs for {record['row_id']}")
        episode_rows[str(record["row_id"])] = episodes
        row_summary = _row_summary(record, episodes, summary)
        main_rows.append(row_summary)
        task_rows.extend(_per_task_rows(record, episodes))
        latency_rows.append(_latency_row(record, steps))
        continuity_rows.append(
            _continuity_row(record, episodes, float(row_summary["success_rate"]))
        )
        offset_metrics[str(record["row_id"])] = _aggregate_offset_metrics(episodes)

    paired_rows = _paired_rows(records, episode_rows, args.bootstrap_iterations)
    parity = _load_parity(args.input_root, args.parity_artifact)
    summary_payload = {
        "schema_version": 1,
        "experiment": "latentloop_segment_length_feedback_density",
        "rows": main_rows,
        "k1_parity": parity,
        "latent_metrics_by_segment_offset": offset_metrics,
        "completeness": {
            "row_count": len(records),
            "all_row_contracts_passed": all(
                bool(record.get("validation", {}).get("pass", False))
                for record in records
            ),
            "all_rows_osmesa_verified": all(
                bool(row.get("renderer_all_ranks_verified"))
                and row.get("renderer_backend") == "osmesa"
                for row in main_rows
            ),
            "input_roots": [str(path.resolve()) for path in args.input_root],
        },
        "definitions": {
            "success_rate": "successful episodes / evaluated episodes",
            "full_seer_call_ratio": "full Seer calls / policy steps",
            "full_query_reduction_ratio": "1 - full Seer calls / policy steps",
            "actual_feedback_density": "observation-conditioned updater calls / all intermediate updater calls",
            "paired_key": ["task_id", "episode_id", "seed"],
            "paired_bootstrap": "resample tasks, then paired episodes within task",
        },
    }
    _write_csv(args.output_dir / "segment_grid_main_table.csv", main_rows)
    _write_csv(args.output_dir / "segment_grid_per_task.csv", task_rows)
    _write_csv(args.output_dir / "segment_grid_paired_flips.csv", paired_rows)
    _write_csv(args.output_dir / "segment_grid_latency.csv", latency_rows)
    _write_csv(args.output_dir / "segment_grid_continuity.csv", continuity_rows)
    atomic_write_json(args.output_dir / "segment_grid_summary.json", summary_payload)
    _plot_outputs(args.output_dir, main_rows, task_rows, continuity_rows)
    _write_report(args.output_dir, summary_payload)
    print(f"[DONE] segment-grid aggregation: {args.output_dir}")


if __name__ == "__main__":
    main()
