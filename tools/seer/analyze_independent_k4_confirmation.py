#!/usr/bin/env python3

"""Analyze paired independent K4 confirmations without pooling teachers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


TEACHERS = (33, 36, 38)
REPEATS = ("repeat_1", "repeat_2", "repeat_3")
MODES = ("base_k1", "adapter_k1", "lrnode_k4")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _single_path(paths, label: str) -> Path:
    paths = list(paths)
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {label}, found {len(paths)}: {paths}")
    return paths[0]


def find_mode_artifacts(run_root: Path) -> dict[str, tuple[Path, Path]]:
    patterns = {
        "base_k1": "baseline_*_full_K1_*/analysis/eval_summary.json",
        "adapter_k1": "ours_*_full_K1_*/analysis/eval_summary.json",
        "lrnode_k4": "ours_*_skip_K4_*/analysis/eval_summary.json",
    }
    found = {}
    for mode, pattern in patterns.items():
        summary = _single_path(run_root.glob(pattern), f"{mode} summary")
        episodes = summary.parent / "eval_episode_metrics.csv"
        if not episodes.is_file():
            raise FileNotFoundError(episodes)
        found[mode] = (summary, episodes)
    return found


def _float(row: dict, key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else math.nan


def _int(row: dict, key: str) -> int:
    return int(float(row[key]))


def paired_bootstrap_ci(
    grouped_differences: dict[int, list[float]],
    *,
    n_bootstrap: int = 20000,
    seed: int = 20260721,
) -> tuple[float, float]:
    """Stratified paired bootstrap over tasks and paired episodes."""
    if not grouped_differences:
        raise ValueError("No paired differences")
    task_ids = sorted(grouped_differences)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for bootstrap_id in range(n_bootstrap):
        task_means = []
        for sampled_task in rng.choice(task_ids, size=len(task_ids), replace=True):
            values = np.asarray(
                grouped_differences[int(sampled_task)], dtype=np.float64
            )
            task_means.append(
                float(rng.choice(values, size=len(values), replace=True).mean())
            )
        samples[bootstrap_id] = float(np.mean(task_means))
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def hierarchical_teacher_bootstrap_ci(
    repeat_task_differences: dict[str, dict[int, list[float]]],
    *,
    n_bootstrap: int = 30000,
    seed: int = 20260721,
) -> tuple[float, float]:
    """Bootstrap repeat, task, and paired episode levels within one teacher."""
    repeat_ids = sorted(repeat_task_differences)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for bootstrap_id in range(n_bootstrap):
        repeat_means = []
        for sampled_repeat in rng.choice(
            repeat_ids, size=len(repeat_ids), replace=True
        ):
            task_groups = repeat_task_differences[str(sampled_repeat)]
            task_ids = sorted(task_groups)
            task_means = []
            for sampled_task in rng.choice(
                task_ids, size=len(task_ids), replace=True
            ):
                values = np.asarray(
                    task_groups[int(sampled_task)], dtype=np.float64
                )
                task_means.append(
                    float(
                        rng.choice(values, size=len(values), replace=True).mean()
                    )
                )
            repeat_means.append(float(np.mean(task_means)))
        samples[bootstrap_id] = float(np.mean(repeat_means))
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def analyze(output_root: Path) -> dict:
    provenance_path = (
        output_root / "init_overlays" / "initial_state_provenance.json"
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    repeat_starts = {
        item["repeat"]: int(item["source_index_start"])
        for item in provenance["repeat_blocks"]
    }

    episode_rows = []
    repeat_rows = []
    task_rows = []
    mode_rows = []
    parity_checks = []
    teacher_repeat_groups = defaultdict(dict)

    for teacher in TEACHERS:
        for repeat_index, repeat in enumerate(REPEATS, start=1):
            run_root = output_root / "raw" / repeat / f"teacher_{teacher}"
            artifacts = find_mode_artifacts(run_root)
            mode_data = {}
            for mode, (summary_path, episode_path) in artifacts.items():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                episodes = read_csv(episode_path)
                keyed = {}
                for row in episodes:
                    task_id = _int(row, "task_id")
                    local_episode_id = _int(row, "episode_id")
                    source_index = repeat_starts[repeat] + local_episode_id
                    key = (task_id, source_index)
                    if key in keyed:
                        raise RuntimeError(
                            f"Duplicate episode key {key} in {episode_path}"
                        )
                    keyed[key] = row
                if len(keyed) != 100:
                    raise RuntimeError(
                        f"Expected 100 episodes in {episode_path}, found {len(keyed)}"
                    )
                mode_data[mode] = {
                    "summary": summary,
                    "summary_path": summary_path,
                    "episode_path": episode_path,
                    "episodes": keyed,
                }
                lrnode = summary["lrnode"]
                query = summary["query_reduction"]
                smooth = summary["action_smoothness"]
                mode_rows.append(
                    {
                        "teacher_checkpoint": teacher,
                        "adapter_checkpoint": (
                            "" if mode == "base_k1" else 39
                        ),
                        "repeat": repeat,
                        "mode": mode,
                        "episodes": len(keyed),
                        "success_rate": summary["success_rate"],
                        "lrnode_enabled": lrnode["enabled"],
                        "skip_full_forward": lrnode[
                            "eval_skip_full_forward"
                        ],
                        "query_interval": lrnode["query_interval"],
                        "num_env_steps": query["num_env_steps"],
                        "num_full_forward_calls": query[
                            "num_full_forward_calls"
                        ],
                        "num_lrnode_update_calls": query[
                            "num_lrnode_update_calls"
                        ],
                        "num_fast_encoder_calls": query[
                            "num_fast_encoder_calls"
                        ],
                        "num_action_head_calls": query[
                            "num_action_head_calls"
                        ],
                        "full_query_reduction_ratio": query[
                            "full_query_reduction_ratio"
                        ],
                        "avg_full_forward_ms": 1000.0
                        * lrnode["avg_full_forward_latency_sec"],
                        "avg_lrnode_ms": 1000.0
                        * lrnode["avg_lrnode_latency_sec"],
                        "avg_fast_encoder_ms": 1000.0
                        * lrnode["avg_fast_encoder_latency_sec"],
                        "avg_node_update_ms": 1000.0
                        * lrnode["avg_node_update_latency_sec"],
                        "avg_skip_action_head_ms": 1000.0
                        * lrnode["avg_action_head_latency_sec"],
                        "avg_policy_step_ms": 1000.0
                        * lrnode["avg_policy_step_latency_sec"],
                        "avg_env_step_ms": 1000.0
                        * lrnode["avg_env_step_latency_sec"],
                        "action_jerk_l2_mean": smooth[
                            "action_jerk_l2_mean"
                        ],
                        "action_jerk_l2_p95": smooth[
                            "action_jerk_l2_p95"
                        ],
                        "gripper_switch_rate": smooth[
                            "gripper_switch_rate"
                        ],
                        "summary_path": str(summary_path),
                        "episode_metrics_path": str(episode_path),
                    }
                )

            key_sets = [set(mode_data[mode]["episodes"]) for mode in MODES]
            paired_keys_match = key_sets[0] == key_sets[1] == key_sets[2]
            if not paired_keys_match:
                raise RuntimeError(
                    f"Episode keys differ for teacher={teacher} repeat={repeat}"
                )

            base = mode_data["base_k1"]["episodes"]
            adapter = mode_data["adapter_k1"]["episodes"]
            k4 = mode_data["lrnode_k4"]["episodes"]
            parity_success = all(
                _int(base[key], "success") == _int(adapter[key], "success")
                for key in base
            )
            parity_steps = all(
                _int(base[key], "num_steps") == _int(adapter[key], "num_steps")
                for key in base
            )
            base_sr = float(mode_data["base_k1"]["summary"]["success_rate"])
            adapter_sr = float(
                mode_data["adapter_k1"]["summary"]["success_rate"]
            )
            k4_sr = float(mode_data["lrnode_k4"]["summary"]["success_rate"])
            parity_sr = math.isclose(base_sr, adapter_sr, abs_tol=1e-12)
            parity_checks.append(
                {
                    "teacher_checkpoint": teacher,
                    "repeat": repeat,
                    "paired_episode_keys": paired_keys_match,
                    "success_vector_equal": parity_success,
                    "num_steps_vector_equal": parity_steps,
                    "success_rate_equal": parity_sr,
                    "passed": (
                        paired_keys_match
                        and parity_success
                        and parity_steps
                        and parity_sr
                    ),
                }
            )

            task_differences = defaultdict(list)
            fail_to_success = 0
            success_to_fail = 0
            for task_id, source_index in sorted(base):
                base_row = base[(task_id, source_index)]
                adapter_row = adapter[(task_id, source_index)]
                k4_row = k4[(task_id, source_index)]
                base_success = _int(base_row, "success")
                adapter_success = _int(adapter_row, "success")
                k4_success = _int(k4_row, "success")
                difference = k4_success - base_success
                task_differences[task_id].append(difference)
                fail_to_success += int(base_success == 0 and k4_success == 1)
                success_to_fail += int(
                    base_success == 1 and k4_success == 0
                )
                episode_rows.append(
                    {
                        "teacher_checkpoint": teacher,
                        "repeat": repeat,
                        "task_id": task_id,
                        "task_name": base_row["task_name"],
                        "local_episode_id": source_index - repeat_starts[repeat],
                        "source_initial_state_index": source_index,
                        "recorded_seed": _int(base_row, "seed"),
                        "base_k1_success": base_success,
                        "adapter_k1_success": adapter_success,
                        "lrnode_k4_success": k4_success,
                        "k4_minus_k1": difference,
                        "base_k1_num_steps": _int(base_row, "num_steps"),
                        "adapter_k1_num_steps": _int(
                            adapter_row, "num_steps"
                        ),
                        "lrnode_k4_num_steps": _int(k4_row, "num_steps"),
                        "base_k1_jerk": _float(
                            base_row, "avg_action_jerk"
                        ),
                        "adapter_k1_jerk": _float(
                            adapter_row, "avg_action_jerk"
                        ),
                        "lrnode_k4_jerk": _float(
                            k4_row, "avg_action_jerk"
                        ),
                        "base_k1_gripper_switch_rate": _float(
                            base_row, "gripper_switch_rate"
                        ),
                        "adapter_k1_gripper_switch_rate": _float(
                            adapter_row, "gripper_switch_rate"
                        ),
                        "lrnode_k4_gripper_switch_rate": _float(
                            k4_row, "gripper_switch_rate"
                        ),
                    }
                )

            ci_low, ci_high = paired_bootstrap_ci(
                task_differences,
                seed=20260721 + teacher * 10 + repeat_index,
            )
            task_delta_values = []
            for task_id in range(10):
                keys = [
                    key for key in sorted(base) if key[0] == task_id
                ]
                task_base = np.mean(
                    [_int(base[key], "success") for key in keys]
                )
                task_adapter = np.mean(
                    [_int(adapter[key], "success") for key in keys]
                )
                task_k4 = np.mean(
                    [_int(k4[key], "success") for key in keys]
                )
                task_delta = float(task_k4 - task_base)
                task_delta_values.append(task_delta)
                task_rows.append(
                    {
                        "teacher_checkpoint": teacher,
                        "repeat": repeat,
                        "task_id": task_id,
                        "task_name": base[keys[0]]["task_name"],
                        "base_k1_sr": task_base,
                        "adapter_k1_sr": task_adapter,
                        "lrnode_k4_sr": task_k4,
                        "k4_minus_k1": task_delta,
                        "fail_to_success": sum(
                            _int(base[key], "success") == 0
                            and _int(k4[key], "success") == 1
                            for key in keys
                        ),
                        "success_to_fail": sum(
                            _int(base[key], "success") == 1
                            and _int(k4[key], "success") == 0
                            for key in keys
                        ),
                    }
                )

            q = mode_data["lrnode_k4"]["summary"]["query_reduction"]
            smooth = mode_data["lrnode_k4"]["summary"]["action_smoothness"]
            lr = mode_data["lrnode_k4"]["summary"]["lrnode"]
            repeat_rows.append(
                {
                    "teacher_checkpoint": teacher,
                    "adapter_checkpoint": 39,
                    "repeat": repeat,
                    "source_index_start": repeat_starts[repeat],
                    "source_index_stop_exclusive": repeat_starts[repeat] + 10,
                    "episodes": 100,
                    "recorded_seed": _int(
                        next(iter(base.values())), "seed"
                    ),
                    "base_k1_sr": base_sr,
                    "adapter_k1_sr": adapter_sr,
                    "lrnode_k4_sr": k4_sr,
                    "k4_minus_k1": k4_sr - base_sr,
                    "paired_bootstrap_ci_low": ci_low,
                    "paired_bootstrap_ci_high": ci_high,
                    "fail_to_success": fail_to_success,
                    "success_to_fail": success_to_fail,
                    "task_delta_variance": float(
                        np.var(task_delta_values, ddof=1)
                    ),
                    "k1_parity_passed": (
                        paired_keys_match
                        and parity_success
                        and parity_steps
                        and parity_sr
                    ),
                    "k4_action_jerk_l2_mean": smooth[
                        "action_jerk_l2_mean"
                    ],
                    "k4_action_jerk_l2_p95": smooth[
                        "action_jerk_l2_p95"
                    ],
                    "k4_gripper_switch_rate": smooth[
                        "gripper_switch_rate"
                    ],
                    "k4_full_forward_calls": q[
                        "num_full_forward_calls"
                    ],
                    "k4_lrnode_update_calls": q[
                        "num_lrnode_update_calls"
                    ],
                    "k4_full_query_reduction_ratio": q[
                        "full_query_reduction_ratio"
                    ],
                    "k4_avg_full_forward_ms": 1000.0
                    * lr["avg_full_forward_latency_sec"],
                    "k4_avg_lrnode_ms": 1000.0
                    * lr["avg_lrnode_latency_sec"],
                    "k4_avg_policy_step_ms": 1000.0
                    * lr["avg_policy_step_latency_sec"],
                }
            )
            teacher_repeat_groups[teacher][repeat] = {
                int(task): list(values)
                for task, values in task_differences.items()
            }

    teacher_rows = []
    for teacher in TEACHERS:
        rows = [
            row
            for row in repeat_rows
            if row["teacher_checkpoint"] == teacher
        ]
        deltas = np.asarray(
            [row["k4_minus_k1"] for row in rows], dtype=np.float64
        )
        ci_low, ci_high = hierarchical_teacher_bootstrap_ci(
            teacher_repeat_groups[teacher],
            seed=20260721 + teacher,
        )
        teacher_rows.append(
            {
                "teacher_checkpoint": teacher,
                "adapter_checkpoint": 39,
                "repeat_count": len(rows),
                "episodes_per_repeat": 100,
                "mean_k4_minus_k1": float(deltas.mean()),
                "std_k4_minus_k1": float(deltas.std(ddof=1)),
                "positive_repeat_count": int((deltas > 0).sum()),
                "zero_repeat_count": int((deltas == 0).sum()),
                "negative_repeat_count": int((deltas < 0).sum()),
                "hierarchical_paired_bootstrap_ci_low": ci_low,
                "hierarchical_paired_bootstrap_ci_high": ci_high,
                "all_k1_parity_passed": all(
                    bool(row["k1_parity_passed"]) for row in rows
                ),
                "total_fail_to_success": sum(
                    int(row["fail_to_success"]) for row in rows
                ),
                "total_success_to_fail": sum(
                    int(row["success_to_fail"]) for row in rows
                ),
            }
        )

    task_aggregate_rows = []
    for teacher in TEACHERS:
        for task_id in range(10):
            rows = [
                row
                for row in task_rows
                if row["teacher_checkpoint"] == teacher
                and row["task_id"] == task_id
            ]
            deltas = np.asarray(
                [row["k4_minus_k1"] for row in rows], dtype=np.float64
            )
            task_aggregate_rows.append(
                {
                    "teacher_checkpoint": teacher,
                    "task_id": task_id,
                    "task_name": rows[0]["task_name"],
                    "mean_k4_minus_k1": float(deltas.mean()),
                    "std_k4_minus_k1": float(deltas.std(ddof=1)),
                    "repeat_values": ";".join(
                        f"{value:.6f}" for value in deltas
                    ),
                }
            )

    all_parity = all(bool(row["passed"]) for row in parity_checks)
    confirmed = (
        all_parity
        and all(row["mean_k4_minus_k1"] > 0 for row in teacher_rows)
        and all(row["positive_repeat_count"] >= 2 for row in teacher_rows)
        and all(
            row["hierarchical_paired_bootstrap_ci_low"] > 0
            for row in teacher_rows
        )
    )
    clearly_not_confirmed = (
        all_parity
        and all(row["mean_k4_minus_k1"] <= 0 for row in teacher_rows)
        and all(
            row["hierarchical_paired_bootstrap_ci_high"] <= 0
            for row in teacher_rows
        )
    )
    if confirmed:
        verdict = "K4_EFFECT_CONFIRMED"
    elif clearly_not_confirmed:
        verdict = "K4_EFFECT_NOT_CONFIRMED"
    else:
        verdict = "K4_EFFECT_INCONCLUSIVE"

    write_csv(output_root / "paired_episode_outcomes.csv", episode_rows)
    write_csv(output_root / "mode_metrics.csv", mode_rows)
    write_csv(output_root / "repeat_summary.csv", repeat_rows)
    write_csv(output_root / "task_repeat_results.csv", task_rows)
    write_csv(output_root / "task_variance_summary.csv", task_aggregate_rows)
    write_csv(output_root / "teacher_repeat_statistics.csv", teacher_rows)
    write_csv(output_root / "k1_parity_checks.csv", parity_checks)

    payload = {
        "verdict": verdict,
        "decision_rule": {
            "confirmed": (
                "All nine K1 parity checks pass; every teacher has positive "
                "mean K4-K1, at least two positive repeats, and a strictly "
                "positive hierarchical paired-bootstrap 95% CI."
            ),
            "not_confirmed": (
                "All parity checks pass and every teacher has non-positive "
                "mean and non-positive CI upper bound."
            ),
            "inconclusive": "Any result between the two predeclared conditions.",
        },
        "teachers_are_not_pooled": True,
        "teacher_statistics": teacher_rows,
        "repeat_statistics": repeat_rows,
        "mode_metrics": mode_rows,
        "k1_parity_checks": parity_checks,
    }
    (output_root / "final_statistics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "final_verdict.txt").write_text(
        verdict + "\n", encoding="utf-8"
    )
    write_report(output_root, payload)
    return payload


def write_report(output_root: Path, payload: dict) -> None:
    teacher_rows = payload["teacher_statistics"]
    repeat_rows = payload["repeat_statistics"]
    lines = [
        "# Independent epoch-39 K4 confirmation",
        "",
        f"## Verdict: `{payload['verdict']}`",
        "",
        "## Protocol",
        "",
        "- Released Seer teachers: 33, 36, 38.",
        "- Each teacher uses only its own fixed LR-NODE adapter `39.pth`.",
        "- Three disjoint repeats use original LIBERO init-state indices "
        "`20-29`, `30-39`, and `40-49`.",
        "- The previous default evaluation used indices `0-19`; changing "
        "`--seed` alone does not select another init state.",
        "- Within each teacher/repeat, base K1, adapter-loaded K1, and K4 use "
        "the identical 100 `(task, source init index)` pairs.",
        "- Teachers are analyzed separately and are not pooled as independent "
        "episodes.",
        "",
        "## Per-repeat paired results",
        "",
        "| Teacher | Repeat | K1 SR | adapter K1 SR | K4 SR | K4-K1 | "
        "F->S | S->F | paired bootstrap 95% CI | parity |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in repeat_rows:
        lines.append(
            f"| {row['teacher_checkpoint']} | {row['repeat']} | "
            f"{100 * row['base_k1_sr']:.1f}% | "
            f"{100 * row['adapter_k1_sr']:.1f}% | "
            f"{100 * row['lrnode_k4_sr']:.1f}% | "
            f"{100 * row['k4_minus_k1']:+.1f} pp | "
            f"{row['fail_to_success']} | {row['success_to_fail']} | "
            f"[{100 * row['paired_bootstrap_ci_low']:+.2f}, "
            f"{100 * row['paired_bootstrap_ci_high']:+.2f}] pp | "
            f"{'PASS' if row['k1_parity_passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Across-repeat statistics by teacher",
            "",
            "| Teacher | mean K4-K1 | std | positive repeats | "
            "hierarchical paired-bootstrap 95% CI | F->S | S->F |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in teacher_rows:
        lines.append(
            f"| {row['teacher_checkpoint']} | "
            f"{100 * row['mean_k4_minus_k1']:+.2f} pp | "
            f"{100 * row['std_k4_minus_k1']:.2f} pp | "
            f"{row['positive_repeat_count']}/3 | "
            f"[{100 * row['hierarchical_paired_bootstrap_ci_low']:+.2f}, "
            f"{100 * row['hierarchical_paired_bootstrap_ci_high']:+.2f}] pp | "
            f"{row['total_fail_to_success']} | "
            f"{row['total_success_to_fail']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `paired_episode_outcomes.csv`: every paired episode outcome and "
            "episode-level jerk/gripper metrics.",
            "- `mode_metrics.csv`: calls, latency, jerk, and gripper rate for "
            "all three evaluation conditions.",
            "- `task_repeat_results.csv`: task-wise K1/K4 outcomes per repeat.",
            "- `task_variance_summary.csv`: per-task mean/std across repeats.",
            "- `repeat_summary.csv`: SR, flips, CI, calls, jerk, gripper, and "
            "latency per teacher/repeat.",
            "- `teacher_repeat_statistics.csv`: mean/std and hierarchical CI "
            "without pooling teachers.",
            "- `k1_parity_checks.csv`: exact success and num-step parity.",
            "- `initial_state_audit.md`: audited current-stack selection semantics.",
            "",
        ]
    )
    (output_root / "final_independent_k4_confirmation_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = analyze(args.output_root.resolve())
    print(
        f"[ANALYSIS][OK] verdict={payload['verdict']} "
        f"report={args.output_root / 'final_independent_k4_confirmation_report.md'}"
    )


if __name__ == "__main__":
    main()
