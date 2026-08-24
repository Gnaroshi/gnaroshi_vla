"""Aggregation and predeclared decisions for SimVLA Generation controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FROZEN_GENERATION_SOURCE_SHA256,
    FULL_ROW,
    GENERATION_ROW,
    NAIVE_ROW,
    ROWS,
    atomic_write_json,
    exact_mcnemar,
    hierarchical_bootstrap_difference,
    load_json,
    validate_row_counters,
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty episode table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    return float(value) if value not in {None, ""} else float("nan")


def _int(row: Mapping[str, Any], name: str) -> int:
    value = row.get(name)
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "yes"}:
        return 1
    if text in {"false", "no"}:
        return 0
    return int(float(text))


def _normalize_gripper_switch_rate(row: dict[str, Any]) -> None:
    if row.get("gripper_switch_rate") in {None, ""}:
        row["gripper_switch_rate"] = _float(row, "switch_disagreement")


def _episode_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (_int(row, "task_id"), _int(row, "trial_id"))


def _quantiles(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _load_npz_records(path: Path) -> dict[tuple[int, int, int], tuple[int, np.ndarray]]:
    # Accessing an array through NpzFile inside the loop re-decompresses the
    # complete member each time and keeps each backing allocation alive.
    # Materialize every member exactly once before constructing record views.
    with np.load(path) as data:
        task_ids = np.asarray(data["task_id"])
        trial_ids = np.asarray(data["trial_id"])
        query_indices = np.asarray(data["policy_query_index"])
        noise_seeds = np.asarray(data["action_noise_seed"])
        action_chunks = np.asarray(data["action_chunk"], dtype=np.float32)
    output: dict[tuple[int, int, int], tuple[int, np.ndarray]] = {}
    for index in range(len(task_ids)):
        key = (
            int(task_ids[index]),
            int(trial_ids[index]),
            int(query_indices[index]),
        )
        if key in output:
            raise RuntimeError(f"duplicate action-chunk key in {path}: {key}")
        output[key] = (
            int(noise_seeds[index]),
            action_chunks[index],
        )
    return output


def _summarize_row(row: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    episodes = len(rows)
    successes = sum(_int(item, "success") for item in rows)
    actions = sum(_int(item, "episode_length") for item in rows)
    queries = sum(_int(item, "num_policy_queries") for item in rows)
    full_vlm_calls = sum(_int(item, "num_full_vlm_calls") for item in rows)
    full_calls = sum(
        _int(item, "num_full_action_transformer_evaluations") for item in rows
    )
    generation_updates = sum(
        _int(item, "num_generation_loop_updates") for item in rows
    )
    integration_updates = sum(_int(item, "num_integration_updates") for item in rows)
    counter_gate = validate_row_counters(
        row,
        policy_queries=queries,
        full_action_transformer_calls=full_calls,
        generation_loop_updates=generation_updates,
        integration_updates=integration_updates,
        full_vlm_calls=full_vlm_calls,
    )
    if counter_gate["verdict"] != "ROW_COUNTER_PASS":
        raise RuntimeError(json.dumps(counter_gate, indent=2, sort_keys=True))
    policy_seconds = sum(_float(item, "policy_wall_time_seconds") for item in rows)
    environment_seconds = sum(
        _float(item, "environment_wall_time_seconds") for item in rows
    )
    task_success: dict[str, Any] = {}
    for task_id in range(10):
        subset = [item for item in rows if _int(item, "task_id") == task_id]
        task_success[str(task_id)] = {
            "episodes": len(subset),
            "successes": sum(_int(item, "success") for item in subset),
            "success_rate": float(np.mean([_int(item, "success") for item in subset])),
        }
    return {
        "row": row,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "per_task": task_success,
        "executed_actions": actions,
        "policy_queries": queries,
        "full_vlm_calls": full_vlm_calls,
        "full_action_transformer_evaluations": full_calls,
        "generation_loop_updates": generation_updates,
        "integration_updates": integration_updates,
        "full_action_transformer_calls_per_query": full_calls / queries,
        "generation_loop_updates_per_query": generation_updates / queries,
        "integration_updates_per_query": integration_updates / queries,
        "vlm_calls_per_query": full_vlm_calls / queries,
        "latency_per_policy_query_ms": float(
            np.average(
                [_float(item, "latency_per_policy_query_ms") for item in rows],
                weights=[_int(item, "num_policy_queries") for item in rows],
            )
        ),
        "latency_per_executed_action_ms": policy_seconds * 1000.0 / actions,
        "model_vlm_encoder_per_query_ms": float(
            np.average(
                [_float(item, "model_vlm_encoder_per_query_ms") for item in rows],
                weights=[_int(item, "num_policy_queries") for item in rows],
            )
        ),
        "model_action_generation_per_query_ms": float(
            np.average(
                [_float(item, "model_action_generation_per_query_ms") for item in rows],
                weights=[_int(item, "num_policy_queries") for item in rows],
            )
        ),
        "policy_wall_time_seconds": policy_seconds,
        "environment_wall_time_seconds": environment_seconds,
        "episode_wall_time_seconds": _quantiles(
            _float(item, "episode_wall_time_seconds") for item in rows
        ),
        "episode_length": _quantiles(_int(item, "episode_length") for item in rows),
        "gripper_switches": sum(_int(item, "gripper_switches") for item in rows),
        "gripper_switch_rate": float(
            np.average(
                [_float(item, "gripper_switch_rate") for item in rows],
                weights=[max(1, _int(item, "episode_length") - 1) for item in rows],
            )
        ),
        "normalized_second_difference": _quantiles(
            _float(item, "normalized_second_difference") for item in rows
        ),
        "counter_gate": counter_gate,
    }


def aggregate_row(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing aggregate output: {output}")
    shard_roots = [Path(value).expanduser().resolve() for value in args.shard]
    if len(shard_roots) not in {1, 2}:
        raise ValueError("row aggregation requires one rb2 shard or two sd1 shards")
    summaries = [load_json(root / "shard_summary.json") for root in shard_roots]
    if any(item.get("verdict") != "GENERATION_CONTROL_SHARD_PASS" for item in summaries):
        raise RuntimeError("one or more shard gates did not pass")
    if {item.get("row") for item in summaries} != {args.row}:
        raise RuntimeError("shard row mismatch")
    if {item.get("manifest_sha256") for item in summaries} != {
        args.expected_manifest_sha256
    }:
        raise RuntimeError("shard manifest mismatch")
    if {item.get("source_combined_sha256") for item in summaries} != {
        FROZEN_GENERATION_SOURCE_SHA256
    }:
        raise RuntimeError("shard source mismatch")
    rows = [item for root in shard_roots for item in _read_csv(root / "episode_metrics.csv")]
    keys = [_episode_key(item) for item in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate task/trial episodes across shards")
    if len(rows) != int(args.expected_episodes):
        raise RuntimeError(f"row has {len(rows)} episodes, expected {args.expected_episodes}")

    chunks: list[dict[tuple[int, int, int], tuple[int, np.ndarray]]] = [
        _load_npz_records(root / "action_chunks.npz") for root in shard_roots
    ]
    all_chunks: dict[tuple[int, int, int], tuple[int, np.ndarray]] = {}
    for shard in chunks:
        overlap = set(all_chunks) & set(shard)
        if overlap:
            raise RuntimeError(f"duplicate action chunk keys: {sorted(overlap)[:3]}")
        all_chunks.update(shard)

    output.mkdir(parents=True)
    _write_csv(output / "episode_metrics.csv", rows)
    ordered = sorted(all_chunks.items())
    np.savez_compressed(
        output / "action_chunks.npz",
        task_id=np.asarray([key[0] for key, _ in ordered], dtype=np.int16),
        trial_id=np.asarray([key[1] for key, _ in ordered], dtype=np.int16),
        policy_query_index=np.asarray([key[2] for key, _ in ordered], dtype=np.int32),
        action_noise_seed=np.asarray([value[0] for _, value in ordered], dtype=np.uint64),
        action_chunk=np.stack([value[1] for _, value in ordered]).astype(np.float32),
    )
    summary = {
        "verdict": "GENERATION_CONTROL_ROW_PASS",
        "classification": summaries[0]["classification"],
        "inference_seed": summaries[0]["inference_seed"],
        "manifest_sha256": args.expected_manifest_sha256,
        "source_combined_sha256": FROZEN_GENERATION_SOURCE_SHA256,
        "paper_runtime_match": all(bool(item["paper_runtime_match"]) for item in summaries),
        "shards": summaries,
        **_summarize_row(args.row, rows),
    }
    atomic_write_json(output / "row_summary.json", summary)
    return summary


def _paired_outcomes(
    reference: Mapping[tuple[int, int], Mapping[str, Any]],
    candidate: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise RuntimeError("paired outcome episode keys differ")

    def counts(keys: Sequence[tuple[int, int]]) -> dict[str, Any]:
        both_success = both_fail = reference_only = candidate_only = 0
        for key in keys:
            left = bool(_int(reference[key], "success"))
            right = bool(_int(candidate[key], "success"))
            if left and right:
                both_success += 1
            elif not left and not right:
                both_fail += 1
            elif left:
                reference_only += 1
            else:
                candidate_only += 1
        return {
            "episodes": len(keys),
            "both_success": both_success,
            "both_fail": both_fail,
            "reference_only": reference_only,
            "candidate_only": candidate_only,
            "exact_mcnemar_p": exact_mcnemar(reference_only, candidate_only),
        }

    ordered = sorted(reference)
    overall = counts(ordered)
    overall["per_task"] = {
        str(task_id): counts([key for key in ordered if key[0] == task_id])
        for task_id in sorted({key[0] for key in ordered})
    }
    return overall


def _first_query_action_error(
    reference_path: Path,
    candidate_path: Path,
    *,
    execution_horizon: int = 5,
) -> dict[str, Any]:
    reference = _load_npz_records(reference_path)
    candidate = _load_npz_records(candidate_path)
    keys = sorted(
        key for key in set(reference) & set(candidate) if int(key[2]) == 0
    )
    if not keys:
        return {"available": False, "reason": "no shared first-query action chunks"}
    errors: list[float] = []
    noise_mismatches = 0
    for key in keys:
        reference_seed, reference_chunk = reference[key]
        candidate_seed, candidate_chunk = candidate[key]
        if reference_seed != candidate_seed:
            noise_mismatches += 1
        errors.append(
            float(
                np.mean(
                    np.abs(
                        candidate_chunk[:execution_horizon]
                        - reference_chunk[:execution_horizon]
                    )
                )
            )
        )
    stats = _quantiles(errors)
    return {
        "available": True,
        "same_observation_scope": "policy_query_index_0_only",
        "executed_prefix_horizon": int(execution_horizon),
        "paired_episodes": len(keys),
        "noise_seed_mismatches": noise_mismatches,
        "first_r_action_l1": stats,
    }


def compare_three_rows(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing comparison output: {output}")
    roots = {
        FULL_ROW: Path(args.full).expanduser().resolve(),
        NAIVE_ROW: Path(args.naive).expanduser().resolve(),
        GENERATION_ROW: Path(args.generation).expanduser().resolve(),
    }
    summaries = {name: load_json(root / "row_summary.json") for name, root in roots.items()}
    if any(item.get("verdict") != "GENERATION_CONTROL_ROW_PASS" for item in summaries.values()):
        raise RuntimeError("all three row gates must pass")
    manifest_hashes = {item.get("manifest_sha256") for item in summaries.values()}
    if len(manifest_hashes) != 1:
        raise RuntimeError("three rows do not share one immutable manifest")
    rows = {name: _read_csv(root / "episode_metrics.csv") for name, root in roots.items()}
    by_key = {
        name: {_episode_key(item): item for item in values} for name, values in rows.items()
    }
    if len({frozenset(mapping) for mapping in by_key.values()}) != 1:
        raise RuntimeError("three row episode identities differ")

    flips = {
        "full_vs_naive": _paired_outcomes(by_key[FULL_ROW], by_key[NAIVE_ROW]),
        "full_vs_generation": _paired_outcomes(
            by_key[FULL_ROW], by_key[GENERATION_ROW]
        ),
        "naive_vs_generation": _paired_outcomes(
            by_key[NAIVE_ROW], by_key[GENERATION_ROW]
        ),
    }
    first_r = {
        "naive_vs_full": _first_query_action_error(
            roots[FULL_ROW] / "action_chunks.npz",
            roots[NAIVE_ROW] / "action_chunks.npz",
        ),
        "generation_vs_full": _first_query_action_error(
            roots[FULL_ROW] / "action_chunks.npz",
            roots[GENERATION_ROW] / "action_chunks.npz",
        ),
    }
    full_sr = float(summaries[FULL_ROW]["success_rate"])
    naive_sr = float(summaries[NAIVE_ROW]["success_rate"])
    generation_sr = float(summaries[GENERATION_ROW]["success_rate"])
    naive_latency = float(summaries[NAIVE_ROW]["latency_per_executed_action_ms"])
    generation_latency = float(
        summaries[GENERATION_ROW]["latency_per_executed_action_ms"]
    )
    naive_tail = float(first_r["naive_vs_full"]["first_r_action_l1"]["p95"])
    generation_tail = float(
        first_r["generation_vs_full"]["first_r_action_l1"]["p95"]
    )
    naive_gripper = float(summaries[NAIVE_ROW]["gripper_switch_rate"])
    generation_gripper = float(summaries[GENERATION_ROW]["gripper_switch_rate"])

    strict = {
        "generation_within_1pp_of_full": generation_sr >= full_sr - 0.01,
        "generation_not_worse_than_naive_by_over_0_5pp": (
            generation_sr >= naive_sr - 0.005
        ),
        "generation_latency_within_5pct_of_naive": (
            generation_latency <= naive_latency * 1.05
        ),
        "generation_sr_advantage_at_least_0_5pp": (
            generation_sr >= naive_sr + 0.005
        ),
        "generation_latency_advantage_at_least_10pct": (
            generation_latency <= naive_latency * 0.90
        ),
        "generation_first_r_p95_error_at_least_20pct_lower": (
            generation_tail <= naive_tail * 0.80
        ),
        "generation_gripper_switch_rate_at_least_10pct_lower": (
            generation_gripper <= naive_gripper * 0.90
        ),
    }
    strict_improvement = any(
        strict[name]
        for name in (
            "generation_sr_advantage_at_least_0_5pp",
            "generation_latency_advantage_at_least_10pct",
            "generation_first_r_p95_error_at_least_20pct_lower",
            "generation_gripper_switch_rate_at_least_10pct_lower",
        )
    )
    promising = (
        strict["generation_within_1pp_of_full"]
        and strict["generation_not_worse_than_naive_by_over_0_5pp"]
        and strict["generation_latency_within_5pct_of_naive"]
        and strict_improvement
    )
    naive_sufficient = (
        naive_sr >= generation_sr - 0.005
        and naive_latency <= generation_latency
        and naive_tail <= generation_tail
    )
    if promising:
        verdict = "SD1_LEARNED_LOOP_PROMISING"
    elif naive_sufficient:
        verdict = "SD1_NAIVE_NFE3_SUFFICIENT"
    else:
        verdict = "SD1_INCONCLUSIVE"

    output.mkdir(parents=True)
    result = {
        "verdict": verdict,
        "classification": "HOST_LOCAL_EGL_DIAGNOSTIC",
        "manifest_sha256": next(iter(manifest_hashes)),
        "source_combined_sha256": FROZEN_GENERATION_SOURCE_SHA256,
        "rows": summaries,
        "paired_outcomes": flips,
        "same_initial_observation_action_tail": first_r,
        "predeclared_checks": strict,
        "strict_improvement": strict_improvement,
        "paper_runtime_match": all(
            bool(item.get("paper_runtime_match")) for item in summaries.values()
        ),
        "normalized_second_difference_is_not_physical_jerk": True,
    }
    atomic_write_json(output / "sd1_generation_control_summary.json", result)
    lines = [
        "# sd1 Generation Control Diagnostic",
        "",
        f"- verdict: `{verdict}`",
        "- classification: `HOST_LOCAL_EGL_DIAGNOSTIC`",
        f"- manifest: `{result['manifest_sha256']}`",
        f"- paper runtime match: `{result['paper_runtime_match']}`",
        "",
        "| Row | Success | SR | ms/query | ms/executed action | Full calls/query | Updater calls/query | Integration updates/query |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ROWS:
        summary = summaries[row]
        lines.append(
            f"| {row} | {summary['successes']}/{summary['episodes']} | "
            f"{100.0 * summary['success_rate']:.2f}% | "
            f"{summary['latency_per_policy_query_ms']:.4f} | "
            f"{summary['latency_per_executed_action_ms']:.4f} | "
            f"{summary['full_action_transformer_calls_per_query']:.1f} | "
            f"{summary['generation_loop_updates_per_query']:.1f} | "
            f"{summary['integration_updates_per_query']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Normalized action second difference is reported as a discrete normalized-action metric, not physical jerk.",
        ]
    )
    (output / "sd1_generation_control_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def _discover_episode_csv(root: Path) -> Path:
    direct = root / "episode_metrics.csv"
    if direct.is_file():
        return direct
    matches = sorted(root.glob("shard_rank*_tasks_*/episode_metrics.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot identify one episode_metrics.csv under {root}")
    return matches[0]


def _legacy_row_summary(root: Path, row: str, inference_seed: str) -> dict[str, Any]:
    row_summary_path = root / "row_summary.json"
    if not row_summary_path.is_file():
        raise FileNotFoundError(f"missing row summary: {row_summary_path}")
    recorded = load_json(row_summary_path)
    accepted_names = {
        FULL_ROW: {FULL_ROW, "baseline_k1"},
        NAIVE_ROW: {NAIVE_ROW},
        GENERATION_ROW: {GENERATION_ROW},
    }[row]
    if recorded.get("row") not in accepted_names:
        raise RuntimeError(f"unexpected row identity under {root}: {recorded.get('row')}")
    if int(recorded.get("episodes", -1)) != 500:
        raise RuntimeError(f"row summary under {root} is not Long-500")
    if recorded.get("source_combined_sha256") != FROZEN_GENERATION_SOURCE_SHA256:
        raise RuntimeError(f"row summary source lock mismatch under {root}")
    manifest_sha256 = str(recorded.get("manifest_sha256", ""))
    if not manifest_sha256:
        raise RuntimeError(f"row summary lacks immutable manifest identity: {root}")

    rows = _read_csv(_discover_episode_csv(root))
    if len(rows) != 500:
        raise RuntimeError(f"confirmatory {inference_seed}/{row} is not Long-500")
    normalized: list[dict[str, Any]] = []
    contract = {
        FULL_ROW: (10, 0, 10),
        NAIVE_ROW: (3, 0, 3),
        GENERATION_ROW: (3, 7, 10),
    }[row]
    for item in rows:
        copy = dict(item)
        copy["inference_seed"] = inference_seed
        queries = _int(copy, "num_policy_queries")
        observed_full = copy.get(
            "num_full_action_transformer_evaluations",
            copy.get("num_action_transformer_flow_iterations"),
        )
        observed_updates = copy.get(
            "num_generation_loop_updates",
            copy.get("num_generation_decoder_only_steps", 0),
        )
        if observed_full in {None, ""}:
            raise RuntimeError(f"{inference_seed}/{row} lacks transformer-call counters")
        copy["num_full_action_transformer_evaluations"] = _int(
            {"value": observed_full}, "value"
        )
        copy["num_generation_loop_updates"] = _int(
            {"value": observed_updates}, "value"
        )
        copy["num_integration_updates"] = _int(
            copy, "num_full_action_transformer_evaluations"
        ) + _int(copy, "num_generation_loop_updates")
        copy.setdefault("num_full_vlm_calls", queries)
        copy.setdefault("policy_wall_time_seconds", _float(copy, "latency_per_executed_action_ms") * _int(copy, "episode_length") / 1000.0)
        copy.setdefault("environment_wall_time_seconds", float("nan"))
        copy.setdefault("episode_wall_time_seconds", float("nan"))
        copy.setdefault("latency_per_policy_query_ms", _float(copy, "latency_per_executed_action_ms") * _int(copy, "episode_length") / max(queries, 1))
        copy.setdefault("model_vlm_encoder_per_query_ms", float("nan"))
        copy.setdefault("model_action_generation_per_query_ms", float("nan"))
        copy.setdefault("gripper_switches", 0)
        _normalize_gripper_switch_rate(copy)
        normalized.append(copy)
    return {
        "rows": normalized,
        "recorded_summary": recorded,
        "manifest_sha256": manifest_sha256,
        "summary": _summarize_row(row, normalized),
    }


def confirmatory(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing confirmatory output: {output}")
    inputs = {
        "seed02": {
            FULL_ROW: Path(args.seed02_full).expanduser().resolve(),
            NAIVE_ROW: Path(args.seed02_naive).expanduser().resolve(),
            GENERATION_ROW: Path(args.seed02_generation).expanduser().resolve(),
        },
        "seed03": {
            FULL_ROW: Path(args.seed03_full).expanduser().resolve(),
            NAIVE_ROW: Path(args.seed03_naive).expanduser().resolve(),
            GENERATION_ROW: Path(args.seed03_generation).expanduser().resolve(),
        },
    }
    seed_results: dict[str, Any] = {}
    bootstrap_rows: list[dict[str, Any]] = []
    for seed_name, roots in inputs.items():
        loaded = {
            row: _legacy_row_summary(root, row, seed_name) for row, root in roots.items()
        }
        manifest_hashes = {payload["manifest_sha256"] for payload in loaded.values()}
        if len(manifest_hashes) != 1:
            raise RuntimeError(f"{seed_name} rows do not share one immutable manifest")
        by_key = {
            row: {_episode_key(item): item for item in payload["rows"]}
            for row, payload in loaded.items()
        }
        if len({frozenset(value) for value in by_key.values()}) != 1:
            raise RuntimeError(f"{seed_name} episode identities differ")
        for key in sorted(by_key[FULL_ROW]):
            bootstrap_rows.append(
                {
                    "inference_seed": seed_name,
                    "task_id": key[0],
                    "trial_id": key[1],
                    "full_success": _int(by_key[FULL_ROW][key], "success"),
                    "naive_success": _int(by_key[NAIVE_ROW][key], "success"),
                    "generation_success": _int(by_key[GENERATION_ROW][key], "success"),
                }
            )
        seed_results[seed_name] = {
            "manifest_sha256": next(iter(manifest_hashes)),
            "rows": {row: payload["summary"] for row, payload in loaded.items()},
            "paired_outcomes": {
                "full_vs_naive": _paired_outcomes(by_key[FULL_ROW], by_key[NAIVE_ROW]),
                "full_vs_generation": _paired_outcomes(
                    by_key[FULL_ROW], by_key[GENERATION_ROW]
                ),
                "naive_vs_generation": _paired_outcomes(
                    by_key[NAIVE_ROW], by_key[GENERATION_ROW]
                ),
            },
        }

    aggregate_rows = {
        row: {
            "mean_success_rate": float(
                np.mean([seed_results[seed]["rows"][row]["success_rate"] for seed in seed_results])
            ),
            "mean_latency_per_executed_action_ms": float(
                np.mean(
                    [
                        seed_results[seed]["rows"][row][
                            "latency_per_executed_action_ms"
                        ]
                        for seed in seed_results
                    ]
                )
            ),
        }
        for row in ROWS
    }
    full_sr = aggregate_rows[FULL_ROW]["mean_success_rate"]
    naive_sr = aggregate_rows[NAIVE_ROW]["mean_success_rate"]
    generation_sr = aggregate_rows[GENERATION_ROW]["mean_success_rate"]
    naive_latency = aggregate_rows[NAIVE_ROW]["mean_latency_per_executed_action_ms"]
    generation_latency = aggregate_rows[GENERATION_ROW][
        "mean_latency_per_executed_action_ms"
    ]
    naive_gripper = float(
        np.mean(
            [seed_results[seed]["rows"][NAIVE_ROW]["gripper_switch_rate"] for seed in seed_results]
        )
    )
    generation_gripper = float(
        np.mean(
            [
                seed_results[seed]["rows"][GENERATION_ROW]["gripper_switch_rate"]
                for seed in seed_results
            ]
        )
    )
    tail: dict[str, Any] = {"available": False}
    if args.learned_offline_screen and args.naive_offline_audit:
        learned_offline = load_json(args.learned_offline_screen)
        naive_offline = load_json(args.naive_offline_audit)
        learned_p95 = float(learned_offline["summaries"]["3"]["first5_l1"]["p95"])
        naive_summary = naive_offline["summaries"]["3"]
        naive_metric = naive_summary.get("first5_action_l1", naive_summary.get("first5_l1"))
        if naive_metric is None:
            raise RuntimeError("naive offline audit lacks first-R action L1")
        naive_p95 = float(naive_metric["p95"])
        tail = {
            "available": True,
            "learned_first_r_p95": learned_p95,
            "naive_first_r_p95": naive_p95,
            "learned_at_least_20pct_lower": learned_p95 <= naive_p95 * 0.80,
            "naive_no_worse": naive_p95 <= learned_p95,
        }
    checks = {
        "generation_within_1pp_of_full": generation_sr >= full_sr - 0.01,
        "generation_not_worse_than_naive_by_over_0_5pp": generation_sr >= naive_sr - 0.005,
        "generation_latency_within_5pct_of_naive": generation_latency <= naive_latency * 1.05,
        "generation_sr_advantage_at_least_0_5pp": generation_sr >= naive_sr + 0.005,
        "generation_latency_advantage_at_least_10pct": generation_latency <= naive_latency * 0.90,
        "generation_first_r_p95_error_at_least_20pct_lower": bool(
            tail.get("learned_at_least_20pct_lower", False)
        ),
        "generation_gripper_switch_rate_at_least_10pct_lower": (
            generation_gripper <= naive_gripper * 0.90
        ),
    }
    strict_improvement = (
        checks["generation_sr_advantage_at_least_0_5pp"]
        or checks["generation_latency_advantage_at_least_10pct"]
        or checks["generation_first_r_p95_error_at_least_20pct_lower"]
        or checks["generation_gripper_switch_rate_at_least_10pct_lower"]
    )
    confirmed = (
        checks["generation_within_1pp_of_full"]
        and checks["generation_not_worse_than_naive_by_over_0_5pp"]
        and checks["generation_latency_within_5pct_of_naive"]
        and strict_improvement
    )
    naive_sufficient = (
        naive_sr >= generation_sr - 0.005
        and naive_latency <= generation_latency
        and bool(tail.get("naive_no_worse", False))
        and naive_gripper <= generation_gripper
    )
    if confirmed:
        verdict = "GENERATION_LOOP_VALUE_CONFIRMED"
    elif naive_sufficient:
        verdict = "NAIVE_NFE3_SUFFICIENT"
    else:
        verdict = "GENERATION_LOOP_INCONCLUSIVE"
    result = {
        "verdict": verdict,
        "source_combined_sha256": FROZEN_GENERATION_SOURCE_SHA256,
        "checkpoint_fixed_across_inference_seeds": True,
        "inference_seeds_are_not_training_seeds": True,
        "selection_seed": "seed01",
        "confirmatory_seeds": ["seed02", "seed03"],
        "seed_results": seed_results,
        "aggregate_rows": aggregate_rows,
        "predeclared_checks": checks,
        "offline_first_r_tail": tail,
        "aggregate_gripper_switch_rate": {
            "naive_nfe3": naive_gripper,
            "generation_ng3": generation_gripper,
        },
        "hierarchical_bootstrap": {
            "generation_minus_full_success": hierarchical_bootstrap_difference(
                bootstrap_rows, value_a="full_success", value_b="generation_success"
            ),
            "generation_minus_naive_success": hierarchical_bootstrap_difference(
                bootstrap_rows, value_a="naive_success", value_b="generation_success"
            ),
        },
        "note": (
            "The preserved baseline/Generation seed02/03 run started before the new "
            "GL-vendor preflight contract. Its manifest and process environment declared EGL; "
            "that limitation remains explicit rather than being retroactively rewritten."
        ),
    }
    output.mkdir(parents=True)
    atomic_write_json(output / "confirmatory_generation_loop_verdict.json", result)
    lines = [
        "# Confirmatory Generation Loop Versus Naive NFE=3",
        "",
        f"- verdict: `{verdict}`",
        "- selection seed: `seed01` (excluded from the primary verdict)",
        "- confirmatory inference-noise seeds: `seed02`, `seed03`",
        "- trained checkpoint count: `1`",
        "",
        "| Row | Mean SR | Mean ms/executed action |",
        "|---|---:|---:|",
    ]
    for row in ROWS:
        aggregate = aggregate_rows[row]
        lines.append(
            f"| {row} | {100.0 * aggregate['mean_success_rate']:.2f}% | "
            f"{aggregate['mean_latency_per_executed_action_ms']:.4f} |"
        )
    lines.extend(
        [
            "",
            "These are inference-noise replicas on one fixed trained checkpoint, not independent training seeds.",
            "The exact per-seed paired flips, McNemar tests, and hierarchical bootstrap are stored in the JSON verdict.",
        ]
    )
    (output / "confirmatory_generation_loop_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    row = commands.add_parser("aggregate-row")
    row.add_argument("--row", choices=ROWS, required=True)
    row.add_argument("--output", required=True)
    row.add_argument("--shard", action="append", required=True)
    row.add_argument("--expected-manifest-sha256", required=True)
    row.add_argument("--expected-episodes", type=int, default=500)
    row.set_defaults(handler=aggregate_row)

    compare = commands.add_parser("compare-three")
    compare.add_argument("--output", required=True)
    compare.add_argument("--full", required=True)
    compare.add_argument("--naive", required=True)
    compare.add_argument("--generation", required=True)
    compare.set_defaults(handler=compare_three_rows)

    final = commands.add_parser("confirmatory")
    final.add_argument("--output", required=True)
    final.add_argument("--seed02-full", required=True)
    final.add_argument("--seed02-naive", required=True)
    final.add_argument("--seed02-generation", required=True)
    final.add_argument("--seed03-full", required=True)
    final.add_argument("--seed03-naive", required=True)
    final.add_argument("--seed03-generation", required=True)
    final.add_argument("--learned-offline-screen", default="")
    final.add_argument("--naive-offline-audit", default="")
    final.set_defaults(handler=confirmatory)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
