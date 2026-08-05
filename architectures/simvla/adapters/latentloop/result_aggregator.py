"""Aggregate offline/online artifacts and apply frozen LatentLoop gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from methods.latentloop.eval import apply_predeclared_decisions

from .source_lock import UPSTREAM, collect_source_lock


def _load(path: str) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _paired_lower(summary: dict[str, Any], row: str) -> float | None:
    ci = summary.get("paired_vs_full", {}).get(row, {}).get(
        "task_hierarchical_paired_ci95_pp"
    )
    return None if not ci or ci[0] is None else float(ci[0])


def _candidate_vs_row_lower(
    summary: dict[str, Any],
    candidate: str,
    baseline: str,
) -> float | None:
    rows = summary.get("rows", {})
    episodes = int(summary.get("episodes_per_row", 0))
    if candidate not in rows or baseline not in rows or episodes < 1:
        return None
    # The online evaluator currently stores hierarchical paired CIs only versus
    # full. For candidate-vs-ablation gates, require a separately aggregated
    # paired comparison rather than substituting an unpaired SR difference.
    value = summary.get("paired_between_rows", {}).get(candidate, {}).get(baseline, {})
    ci = value.get("task_hierarchical_paired_ci95_pp")
    return None if not ci or ci[0] is None else float(ci[0])


def _candidate_vs_row_mean(
    summary: dict[str, Any],
    candidate: str,
    baseline: str,
) -> float | None:
    value = summary.get("paired_between_rows", {}).get(candidate, {}).get(baseline, {})
    difference = value.get("candidate_minus_baseline_pp")
    return None if difference is None else float(difference)


def _all_credible_comparisons_nonpositive(
    comparisons: list[tuple[bool, float | None]],
) -> bool:
    """Return true when every credible protocol says the baseline matches/beats recurrent."""

    credible = [value for enabled, value in comparisons if enabled]
    return bool(credible) and all(value is not None and value <= 0.0 for value in credible)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    """Build immutable decision inputs from supplied experiment summaries."""

    k1_r1 = _load(args.k1_r1_summary)
    k1_r5 = _load(args.k1_r5_summary)
    offline_r1 = _load(args.offline_r1)
    offline_r5 = _load(args.offline_r5)
    online_r1 = _load(args.online_r1)
    online_r5 = _load(args.online_r5)
    confirmation_r1 = _load(args.confirmation_r1)
    k1_results = [
        k1_r1.get("k1_parity") or {},
        k1_r5.get("k1_parity") or {},
    ]
    r1_chunk = "chunk_aware_latentloop_k4"
    r5_chunk = "chunk_aware_latentloop_k2"
    r1_action = "action_chunk_correction_k4"
    r1_nonrec = "nonrecurrent_condition_k4"
    r5_old_difference = _candidate_vs_row_mean(
        online_r5, r5_chunk, "old_observation_only_k2"
    )
    offline_beats_old = bool(
        offline_r5.get("gate", {}).get(
            "offline_chunk_aware_beats_old_observation_only_prefix_error", False
        )
    )
    input_summary: dict[str, Any] = {
        "k1_exact_action_chunk_equality": all(
            bool(result.get("exact_action_chunk_equality", False)) for result in k1_results
        ),
        "k1_identical_paired_outcomes": all(
            bool(result.get("identical_paired_outcomes", False)) for result in k1_results
        ),
        "k1_updater_calls": sum(int(result.get("updater_calls", -1)) for result in k1_results),
        "k1_observation_encoder_calls": sum(
            int(result.get("observation_encoder_calls", -1)) for result in k1_results
        ),
        "k1_action_encoder_calls": sum(
            int(result.get("action_encoder_calls", -1)) for result in k1_results
        ),
        "offline_chunk_aware_beats_hold_prefix_error": offline_r5.get("gate", {}).get(
            "offline_chunk_aware_beats_hold_prefix_error"
        ),
        "offline_chunk_aware_beats_old_observation_only_prefix_error": offline_r5.get(
            "gate", {}
        ).get("offline_chunk_aware_beats_old_observation_only_prefix_error"),
        "r1_confirmation_complete": bool(
            confirmation_r1
            and int(confirmation_r1.get("episodes_per_row", 0)) >= 200
        ),
        "r1_k4_paired_ci_lower_pp": _paired_lower(confirmation_r1, r1_chunk),
        "r1_k4_vs_no_observation_ci_lower_pp": _candidate_vs_row_lower(
            confirmation_r1, r1_chunk, "no_observation_k4"
        ),
        "r1_k4_full_condition_reduction": confirmation_r1.get("rows", {})
        .get(r1_chunk, {})
        .get("full_condition_reduction_per_policy_query"),
        "r1_k4_worse_than_both_matched_baselines": bool(
            confirmation_r1.get("rows", {}).get(r1_chunk, {}).get("success_rate", -1)
            < confirmation_r1.get("rows", {}).get(r1_action, {}).get("success_rate", 2)
            and confirmation_r1.get("rows", {}).get(r1_chunk, {}).get("success_rate", -1)
            < confirmation_r1.get("rows", {}).get(r1_nonrec, {}).get("success_rate", 2)
        ),
        "r5_k2_paired_ci_lower_pp": _paired_lower(online_r5, r5_chunk),
        "r5_k2_full_condition_reduction": online_r5.get("rows", {})
        .get(r5_chunk, {})
        .get("full_condition_reduction_per_policy_query"),
        "r5_k2_beats_hold": bool(
            (_candidate_vs_row_mean(online_r5, r5_chunk, "hold_condition_k2") or -999) > 0
        ),
        "r5_k2_beats_no_observation": bool(
            (_candidate_vs_row_mean(online_r5, r5_chunk, "no_observation_k2") or -999) > 0
        ),
        "r5_k2_improves_old_observation_only": bool(
            r5_old_difference is not None
            and (
                r5_old_difference > 0.0
                or (r5_old_difference >= -3.0 and offline_beats_old)
            )
        ),
        "simvla_r1_credible": False,
        "simvla_r5_credible": False,
        "simvla_screening_complete": bool(online_r1 and online_r5),
    }
    preliminary = apply_predeclared_decisions(input_summary)
    r1_credible = bool(preliminary["R1_K4_PASS"])
    r5_credible = bool(preliminary["R5_K2_PASS"])
    r1_comparison = confirmation_r1 or online_r1
    r1_nonrec_difference = _candidate_vs_row_mean(
        r1_comparison, r1_chunk, r1_nonrec
    )
    r1_action_difference = _candidate_vs_row_mean(
        r1_comparison, r1_chunk, r1_action
    )
    r5_nonrec_difference = _candidate_vs_row_mean(
        online_r5, r5_chunk, "nonrecurrent_condition_k2"
    )
    r5_action_difference = _candidate_vs_row_mean(
        online_r5, r5_chunk, "action_chunk_correction_k2"
    )
    input_summary.update(
        {
            "simvla_r1_credible": r1_credible,
            "simvla_r5_credible": r5_credible,
            "simvla_r5_chunk_aware_supported": bool(
                r5_credible
                and input_summary["r5_k2_improves_old_observation_only"]
            ),
            "nonrecurrent_matches_or_beats_recurrent": _all_credible_comparisons_nonpositive(
                [
                    (r1_credible, r1_nonrec_difference),
                    (r5_credible, r5_nonrec_difference),
                ]
            ),
            "action_correction_matches_or_beats_recurrent": _all_credible_comparisons_nonpositive(
                [
                    (r1_credible, r1_action_difference),
                    (r5_credible, r5_action_difference),
                ]
            ),
            "recurrent_beats_nonrecurrent_both_architectures": False,
            "recurrent_beats_action_correction_both_architectures": False,
            "cross_architecture_latent_location_blocker": (
                "No machine-readable Seer matched nonrecurrent/action-correction input was supplied."
            ),
            "paired_candidate_minus_nonrecurrent_pp": {
                "r1": r1_nonrec_difference,
                "r5": r5_nonrec_difference,
            },
            "paired_candidate_minus_action_correction_pp": {
                "r1": r1_action_difference,
                "r5": r5_action_difference,
            },
        }
    )
    decisions = apply_predeclared_decisions(input_summary)
    t2_gate_r5 = {
        "execution_horizon": 5,
        "T1_K2_OFFLINE_PASS": bool(
            offline_r5.get("gate", {}).get("OFFLINE_PREFIX_GATE_PASS", False)
        ),
        "T1_K2_ONLINE_PASS": bool(decisions["R5_K2_PASS"]),
    }
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_lock = collect_source_lock(
        checkpoint=getattr(args, "checkpoint", "YuankaiLuo/SimVLA-LIBERO"),
        norm_stats_path=getattr(
            args,
            "norm_stats",
            str(UPSTREAM / "norm_stats" / "libero_norm.json"),
        ),
    )
    result = {
        "inputs": input_summary,
        "decisions": decisions,
        "t2_gate_r5": t2_gate_r5,
        "artifacts": {
            "k1_r1": args.k1_r1_summary,
            "k1_r5": args.k1_r5_summary,
            "offline_r1": args.offline_r1,
            "offline_r5": args.offline_r5,
            "online_r1": args.online_r1,
            "online_r5": args.online_r5,
            "confirmation_r1": args.confirmation_r1,
        },
    }
    (output / "combined_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "source_lock.json").write_text(
        json.dumps(source_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "go_no_go.json").write_text(
        json.dumps(decisions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "t2_gate_r5.json").write_text(
        json.dumps(t2_gate_r5, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "confirmation_gate_template.json").write_text(
        json.dumps(
            {
                "CONFIRMATION_APPROVED": False,
                "CONFIRMATION_ROWS_APPROVED": [],
                "note": "Fill only after screening analysis; the evaluator requires an exact row-name match.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    """Aggregate completed artifacts; missing inputs remain explicit blockers."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument(
        "--norm-stats",
        default=str(UPSTREAM / "norm_stats" / "libero_norm.json"),
    )
    parser.add_argument("--k1-r1-summary", required=True)
    parser.add_argument("--k1-r5-summary", required=True)
    parser.add_argument("--offline-r1", default="")
    parser.add_argument("--offline-r5", default="")
    parser.add_argument("--online-r1", default="")
    parser.add_argument("--online-r5", default="")
    parser.add_argument("--confirmation-r1", default="")
    args = parser.parse_args()
    result = aggregate(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
