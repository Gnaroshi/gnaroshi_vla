"""Aggregate paired K_C and learned-versus-naive action-generation controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_aggregate import (
    _key,
    _read_csv,
    _summarize,
    _validate_episode_table,
    _write_csv,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FULL_ROW,
    atomic_write_json,
    load_json,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_aggregate import (
    _effect,
    _observed_pareto,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import (
    condition_row_name,
    naive_condition_row_name,
    row_spec,
)


ROW_ARGUMENTS = {
    FULL_ROW: "baseline",
    condition_row_name(2, 10): "kc2_ng10",
    condition_row_name(2, 3): "kc2_ng3",
    condition_row_name(2, 2): "kc2_ng2",
    naive_condition_row_name(2, 3): "kc2_naive_nfe3",
    naive_condition_row_name(2, 2): "kc2_naive_nfe2",
    condition_row_name(3, 10): "kc3_ng10",
    condition_row_name(3, 3): "kc3_ng3",
    naive_condition_row_name(3, 3): "kc3_naive_nfe3",
}


def _load(root: Path, row_name: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    summary = load_json(root / "row_summary.json")
    accepted = {
        "FIXED_2X2_ROW_PASS",
        "GENERATION_CONTROL_ROW_PASS",
        "KC_FRONTIER_ROW_PASS",
    }
    if summary.get("verdict") not in accepted or summary.get("row") != row_name:
        raise RuntimeError(f"row gate mismatch for {row_name}: {root}")
    rows = _read_csv(root / "episode_metrics.csv")
    _validate_episode_table(row_name, rows)
    return summary, rows


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    roots = {
        row_name: Path(getattr(args, argument)).expanduser().resolve()
        for row_name, argument in ROW_ARGUMENTS.items()
    }
    loaded = {name: _load(root, name) for name, root in roots.items()}
    metadata = {name: value[0] for name, value in loaded.items()}
    tables = {name: value[1] for name, value in loaded.items()}
    manifest_hashes = {item.get("manifest_sha256") for item in metadata.values()}
    inference_seeds = {item.get("inference_seed") for item in metadata.values()}
    classifications = {item.get("classification") for item in metadata.values()}
    if len(manifest_hashes) != 1 or len(inference_seeds) != 1 or len(classifications) != 1:
        raise RuntimeError("all rows must share manifest, seed, and runtime classification")
    keyed = {name: {_key(row): row for row in rows} for name, rows in tables.items()}
    if len({frozenset(items) for items in keyed.values()}) != 1:
        raise RuntimeError("all rows must contain the same paired 500 episodes")
    metrics = {name: _summarize(name, rows) for name, rows in tables.items()}
    baseline = metrics[FULL_ROW]
    effects_vs_baseline = {
        name: _effect(baseline, metrics[name], keyed[FULL_ROW], keyed[name])
        for name in metrics
        if name != FULL_ROW
    }
    learned_vs_naive_pairs = {}
    for k_c, n_g in ((2, 3), (2, 2), (3, 3)):
        learned = condition_row_name(k_c, n_g)
        naive = naive_condition_row_name(k_c, n_g)
        learned_vs_naive_pairs[f"kc{k_c}_nfe{n_g}"] = {
            "learned_row": learned,
            "naive_row": naive,
            "learned_vs_naive": _effect(
                metrics[naive], metrics[learned], keyed[naive], keyed[learned]
            ),
        }

    comparison_rows = []
    for key in sorted(keyed[FULL_ROW]):
        comparison_rows.append(
            {
                "task_id": key[0],
                "trial_id": key[1],
                **{
                    f"{name}_success": int(float(keyed[name][key]["success"]))
                    for name in roots
                },
            }
        )
    result = {
        "verdict": "JOINT_NFE_FRONTIER_COMPLETE",
        "paper_table_eligible": False,
        "classification": next(iter(classifications)),
        "inference_seed": next(iter(inference_seeds)),
        "manifest_sha256": next(iter(manifest_hashes)),
        "episodes_per_row": 500,
        "rows": metrics,
        "effects_vs_full_baseline": effects_vs_baseline,
        "learned_vs_naive": learned_vs_naive_pairs,
        "observed_success_latency_pareto_rows": _observed_pareto(metrics),
        "nominal_compute_contracts": {
            name: {
                "k_c": row_spec(name).k_c,
                "n_g_or_naive_nfe": row_spec(name).n_g,
                "learned_generation": row_spec(name).uses_generation,
                "naive_nfe": row_spec(name).naive_nfe,
            }
            for name in metrics
        },
        "interpretation": (
            "Single-seed same-host paired mechanism frontier. Seed replication is deferred; "
            "latency must later be remeasured in isolated runs."
        ),
    }
    output.mkdir(parents=True)
    atomic_write_json(output / "joint_nfe_frontier_summary.json", result)
    _write_csv(output / "paired_episode_outcomes.csv", comparison_rows)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    for argument in ROW_ARGUMENTS.values():
        parser.add_argument(f"--{argument.replace('_', '-')}", dest=argument, required=True)
    return parser


def main() -> int:
    result = aggregate(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
