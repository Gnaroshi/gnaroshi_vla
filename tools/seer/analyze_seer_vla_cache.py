#!/usr/bin/env python3
"""Validate and summarize a Seer VLA-Cache comparison campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODES = ("off", "matched_full", "reuse")


def _load_summary(campaign_root: Path, mode: str) -> tuple[Path, dict]:
    path = campaign_root / mode / "analysis" / "eval_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing evaluation summary for {mode}: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _episode_count(summary: dict) -> int:
    return sum(int(row.get("num_episodes", 0)) for row in summary.get("task_results", []))


def _renderer(summary: dict) -> str:
    metadata = summary.get("renderer_backend", {})
    return str(
        metadata.get("backend_classification")
        or metadata.get("effective_backend")
        or metadata.get("requested_backend")
        or "unknown"
    )


def _row(mode: str, path: Path, summary: dict) -> dict[str, object]:
    cache = summary.get("vla_cache", {})
    calls = int(cache.get("calls", 0))
    first_queries = int(cache.get("first_queries", 0))
    reuse_calls = int(cache.get("actual_kv_reuse_calls", 0))
    nonfirst = max(0, calls - first_queries)
    return {
        "mode": mode,
        "summary": str(path),
        "episodes": _episode_count(summary),
        "success_rate_pct": 100.0 * float(summary.get("success_rate", 0.0)),
        "num_env_steps": int(summary.get("num_env_steps", 0)),
        "avg_policy_step_latency_ms": float(
            summary.get("avg_policy_step_latency_ms", 0.0)
        ),
        "avg_full_forward_latency_ms": float(
            summary.get("avg_full_forward_latency_ms", 0.0)
        ),
        "renderer": _renderer(summary),
        "cache_calls": calls,
        "cache_first_queries": first_queries,
        "cache_nonfirst_queries": nonfirst,
        "actual_kv_reuse_calls": reuse_calls,
        "actual_kv_reuse_rate_nonfirst": reuse_calls / nonfirst if nonfirst else 0.0,
        "avg_reusable_candidates": float(cache.get("avg_reusable_candidates", 0.0)),
        "avg_removed_final": float(cache.get("avg_removed_final", 0.0)),
        "full_token_layers": int(cache.get("full_token_layers", 0)),
        "computed_token_layers": int(cache.get("computed_token_layers", 0)),
        "token_layer_reduction_pct": 100.0
        * float(cache.get("token_layer_reduction", 0.0)),
    }


def _validate(rows: list[dict[str, object]], expected_episodes: int | None) -> None:
    by_mode = {str(row["mode"]): row for row in rows}
    if set(by_mode) != set(MODES):
        raise RuntimeError(f"expected modes={MODES}, found={tuple(by_mode)}")
    episodes = {int(row["episodes"]) for row in rows}
    if len(episodes) != 1:
        raise RuntimeError(f"episode counts differ across modes: {episodes}")
    if expected_episodes is not None and episodes != {expected_episodes}:
        raise RuntimeError(
            f"expected {expected_episodes} episodes per mode, found={episodes}"
        )
    renderers = {str(row["renderer"]) for row in rows}
    if len(renderers) != 1:
        raise RuntimeError(f"renderer backends differ across modes: {renderers}")

    off = by_mode["off"]
    matched = by_mode["matched_full"]
    reuse = by_mode["reuse"]
    if int(off["cache_calls"]) != 0:
        raise RuntimeError("off control unexpectedly entered the cache runtime")
    if int(matched["cache_calls"]) <= 0:
        raise RuntimeError("matched_full did not enter the cache runtime")
    if int(matched["actual_kv_reuse_calls"]) != 0:
        raise RuntimeError("matched_full performed K/V reuse")
    if abs(float(matched["token_layer_reduction_pct"])) > 1e-9:
        raise RuntimeError("matched_full reduced token-layer computation")
    if int(reuse["cache_calls"]) <= 0:
        raise RuntimeError("reuse did not enter the cache runtime")
    if int(reuse["actual_kv_reuse_calls"]) <= 0:
        raise RuntimeError(
            "reuse completed without any actual K/V reuse; inspect the threshold and trace"
        )
    if float(reuse["token_layer_reduction_pct"]) <= 0.0:
        raise RuntimeError("reuse did not reduce computed token-layers")


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    by_mode = {str(row["mode"]): row for row in rows}
    off_ms = float(by_mode["off"]["avg_policy_step_latency_ms"])
    matched_ms = float(by_mode["matched_full"]["avg_policy_step_latency_ms"])
    reuse_ms = float(by_mode["reuse"]["avg_policy_step_latency_ms"])
    lines = [
        "# Seer VLA-Cache campaign summary",
        "",
        "| Mode | Episodes | SR (%) | Policy (ms) | Full forward (ms) | Reuse calls | Reuse rate (%) | Token-layer reduction (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {episodes} | {success_rate_pct:.3f} | "
            "{avg_policy_step_latency_ms:.3f} | {avg_full_forward_latency_ms:.3f} | "
            "{actual_kv_reuse_calls} | {reuse_rate:.3f} | "
            "{token_layer_reduction_pct:.3f} |".format(
                **row,
                reuse_rate=100.0 * float(row["actual_kv_reuse_rate_nonfirst"]),
            )
        )
    lines.extend(
        [
            "",
            "## Derived comparisons",
            "",
            f"- `off / reuse` policy-latency speedup: {off_ms / reuse_ms:.4f}x"
            if reuse_ms > 0
            else "- `off / reuse` policy-latency speedup: unavailable",
            f"- `matched_full / reuse` policy-latency speedup: {matched_ms / reuse_ms:.4f}x"
            if reuse_ms > 0
            else "- `matched_full / reuse` policy-latency speedup: unavailable",
            "- `off` is native Seer. `matched_full` uses the indexed cache backend but recomputes every token. `reuse` is the actual VLA-Cache treatment.",
            "- Token-layer reduction counts Seer's causal action-transformer token computations; it does not claim that the MAE encoder, Perceiver, image decoder, action head, or simulator were skipped.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()

    rows = []
    for mode in MODES:
        path, summary = _load_summary(args.campaign_root, mode)
        rows.append(_row(mode, path, summary))
    _validate(rows, args.expected_episodes)

    output_dir = args.campaign_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vla_cache_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = output_dir / "vla_cache_comparison.md"
    _write_markdown(markdown_path, rows)
    print(f"[VERIFY][PASS] {markdown_path}")


if __name__ == "__main__":
    main()
