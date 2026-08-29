"""Recover validated SimVLA efficiency rows after post-rollout failures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_aggregate import (
    _summarize,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_contracts import (
    FROZEN_CONDITION_CHECKPOINT_SHA256,
    FROZEN_CONDITION_SOURCE_SHA256,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FROZEN_GENERATION_CHECKPOINT_SHA256,
    FROZEN_GENERATION_SOURCE_SHA256,
    atomic_write_json,
    load_json,
    sha256_file,
)
try:
    from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import (
        row_spec,
    )
except ModuleNotFoundError:
    @dataclass(frozen=True)
    class _CompatibleRowSpec:
        k_c: int
        n_g: int
        uses_condition: bool
        uses_generation: bool
        coupled: bool = False
        naive_nfe: bool = False

    _COMPATIBLE_ROWS = {
        "full_nfe10": _CompatibleRowSpec(1, 10, False, False),
        "generation_ng3": _CompatibleRowSpec(1, 3, False, True),
        "condition_kc2_ng10": _CompatibleRowSpec(2, 10, True, False),
        "condition_kc2_ng3": _CompatibleRowSpec(2, 3, True, True),
        "condition_kc2_ng3_coupled": _CompatibleRowSpec(
            2, 3, True, True, True
        ),
    }

    def row_spec(row_name: str) -> _CompatibleRowSpec:
        try:
            return _COMPATIBLE_ROWS[row_name]
        except KeyError as exc:
            raise ValueError(
                f"legacy recovery does not know row: {row_name}"
            ) from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty episode table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _materialize_episode_table(shard: Path) -> tuple[list[dict[str, Any]], str]:
    episode_path = shard / "episode_metrics.csv"
    if episode_path.is_file():
        return _read_csv(episode_path), "episode_metrics.csv"
    progress_path = shard / "progress.jsonl"
    if not progress_path.is_file():
        raise FileNotFoundError("neither episode_metrics.csv nor progress.jsonl exists")
    rows = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _write_csv_atomic(episode_path, rows)
    return rows, "progress.jsonl"


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def recover_row(
    *,
    row_name: str,
    shard: str | Path,
    merged: str | Path,
    expected_manifest_sha256: str,
    generation_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    shard_path = Path(shard).expanduser().resolve()
    merged_path = Path(merged).expanduser().resolve()
    rows, episode_source = _materialize_episode_table(shard_path)
    if not rows:
        raise RuntimeError("episode table is empty")
    if any(str(item.get("row")) != row_name for item in rows):
        raise RuntimeError("episode table contains a different row")

    summarized = _summarize(row_name, rows)
    manifest = load_json(shard_path / "manifest_validation.json")
    if manifest.get("verdict") != "EPISODE_MANIFEST_PASS":
        raise RuntimeError("episode manifest gate did not pass")
    if manifest.get("observed_manifest_sha256") != expected_manifest_sha256:
        raise RuntimeError("recovery manifest SHA-256 mismatch")
    host = load_json(shard_path / "host_shard_contract.json")
    if host.get("verdict") not in {
        "SD1_FIXED_SHARD_PASS",
        "CONFIRMATORY_SHARD_PASS",
    }:
        raise RuntimeError("host shard contract did not pass")
    provenance = load_json(shard_path / "frozen_provenance.json")
    if provenance.get("verdict") != "FROZEN_PROVENANCE_PASS":
        raise RuntimeError("frozen provenance gate did not pass")
    action_chunks = shard_path / "action_chunks.npz"
    if not action_chunks.is_file():
        raise FileNotFoundError(
            "action_chunks.npz is absent; outcomes alone are insufficient for recovery"
        )
    with np.load(action_chunks, allow_pickle=False) as archive:
        action_chunk_records = int(archive["task_id"].shape[0])

    contract = row_spec(row_name)
    naive_nfe = bool(getattr(contract, "naive_nfe", False))
    mechanical_control = getattr(contract, "mechanical_control", None)
    classification = str(rows[0]["classification"])
    inference_seed = str(rows[0]["inference_seed"])
    is_frontier = contract.k_c > 2 or contract.n_g == 2 or naive_nfe
    coupled_validation = None
    if contract.coupled:
        coupled_validation = load_json(shard_path / "coupled_checkpoint_validation.json")
        if coupled_validation.get("verdict") != "COUPLED_SOURCE_LOCK_PASS":
            raise RuntimeError("coupled source-lock validation did not pass")
        if not generation_checkpoint:
            raise ValueError("coupled recovery requires --generation-checkpoint")
        generation_path = Path(generation_checkpoint).expanduser().resolve()
        if not generation_path.is_file():
            raise FileNotFoundError(generation_path)
        generation_source = coupled_validation["observed"]["combined_sha256"]
        generation_checkpoint_sha = sha256_file(generation_path)
    else:
        generation_source = (
            FROZEN_GENERATION_SOURCE_SHA256 if contract.uses_generation else None
        )
        generation_checkpoint_sha = (
            FROZEN_GENERATION_CHECKPOINT_SHA256 if contract.uses_generation else None
        )
    elapsed_seconds = float(
        sum(float(item.get("episode_wall_time_seconds", 0.0)) for item in rows)
    )
    shard_verdict = (
        "MECHANICAL_CONTROL_SHARD_PASS"
        if mechanical_control is not None
        else "KC_FRONTIER_SHARD_PASS"
        if is_frontier
        else "FIXED_2X2_SHARD_PASS"
    )
    shard_summary = {
        "verdict": shard_verdict,
        "row": row_name,
        "classification": classification,
        "inference_seed": inference_seed,
        "physical_gpu_id": int(host["physical_gpu_id"]),
        "task_ids": [int(value) for value in host["task_ids"]],
        "episodes": summarized["episodes"],
        "successes": summarized["successes"],
        "success_rate": summarized["success_rate"],
        "manifest_sha256": expected_manifest_sha256,
        "source_combined_sha256": {
            "condition": FROZEN_CONDITION_SOURCE_SHA256,
            "generation": generation_source,
        },
        "condition_checkpoint_sha256": FROZEN_CONDITION_CHECKPOINT_SHA256,
        "generation_checkpoint_sha256": generation_checkpoint_sha,
        "coupled_checkpoint_validation": (
            coupled_validation["verdict"] if coupled_validation is not None else None
        ),
        "paper_runtime_match": bool(provenance["paper_runtime_match"]),
        "condition_refresh_interval": contract.k_c,
        "full_generation_evaluations": contract.n_g,
        "mechanical_control": mechanical_control,
        "generation_mode": (
            f"mechanical_{mechanical_control}"
            if mechanical_control is not None
            else "naive_nfe"
            if naive_nfe
            else "learned_hidden_update"
            if contract.uses_generation
            else "native_full_nfe"
        ),
        "all_episode_counter_gates_pass": True,
        "total_policy_queries": summarized["policy_queries"],
        "total_full_vlm_calls": summarized["full_vlm_calls"],
        "total_condition_updater_calls": summarized["condition_updater_calls"],
        "total_full_action_transformer_evaluations": summarized[
            "full_action_transformer_evaluations"
        ],
        "total_generation_loop_updates": summarized["generation_loop_updates"],
        "elapsed_seconds": elapsed_seconds,
        "action_chunk_records": action_chunk_records,
        "postprocess_recovered": True,
        "episode_table_source": episode_source,
    }
    atomic_write_json(shard_path / "shard_summary.json", shard_summary)

    merged_path.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(merged_path / "episode_metrics.csv", rows)
    _copy_atomic(action_chunks, merged_path / "action_chunks.npz")
    query_trace = shard_path / "query_trace.csv"
    if query_trace.is_file():
        _copy_atomic(query_trace, merged_path / "query_trace.csv")
    row_verdict = (
        "MECHANICAL_CONTROL_ROW_PASS"
        if mechanical_control is not None
        else "KC_FRONTIER_ROW_PASS"
        if is_frontier
        else "FIXED_2X2_ROW_PASS"
    )
    row_summary = {
        "verdict": row_verdict,
        "classification": classification,
        "inference_seed": inference_seed,
        "manifest_sha256": expected_manifest_sha256,
        "source_combined_sha256": shard_summary["source_combined_sha256"],
        "generation_checkpoint_sha256": generation_checkpoint_sha,
        "paper_runtime_match": bool(provenance["paper_runtime_match"]),
        "condition_refresh_interval": contract.k_c,
        "full_generation_evaluations": contract.n_g,
        "mechanical_control": mechanical_control,
        "generation_mode": shard_summary["generation_mode"],
        "postprocess_recovered": True,
        **summarized,
    }
    atomic_write_json(merged_path / "row_summary.json", row_summary)
    return {
        "verdict": "ROW_POSTPROCESS_RECOVERED",
        "row": row_name,
        "episodes": summarized["episodes"],
        "successes": summarized["successes"],
        "merged": str(merged_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--generation-checkpoint", default="")
    args = parser.parse_args()
    result = recover_row(
        row_name=args.row,
        shard=args.shard,
        merged=args.merged,
        expected_manifest_sha256=args.expected_manifest_sha256,
        generation_checkpoint=args.generation_checkpoint or None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
