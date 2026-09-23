"""Hard completion and integrity gate for the deterministic OSMesa R5 cache."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from methods.latentloop.training.query_cache_dataset import (
    iter_query_records,
    load_manifest,
    validate_query_cache,
)
from methods.simvla_exact_q2.dataset import build_exact_q2_index, sha256_file


EXPECTED_RENDERER = {
    "GALLIUM_DRIVER": "llvmpipe",
    "LIBGL_ALWAYS_SOFTWARE": "true",
    "LP_NUM_THREADS": "0",
    "MUJOCO_GL": "osmesa",
    "PYOPENGL_PLATFORM": "osmesa",
}


def _assignment(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}=(.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"launcher does not define {name}")
    return match.group(1).strip()


def discover_pipeline_paths(launcher: str | Path) -> dict[str, str]:
    """Derive actual cache/result paths from the completed launcher contract."""

    launcher_path = Path(launcher).resolve()
    text = launcher_path.read_text(encoding="utf-8")
    shared_root = _assignment(text, "SHARED_ROOT")
    tag_expression = _assignment(text, "TAG")
    match = re.fullmatch(r"\$\{TAG:-([^}]+)\}", tag_expression)
    if match is None:
        raise ValueError(f"unsupported TAG expression: {tag_expression}")
    tag = match.group(1)
    experiment_root = Path(shared_root) / "results" / "simvla" / "latentloop" / tag
    return {
        "launcher": str(launcher_path),
        "shared_root": shared_root,
        "tag": tag,
        "experiment_root": str(experiment_root),
        "cache_root": str(experiment_root / "cache"),
        "pipeline_root": str(experiment_root / "cache_pipeline"),
        "r1_production_cache": str(experiment_root / "cache" / "query_v3_r1_full_10x20"),
        "r5_production_cache": str(experiment_root / "cache" / "query_v3_r5_full_10x20"),
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_row(summary: dict[str, Any], phase: str, horizon: str) -> dict[str, Any]:
    row = summary[phase][horizon]
    return {
        "cache_dir": row["cache_dir"],
        "episodes": int(row["episodes"]),
        "records": int(row["records"]),
        "execution_horizon": int(row["phase_config"]["execution_horizon"]),
        "renderer": row["phase_config"]["render_backend"],
        "validation_passed": bool(row["validation"]["passed"]),
        "errors": list(row["validation"]["errors"]),
    }


def audit_completed_osmesa_pipeline(
    launcher: str | Path,
    *,
    verify_shard_hashes: bool = True,
) -> dict[str, Any]:
    """Apply all fourteen exact-q2 cache gates and return audit evidence."""

    paths = discover_pipeline_paths(launcher)
    pipeline_root = Path(paths["pipeline_root"])
    cache_root = Path(paths["cache_root"])
    pipeline = _load(pipeline_root / "cache_pipeline_summary.json")
    osmesa = _load(pipeline_root / "osmesa_cache_pipeline_summary.json")
    phases = {
        "r1_smoke": _phase_row(pipeline, "smoke", "r1"),
        "r5_smoke": _phase_row(pipeline, "smoke", "r5"),
        "r1_pilot": _phase_row(pipeline, "pilot", "r1"),
        "r5_pilot": _phase_row(pipeline, "pilot", "r5"),
        "r1_production": _phase_row(pipeline, "production", "r1"),
        "r5_production": _phase_row(pipeline, "production", "r5"),
    }
    phase_completion = all(
        row["validation_passed"] and not row["errors"] and row["renderer"] == "osmesa"
        for row in phases.values()
    )
    r5 = Path(paths["r5_production_cache"])
    manifest = load_manifest(r5)
    cache_validation = validate_query_cache(r5, verify_shard_hashes=verify_shard_hashes)
    dataset_manifest, split = build_exact_q2_index(r5)

    renderer_errors: list[str] = []
    provenance_signatures: set[tuple[Any, ...]] = set()
    task_episodes: set[tuple[int, str]] = set()
    for record in iter_query_records(r5):
        provenance = record["provenance"]
        if provenance.get("render_backend") != "osmesa":
            renderer_errors.append(
                f"{record['episode_id']} q{record['query_index']} is not OSMesa"
            )
        task_episodes.add((int(record["task_id"]), str(record["episode_id"])))
        provenance_signatures.add(
            (
                provenance.get("checkpoint"),
                provenance.get("norm_stats"),
                provenance.get("experiment_seed"),
                provenance.get("render_backend"),
            )
        )
    expected_episodes = {
        (task, f"task{task:02d}_trial{trial:03d}")
        for task in range(10)
        for trial in range(20)
    }
    source_lock = _load(r5 / "source_lock.json")
    worker_signatures: set[tuple[Any, ...]] = set()
    for part in manifest["metadata"]["parts"]:
        lock = _load(r5 / part["path"] / "source_lock.json")
        worker_signatures.add(
            (
                lock["root_commit"],
                lock["simvla_upstream_commit"],
                lock["norm_stats_sha256"],
                lock["checkpoint"]["revision"],
                lock["checkpoint"]["hf_blob_key_sha256"],
            )
        )
    source_hashes_match = len(worker_signatures) == 1 and next(iter(worker_signatures)) == (
        source_lock["root_commit"],
        source_lock["simvla_upstream_commit"],
        source_lock["norm_stats_sha256"],
        source_lock["checkpoint"]["revision"],
        source_lock["checkpoint"]["hf_blob_key_sha256"],
    )
    tuples = dataset_manifest["tuples"]
    tuple_ids = [row["tuple_id"] for row in tuples]
    native = dataset_manifest["native_semantics"]
    checks = {
        "01_osmesa_in_every_shard": not renderer_errors
        and manifest["metadata"]["protocol"]["render_backend"] == "osmesa"
        and osmesa["renderer"] == EXPECTED_RENDERER,
        "02_all_libero10_tasks_present": sorted({task for task, _ in task_episodes})
        == list(range(10)),
        "03_production_coverage_complete": task_episodes == expected_episodes
        and int(manifest["total_records"]) == 12_509,
        "04_episode_query_indices_contiguous": bool(cache_validation["passed"]),
        "05_tuple_same_task_episode": all(
            row["episode_id"].startswith(f"task{int(row['task_id']):02d}_") for row in tuples
        ),
        "06_q0_q1_q2_observations_exist": int(dataset_manifest["tuple_count"]) > 0,
        "07_q0_q1_q2_conditions_exist": int(dataset_manifest["tuple_count"]) > 0,
        "08_x0_x1_exist": int(dataset_manifest["tuple_count"]) > 0,
        "09_native_executed_subchunk_length": int(native["execution_horizon_R"]) == 5,
        "10_explicit_q1_q2_noise": all(
            len(row["epsilon1_sha256"]) == 64 and len(row["epsilon2_sha256"]) == 64
            for row in tuples
        ),
        "11_condition_action_tensors_finite": bool(cache_validation["passed"]),
        "12_source_and_environment_hashes_match": source_hashes_match
        and len(provenance_signatures) == 1,
        "13_no_duplicate_tuple": len(tuple_ids) == len(set(tuple_ids)),
        "14_no_episode_boundary_crossing": bool(cache_validation["passed"]),
    }
    errors = list(cache_validation["errors"]) + renderer_errors
    if not phase_completion:
        errors.append("one or more smoke/pilot/production phases did not complete cleanly")
    verdict = (
        "R5_EXACT_Q2_CACHE_GATE_PASS"
        if phase_completion and all(checks.values()) and not errors
        else "R5_EXACT_Q2_CACHE_GATE_FAIL"
    )
    return {
        "schema_version": "simvla_exact_q2_cache_audit_v1",
        "experiment_identifier": "simvla_r5_exact_q2_regeneration",
        "verdict": verdict,
        "pipeline_completion_verdict": osmesa["verdict"],
        "paths": paths,
        "phases": phases,
        "checks": checks,
        "errors": errors,
        "r5": {
            "manifest_sha256": sha256_file(r5 / "manifest.json"),
            "records": int(manifest["total_records"]),
            "shards": len(manifest["shards"]),
            "episodes": len(task_episodes),
            "tuple_count": int(dataset_manifest["tuple_count"]),
            "train_tuples": int(split["train_tuple_count"]),
            "validation_tuples": int(split["validation_tuple_count"]),
            "source_signature": dataset_manifest["source_signature"],
        },
        "renderer": dict(EXPECTED_RENDERER),
        "pipeline_elapsed_seconds": float(osmesa["pipeline_elapsed_seconds"]),
    }
