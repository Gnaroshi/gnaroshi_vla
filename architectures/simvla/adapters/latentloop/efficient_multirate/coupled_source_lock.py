"""Source and artifact identity for the SimVLA condition-code coupling lane."""

from __future__ import annotations

import hashlib
import json
import subprocess
import argparse
from pathlib import Path
from typing import Any

from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_contracts import (
    FROZEN_CONDITION_CHECKPOINT_SHA256,
    FROZEN_CONDITION_SOURCE_SHA256,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FROZEN_GENERATION_CHECKPOINT_SHA256,
    FROZEN_GENERATION_SOURCE_SHA256,
    FROZEN_NORM_STATS_SHA256,
)


ROOT = Path(__file__).resolve().parents[5]
SOURCE_FILES = (
    "architectures/simvla/adapters/latentloop/efficient_multirate/coupled_condition_generation.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/coupled_condition_parity.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/coupled_generation_offline.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/coupled_generation_train.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/coupled_source_lock.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/fixed_2x2_aggregate.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/fixed_2x2_eval.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/kc_frontier_aggregate.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/kc_frontier_contracts.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/joint_nfe_aggregate.py",
    "architectures/simvla/adapters/latentloop/efficient_multirate/row_postprocess_recovery.py",
    "architectures/simvla/wrappers/run_coupled_condition_generation.sh",
    "architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh",
    "architectures/simvla/wrappers/run_kc_efficiency_frontier_sd1.sh",
    "architectures/simvla/wrappers/run_joint_nfe_frontier_sd1.sh",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(ROOT), *args), text=True, stderr=subprocess.STDOUT
    ).strip()


def build_coupled_source_lock(
    *,
    parent_generation_checkpoint: str | Path,
    condition_checkpoint: str | Path,
    norm_stats: str | Path,
    exact_cache: str | Path,
) -> dict[str, Any]:
    files = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    cache_manifest = Path(exact_cache).expanduser().resolve() / "manifest.json"
    payload: dict[str, Any] = {
        "schema_version": "simvla_condition_generation_coupling_source_v1",
        "root_commit": _git("rev-parse", "HEAD"),
        "upstream_commit": _git(
            "-C", str(ROOT / "architectures" / "simvla" / "upstream"), "rev-parse", "HEAD"
        ),
        "source_file_sha256": files,
        "parent_generation_source_combined_sha256": FROZEN_GENERATION_SOURCE_SHA256,
        "condition_source_combined_sha256": FROZEN_CONDITION_SOURCE_SHA256,
        "parent_generation_checkpoint_sha256": sha256_file(
            parent_generation_checkpoint
        ),
        "condition_checkpoint_sha256": sha256_file(condition_checkpoint),
        "norm_stats_sha256": sha256_file(norm_stats),
        "exact_cache_manifest_sha256": sha256_file(cache_manifest),
    }
    expected = {
        "parent_generation_checkpoint_sha256": FROZEN_GENERATION_CHECKPOINT_SHA256,
        "condition_checkpoint_sha256": FROZEN_CONDITION_CHECKPOINT_SHA256,
        "norm_stats_sha256": FROZEN_NORM_STATS_SHA256,
    }
    mismatches = {
        name: {"expected": value, "observed": payload[name]}
        for name, value in expected.items()
        if payload[name] != value
    }
    if mismatches:
        raise RuntimeError(f"coupled parent artifact mismatch: {mismatches}")
    payload["combined_sha256"] = _canonical_sha256(payload)
    return payload


def verify_coupled_source_lock(
    source: dict[str, Any],
    *,
    parent_generation_checkpoint: str | Path,
    condition_checkpoint: str | Path,
    norm_stats: str | Path,
    exact_cache: str | Path,
) -> dict[str, Any]:
    observed = build_coupled_source_lock(
        parent_generation_checkpoint=parent_generation_checkpoint,
        condition_checkpoint=condition_checkpoint,
        norm_stats=norm_stats,
        exact_cache=exact_cache,
    )
    checks = {
        "schema_version": source.get("schema_version") == observed["schema_version"],
        "combined_sha256": source.get("combined_sha256")
        == observed["combined_sha256"],
        "root_commit": source.get("root_commit") == observed["root_commit"],
        "upstream_commit": source.get("upstream_commit") == observed["upstream_commit"],
        "source_files": source.get("source_file_sha256")
        == observed["source_file_sha256"],
        "parent_generation_checkpoint": source.get(
            "parent_generation_checkpoint_sha256"
        )
        == observed["parent_generation_checkpoint_sha256"],
        "condition_checkpoint": source.get("condition_checkpoint_sha256")
        == observed["condition_checkpoint_sha256"],
        "norm_stats": source.get("norm_stats_sha256")
        == observed["norm_stats_sha256"],
        "exact_cache": source.get("exact_cache_manifest_sha256")
        == observed["exact_cache_manifest_sha256"],
    }
    return {
        "verdict": "COUPLED_SOURCE_LOCK_PASS" if all(checks.values()) else "COUPLED_SOURCE_LOCK_FAIL",
        "checks": checks,
        "expected": source,
        "observed": observed,
    }


def prepare_eval_provenance(
    *,
    base_fixed_source_lock: str | Path,
    base_control_manifest: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing existing provenance output: {destination}")
    destination.mkdir(parents=True)
    fixed = json.loads(Path(base_fixed_source_lock).read_text(encoding="utf-8"))
    fixed["schema_version"] = "simvla_coupled_fixed_eval_source_lock_v1"
    fixed["root_commit"] = _git("rev-parse", "HEAD")
    locked_names = set(fixed.get("file_sha256", {})) | set(SOURCE_FILES)
    fixed["file_sha256"] = {
        name: sha256_file(ROOT / name) for name in sorted(locked_names)
    }
    control = json.loads(Path(base_control_manifest).read_text(encoding="utf-8"))
    control_files = dict(control.get("control_file_sha256", {}))
    for name in tuple(control_files):
        path = ROOT / name
        if path.is_file():
            control_files[name] = sha256_file(path)
    control["control_file_sha256"] = control_files
    fixed_path = destination / "fixed_eval_source_lock.json"
    control_path = destination / "control_manifest.json"
    fixed_path.write_text(
        json.dumps(fixed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    control_path.write_text(
        json.dumps(control, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "verdict": "COUPLED_EVAL_PROVENANCE_PREPARED",
        "root_commit": fixed["root_commit"],
        "fixed_eval_source_lock": str(fixed_path),
        "control_manifest": str(control_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-fixed-source-lock", required=True)
    parser.add_argument("--base-control-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = prepare_eval_provenance(
        base_fixed_source_lock=args.base_fixed_source_lock,
        base_control_manifest=args.base_control_manifest,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
