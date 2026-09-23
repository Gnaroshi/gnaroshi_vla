"""Build immutable diagnostic bundles for V3 learning-trajectory checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    BUNDLE_SCHEMA,
    atomic_write_json,
    canonical_sha256,
    load_json,
    sha256_file,
)


READY_NAME = "READY_SHORT_DIAGNOSTIC_FOR_RB2.json"
READY_VERDICT = "READY_SHORT_DIAGNOSTIC_FOR_RB2"


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def build(
    *,
    checkpoint: str | Path,
    offline_gate: str | Path,
    norm_stats: str | Path,
    training_root: str | Path,
    output: str | Path,
    branch: str,
    optimizer_step: int,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    gate_path = Path(offline_gate).expanduser().resolve()
    norm_path = Path(norm_stats).expanduser().resolve()
    train_root = Path(training_root).expanduser().resolve()
    output_root = Path(output).expanduser().resolve()
    step = int(optimizer_step)

    gate = load_json(gate_path)
    if str(gate.get("branch")) != str(branch):
        raise RuntimeError("V3 diagnostic gate branch changed")
    if int(gate.get("optimizer_step", -1)) != step:
        raise RuntimeError("V3 diagnostic gate optimizer step changed")
    checkpoint_sha = sha256_file(checkpoint_path)
    if gate.get("candidate_sha256") != checkpoint_sha:
        raise RuntimeError("V3 diagnostic gate and checkpoint hashes differ")

    ready_path = output_root / READY_NAME
    if ready_path.is_file():
        ready = load_json(ready_path)
        if (
            ready.get("checkpoint_sha256") == checkpoint_sha
            and int(ready.get("optimizer_step", -1)) == step
        ):
            return ready
        raise FileExistsError(f"incompatible V3 diagnostic bundle: {output_root}")
    if output_root.exists():
        raise FileExistsError(f"incomplete V3 diagnostic bundle exists: {output_root}")

    output_root.mkdir(parents=True)
    checkpoint_name = f"stability_v3_{str(branch).lower()}_step_{step:06d}.pt"
    _copy(checkpoint_path, output_root / checkpoint_name)
    _copy(gate_path, output_root / "offline_gate.json")
    _copy(norm_path, output_root / "libero_norm.json")
    for name in (
        "source_lock.json",
        "training_contract.json",
        "determinism.json",
        "parameter_audit.json",
    ):
        source = train_root / name
        if source.is_file():
            _copy(source, output_root / name)

    failed_checks = sorted(
        name
        for name, passed in gate.get("gate", {}).get("checks", {}).items()
        if not bool(passed)
    )
    diagnostic_contract = {
        "schema_version": "simvla_stability_v3_trajectory_bundle_v1",
        "classification": "DIAGNOSTIC_ONLY",
        "method_version": "stability_v3",
        "branch": str(branch),
        "optimizer_step": step,
        "offline_gate_verdict": gate.get("verdict"),
        "offline_gate_passed": bool(gate.get("passed")),
        "failed_offline_checks": failed_checks,
        "selection_policy": "online results must not select a checkpoint",
        "purpose": "fixed-manifest learning-trajectory diagnosis",
    }
    atomic_write_json(output_root / "diagnostic_contract.json", diagnostic_contract)

    payload_files = sorted(path for path in output_root.iterdir() if path.is_file())
    files = {path.name: sha256_file(path) for path in payload_files}
    atomic_write_json(output_root / "SHA256_MANIFEST.json", files)
    ready = {
        "schema_version": BUNDLE_SCHEMA,
        "verdict": READY_VERDICT,
        "classification": "DIAGNOSTIC_ONLY",
        "diagnostic_only": True,
        "method_version": "stability_v3",
        "offline_gate_passed": False,
        "measured_stage_gate_passed": bool(gate.get("passed")),
        "offline_gate_verdict": gate.get("verdict"),
        "selected_branch": str(branch),
        "optimizer_step": step,
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha,
        "kc3_offline_ready": False,
        "kc4_offline_ready": False,
        "online_must_not_select_checkpoint": True,
        "generation_ng3_preserved": True,
        "n_g": 3,
        "full_generation_indices": [0, 4, 8],
        "files": files,
    }
    ready["combined_sha256"] = canonical_sha256(ready)
    atomic_write_json(ready_path, ready)
    return ready


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--offline-gate", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--branch", choices=("R50", "R150"), required=True)
    parser.add_argument("--optimizer-step", type=int, required=True)
    args = parser.parse_args()
    result = build(
        checkpoint=args.checkpoint,
        offline_gate=args.offline_gate,
        norm_stats=args.norm_stats,
        training_root=args.training_root,
        output=args.output,
        branch=args.branch,
        optimizer_step=args.optimizer_step,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
