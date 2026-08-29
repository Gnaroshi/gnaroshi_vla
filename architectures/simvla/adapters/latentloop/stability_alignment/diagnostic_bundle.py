"""Build a provenance-locked rb2 bundle from a failed offline checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    BUNDLE_SCHEMA,
    atomic_write_json,
    load_json,
    sha256_file,
)


READY_NAME = "READY_SHORT_DIAGNOSTIC_FOR_RB2.json"
READY_VERDICT = "READY_SHORT_DIAGNOSTIC_FOR_RB2"


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, destination)


def build(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    offline_gate = Path(args.offline_gate).expanduser().resolve()
    norm_stats = Path(args.norm_stats).expanduser().resolve()
    loss_weights = Path(args.loss_weights).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    gate = load_json(offline_gate)
    if gate.get("verdict") != "STABILITY_10K_GATE_FAIL":
        raise RuntimeError("diagnostic bundle requires STABILITY_10K_GATE_FAIL")
    if bool(gate.get("gate", {}).get("passed")):
        raise RuntimeError("diagnostic bundle must not contain a passing gate")
    if int(gate.get("optimizer_step", -1)) != 10_000:
        raise RuntimeError("diagnostic bundle requires the 10K checkpoint")

    checkpoint_sha = sha256_file(checkpoint)
    if output.exists():
        ready_path = output / READY_NAME
        if ready_path.is_file():
            ready = load_json(ready_path)
            if ready.get("checkpoint_sha256") == checkpoint_sha:
                return ready
        raise FileExistsError(f"refusing incompatible diagnostic bundle: {output}")

    output.mkdir(parents=True)
    _copy(checkpoint, output / "stability_aligned_s50_10k.pt")
    _copy(offline_gate, output / "offline_gate.json")
    _copy(norm_stats, output / "libero_norm.json")
    _copy(loss_weights, output / "stability_alignment_loss_weights.json")

    train_root = checkpoint.parents[1]
    for name in (
        "source_lock.json",
        "training_contract.json",
        "determinism.json",
        "parameter_audit.json",
    ):
        _copy(train_root / name, output / name)

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    atomic_write_json(
        output / "parent_identity.json",
        dict(checkpoint_payload.get("parent_identity", {})),
    )
    failed_checks = sorted(
        name
        for name, passed in gate.get("gate", {}).get("checks", {}).items()
        if not bool(passed)
    )
    atomic_write_json(
        output / "diagnostic_contract.json",
        {
            "schema_version": "simvla_s50_10k_diagnostic_bundle_v1",
            "classification": "DIAGNOSTIC_ONLY",
            "branch": "S50",
            "optimizer_step": 10_000,
            "offline_gate_verdict": gate["verdict"],
            "offline_gate_passed": False,
            "failed_offline_checks": failed_checks,
            "online_result_must_not_be_reclassified_as_gate_passing": True,
        },
    )

    files = sorted(path for path in output.iterdir() if path.is_file())
    atomic_write_json(
        output / "SHA256_MANIFEST.json",
        {path.name: sha256_file(path) for path in files},
    )
    ready = {
        "schema_version": BUNDLE_SCHEMA,
        "verdict": READY_VERDICT,
        "classification": "DIAGNOSTIC_ONLY",
        "diagnostic_only": True,
        "offline_gate_passed": False,
        "offline_gate_verdict": gate["verdict"],
        "selected_branch": "S50",
        "checkpoint": "stability_aligned_s50_10k.pt",
        "checkpoint_sha256": checkpoint_sha,
        "optimizer_step": 10_000,
        "kc3_offline_ready": False,
        "kc4_offline_ready": False,
        "generation_ng3_preserved": True,
        "n_g": 3,
        "full_generation_indices": [0, 4, 8],
    }
    atomic_write_json(output / READY_NAME, ready)
    return ready


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--offline-gate", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--loss-weights", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
