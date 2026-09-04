"""Create a fail-closed baseline/LatentLoop real deployment manifest."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from architectures.simvla.adapters.latentloop_real_deploy.contracts import (
    load_deployment_contract,
    sha256_directory,
)

from .io_utils import atomic_write_json, sha256_file
from .model_io import load_real_action_payload, official_base_identity
from .updater_io import load_real_updater


def build(args: argparse.Namespace) -> dict[str, Any]:
    template_path = Path(args.template).expanduser().resolve()
    payload = deepcopy(json.loads(template_path.read_text(encoding="utf-8")))
    official = Path(args.checkpoint).expanduser().resolve()
    processor = Path(args.processor).expanduser().resolve()
    norm = Path(args.norm_stats).expanduser().resolve()
    baseline = Path(args.baseline_action_checkpoint).expanduser().resolve()
    condition = Path(args.condition_updater).expanduser().resolve()
    generation = Path(args.generation_updater).expanduser().resolve()
    base_identity = official_base_identity(official, processor)
    baseline_payload = load_real_action_payload(baseline)
    if (
        baseline_payload["official_base"]["model_weights_sha256"]
        != base_identity.model_weights_sha256
    ):
        raise ValueError("real baseline was not initialized from the selected official checkpoint")
    if baseline_payload["norm_stats_sha256"] != sha256_file(norm):
        raise ValueError("real baseline and deployment norm statistics differ")
    baseline_sha = sha256_file(baseline)
    load_real_updater(
        condition,
        kind="condition",
        device="cpu",
        expected_baseline_sha256=baseline_sha,
    )
    load_real_updater(
        generation,
        kind="generation",
        device="cpu",
        expected_baseline_sha256=baseline_sha,
    )
    weights = official / "model.safetensors"
    payload["deployment_id"] = args.deployment_id
    payload["artifacts"] = {
        "official_base_model_directory": {
            "path": str(official),
            "sha256": sha256_directory(official),
        },
        "official_base_model_weights": {
            "path": str(weights),
            "sha256": sha256_file(weights),
        },
        "processor_directory": {
            "path": str(processor),
            "sha256": sha256_directory(processor),
        },
        "norm_stats": {"path": str(norm), "sha256": sha256_file(norm)},
        "real_action_transformer": {
            "path": str(baseline),
            "sha256": baseline_sha,
        },
        "condition_updater": {
            "path": str(condition),
            "sha256": sha256_file(condition),
        },
        "generation_updater": {
            "path": str(generation),
            "sha256": sha256_file(generation),
        },
    }
    payload["pairing"] = {
        "official_base_model_identity": base_identity.model_weights_sha256,
        "real_baseline_identity": baseline_sha,
        "condition_source_real_baseline_identity": baseline_sha,
        "generation_source_real_baseline_identity": baseline_sha,
    }
    payload["runtime"]["instructions"] = [args.instruction]
    payload["safety_review"] = {
        "live_authorized": False,
        "model_preflight_passed": False,
        "read_only_profile_passed": False,
        "approved_by": "",
        "approved_at": "",
    }
    output = atomic_write_json(args.output, payload)
    contract = load_deployment_contract(output, verify_artifacts=True)
    return {
        "verdict": "REAL_DEPLOYMENT_MANIFEST_PASS",
        "path": str(output),
        "deployment_id": contract.deployment_id,
        "official_base_sha256": base_identity.model_weights_sha256,
        "real_baseline_sha256": baseline_sha,
        "live_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--template", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--processor", required=True)
    result.add_argument("--norm-stats", required=True)
    result.add_argument("--baseline-action-checkpoint", required=True)
    result.add_argument("--condition-updater", required=True)
    result.add_argument("--generation-updater", required=True)
    result.add_argument("--deployment-id", required=True)
    result.add_argument("--instruction", required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> int:
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

