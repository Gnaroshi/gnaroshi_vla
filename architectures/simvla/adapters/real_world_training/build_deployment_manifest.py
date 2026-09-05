"""Create a fail-closed baseline/LatentLoop real deployment manifest."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from architectures.simvla.adapters.latentloop_real_deploy.contracts import (
    load_deployment_contract,
    runtime_source_identity,
    sha256_directory,
)

from .io_utils import atomic_write_json, sha256_file
from .model_io import load_real_action_payload, official_base_identity
from .updater_io import (
    audit_real_coupled_checkpoint,
    load_real_coupled_generation,
    load_real_updater,
)
from .verify_condition_cache import validate_condition_cache_attestation


def build(args: argparse.Namespace) -> dict[str, Any]:
    template_path = Path(args.template).expanduser().resolve()
    payload = deepcopy(json.loads(template_path.read_text(encoding="utf-8")))
    official = Path(args.checkpoint).expanduser().resolve()
    processor = Path(args.processor).expanduser().resolve()
    norm = Path(args.norm_stats).expanduser().resolve()
    dataset_manifest = Path(args.dataset_manifest).expanduser().resolve()
    cache_manifest = Path(args.condition_cache_manifest).expanduser().resolve()
    cache_attestation = Path(args.condition_cache_attestation).expanduser().resolve()
    baseline = Path(args.baseline_action_checkpoint).expanduser().resolve()
    condition = Path(args.condition_updater).expanduser().resolve()
    generation = Path(args.generation_updater).expanduser().resolve()
    coupled = Path(args.coupled_generation_updater).expanduser().resolve()
    base_identity = official_base_identity(official, processor)
    payload["runtime_source_identity_sha256"] = runtime_source_identity()[
        "combined_sha256"
    ]
    baseline_payload = load_real_action_payload(baseline)
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    cache_payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
    cache_attestation_payload = validate_condition_cache_attestation(
        cache_attestation,
        condition_cache=cache_manifest.parent,
        checkpoint=official,
        processor=processor,
        norm_stats=norm,
        verify_cache_array_checksums=False,
    )
    if (
        baseline_payload["official_base"]["model_weights_sha256"]
        != base_identity.model_weights_sha256
    ):
        raise ValueError("real baseline was not initialized from the selected official checkpoint")
    if baseline_payload["norm_stats_sha256"] != sha256_file(norm):
        raise ValueError("real baseline and deployment norm statistics differ")
    if int(baseline_payload.get("optimizer_step", -1)) != 3000:
        raise ValueError("real deployment baseline must be the predeclared step-3000 model")
    dataset_identity = str(dataset_payload["dataset_identity_sha256"])
    cache_identity = str(cache_payload["condition_cache_identity_sha256"])
    if baseline_payload.get("dataset_identity_sha256") != dataset_identity:
        raise ValueError("real baseline and deployment dataset identities differ")
    if cache_payload.get("dataset_identity_sha256") != dataset_identity:
        raise ValueError("condition cache and deployment dataset identities differ")
    cache_processor = Path(
        cache_payload.get("official_base", {}).get("processor_directory", "")
    ).expanduser()
    if not cache_processor.is_absolute() or cache_processor.resolve() != processor:
        raise ValueError(
            "condition cache was not produced with the selected processor directory"
        )
    if dataset_payload.get("norm_stats", {}).get("sha256") != sha256_file(norm):
        raise ValueError("dataset manifest and deployment norm statistics differ")
    if (
        baseline_payload.get("training_config", {}).get(
            "condition_cache_identity_sha256"
        )
        != cache_identity
    ):
        raise ValueError("real baseline and deployment condition cache identities differ")
    attestation_identity = str(
        cache_attestation_payload["attestation_identity_sha256"]
    )
    if (
        baseline_payload.get("training_config", {}).get(
            "condition_cache_attestation_identity_sha256"
        )
        != attestation_identity
    ):
        raise ValueError(
            "real baseline and deployment Condition cache attestations differ"
        )
    baseline_sha = sha256_file(baseline)
    norm_sha = sha256_file(norm)
    updater_expectations = {
        "expected_baseline_sha256": baseline_sha,
        "expected_norm_sha256": norm_sha,
        "expected_dataset_identity_sha256": dataset_identity,
        "expected_cache_identity_sha256": cache_identity,
        "expected_cache_attestation_identity_sha256": attestation_identity,
        "expected_optimizer_step": 10_000,
    }
    load_real_updater(
        condition,
        kind="condition",
        device="cpu",
        **updater_expectations,
    )
    load_real_updater(
        generation,
        kind="generation",
        device="cpu",
        **updater_expectations,
    )
    load_real_coupled_generation(
        coupled,
        device="cpu",
        expected_parent_generation_sha256=sha256_file(generation),
        expected_condition_updater_sha256=sha256_file(condition),
        expected_cache_manifest_sha256=sha256_file(cache_manifest),
        **updater_expectations,
    )
    projection_audit = audit_real_coupled_checkpoint(
        parent_generation_checkpoint=generation,
        coupled_generation_checkpoint=coupled,
    )
    if projection_audit["verdict"] != "PROJECTION_ONLY_STATE_PASS":
        raise ValueError("coupled checkpoint changed tensors outside the coupling projection")
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
        "norm_stats": {"path": str(norm), "sha256": norm_sha},
        "dataset_manifest": {
            "path": str(dataset_manifest),
            "sha256": sha256_file(dataset_manifest),
        },
        "condition_cache_manifest": {
            "path": str(cache_manifest),
            "sha256": sha256_file(cache_manifest),
        },
        "condition_cache_attestation": {
            "path": str(cache_attestation),
            "sha256": sha256_file(cache_attestation),
        },
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
        "coupled_generation_updater": {
            "path": str(coupled),
            "sha256": sha256_file(coupled),
        },
    }
    payload["pairing"] = {
        "official_base_model_identity": base_identity.model_weights_sha256,
        "real_baseline_identity": baseline_sha,
        "norm_stats_identity": norm_sha,
        "dataset_identity": dataset_identity,
        "condition_cache_identity": cache_identity,
        "condition_cache_attestation_identity": cache_attestation_payload[
            "attestation_identity_sha256"
        ],
        "condition_source_real_baseline_identity": baseline_sha,
        "generation_source_real_baseline_identity": baseline_sha,
        "coupled_source_real_baseline_identity": baseline_sha,
        "coupled_parent_generation_identity": sha256_file(generation),
        "coupled_condition_updater_identity": sha256_file(condition),
        "coupled_generation_identity": sha256_file(coupled),
    }
    payload["runtime"]["instructions"] = [args.instruction]
    payload["safety_review"] = {
        "live_authorized": False,
        "model_preflight_passed": False,
        "read_only_profile_passed": False,
        "hardware_configuration_reviewed": False,
        "camera_role_mapping_verified": False,
        "task_home_pose_verified": False,
        "workspace_bounds_verified": False,
        "control_limits_reviewed": False,
        "gripper_startup_behavior_reviewed": False,
        "gripper_no_software_stop_acknowledged": False,
        "physical_emergency_stop_verified": False,
        "runtime_timing_reviewed": False,
        "baseline_bounded_canary_passed": False,
        "approved_by": "",
        "approved_at": "",
    }
    output = atomic_write_json(args.output, payload)
    contract = load_deployment_contract(output, verify_artifacts=True)
    return {
        "verdict": "REAL_DEPLOYMENT_MANIFEST_TEMPLATE_READY",
        "path": str(output),
        "deployment_id": contract.deployment_id,
        "official_base_sha256": base_identity.model_weights_sha256,
        "real_baseline_sha256": baseline_sha,
        "coupled_generation_sha256": sha256_file(coupled),
        "projection_only_state_audit": projection_audit["verdict"],
        "hardware_review_required": True,
        "live_authorized": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--template", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--processor", required=True)
    result.add_argument("--norm-stats", required=True)
    result.add_argument("--dataset-manifest", required=True)
    result.add_argument("--condition-cache-manifest", required=True)
    result.add_argument("--condition-cache-attestation", required=True)
    result.add_argument("--baseline-action-checkpoint", required=True)
    result.add_argument("--condition-updater", required=True)
    result.add_argument("--generation-updater", required=True)
    result.add_argument("--coupled-generation-updater", required=True)
    result.add_argument("--deployment-id", required=True)
    result.add_argument("--instruction", required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> int:
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
