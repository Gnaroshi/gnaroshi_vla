"""Build a relocatable, provenance-complete SimVLA real deployment bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from architectures.simvla.adapters.latentloop_real_deploy.contracts import (
    load_deployment_contract,
)

from .io_utils import atomic_write_json, sha256_file


FILE_LAYOUT = {
    "official_base_model_weights": "official_base_model/model.safetensors",
    "norm_stats": "norm_stats/real_norm.json",
    "dataset_manifest": "provenance/dataset_manifest.json",
    "condition_cache_manifest": "provenance/condition_cache_manifest.json",
    "condition_cache_attestation": "provenance/condition_cache_attestation.json",
    "real_action_transformer": "checkpoints/real_action_transformer.pt",
    "condition_updater": "checkpoints/condition_updater.pt",
    "generation_updater": "checkpoints/generation_updater.pt",
    "coupled_generation_updater": "checkpoints/coupled_generation_updater.pt",
}
DIRECTORY_LAYOUT = {
    "official_base_model_directory": "official_base_model",
    "processor_directory": "processor",
}


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    source_contract = load_deployment_contract(args.manifest, verify_artifacts=True)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"deployment bundle destination already exists: {output}")
    building = output.with_name(f".{output.name}.building-{os.getpid()}")
    if building.exists():
        raise FileExistsError(f"stale deployment bundle staging directory: {building}")
    building.mkdir(parents=True)
    try:
        for name, relative in DIRECTORY_LAYOUT.items():
            shutil.copytree(
                source_contract.artifacts[name].path,
                building / relative,
                copy_function=shutil.copy2,
            )
        # The weights are already included by the official model directory.
        for name, relative in FILE_LAYOUT.items():
            destination = building / relative
            if name == "official_base_model_weights":
                if not destination.is_file():
                    raise FileNotFoundError(
                        "copied official model directory lacks model.safetensors"
                    )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_contract.artifacts[name].path, destination)

        payload = deepcopy(source_contract.payload)
        for name, relative in {**DIRECTORY_LAYOUT, **FILE_LAYOUT}.items():
            payload["artifacts"][name]["path"] = f"./{relative}"
        payload["runtime"]["results_directory"] = "./runtime_results"
        payload["runtime"]["camera_serials_file"] = (
            "./runtime_results/camera_serials.json"
        )
        # Authorization is deliberately invalidated when a bundle is relocated.
        payload["safety_review"].update(
            {
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
        )
        source_copy = building / "provenance" / "source_deployment_manifest.json"
        shutil.copy2(source_contract.path, source_copy)
        manifest = atomic_write_json(building / "deployment_manifest.json", payload)
        verified = load_deployment_contract(manifest, verify_artifacts=True)
        inventory = {
            name: {
                "relative_path": payload["artifacts"][name]["path"],
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            }
            for name, artifact in verified.artifacts.items()
        }
        report = {
            "verdict": "REAL_SIMVLA_DEPLOYMENT_BUNDLE_PASS",
            "deployment_id": verified.deployment_id,
            "source_manifest_sha256": sha256_file(source_contract.path),
            "manifest": "./deployment_manifest.json",
            "inventory": inventory,
            "condition_cache_arrays_included": False,
            "dataset_episode_files_included": False,
            "live_authorized": False,
        }
        atomic_write_json(building / "bundle_inventory.json", report)
        os.replace(building, output)
        return {**report, "output": str(output)}
    finally:
        if building.exists():
            shutil.rmtree(building)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    print(json.dumps(build_bundle(build_parser().parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
