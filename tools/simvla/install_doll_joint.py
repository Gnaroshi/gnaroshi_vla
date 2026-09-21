"""Install a baseline-only Doll checkpoint without changing the existing deployment."""

from __future__ import annotations

import argparse
import copy
import errno
import json
import os
from pathlib import Path
import shutil
import tempfile

from architectures.simvla.adapters.latentloop_real_deploy.contracts import (
    load_deployment_contract, runtime_source_identity, sha256_file,
)
from architectures.simvla.adapters.real_world_training.build_deployment_bundle import (
    DIRECTORY_LAYOUT, FILE_LAYOUT,
)


def link_or_copy(source, destination):
    source = Path(source).resolve(strict=True)
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)


def relocation_payload(source, reference, checkpoint_digest, runtime_digest):
    if source.get("enabled_methods") != ["baseline"]:
        raise ValueError("Only a baseline-only deployment can be installed here")
    if source.get("task_id") != "stackcupanddoll":
        raise ValueError("This installer is for the Doll task")
    if source["runtime_source_identity_sha256"] != runtime_digest:
        raise ValueError("Install the matching runtime before transferring this checkpoint")
    if source["artifacts"]["real_action_transformer"]["sha256"] != checkpoint_digest:
        raise ValueError("Transferred checkpoint hash differs from the selected baseline")
    if source["pairing"]["real_baseline_identity"] != checkpoint_digest:
        raise ValueError("Baseline pairing hash differs from the checkpoint")
    if source["selection"].get("old_ours_weights_compatible") is not False:
        raise ValueError("The new baseline must explicitly reject old Ours weights")
    for field in ("policy", "state", "action"):
        if source[field] != reference[field]:
            raise ValueError(f"Review changed {field} before reusing the existing site")
    for field in ("ip", "home_pose", "gripper", "control"):
        if source["hardware"]["robot"][field] != reference["hardware"]["robot"][field]:
            raise ValueError(f"Robot {field} differs from the existing Doll site")
    if source["hardware"]["cameras"] != reference["hardware"]["cameras"]:
        raise ValueError("Camera mapping/configuration differs from the existing Doll site")
    if not reference["safety_review"]["camera_role_mapping_verified"]:
        raise ValueError("The reference site has no reviewed camera mapping")
    payload = copy.deepcopy(source)
    for name in source["artifacts"]:
        if name != "real_action_transformer":
            if reference["artifacts"][name]["sha256"] != source["artifacts"][name]["sha256"]:
                raise ValueError(f"Shared asset changed: {name}")
        relative = (DIRECTORY_LAYOUT | FILE_LAYOUT)[name]
        payload["artifacts"][name]["path"] = "./" + relative
    # Reuse only the unchanged camera-role review, never old model/live approval.
    payload["safety_review"].update(
        live_authorized=False, model_preflight_passed=False,
        read_only_profile_passed=False, camera_role_mapping_verified=True,
        physical_emergency_stop_verified=False, baseline_bounded_canary_passed=False,
        approved_by="", approved_at="",
    )
    payload["runtime"]["results_directory"] = "./runtime_results"
    payload["runtime"]["camera_serials_file"] = "./runtime_results/camera_serials.json"
    return payload


def install(source_path, reference_path, checkpoint, output):
    source_path, reference_path, checkpoint, output = map(
        lambda x: Path(x).expanduser().resolve(), (source_path, reference_path, checkpoint, output)
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing deployment: {output}")
    source = json.loads(source_path.read_text())
    reference = json.loads(reference_path.read_text())
    digest = sha256_file(checkpoint)
    payload = relocation_payload(source, reference, digest, runtime_source_identity()["combined_sha256"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.install-", dir=output.parent))
    try:
        for name, relative in DIRECTORY_LAYOUT.items():
            location = (reference_path.parent / reference["artifacts"][name]["path"]).resolve()
            shutil.copytree(location, staging / relative, copy_function=link_or_copy)
        for name in payload["artifacts"]:
            if name in DIRECTORY_LAYOUT or name == "official_base_model_weights":
                continue
            destination = staging / FILE_LAYOUT[name]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name == "real_action_transformer":
                link_or_copy(checkpoint, destination)
            else:
                location = (reference_path.parent / reference["artifacts"][name]["path"]).resolve()
                shutil.copy2(location, destination)
        provenance = staging / "provenance"
        shutil.copy2(source_path, provenance / "source_deployment_manifest.json")
        manifest = staging / "deployment_manifest.site.json"
        manifest.write_text(json.dumps(payload, indent=2) + "\n")
        verified = load_deployment_contract(manifest, verify_artifacts=True)
        report = {
            "deployment_id": verified.deployment_id, "checkpoint_sha256": digest,
            "optimizer_step": payload["pairing"]["baseline_optimizer_step"],
            "source_manifest_sha256": sha256_file(source_path),
            "reference_site_sha256": sha256_file(reference_path),
            "runtime_sha256": payload["runtime_source_identity_sha256"],
            "verdict": "JOINT_BASELINE_FILES_INSTALLED", "enabled_methods": ["baseline"],
            "robot_connected": False, "robot_commands_issued": 0,
            "model_inference_verified": False, "live_authorized": False,
            "immutable_model_files_hardlinked_when_possible": True,
        }
        (staging / "installation.json").write_text(json.dumps(report, indent=2) + "\n")
        os.rename(staging, output)
        return {**report, "manifest": str(output / manifest.name)}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.source_manifest, args.reference_manifest, args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
