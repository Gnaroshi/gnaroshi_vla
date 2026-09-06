"""Export the selected joint baseline contract without reusing old Ours weights."""
import argparse
import copy
import json
from pathlib import Path

from architectures.simvla.adapters.latentloop_real_deploy.contracts import (
    load_deployment_contract, runtime_source_identity, sha256_directory,
)
from architectures.simvla.adapters.real_world_training.io_utils import atomic_write_json, sha256_file
from architectures.simvla.adapters.real_world_training.model_io import (
    REAL_JOINT_CHECKPOINT_FORMAT, load_real_action_payload, official_base_identity,
)

ROOT = Path(__file__).resolve().parents[2]


def build(args):
    selection = json.loads((Path(args.run) / "selection.json").read_text())
    checkpoint = Path(selection["best_checkpoint"]).resolve()
    weights = load_real_action_payload(checkpoint)
    if weights["checkpoint_format"] != REAL_JOINT_CHECKPOINT_FORMAT:
        raise ValueError("selected baseline must include its fine-tuned VLM")
    dataset_path = Path(args.dataset).resolve() / "manifest.json"
    dataset = json.loads(dataset_path.read_text())
    norm = dataset_path.parent / dataset["norm_stats"]["path"]
    base = official_base_identity(args.checkpoint, args.processor)
    if weights["official_base"]["model_weights_sha256"] != base.model_weights_sha256:
        raise ValueError("official parent mismatch")
    if weights["dataset_identity_sha256"] != dataset["dataset_identity_sha256"]:
        raise ValueError("dataset mismatch")
    if weights["norm_stats_sha256"] != sha256_file(norm):
        raise ValueError("normalization mismatch")
    payload = json.loads((ROOT / "artifacts/simvla/real_world/deployment_manifest.example.json").read_text())
    site = json.loads((ROOT / "artifacts/simvla/real_world/seer_doll_site_profile.json").read_text())
    payload.update(deployment_id=Path(args.run).name, task_id="stackcupanddoll",
                   enabled_methods=["baseline"],
                   runtime_source_identity_sha256=runtime_source_identity()["combined_sha256"])
    robot = payload["hardware"]["robot"]
    robot["ip"] = site["robot"]["ip"]
    robot["home_pose"] = site["robot"]["home_pose"]
    robot["home_pose_source"] = "User-approved existing Doll robot site"
    robot["gripper"] = copy.deepcopy(site["robot"]["gripper"])
    robot["control"].update(site["robot"]["control"])
    payload["hardware"]["cameras"].update(site["cameras"])
    payload["runtime"].update(
        control_frequency_hz=60, training_sample_hz=15, max_steps=5000,
        num_rollouts_per_instruction=15, warmup_steps=3,
        instructions=[dataset["instruction"]],
    )
    paths = {
        "official_base_model_directory": Path(args.checkpoint).resolve(),
        "official_base_model_weights": Path(args.checkpoint).resolve() / "model.safetensors",
        "processor_directory": Path(args.processor).resolve(),
        "norm_stats": norm, "dataset_manifest": dataset_path,
        "real_action_transformer": checkpoint,
    }
    payload["artifacts"] = {
        name: {"path": str(path), "sha256": sha256_directory(path) if path.is_dir() else sha256_file(path)}
        for name, path in paths.items()
    }
    payload["pairing"] = {
        "official_base_model_identity": base.model_weights_sha256,
        "real_baseline_identity": sha256_file(checkpoint),
        "norm_stats_identity": sha256_file(norm),
        "dataset_identity": dataset["dataset_identity_sha256"],
        "baseline_optimizer_step": weights["optimizer_step"],
    }
    payload["selection"] = {**selection, "robot_success_measured": False,
                            "old_ours_weights_compatible": False}
    path = Path(args.output).resolve()
    atomic_write_json(path, payload)
    load_deployment_contract(path, verify_artifacts=True)
    return {"verdict": "JOINT_BASELINE_MANIFEST_READY", "path": str(path),
            "checkpoint": str(checkpoint), "enabled_methods": ["baseline"],
            "live_authorized": False, "robot_success_measured": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "dataset", "checkpoint", "processor", "output"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(build(parser.parse_args()), indent=2))

