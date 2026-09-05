"""Fail-closed validation for the real SimVLA training artifact chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .condition_cache import validate_real_condition_cache
from .dataset import DATASET_SCHEMA, valid_action_window_starts
from .geometry import RealActionScales, transition_to_normalized_action
from .io_utils import sha256_file, sha256_text
from .model_io import load_real_action_payload, official_base_identity
from .updater_io import (
    audit_real_coupled_checkpoint,
    load_real_coupled_generation,
    load_real_updater,
)
from .verify_condition_cache import validate_condition_cache_attestation


EXPECTED_SPLIT_SEED = 20260904
EXPECTED_TARGET_HZ = 15.0
EXPECTED_IMAGE_CONTRACT = {
    "source_order": ["base_rgb", "wrist_rgb"],
    "source_orientation": "as_captured_no_flip",
    "storage": "JPEG quality=95 subsampling=0",
    "model_preprocessing": "resize_with_pad_224_then_bicubic_384_imagenet_norm",
}
EXPECTED_STATE_CONTRACT = {
    "representation": "tcp_xyz_rotvec_plus_opposed_finger_positions",
    "condition_updater_rotation_delta": (
        "current rotvec mapped to equivalent 2pi branch nearest previous rotvec"
    ),
    "gripper_max_opening_m": 0.04,
}
EXPECTED_ACTION_CONTRACT = {
    "representation": (
        "inv(T_current)@T_next local xyz plus xyz_euler plus continuous gripper target"
    ),
    "pose_label_source": "measured transition observation_t to observation_t+1",
    "gripper_label_source": "1 - 2 * command_t stored in current frame",
    "translation_scale_m": 0.02,
    "rotation_scale_rad": 0.05,
    "clip_abs": 1.0,
    "model_positive_gripper_means": "open",
}


def _recompute_stats(arrays: list[np.ndarray]) -> dict[str, np.ndarray]:
    if not arrays:
        raise ValueError("normalization audit received no arrays")
    values = np.concatenate(arrays, axis=0).astype(np.float64)
    return {
        "mean": values.mean(axis=0),
        "std": np.maximum(values.std(axis=0), 1e-6),
        "q01": np.quantile(values, 0.01, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }


def _validate_norm_statistics(
    path: Path,
    *,
    state_values: list[np.ndarray],
    action_values: list[np.ndarray],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    expected_metadata = {
        "schema_version": DATASET_SCHEMA,
        "split": "train_only",
        "num_episodes": 32,
        "state_dim": 8,
        "action_dim": 7,
        "state_labels": [
            "tcp_x",
            "tcp_y",
            "tcp_z",
            "rotvec_x",
            "rotvec_y",
            "rotvec_z",
            "finger_positive_m",
            "finger_negative_m",
        ],
        "action_labels": [
            "dx",
            "dy",
            "dz",
            "droll",
            "dpitch",
            "dyaw",
            "gripper",
        ],
    }
    if metadata != expected_metadata:
        raise ValueError("real dataset normalization metadata changed")
    observed = payload.get("norm_stats", {})
    for name, arrays in (("state", state_values), ("actions", action_values)):
        recomputed = _recompute_stats(arrays)
        section = observed.get(name, {})
        for statistic, expected in recomputed.items():
            value = np.asarray(section.get(statistic), dtype=np.float64)
            if value.shape != expected.shape or not np.allclose(
                value, expected, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"real dataset {name} normalization statistic changed: {statistic}"
                )


def validate_real_dataset_manifest(
    manifest: str | Path, *, verify_episode_checksums: bool
) -> dict[str, Any]:
    """Recompute the compact real-dataset contract instead of trusting PASS text."""

    manifest_path = Path(manifest).expanduser().resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("unsupported real SimVLA dataset schema")
    if payload.get("verdict") != "REAL_DATASET_CONTRACT_PASS":
        raise ValueError("real SimVLA dataset conversion did not pass")
    if payload.get("dataset_id") != "stackcupanddoll":
        raise ValueError("real SimVLA dataset task identity changed")
    if payload.get("action_horizon") != 10 or payload.get("execution_horizon") != 5:
        raise ValueError("real SimVLA dataset must use H=10,R=5")
    if payload.get("sampling_mode") != "native_capture_order_with_timing_valid_windows":
        raise ValueError("real SimVLA dataset sampling contract changed")
    if payload.get("split_seed") != EXPECTED_SPLIT_SEED:
        raise ValueError("real SimVLA dataset split seed changed")
    if float(payload.get("target_hz", -1)) != EXPECTED_TARGET_HZ:
        raise ValueError("real SimVLA dataset must preserve the native 15 Hz stream")
    if payload.get("image_contract") != EXPECTED_IMAGE_CONTRACT:
        raise ValueError("real SimVLA image contract changed")
    if payload.get("state_contract") != EXPECTED_STATE_CONTRACT:
        raise ValueError("real SimVLA state contract changed")

    source = payload.get("source", {})
    expected_source = {
        "format": "3dflow_teleoperation_pickle",
        "control_pose_channels_used": False,
        "tcp_pose_source": "ee_pos_quat interpreted as xyz+rotation_vector",
        "observation_action_alignment": (
            "frame_t stores observation_t and command_t before env.step(command_t)"
        ),
        "gripper_command_source": (
            "current_frame.control[6] recorded with observation_t before "
            "env.step(command_t); 0=open and 1=close continuous target"
        ),
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"real dataset source contract changed: {key}")
    action_contract = payload.get("action_contract", {})
    if action_contract != EXPECTED_ACTION_CONTRACT:
        raise ValueError("real SimVLA action contract changed")

    episodes = payload.get("episodes")
    splits = payload.get("splits", {})
    if not isinstance(episodes, list) or len(episodes) != 40:
        raise ValueError("real SimVLA dataset requires exactly 40 episodes")
    train = [str(value) for value in splits.get("train", [])]
    validation = [str(value) for value in splits.get("validation", [])]
    if len(train) != 32 or len(validation) != 8 or set(train).intersection(validation):
        raise ValueError("real SimVLA dataset requires a disjoint 32/8 episode split")
    episode_ids = [str(item.get("episode_id", "")) for item in episodes]
    if len(set(episode_ids)) != 40 or set(episode_ids) != set(train).union(validation):
        raise ValueError("real SimVLA split does not cover its 40 unique episodes")

    episode_sha256: dict[str, str] = {}
    total_samples = 0
    total_transitions = 0
    total_excluded_transitions = 0
    total_candidate_samples = 0
    state_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    scales = RealActionScales()
    for specification in episodes:
        episode_id = str(specification["episode_id"])
        episode_path = (root / str(specification["path"])).resolve()
        if not episode_path.is_file():
            raise FileNotFoundError(f"real dataset episode is missing: {episode_path}")
        if int(specification.get("size_bytes", -1)) != episode_path.stat().st_size:
            raise ValueError(f"real dataset episode size changed: {episode_id}")
        expected_sha = str(specification.get("sha256", ""))
        if len(expected_sha) != 64:
            raise ValueError(f"real dataset episode SHA-256 is invalid: {episode_id}")
        if verify_episode_checksums and sha256_file(episode_path) != expected_sha:
            raise ValueError(f"real dataset episode SHA-256 changed: {episode_id}")
        episode_sha256[episode_id] = expected_sha
        with h5py.File(episode_path, "r") as handle:
            if handle.attrs.get("schema_version") != DATASET_SCHEMA:
                raise ValueError(f"real dataset HDF5 schema changed: {episode_id}")
            if str(handle.attrs.get("episode_id")) != episode_id:
                raise ValueError(f"real dataset HDF5 episode ID changed: {episode_id}")
            frames = int(specification["frames"])
            expected_attributes = {
                "instruction": payload.get("instruction"),
                "source_frame_count": frames,
                "target_hz": EXPECTED_TARGET_HZ,
                "sampling_mode": "native_capture_order",
                "observation_action_alignment": expected_source[
                    "observation_action_alignment"
                ],
            }
            for name, expected in expected_attributes.items():
                if handle.attrs.get(name) != expected:
                    raise ValueError(
                        f"real dataset HDF5 attribute changed: {episode_id}/{name}"
                    )
            required_shapes = {
                "timestamp_s": (frames,),
                "source_index": (frames,),
                "valid_transition": (frames - 1,),
                "state": (frames, 8),
                "gripper_command": (frames,),
                "step_action": (frames - 1, 7),
                "raw_normalized_step_action": (frames - 1, 7),
                "base_rgb_jpeg": (frames,),
                "wrist_rgb_jpeg": (frames,),
            }
            for name, shape in required_shapes.items():
                if name not in handle or tuple(handle[name].shape) != shape:
                    raise ValueError(
                        f"real dataset HDF5 field changed: {episode_id}/{name}"
                    )
            timestamps = np.asarray(handle["timestamp_s"], dtype=np.float64)
            if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0):
                raise ValueError(f"real dataset timestamps changed: {episode_id}")
            source_indices = np.asarray(handle["source_index"], dtype=np.int64)
            if not np.array_equal(source_indices, np.arange(frames, dtype=np.int64)):
                raise ValueError(f"real dataset native frame order changed: {episode_id}")
            states = np.asarray(handle["state"], dtype=np.float32)
            commands = np.asarray(handle["gripper_command"], dtype=np.float32)
            actions = np.asarray(handle["step_action"], dtype=np.float32)
            raw_actions = np.asarray(
                handle["raw_normalized_step_action"], dtype=np.float32
            )
            if not all(
                np.isfinite(value).all()
                for value in (states, commands, actions, raw_actions)
            ):
                raise ValueError(f"real dataset contains non-finite values: {episode_id}")
            if np.any(commands < 0.0) or np.any(commands > 1.0):
                raise ValueError(f"real dataset gripper command is outside [0,1]: {episode_id}")
            if np.any(states[:, 6] < 0.0) or np.any(states[:, 6] > 0.04):
                raise ValueError(f"real dataset positive finger state is outside [0,0.04]: {episode_id}")
            if not np.allclose(states[:, 6], -states[:, 7], rtol=0.0, atol=1e-7):
                raise ValueError(f"real dataset opposed finger encoding changed: {episode_id}")
            expected_raw = np.stack(
                [
                    transition_to_normalized_action(
                        states[index, :6],
                        states[index + 1, :6],
                        float(commands[index]),
                        scales,
                        clip=False,
                    )[1]
                    for index in range(frames - 1)
                ],
                axis=0,
            )
            if not np.allclose(
                raw_actions, expected_raw, rtol=0.0, atol=2e-5
            ):
                raise ValueError(
                    f"real dataset actions do not match measured TCP transitions: {episode_id}"
                )
            if not np.allclose(
                actions,
                np.clip(raw_actions, -1.0, 1.0),
                rtol=0.0,
                atol=1e-7,
            ):
                raise ValueError(f"real dataset action clipping contract changed: {episode_id}")
            if np.any(np.abs(raw_actions[:, :6]) > 1.0 + 1e-7):
                raise ValueError(f"real dataset contains clipped pose labels: {episode_id}")
            if not np.allclose(actions[:, 6], 1.0 - 2.0 * commands[:-1], atol=1e-7):
                raise ValueError(f"real dataset does not use command_t labels: {episode_id}")
            valid = np.asarray(handle["valid_transition"], dtype=bool)
            samples = int(valid_action_window_starts(valid, 10).size)
            if samples != int(specification.get("training_samples", -1)):
                raise ValueError(f"real dataset H=10 sample count changed: {episode_id}")
            total_samples += samples
            total_transitions += frames - 1
            total_excluded_transitions += int((~valid).sum())
            total_candidate_samples += frames - 10
            if episode_id in train:
                state_values.append(states)
                action_values.append(actions[valid])

    audit = payload.get("audit", {})
    expected_audit_counts = {
        "training_samples": total_samples,
        "transition_count": total_transitions,
        "excluded_transition_count": total_excluded_transitions,
        "candidate_training_samples": total_candidate_samples,
        "excluded_training_samples": total_candidate_samples - total_samples,
        "gripper_label_alignment_error_count": 0,
        "pose_clip_transition_count": 0,
    }
    for name, expected in expected_audit_counts.items():
        if int(audit.get(name, -1)) != expected:
            raise ValueError(f"real dataset aggregate audit changed: {name}")
    if float(audit.get("pose_clip_transition_fraction", -1)) != 0.0:
        raise ValueError("real dataset aggregate pose clipping changed")
    if float(audit.get("max_allowed_pose_clip_fraction", -1)) != 0.0:
        raise ValueError("real dataset allows pose-label clipping")
    if total_samples != int(audit.get("training_samples", -1)):
        raise ValueError("real dataset aggregate H=10 sample count changed")
    norm = payload.get("norm_stats", {})
    norm_path = (root / str(norm.get("path", ""))).resolve()
    if not norm_path.is_file() or sha256_file(norm_path) != norm.get("sha256"):
        raise ValueError("real dataset normalization statistics changed")
    _validate_norm_statistics(
        norm_path, state_values=state_values, action_values=action_values
    )
    identity_payload = {
        "schema": payload["schema_version"],
        "episode_sha256": episode_sha256,
        "split_seed": payload.get("split_seed"),
        "train": train,
        "validation": validation,
        "instruction": payload.get("instruction"),
    }
    if payload.get("dataset_identity_sha256") != sha256_text(
        json.dumps(identity_payload, sort_keys=True)
    ):
        raise ValueError("real SimVLA dataset identity is invalid")
    return payload


def validate_real_training_sources(
    *,
    condition_cache: str | Path,
    checkpoint: str | Path,
    processor: str | Path,
    norm_stats: str | Path,
    verify_cache_array_checksums: bool,
    condition_cache_attestation: str | Path | None = None,
) -> dict[str, Any]:
    cache_root = Path(condition_cache).expanduser().resolve()
    cache = validate_real_condition_cache(
        cache_root, verify_array_checksums=verify_cache_array_checksums
    )
    dataset_path = Path(cache["dataset_manifest"]).expanduser().resolve()
    dataset = validate_real_dataset_manifest(
        dataset_path, verify_episode_checksums=False
    )
    base = official_base_identity(checkpoint, processor)
    norm_sha = sha256_file(norm_stats)
    recorded_processor = Path(
        cache.get("official_base", {}).get("processor_directory", "")
    ).expanduser()
    checks = {
        "dataset_verdict": dataset.get("verdict") == "REAL_DATASET_CONTRACT_PASS",
        "dataset_identity": dataset.get("dataset_identity_sha256")
        == cache.get("dataset_identity_sha256"),
        "norm": dataset.get("norm_stats", {}).get("sha256") == norm_sha,
        "base_weights": cache.get("official_base", {}).get("model_weights_sha256")
        == base.model_weights_sha256,
        "action_mode": base.action_mode == "libero_joint",
        "action_horizon": base.action_horizon == 10,
        "processor_path": recorded_processor.is_absolute()
        and recorded_processor.resolve() == Path(processor).expanduser().resolve(),
    }
    if not all(checks.values()):
        raise ValueError(f"real SimVLA training source mismatch: {checks}")
    result = {
        "condition_cache": cache,
        "dataset": dataset,
        "official_base": base.to_dict(),
        "norm_stats_sha256": norm_sha,
        "checks": checks,
    }
    if condition_cache_attestation is not None:
        result["condition_cache_attestation"] = (
            validate_condition_cache_attestation(
                condition_cache_attestation,
                condition_cache=cache_root,
                checkpoint=checkpoint,
                processor=processor,
                norm_stats=norm_stats,
                verify_cache_array_checksums=False,
            )
        )
    return result


def validate_real_baseline_checkpoint(
    checkpoint: str | Path,
    *,
    source: dict[str, Any],
    expected_optimizer_step: int | None,
) -> dict[str, Any]:
    payload = load_real_action_payload(checkpoint)
    cache = source["condition_cache"]
    expected = {
        "base": source["official_base"]["model_weights_sha256"],
        "norm": source["norm_stats_sha256"],
        "dataset": cache["dataset_identity_sha256"],
        "cache": cache["condition_cache_identity_sha256"],
        "attestation": source["condition_cache_attestation"][
            "attestation_identity_sha256"
        ],
    }
    observed = {
        "base": payload.get("official_base", {}).get("model_weights_sha256"),
        "norm": payload.get("norm_stats_sha256"),
        "dataset": payload.get("dataset_identity_sha256"),
        "cache": payload.get("training_config", {}).get(
            "condition_cache_identity_sha256"
        ),
        "attestation": payload.get("training_config", {}).get(
            "condition_cache_attestation_identity_sha256"
        ),
    }
    mismatches = {
        name: {"observed": observed[name], "expected": value}
        for name, value in expected.items()
        if observed[name] != value
    }
    step = int(payload.get("optimizer_step", -1))
    if expected_optimizer_step is not None and step != int(expected_optimizer_step):
        mismatches["optimizer_step"] = {
            "observed": step,
            "expected": int(expected_optimizer_step),
        }
    training = payload.get("training_config", {})
    if training.get("protocol") != (
        "official_full_checkpoint_then_frozen_vlm_action_transformer_finetune"
    ):
        mismatches["protocol"] = {
            "observed": training.get("protocol"),
            "expected": "official_full_checkpoint_then_frozen_vlm_action_transformer_finetune",
        }
    if training.get("action_transformer_reinitialized") is not False:
        mismatches["action_transformer_reinitialized"] = {
            "observed": training.get("action_transformer_reinitialized"),
            "expected": False,
        }
    if mismatches:
        raise ValueError(f"real SimVLA baseline artifact mismatch: {mismatches}")
    return {
        "path": str(Path(checkpoint).expanduser().resolve()),
        "sha256": sha256_file(checkpoint),
        "optimizer_step": step,
        "checks": {name: True for name in (*expected, "protocol", "initialization")},
    }


def validate_real_artifact_chain(args: argparse.Namespace) -> dict[str, Any]:
    source = validate_real_training_sources(
        condition_cache=args.condition_cache,
        checkpoint=args.checkpoint,
        processor=args.processor,
        norm_stats=args.norm_stats,
        verify_cache_array_checksums=args.verify_cache_array_checksums,
        condition_cache_attestation=args.condition_cache_attestation,
    )
    result: dict[str, Any] = {
        "verdict": "REAL_SIMVLA_ARTIFACT_CHAIN_PASS",
        "cache_identity": source["condition_cache"][
            "condition_cache_identity_sha256"
        ],
        "dataset_identity": source["condition_cache"]["dataset_identity_sha256"],
        "official_base_identity": source["official_base"]["model_weights_sha256"],
        "condition_cache_attestation_identity": source[
            "condition_cache_attestation"
        ]["attestation_identity_sha256"],
        "validated": ["sources", "condition_cache", "condition_cache_attestation"],
    }
    if not args.baseline_action_checkpoint:
        return result
    baseline = validate_real_baseline_checkpoint(
        args.baseline_action_checkpoint,
        source=source,
        expected_optimizer_step=args.baseline_optimizer_step,
    )
    result["baseline"] = baseline
    result["validated"].append("baseline")
    expected = {
        "expected_baseline_sha256": baseline["sha256"],
        "expected_norm_sha256": source["norm_stats_sha256"],
        "expected_dataset_identity_sha256": result["dataset_identity"],
        "expected_cache_identity_sha256": result["cache_identity"],
        "expected_cache_attestation_identity_sha256": source[
            "condition_cache_attestation"
        ]["attestation_identity_sha256"],
        "expected_optimizer_step": args.updater_optimizer_step,
    }
    for kind, path in (
        ("condition", args.condition_updater),
        ("generation", args.generation_updater),
    ):
        if not path:
            continue
        _, payload = load_real_updater(path, kind=kind, device="cpu", **expected)
        result[kind] = {
            "path": str(Path(path).expanduser().resolve()),
            "sha256": sha256_file(path),
            "optimizer_step": int(payload["optimizer_step"]),
        }
        result["validated"].append(kind)
    if args.coupled_generation_updater:
        if not args.condition_updater or not args.generation_updater:
            raise ValueError("coupled validation requires both parent updaters")
        cache_manifest = Path(args.condition_cache).expanduser().resolve() / "manifest.json"
        _, payload = load_real_coupled_generation(
            args.coupled_generation_updater,
            device="cpu",
            expected_parent_generation_sha256=result["generation"]["sha256"],
            expected_condition_updater_sha256=result["condition"]["sha256"],
            expected_cache_manifest_sha256=sha256_file(cache_manifest),
            expected_optimizer_step=args.coupled_optimizer_step,
            **{key: value for key, value in expected.items() if key != "expected_optimizer_step"},
        )
        projection = audit_real_coupled_checkpoint(
            parent_generation_checkpoint=args.generation_updater,
            coupled_generation_checkpoint=args.coupled_generation_updater,
        )
        if projection["verdict"] != "PROJECTION_ONLY_STATE_PASS":
            raise ValueError("coupled Generation artifact is not projection-only")
        result["coupled"] = {
            "path": str(Path(args.coupled_generation_updater).expanduser().resolve()),
            "sha256": sha256_file(args.coupled_generation_updater),
            "optimizer_step": int(payload["optimizer_step"]),
            "projection_only_state_audit": projection,
        }
        result["validated"].append("coupled")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--condition-cache-attestation", required=True)
    parser.add_argument("--baseline-action-checkpoint")
    parser.add_argument("--condition-updater")
    parser.add_argument("--generation-updater")
    parser.add_argument("--coupled-generation-updater")
    parser.add_argument("--baseline-optimizer-step", type=int, default=3000)
    parser.add_argument("--updater-optimizer-step", type=int, default=10_000)
    parser.add_argument("--coupled-optimizer-step", type=int, default=10_000)
    parser.add_argument("--verify-cache-array-checksums", action="store_true")
    return parser


def main() -> int:
    print(json.dumps(validate_real_artifact_chain(build_parser().parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
