"""Convert 3DFlow teleoperation PKLs into compact, audited SimVLA HDF5 data."""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

from .dataset import DATASET_SCHEMA, valid_action_window_starts
from .geometry import (
    ACTION_DIM,
    ACTION_HORIZON,
    PROPRIO_DIM,
    RealActionScales,
    apply_normalized_action,
    encode_opposed_finger_state,
    pose6d_to_matrix,
    transition_to_normalized_action,
)
from .io_utils import atomic_write_json, sha256_file, sha256_text


DEFAULT_INSTRUCTION = (
    "Pick up the white cup and place it on top of the upside-down pink cup, "
    "then pick up the blue penguin plush toy and put it in the white cup"
)
REQUIRED_KEYS = {"base_rgb", "wrist_rgb", "ee_pos_quat", "gripper_position", "control"}


def timestamp_from_path(path: Path) -> float:
    parsed = datetime.fromisoformat(path.stem)
    return (parsed - datetime(1970, 1, 1)).total_seconds()


def _load_raw_frame(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"raw frame is not a dictionary: {path}")
    missing = sorted(REQUIRED_KEYS - set(payload))
    if missing:
        raise ValueError(f"raw frame {path} is missing keys: {missing}")
    base = np.asarray(payload["base_rgb"])
    wrist = np.asarray(payload["wrist_rgb"])
    pose = np.asarray(payload["ee_pos_quat"], dtype=np.float64).reshape(-1)
    gripper = np.asarray(payload["gripper_position"], dtype=np.float64).reshape(-1)
    control = np.asarray(payload["control"], dtype=np.float64).reshape(-1)
    if base.ndim != 3 or wrist.ndim != 3 or base.shape[-1] != 3 or wrist.shape[-1] != 3:
        raise ValueError(f"RGB arrays must be HWC three-channel images: {path}")
    if base.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise ValueError(f"RGB arrays must be uint8: {path}")
    if pose.shape != (6,) or gripper.shape != (1,) or control.shape != (7,):
        raise ValueError(f"unexpected pose/gripper/control shape in {path}")
    if not all(np.isfinite(value).all() for value in (pose, gripper, control)):
        raise ValueError(f"non-finite robot value in {path}")
    return {
        "base_rgb": np.ascontiguousarray(base),
        "wrist_rgb": np.ascontiguousarray(wrist),
        "pose": pose,
        "gripper_position": float(gripper[0]),
        "gripper_control": float(control[6]),
    }


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(
        buffer, format="JPEG", quality=int(quality), subsampling=0, optimize=False
    )
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def _roundtrip_error(current: np.ndarray, following: np.ndarray, raw_action: np.ndarray, scales: RealActionScales) -> float:
    reconstructed = apply_normalized_action(current, raw_action, scales)
    left = pose6d_to_matrix(reconstructed)
    right = pose6d_to_matrix(following)
    return float(np.max(np.abs(left - right)))


def convert_episode(
    episode_dir: Path,
    destination: Path,
    *,
    instruction: str,
    target_hz: float,
    max_transition_period_error_ms: float,
    jpeg_quality: int,
    scales: RealActionScales,
) -> dict[str, Any]:
    source_files = sorted((*episode_dir.glob("*.pkl"), *episode_dir.glob("*.pickle")))
    if len(source_files) < ACTION_HORIZON + 1:
        raise ValueError(f"episode {episode_dir.name} is too short")
    timestamps = np.asarray([timestamp_from_path(path) for path in source_files], dtype=np.float64)
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"source timestamps must be strictly increasing: {episode_dir.name}")
    # The capture stream is already nominally 15 Hz. Keep its synchronized RGB,
    # pose, and control records intact instead of snapping them to another grid.
    selected_indices = np.arange(timestamps.size, dtype=np.int64)
    frames = [_load_raw_frame(source_files[int(index)]) for index in selected_indices]
    selected_times = timestamps[selected_indices]
    source_interval_ms = np.diff(selected_times) * 1000.0
    nominal_period_ms = 1000.0 / float(target_hz)
    transition_period_error_ms = np.abs(source_interval_ms - nominal_period_ms)
    valid_transition = transition_period_error_ms <= float(max_transition_period_error_ms)
    valid_starts = valid_action_window_starts(valid_transition, ACTION_HORIZON)
    if valid_starts.size == 0:
        raise ValueError(f"episode {episode_dir.name} has no timing-valid H={ACTION_HORIZON} window")
    states = np.stack(
        [
            encode_opposed_finger_state(frame["pose"], frame["gripper_position"], scales)
            for frame in frames
        ],
        axis=0,
    )
    clipped_actions: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    roundtrip_errors: list[float] = []
    for current, following in zip(frames[:-1], frames[1:]):
        clipped, raw = transition_to_normalized_action(
            current["pose"],
            following["pose"],
            following["gripper_control"],
            scales,
            clip=True,
        )
        clipped_actions.append(clipped)
        raw_actions.append(raw)
        roundtrip_errors.append(
            _roundtrip_error(current["pose"], following["pose"], raw, scales)
        )
    actions = np.stack(clipped_actions, axis=0)
    raw_action_array = np.stack(raw_actions, axis=0)
    pose_exceeded = np.any(np.abs(raw_action_array[:, :6]) > scales.clip_abs, axis=1)
    component_exceeded = np.abs(raw_action_array[:, :6]) > scales.clip_abs

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_version"] = DATASET_SCHEMA
            handle.attrs["episode_id"] = episode_dir.name
            handle.attrs["instruction"] = instruction
            handle.attrs["source_frame_count"] = len(source_files)
            handle.attrs["target_hz"] = float(target_hz)
            handle.attrs["sampling_mode"] = "native_capture_order"
            handle.create_dataset("timestamp_s", data=selected_times)
            handle.create_dataset("source_index", data=selected_indices)
            handle.create_dataset("valid_transition", data=valid_transition.astype(np.uint8))
            handle.create_dataset("state", data=states, compression="gzip", shuffle=True)
            handle.create_dataset("step_action", data=actions, compression="gzip", shuffle=True)
            handle.create_dataset(
                "raw_normalized_step_action", data=raw_action_array, compression="gzip", shuffle=True
            )
            base_dataset = handle.create_dataset(
                "base_rgb_jpeg", shape=(len(frames),), dtype=jpeg_dtype
            )
            wrist_dataset = handle.create_dataset(
                "wrist_rgb_jpeg", shape=(len(frames),), dtype=jpeg_dtype
            )
            for index, frame in enumerate(frames):
                base_dataset[index] = _jpeg(frame["base_rgb"], jpeg_quality)
                wrist_dataset[index] = _jpeg(frame["wrist_rgb"], jpeg_quality)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "episode_id": episode_dir.name,
        "path": str(Path("episodes") / destination.name),
        "source_frames": len(source_files),
        "frames": int(states.shape[0]),
        "candidate_training_samples": int(states.shape[0] - ACTION_HORIZON),
        "training_samples": int(valid_starts.size),
        "excluded_training_samples": int(states.shape[0] - ACTION_HORIZON - valid_starts.size),
        "duration_s": float(selected_times[-1] - selected_times[0]),
        "source_interval_ms": {
            "min": float(source_interval_ms.min()),
            "mean": float(source_interval_ms.mean()),
            "p95": float(np.percentile(source_interval_ms, 95)),
            "max": float(source_interval_ms.max()),
        },
        "transition_period_error_ms": {
            "mean": float(transition_period_error_ms.mean()),
            "p95": float(np.percentile(transition_period_error_ms, 95)),
            "max": float(transition_period_error_ms.max()),
            "max_included": float(transition_period_error_ms[valid_transition].max()),
        },
        "excluded_transition_count": int((~valid_transition).sum()),
        "pose_clip_transition_count": int(pose_exceeded.sum()),
        "pose_clip_transition_fraction": float(pose_exceeded.mean()),
        "pose_clip_component_count": int(component_exceeded.sum()),
        "pose_clip_component_fraction": float(component_exceeded.mean()),
        "raw_pose_action_abs_max": np.abs(raw_action_array[:, :6]).max(axis=0).tolist(),
        "roundtrip_matrix_max_abs": float(max(roundtrip_errors)),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
    }


def _stats(arrays: list[np.ndarray]) -> dict[str, list[float]]:
    values = np.concatenate(arrays, axis=0).astype(np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": np.maximum(values.std(axis=0), 1e-6).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _write_norm_stats(output: Path, episodes: list[dict[str, Any]], train_ids: set[str]) -> Path:
    state_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    for episode in episodes:
        if episode["episode_id"] not in train_ids:
            continue
        with h5py.File(output / episode["path"], "r") as handle:
            state_values.append(np.asarray(handle["state"], dtype=np.float32))
            action = np.asarray(handle["step_action"], dtype=np.float32)
            valid = np.asarray(handle["valid_transition"], dtype=bool)
            action_values.append(action[valid])
    payload = {
        "norm_stats": {
            "state": _stats(state_values),
            "actions": _stats(action_values),
        },
        "metadata": {
            "schema_version": DATASET_SCHEMA,
            "split": "train_only",
            "num_episodes": len(train_ids),
            "state_dim": PROPRIO_DIM,
            "action_dim": ACTION_DIM,
            "state_labels": [
                "tcp_x", "tcp_y", "tcp_z", "rotvec_x", "rotvec_y", "rotvec_z",
                "finger_positive_m", "finger_negative_m",
            ],
            "action_labels": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        },
    }
    return atomic_write_json(output / "real_norm.json", payload)


def convert_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    episode_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    if len(episode_dirs) != args.expected_episodes:
        raise RuntimeError(
            f"expected exactly {args.expected_episodes} episodes, found {len(episode_dirs)}"
        )
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"output is non-empty; pass --resume after inspection: {output}")
    (output / "episodes").mkdir(parents=True, exist_ok=True)
    scales = RealActionScales(
        translation_m=args.translation_scale_m,
        rotation_rad=args.rotation_scale_rad,
        clip_abs=args.clip_abs,
        gripper_max_opening_m=args.gripper_max_opening_m,
    )
    max_transition_period_error_ms = args.max_transition_period_error_ms
    if max_transition_period_error_ms is None:
        # A sample farther than half a nominal period belongs closer to a
        # different control tick and must not be treated as one 15 Hz action.
        max_transition_period_error_ms = 500.0 / float(args.target_hz)
    summaries = []
    for number, episode_dir in enumerate(episode_dirs, start=1):
        destination = output / "episodes" / f"{episode_dir.name}.h5"
        print(f"[{number}/{len(episode_dirs)}] converting {episode_dir.name}", flush=True)
        summaries.append(
            convert_episode(
                episode_dir,
                destination,
                instruction=args.instruction,
                target_hz=args.target_hz,
                max_transition_period_error_ms=max_transition_period_error_ms,
                jpeg_quality=args.jpeg_quality,
                scales=scales,
            )
        )

    rng = np.random.default_rng(args.split_seed)
    shuffled = [episode_dirs[index].name for index in rng.permutation(len(episode_dirs))]
    train_count = int(args.train_episodes)
    if train_count < 1 or train_count >= len(shuffled):
        raise ValueError("train_episodes must leave at least one validation episode")
    train_ids = sorted(shuffled[:train_count])
    validation_ids = sorted(shuffled[train_count:])
    norm_path = _write_norm_stats(output, summaries, set(train_ids))
    transition_total = sum(item["frames"] - 1 for item in summaries)
    clip_total = sum(item["pose_clip_transition_count"] for item in summaries)
    clip_fraction = float(clip_total / max(transition_total, 1))
    timing_max = max(item["transition_period_error_ms"]["max"] for item in summaries)
    included_timing_max = max(
        item["transition_period_error_ms"]["max_included"] for item in summaries
    )
    excluded_transition_count = sum(item["excluded_transition_count"] for item in summaries)
    candidate_training_samples = sum(item["candidate_training_samples"] for item in summaries)
    training_samples = sum(item["training_samples"] for item in summaries)
    roundtrip_max = max(item["roundtrip_matrix_max_abs"] for item in summaries)
    data_identity_payload = {
        "schema": DATASET_SCHEMA,
        "episode_sha256": {item["episode_id"]: item["sha256"] for item in summaries},
        "split_seed": args.split_seed,
        "train": train_ids,
        "validation": validation_ids,
        "instruction": args.instruction,
    }
    manifest = {
        "schema_version": DATASET_SCHEMA,
        "dataset_id": "stackcupanddoll",
        "dataset_identity_sha256": sha256_text(json.dumps(data_identity_payload, sort_keys=True)),
        "source": {
            "path": str(source),
            "format": "3dflow_teleoperation_pickle",
            "control_pose_channels_used": False,
            "tcp_pose_source": "ee_pos_quat interpreted as xyz+rotation_vector",
            "gripper_command_source": "control[6], <0.5=open and >=0.5=close",
        },
        "instruction": args.instruction,
        "target_hz": float(args.target_hz),
        "sampling_mode": "native_capture_order_with_timing_valid_windows",
        "action_horizon": ACTION_HORIZON,
        "execution_horizon": 5,
        "image_contract": {
            "source_order": ["base_rgb", "wrist_rgb"],
            "source_orientation": "as_captured_no_flip",
            "storage": f"JPEG quality={args.jpeg_quality} subsampling=0",
            "model_preprocessing": "resize_with_pad_224_then_bicubic_384_imagenet_norm",
        },
        "state_contract": {
            "representation": "tcp_xyz_rotvec_plus_opposed_finger_positions",
            "gripper_max_opening_m": scales.gripper_max_opening_m,
        },
        "action_contract": {
            "representation": "inv(T_current)@T_next local xyz plus xyz_euler plus gripper",
            "translation_scale_m": scales.translation_m,
            "rotation_scale_rad": scales.rotation_rad,
            "clip_abs": scales.clip_abs,
            "model_positive_gripper_means": "open",
        },
        "splits": {"train": train_ids, "validation": validation_ids},
        "split_seed": int(args.split_seed),
        "episodes": summaries,
        "norm_stats": {"path": norm_path.name, "sha256": sha256_file(norm_path)},
        "audit": {
            "pose_clip_transition_fraction": clip_fraction,
            "pose_clip_transition_count": clip_total,
            "transition_count": transition_total,
            "excluded_transition_count": excluded_transition_count,
            "excluded_transition_fraction": float(excluded_transition_count / max(transition_total, 1)),
            "candidate_training_samples": candidate_training_samples,
            "training_samples": training_samples,
            "excluded_training_samples": candidate_training_samples - training_samples,
            "excluded_training_sample_fraction": float(
                (candidate_training_samples - training_samples) / max(candidate_training_samples, 1)
            ),
            "max_transition_period_error_ms": timing_max,
            "max_included_transition_period_error_ms": included_timing_max,
            "max_pose_roundtrip_matrix_abs": roundtrip_max,
            "max_allowed_pose_clip_fraction": float(args.max_pose_clip_fraction),
            "max_allowed_transition_period_error_ms": float(max_transition_period_error_ms),
        },
    }
    passed = (
        clip_fraction <= args.max_pose_clip_fraction
        and included_timing_max <= max_transition_period_error_ms
        and roundtrip_max <= 1e-6
        and all(item["training_samples"] > 0 for item in summaries)
    )
    manifest["verdict"] = "REAL_DATASET_CONTRACT_PASS" if passed else "REAL_DATASET_CONTRACT_FAIL"
    atomic_write_json(output / "manifest.json", manifest)
    if not passed:
        raise RuntimeError(
            "converted data failed the action/timing contract; inspect manifest.json"
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--expected-episodes", type=int, default=40)
    parser.add_argument("--train-episodes", type=int, default=32)
    parser.add_argument("--split-seed", type=int, default=20260904)
    # The demonstrations and copied UR5e runtime both operate at 15 Hz.  Keeping
    # this rate makes one training action equal one deployed control command.
    parser.add_argument("--target-hz", type=float, default=15.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--translation-scale-m", type=float, default=0.02)
    parser.add_argument("--rotation-scale-rad", type=float, default=0.05)
    parser.add_argument("--clip-abs", type=float, default=1.0)
    parser.add_argument("--gripper-max-opening-m", type=float, default=0.04)
    # Pose-label clipping changes the demonstrated trajectory.  Fail closed by
    # default instead of accepting an arbitrary percentage of distorted labels.
    parser.add_argument("--max-pose-clip-fraction", type=float, default=0.0)
    parser.add_argument(
        "--max-transition-period-error-ms",
        type=float,
        default=None,
        help="exclude transitions farther than this from one nominal period; default is half a period",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    result = convert_dataset(build_parser().parse_args())
    print(json.dumps({"verdict": result["verdict"], "dataset_identity_sha256": result["dataset_identity_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
