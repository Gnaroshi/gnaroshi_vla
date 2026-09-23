#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


REQUIRED_STEP_FILES = ("image_primary.jpg", "image_wrist.jpg", "other.npz")
REQUIRED_NPZ_KEYS = (
    "gripper_pose",
    "gripper_open_state",
    "joints",
    "action_gripper_pose",
    "delta_cur_2_last_action",
    "language_instruction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(label: str, path: Path, expected: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected={expected}, actual={actual}, path={path}"
        )
    print(f"[VERIFY][OK] {label} sha256={actual} bytes={path.stat().st_size}")


def scalar_string(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def verify_dataset(args: argparse.Namespace) -> None:
    dataset_path = args.dataset_root / args.dataset_name
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"nested dataset path is missing: {dataset_path}")

    with args.data_info.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if len(records) != args.expected_episodes:
        raise RuntimeError(
            f"episode count mismatch: expected={args.expected_episodes}, actual={len(records)}"
        )

    expected_ids = {f"0000/{index:06d}" for index in range(args.expected_episodes)}
    actual_ids = {str(record[0]) for record in records}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise RuntimeError(f"episode IDs mismatch: missing={missing}, extra={extra}")

    total_frames = sum(int(record[1]) for record in records)
    total_windows = sum(len(record) - 2 for record in records)
    if total_frames != args.expected_total_frames:
        raise RuntimeError(
            f"metadata frame total mismatch: expected={args.expected_total_frames}, actual={total_frames}"
        )
    if total_windows != args.expected_train_windows:
        raise RuntimeError(
            f"training-window total mismatch: expected={args.expected_train_windows}, actual={total_windows}"
        )

    checked_steps = 0
    sampled_images = []
    for record in records:
        episode_id = str(record[0])
        expected_windows = int(record[1]) - args.window_size
        windows = record[2:]
        if len(windows) != expected_windows:
            raise RuntimeError(
                f"window count mismatch for {episode_id}: expected={expected_windows}, actual={len(windows)}"
            )

        referenced_steps = set()
        for pair in windows:
            if not isinstance(pair, list) or len(pair) != 2:
                raise RuntimeError(f"invalid window record for {episode_id}: {pair!r}")
            start, end = map(int, pair)
            if start < 0 or end - start != args.window_size:
                raise RuntimeError(
                    f"invalid window range for {episode_id}: start={start}, end={end}"
                )
            referenced_steps.update(range(start, end))

        if not referenced_steps:
            raise RuntimeError(f"no referenced steps for {episode_id}")
        episode_steps = dataset_path / episode_id / "steps"
        for step in sorted(referenced_steps):
            step_dir = episode_steps / f"{step:04d}"
            for filename in REQUIRED_STEP_FILES:
                path = step_dir / filename
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"missing dataset file: {path}")
        checked_steps += len(referenced_steps)

        sample_step = episode_steps / f"{min(referenced_steps):04d}"
        with np.load(sample_step / "other.npz", allow_pickle=True) as payload:
            missing_keys = [key for key in REQUIRED_NPZ_KEYS if key not in payload.files]
            if missing_keys:
                raise RuntimeError(f"missing NPZ keys in {sample_step}: {missing_keys}")
            instruction = scalar_string(payload["language_instruction"])
        if instruction != args.expected_instruction:
            raise RuntimeError(
                f"instruction mismatch in {episode_id}: expected={args.expected_instruction!r}, "
                f"actual={instruction!r}"
            )
        sampled_images.extend(
            [sample_step / "image_primary.jpg", sample_step / "image_wrist.jpg"]
        )

    for image_path in (sampled_images[0], sampled_images[1], sampled_images[-2], sampled_images[-1]):
        with Image.open(image_path) as image:
            if image.size != (640, 480) or image.mode != "RGB":
                raise RuntimeError(
                    f"unexpected image format: path={image_path}, size={image.size}, mode={image.mode}"
                )

    print(
        "[VERIFY][OK] dataset "
        f"task={args.task} episodes={len(records)} metadata_frames={total_frames} "
        f"training_windows={total_windows} unique_episode_steps_checked={checked_steps} "
        "image_size=640x480 image_mode=RGB instruction=exact"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        required=True,
        choices=("doll", "cabinet", "rings", "stacking_cups"),
    )
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--vit-checkpoint", required=True, type=Path)
    parser.add_argument("--vit-sha256", required=True)
    parser.add_argument("--clip-checkpoint", required=True, type=Path)
    parser.add_argument("--clip-sha256", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-info", required=True, type=Path)
    parser.add_argument("--data-info-sha256", required=True)
    parser.add_argument("--expected-instruction", required=True)
    parser.add_argument("--expected-episodes", required=True, type=int)
    parser.add_argument("--expected-total-frames", required=True, type=int)
    parser.add_argument("--expected-train-windows", required=True, type=int)
    parser.add_argument("--window-size", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_hash("teacher", args.teacher_checkpoint, args.teacher_sha256)
    verify_hash("ViT-MAE", args.vit_checkpoint, args.vit_sha256)
    verify_hash("CLIP", args.clip_checkpoint, args.clip_sha256)
    verify_hash("data_info", args.data_info, args.data_info_sha256)
    verify_dataset(args)
    print(f"[PREFLIGHT][ASSETS][PASS] task={args.task}")


if __name__ == "__main__":
    main()
