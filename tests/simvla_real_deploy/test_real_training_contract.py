import argparse
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from architectures.simvla.adapters.real_world_training.convert_dataset import (
    convert_dataset,
)
from architectures.simvla.adapters.real_world_training.dataset import RealSimVLADataset
from architectures.simvla.adapters.real_world_training.geometry import (
    RealActionScales,
    apply_normalized_action,
    transition_to_normalized_action,
)
from architectures.simvla.adapters.real_world_training.io_utils import sha256_file
from architectures.simvla.adapters.real_world_training.model_io import (
    OfficialBaseIdentity,
    apply_real_action_checkpoint,
    save_real_action_checkpoint,
)
from architectures.simvla.adapters.real_world_training.updater_io import (
    RealConditionConfig,
    load_real_updater,
    save_real_updater,
)


def test_local_delta_pose_round_trip_matches_deployment_composition():
    scales = RealActionScales()
    current = np.asarray([0.45, -0.1, 0.25, 0.2, -0.1, 0.3])
    following = np.asarray([0.451, -0.099, 0.249, 0.201, -0.102, 0.304])
    _, raw = transition_to_normalized_action(current, following, 0.0, scales, clip=False)
    reconstructed = apply_normalized_action(current, raw, scales)
    np.testing.assert_allclose(reconstructed, following, atol=1e-6)
    assert raw[-1] == 1.0


def _write_episode(root: Path, episode: str, offset: float) -> None:
    directory = root / episode
    directory.mkdir(parents=True)
    started = datetime(2026, 9, 4, 12, 0, 0)
    for index in range(14):
        timestamp = started + timedelta(seconds=index / 15.0)
        payload = {
            "base_rgb": np.full((12, 16, 3), index, dtype=np.uint8),
            "wrist_rgb": np.full((12, 16, 3), 255 - index, dtype=np.uint8),
            "ee_pos_quat": np.asarray([offset + index * 0.001, 0, 0.2, 0, 0, 0]),
            "gripper_position": np.asarray([0.0]),
            "control": np.asarray([0, 0, 0, 0, 0, 0, 0.0]),
        }
        with (directory / f"{timestamp.isoformat(timespec='microseconds')}.pkl").open("wb") as handle:
            pickle.dump(payload, handle)


def test_conversion_uses_episode_split_and_h10_local_actions(tmp_path):
    source = tmp_path / "raw"
    _write_episode(source, "episode_a", 0.0)
    _write_episode(source, "episode_b", 0.1)
    output = tmp_path / "converted"
    args = argparse.Namespace(
        source=str(source),
        output=str(output),
        instruction="stack the cup and doll",
        expected_episodes=2,
        train_episodes=1,
        split_seed=7,
        target_hz=15.0,
        jpeg_quality=95,
        translation_scale_m=0.02,
        rotation_scale_rad=0.05,
        clip_abs=1.0,
        gripper_max_opening_m=0.04,
        max_pose_clip_fraction=0.0,
        max_resample_error_ms=34.0,
        resume=False,
    )
    manifest = convert_dataset(args)
    assert manifest["verdict"] == "REAL_DATASET_CONTRACT_PASS"
    assert len(manifest["splits"]["train"]) == 1
    assert len(manifest["splits"]["validation"]) == 1
    assert set(manifest["splits"]["train"]).isdisjoint(manifest["splits"]["validation"])
    sample = RealSimVLADataset(output / "manifest.json", split="train", training=False)[0]
    assert sample["image_input"].shape == (3, 3, 384, 384)
    assert sample["action"].shape == (10, 7)
    np.testing.assert_allclose(sample["action"][:, 0], 0.05, atol=1e-5)


class _TinyModel:
    def __init__(self):
        self.transformer = torch.nn.Linear(3, 2)


def test_real_action_overlay_is_strict_and_source_locked(tmp_path):
    norm = tmp_path / "norm.json"
    norm.write_text("{}", encoding="utf-8")
    source = _TinyModel()
    base = OfficialBaseIdentity("base", "a" * 64, "processor", "libero_joint", 10, 3, 1)
    checkpoint = save_real_action_checkpoint(
        tmp_path / "real_action.pt",
        transformer=source.transformer,
        official_base=base,
        norm_stats_path=norm,
        dataset_identity_sha256="dataset",
        optimizer_step=3,
        training_config={},
        validation={},
    )
    target = _TinyModel()
    report = apply_real_action_checkpoint(
        target,
        checkpoint,
        expected_base_sha256="a" * 64,
        expected_norm_sha256=sha256_file(norm),
    )
    assert report["strict_state_dict_load"]
    for left, right in zip(source.transformer.parameters(), target.transformer.parameters()):
        assert torch.equal(left, right)


def test_real_condition_updater_rejects_wrong_baseline(tmp_path):
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    config = RealConditionConfig()
    updater = config.build()
    checkpoint = save_real_updater(
        tmp_path / "condition.pt",
        kind="condition",
        updater=updater,
        config=config,
        baseline_action_checkpoint=baseline,
        norm_stats_sha256="n",
        dataset_identity_sha256="d",
        condition_cache_identity_sha256="c",
        optimizer_step=1,
        objective={},
        validation={},
    )
    loaded, _ = load_real_updater(
        checkpoint,
        kind="condition",
        device="cpu",
        expected_baseline_sha256=sha256_file(baseline),
    )
    assert loaded.parameter_audit()["total"] == updater.parameter_audit()["total"]

