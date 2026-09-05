import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from types import SimpleNamespace
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from architectures.simvla.adapters.real_world_training.convert_dataset import (
    convert_dataset,
)
from architectures.simvla.adapters.real_world_training.artifact_validation import (
    validate_real_dataset_manifest,
)
from architectures.simvla.adapters.real_world_training.condition_cache import (
    _AllSplits,
    _building_contract,
    _create_arrays,
    _validate_building_state,
    validate_real_condition_cache,
)
from architectures.simvla.adapters.real_world_training.dataset import (
    RealSimVLADataset,
    align_current_rotvec_proprio,
)
from architectures.simvla.adapters.real_world_training.geometry import (
    RealActionScales,
    apply_normalized_action,
    transition_to_normalized_action,
)
from architectures.simvla.adapters.real_world_training.io_utils import (
    sha256_file,
    sha256_text,
)
from architectures.simvla.adapters.real_world_training.migrate_condition_cache import (
    migrate,
)
from architectures.simvla.adapters.real_world_training.model_io import (
    OfficialBaseIdentity,
    apply_real_action_checkpoint,
    save_real_action_checkpoint,
)
from architectures.simvla.adapters.real_world_training.updater_io import (
    RealConditionConfig,
    RealGenerationConfig,
    load_real_coupled_generation,
    load_real_updater,
    save_real_coupled_generation,
    save_real_updater,
)
from architectures.simvla.adapters.real_world_training.verify_condition_cache import (
    select_episode_records,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    prepare_projection_only_coupling,
)


def test_training_wrapper_supports_one_gpu_with_matched_effective_batch():
    repo = Path(__file__).resolve().parents[2]
    wrapper = repo / "architectures/simvla/wrappers/train_real_stackcupanddoll.sh"
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    source = wrapper.read_text(encoding="utf-8")
    assert '--nproc_per_node="${world_size}"' in source
    assert "gradient_accumulation_steps=$((effective_batch_size / microbatches_per_step))" in source
    assert 'generation_gpu="${gpu_array[1]:-${gpu_array[0]}}"' in source
    assert 'dataset_root="${SIMVLA_REAL_DATASET:-${storage}/dataset_v3}"' in source
    assert 'run_condition_updater 2>&1 | tee "${storage}/logs/04_condition_train.log"' in source
    assert "real_world_training.verify_condition_cache" in source
    assert '--condition-cache-attestation "${condition_cache_attestation}"' in source
    assert 'quarantine_path "${baseline_root}" baseline' in source
    assert 'quarantine_path "${condition_root}" condition_kc2' in source
    assert 'quarantine_path "${generation_root}" generation_ng3' in source
    assert 'quarantine_path "${coupled_root}" coupled_kc2_ng3' in source
    assert "baseline_args" not in source
    assert "condition_args" not in source
    assert "generation_args" not in source
    assert "coupled_args" not in source
    assert "real_world_training.migrate_condition_cache" in source


def test_local_processor_does_not_fall_back_to_hub(tmp_path, monkeypatch):
    from architectures.simvla.adapters.real_world_training import model_io

    calls = []

    class Processor:
        def __init__(self, *, smolvlm_model_path):
            calls.append(smolvlm_model_path)
            raise OSError("incomplete local tokenizer")

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            pytest.fail("upstream fallback factory must not be called")

    monkeypatch.setattr(model_io, "configure_model_imports", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "models.processing_smolvlm_vla",
        SimpleNamespace(SmolVLMVLAProcessor=Processor),
    )
    with pytest.raises(FileNotFoundError, match="local processor snapshot"):
        model_io._load_local_processor(tmp_path / "absent")
    with pytest.raises(OSError, match="incomplete local tokenizer"):
        model_io._load_local_processor(tmp_path)
    assert calls == [str(tmp_path.resolve())]


def test_deploy_wrapper_rejects_sd1_gpu_zero_before_model_or_hardware(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    tools = tmp_path / "bin"
    tools.mkdir()
    hostname = tools / "hostname"
    hostname.write_text("#!/bin/sh\nprintf 'jbrserver1\\n'\n", encoding="utf-8")
    hostname.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "SIMVLA_REAL_CUDA_DEVICE": "0",
        "SIMVLA_REAL_LOG_ROOT": str(tmp_path / "logs"),
    }
    result = subprocess.run(
        [
            "bash",
            str(repo / "architectures/simvla/wrappers/deploy_latentloop_real.sh"),
            "artifact-preflight",
            "--manifest",
            str(repo / "artifacts/simvla/real_world/deployment_manifest.example.json"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "sd1 permits only physical GPU IDs 4,5,6,7" in result.stderr
    assert not (tmp_path / "logs").exists()


def test_interrupted_cache_resume_binds_processor_contents(tmp_path, monkeypatch):
    from architectures.simvla.adapters.real_world_training import condition_cache
    from architectures.simvla.adapters.real_world_training.model_io import (
        OfficialBaseIdentity,
    )

    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "processor_config.json").write_text("v1", encoding="utf-8")
    dataset_manifest = tmp_path / "manifest.json"
    dataset_manifest.write_text("{}", encoding="utf-8")
    norm_stats = tmp_path / "norm.json"
    norm_stats.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        condition_cache,
        "official_base_identity",
        lambda checkpoint, processor_path: OfficialBaseIdentity(
            model_directory=str(checkpoint),
            model_weights_sha256="a" * 64,
            processor_directory=str(processor_path),
            action_mode="libero_joint",
            action_horizon=10,
            transformer_hidden_size=1024,
            transformer_depth=24,
        ),
    )
    kwargs = {
        "dataset_manifest": dataset_manifest,
        "dataset_payload": {"dataset_identity_sha256": "b" * 64},
        "records_sha256": "c" * 64,
        "count": 1,
        "checkpoint": tmp_path / "model",
        "processor": processor,
        "norm_stats": norm_stats,
    }
    contract = _building_contract(**kwargs)
    building = tmp_path / ".cache.building"
    _create_arrays(building, 1)
    (building / "records.json").write_text("records", encoding="utf-8")
    contract["records_sha256"] = sha256_file(building / "records.json")
    (building / "building_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    _validate_building_state(building, contract)

    (processor / "processor_config.json").write_text("v2", encoding="utf-8")
    changed = _building_contract(**{**kwargs, "records_sha256": contract["records_sha256"]})
    with pytest.raises(ValueError, match="different source contract"):
        _validate_building_state(building, changed)


def test_condition_cache_attestation_selects_one_lower_median_per_episode():
    records = [
        {"episode_id": "b", "split": "train", "frame_index": 7, "cache_index": 3},
        {"episode_id": "a", "split": "train", "frame_index": 9, "cache_index": 2},
        {"episode_id": "a", "split": "train", "frame_index": 1, "cache_index": 0},
        {"episode_id": "a", "split": "train", "frame_index": 5, "cache_index": 1},
        {"episode_id": "b", "split": "train", "frame_index": 2, "cache_index": 4},
    ]
    selected = select_episode_records(records, ["a", "b"])
    assert selected == [
        {"episode_id": "a", "split": "train", "frame_index": 5, "cache_index": 1},
        {"episode_id": "b", "split": "train", "frame_index": 2, "cache_index": 4},
    ]


def test_local_delta_pose_round_trip_matches_deployment_composition():
    scales = RealActionScales()
    current = np.asarray([0.45, -0.1, 0.25, 0.2, -0.1, 0.3])
    following = np.asarray([0.451, -0.099, 0.249, 0.201, -0.102, 0.304])
    _, raw = transition_to_normalized_action(current, following, 0.0, scales, clip=False)
    reconstructed = apply_normalized_action(current, raw, scales)
    np.testing.assert_allclose(reconstructed, following, atol=1e-6)
    assert raw[-1] == 1.0


def test_real_condition_proprio_uses_nearest_equivalent_rotvec_branch():
    previous = torch.tensor([[0.4, 0.0, 0.2, 0.0, 0.0, 3.13, 0.02, -0.02]])
    current = torch.tensor([[0.4, 0.0, 0.2, 0.0, 0.0, -3.13, 0.02, -0.02]])
    aligned = align_current_rotvec_proprio(previous, current)
    assert float(torch.linalg.vector_norm(aligned[:, 3:6] - previous[:, 3:6])) < 0.03
    np.testing.assert_allclose(aligned[:, :3], current[:, :3])
    np.testing.assert_allclose(aligned[:, 6:], current[:, 6:])


def _write_episode(root: Path, episode: str, offset: float, frames: int = 14) -> None:
    directory = root / episode
    directory.mkdir(parents=True)
    started = datetime(2026, 9, 4, 12, 0, 0)
    for index in range(frames):
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
        max_transition_period_error_ms=None,
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


def test_conversion_pairs_observation_t_with_gripper_command_t(tmp_path):
    source = tmp_path / "raw"
    _write_episode(source, "episode_a", 0.0)
    _write_episode(source, "episode_b", 0.1)
    for episode in ("episode_a", "episode_b"):
        path = sorted((source / episode).glob("*.pkl"))[5]
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        payload["control"][6] = 0.75
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    output = tmp_path / "converted"
    args = argparse.Namespace(
        source=str(source), output=str(output), instruction="stack the cup and doll",
        expected_episodes=2, train_episodes=1, split_seed=7, target_hz=15.0,
        jpeg_quality=95, translation_scale_m=0.02, rotation_scale_rad=0.05,
        clip_abs=1.0, gripper_max_opening_m=0.04, max_pose_clip_fraction=0.0,
        max_transition_period_error_ms=None, resume=False,
    )
    manifest = convert_dataset(args)
    assert manifest["source"]["observation_action_alignment"].startswith(
        "frame_t stores observation_t and command_t"
    )
    episode = next(item for item in manifest["episodes"] if item["episode_id"] == "episode_a")
    with h5py.File(output / episode["path"], "r") as handle:
        assert float(handle["step_action"][4, 6]) == 1.0
        assert float(handle["step_action"][5, 6]) == -0.5
        assert float(handle["gripper_command"][5]) == 0.75
    assert manifest["audit"]["gripper_label_alignment_error_count"] == 0


def test_full_dataset_validator_recomputes_pose_labels_and_norm_stats(tmp_path):
    source = tmp_path / "raw"
    for index in range(40):
        _write_episode(source, f"episode_{index:02d}", index * 0.001)
    output = tmp_path / "converted"
    args = argparse.Namespace(
        source=str(source), output=str(output), instruction="stack the cup and doll",
        expected_episodes=40, train_episodes=32, split_seed=20260904,
        target_hz=15.0, jpeg_quality=95, translation_scale_m=0.02,
        rotation_scale_rad=0.05, clip_abs=1.0, gripper_max_opening_m=0.04,
        max_pose_clip_fraction=0.0, max_transition_period_error_ms=None,
        resume=False,
    )
    manifest = convert_dataset(args)
    validated = validate_real_dataset_manifest(
        output / "manifest.json", verify_episode_checksums=True
    )
    assert validated["dataset_identity_sha256"] == manifest["dataset_identity_sha256"]

    episode_path = output / manifest["episodes"][0]["path"]
    with h5py.File(episode_path, "r+") as handle:
        handle["raw_normalized_step_action"][0, 0] += np.float32(0.1)
    manifest_path = output / "manifest.json"
    changed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_manifest["episodes"][0]["size_bytes"] = episode_path.stat().st_size
    manifest_path.write_text(json.dumps(changed_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="measured TCP transitions"):
        validate_real_dataset_manifest(
            manifest_path, verify_episode_checksums=False
        )


def test_conversion_preserves_native_frames_and_excludes_gap_crossing_windows(tmp_path):
    source = tmp_path / "raw"
    _write_episode(source, "episode_a", 0.0, frames=24)
    _write_episode(source, "episode_b", 0.1, frames=24)
    episode = source / "episode_a"
    files = sorted(episode.glob("*.pkl"))
    for path in files[12:]:
        payload = path.read_bytes()
        timestamp = datetime.fromisoformat(path.stem) + timedelta(milliseconds=40)
        path.unlink()
        (episode / f"{timestamp.isoformat(timespec='microseconds')}.pkl").write_bytes(payload)

    output = tmp_path / "converted"
    args = argparse.Namespace(
        source=str(source), output=str(output), instruction="stack the cup and doll",
        expected_episodes=2, train_episodes=1, split_seed=7, target_hz=15.0,
        jpeg_quality=95, translation_scale_m=0.02, rotation_scale_rad=0.05,
        clip_abs=1.0, gripper_max_opening_m=0.04, max_pose_clip_fraction=0.0,
        max_transition_period_error_ms=None, resume=False,
    )
    manifest = convert_dataset(args)
    summary = next(
        item for item in manifest["episodes"] if item["episode_id"] == "episode_a"
    )
    assert summary["source_frames"] == 24
    assert summary["frames"] == 24
    assert summary["excluded_transition_count"] == 1
    assert summary["training_samples"] == 4
    with h5py.File(output / summary["path"], "r") as handle:
        np.testing.assert_array_equal(handle["source_index"][:], np.arange(24))


def _legacy_dataset_and_cache_from_corrected(
    corrected_root: Path, legacy_dataset_root: Path, legacy_cache_root: Path
) -> None:
    shutil.copytree(corrected_root, legacy_dataset_root)
    manifest_path = legacy_dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "simvla_real_hdf5_v2"
    manifest["source"].pop("observation_action_alignment", None)
    manifest["source"]["gripper_command_source"] = (
        "control[6], <0.5=open and >=0.5=close"
    )
    manifest["action_contract"].pop("pose_label_source", None)
    manifest["action_contract"].pop("gripper_label_source", None)
    manifest["state_contract"].pop("condition_updater_rotation_delta", None)
    manifest["audit"].pop("gripper_label_alignment_error_count", None)
    for episode in manifest["episodes"]:
        path = legacy_dataset_root / episode["path"]
        with h5py.File(path, "r+") as handle:
            handle.attrs["schema_version"] = "simvla_real_hdf5_v2"
            command = np.asarray(handle["gripper_command"], dtype=np.float32)
            following = np.where(command[1:] < 0.5, 1.0, -1.0).astype(np.float32)
            handle["step_action"][:, 6] = following
            handle["raw_normalized_step_action"][:, 6] = following
            del handle["gripper_command"]
            del handle.attrs["observation_action_alignment"]
        episode["sha256"] = sha256_file(path)
        episode.pop("gripper_label_alignment_error_count", None)
    identity = {
        "schema": manifest["schema_version"],
        "episode_sha256": {
            item["episode_id"]: item["sha256"] for item in manifest["episodes"]
        },
        "split_seed": manifest["split_seed"],
        "train": manifest["splits"]["train"],
        "validation": manifest["splits"]["validation"],
        "instruction": manifest["instruction"],
    }
    manifest["dataset_identity_sha256"] = sha256_text(
        json.dumps(identity, sort_keys=True)
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    corrected = _AllSplits(corrected_root / "manifest.json")
    records = corrected.records()
    legacy_cache_root.mkdir()
    (legacy_cache_root / "records.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    count = len(records)
    condition = np.arange(count * 122 * 960, dtype=np.float32).reshape(
        count, 122, 960
    )
    proprio = np.empty((count, 8), dtype=np.float32)
    action = np.empty((count, 10, 7), dtype=np.float32)
    legacy_paths = {
        item["episode_id"]: legacy_dataset_root / item["path"]
        for item in manifest["episodes"]
    }
    for row in records:
        index = int(row["cache_index"])
        with h5py.File(legacy_paths[row["episode_id"]], "r") as handle:
            frame = int(row["frame_index"])
            proprio[index] = handle["state"][frame]
            action[index] = handle["step_action"][frame : frame + 10]
    np.save(legacy_cache_root / "condition.npy", condition)
    np.save(legacy_cache_root / "proprio.npy", proprio)
    np.save(legacy_cache_root / "action.npy", action)
    np.save(legacy_cache_root / "complete.npy", np.ones(count, dtype=np.uint8))
    arrays = {
        name: {
            "sha256": sha256_file(legacy_cache_root / name),
            "size_bytes": (legacy_cache_root / name).stat().st_size,
        }
        for name in ("condition.npy", "proprio.npy", "action.npy", "complete.npy")
    }
    records_sha = sha256_file(legacy_cache_root / "records.json")
    official_sha = "a" * 64
    cache_identity = {
        "schema": "simvla_real_condition_cache_v1",
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "official_model_weights_sha256": official_sha,
        "records_sha256": records_sha,
        "preprocessing": manifest["image_contract"]["model_preprocessing"],
    }
    cache_manifest = {
        "schema_version": "simvla_real_condition_cache_v1",
        "verdict": "REAL_CONDITION_CACHE_PASS",
        "count": count,
        "shape": [count, 122, 960],
        "dtype": "float32",
        "dataset_manifest": str(manifest_path),
        "dataset_identity_sha256": manifest["dataset_identity_sha256"],
        "official_base": {"model_weights_sha256": official_sha},
        "exact_loading": {"verdict": "EXACT_OFFICIAL_INITIALIZATION_PASS"},
        "records_sha256": records_sha,
        "condition_cache_identity_sha256": sha256_text(
            json.dumps(cache_identity, sort_keys=True)
        ),
        "arrays": arrays,
        "token_layout": {
            "valid_mask": [[True] * 122],
            "group_ids": [[0] * 122],
            "group_names": {"0": "padding"},
            "image_tokens_per_view": 49,
            "text_tokens": 24,
            "sample_ranges": [
                {
                    "sample": 0,
                    "image_views": [
                        {"view": 0, "start": 0, "end": 49},
                        {"view": 1, "start": 49, "end": 98},
                    ],
                    "language": {"start": 98, "end": 122},
                    "batch_padding": {"start": 122, "end": 122},
                }
            ],
            "source_attention_quirk": "test fixture",
        },
    }
    (legacy_cache_root / "manifest.json").write_text(
        json.dumps(cache_manifest), encoding="utf-8"
    )


def test_condition_cache_label_migration_reuses_only_identical_vlm_inputs(tmp_path):
    source = tmp_path / "raw"
    _write_episode(source, "episode_a", 0.0)
    _write_episode(source, "episode_b", 0.1)
    for episode in ("episode_a", "episode_b"):
        path = sorted((source / episode).glob("*.pkl"))[5]
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        payload["control"][6] = 1.0
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
    corrected = tmp_path / "corrected"
    convert_dataset(
        argparse.Namespace(
            source=str(source), output=str(corrected), instruction="stack the cup and doll",
            expected_episodes=2, train_episodes=1, split_seed=7, target_hz=15.0,
            jpeg_quality=95, translation_scale_m=0.02, rotation_scale_rad=0.05,
            clip_abs=1.0, gripper_max_opening_m=0.04, max_pose_clip_fraction=0.0,
            max_transition_period_error_ms=None, resume=False,
        )
    )
    legacy_dataset = tmp_path / "legacy_dataset"
    legacy_cache = tmp_path / "legacy_cache"
    _legacy_dataset_and_cache_from_corrected(
        corrected, legacy_dataset, legacy_cache
    )
    migrated = tmp_path / "migrated_cache"
    result = migrate(
        argparse.Namespace(
            legacy_condition_cache=str(legacy_cache),
            corrected_dataset_manifest=str(corrected / "manifest.json"),
            output=str(migrated),
            allow_condition_copy=False,
        )
    )
    validate_real_condition_cache(migrated, verify_array_checksums=True)
    assert result["migration"]["condition_storage_mode"] == "hardlink"
    assert result["migration"]["condition_inputs_bitwise_equal"] is True
    assert result["migration"]["changed_gripper_transition_count"] == 4
    assert (legacy_cache / "condition.npy").stat().st_ino == (
        migrated / "condition.npy"
    ).stat().st_ino
    assert not np.array_equal(
        np.load(legacy_cache / "action.npy"), np.load(migrated / "action.npy")
    )


def test_condition_cache_label_migration_rejects_changed_images(tmp_path):
    source = tmp_path / "raw"
    _write_episode(source, "episode_a", 0.0)
    _write_episode(source, "episode_b", 0.1)
    corrected = tmp_path / "corrected"
    convert_dataset(
        argparse.Namespace(
            source=str(source), output=str(corrected), instruction="stack the cup and doll",
            expected_episodes=2, train_episodes=1, split_seed=7, target_hz=15.0,
            jpeg_quality=95, translation_scale_m=0.02, rotation_scale_rad=0.05,
            clip_abs=1.0, gripper_max_opening_m=0.04, max_pose_clip_fraction=0.0,
            max_transition_period_error_ms=None, resume=False,
        )
    )
    legacy_dataset = tmp_path / "legacy_dataset"
    legacy_cache = tmp_path / "legacy_cache"
    _legacy_dataset_and_cache_from_corrected(
        corrected, legacy_dataset, legacy_cache
    )
    first = corrected / json.loads(
        (corrected / "manifest.json").read_text(encoding="utf-8")
    )["episodes"][0]["path"]
    with h5py.File(first, "r+") as handle:
        payload = np.asarray(handle["base_rgb_jpeg"][0]).copy()
        payload[-1] ^= np.uint8(1)
        handle["base_rgb_jpeg"][0] = payload
    corrected_manifest_path = corrected / "manifest.json"
    corrected_manifest = json.loads(corrected_manifest_path.read_text(encoding="utf-8"))
    changed_episode = corrected_manifest["episodes"][0]
    changed_episode["sha256"] = sha256_file(corrected / changed_episode["path"])
    corrected_identity = {
        "schema": corrected_manifest["schema_version"],
        "episode_sha256": {
            item["episode_id"]: item["sha256"]
            for item in corrected_manifest["episodes"]
        },
        "split_seed": corrected_manifest["split_seed"],
        "train": corrected_manifest["splits"]["train"],
        "validation": corrected_manifest["splits"]["validation"],
        "instruction": corrected_manifest["instruction"],
    }
    corrected_manifest["dataset_identity_sha256"] = sha256_text(
        json.dumps(corrected_identity, sort_keys=True)
    )
    corrected_manifest_path.write_text(json.dumps(corrected_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="changed condition input base_rgb_jpeg"):
        migrate(
            argparse.Namespace(
                legacy_condition_cache=str(legacy_cache),
                corrected_dataset_manifest=str(corrected / "manifest.json"),
                output=str(tmp_path / "must_not_exist"),
                allow_condition_copy=False,
            )
        )


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
        condition_cache_attestation_identity_sha256="a",
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


def test_real_updater_loader_rejects_unknown_kind(tmp_path):
    checkpoint = tmp_path / "updater.pt"
    checkpoint.write_bytes(b"not-loaded")
    with pytest.raises(ValueError, match="unsupported real updater kind"):
        load_real_updater(checkpoint, kind="typo", device="cpu")


def test_coupled_checkpoint_save_enforces_projection_only_change(tmp_path):
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"baseline")
    cache_manifest = tmp_path / "cache_manifest.json"
    cache_manifest.write_text("{}", encoding="utf-8")
    condition_config = RealConditionConfig()
    generation_config = RealGenerationConfig()
    condition = save_real_updater(
        tmp_path / "condition.pt",
        kind="condition",
        updater=condition_config.build(),
        config=condition_config,
        baseline_action_checkpoint=baseline,
        norm_stats_sha256="norm",
        dataset_identity_sha256="dataset",
        condition_cache_identity_sha256="cache",
        condition_cache_attestation_identity_sha256="attestation",
        optimizer_step=10_000,
        objective={},
        validation={},
    )
    parent_module = generation_config.build()
    with torch.no_grad():
        parent_module.condition_code_projection.weight.fill_(0.25)
    parent = save_real_updater(
        tmp_path / "generation.pt",
        kind="generation",
        updater=parent_module,
        config=generation_config,
        baseline_action_checkpoint=baseline,
        norm_stats_sha256="norm",
        dataset_identity_sha256="dataset",
        condition_cache_identity_sha256="cache",
        condition_cache_attestation_identity_sha256="attestation",
        optimizer_step=10_000,
        objective={},
        validation={},
    )
    candidate, _ = load_real_updater(
        parent, kind="generation", device="cpu"
    )
    coupling = prepare_projection_only_coupling(candidate)
    training_config = {
        "condition_refresh_interval": 2,
        "generation_full_evaluations": 3,
        "full_step_indices": [0, 4, 8],
        "trainable_parameter_names": ["condition_code_projection.weight"],
        "trainable_parameters": 16_384,
        "condition_change_code": "condition_updater_delta_encoder",
    }
    assert coupling["trainable_parameters"] == 16_384
    coupled = save_real_coupled_generation(
        tmp_path / "coupled.pt",
        updater=candidate,
        config=generation_config,
        parent_generation_checkpoint=parent,
        condition_updater_checkpoint=condition,
        condition_cache_manifest=cache_manifest,
        optimizer_step=10_000,
        training_config=training_config,
        validation={},
    )
    _, payload = load_real_coupled_generation(
        coupled, device="cpu", expected_optimizer_step=10_000
    )
    assert payload["projection_only_state_audit"]["verdict"] == (
        "PROJECTION_ONLY_STATE_PASS"
    )

    changed = next(
        parameter
        for name, parameter in candidate.named_parameters()
        if name != "condition_code_projection.weight"
    )
    with torch.no_grad():
        changed.add_(1.0)
    with pytest.raises(ValueError, match="outside condition_code_projection.weight"):
        save_real_coupled_generation(
            tmp_path / "invalid_coupled.pt",
            updater=candidate,
            config=generation_config,
            parent_generation_checkpoint=parent,
            condition_updater_checkpoint=condition,
            condition_cache_manifest=cache_manifest,
            optimizer_step=10_000,
            training_config=training_config,
            validation={},
        )
