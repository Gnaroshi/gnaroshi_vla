#!/usr/bin/env python3
"""Lock checkpoints, evaluator semantics, source, and paired episode identity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


EXPECTED_TEACHER_SHA256 = "a999bf839acfb6f77beb8b86576933254f1981d2bacd1f0d269da093d7205cc5"
EXPECTED_ADAPTER_SHA256 = "badc74e135003fee91ccc69c76fe4f225aece856f487236ca7f626424504f132"
EXPECTED_VIT_SHA256 = "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"
EXPECTED_DATA_INFO_SHA256 = "4b8241c1dd39b62c56aa6bbd7dca1afb397a4f9862e74f7718a3b00ca4679120"
EXPECTED_DATA_META_SHA256 = "08f765dbc4695e9618517762a4e1b297d3062a485a06f15ff52f6e52000e3352"
EXPECTED_TASK_ORDER = (
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
)
LIBERO_DATASET_NAME = "libero_10_converted"

SOURCE_FILES = (
    "architectures/seer/upstream/train.py",
    "architectures/seer/upstream/eval_libero.py",
    "architectures/seer/upstream/models/seer_model.py",
    "architectures/seer/upstream/models/lrnode_modules.py",
    "architectures/seer/upstream/data_info/libero_10_converted.json",
    "architectures/seer/upstream/utils/arguments_utils.py",
    "architectures/seer/upstream/utils/data_utils.py",
    "architectures/seer/upstream/utils/train_utils.py",
    "architectures/seer/upstream/utils/eval_utils_libero.py",
    "architectures/seer/upstream/utils/lrnode_mechanism_utils.py",
    "architectures/seer/upstream/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh",
    "architectures/seer/wrappers/lrnode/distill_node.sh",
    "architectures/seer/wrappers/lrnode/train_action_correction_baseline.sh",
    "architectures/seer/wrappers/lrnode/train_nonrecurrent_latent_baseline.sh",
    "architectures/seer/wrappers/lrnode/eval_latentloop_comparison.sh",
    "architectures/seer/wrappers/lrnode/run_latentloop_comparison_sequential.sh",
    "architectures/seer/adapters/latentloop_comparison/action_token_alignment.py",
    "architectures/seer/adapters/latentloop_comparison/__init__.py",
    "architectures/seer/adapters/latentloop_comparison/factory.py",
    "architectures/seer/adapters/latentloop_comparison/seer_action_correction.py",
    "architectures/seer/adapters/latentloop_comparison/seer_nonrecurrent_latent.py",
    "architectures/seer/adapters/latentloop_comparison/seer_teacher_pairs.py",
    "methods/latentloop_comparison/action_space_correction.py",
    "methods/latentloop_comparison/__init__.py",
    "methods/latentloop_comparison/nonrecurrent_latent.py",
    "methods/latentloop_comparison/training_losses.py",
    "methods/latentloop_comparison/fairness.py",
    "methods/latentloop_comparison/decisions.py",
    "methods/latentloop_comparison/metrics.py",
    "tools/seer/lock_latentloop_comparison_source.py",
    "tools/seer/calibrate_latentloop_comparison_losses.py",
    "tools/seer/audit_action_token_alignment.py",
    "tools/seer/evaluate_latentloop_comparison_offline.py",
    "tools/seer/select_latentloop_comparison_checkpoint.py",
    "tools/seer/aggregate_latentloop_comparison.py",
    "tools/seer/apply_seer_comparison_decisions.py",
    "tools/seer/check_latentloop_comparison_k1_parity.py",
    "tests/test_latentloop_comparison.py",
)

LOCK_OUTPUT_FILES = (
    "source_lock_report.md",
    "source_lock_manifest.json",
    "source_hashes.json",
    "canonical_episode_manifest.csv",
)


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into accelerator memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_dataset_path(dataset_root: Path) -> Path:
    """Return the path produced by Seer's ``root_dir/dataset_name`` lookup."""

    return dataset_root / LIBERO_DATASET_NAME


def _refuse_output(output_dir: Path) -> None:
    collisions = [
        output_dir / name for name in LOCK_OUTPUT_FILES if (output_dir / name).exists()
    ]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite source lock files: {collisions}")
    output_dir.mkdir(parents=True, exist_ok=True)


def lock_source(
    repo_root: Path,
    output_dir: Path,
    teacher: Path,
    adapter: Path,
    canonical_row: Path,
    vit_checkpoint: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    """Validate and persist the complete primary comparison identity."""

    _refuse_output(output_dir)
    teacher_sha = sha256_file(teacher)
    adapter_sha = sha256_file(adapter)
    if teacher_sha != EXPECTED_TEACHER_SHA256:
        raise RuntimeError(f"Local teacher-33 SHA mismatch: {teacher_sha}")
    if adapter_sha != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError(f"Local adapter-39 SHA mismatch: {adapter_sha}")
    vit_sha = sha256_file(vit_checkpoint)
    if vit_sha != EXPECTED_VIT_SHA256:
        raise RuntimeError(f"MAE-ViT checkpoint SHA mismatch: {vit_sha}")
    data_info = (
        repo_root
        / "architectures/seer/upstream/data_info/libero_10_converted.json"
    )
    data_info_sha = sha256_file(data_info)
    if data_info_sha != EXPECTED_DATA_INFO_SHA256:
        raise RuntimeError(f"LIBERO data-info SHA mismatch: {data_info_sha}")
    runtime_dataset_path = _runtime_dataset_path(dataset_root)
    episode_root = runtime_dataset_path / "episodes"
    if not episode_root.is_dir():
        raise FileNotFoundError(
            "Seer runtime dataset root is invalid; expected episodes at "
            f"{episode_root}. Pass ROOT_DIR as the parent of the inner "
            f"{LIBERO_DATASET_NAME}/ directory."
        )
    data_meta = runtime_dataset_path / "meta_info.h5"
    if not data_meta.is_file():
        raise FileNotFoundError(data_meta)
    data_meta_sha = sha256_file(data_meta)
    if data_meta_sha != EXPECTED_DATA_META_SHA256:
        raise RuntimeError(f"LIBERO dataset meta SHA mismatch: {data_meta_sha}")

    row = json.loads(canonical_row.read_text(encoding="utf-8"))
    required_row = {
        "num_tasks": 10,
        "episodes_per_task": 20,
        "actual_episodes": 200,
        "action_pred_steps": 3,
        "segment_length": 4,
        "feedback_schedule": "dense",
    }
    for key, expected in required_row.items():
        if row.get(key) != expected:
            raise RuntimeError(
                f"Canonical row mismatch for {key}: {row.get(key)!r} != {expected!r}"
            )
    renderer = row.get("renderer_backend", {})
    if renderer.get("effective_backend") != "osmesa" or not renderer.get(
        "actual_context_verified"
    ):
        raise RuntimeError("Canonical row does not have verified OSMesa rendering")
    if row.get("baseline_checkpoint_sha256") != teacher_sha:
        raise RuntimeError("Canonical row does not reference local teacher 33")
    if row.get("adapter_checkpoint_sha256") != adapter_sha:
        raise RuntimeError("Canonical row does not reference local adapter 39")

    episode_path = Path(row["episode_metrics_path"])
    if not episode_path.is_file():
        raise FileNotFoundError(episode_path)
    episodes: list[dict[str, Any]] = []
    with episode_path.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            episodes.append(
                {
                    "task_id": int(item["task_id"]),
                    "task_name": item["task_name"],
                    "episode_id": int(item["episode_id"]),
                    "seed": int(item["seed"]),
                    "episode_key": f"{int(item['task_id'])}:{int(item['episode_id'])}:{int(item['seed'])}",
                }
            )
    if len(episodes) != 200 or len({row["episode_key"] for row in episodes}) != 200:
        raise RuntimeError("Canonical episode manifest is not 200 unique episode keys")
    observed_order = tuple(
        next(item["task_name"] for item in episodes if item["task_id"] == task_id)
        for task_id in range(10)
    )
    if observed_order != EXPECTED_TASK_ORDER:
        raise RuntimeError("Canonical LIBERO-LONG task order changed")
    for task_id in range(10):
        ids = sorted(
            item["episode_id"] for item in episodes if item["task_id"] == task_id
        )
        if ids != list(range(20)):
            raise RuntimeError(f"Task {task_id} episode IDs are not 0..19")

    mechanism_source = (
        repo_root / "architectures/seer/upstream/utils/lrnode_mechanism_utils.py"
    ).read_text(encoding="utf-8")
    evaluator_source = (
        repo_root / "architectures/seer/upstream/utils/eval_utils_libero.py"
    ).read_text(encoding="utf-8")
    semantic_fragments = {
        "temporal_validity": "torch.all(candidates != 0, dim=1)",
        "temporal_weights": "weights = np.exp(-float(temperature) * np.arange(len(candidates)))",
        "temporal_numpy_to_torch": "torch.from_numpy(weights)",
        "gripper_threshold": "action[:, 6:] > 0.5",
        "gripper_remap": "action[:, -1] = (action[:, -1] - 0.5) * 2",
    }
    for name, fragment in semantic_fragments.items():
        haystack = mechanism_source if name.startswith("temporal") else evaluator_source
        if fragment not in haystack:
            raise RuntimeError(f"Evaluator semantic lock failed: {name}")

    missing = [relative for relative in SOURCE_FILES if not (repo_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Comparison source files are missing: {missing}")
    source_hashes = {
        relative: sha256_file(repo_root / relative) for relative in SOURCE_FILES
    }
    status = _git(repo_root, "status", "--short")
    tracked_diff = subprocess.check_output(
        ["git", "-C", str(repo_root), "diff", "--binary", "--no-ext-diff", "HEAD"]
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS",
        "repository": str(repo_root),
        "git": {
            "branch": _git(repo_root, "branch", "--show-current"),
            "commit": _git(repo_root, "rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_short": status.splitlines(),
            "tracked_dirty_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        },
        "primary_identity": {
            "teacher": {"path": str(teacher), "sha256": teacher_sha, "kind": "local_scratch"},
            "adapter": {"path": str(adapter), "sha256": adapter_sha, "id": 39},
            "k1": {"successes": 166, "episodes": 200, "sr": 0.83},
            "k4": {"successes": 182, "episodes": 200, "sr": 0.91},
            "full_query_reduction": 0.74892,
            "mean_policy_latency_reduction": 0.59536,
        },
        "training_inputs": {
            "vit_checkpoint": {
                "path": str(vit_checkpoint),
                "sha256": vit_sha,
            },
            "dataset_root": str(dataset_root),
            "dataset_name": LIBERO_DATASET_NAME,
            "runtime_dataset_path": str(runtime_dataset_path),
            "runtime_episode_root": str(episode_root),
            "data_info": {"path": str(data_info), "sha256": data_info_sha},
            "dataset_meta": {"path": str(data_meta), "sha256": data_meta_sha},
            "content_scope": (
                "data-info and HDF5 meta/index are byte-locked; per-run selected "
                "episode/window indices are locked by dataset_split_manifest.json"
            ),
        },
        "evaluator": {
            "canonical_row": str(canonical_row),
            "canonical_episode_csv": str(episode_path),
            "renderer": renderer,
            "task_order": list(EXPECTED_TASK_ORDER),
            "episodes_per_task": 20,
            "episode_keys": [item["episode_key"] for item in episodes],
            "query_interval": 4,
            "refresh_policy": "periodic",
            "full_k_schedule": ["full", "skip", "skip", "skip"],
            "full_refresh_predicate": "cache empty or timestep % 4 == 0",
            "precision": "fp32",
            "action_pred_steps": 3,
            "training_sequence_length": 7,
            "evaluation_window_size": 13,
            "temporal_ensemble_temperature": 0.01,
            "temporal_validity": "all seven candidate values are nonzero",
            "temporal_weight_dtype": "numpy.float64",
            "temporal_output_dtype": "torch.float64",
            "gripper_postprocessing": "probability > 0.5, then {0,1} to {-1,+1}",
            "seed": 42,
            "max_steps": 600,
        },
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "interpreter_prefix": sys.prefix,
            "interpreter_environment": Path(sys.prefix).name,
            "launcher_conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "launcher_conda_prefix": os.environ.get("CONDA_PREFIX"),
            "torch": _package_version("torch"),
            "numpy": _package_version("numpy"),
            "mujoco": _package_version("mujoco"),
            "robosuite": _package_version("robosuite"),
            "libero": _package_version("libero"),
            "PyOpenGL": _package_version("PyOpenGL"),
        },
        "source_sha256": source_hashes,
    }
    (output_dir / "source_lock_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "source_hashes.json").write_text(
        json.dumps(source_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "canonical_episode_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    report = [
        "# Seer LatentLoop Q1/Q2 source lock",
        "",
        "**Status: PASS.**",
        "",
        f"- Branch / commit: `{manifest['git']['branch']}` / `{manifest['git']['commit']}`",
        f"- Dirty source fingerprint: `{manifest['git']['tracked_dirty_diff_sha256']}`",
        f"- Local scratch teacher 33: `{teacher_sha}`",
        f"- Fixed local adapter 39: `{adapter_sha}`",
        f"- MAE-ViT checkpoint: `{vit_sha}`",
        f"- LIBERO data-info/meta: `{data_info_sha}` / `{data_meta_sha}`",
        "- Primary identity: Full Seer K1 `166/200`; dense LatentLoop K4 `182/200`.",
        "- Evaluator: 10 tasks x 20 episode IDs, seed 42, P=3, FP32, OSMesa.",
        "- K4 schedule: periodic `full, skip, skip, skip`; cache-empty always forces full refresh.",
        "- Temporal ensemble: all-seven-nonzero mask, NumPy float64 exponential weights, torch float64 output.",
        "- Action postprocessing: soft gripper probability threshold at 0.5 and remap to {-1,+1}.",
        "",
        "This lock fingerprints the dirty working tree and every comparison source file; it does not require a clean checkout.",
    ]
    (output_dir / "source_lock_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--canonical-row", type=Path, required=True)
    parser.add_argument("--vit-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    lock_source(
        args.repo_root.resolve(),
        args.output_dir.resolve(),
        args.teacher.resolve(),
        args.adapter.resolve(),
        args.canonical_row.resolve(),
        args.vit_checkpoint.resolve(),
        args.dataset_root.resolve(),
    )


if __name__ == "__main__":
    main()
