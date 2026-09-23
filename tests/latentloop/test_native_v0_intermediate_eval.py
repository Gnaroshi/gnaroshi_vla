from __future__ import annotations

import copy

import pytest

from tools.simvla.native_v0_intermediate_eval import require_source_compatible


def _source(gpus: list[int]) -> dict:
    return {
        "combined_sha256": "derived-digest",
        "selected_physical_gpu_ids": gpus,
        "norm_stats_sha256": "official-norm",
        "critical_file_sha256": {"model.py": "source-hash"},
        "environment": {"torch": "2.6.0"},
        "complete_source_lock": {
            "command": ["train.py"],
            "conda_env": "simvla_libero",
            "cuda_visible_devices": "4,5",
            "root_branch": "training-branch",
            "root_commit": "training-commit",
            "root_status_short": "dirty-at-training",
            "simvla_upstream_status_short": "dirty-upstream-at-training",
            "packages": {"mujoco": "2.3.7"},
        },
    }


def test_source_compatibility_ignores_only_physical_gpu_ordinals() -> None:
    runtime = _source([6, 7])
    runtime["complete_source_lock"].update(
        {
            "command": ["evaluate.py"],
            "conda_env": "base",
            "cuda_visible_devices": "6,7",
            "root_branch": "evaluation-branch",
            "root_commit": "evaluation-commit",
            "root_status_short": "dirty-at-evaluation",
            "simvla_upstream_status_short": "clean-at-evaluation",
        }
    )
    require_source_compatible(
        checkpoint_source=_source([4, 5]),
        runtime_source=runtime,
    )


def test_source_compatibility_rejects_scientific_input_change() -> None:
    runtime = copy.deepcopy(_source([6, 7]))
    runtime["norm_stats_sha256"] = "wrong-norm"
    with pytest.raises(RuntimeError, match="norm_stats_sha256"):
        require_source_compatible(
            checkpoint_source=_source([4, 5]),
            runtime_source=runtime,
        )


def test_source_compatibility_rejects_package_change() -> None:
    runtime = copy.deepcopy(_source([6, 7]))
    runtime["complete_source_lock"]["packages"]["mujoco"] = "3.1.0"
    with pytest.raises(RuntimeError, match="complete_source_lock"):
        require_source_compatible(
            checkpoint_source=_source([4, 5]),
            runtime_source=runtime,
        )
