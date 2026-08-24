from __future__ import annotations

import copy
from collections import defaultdict, deque
from types import SimpleNamespace

import pytest
import torch

from architectures.simvla.adapters.latentloop.native_v0_checkpoint import CHECKPOINT_FORMAT
from tools.simvla.native_v0_intermediate_eval import (
    FINAL_DIAGNOSTIC_CLASS,
    FINAL_MANIFEST_SCHEMA,
    _NativeV0K2Policy,
    _canonical_hash,
    _load_checkpoint_payload,
    _validate_manifest,
    require_source_compatible,
)


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


def test_kc2_schedule_alternates_full_and_age1_updates() -> None:
    policy = _NativeV0K2Policy.__new__(_NativeV0K2Policy)
    policy.metrics = SimpleNamespace(counters=defaultdict(int))
    policy.query_index = 0
    policy.action_queue = deque()
    policy.query_trace = []
    action_chunk = torch.zeros(1, 10, 7)
    policy._full_refresh = lambda batch, policy_query_index: (None, action_chunk, policy_query_index)
    policy._v0_update = lambda batch, age, policy_query_index: (None, action_chunk, policy_query_index)

    for _ in range(6):
        policy._refill_action_queue({})

    assert [row["age"] for row in policy.query_trace] == [0, 1, 0, 1, 0, 1]
    assert [row["source"] for row in policy.query_trace] == ["full_refresh", "native_v0"] * 3
    assert len(policy.action_queue) == 5


def test_final_checkpoint_requires_explicit_diagnostic_flag(tmp_path) -> None:
    checkpoint = tmp_path / "final.pt"
    torch.save(
        {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "global_optimizer_step": 150_000,
            "scientific_primary_checkpoint": True,
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="explicit diagnostic"):
        _load_checkpoint_payload(checkpoint)
    assert _load_checkpoint_payload(checkpoint, allow_final_diagnostic=True)[
        "global_optimizer_step"
    ] == 150_000


def test_gpu_ordinal_remap_is_explicit() -> None:
    manifest = {
        "schema_version": FINAL_MANIFEST_SCHEMA,
        "evaluation_class": FINAL_DIAGNOSTIC_CLASS,
        "scientific_claim_allowed": False,
        "selected_physical_gpu_ids": [4, 5],
        "renderer": {},
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    with pytest.raises(RuntimeError, match="SIMVLA_GPU_IDS"):
        _validate_manifest(manifest, (6, 7))
    _validate_manifest(manifest, (6, 7), allow_gpu_ordinal_remap=True)


def test_manifest_can_lock_disjoint_gpu_pairs_per_row() -> None:
    manifest = {
        "schema_version": FINAL_MANIFEST_SCHEMA,
        "evaluation_class": FINAL_DIAGNOSTIC_CLASS,
        "scientific_claim_allowed": False,
        "selected_physical_gpu_ids": [4, 5],
        "row_runtime_gpu_ids": {
            "native_v0_k2": [4, 5],
            "baseline_k1": [6, 7],
        },
        "renderer": {},
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    _validate_manifest(manifest, (4, 5), row="native_v0_k2")
    _validate_manifest(manifest, (6, 7), row="baseline_k1")
