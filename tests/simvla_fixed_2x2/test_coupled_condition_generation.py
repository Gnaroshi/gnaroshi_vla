from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from architectures.simvla.adapters.latentloop.efficient_multirate import (
    coupled_source_lock,
)

from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    audit_projection_only_state,
    build_coupled_query,
    build_kc2_coupled_query,
    prepare_projection_only_coupling,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    GenerationLoopConfig,
)
from methods.latentloop.modules.native_simvla_v0 import (
    NativeV0ObservationPair,
    NativeV0UpdateOutput,
)
from methods.latentloop.modules.simvla_generation_loop import (
    SimVLAGenerationHiddenUpdater,
)


class _Delta(nn.Module):
    def forward(self, pair):  # type: ignore[no-untyped-def]
        image_delta = pair.current_images.float().mean(
            (1, 2, 3, 4)
        ) - pair.previous_images.float().mean((1, 2, 3, 4))
        proprio_delta = (pair.current_proprio - pair.previous_proprio).mean(-1)
        value = image_delta + proprio_delta
        return value[:, None].repeat(1, 4)


class _Updater(nn.Module):
    def forward(  # type: ignore[no-untyped-def]
        self, previous_condition, code, *, valid_mask, group_ids, age
    ):
        del group_ids, age
        residual = code[:, None, :].repeat(1, previous_condition.shape[1], 1)
        condition = torch.where(
            valid_mask[..., None], previous_condition + residual, previous_condition
        )
        gate = torch.ones_like(residual[..., :1])
        return NativeV0UpdateOutput(condition=condition, residual=residual, gate=gate)


class _Adapter(nn.Module):
    delta_dim = 4

    def __init__(self) -> None:
        super().__init__()
        self.delta_encoder = _Delta()
        self.condition_updater = _Updater()


def _sequence() -> dict[str, torch.Tensor]:
    return {
        "image_sequence": torch.arange(
            2 * 4 * 2 * 2 * 2 * 3, dtype=torch.uint8
        ).reshape(2, 4, 2, 2, 2, 3),
        "proprio_sequence": torch.arange(
            2 * 4 * 8, dtype=torch.float32
        ).reshape(2, 4, 8),
        "anchor_condition": torch.zeros(2, 3, 4),
        "teacher_conditions": torch.stack(
            [torch.full((2, 3, 4), float(age)) for age in (1, 2, 3)], dim=1
        ),
        "valid_mask": torch.ones(2, 3, dtype=torch.bool),
        "group_ids": torch.zeros(2, 3, dtype=torch.long),
        "teacher_actions": torch.zeros(2, 3, 10, 7),
        "explicit_noises": torch.zeros(2, 3, 10, 7),
    }


def test_exposed_code_is_the_exact_condition_update_input() -> None:
    adapter = _Adapter()
    sequence = _sequence()
    query = build_kc2_coupled_query(
        adapter, sequence, query_ages=torch.tensor([1, 3])
    )
    assert query["condition_change_code"].shape == (2, 4)
    assert torch.count_nonzero(query["condition_change_code"]) > 0
    assert torch.equal(query["proprio"], sequence["proprio_sequence"][[0, 1], [1, 3]])


def test_full_refresh_age_uses_zero_code_and_exact_teacher_condition() -> None:
    adapter = _Adapter()
    sequence = _sequence()
    query = build_kc2_coupled_query(
        adapter, sequence, query_ages=torch.tensor([2, 2])
    )
    assert torch.count_nonzero(query["condition_change_code"]) == 0
    assert torch.equal(query["condition"], sequence["teacher_conditions"][:, 1])


def test_kc3_age2_recursively_updates_condition_and_exposes_latest_code() -> None:
    adapter = _Adapter()
    sequence = _sequence()
    query = build_coupled_query(
        adapter,
        sequence,
        query_ages=torch.tensor([2, 2]),
        k_c=3,
    )
    first_pair_code = adapter.delta_encoder(
        NativeV0ObservationPair(
            previous_images=sequence["image_sequence"][:, 0],
            current_images=sequence["image_sequence"][:, 1],
            previous_proprio=sequence["proprio_sequence"][:, 0],
            current_proprio=sequence["proprio_sequence"][:, 1],
        )
    )
    second_pair_code = adapter.delta_encoder(
        NativeV0ObservationPair(
            previous_images=sequence["image_sequence"][:, 1],
            current_images=sequence["image_sequence"][:, 2],
            previous_proprio=sequence["proprio_sequence"][:, 1],
            current_proprio=sequence["proprio_sequence"][:, 2],
        )
    )
    expected = sequence["anchor_condition"]
    expected = expected + first_pair_code[:, None, :]
    expected = expected + second_pair_code[:, None, :]
    assert torch.equal(query["condition"], expected)
    assert torch.equal(query["condition_change_code"], second_pair_code)


def test_kc3_age3_is_full_refresh_with_zero_code() -> None:
    adapter = _Adapter()
    sequence = _sequence()
    query = build_coupled_query(
        adapter,
        sequence,
        query_ages=torch.tensor([3, 3]),
        k_c=3,
    )
    assert torch.count_nonzero(query["condition_change_code"]) == 0
    assert torch.equal(query["condition"], sequence["teacher_conditions"][:, 2])


def test_projection_only_setup_matches_synthetic_dimensions() -> None:
    updater = SimVLAGenerationHiddenUpdater(
        hidden_dim=32,
        condition_dim=16,
        condition_code_dim=128,
        rank_dim=64,
    )
    audit = prepare_projection_only_coupling(updater)
    assert audit["trainable_parameters"] == 8192
    assert audit["trainable_parameter_names"] == ["condition_code_projection.weight"]
    assert torch.count_nonzero(updater.condition_code_projection.weight) == 0
    assert updater.condition_code_projection.bias.requires_grad is False


def test_production_projection_only_setup_trains_exactly_16384_parameters() -> None:
    updater = GenerationLoopConfig().build()
    audit = prepare_projection_only_coupling(updater)
    assert audit["condition_code_dim"] == 128
    assert updater.rank_dim == 128
    assert audit["trainable_parameters"] == 16_384


def test_projection_only_state_audit_rejects_any_other_change() -> None:
    parent = GenerationLoopConfig().build()
    candidate = GenerationLoopConfig().build()
    candidate.load_state_dict(parent.state_dict())
    prepare_projection_only_coupling(candidate)
    assert (
        audit_projection_only_state(parent, candidate)["verdict"]
        == "PROJECTION_ONLY_STATE_PASS"
    )
    with torch.no_grad():
        candidate.gate_head.bias.add_(1.0)
    assert (
        audit_projection_only_state(parent, candidate)["verdict"]
        == "PROJECTION_ONLY_STATE_FAIL"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_checkpoint_lineage_is_independent_of_current_eval_commit(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    parent = tmp_path / "parent.pt"
    condition = tmp_path / "condition.pt"
    norm = tmp_path / "norm.json"
    cache = tmp_path / "cache"
    cache.mkdir()
    checkpoint.write_bytes(b"frozen-coupled-checkpoint")
    parent.write_bytes(b"parent")
    condition.write_bytes(b"condition")
    norm.write_bytes(b"norm")
    (cache / "manifest.json").write_bytes(b"cache")

    monkeypatch.setattr(
        coupled_source_lock, "FROZEN_GENERATION_CHECKPOINT_SHA256", _sha256(parent)
    )
    monkeypatch.setattr(
        coupled_source_lock, "FROZEN_CONDITION_CHECKPOINT_SHA256", _sha256(condition)
    )
    monkeypatch.setattr(
        coupled_source_lock, "FROZEN_NORM_STATS_SHA256", _sha256(norm)
    )
    monkeypatch.setattr(
        coupled_source_lock,
        "FROZEN_EXACT_CACHE_MANIFEST_SHA256",
        _sha256(cache / "manifest.json"),
    )
    monkeypatch.setattr(
        coupled_source_lock, "FROZEN_GENERATION_SOURCE_SHA256", "1" * 64
    )
    monkeypatch.setattr(
        coupled_source_lock, "FROZEN_CONDITION_SOURCE_SHA256", "2" * 64
    )
    monkeypatch.setattr(coupled_source_lock, "FROZEN_UPSTREAM_COMMIT", "3" * 40)

    historical_root = "4" * 40
    source = {
        "schema_version": "simvla_condition_generation_coupling_source_v1",
        "root_commit": historical_root,
        "upstream_commit": "3" * 40,
        "source_file_sha256": {"historical_training.py": "5" * 64},
        "parent_generation_source_combined_sha256": "1" * 64,
        "condition_source_combined_sha256": "2" * 64,
        "parent_generation_checkpoint_sha256": _sha256(parent),
        "condition_checkpoint_sha256": _sha256(condition),
        "norm_stats_sha256": _sha256(norm),
        "exact_cache_manifest_sha256": _sha256(cache / "manifest.json"),
    }
    source["combined_sha256"] = coupled_source_lock._canonical_sha256(source)
    row = "synthetic_coupled"
    monkeypatch.setitem(
        coupled_source_lock.FROZEN_COUPLED_CHECKPOINTS,
        row,
        {
            "k_c": 2,
            "n_g": 3,
            "checkpoint_sha256": _sha256(checkpoint),
            "source_combined_sha256": source["combined_sha256"],
            "source_root_commit": historical_root,
        },
    )
    payload = {
        "source_lock": source,
        "training_config": {
            "source_combined_sha256": source["combined_sha256"],
            "k_c": 2,
            "n_g": 3,
        },
    }
    report = coupled_source_lock.verify_frozen_coupled_checkpoint(
        checkpoint,
        payload,
        row=row,
        parent_generation_checkpoint=parent,
        condition_checkpoint=condition,
        norm_stats=norm,
        exact_cache=cache,
    )
    assert report["verdict"] == "FROZEN_COUPLED_CHECKPOINT_PASS"

    tampered = deepcopy(payload)
    tampered["source_lock"]["root_commit"] = "6" * 40
    failed = coupled_source_lock.verify_frozen_coupled_checkpoint(
        checkpoint,
        tampered,
        row=row,
        parent_generation_checkpoint=parent,
        condition_checkpoint=condition,
        norm_stats=norm,
        exact_cache=cache,
    )
    assert failed["verdict"] == "FROZEN_COUPLED_CHECKPOINT_FAIL"
