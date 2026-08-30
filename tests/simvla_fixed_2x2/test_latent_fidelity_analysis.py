from __future__ import annotations

import math

import torch

from architectures.simvla.adapters.latentloop.efficient_multirate.latent_fidelity_analysis import (
    assign_change_quartiles,
    deterministic_task_partner_indices,
    masked_condition_metrics,
)


def test_masked_condition_metrics_identical_target() -> None:
    generator = torch.Generator().manual_seed(7)
    previous = torch.randn(2, 4, 6, generator=generator)
    target = previous + torch.randn(2, 4, 6, generator=generator) * 0.1
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    rows = masked_condition_metrics(target, target, previous, mask)
    assert len(rows) == 2
    for row in rows:
        assert row["condition_mse"] == 0.0
        assert row["condition_normalized_mse"] == 0.0
        assert math.isclose(row["condition_cosine"], 1.0, rel_tol=0, abs_tol=1e-6)
        assert math.isclose(row["condition_delta_cosine"], 1.0, rel_tol=0, abs_tol=1e-6)
        assert math.isclose(row["condition_delta_norm_ratio"], 1.0, rel_tol=0, abs_tol=1e-6)


def test_partner_indices_never_cross_task_or_self_pair() -> None:
    identities = (
        (0, "a", 0),
        (0, "b", 0),
        (1, "c", 0),
        (1, "d", 0),
        (1, "e", 0),
    )
    partners = deterministic_task_partner_indices(identities)
    assert set(partners) == set(range(len(identities)))
    for index, partner in partners.items():
        assert index != partner
        assert identities[index][0] == identities[partner][0]


def test_change_quartiles_are_shared_across_variants() -> None:
    rows = []
    for index in range(8):
        for variant in ("ours_full_observation", "stale_observation"):
            rows.append(
                {
                    "dataset_index": index,
                    "age": 1,
                    "regime": "kc2_local",
                    "variant": variant,
                    "visual_change_l1": float(index),
                    "proprio_change_l2": float(index),
                }
            )
    assign_change_quartiles(rows)
    by_index = {}
    for row in rows:
        by_index.setdefault(row["dataset_index"], set()).add(
            row["observation_change_quartile"]
        )
    assert all(len(values) == 1 for values in by_index.values())
    assert {next(iter(values)) for values in by_index.values()} == {1, 2, 3, 4}
