from collections import Counter
from types import SimpleNamespace

import pytest

from tools.simvla.error_compensation_kc_transfer import (
    ROWS, candidate_query, expected_counts, summary_valid, validate_episode,
)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_candidate_only_used_between_refresh_queries(k):
    for q in range(13):
        assert candidate_query("true_condition", k, q) == (q % k != 0)
        assert candidate_query("true_condition_no_code", k, q) == (q % k != 0)
        assert not candidate_query("parent", k, q)
        assert not candidate_query("condition_naive3", k, q)


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("row", ROWS)
def test_counters_include_partial_last_chunk(k, row):
    for actions in (1, 5, 6, 11, 16, 21, 26, 900):
        queries = list(range(0, actions, 5))
        n = len(queries)
        full = sum(q % k == 0 for q in range(n))
        c = expected_counts(row, k, actions)
        assert c == dict(queries=n, full_vlm=full, condition=n-full,
            transformer=3*n, generation=0 if row == "condition_naive3" else 7*n)


def test_no_accidental_kc2_modulo_for_longer_intervals():
    assert candidate_query("true_condition", 3, 2)
    assert not candidate_query("true_condition", 3, 3)
    assert candidate_query("true_condition", 4, 2)
    assert not candidate_query("true_condition", 4, 4)


def test_resume_rejects_other_horizon_and_partial_summary(tmp_path):
    spec = {"task_id": 9, "trial_id": 0}
    saved = dict(identity="id", row="parent", k_c=3, **spec,
        episode_length=26, counters=expected_counts("parent", 3, 26))
    validate_episode(saved, "id", "parent", 3, spec)
    with pytest.raises(RuntimeError):
        validate_episode(saved, "id", "parent", 4, spec)
    from tools.simvla.error_compensation_common import write_json
    path = tmp_path / "summary.json"
    write_json(path, dict(identity="id", verdict="EVALUATION_COMPLETE", episodes=499))
    assert not summary_valid(path, "id")
    with pytest.raises(RuntimeError):
        summary_valid(path, "other")


def test_followup_does_not_edit_original_evaluator():
    import inspect
    from tools.simvla import error_compensation_kc_transfer as m
    source = inspect.getsource(m.make_policy)
    assert "candidate_query(row, k_c, policy_query_index)" in source
    assert "full_step_indices=(0, 4, 8)" in source
    assert "self.condition_layout.valid_mask if updated else None" in source
    assert "original_policy(source, row)" in source


@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("row", ["parent", "true_condition", "true_condition_no_code"])
def test_real_decode_binding_selects_candidate_and_code(monkeypatch, k, row):
    import torch
    from torch import nn
    from tools.simvla import error_compensation_eval as old
    from tools.simvla.error_compensation_kc_transfer import make_policy
    class Loop:
        def __init__(self):
            self.updater = SimpleNamespace(condition_code_dim=128)
            self.inputs = []
        def __call__(self, noise, **kwargs):
            self.inputs.append(kwargs)
            return SimpleNamespace(final_noisy_action=noise)
    parent, candidate = Loop(), Loop()
    transformer = nn.Module()
    transformer.action_decoder = nn.Identity()
    code = torch.full((1, 128), 2.0)
    mask = torch.ones(1, 2, dtype=torch.bool)
    policy = SimpleNamespace(model=SimpleNamespace(transformer=transformer),
        _experiment_loops=[parent, candidate], _condition_code=code,
        _paired_initial_noise=lambda *_: (torch.zeros(1, 10, 7), 42),
        condition_layout=SimpleNamespace(valid_mask=mask),
        action_adapter=SimpleNamespace(normalize_proprio=lambda p: p,
            action_space=SimpleNamespace(postprocess=lambda x: x)),
        metrics=SimpleNamespace(counters=Counter()))
    monkeypatch.setattr(old, "make_policy", lambda *_: policy)
    made = make_policy({}, row, k)
    for q in range(9):
        made._decode(torch.zeros(1, 2, 960), torch.zeros(1, 8), policy_query_index=q)
        updated = row != "parent" and q % k != 0
        inputs = (candidate if updated else parent).inputs[-1]
        assert inputs["condition_valid_mask"] is (mask if updated else None)
        expected_code = code if updated and row == "true_condition" else torch.zeros_like(code)
        torch.testing.assert_close(inputs["condition_change_code"], expected_code)
    assert len(candidate.inputs) == (sum(q % k != 0 for q in range(9)) if row != "parent" else 0)
    assert policy.k_c == policy.refresh_every == k
