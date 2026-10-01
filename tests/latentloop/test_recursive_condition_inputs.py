from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from architectures.simvla.adapters.latentloop.efficient_multirate.recursive_condition_inputs import query_inputs
from tools.simvla.error_compensation_campaign import job_complete
from tools.simvla.error_compensation_common import ARMS, write_json
from tools.simvla.error_compensation_eval import check_counts


@pytest.mark.parametrize("age", [1, 2, 3])
def test_predictions_are_recursive_and_never_read_teacher(age):
    inputs, pairs = [], []
    def encode(pair):
        pairs.append(pair)
        return pair.current_proprio - pair.previous_proprio
    def update(condition, code, *, age, **kwargs):
        inputs.append(condition.clone())
        return SimpleNamespace(condition=condition + code[:, None, :] + age)
    adapter = SimpleNamespace(delta_encoder=encode, condition_updater=update)
    action = SimpleNamespace(normalize_proprio=lambda p: 2*p,
        action_space=SimpleNamespace(normalize_action=lambda x: x))
    sequence = dict(anchor_condition=torch.zeros(1, 2, 1),
        image_sequence=torch.arange(4).reshape(1, 4, 1),
        proprio_sequence=torch.arange(4.).reshape(1, 4, 1),
        valid_mask=torch.ones(1, 2, dtype=torch.bool), group_ids=torch.zeros(1, 2),
        explicit_noises=torch.arange(3.).reshape(1, 3, 1),
        teacher_actions=torch.zeros(1, 3, 1))
    # Deliberately omit teacher_conditions: they must not initialize later updates.
    context, raw, noise, target = query_inputs(adapter, action, sequence, age)
    assert context.condition.unique().item() == sum(1+i for i in range(1, age+1))
    for i in range(age):
        assert inputs[i].unique().item() == sum(1+j for j in range(1, i+1))
        assert pairs[i].previous_images.item() == i
        assert pairs[i].current_images.item() == i+1
    assert raw.item() == age
    assert context.proprio.item() == 2*age
    assert context.global_code.item() == 1
    assert noise.item() == age-1
    assert target.item() == 0
    assert not context.condition.requires_grad


@pytest.mark.parametrize("age", [0, 4, -1])
def test_invalid_ages_fail_before_accessing_data(age):
    with pytest.raises(ValueError):
        query_inputs(None, None, {}, age)


@pytest.mark.parametrize("k_c", [3, 4])
@pytest.mark.parametrize("row", ["condition_full10", *ARMS])
def test_actual_invocation_counters(k_c, row):
    for actions in (1, 5, 6, 16, 26, 900):
        q = (actions+4)//5
        full = (q+k_c-1)//k_c
        policy = SimpleNamespace(step_index=actions,
            metrics=SimpleNamespace(counters=Counter(num_policy_queries=q, num_full_vlm_calls=full)))
        counts = dict(transformer=(10 if row == "condition_full10" else 3)*q,
            generation=0 if row == "condition_full10" else 7*q, condition=q-full)
        assert check_counts(policy, row, counts, k_c)["full_vlm"] == full
        with pytest.raises(RuntimeError):
            check_counts(policy, row, {**counts, "transformer": counts["transformer"]-1}, k_c)


def test_completed_jobs_require_full_count_and_identity(tmp_path):
    path = tmp_path / "summary.json"
    job = dict(id="eval_kc4_true_condition", summary=str(path))
    assert not job_complete(job, "id", False, 5000)
    write_json(path, dict(identity="id", verdict="EVALUATION_COMPLETE", episodes=499))
    assert not job_complete(job, "id", False, 5000)
    write_json(path, dict(identity="id", verdict="EVALUATION_COMPLETE", episodes=500))
    assert job_complete(job, "id", False, 5000)
    with pytest.raises(RuntimeError):
        job_complete(job, "other", False, 5000)
    write_json(path, dict(identity="id", verdict="SMOKE_PASS", steps=3))
    assert job_complete(dict(id="train_true_condition", summary=str(path)), "id", True, 5000)
