import torch
import pytest

from tools.simvla.error_compensation_common import ARMS, arm_inputs, environment
from tools.simvla.error_compensation_campaign import jobs
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import lr_factor


def test_counterfactual_targets_and_code():
    predicted, teacher, code = (torch.randn(2, 4, 3) for _ in range(3))
    assert arm_inputs(ARMS[0], predicted, teacher, code)[0] is predicted
    assert arm_inputs(ARMS[1], predicted, teacher, code) == (teacher, code)
    target, empty = arm_inputs(ARMS[2], predicted, teacher, code)
    assert target is teacher
    assert torch.count_nonzero(empty) == 0
    assert torch.count_nonzero(code) > 0


def test_no_unapproved_gpus():
    for gpu in (0, 1, 2, 3, 8):
        with pytest.raises(ValueError): environment({}, gpu)


def test_short_horizon_schedule():
    assert lr_factor(0, 5000, 250) == 1 / 250
    assert lr_factor(250, 5000, 250) == 1
    assert lr_factor(5000, 5000, 250) == pytest.approx(0.1)


def test_every_trained_arm_has_online_evaluation():
    plan = jobs({"python": "python", "output": "/tmp/results"}, "/tmp/c.json", False)
    by_id = {j["id"]: j for j in plan}
    assert len(plan) == 11
    for arm in ARMS:
        assert by_id["eval_" + arm]["deps"] == ["train_" + arm]
    assert all("--smoke" not in j["cmd"] for j in plan)
