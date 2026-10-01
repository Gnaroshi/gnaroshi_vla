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
    plan = jobs({"python": "python", "output": "/tmp/results", "evaluation_condition_intervals": [4, 3]}, "/tmp/c.json", False)
    by_id = {j["id"]: j for j in plan}
    assert len(plan) == 11
    for arm in ARMS:
        for k_c in (3, 4):
            assert by_id[f"eval_kc{k_c}_{arm}"]["deps"] == ["train_" + arm]
    assert plan[3]["id"] == "eval_kc4_condition_full10"
    assert not plan[3]["deps"]
    assert all("--smoke" not in j["cmd"] for j in plan)


def test_oracle_condition_never_leaks_into_student_full_steps():
    from types import SimpleNamespace
    from torch import nn
    from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationHiddenUpdater, SimVLAGenerationLoop
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_objective import generation_local_oracle_loss
    torch.set_num_threads(1)
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_decoder = nn.Linear(8, 7)
            self.conditions = []
        def forward(self, *, vlm_features, action_with_noise, proprio, t):
            self.conditions.append(vlm_features.detach().clone())
            hidden = vlm_features.mean(1)[:, None, :].expand(-1, 10, -1)
            return self.action_decoder(hidden)
    transformer = Transformer().requires_grad_(False)
    updater = SimVLAGenerationHiddenUpdater(hidden_dim=8, condition_dim=8, rank_dim=4)
    loop = SimVLAGenerationLoop(updater, transformer.action_decoder)
    predicted, teacher = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    arguments = dict(loop=loop, transformer=transformer, action_space=SimpleNamespace(postprocess=lambda x: x),
        condition=predicted, initial_noise=torch.randn(1, 10, 7), normalized_proprio=torch.randn(1, 8),
        condition_valid_mask=None, condition_change_code=torch.zeros(1, 128),
        full_step_indices=(0, 4, 8), teacher_final_action=torch.randn(1, 10, 7))
    result = generation_local_oracle_loss(**arguments, oracle_condition=teacher)
    assert len(transformer.conditions) == 4
    for condition in transformer.conditions[:3]:
        torch.testing.assert_close(condition, predicted, rtol=0, atol=0)
    torch.testing.assert_close(transformer.conditions[3], teacher.repeat(7, 1, 1), rtol=0, atol=0)
    result.total.backward()
    assert all(p.grad is None for p in transformer.parameters())
    default = generation_local_oracle_loss(**arguments)
    explicit = generation_local_oracle_loss(**arguments, oracle_condition=predicted)
    torch.testing.assert_close(default.total, explicit.total, rtol=0, atol=0)
