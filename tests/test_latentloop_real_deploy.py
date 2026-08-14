import json

import numpy as np
import pytest
import torch

from architectures.seer.adapters.latentloop_real_deploy.controller import (
    LatentLoopSeerController,
    load_artifact_profile,
    remove_ddp_prefix,
    should_use_latentloop,
    temporal_ensemble_probability,
)


def test_remove_ddp_prefix_for_single_gpu_inference():
    tensor = torch.tensor([1.0])
    assert remove_ddp_prefix({"module.lrnode_x": tensor}) == {"lrnode_x": tensor}
    with pytest.raises(ValueError, match="collision"):
        remove_ddp_prefix({"module.x": tensor, "x": tensor})


@pytest.mark.parametrize(
    ("timestep", "has_cache", "expected"),
    [
        (0, False, False),
        (0, True, False),
        (1, True, True),
        (2, True, True),
        (3, True, True),
        (4, True, False),
    ],
)
def test_k4_schedule(timestep, has_cache, expected):
    assert should_use_latentloop(timestep, 4, has_cache) is expected


def test_temporal_ensemble_matches_legacy_probability_domain_rule():
    buffer = torch.zeros(6, 9, 7)
    first = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]] * 3], dtype=torch.float32
    )
    second = torch.tensor(
        [[[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]] * 3], dtype=torch.float32
    )

    action0, count0 = temporal_ensemble_probability(first, 0, buffer, 0.01)
    assert count0 == 1
    assert action0.dtype == torch.float64
    torch.testing.assert_close(action0, first[:, 0].double())

    action1, count1 = temporal_ensemble_probability(second, 1, buffer, 0.01)
    weights = np.exp(-0.01 * np.arange(2))
    weights /= weights.sum()
    expected = first[:, 1].double() * weights[0] + second[:, 0].double() * weights[1]
    assert count1 == 2
    torch.testing.assert_close(action1, expected)


def test_manifest_requires_teacher_specific_adapter_pair(tmp_path):
    manifest = {
        "schema_version": 1,
        "task": "basketball",
        "latentloop_architecture": {"hidden_dim": 256},
        "vit": {"filename": "vit.pth"},
        "teachers": {
            "37": {
                "filename": "teacher_37.pth",
                "adapters": {"39": {"filename": "teacher_37_adapter_39.pth"}},
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    profile = load_artifact_profile(path, teacher_id=37, adapter_id=39)
    assert profile["teacher"]["filename"] == "teacher_37.pth"
    assert profile["adapter"]["filename"] == "teacher_37_adapter_39.pth"
    with pytest.raises(KeyError, match="Teacher 34"):
        load_artifact_profile(path, teacher_id=34, adapter_id=39)


def test_completed_rollout_warmup_is_not_written_into_previous_summary():
    controller = object.__new__(LatentLoopSeerController)
    controller.history_len = 7
    controller.real_eval_max_steps = 20
    controller.action_pred_steps = 3
    controller.device_id = torch.device("cpu")
    controller.use_ensembling = False
    controller.rollout_index = 1
    controller._rollout_complete = False
    controller.step_records = [{"timestep": 0}]
    writes = []
    controller.write_runtime_summary = lambda: writes.append("write")

    controller.mark_rollout_complete()
    assert controller._rollout_complete is True
    assert writes == ["write"]

    controller.reset()
    assert writes == ["write"]
    assert controller.rollout_index == 2
    assert controller._rollout_complete is False
    assert controller.step_records == []
