from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from methods.latentloop.modules.shared_refinement import (
    VARIANTS, RefinementContext, SharedRefiner, refine_from_anchor,
)
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0, NativeV0ObservationPair
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement import condition_query, frozen_anchor
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import models_for_seed, lr_factor
from tools.simvla.shared_refinement_pipeline import remaining_blockers


@pytest.fixture(autouse=True)
def one_thread():
    torch.set_num_threads(1)


def context():
    return RefinementContext(torch.randn(2, 6, 960), torch.tensor([[1,1,1,0,0,0], [1,1,1,1,1,1]]).bool(),
        torch.randn(2, 8), torch.randn(2, 128), torch.randn(2, 6, 65), torch.ones(2).bool())


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("cheap", [1, 2])
def test_whole_flow_interval_and_gradients(variant, cheap):
    torch.manual_seed(17)
    model = SharedRefiner(variant)
    decoder = nn.Linear(1024, 7).requires_grad_(False)
    h, x = torch.randn(2, 10, 1024), torch.randn(2, 10, 7)
    velocity = decoder(h)
    output = refine_from_anchor(model, decoder, anchor_hidden=h, anchor_velocity=velocity,
        noise=x, context=context(), cheap_steps=cheap)
    # Zero residual must traverse the WHOLE interval, not stop at t=.8/.7.
    torch.testing.assert_close(output, x - velocity)
    output.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None for p in decoder.parameters())


def test_shared_code_mask_and_fresh_query():
    torch.manual_seed(3)
    model = SharedRefiner("token_code_hidden")
    nn.init.normal_(model.output.weight, std=.01)
    h, x = torch.randn(2, 10, 1024), torch.randn(2, 10, 7)
    c = context()
    kwargs = dict(tau=.5, dt=-.5, index=1)
    a = model(h, x, x + .1, c, **kwargs)
    altered = c.token_code.clone()
    altered[0, 3:] = 999
    torch.testing.assert_close(a, model(h, x, x + .1, replace(c, token_code=altered), **kwargs), rtol=0, atol=0)
    altered[0, :3] += 10
    assert not torch.equal(a, model(h, x, x + .1, replace(c, token_code=altered), **kwargs))
    fresh = replace(c, updated=torch.zeros(2).bool())
    torch.testing.assert_close(model(h, x, x, fresh, **kwargs),
        model(h, x, x, replace(fresh, token_code=altered), **kwargs), rtol=0, atol=0)


def test_native_condition_hook_preserves_output():
    torch.manual_seed(7)
    native = NativeSimVLAV0().eval()
    sequence = {"anchor_condition": torch.randn(1, 6, 960), "teacher_conditions": torch.randn(1, 3, 6, 960),
        "image_sequence": torch.rand(1, 4, 2, 3, 64, 64), "proprio_sequence": torch.randn(1, 4, 8),
        "valid_mask": torch.tensor([[1,1,1,1,0,0]]).bool(), "group_ids": torch.zeros(1, 6).long()}
    for age in (1, 3):
        prev = sequence["anchor_condition"] if age == 1 else sequence["teacher_conditions"][:, 1]
        pair = NativeV0ObservationPair(sequence["image_sequence"][:, age-1], sequence["image_sequence"][:, age],
            sequence["proprio_sequence"][:, age-1], sequence["proprio_sequence"][:, age])
        with torch.no_grad():
            original = native.condition_updater(prev, native.delta_encoder(pair), valid_mask=sequence["valid_mask"],
                group_ids=sequence["group_ids"], age=1)
        exposed = condition_query(native, sequence, age)
        torch.testing.assert_close(exposed.condition, original.condition, rtol=0, atol=0)
        encoded, gate = exposed.token_code[..., :-1], exposed.token_code[..., -1:]
        reconstructed = torch.nn.functional.linear(encoded, native.condition_updater.up.weight)
        reconstructed += gate * native.condition_updater.up.bias
        torch.testing.assert_close(reconstructed, original.condition - prev, atol=1e-6, rtol=1e-4)
    fresh = condition_query(native, sequence, 2)
    assert not fresh.updated.any()
    assert not fresh.token_code.any()
    assert not native.condition_updater.up._forward_pre_hooks


def test_anchor_one_call_and_hook_cleanup():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_decoder = nn.Linear(1024, 7)
            self.calls = 0
        def forward(self, **kwargs):
            self.calls += 1
            return self.action_decoder(torch.ones(2, 10, 1024))
    transformer = Transformer().requires_grad_(False)
    h, v = frozen_anchor(transformer, context(), torch.randn(2, 10, 7))
    assert transformer.calls == 1
    torch.testing.assert_close(transformer.action_decoder(h), v)
    assert not transformer.action_decoder._forward_pre_hooks


def test_initial_common_weights_and_lr_horizon():
    models = models_for_seed(7, torch.device("cpu"))
    for m in list(models.values())[1:]:
        torch.testing.assert_close(m.hidden.weight, models[VARIANTS[0]].hidden.weight, rtol=0, atol=0)
    assert lr_factor(0, 10000, 500) == .002
    assert lr_factor(500, 10000, 500) == 1
    assert lr_factor(10000, 10000, 500) == .1


def test_wait_parent_survives_gpu_gap_and_pid_reuse():
    parent = {"pid": 111, "start_ticks": "123", "state": "S", "command": "bash wrappers/run_pi05_dual_loop_long.sh --all"}
    captured = [parent]
    assert remaining_blockers([parent], captured) == [parent]
    reused = dict(parent, start_ticks="999", command="bash")
    assert remaining_blockers([reused], captured) == []
    restarted = dict(parent, pid=222, start_ticks="456")
    assert remaining_blockers([restarted], captured) == [restarted]
    assert remaining_blockers([dict(parent, state="Z")], captured) == []


def test_train_export_resume_and_offline_completion(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import (
        run_training, evaluate, gpu_contract_smoke,
    )
    from tools.simvla.shared_refinement_pipeline import write_json
    from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import install_exact_uint8_delta_path

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(7, 1024)
            self.action_decoder = nn.Linear(1024, 7)
        def forward(self, *, vlm_features, action_with_noise, proprio, t):
            return self.action_decoder(torch.tanh(self.project(action_with_noise) + t[:, None, None]))

    frozen = nn.Module()
    frozen.transformer = Transformer()
    frozen.requires_grad_(False)

    class Action:
        action_space = SimpleNamespace(normalize_action=lambda x: x, postprocess=lambda x: x)
        def normalize_proprio(self, x):
            return x
        def decode_action_from_condition(self, c, q, *, steps, initial_noise, return_debug=False):
            x = initial_noise
            for i in range(steps):
                x = x - frozen.transformer(vlm_features=c, action_with_noise=x, proprio=q,
                    t=x.new_full((x.shape[0],), 1-i/steps)) / steps
            return SimpleNamespace(action=x) if return_debug else x

    action = Action()
    native = NativeSimVLAV0().eval().requires_grad_(False)
    install_exact_uint8_delta_path(native)
    items = []
    for index in range(4):
        item = {"task_id": index, "episode_id": str(index), "anchor_query_index": 0,
            "language_instruction": "test", "query_ids": [str(i) for i in range(4)],
            "anchor_condition": torch.randn(122, 960), "teacher_conditions": torch.randn(3, 122, 960),
            "image_sequence": torch.randint(0, 256, (4,2,64,64,3), dtype=torch.uint8),
            "proprio_sequence": torch.randn(4,8), "valid_mask": torch.ones(122).bool(),
            "group_ids": torch.zeros(122).long(), "explicit_noises": torch.randn(3,10,7)}
        item["teacher_actions"] = torch.stack([action.decode_action_from_condition(item["teacher_conditions"][i:i+1],
            item["proprio_sequence"][i+1:i+2], steps=10, initial_noise=item["explicit_noises"][i:i+1])[0] for i in range(3)])
        items.append(item)

    class Dataset:
        store = SimpleNamespace(_loaded={})
        identities = [(i, str(i), 0) for i in range(4)]
        split_sha256 = "test"
        def __len__(self):
            return len(items)
        def __getitem__(self, i):
            return items[i]
        def contract(self):
            return {"fake": True}

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 0)
    c = dict(seed=7, learning_rate=1e-4, steps=4, warmup_steps=1, batch_size=1,
        num_workers=0, log_interval=1, save_interval=2, heldout_windows=4)
    runtime = (torch.device("cpu"), native, frozen, action, Dataset(), Dataset())
    gpu_contract_smoke(c, tmp_path, runtime, write_json)
    models = run_training(c, tmp_path, "test", runtime, write_json)
    before = {k: v.clone() for k, v in models.state_dict().items()}
    for path in (tmp_path / "checkpoints").glob("*_final.pt"):
        path.unlink()
    resumed = run_training(c, tmp_path, "test", runtime, write_json)
    for k, v in resumed.state_dict().items():
        torch.testing.assert_close(before[k], v, rtol=0, atol=0)
    assert len(list((tmp_path / "checkpoints").glob("*_final.pt"))) == 4
    report = evaluate(c, tmp_path, "test", runtime, resumed, write_json)
    assert len(report["results"]) == 12
    repeated = evaluate(c, tmp_path, "test", runtime, resumed, write_json)
    assert repeated == report
