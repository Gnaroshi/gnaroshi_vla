from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from architectures.openpi.adapters.latentloop.aligned_condition import (
    AlignedConditionUpdater, ConditionConfig, load_condition, pack_kv, unpack_kv,
)
from architectures.openpi.adapters.latentloop.dual_loop import ANCHORS, DualLoopPolicy
from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixKVState
from methods.latentloop.modules.native_simvla_v0 import NativeV0DeltaEncoder, TokenSharedConditionUpdater


def observation():
    return SimpleNamespace(images={k: torch.rand(1, 3, 64, 64) * 2 - 1 for k in
                                  ("base_0_rgb", "left_wrist_0_rgb")},
                           image_masks={k: torch.ones(1, dtype=torch.bool) for k in
                                        ("base_0_rgb", "left_wrist_0_rgb")},
                           state=torch.randn(1, 8), tokenized_prompt=torch.tensor([[1, 2, 0]]),
                           tokenized_prompt_mask=torch.tensor([[True, True, False]]))


def prefix():
    mask = torch.tensor([[True, True, True, False]])
    return PrefixKVState(embeddings=torch.randn(1, 4, 16), pad_mask=mask,
                         attention_pattern=torch.zeros_like(mask), position_ids=mask.cumsum(1) - 1,
                         pre_rope_keys=tuple(torch.randn(1, 2, 4, 8) for _ in range(2)),
                         values=tuple(torch.randn(1, 2, 4, 8) for _ in range(2)))


def test_pack_roundtrip_and_zero_update():
    p, a, b = prefix(), observation(), observation()
    model = AlignedConditionUpdater(ConditionConfig.from_prefix(p, a))
    assert isinstance(model.delta_encoder, NativeV0DeltaEncoder)
    assert isinstance(model.condition_updater, TokenSharedConditionUpdater)
    assert torch.equal(pack_kv(unpack_kv(p, pack_kv(p))), pack_kv(p))
    predicted, update = model(p, a, b, age=1)
    assert torch.equal(pack_kv(predicted), pack_kv(p))
    assert torch.all(update.gate[update.gate != 0] < 0.02)


def test_recursive_graph_mask_and_current_observation():
    p = prefix()
    observations = [observation() for _ in range(4)]
    model = AlignedConditionUpdater(ConditionConfig.from_prefix(p, observations[0]))
    torch.nn.init.normal_(model.condition_updater.up.weight, std=0.05)
    first, _ = model(p, observations[0], observations[1], age=1)
    second, _ = model(first, observations[1], observations[2], age=2)
    third, _ = model(second, observations[2], observations[3], age=3)
    grad = torch.autograd.grad(pack_kv(third).square().sum(), first.values[0], retain_graph=True)[0]
    assert grad.abs().sum() > 0
    pack_kv(third).square().mean().backward()
    assert model.delta_encoder.image_encoder[0].weight.grad.abs().sum() > 0
    assert torch.equal(pack_kv(third)[..., -1, :], pack_kv(p)[..., -1, :])
    changed = observation()
    changed.state = observations[1].state + 10
    other, _ = model(p, observations[0], changed, age=1)
    assert not torch.equal(pack_kv(first), pack_kv(other))
    with pytest.raises(ValueError, match="prompt"):
        changed.tokenized_prompt = torch.tensor([[2, 2, 0]])
        model(p, observations[0], changed, age=1)


def test_legacy_checkpoint_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"adapter": {}, "step": 24000}, path)
    with pytest.raises(ValueError, match="legacy"):
        load_condition(path, device="cpu")


def test_schedule_matches_simvla():
    assert ANCHORS[2] == (0, 5)
    assert ANCHORS[3] == (0, 4, 8)


def test_condition_skip_does_not_embed_or_accept_actions(monkeypatch):
    import architectures.openpi.adapters.latentloop.dual_loop as runtime
    p, previous, current = prefix(), observation(), observation()
    updater = AlignedConditionUpdater(ConditionConfig.from_prefix(p, previous))
    policy = DualLoopPolicy(SimpleNamespace(), updater, None, "condition_k2")
    calls = []
    policy.hook = SimpleNamespace(extract=lambda obs: (calls.append(obs) or SimpleNamespace(
        state=p, robot_state=obs.state, prefix_embedding_ms=1., full_prefix_ms=1.)))
    monkeypatch.setattr(runtime, "generate", lambda *a, **k: SimpleNamespace(
        actions=torch.zeros(1, 10, 7), metrics={}))
    noise = torch.zeros(1, 10, 7)
    policy.query(previous, noise)
    _, metrics = policy.query(current, noise)
    assert len(calls) == 1
    assert metrics["condition_updater_calls"] == 1
    assert metrics["prefix_embedding_ms"] == 0
    policy.query(current, noise)
    assert len(calls) == 2
    policy.reset()
    assert policy.prefix is None and policy.previous_observation is None
