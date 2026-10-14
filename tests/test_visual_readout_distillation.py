import copy

import pytest
import torch
from torch.nn import functional as F

from methods.latentloop.modules.bounded_history import BoundedHistoryCondition
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.visual_memory import MemoryReference, VisualMemoryCondition
from methods.latentloop.modules.visual_midpoint import VisualMidpointCondition, VisualReference
from methods.latentloop.modules.visual_readout_distillation import (
    ARMS,
    DistilledVisualReference,
    ReadoutDistillationCondition,
)


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def parent():
    torch.manual_seed(71)
    short = BoundedHistoryCondition(NativeSimVLAV0(condition_dim=32), 'recurrent')
    torch.nn.init.normal_(short.condition_updater.up.weight, std=.04)
    torch.nn.init.normal_(short.condition_updater.gate_head.weight, std=.02)
    torch.nn.init.constant_(short.condition_updater.gate_head.bias, -1.)
    model = VisualMidpointCondition(short, 'late_every')
    # Simulate a completed parent: late differs from early and attention is active.
    with torch.no_grad():
        for parameter in model.late.parameters():
            parameter.add_(.013)
    torch.nn.init.normal_(model.visual.output.weight, std=.08)
    torch.nn.init.normal_(model.visual.output.bias, std=.03)
    return model.requires_grad_(False).eval()


def inputs(seed=17, batch=1, interval=8, uint8=False, condition_dim=32):
    generator = torch.Generator().manual_seed(seed)
    images = [torch.rand(batch, 2, 13, 17, 3, generator=generator) for _ in range(8)]
    if uint8:
        images = [(image * 255).to(torch.uint8) for image in images]
    proprio = [torch.randn(batch, 8, generator=generator) for _ in range(8)]
    anchor = torch.randn(batch, 6, condition_dim, generator=generator)
    valid = torch.tensor([[True, True, False, True, False, True]]).expand(batch, -1)
    groups = torch.arange(6)[None].expand(batch, -1)
    features = [torch.randn(batch, 72, condition_dim, generator=generator) for _ in range(8)]
    args = (anchor, images[0], proprio[0], valid, groups, interval)
    return args, images, proprio, features


def unroll(model, data, through=7):
    args, images, proprio, _ = data
    context = model.prepare(*args)
    outputs = []
    for age in range(1, through + 1):
        output, diagnostics = model.predict(context, age, images[age], proprio[age])
        outputs.append((output, diagnostics))
    return context, outputs


def trainable(name, arm):
    if arm == 'warm_retain':
        return name.startswith(('late.', 'visual.'))
    return name.startswith('visual.predictor.') or (arm == 'joint_readout' and name.startswith('late.'))


@pytest.mark.parametrize('arm', ARMS)
def test_full_parent_transfer_and_exact_freezes(parent, arm):
    original = {name: tensor.clone() for name, tensor in parent.state_dict().items()}
    model = ReadoutDistillationCondition(parent, arm)
    state = model.state_dict()
    for name, value in original.items():
        torch.testing.assert_close(state[name], value, rtol=0, atol=0)
        assert state[name].data_ptr() != parent.state_dict()[name].data_ptr()
    extra = set(state) - set(original)
    result = model.load_parent(parent.state_dict())
    assert set(result.missing_keys) == extra
    assert result.unexpected_keys == []
    if arm == 'warm_retain':
        assert not extra
        assert isinstance(model.visual, MemoryReference) and model.visual.spatial is None
    else:
        assert extra and all(name.startswith('visual.predictor.') for name in extra)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == trainable(name, arm), name
    assert all(not parameter.requires_grad for parameter in parent.parameters())
    assert model.feature_provider is None


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('seed,batch,uint8', [(19, 1, False), (29, 2, False), (39, 1, True)])
def test_exact_initial_hold_equivalence_all_seven_ages(parent, arm, seed, batch, uint8):
    model = ReadoutDistillationCondition(parent, arm).eval()
    hold = VisualMemoryCondition(parent.early, spatial=False).eval()
    hold.load_state_dict(parent.state_dict(), strict=True)
    data = inputs(seed, batch, uint8=uint8)
    vision = data[3][4]
    model.feature_provider = hold.feature_provider = lambda age, images: vision
    with torch.no_grad():
        _, actual = unroll(model, data)
        _, expected = unroll(hold, data)
    for age, ((output, diagnostics), (reference, reference_diagnostics)) in enumerate(
            zip(actual, expected), 1):
        torch.testing.assert_close(output, reference, rtol=0, atol=0, msg=f'age {age}')
        assert diagnostics.keys() == reference_diagnostics.keys()
        for name in diagnostics:
            torch.testing.assert_close(diagnostics[name], reference_diagnostics[name], rtol=0, atol=0)
        assert torch.equal(output[~data[0][3]], data[0][0][~data[0][3]])


@pytest.mark.parametrize('arm', ARMS)
def test_k4_parity_after_trainable_weights_change(parent, arm):
    model = ReadoutDistillationCondition(parent, arm)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(.1)
    assert model.feature_provider is None
    for seed in (31, 41):
        data = inputs(seed, batch=2, interval=4)
        with torch.no_grad():
            context, actual = unroll(model, data, through=3)
            _, expected = unroll(parent.early, data, through=3)
        for (output, _), (reference, _) in zip(actual, expected):
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        assert not hasattr(context, 'visual_memory')


@pytest.mark.parametrize('arm', ARMS)
def test_one_age4_vision_call_no_online_targets_and_fresh_context(parent, arm, monkeypatch):
    model = ReadoutDistillationCondition(parent, arm)
    data = inputs()
    calls = []

    def provide(age, images):
        assert age == 4 and images is data[1][4]
        calls.append(age)
        return data[3][4]

    def forbidden(*args, **kwargs):
        raise AssertionError('Online predict requested a feature target')

    model.feature_provider = provide
    if arm != 'warm_retain':
        monkeypatch.setattr(model.visual, 'target_readout', forbidden)
    with torch.no_grad():
        context, _ = unroll(model, data)
    assert calls == [4]
    assert context.visual_memory is data[3][4]
    assert context.visual_images is data[1][4]
    assert context.previous_images is data[1][7]
    assert context.previous_proprio is data[2][7]
    assert context.age == 7
    assert [model.uses_vision(age) for age in range(8)] == [False] * 4 + [True] + [False] * 3
    fresh = model.prepare(*data[0])
    assert not hasattr(fresh, 'visual_memory') and not hasattr(fresh, 'visual_images')
    with torch.no_grad():
        unroll(model, data)
    assert calls == [4, 4]


@pytest.mark.parametrize('condition_dim', [32, 960])
def test_target_shapes_fixed_projection_without_positions_and_detached(condition_dim):
    original = VisualReference(condition_dim, 64)
    visual = DistilledVisualReference(original)
    features = torch.randn(2, 72, condition_dim, requires_grad=True)
    target = visual.target_readout(features)
    source = F.layer_norm(features, (condition_dim,))
    for actual, expected in zip(target, (original.key(source), original.value(source))):
        assert actual.shape == (2, 72, 64)
        assert not actual.requires_grad and actual.grad_fn is None
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.no_grad():
        visual.position.weight.add_(10.)
    for before, after in zip(target, visual.target_readout(features)):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert features.grad is None


def test_age4_bypasses_predictor_after_training_and_matches_original(parent, monkeypatch):
    visual = DistilledVisualReference(parent.visual)
    with torch.no_grad():
        visual.predictor.mlp[-1].weight.fill_(.5)
        visual.predictor.mlp[-1].bias.fill_(.5)

    def forbidden(*args):
        raise AssertionError('Age 4 must bypass the predictor and RGB encoder')

    monkeypatch.setattr(visual.predictor, 'forward', forbidden)
    args, _, _, features = inputs(batch=2)
    query = torch.randn(2, 6, 64)
    actual = visual(query, features[4], args[3], None, None, 4)
    expected = parent.visual(query, features[4], args[3])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(visual.predicted_readout(features[4], None, None, 4),
                                visual.target_readout(features[4])):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in visual.predictor.parameters())


@pytest.mark.parametrize('age', [5, 6, 7])
def test_exact_current_readout_is_sufficient_for_original_attention(parent, monkeypatch, age):
    visual = DistilledVisualReference(parent.visual)
    args, images, _, features = inputs(seed=50 + age, batch=2)
    query = torch.randn(2, 6, 64)
    target = visual.target_readout(features[age])
    calls = []

    def exact_readout(reference, reference_images, current_images, requested_age):
        assert reference is features[4] and reference_images is images[4]
        assert current_images is images[age] and requested_age == age
        calls.append(requested_age)
        return target

    monkeypatch.setattr(visual, 'predicted_readout', exact_readout)
    actual = visual(query, features[4], args[3], images[4], images[age], age)
    expected = parent.visual(query, features[age], args[3])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [age]
    assert torch.count_nonzero(actual[~args[3]]) == 0
    assert not torch.equal(expected, parent.visual(query, features[4], args[3]))


def test_predictor_input_grid_age_and_residual_computation(parent):
    visual = DistilledVisualReference(parent.visual)
    _, images, _, features = inputs(batch=2)
    captured = []
    handle = visual.predictor.mlp.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    with torch.no_grad():
        visual.predictor.mlp[-1].bias[:64].fill_(.25)
        visual.predictor.mlp[-1].bias[64:].fill_(-.5)
    reference_key, reference_value = visual.target_readout(features[4])
    for age in (5, 6, 7):
        key, value = visual.predicted_readout(features[4], images[4], images[age], age)
        torch.testing.assert_close(key, reference_key + .25, rtol=0, atol=0)
        torch.testing.assert_close(value, reference_value - .5, rtol=0, atol=0)
        assert captured[-1].shape == (2, 72, 193)
        torch.testing.assert_close(captured[-1][..., :64], reference_key, rtol=0, atol=0)
        torch.testing.assert_close(captured[-1][..., 64:128], reference_value, rtol=0, atol=0)
        torch.testing.assert_close(captured[-1][..., 128:192],
                                   visual.predictor.spatial(images[4], images[age]), rtol=0, atol=0)
        torch.testing.assert_close(captured[-1][..., 192],
                                   reference_key.new_full((2, 72), (age - 4) / 3), rtol=0, atol=0)
    handle.remove()


def test_direct_kv_gradient_reaches_cnn_after_zero_head_update(parent):
    model = ReadoutDistillationCondition(parent, 'distilled_readout')
    _, images, _, features = inputs()
    visual = model.visual
    original = {name: value.clone() for name, value in model.state_dict().items()
                if not name.startswith('visual.predictor.')}
    target_features = features[7].requires_grad_(True)
    target = visual.target_readout(target_features)
    optimizer = torch.optim.SGD(visual.predictor.parameters(), lr=.2)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = visual.predicted_readout(features[4], images[4], images[7], 7)
        loss = sum(F.mse_loss(actual, expected) for actual, expected in zip(prediction, target))
        loss.backward()
        assert visual.predictor.mlp[-1].weight.grad.abs().sum() > 0
        cnn_gradient = visual.predictor.spatial.net[0].weight.grad
        assert cnn_gradient is not None
        assert (cnn_gradient.abs().sum() == 0) if step == 0 else (cnn_gradient.abs().sum() > 0)
        assert all(parameter.grad is None for name, parameter in model.named_parameters()
                   if not name.startswith('visual.predictor.'))
        assert target_features.grad is None
        optimizer.step()
    for name, value in original.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    current = visual.predicted_readout(features[4], images[4], images[7], 7)
    withheld = visual.predicted_readout(features[4], images[4], images[4], 7)
    assert any(not torch.equal(a, b) for a, b in zip(current, withheld))


def test_readout_arms_start_from_same_parent_and_zero_predictor_seed(parent):
    torch.manual_seed(123)
    distilled = ReadoutDistillationCondition(parent, 'distilled_readout')
    torch.manual_seed(123)
    joint = ReadoutDistillationCondition(parent, 'joint_readout')
    for name, value in distilled.state_dict().items():
        torch.testing.assert_close(joint.state_dict()[name], value, rtol=0, atol=0)
    assert torch.count_nonzero(joint.visual.predictor.mlp[-1].weight) == 0
    assert torch.count_nonzero(joint.visual.predictor.mlp[-1].bias) == 0
    for name, parameter in joint.named_parameters():
        assert parameter.requires_grad == trainable(name, 'joint_readout'), name
    data = inputs()
    joint.feature_provider = distilled.feature_provider = lambda age, images: data[3][4]
    with torch.no_grad():
        _, actual = unroll(joint, data)
        _, expected = unroll(distilled, data)
    for (output, _), (reference, _) in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_joint_late_age4_unroll_gradient_and_fixed_reader(parent):
    joint = ReadoutDistillationCondition(parent, 'joint_readout')
    data = inputs()
    joint.feature_provider = lambda age, images: data[3][4]
    before = {name: parameter.detach().clone() for name, parameter in joint.named_parameters()
              if not parameter.requires_grad}
    target_before = joint.visual.target_readout(data[3][7])
    optimizer = torch.optim.SGD([p for p in joint.parameters() if p.requires_grad], lr=.1)
    # A downstream condition surrogate exercises the action/condition gradient path.
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        context, actual = unroll(joint, data)
        age4_output = actual[3][0]
        age4_output.retain_grad()
        actual[-1][0].square().mean().backward()
        assert age4_output.grad.abs().sum() > 0
        assert joint.late.delta_encoder.image_encoder[0].weight.grad.abs().sum() > 0
        assert joint.late.condition_updater.up.weight.grad.abs().sum() > 0
        assert joint.visual.predictor.mlp[-1].weight.grad.abs().sum() > 0
        cnn_gradient = joint.visual.predictor.spatial.net[0].weight.grad
        assert (cnn_gradient.abs().sum() == 0) if step == 0 else (cnn_gradient.abs().sum() > 0)
        assert all(parameter.grad is None for name, parameter in joint.named_parameters() if name in before)
        optimizer.step()
    for name, parameter in joint.named_parameters():
        if name in before:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert context.visual_memory is data[3][4]
    for a, b in zip(joint.visual.target_readout(data[3][7]), target_before):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_warm_retain_trains_late_and_original_reader(parent):
    model = ReadoutDistillationCondition(parent, 'warm_retain')
    data = inputs()
    model.feature_provider = lambda age, images: data[3][4]
    _, outputs = unroll(model, data)
    outputs[-1][0].square().mean().backward()
    assert model.late.delta_encoder.image_encoder[0].weight.grad.abs().sum() > 0
    assert model.late.condition_updater.up.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.visual.parameters())
    assert all(p.grad is None for p in model.early.parameters())


def test_predictions_always_use_reference4_and_never_recur(parent):
    model = ReadoutDistillationCondition(parent, 'distilled_readout')
    torch.nn.init.normal_(model.visual.predictor.mlp[-1].weight, std=.05)
    data = inputs()
    args, images, proprio, features = data
    model.feature_provider = lambda age, images: features[4]
    seen = []
    handle = model.visual.predictor.register_forward_pre_hook(lambda module, args: seen.append(args))
    late_pairs = []
    late_handle = model.late.delta_encoder.register_forward_pre_hook(
        lambda module, args: late_pairs.append(args[0]))
    with torch.no_grad():
        context, _ = unroll(model, data)
    handle.remove()
    late_handle.remove()
    reference_key, reference_value = model.visual.target_readout(features[4])
    assert len(seen) == 3
    for age, (key, value, reference_images, current_images, requested_age) in zip((5, 6, 7), seen):
        torch.testing.assert_close(key, reference_key, rtol=0, atol=0)
        torch.testing.assert_close(value, reference_value, rtol=0, atol=0)
        assert reference_images is images[4] and current_images is images[age] and requested_age == age
    for age, pair in zip((4, 5, 6, 7), late_pairs):
        assert pair.previous_images is images[age - 1] and pair.current_images is images[age]
        assert pair.previous_proprio is proprio[age - 1] and pair.current_proprio is proprio[age]
    assert context.visual_memory is features[4] and context.visual_images is images[4]
    first = model.visual.predicted_readout(features[4], images[4], images[7], 7)
    model.visual.predicted_readout(features[0], images[0], images[5], 5)
    second = model.visual.predicted_readout(features[4], images[4], images[7], 7)
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('fault', ['missing', 'unexpected', 'shape'])
def test_parent_transfer_rejects_any_original_mismatch(parent, arm, fault):
    model = ReadoutDistillationCondition(parent, arm)
    state = copy.deepcopy(parent.state_dict())
    if fault == 'missing':
        del state['visual.value.weight']
    elif fault == 'unexpected':
        state['visual.predictor.unexpected'] = torch.zeros(1)
    else:
        state['late.condition_updater.up.weight'] = torch.zeros(1)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match='Parent state mismatch'):
        model.load_parent(state)
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_constructor_validation(parent):
    with pytest.raises(ValueError, match='Unknown readout arm'):
        ReadoutDistillationCondition(parent, 'spatial')
    with pytest.raises(ValueError, match='late_every'):
        ReadoutDistillationCondition(parent.early, 'warm_retain')
    parent.arm = 'midpoint'
    with pytest.raises(ValueError, match='late_every'):
        ReadoutDistillationCondition(parent, 'distilled_readout')


@pytest.mark.parametrize('age', [3, 8, 5.0, torch.tensor(5)])
def test_readout_rejects_invalid_age(parent, age):
    visual = DistilledVisualReference(parent.visual)
    with pytest.raises(ValueError, match='Python int in 4..7'):
        visual.predicted_readout(torch.randn(1, 72, 32), None, None, age)


def test_forward_compiles_fullgraph_and_accepts_new_inputs_without_captured_state(parent):
    torch._dynamo.reset()
    visual = DistilledVisualReference(parent.visual)
    torch.nn.init.normal_(visual.predictor.mlp[-1].weight, std=.03)
    original_forward = visual.forward
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph.forward

    # Match attach_visual: wrap the bound visual.forward, not just predictor.
    visual.forward = torch.compile(original_forward, backend=backend, fullgraph=True, dynamic=False)
    try:
        for seed in (101, 102):
            args, images, _, features = inputs(seed)
            query = torch.randn(1, 6, 64)
            for age in (4, 5, 6, 7):
                actual = visual(query, features[4], args[3], images[4], images[age], age)
                expected = original_forward(query, features[4], args[3], images[4], images[age], age)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if seed == 101:
                compiled_count = len(graphs)
        assert compiled_count >= 2
        assert len(graphs) == compiled_count
        assert not list(visual.named_buffers())
    finally:
        torch._dynamo.reset()
