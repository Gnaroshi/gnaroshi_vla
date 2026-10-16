import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from methods.latentloop.modules.bounded_history import BoundedHistoryCondition
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.modules.short_visual_training import (
    ARMS, ShortVisualCondition, build_model, gradient_coverage, parameter_counts,
)
from methods.latentloop.modules.visual_token_transport import TokenTransport
from tools.simvla import short_visual_train as training
from tools.simvla.condition_output_split_train import unroll


@pytest.fixture(scope='module', autouse=True)
def cpu_only():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(threads)


@pytest.fixture
def native():
    torch.manual_seed(37)
    return NativeSimVLAV0(condition_dim=32, max_tokens=16)


@pytest.fixture
def parent(native):
    short = BoundedHistoryCondition(copy.deepcopy(native), 'recurrent')
    nn.init.normal_(short.condition_updater.up.weight, std=.04)
    nn.init.normal_(short.condition_updater.gate_head.weight, std=.02)
    nn.init.constant_(short.condition_updater.gate_head.bias, -1.)
    return short.eval().requires_grad_(False)


@pytest.fixture
def contract(tmp_path, parent):
    path = tmp_path / 'parent.pt'
    torch.save(dict(step=10000, model=parent.state_dict(), contract=dict(
        training_intervals=[4], action_mode='naive1', bounded_history_variant='recurrent',
        initialization='fresh', seed=7, total_training_steps=10000, continuation=None)), path)
    checksum = training.sha(path)
    return dict(identity='test-contract', seed=7, steps=10000, batch_size=2, warmup=500,
        learning_rate=1e-4, condition_weight=.05, training_ages=[4, 5, 6, 7],
        objective='first5 action L1 +0.05 all-unrolled scaled MSE', gradient_routing='unroll',
        gradient_audit='post-update2 backward, no optimizer step',
        snapshots={}, features=dict(root=str(tmp_path / 'features'), identity='features'),
        model=dict(checkpoint=str(path), sha256=checksum, config={'seed': 7}),
        initialization={arm: dict(parent_fresh_sha256=checksum if arm == 'protected_short' else 'NONE',
            parent_training_steps=10000 if arm == 'protected_short' else 0,
            parent_training_seconds=1906.476 if arm == 'protected_short' else 0.,
            total_training_steps=20000 if arm == 'protected_short' else 10000)
            for arm in ARMS})


def sequence(batch=2):
    g = torch.Generator().manual_seed(91)
    anchor = torch.randn(batch, 6, 32, generator=g)
    return dict(anchor_condition=anchor,
        image_sequence=torch.randint(0, 256, (batch, 8, 2, 13, 17, 3), generator=g, dtype=torch.uint8),
        proprio_sequence=torch.randn(batch, 8, 8, generator=g),
        teacher_conditions=anchor[:, None] + .3 * torch.randn(batch, 7, 6, 32, generator=g),
        valid_mask=torch.tensor([[True, True, False, True, False, True]]).expand(batch, -1),
        group_ids=torch.arange(6)[None].expand(batch, -1),
        explicit_noises=torch.randn(batch, 7, 10, 7, generator=g),
        teacher_actions=torch.randn(batch, 7, 10, 7, generator=g))


class Features:
    def __init__(self, *args):
        self.calls = []

    def bind(self, seq):
        self.seq = seq

    def __call__(self, age, images):
        assert age == 4
        self.calls.append(age)
        return self.seq['anchor_condition'].repeat(1, 12, 1)


class Action:
    """Differentiable frozen CPU action surrogate, not a SimVLA measurement."""

    def __init__(self):
        self.original = nn.Linear(32, 7).requires_grad_(False)
        self.action_space = SimpleNamespace(normalize_action=lambda value: value)

    def decode_action_from_condition(self, condition, proprio, *, steps, initial_noise,
                                     requires_grad=False, return_debug=False):
        result = self.original(condition.mean(1))[:, None].expand(-1, 10, -1) + .01 * initial_noise
        return SimpleNamespace(final_action_latent=result) if return_debug else result


@pytest.mark.parametrize('arm', ARMS)
def test_factory_shapes_freeze_and_no_alias(contract, native, parent, arm):
    before = {n: v.clone() for n, v in native.state_dict().items()}
    model = training.initial_model(contract, arm, native)
    assert isinstance(model, ShortVisualCondition)
    assert model.arm == 'joint_readout' and model.short_training_arm == arm
    assert isinstance(model.visual.predictor, TokenTransport)
    assert model.visual.predictor.appearance is not None
    assert not hasattr(model, 'teacher_visual')
    assert not model.visual.output.weight.count_nonzero()
    assert not model.visual.predictor.appearance[-1].weight.count_nonzero()
    for name, value in native.state_dict().items():
        assert torch.equal(value, before[name])
    for name, p in model.named_parameters():
        assert p.requires_grad == (arm == 'fresh_joint' or not name.startswith('early.'))
    for name, value in model.early.state_dict().items():
        assert torch.equal(value, model.late.state_dict()[name])
        assert value.data_ptr() != model.late.state_dict()[name].data_ptr()
        if arm == 'protected_short':
            assert torch.equal(value, parent.state_dict()[name])
    assert not ({p.data_ptr() for p in model.parameters()} & {p.data_ptr() for p in native.parameters()})
    seq = sequence()
    features = Features()
    features.bind(seq)
    model.feature_provider = features
    with torch.no_grad():
        result, _, _ = unroll(model, seq, 7, 8)
    assert result.shape == (2, 6, 32)
    assert features.calls == [4]
    assert [model.uses_vision(age) for age in range(1, 8)] == [False, False, False, True, False, False, False]
    assert torch.equal(result[~seq['valid_mask']], seq['anchor_condition'][~seq['valid_mask']])


def test_fresh_independent_of_native_weights_and_parent_file(contract, native, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Fresh construction accessed checkpoint weights')
    monkeypatch.setattr(training, '_protected_payload', forbidden)
    first = training.initial_model(contract, 'fresh_joint', native)
    with torch.no_grad():
        for p in native.parameters():
            p.fill_(float('nan'))
    second = training.initial_model(contract, 'fresh_joint', native)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    assert not first.early.condition_updater.up.weight.count_nonzero()


def test_same_visual_seed_and_exact_parameter_counts(contract, native):
    fresh = training.initial_model(contract, 'fresh_joint', native)
    protected = training.initial_model(contract, 'protected_short', native)
    for name, value in fresh.visual.state_dict().items():
        assert torch.equal(value, protected.visual.state_dict()[name])
    a, b = parameter_counts(fresh), parameter_counts(protected)
    assert a['total'] == b['total'] == a['trainable']
    assert a['total'] - b['trainable'] == b['blocks']['early']['total']
    assert b['blocks']['early']['trainable'] == 0
    assert b['blocks']['visual']['total'] == b['blocks']['visual']['trainable']
    assert a['total'] == sum(block['total'] for block in a['blocks'].values())


def test_production_shape_parameter_count():
    native = NativeSimVLAV0()
    model = build_model(native, 'fresh_joint')
    counts = parameter_counts(model)
    assert counts['total'] == counts['trainable'] == 1387523
    assert counts['blocks']['early']['total'] == counts['blocks']['late']['total'] == 584577
    assert counts['blocks']['visual']['trainable'] == 218369
    model.early.requires_grad_(False)
    assert parameter_counts(model)['trainable'] == 802946


@pytest.mark.parametrize('arm', ARMS)
def test_k1_bypass_and_invalid_intervals(contract, native, arm, monkeypatch):
    model = training.initial_model(contract, arm, native)
    def forbidden(*args, **kwargs):
        raise AssertionError('K1 called an updater or feature provider')
    monkeypatch.setattr(model.early, 'prepare', forbidden)
    monkeypatch.setattr(model.early, 'predict', forbidden)
    monkeypatch.setattr(model.late, 'predict', forbidden)
    model.feature_provider = forbidden
    assert model.prepare(None, None, None, None, None, interval=1) is None
    monkeypatch.undo()
    seq = sequence()
    for interval in (0, 5, 9, True, 1., None):
        with pytest.raises(ValueError):
            model.prepare(seq['anchor_condition'], seq['image_sequence'][:, 0],
                seq['proprio_sequence'][:, 0], seq['valid_mask'], seq['group_ids'], interval)


@pytest.mark.parametrize('arm', ARMS)
def test_k4_parent_parity_only_protected(contract, native, parent, arm):
    model = training.initial_model(contract, arm, native)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(.01)
        seq = sequence()
        actual, _, _ = unroll(model, seq, 3, 4)
        expected, _, _ = unroll(parent, seq, 3, 4)
    assert torch.equal(actual, expected) == (arm == 'protected_short')
    assert model.feature_provider is None


@pytest.mark.parametrize('arm', ARMS)
def test_two_updates_then_gradient_audit(contract, native, arm):
    model = training.initial_model(contract, arm, native)
    features, action, seq = Features(), Action(), sequence()
    model.feature_provider = features
    early_before = {n: v.clone() for n, v in model.early.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-7, weight_decay=0)
    observations = []
    for step, age in enumerate((5, 7), 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = training.loss_for(contract, model, action, seq, age, features)
        loss.backward()
        observations.append(gradient_coverage(model))
        assert observations[-1]['visual.predictor.encoder'] == (arm == 'protected_short' and step == 2)
        if arm == 'protected_short' and step == 2:
            assert observations[-1]['visual.output']
        training.frozen_audit(model, action.original)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.param_groups[0]['lr'] = 4e-7
    assert not observations[0]['visual.key']
    before_audit = {n: v.clone() for n, v in model.state_dict().items()}
    optimizer.zero_grad(set_to_none=True)
    training.loss_for(contract, model, action, seq, 7, features)[0].backward()
    coverage = gradient_coverage(model)
    assert all(coverage.values()), coverage
    for name, value in model.state_dict().items():
        assert torch.equal(value, before_audit[name])
    if arm == 'fresh_joint':
        assert coverage['early.delta_encoder'] and coverage['early.condition_updater']
        assert any(not torch.equal(v, early_before[n]) for n, v in model.early.state_dict().items())
    else:
        assert all(p.grad is None for p in model.early.parameters())
        assert all(torch.equal(v, early_before[n]) for n, v in model.early.state_dict().items())
    assert training.frozen_audit(model, action.original)['gradient_bytes'] == 0


def test_strict_parent_provenance_and_shapes(contract, native):
    path = contract['model']['checkpoint']
    saved = torch.load(path, weights_only=False)
    saved['contract']['initialization'] = 'continuation'
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match='parent changed'):
        training.initial_model(contract, 'protected_short', native)
    contract['model']['sha256'] = training.sha(path)
    with pytest.raises(RuntimeError, match='fresh10K'):
        training.initial_model(contract, 'protected_short', native)
    with pytest.raises(RuntimeError):
        build_model(native, 'protected_short', early_state={})
    with pytest.raises(ValueError):
        build_model(native, 'fresh_joint', early_state={})


def test_sampling_and_source_lock(contract):
    assert [training.sampling_step(contract, i)[1] for i in range(1, 9)] == [4, 5, 6, 7] * 2
    assert [training.sampling_step(contract, i, True)[1] for i in (1, 2)] == [5, 7]
    for step in range(1, 8):
        a, _ = training.sampling_step(contract, step)
        b, _ = training.sampling_step(contract, step)
        assert [a.randrange(100) for _ in range(2)] == [b.randrange(100) for _ in range(2)]
    assert len(training.FILES) == len(set(training.FILES))
    assert all((training.ROOT / path).is_file() for path in training.FILES)


@pytest.mark.parametrize('arm', ARMS)
def test_cpu_smoke_save_resume_validation_and_payload(contract, native, arm, monkeypatch, tmp_path):
    from architectures.simvla.adapters.latentloop.efficient_multirate import shared_refinement_train as runtime
    from tools.simvla import error_compensation_common, visual_midpoint_features
    action = Action()
    seq = sequence(batch=1)
    seq['teacher_actions'] = torch.stack([action.decode_action_from_condition(
        seq['teacher_conditions'][:, j], seq['proprio_sequence'][:, j + 1],
        steps=10, initial_noise=seq['explicit_noises'][:, j]) for j in range(7)], 1)
    item = {name: value[0] for name, value in seq.items()}
    item.update(task_id=0, episode_id='cpu', anchor_query_index=0,
                language_instruction='cpu', query_ids=[str(i) for i in range(8)])
    class Dataset:
        identities = [(0, 'cpu', i) for i in range(30)]
        def __len__(self):
            return 30
        def __getitem__(self, index):
            return item
        def contract(self):
            return dict(split='cpu')
    dataset = Dataset()
    monkeypatch.setattr(training, 'OUTPUT', tmp_path / 'run')
    monkeypatch.setattr(training.prior, 'IS_RB2', False)
    monkeypatch.setattr(training, 'checked', lambda: contract)
    monkeypatch.setattr(error_compensation_common, 'configure', lambda cfg: None)
    monkeypatch.setattr(visual_midpoint_features, 'CachedFeatures', Features)
    monkeypatch.setattr(runtime, 'load_runtime', lambda *args:
        (torch.device('cpu'), native, action.original, action, dataset, dataset))
    training.train(arm, smoke=True)
    out = training.OUTPUT / 'train_smoke' / arm
    summary = training.read_json(out / 'summary.json')
    assert summary['steps'] == 2 and summary['verdict'] == 'TRAIN_SMOKE_PASS'
    assert summary['total_training_steps'] == (10002 if arm == 'protected_short' else 2)
    assert summary['total_training_seconds'] == summary['prior_training_seconds'] + summary['training_seconds']
    assert summary['validation_seconds'] > 0
    assert len(training.read_json(out / 'validation.json')['records']) == 14
    assert len((out / 'progress.jsonl').read_text().splitlines()) == 2
    audit = training.read_json(out / 'gradient_audit.json')
    assert audit['additional_optimizer_steps'] == 0 and all(audit['coverage'].values())
    payload, checksum = training.model_payload(contract, arm, smoke=True)
    assert payload['step'] == 2 and payload['identity'] == contract['identity']
    training.train(arm, smoke=True)
    assert training.sha(out / 'latest.pt') == checksum
    assert len((out / 'progress.jsonl').read_text().splitlines()) == 2
    with pytest.raises(RuntimeError, match='contract mismatch'):
        training._check_saved({**payload, 'contract_sha256': 'bad'}, payload['contract'], 2)
    with pytest.raises(RuntimeError, match='contract mismatch'):
        training._check_saved({**payload, 'step': 3}, payload['contract'], 2)
    summary['arm'] = 'wrong'
    training.write_json(out / 'summary.json', summary)
    with pytest.raises(RuntimeError, match='mixed checkpoint'):
        training.model_payload(contract, arm, smoke=True)


def test_checked_self_contained_and_rejects_drift(contract, monkeypatch, tmp_path):
    from tools.simvla import geometry_target_train
    monkeypatch.setattr(training, 'OUTPUT', tmp_path)
    def forbidden():
        raise AssertionError('Nested old training checked() must not run')
    monkeypatch.setattr(geometry_target_train, 'checked', forbidden)
    c = dict(contract, source_sha256={}, input_sources={}, frozen_original_sha256={})
    c.pop('identity')
    c['identity'] = training.prior.digest(c)
    training.write_json(tmp_path / 'contract.json', c)
    assert training.checked() == c
    c['steps'] += 1
    training.write_json(tmp_path / 'contract.json', c)
    with pytest.raises(RuntimeError, match='identity'):
        training.checked()


def test_prepare_copies_only_inputs_and_costs(contract, monkeypatch, tmp_path):
    from tools.simvla import geometry_target_train, visual_midpoint, error_compensation_common
    monkeypatch.setattr(training, 'OUTPUT', tmp_path / 'new')
    monkeypatch.setattr(geometry_target_train, 'OUTPUT', tmp_path / 'geometry')
    monkeypatch.setattr(visual_midpoint, 'OUTPUT', tmp_path / 'vision')
    checkpoint = contract['model']['checkpoint']
    training.write_json(tmp_path / 'summary.json', dict(steps=10000, prior_training_seconds=0,
        verdict='TRAIN_AND_OFFLINE_COMPLETE', checkpoint=checkpoint,
        checkpoint_sha256=contract['model']['sha256'], training_seconds=1906.476))
    cfg = dict(seed=7, training_k_c=8, cache=str(tmp_path / 'cache'),
               condition_checkpoint=checkpoint, norm_stats=str(tmp_path / 'norm.json'))
    training.write_json(tmp_path / 'norm.json', {'normalization': 'test'})
    training.write_json(tmp_path / 'cache/manifest.json', {'identity': 'teacher'})
    training.write_json(tmp_path / 'snapshot/config.json', {'original': 'test'})
    training.write_json(tmp_path / 'k4_eval.json', {'successes': 454, 'episodes': 500})
    model = dict(contract['model'], config=cfg, source_summary=str(tmp_path / 'k4_eval.json'),
                 source_summary_sha256=training.sha(tmp_path / 'k4_eval.json'))
    old = dict(model=model, manifest={'episodes': []}, trained_reader_sha256='forbidden60k')
    training.write_json(geometry_target_train.OUTPUT / 'contract.json', old)
    training.write_json(visual_midpoint.OUTPUT / 'contract.json', dict(identity='vision', model=model))
    calls = []
    monkeypatch.setattr(geometry_target_train, 'checked', lambda: calls.append('prepare') or old)
    monkeypatch.setattr(error_compensation_common, 'snapshots', lambda cfg:
        dict(checkpoint_snapshot=str(tmp_path / 'snapshot'), backbone_snapshot=str(tmp_path / 'snapshot')))
    c = training.prepare()
    assert c['model'] == model and c['model'] is not model
    assert 'trained_reader_sha256' not in c and calls == ['prepare']
    assert c['initialization']['fresh_joint']['parent_training_seconds'] == 0
    assert c['initialization']['protected_short']['parent_training_seconds'] == 1906.476
    assert c['initialization']['protected_short']['total_training_steps'] == 20000
    assert training.checked() == c and calls == ['prepare']
    assert training.prepare() == c
    training.write_json(tmp_path / 'norm.json', {'normalization': 'changed'})
    with pytest.raises(RuntimeError, match='Frozen original asset changed'):
        training.checked()


@pytest.mark.parametrize('arm', ARMS)
def test_early_action_unroll_without_extra_direct_action_examples(contract, native, arm):
    model = training.initial_model(contract, arm, native)
    features, action, seq = Features(), Action(), sequence()
    model.feature_provider = features
    calls = []
    decode = action.decode_action_from_condition
    def count(*args, **kwargs):
        calls.append(kwargs['steps'])
        return decode(*args, **kwargs)
    action.decode_action_from_condition = count
    loss, _, _ = training.loss_for(dict(contract, condition_weight=0.), model, action, seq, 7, features)
    loss.backward()
    assert calls == [1]
    if arm == 'fresh_joint':
        assert model.early.condition_updater.up.weight.grad.count_nonzero()
    else:
        assert all(p.grad is None for p in model.early.parameters())
