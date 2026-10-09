import copy
import csv
import json
from types import SimpleNamespace

import pytest
import torch

from methods.latentloop.modules.condition_output_split import ConditionOutputSplit, SplitContext
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from tools.simvla import condition_interval_compare as subject


@pytest.mark.parametrize('k', [2, 3, 4, 8])
def test_shorter_intervals_have_exactly_same_prediction_prefix(k):
    torch.set_num_threads(1)
    torch.manual_seed(81)
    model = ConditionOutputSplit(NativeSimVLAV0(condition_dim=16, max_tokens=8), 'carry_base').eval()
    for head in (model.condition_updater, model.action_condition_updater):
        torch.nn.init.normal_(head.up.weight, std=.1)
    z = torch.randn(1, 4, 16)
    images = torch.randint(0, 256, (8, 1, 2, 64, 64, 3), dtype=torch.uint8)
    proprio = torch.randn(8, 1, 8)
    valid = torch.ones(1, 4, dtype=torch.bool); groups = torch.zeros(1, 4, dtype=torch.long)
    short = model.prepare(z, images[0], proprio[0], valid, groups, k)
    original = SplitContext(z, z, images[0], proprio[0], valid, groups, 8)
    with torch.inference_mode():
        for age in range(1, k):
            a, ax = model.predict(short, age, images[age], proprio[age])
            b, bx = model.predict(original, age, images[age], proprio[age])
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(ax['carried'], bx['carried'], rtol=0, atol=0)
    with pytest.raises(ValueError):
        model.predict(short, k, images[0], proprio[0])


@pytest.mark.parametrize('k', [0, 1, 5, 9, True, 2.0, '2'])
def test_invalid_interval_rejected(k):
    model = ConditionOutputSplit(NativeSimVLAV0(condition_dim=16, max_tokens=8), 'carry_base')
    with pytest.raises(ValueError):
        model.prepare(None, None, None, None, None, k)


def test_only_three_missing_rows_are_queued():
    plan = subject.jobs()
    assert {x['id'] for x in plan} == {'ours_k2', 'ours_k3', 'bridge_k4'}
    assert len(plan) == 3
    assert all(x['completion']['episodes'] == 500 for x in plan)
    assert set(subject.REFERENCES).isdisjoint(x['id'] for x in plan)
    assert len(subject.ROWS) == 6


@pytest.mark.parametrize('row', list(subject.ROWS))
@pytest.mark.parametrize('q', [1, 2, 3, 4, 9, 181])
def test_invocation_counts_preserve_nfe1_and_condition_interval(row, q):
    method, k = subject.ROWS[row]
    values = subject.expected_counts(row, q)
    refresh = sum(t % k == 0 for t in range(q))
    assert values['num_full_vlm_calls'] == refresh
    assert values['num_condition_updater_calls'] == q-refresh
    assert values['num_action_transformer_calls'] == q
    assert values['num_generation_decoder_only_steps'] == 0
    assert values['num_latent_bridge_calls'] == (q-refresh if method == 'bridge' else 0)


def test_bridge_uses_existing_euler_decode_without_changing_noise_key(monkeypatch):
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_eval import SynchronizedNaiveNFE3Policy
    policy = SimpleNamespace(flow_steps=10)
    monkeypatch.setattr(subject, 'attach_policy', lambda *a: policy)
    result = subject.policy_factory(SimpleNamespace(loop=None), {}, 'bridge_k4', {})
    assert result._decode.__func__ is SynchronizedNaiveNFE3Policy._decode
    assert result.NFE == result.nfe == 1
    assert result.refresh_every == 4 and result.flow_steps == 10


def fixture_reference(tmp_path):
    row = 'bridge_f2_nfe1'; contract = {'fixture': True}
    identity = subject.campaign.digest(dict(campaign=subject.campaign.digest(contract), suite='libero_10', seed='seed01', row=row, smoke=False))
    records = [dict(task_id=t, trial_id=i, success=1, episode_length=5, policy_ms_total=20., timing_valid=True)
        for t in range(10) for i in range(50)]
    with (tmp_path/'outcomes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    result = dict(identity=identity, episodes=500, verdict='EPISODES_COMPLETE', timing_valid_episodes=500,
        successes=500, success_rate=1., executed_actions=2500, pooled_policy_ms_per_action=4.)
    (tmp_path/'summary.json').write_text(json.dumps(result))
    return row, contract, result


def test_reuse_recalculates_success_and_latency(tmp_path):
    row, contract, result = fixture_reference(tmp_path)
    assert subject.validate_reference(tmp_path, contract, row) == result
    for key, value in [('successes', 499), ('pooled_policy_ms_per_action', 3.9), ('identity', 'bad'), ('timing_valid_episodes', 499)]:
        (tmp_path/'summary.json').write_text(json.dumps({**result, key: value}))
        with pytest.raises(RuntimeError): subject.validate_reference(tmp_path, contract, row)


def test_reuse_rejects_duplicate_episode_ids(tmp_path):
    row, contract, _ = fixture_reference(tmp_path)
    p = tmp_path/'outcomes.csv'
    lines = p.read_text().splitlines()
    lines[-1] = lines[1]
    p.write_text('\n'.join(lines)+'\n')
    with pytest.raises(RuntimeError): subject.validate_reference(tmp_path, contract, row)


def test_reference_cell_cannot_be_rerun():
    with pytest.raises(RuntimeError): subject.cell({}, 'ours_k4')


def test_model_spec_is_exact_requested_checkpoint(monkeypatch):
    monkeypatch.setattr(subject, 'ready_spec', lambda *a: dict(sha256=subject.OURS_SHA))
    assert subject.model_spec()['sha256'] == subject.OURS_SHA
    monkeypatch.setattr(subject, 'ready_spec', lambda *a: dict(sha256='another'))
    with pytest.raises(RuntimeError): subject.model_spec()


def test_package_audit_rejects_other_runtime_changes():
    assert subject.audit_packages(dict(packages=['torch==x']), dict(packages=['torch==x'])) is None
    with pytest.raises(RuntimeError):
        subject.audit_packages(dict(packages=['torch==x']), dict(packages=['torch==y']))


def test_package_audit_checks_explicit_import_root_and_revision(tmp_path, monkeypatch):
    revision = '8f1084e3132a39270c3a13ebe37270a43ece2a01'
    old = dict(packages=['torch==x', 'libero==0.1.0'], config=dict(libero_root=str(tmp_path)))
    new = dict(packages=['torch==x', f'-e git+https://github.com/Lifelong-Robot-Learning/LIBERO.git@{revision}#egg=libero'], config=dict(libero_root=str(tmp_path)))
    path = tmp_path/'libero/libero/__init__.py'; path.parent.mkdir(parents=True); path.write_text('')
    monkeypatch.setattr(subject.subprocess, 'check_output', lambda *a, **kw: revision+'\n')
    assert subject.audit_packages(old, new)['git_revision'] == revision
    with pytest.raises(RuntimeError):
        subject.audit_packages(old, {**new, 'config': dict(libero_root='different')})
    monkeypatch.setattr(subject.subprocess, 'check_output', lambda *a, **kw: 'other\n')
    with pytest.raises(RuntimeError): subject.audit_packages(old, new)
