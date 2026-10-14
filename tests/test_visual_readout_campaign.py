"""CPU-only orchestration tests; no model runtime, rollout or GPU discovery."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools.simvla import visual_readout_campaign as study


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(study.prior, 'IS_RB2', False)
    monkeypatch.setattr(study, 'OUTPUT', tmp_path / 'output')
    monkeypatch.setattr(study, 'INCOMING', tmp_path / 'incoming')
    monkeypatch.setattr(study, 'ROOT', tmp_path / 'source')
    sources = {}
    for source in study.training.FILES:
        study.write_json(study.ROOT / source, dict(test_source=source))
        sources[source] = study.sha(study.ROOT / source)
    c = dict(identity='test', arms=list(study.ARMS), steps=10000, source_sha256=sources,
             manifest=dict(suite='libero_10', environment_seed=7,
                           episodes=[dict(task_id=t, trial_id=r) for t in range(10) for r in range(50)]))
    monkeypatch.setattr(study, 'checked', lambda: c)
    return c


def test_plan_unique_ids_outputs_commands_and_independent_training(campaign):
    jobs = study.jobs(campaign)
    assert len(jobs) == 27
    assert len({j['id'] for j in jobs}) == len({j['summary'] for j in jobs}) == 27
    assert len({tuple(j['cmd']) for j in jobs}) == 27
    plan = {j['id']: j for j in jobs}
    for index, arm in enumerate(study.ARMS):
        smoke, train, export = (plan[prefix + arm] for prefix in ('train_smoke_', 'train_', 'export_'))
        assert smoke['deps'] == []
        assert smoke['completion']['steps'] == 2
        assert train['deps'] == ['train_smoke_' + arm]
        assert train['completion']['steps'] == 10000
        assert export['deps'] == ['train_' + arm]
        deps = (['merge_' + study.ARMS[index - 1]] if index else []) + ['train_' + arm]
        assert plan['eval_smoke_' + arm]['deps'] == deps
        assert plan['merge_' + arm]['deps'] == [f'eval_{arm}_{s}' for s in range(4)]
        for shard in range(4):
            job = plan[f'eval_{arm}_{shard}']
            assert job['deps'] == ['eval_smoke_' + arm]
            assert job['completion']['episodes'] == 125
        assert plan['merge_' + arm]['completion']['episodes'] == 500
    assert study.gpu_ids() == (4, 5, 6, 7)
    assert all(j['cmd'][:4] == [sys.executable, '-u', '-m', study.MODULE] for j in jobs)


def test_rb2_only_ordered_evaluations_wait_for_ready(campaign, monkeypatch):
    monkeypatch.setattr(study.prior, 'IS_RB2', True)
    jobs = study.jobs(campaign)
    assert len(jobs) == len({j['id'] for j in jobs}) == 9
    assert study.gpu_ids() == (0,)
    plan = {j['id']: j for j in jobs}
    for index, arm in enumerate(study.ARMS):
        assert study.shard_count(arm) == 1
        smoke = plan['eval_smoke_' + arm]
        assert smoke['deps'] == (['merge_' + study.ARMS[index - 1]] if index else [])
        done = set(smoke['deps'])
        assert not study.queue.ready(smoke, done)
        study.write_json(Path(smoke['ready_file']), dict(manifest_sha256='manifest'))
        assert study.queue.ready(smoke, done)
        if index:
            assert not study.queue.ready(smoke, set())
        full = plan[f'eval_{arm}_0']
        assert full['ready_file'] == smoke['ready_file']
        assert full['completion']['episodes'] == 500
        assert plan['merge_' + arm]['deps'] == [f'eval_{arm}_0']
    assert all(not j['id'].startswith(('train', 'export')) for j in jobs)


@pytest.mark.parametrize('count', [1, 4])
def test_shards_preserve_original_manifest_and_cover_500_once(campaign, count):
    manifest = campaign['manifest']
    original = list(manifest['episodes'])
    ids = []
    for shard in range(count):
        selected = study.selected_manifest(manifest, shard, count)
        assert selected['episodes'] == original[shard::count]
        assert selected['suite'] == manifest['suite']
        ids.extend((e['task_id'], e['trial_id']) for e in selected['episodes'])
    assert len(ids) == len(set(ids)) == 500
    assert set(ids) == set(study.manifest_ids(manifest))
    assert manifest['episodes'] == original


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'extra', 'foreign', 'boolean'])
def test_invalid_manifest_is_rejected(campaign, kind):
    episodes = campaign['manifest']['episodes']
    if kind == 'missing':
        episodes.pop()
    elif kind == 'duplicate':
        episodes[-1] = episodes[0]
    elif kind == 'extra':
        episodes.append(episodes[0])
    elif kind == 'foreign':
        episodes[-1] = dict(task_id=10, trial_id=49)
    else:
        episodes[0] = dict(task_id=False, trial_id=0)
    with pytest.raises(RuntimeError, match='Long500'):
        study.jobs(campaign)


@pytest.mark.parametrize('shard,count', [(-1, 4), (4, 4), (1, 1), (0, 2), (0, 0)])
def test_invalid_shard(campaign, shard, count):
    with pytest.raises(ValueError, match='Invalid shard'):
        study.selected_manifest(campaign['manifest'], shard, count)


def write_shards(campaign, arm='warm_retain', all_fail=False):
    for shard in range(study.shard_count(arm)):
        out = study.shard_output(arm, shard) / 'rows' / arm
        subset = campaign['manifest']['episodes'][shard::study.shard_count(arm)]
        successes = 0
        for spec in subset:
            success = spec['trial_id'] < 40 and not all_fail
            successes += success
            study.write_json(out / 'episodes' / f"task{spec['task_id']}_trial{spec['trial_id']:02d}.json",
                dict(**spec, identity=campaign['identity'], arm=arm, model_sha256='model',
                     success=success, episode_length=10, policy_ms_total=100., timing_valid=spec['trial_id'] != 0))
        study.write_json(out / 'summary.json', dict(verdict='EVAL_COMPLETE', identity=campaign['identity'],
            arm=arm, episodes=len(subset), successes=successes, success_rate=successes / len(subset),
            hardware='test', elapsed_seconds=5, timing_scope='test'))


@pytest.mark.parametrize('rb2', [False, True])
@pytest.mark.parametrize('all_fail', [False, True])
def test_merge_exact500_and_no_success_threshold(campaign, monkeypatch, rb2, all_fail):
    monkeypatch.setattr(study.prior, 'IS_RB2', rb2)
    write_shards(campaign, all_fail=all_fail)
    summary = study.merge('warm_retain')
    assert summary['episodes'] == 500
    assert summary['successes'] == (0 if all_fail else 400)
    assert summary['success_rate'] == (0. if all_fail else .8)
    assert summary['policy_ms_per_action'] == 10.
    assert summary['timing_valid_episodes'] == 490
    assert summary['shards'] == (1 if rb2 else 4)
    rows = study.read_json(study.OUTPUT / 'rows/warm_retain/episode_outcomes.json')
    assert [(r['task_id'], r['trial_id']) for r in rows] == study.manifest_ids(campaign['manifest'])
    job = next(j for j in study.jobs(campaign) if j['id'] == 'merge_warm_retain')
    assert study.queue.completed(job)
    assert study.audit(campaign)['rows']['warm_retain']['complete']


@pytest.mark.parametrize('corruption', [
    'missing_summary', 'missing_episode', 'duplicate_episode', 'foreign_episode',
    'episode_identity', 'episode_arm', 'model_hash', 'summary_identity', 'summary_arm',
    'summary_verdict', 'summary_count', 'summary_successes', 'runtime', 'timing'])
def test_merge_rejects_incomplete_or_mixed_evidence(campaign, corruption):
    write_shards(campaign)
    out = study.shard_output('warm_retain', 0) / 'rows/warm_retain'
    path = out / 'episodes/task0_trial00.json'
    if corruption == 'missing_summary':
        (out / 'summary.json').unlink()
    elif corruption == 'missing_episode':
        path.unlink()
    elif corruption == 'duplicate_episode':
        study.write_json(path.with_name('duplicate.json'), study.read_json(path))
    elif corruption.startswith('summary_'):
        values = dict(summary_identity=('identity', 'other'), summary_arm=('arm', 'joint_readout'),
                      summary_verdict=('verdict', 'PARTIAL'), summary_count=('episodes', 124),
                      summary_successes=('successes', 0))
        key, value = values[corruption]
        path = out / 'summary.json'
        study.write_json(path, dict(study.read_json(path), **{key: value}))
    elif corruption == 'runtime':
        path = out / 'summary.json'
        study.write_json(path, dict(study.read_json(path), hardware='other'))
    else:
        key, value = dict(foreign_episode=('trial_id', 100), episode_identity=('identity', 'other'),
                          episode_arm=('arm', 'joint_readout'), model_hash=('model_sha256', 'other'),
                          timing=('episode_length', 0))[corruption]
        study.write_json(path, dict(study.read_json(path), **{key: value}))
    with pytest.raises((RuntimeError, FileNotFoundError)):
        study.merge('warm_retain')
    assert not (study.OUTPUT / 'rows/warm_retain/summary.json').exists()


@pytest.mark.parametrize('raises', [False, True])
def test_evaluator_adapter_and_restore(campaign, monkeypatch, raises):
    old_study = object()
    fake = SimpleNamespace(study=old_study, write_json=study.write_json)

    def evaluate(arm, smoke):
        adapter = fake.study
        assert adapter.OUTPUT == study.shard_output(arm, 2)
        assert adapter.initial_model is study.training.initial_model
        assert adapter.model_payload is study.training.model_payload
        assert adapter.previous is study.training.previous
        assert len(adapter.checked()['manifest']['episodes']) == 125
        assert len(campaign['manifest']['episodes']) == 500
        fake.write_json(adapter.OUTPUT / 'progress.json', dict(total=500, completed=0))
        if raises:
            raise RuntimeError('simulated evaluation failure')
        assert smoke

    fake.evaluate = evaluate
    monkeypatch.setitem(sys.modules, 'tools.simvla.visual_midpoint_eval', fake)
    import tools.simvla
    monkeypatch.setattr(tools.simvla, 'visual_midpoint_eval', fake, raising=False)
    if raises:
        with pytest.raises(RuntimeError, match='simulated'):
            study.evaluate('warm_retain', 2, smoke=True)
    else:
        study.evaluate('warm_retain', 2, smoke=True)
    assert fake.study is old_study and fake.write_json is study.write_json
    assert study.read_json(study.shard_output('warm_retain', 2) / 'progress.json')['total'] == 125


def test_cpu_audit_hashes_and_merged_evidence(campaign):
    assert study.audit(campaign)['verdict'] == 'AUDIT_PASS'
    write_shards(campaign)
    study.merge('warm_retain')
    path = study.OUTPUT / 'rows/warm_retain/episode_outcomes.json'
    study.write_json(path, [])
    with pytest.raises(RuntimeError, match='Merged row'):
        study.audit(campaign)
    path.unlink()
    (study.OUTPUT / 'rows/warm_retain/summary.json').unlink()
    source = next(iter(campaign['source_sha256']))
    study.write_json(study.ROOT / source, dict(changed=True))
    with pytest.raises(RuntimeError, match='Source changed'):
        study.audit(campaign)


def test_rb2_cpu_audit_can_wait_and_validates_ready(campaign, monkeypatch):
    monkeypatch.setattr(study.prior, 'IS_RB2', True)
    assert set(study.audit(campaign)['transfers'].values()) == {'waiting_for_READY'}
    directory = study.INCOMING / 'warm_retain'
    study.write_json(directory / 'model.pt', dict(fake_checkpoint=True))
    study.write_json(directory / 'manifest.json', dict(source_sha256=campaign['source_sha256'],
                                                      checkpoint_sha256=study.sha(directory / 'model.pt')))
    study.write_json(directory / 'READY.json', dict(manifest_sha256=study.sha(directory / 'manifest.json')))
    assert study.audit(campaign)['transfers']['warm_retain'] == 'hashes_verified'
    study.write_json(directory / 'model.pt', dict(changed=True))
    with pytest.raises(RuntimeError, match='checkpoint changed'):
        study.audit(campaign)


def test_source_hashes_include_training(campaign):
    campaign['source_sha256'].pop('tools/simvla/visual_readout_train.py')
    with pytest.raises(RuntimeError, match='shared training source'):
        study.validate_contract(campaign)


@pytest.mark.parametrize('rb2', [False, True])
def test_main_uses_standard_lease_queue_and_only_allowed_gpus(campaign, monkeypatch, rb2):
    monkeypatch.setattr(study.prior, 'IS_RB2', rb2)
    monkeypatch.setattr(study, 'prepare', lambda: campaign)
    previous = dict(path='previous', lock='queue.lock')
    monkeypatch.setattr(study, 'wait_for_predecessor', lambda: previous)
    calls = []
    monkeypatch.setattr(study.queue, 'run_queue', lambda *a, **kw: calls.append((a, kw)) or 1)
    assert study.main(['preflight']) == 0
    assert not calls
    assert study.main([]) == 1
    args, kwargs = calls[0]
    assert args[0] == study.OUTPUT
    assert kwargs['gpus'] == ((0,) if rb2 else (4, 5, 6, 7))
    assert kwargs['predecessor'] == [previous]
    assert kwargs['environment'] is study.prior.environment
    assert kwargs['timeout'] == 24 * 3600


@pytest.mark.parametrize('command', ['train-smoke', 'train', 'export'])
def test_training_command_dispatch(campaign, monkeypatch, command):
    calls = []
    monkeypatch.setattr(study.training, 'train', lambda arm, smoke: calls.append((arm, smoke)))
    monkeypatch.setattr(study.training, 'export', lambda arm: calls.append((arm, 'export')))
    assert study.main([command, '--arm', 'joint_readout']) == 0
    assert calls == [('joint_readout', 'export' if command == 'export' else command == 'train-smoke')]


@pytest.mark.parametrize('args', [
    ['eval'], ['train'], ['export'], ['merge'], ['eval', '--arm', 'wrong'],
    ['eval', '--arm', 'warm_retain', '--shard', '4'],
    ['eval', '--arm', 'warm_retain', '--shard', '-1']])
def test_cli_rejects_invalid_arguments(campaign, args):
    with pytest.raises(SystemExit) as error:
        study.main(args)
    assert error.value.code == 2


def test_rb2_rejects_training_and_extra_shards(campaign, monkeypatch):
    monkeypatch.setattr(study.prior, 'IS_RB2', True)
    for args in (['train'], ['train-smoke'], ['export'], ['eval', '--shard', '1']):
        with pytest.raises(SystemExit) as error:
            study.main([*args, '--arm', 'joint_readout'])
        assert error.value.code == 2


def test_predecessor_waits_for_completion_not_all_assigned(tmp_path, monkeypatch):
    monkeypatch.setattr(study.predecessor, 'OUTPUT', tmp_path)
    (tmp_path / 'queue.lock').touch()
    state = dict(phase='running', pending=[], active={'4': {}}, completed=[], failed={}, total_jobs=1)
    study.write_json(tmp_path / 'queue_status.json', state)
    sleeps = []

    def finish(_seconds):
        sleeps.append(True)
        study.write_json(tmp_path / 'queue_status.json', dict(state, phase='complete', active={}, completed=['last']))

    monkeypatch.setattr(study.time, 'sleep', finish)
    result = study.wait_for_predecessor()
    assert len(sleeps) == 1
    assert result == dict(path=str(tmp_path), lock='queue.lock')


def test_predecessor_failure_and_timeout_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(study.predecessor, 'OUTPUT', tmp_path)
    with pytest.raises(TimeoutError):
        study.wait_for_predecessor(timeout=0)
    study.write_json(tmp_path / 'queue_status.json', dict(phase='finished_with_failures'))
    with pytest.raises(RuntimeError, match='with failures'):
        study.wait_for_predecessor()


def test_queue_retries_once_and_preserves_failure_on_resume(tmp_path, monkeypatch):
    queue = study.queue
    monkeypatch.setattr(queue.socket, 'gethostname', lambda: 'jbrserver1')
    monkeypatch.setattr(queue.Path, 'home', lambda: tmp_path / 'home')
    monkeypatch.setattr(queue, 'gpu_idle', lambda gpu: True)
    monkeypatch.setattr(queue, 'predecessor_pending', lambda spec: False)
    ticks = iter(range(0, 10000, 10))
    monkeypatch.setattr(queue.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(queue.time, 'sleep', lambda seconds: None)
    starts = []

    def fail(cmd, **kwargs):
        starts.append(cmd)
        assert 'GNAROSHI_GPU_LEASE_FD' in kwargs['env'] and kwargs['pass_fds']
        return SimpleNamespace(pid=123, returncode=1, poll=lambda: 1)

    monkeypatch.setattr(queue.subprocess, 'Popen', fail)
    jobs = [dict(id='failure', cmd=['mock'], deps=[], summary=str(tmp_path / 'missing.json'),
                 completion=dict(verdict='PASS')),
            dict(id='dependent', cmd=['mock'], deps=['failure'], summary=str(tmp_path / 'dependent.json'),
                 completion=dict(verdict='PASS'))]
    for _ in range(2):
        assert queue.run_queue(tmp_path, jobs, gpus=(4,), predecessor=[], environment=lambda gpu: {}, cwd=tmp_path) == 1
    state = study.read_json(tmp_path / 'queue_status.json')
    assert len(starts) == 2
    assert state['attempts'] == {'failure': 2}
    assert state['failed']['dependent']['reason'] == 'dependency_failed'


@pytest.mark.parametrize('python_rc,tee_rc,expected', [(7, 0, 7), (0, 13, 13), (0, 0, 0)])
def test_wrapper_logs_stderr_real_status_and_rb2_libraries(tmp_path, python_rc, tee_rc, expected):
    root = Path(__file__).resolve().parents[1]
    source = root / 'architectures/simvla/wrappers/run_visual_readout_distillation.sh'
    subprocess.run(['bash', '-n', str(source)], check=True)
    wrapper = tmp_path / 'repo/architectures/simvla/wrappers/run_visual_readout_distillation.sh'
    wrapper.parent.mkdir(parents=True)
    storage = tmp_path / 'storage'
    wrapper.write_text(source.read_text().replace('/home/mingyujung/private/gnaroshi_vla_storage', str(storage)))
    python = storage / 'envs/simvla/libero_mujoco237/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/usr/bin/env bash\nprintf "stdout\\n"\nprintf "stderr\\n" >&2\n'
                      'printf "%s\\n" "$LD_LIBRARY_PATH" "$__EGL_VENDOR_LIBRARY_FILENAMES"\n'
                      f'exit {python_rc}\n')
    python.chmod(0o755)
    script = 'hostname() { printf "jbr-TRX50\\n"; }; export -f hostname; '
    if tee_rc:
        script += f'tee() {{ command tee "$@"; return {tee_rc}; }}; export -f tee; '
    script += 'export LD_LIBRARY_PATH=existing; bash "$1" preflight'
    result = subprocess.run(['bash', '-c', script, 'test', str(wrapper)], capture_output=True, text=True)
    assert result.returncode == 0
    logs = storage / 'results/simvla/visual_readout_distillation/preserved_reader_nfe1_v1/logs'
    assert (logs / 'preflight.status').read_text().strip() == str(expected)
    output = (logs / 'preflight.log').read_text()
    assert 'stdout' in output and 'stderr' in output
    runtime = storage / 'runtime/nvidia_580_173/root'
    assert str(runtime / 'usr/lib/x86_64-linux-gnu') + ':existing' in output
    assert str(runtime / 'usr/share/glvnd/egl_vendor.d/10_nvidia.json') in output
