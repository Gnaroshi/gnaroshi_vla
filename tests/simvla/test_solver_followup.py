from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from tools.simvla import bridge_coarse_followup as bridge
from tools.simvla import gpu_followup_queue as queue
from tools.simvla.paired_error_analysis import paired_metrics, aggregate, query_record


@pytest.mark.parametrize('row', bridge.ROWS)
@pytest.mark.parametrize('actions', [0, 1, 5, 6, 41, 900])
def test_bridge_counters_and_original_noise_key(row, actions):
    queries = (actions + 4) // 5
    counts = {**bridge.expected_counts(row, queries), 'num_policy_queries': queries}
    policy = SimpleNamespace(metrics=SimpleNamespace(counters=counts), step_index=actions, flow_steps=10)
    bridge.check_policy(policy, row)
    policy.flow_steps = 3
    with pytest.raises(RuntimeError, match='noise'):
        bridge.check_policy(policy, row)


def test_actual_coarse_decoder_uses_three_steps_and_identical_noise(monkeypatch):
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_eval import SynchronizedNaiveNFE3Policy
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import RealSimVLADCLDPolicy
    calls = []

    def sample(batch, *, generator, dtype, device, deterministic):
        return torch.randn(batch, 10, 7, generator=generator, dtype=dtype, device=device)

    def decode(condition, proprio, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(action=kwargs['initial_noise'], debug={'iterations': kwargs['steps']})

    class Stub:
        _action_noise_seed = RealSimVLADCLDPolicy._action_noise_seed
        _paired_initial_noise = RealSimVLADCLDPolicy._paired_initial_noise
        _sync = staticmethod(lambda: None)
        action_noise_seed_base = 77
        suite = 'libero_10'
        task_id = 3
        trial_id = 4
        flow_steps = 10
        paired_action_noise = True
        metrics = SimpleNamespace(counters=defaultdict(int), latencies=defaultdict(list))
        action_adapter = SimpleNamespace(num_actions=10, dim_action=7,
            sample_initial_action_noise=sample, decode_action_from_condition=decode)

    stub = Stub()
    monkeypatch.setattr(bridge, 'attach_policy', lambda *args: stub)
    policy = bridge.make_policy(SimpleNamespace(loop=None), {}, 'bridge_f4_naive3', {})
    c, p = torch.zeros(1, 2, 8), torch.zeros(1, 8)
    expected, seed = policy._paired_initial_noise(c, p, 7)
    result, actual_seed = policy._decode(c, p, policy_query_index=7)
    assert torch.equal(result, expected)
    assert actual_seed == seed == SynchronizedNaiveNFE3Policy._action_noise_seed(
        SimpleNamespace(**{k: getattr(stub, k) for k in ('action_noise_seed_base','suite','task_id','trial_id','action_adapter')},
            NOISE_KEY_FLOW_STEPS=10), 7)
    assert policy.refresh_every == 4 and policy.flow_steps == 10
    assert calls[0]['steps'] == 3
    assert policy.metrics.counters['num_action_transformer_calls'] == 3
    with pytest.raises(RuntimeError, match='Generation Loop'):
        bridge.make_policy(SimpleNamespace(loop=object()), {}, 'bridge_f2_naive3', {})


def test_paired_vector_errors_and_non_additive_norms():
    zero = torch.zeros(1, 10, 7)
    r = paired_metrics(zero, -torch.ones_like(zero), torch.ones_like(zero), zero)
    assert r['condition_l1'] == r['solver_l1'] == 1
    assert r['interaction_l1'] == r['combined_l1'] == 0
    assert r['interaction_identity_max_abs'] == 0
    r = paired_metrics(zero, zero, zero, torch.ones_like(zero))
    assert r['interaction_l1'] == r['combined_l1'] == 1
    assert r['combined_gripper_sign_mismatches'] == 5
    with pytest.raises(RuntimeError, match='Nonfinite'):
        paired_metrics(zero, zero, zero, zero + float('nan'))
    with pytest.raises(ValueError, match='matched'):
        paired_metrics(zero, zero[:, :5], zero, zero)


def test_analysis_groups_preserve_source_interval_age():
    rows = [dict(source='heldout', interval=8, age=i, window=i, condition_l1=float(i)) for i in (1, 2)]
    result = aggregate(rows)
    assert result['heldout/k8']['metrics']['condition_l1']['mean'] == 1.5
    assert result['heldout/k8/age2']['queries'] == 1


def test_real_record_assembly_and_serialization_keep_latent_and_action_errors_separate(tmp_path):
    zero = torch.zeros(1,10,7)
    record = query_record(dict(source='heldout', interval=4, age=1, window=0),
        (zero, zero, zero+2, zero+3), dict(a00_decode_ms=1.0),
        condition_ms=2.0, teacher_diff=0.0, latent_mse=torch.tensor(.25))
    assert record['condition_mse'] == 4.0
    assert record['latent_condition_normalized_mse'] == .25
    queue.write_json(tmp_path/'query_metrics.json', [record])
    summary = aggregate(queue.read_json(tmp_path/'query_metrics.json'))
    assert summary['all']['metrics']['condition_mse']['mean'] == 4.0
    assert summary['all']['metrics']['latent_condition_normalized_mse']['mean'] == .25


def test_predecessor_and_gpu_lease_exclusion(tmp_path):
    path = tmp_path/'pipeline.lock'
    lock = queue.acquire_lock(path)
    assert queue.acquire_lock(path) is None
    assert queue.predecessor_pending(tmp_path)
    queue.write_json(tmp_path/'status.json', dict(total_jobs=3, completed=['a'], failed=[], active={'4':'b'}))
    assert queue.predecessor_pending(tmp_path)
    queue.write_json(tmp_path/'status.json', dict(total_jobs=3, completed=['a'], failed=[], active={'4':'b','5':'c'}))
    assert not queue.predecessor_pending(tmp_path)
    lock.close()
    assert not queue.predecessor_pending(tmp_path)
    with pytest.raises(FileNotFoundError):
        queue.predecessor_pending(tmp_path/'missing')


@pytest.mark.parametrize('pids,usage,expected', [('123\n', '0,0', False), ('', '511,4', True),
    ('', '512,0', False), ('', '0,5', False), ('', 'N/A,0', False)])
def test_idle_requires_no_process_low_memory_and_low_utilization(monkeypatch, pids, usage, expected):
    def check(cmd, **kwargs):
        assert cmd[1:3] == ['-i','4']
        return pids if '--query-compute-apps=pid' in cmd else usage
    monkeypatch.setattr(queue.subprocess, 'check_output', check)
    assert queue.gpu_idle(4) is expected


def test_queue_retry_recovery_skip_completed_and_continue_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(queue.socket, 'gethostname', lambda: 'jbrserver1')
    monkeypatch.setattr(queue.Path, 'home', lambda: tmp_path/'home')
    clock = [0.0]
    monkeypatch.setattr(queue.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(queue.time, 'sleep', lambda n: clock.__setitem__(0, clock[0]+n))
    monkeypatch.setattr(queue, 'gpu_idle', lambda g: True)
    monkeypatch.setattr(queue, 'predecessor_pending', lambda _: clock[0] < 10)
    jobs = [dict(id=k, cmd=[k], summary=str(tmp_path/(k+'.json')), completion={'verdict':'OK'})
        for k in ('existing', 'bad', 'recover', 'good')]
    queue.write_json(jobs[0]['summary'], {'verdict':'OK'})
    starts = []

    class Process:
        pid = 123
        returncode = 1
        def __init__(self, cmd, **kwargs):
            starts.append(cmd[0])
            assert clock[0] >= 15 and kwargs['env']['CUDA_VISIBLE_DEVICES'] == '4'
            assert kwargs['pass_fds'] and kwargs['start_new_session']
            if cmd[0] in ('recover', 'good'):
                queue.write_json(tmp_path/(cmd[0]+'.json'), {'verdict':'OK'})
        def poll(self):
            return self.returncode

    monkeypatch.setattr(queue.subprocess, 'Popen', Process)
    rc = queue.run_queue(tmp_path/'out', jobs, gpus=(4,), predecessor=tmp_path,
        environment=lambda g: {'CUDA_VISIBLE_DEVICES':str(g)}, cwd=tmp_path)
    assert rc == 1 and starts == ['bad', 'bad', 'recover', 'good']
    status = queue.read_json(tmp_path/'out/queue_status.json')
    assert status['completed'] == ['existing','good','recover']
    assert list(status['failed']) == ['bad']
    assert status['attempts']['bad'] == 2
    with pytest.raises(ValueError, match='Unauthorized'):
        queue.run_queue(tmp_path/'other', jobs, gpus=(2,3), predecessor=tmp_path,
            environment=lambda _: {}, cwd=tmp_path)
