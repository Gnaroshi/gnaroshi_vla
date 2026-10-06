import fcntl
import json
from tools.simvla.gpu_followup_queue import predecessor_pending, ready, completed, upstream_finished_without_artifact
from architectures.simvla.adapters.refresh_calibration.train import select_queries, language_key


def test_predecessor_waits_for_actual_queue_owner(tmp_path):
    path = tmp_path / 'queue.lock'
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert predecessor_pending([dict(path=str(tmp_path), lock='queue.lock')])
    assert not predecessor_pending(dict(path=str(tmp_path), lock='queue.lock'))


def test_dependencies_and_transferred_marker(tmp_path):
    marker = tmp_path / 'READY.json'
    job = dict(deps=['bootstrap'], ready_file=str(marker), ready_fields=dict(verdict='TRANSFER_READY'))
    assert not ready(job, {'bootstrap'})
    marker.write_text(json.dumps(dict(verdict='TRANSFER_READY')))
    assert not ready(job, set())
    assert ready(job, {'bootstrap'})


def test_completed_does_not_accept_wrong_contract(tmp_path):
    summary = tmp_path / 'result.json'
    summary.write_text(json.dumps(dict(identity='old', episodes=500)))
    assert not completed(dict(summary=str(summary), completion=dict(identity='new', episodes=500)))


def test_matched_sampling_and_instruction_keys():
    assert select_queries(7, 10, 100, 8) == select_queries(7, 10, 100, 8)
    for k in (4, 8):
        assert all(1 <= age < k for step in range(10) for _, age in select_queries(7, step, 100, k))
    assert language_key('  Put_the cup  ') == language_key('put the cup')


def test_upstream_failure_does_not_wait_forever(tmp_path):
    status = tmp_path / 'status.json'
    marker = tmp_path / 'READY.json'
    job = dict(upstream_status_file=str(status), ready_file=str(marker))
    status.write_text(json.dumps(dict(phase='running')))
    assert not upstream_finished_without_artifact(job)
    status.write_text(json.dumps(dict(phase='finished_with_failures')))
    assert upstream_finished_without_artifact(job)
    marker.write_text('{}')
    assert not upstream_finished_without_artifact(job)
