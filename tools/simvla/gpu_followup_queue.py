"""Cooperative per-GPU follow-up queue; never preempt another process."""
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from tools.simvla.compile_benchmark import read_json, write_json


def acquire_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def predecessor_pending(directory):
    if isinstance(directory, (list, tuple)):
        return any(predecessor_pending(item) for item in directory)
    if isinstance(directory, dict):
        path = Path(directory['path']) / directory.get('lock', 'queue.lock')
        if not path.exists():
            raise FileNotFoundError('Missing predecessor lock: ' + str(path))
        lock = acquire_lock(path)
        if lock is None:
            return True
        lock.close()
        return False
    directory = Path(directory)
    path = directory / 'pipeline.lock'
    if not path.exists():
        raise FileNotFoundError('Missing predecessor lock: ' + str(path))
    lock = acquire_lock(path)
    if lock is not None:
        lock.close()
        return False
    state = directory / 'status.json'
    if state.exists():
        d = read_json(state)
        # A parallel predecessor may still be finishing its last assigned jobs.
        # Its now-idle devices are usable without competing for pending jobs.
        if all(k in d for k in ('total_jobs', 'completed', 'failed', 'active')):
            assigned = len(d['completed']) + len(d['failed']) + len(d['active'])
            return assigned < d['total_jobs']
    return True


def gpu_idle(gpu):
    try:
        pids = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
        if pids.strip():
            return False
        value = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
            '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        memory, utilization = map(int, value.strip().split(','))
        return memory < 512 and utilization < 5
    except (subprocess.SubprocessError, ValueError):
        return False


def completed(job):
    p = Path(job['summary'])
    if not p.exists():
        return False
    d = read_json(p)
    return all(d.get(k) == v for k, v in job['completion'].items())


def ready(job, done):
    if not set(job.get('deps', ())).issubset(done):
        return False
    marker = job.get('ready_file')
    if marker:
        if not Path(marker).is_file():
            return False
        d = read_json(marker)
        return all(d.get(k) == v for k, v in job.get('ready_fields', {}).items())
    return True


def upstream_finished_without_artifact(job):
    marker = job.get('upstream_status_file')
    if not marker or not Path(marker).is_file():
        return False
    state = read_json(marker)
    return state.get('phase') in ('complete', 'finished_with_failures') and not ready(job, set(job.get('deps', ())))


def run_queue(output, jobs, *, gpus, predecessor, environment, cwd, timeout=28800):
    output = Path(output)
    host = socket.gethostname()
    allowed = (4, 5, 6, 7) if host == 'jbrserver1' else (0,) if host == 'jbr-TRX50' else ()
    if not gpus or any(g not in allowed for g in gpus):
        raise ValueError(f'Unauthorized GPU pool: {host} {gpus}')
    ids = {j['id'] for j in jobs}
    if len(ids) != len(jobs) or any(not set(j.get('deps', ())).issubset(ids) for j in jobs):
        raise ValueError('Duplicate job IDs or unknown dependencies')
    owner = acquire_lock(output / 'queue.lock')
    if owner is None:
        raise RuntimeError('This queue already has an owner')
    lease_root = Path.home() / '.cache/gnaroshi_vla/gpu_leases' / host
    write_json(output / 'queue_plan.json', dict(host=host, gpus=gpus, jobs=jobs,
        predecessor=str(predecessor), gpu_lease_root=str(lease_root),
        policy='Wait for predecessor pending jobs, then acquire an idle GPU and shared lease; retry technical failures once; no SR gate.'))
    status_path = output / 'queue_status.json'
    old = read_json(status_path) if status_path.exists() else {}
    attempts = old.get('attempts', {})
    done = {j['id'] for j in jobs if completed(j)}
    failed = {k: v for k, v in old.get('failed', {}).items() if k not in done}
    active, idle_since = {}, {}
    last_report = 0
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    def state(phase):
        return dict(phase=phase, completed=sorted(done), failed=failed, attempts=attempts,
            pending=[j['id'] for j in jobs if j['id'] not in done | set(failed)
                     and j['id'] not in {v[0]['id'] for v in active.values()}],
            active={str(g): dict(job=v[0]['id'], pid=v[1].pid,
                                elapsed_seconds=time.monotonic()-v[4]) for g, v in active.items()},
            total_jobs=len(jobs), updated_unix=time.time())
    try:
        while len(done) + len(failed) < len(jobs):
            for gpu, (job, proc, log, lease, began) in list(active.items()):
                if proc.poll() is None and time.monotonic()-began > timeout:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                if proc.poll() is None:
                    continue
                log.close(); lease.close(); del active[gpu]
                idle_since.pop(gpu, None)
                # A summary can be complete even if logging failed at shutdown.
                if completed(job):
                    done.add(job['id'])
                    print('DONE ' + job['id'], flush=True)
                elif attempts[job['id']] >= 2:
                    failed[job['id']] = dict(returncode=proc.returncode, log=str(log.name))
                    print('FAILED ' + job['id'] + ' log=' + str(log.name), flush=True)
                else:
                    print('RETRY ' + job['id'] + ' log=' + str(log.name), flush=True)
            waiting = predecessor_pending(predecessor)
            for job in jobs:
                if job['id'] not in done | set(failed) and any(d in failed for d in job.get('deps', ())):
                    failed[job['id']] = dict(reason='dependency_failed', dependencies=job['deps'])
                elif job['id'] not in done | set(failed) and upstream_finished_without_artifact(job):
                    failed[job['id']] = dict(reason='upstream_finished_without_required_artifact',
                        status=job['upstream_status_file'])
            active_ids = {v[0]['id'] for v in active.values()}
            if not waiting:
                for gpu in gpus:
                    if gpu in active or not gpu_idle(gpu):
                        idle_since.pop(gpu, None)
                        continue
                    idle_since.setdefault(gpu, time.monotonic())
                    if time.monotonic()-idle_since[gpu] < 5:
                        continue
                    job = next((j for j in jobs if j['id'] not in done | set(failed) | active_ids
                                and ready(j, done)), None)
                    if job is None:
                        break
                    lease = acquire_lock(lease_root / f'gpu{gpu}.lock')
                    if lease is None:
                        continue
                    if not gpu_idle(gpu):
                        lease.close()
                        continue
                    log_path = output / 'queue_logs' / (job['id'] + '.log')
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log = log_path.open('a', buffering=1)
                    attempts[job['id']] = attempts.get(job['id'], 0) + 1
                    env = environment(gpu)
                    env['GNAROSHI_GPU_LEASE_FD'] = str(lease.fileno())
                    try:
                        proc = subprocess.Popen(job['cmd'], cwd=cwd, env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            pass_fds=(lease.fileno(),))
                    except Exception:
                        log.close(); lease.close()
                        raise
                    active[gpu] = (job, proc, log, lease, time.monotonic())
                    active_ids.add(job['id'])
                    print(f'START gpu={gpu} job={job["id"]} pid={proc.pid}', flush=True)
            awaiting_artifact = not any(ready(j, done) for j in jobs if j['id'] not in done | set(failed))
            phase = ('waiting_for_predecessor' if waiting else 'running' if active
                     else 'waiting_for_artifact' if awaiting_artifact else 'waiting_for_idle_gpu')
            write_json(status_path, state(phase))
            if time.monotonic()-last_report > 60:
                print(f'QUEUE {phase}; completed={len(done)}/{len(jobs)}; active={active_ids}; failed={list(failed)}', flush=True)
                last_report = time.monotonic()
            time.sleep(5)
        write_json(status_path, state('complete' if not failed else 'finished_with_failures'))
        return int(bool(failed))
    finally:
        for _, proc, _, _, _ in active.values():
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for _, proc, log, lease, _ in active.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            log.close(); lease.close()
        if active:
            write_json(status_path, state('interrupted'))
        owner.close()
        signal.signal(signal.SIGTERM, previous_handler)
