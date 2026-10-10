"""Defer dispatch of an owned queue without interrupting its active workers."""
import argparse
import os
from pathlib import Path
import signal
import time

from tools.simvla.compile_benchmark import read_json, write_json


def process(pid):
    path = Path('/proc') / str(pid)
    try:
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), started=fields[19], state=fields[0],
                    command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except FileNotFoundError:
        return None


def alive(saved):
    current = process(saved['pid'])
    return current is not None and current['started'] == saved['started'] and current['state'] != 'Z'


def prepare(queue, owner, module, output):
    output = Path(output)
    if output.exists():
        raise RuntimeError('Existing handoff record; use the recorded handoff')
    saved = process(owner)
    if saved is None or saved['command'].strip().split()[-2:] != ['-m', module]:
        raise RuntimeError('Owner must be exactly the requested queue module, without worker arguments')
    os.kill(owner, signal.SIGSTOP)
    try:
        for _ in range(100):
            current = process(owner)
            if current and current['state'] in ('T', 't'):
                break
            time.sleep(.02)
        else:
            raise RuntimeError('Dispatch owner did not stop')
        state = read_json(Path(queue) / 'queue_status.json')
        workers = [process(v['pid']) for v in state['active'].values()]
        workers = [w for w in workers if w is not None and w['state'] != 'Z']
        # A child may have launched just before the last status write.
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[1]) == owner:
                    child = process(int(entry.name))
                    if child and child['state'] != 'Z' and child['pid'] not in {w['pid'] for w in workers}:
                        workers.append(child)
            except (FileNotFoundError, ProcessLookupError):
                continue
        record = dict(phase='dispatch_deferred_workers_preserved', owner=saved,
                      workers=workers, queue=str(queue), previous_status=state,
                      reason='User requested bounded-history experiment before old pending sweeps',
                      created_unix=time.time())
        write_json(output, record)
    except BaseException:
        os.kill(owner, signal.SIGCONT)
        raise
    if not workers:
        finish(output)
    print('HANDOFF_PREPARED', output, 'workers=', [w['pid'] for w in workers], flush=True)


def finish(path):
    path = Path(path)
    record = read_json(path)
    if record['phase'] == 'HANDOFF_COMPLETE':
        return
    while any(alive(w) for w in record['workers']):
        print('Waiting for preserved workers', [w['pid'] for w in record['workers'] if alive(w)], flush=True)
        time.sleep(15)
    owner = record['owner']
    if alive(owner):
        # Deliver termination before resuming so no pending job can be dispatched.
        os.kill(owner['pid'], signal.SIGTERM)
        os.kill(owner['pid'], signal.SIGCONT)
        for _ in range(300):
            if not alive(owner):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('Queue owner did not exit; no forced termination was used')
    status_path = Path(record['queue']) / 'queue_status.json'
    state = read_json(status_path)
    state.update(phase='deferred_by_user_priority', priority_handoff=str(path), updated_unix=time.time())
    write_json(status_path, state)
    record.update(phase='HANDOFF_COMPLETE', finished_unix=time.time())
    write_json(path, record)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--record', required=True)
    p.add_argument('--queue')
    p.add_argument('--owner', type=int)
    p.add_argument('--module')
    p.add_argument('--finish', action='store_true')
    a = p.parse_args()
    if a.finish:
        finish(a.record)
    else:
        if not all((a.queue, a.owner, a.module)):
            p.error('prepare requires queue, owner and module')
        prepare(a.queue, a.owner, a.module, a.record)
