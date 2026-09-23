"""Three new controls; preserve and reuse the completed public33 reference campaign."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from ..latentloop_freshness import campaign as base
from ..latentloop_freshness.config import ASSETS, DEFAULT_RESULT, assigned_ids, evaluator_argv
from .control import ROWS, calibration_means

REPO = base.REPO
DEFAULT_ROOT = str(Path(DEFAULT_RESULT).with_name('public33_seed42_controls_r1'))
ENTRY = REPO / 'tools/seer/run_latentloop_freshness_controls.py'


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sources():
    result = base.sources()
    paths = list((REPO / 'architectures/seer/adapters/latentloop_freshness_controls').glob('*.py'))
    paths += [ENTRY, REPO / 'architectures/seer/wrappers/lrnode/run_latentloop_freshness_controls.sh']
    for path in paths:
        result[str(path.relative_to(REPO))] = base.digest(path)
    return result


def verify_reference():
    root = Path(DEFAULT_RESULT)
    locked = json.loads((root / 'source_hashes.json').read_text())
    for name, expected in locked.items():
        if base.digest(REPO / name) != expected:
            raise RuntimeError(f'Reference source changed: {name}; do not silently mix results')
    config = json.loads((root / 'config.json').read_text())
    for row in ('full', 'normal', 'zero', 'gate_observed_residual_zero', 'repeated_observation'):
        if not base.verify_stage(root, row, config, 4):
            raise RuntimeError(f'Reference row incomplete: {row}')
    expected = json.loads((root / 'normal/environment_rank0.json').read_text())['packages']
    for package, version in expected.items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'Reference environment differs: {package}')
    print('[REFERENCE PASS] retained 500-episode rows, 119 sources and environment; no reruns', flush=True)


def calibration_cases(config):
    """Reserve original initial states not present in the 50-state evaluation set."""
    import torch
    from utils.eval_utils_libero import benchmark
    suite = benchmark.get_benchmark_dict()['libero_10']()
    cases, hashes = [], {}
    def state_hash(value):
        return hashlib.sha256(torch.as_tensor(value).double().contiguous().numpy().tobytes()).hexdigest()
    for task_id in range(10):
        task = suite.get_task(task_id)
        parent = Path(config['libero']) / 'libero/libero/init_files' / task.problem_folder
        evaluated = parent / task.init_states_file
        reserve = parent / f'{task.name}.init'
        test_states = torch.load(evaluated, map_location='cpu')
        other_states = torch.load(reserve, map_location='cpu')
        forbidden = {state_hash(value) for value in test_states[:50]}
        selected = []
        for index, value in enumerate(other_states):
            digest = state_hash(value)
            if digest not in forbidden:
                selected.append({'task_id': task_id, 'trial': index, 'init_states_file': reserve.name, 'state_sha256': digest})
                forbidden.add(digest)
            if len(selected) == 2:
                break
        if len(selected) != 2:
            raise RuntimeError(f'Cannot form independent calibration for {task.name}; no test-set fallback')
        cases.extend(selected)
        hashes[str(evaluated)] = base.digest(evaluated)
        hashes[str(reserve)] = base.digest(reserve)
    print('[CALIBRATION SPLIT] 20 reserved original initial states; zero overlap with 500 evaluation initial states', flush=True)
    return cases, hashes


def verify_stage(root, stage, config):
    if stage == 'smoke':
        for rank in range(4):
            path = root / stage / f'rank{rank}.json'
            if not path.exists():
                return False
            row = json.loads(path.read_text())
            if row['normal_action_max_abs_error'] != 0 or row['formula_checks'] != 18:
                raise RuntimeError('Smoke semantic validation failed')
        return True
    adjusted = dict(config, episodes_per_task=2 if stage == 'calibration' else 50)
    return base.verify_stage(root, stage, adjusted, 4)


def finalize_calibration(root, config):
    rows = [json.loads((root / 'calibration' / f'episode_{i:04d}.json').read_text()) for i in range(20)]
    value = {'gates': calibration_means(rows), 'method': 'episode-balanced age-specific mean; no SR tuning',
             'calibration_seed': 4242, 'initial_state_policy': 'first two distinct original states absent from evaluation set, per task',
             'cases': config['calibration_cases'], 'episodes': 20}
    path = root / 'calibration.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError('Frozen calibration changed')
    atomic_json(path, value)
    print(f'[CALIBRATION FROZEN] {value["gates"]}', flush=True)


def run_stage(root, stage, config, requested_port):
    if verify_stage(root, stage, config):
        print(f'[SKIP VERIFIED] {stage}', flush=True)
        return
    base.wait_for_gpus(config['physical_gpus'])
    for port in range(requested_port, requested_port + 100):
        try:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', port))
            break
        except OSError:
            continue
    else:
        raise RuntimeError('No available master port')
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node', '4',
        '--master_addr', '127.0.0.1', '--master_port', str(port), str(ENTRY), 'worker',
        '--config', str(root / 'config.json'), '--stage', stage]
    log_path = root / 'logs' / f'{stage}.log'
    log_path.parent.mkdir(exist_ok=True)
    atomic_json(root / 'logs' / f'{stage}_command.json', command)
    env = dict(os.environ, LOG_DIR=str(root / stage),
        PYTHONPATH=os.pathsep.join([str(REPO / 'architectures/seer/upstream'), str(REPO), config['libero']]))
    started = time.monotonic()
    adjusted = dict(config, episodes_per_task=2 if stage == 'calibration' else 50)
    with log_path.open('a', buffering=1) as log:
        child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        last = 0.
        try:
            while child.poll() is None:
                time.sleep(5)
                if time.monotonic() - last >= 30:
                    base.stage_progress(root, stage, adjusted, started)
                    last = time.monotonic()
            if child.returncode or not verify_stage(root, stage, config):
                print(log_path.read_text()[-12000:], flush=True)
                raise RuntimeError(f'{stage} failed: exit={child.returncode}. Completed episodes retained for resume.')
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait()
            raise
    print(f'[STAGE COMPLETE] {stage}: {(time.monotonic()-started)/60:.1f} min', flush=True)


def summarize(root):
    reference = Path(DEFAULT_RESULT)
    lines = ['# Seer feature/gate controls', '', 'One eval seed (42), public33 + adapter39, K4, EGL, 500 episodes per row.',
        'No training. Reference rows are reused only after source/environment checks.',
        'Calibration uses 20 reserved unpruned initial states not in the 500 evaluation starts, seed4242.',
        'This tests gate calibration; it is not a new latency benchmark or proof that observations are unnecessary.', '',
        '| Row | Source | SR (%) | Gain vs full (pp) | Both success | Mean steps saved |',
        '|---|---|---:|---:|---:|---:|']
    def load(folder):
        return {i: json.loads((folder / f'episode_{i:04d}.json').read_text()) for i in range(500)}
    baseline = load(reference / 'full')
    baseline_sr = sum(v['success'] for v in baseline.values()) / 5
    results = []
    for row in ('full', 'normal', 'zero', 'gate_observed_residual_zero', 'repeated_observation') + ROWS:
        origin = root if row in ROWS else reference
        values = load(origin / row)
        sr = sum(v['success'] for v in values.values()) / 5
        both = [i for i in values if values[i]['success'] and baseline[i]['success']]
        saved = sum(baseline[i]['env_steps']-values[i]['env_steps'] for i in both)/len(both) if both else None
        results.append(dict(row=row, source=str(origin / row), episodes=500, sr_percent=sr,
            gain_pp=sr-baseline_sr, both_success=len(both), mean_steps_saved=saved))
        saved_text = f'{saved:.2f}' if saved is not None else 'NA'
        lines.append(f'| {row} | {"new" if row in ROWS else "reference"} | {sr:.1f} | {sr-baseline_sr:+.1f} | {len(both)} | {saved_text} |')
    report = REPO / 'codex_outputs/seer/freshness' / root.name
    report.mkdir(parents=True, exist_ok=True)
    atomic_json(report / 'results.json', results)
    atomic_json(report / 'calibration.json', json.loads((root / 'calibration.json').read_text()))
    (report / 'report.md').write_text('\n'.join(lines) + '\n')
    print(f'[DONE] {report / "report.md"}', flush=True)


def worker(options):
    root = Path(options.config).parent
    config = json.loads(Path(options.config).read_text())
    if sources() != json.loads((root / 'source_hashes.json').read_text()):
        raise RuntimeError('Source changed since controls preflight')
    for path, expected in config['initial_state_hashes'].items():
        if base.digest(path) != expected:
            raise RuntimeError(f'Initial states changed: {path}')
    sys.path[:0] = [str(REPO / 'architectures/seer/upstream'), config['libero']]
    import eval_libero
    from .runtime import evaluate_loaded
    local = dict(config, seed=4242) if options.stage == 'calibration' else config
    sys.argv = ['eval_libero.py'] + evaluator_argv(local, root / options.stage)
    eval_libero.eval_one_epoch_libero_ddp = lambda args, model, image_processor, tokenizer: evaluate_loaded(
        args, model, image_processor, tokenizer, config, options.stage, root / options.stage)
    try:
        eval_libero.main()
    finally:
        import torch
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'preflight', 'worker'))
    parser.add_argument('--config')
    parser.add_argument('--stage', choices=('smoke', 'calibration') + ROWS)
    parser.add_argument('--result-root', default=os.environ.get('RESULT_ROOT', DEFAULT_ROOT))
    parser.add_argument('--master-port', type=int, default=int(os.environ.get('MASTER_PORT_BASE', '18200')))
    options = parser.parse_args()
    base.configure_environment()
    os.environ['LIBERO_GL_REQUIRE_ACTUAL'] = '1'
    if options.mode == 'worker':
        return worker(options)
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '4,5,6,7').split(',')
    if socket.gethostname() != 'jbrserver1' or devices != ['4', '5', '6', '7']:
        raise ValueError('This follow-up uses only sd1 GPUs 4,5,6,7')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
    root = Path(options.result_root).resolve()
    if root == Path(DEFAULT_RESULT) or not root.is_relative_to(Path(DEFAULT_RESULT).parent):
        raise ValueError('Use a separate follow-up result folder on shared storage')
    config = {'schema': 1, 'seed': 42, 'tasks': 10, 'episodes_per_task': 50, 'diagnostic_trials': 2,
        'max_steps': 600, 'event_step': 41, 'threads': 4, 'libero': '/home/mingyujung/private/LIBERO',
        'assets': {k:v[0] for k,v in ASSETS.items()}, 'world_size': 4, 'physical_gpus': devices,
        'result_root': str(root), 'reference_root': DEFAULT_RESULT, 'calibration_seed': 4242}
    verify_reference()
    base.preflight(config)
    cases, hashes = calibration_cases(config)
    config.update(calibration_cases=cases, initial_state_hashes=hashes)
    if options.mode == 'preflight':
        print('[CONTROLS PREFLIGHT PASS] CPU/static checks only; GPU smoke runs before long evaluation', flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name, value in (('config.json', config), ('source_hashes.json', sources())):
        path = root / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise RuntimeError(f'Existing controls campaign differs: {name}')
        atomic_json(path, value)
    for name, command in (('commit.txt', ['git','rev-parse','HEAD']), ('git_status.txt',['git','status','--short'])):
        path = root / name
        if not path.exists():
            path.write_text(subprocess.check_output(command, cwd=REPO, text=True))
    print(f'[CAMPAIGN] {root}', flush=True)
    print('[PLAN] GPU/formula smoke -> 20 independent gate calibration episodes -> 3 x 500 evaluation episodes -> comparison report', flush=True)
    for index, stage in enumerate(('smoke', 'calibration') + ROWS):
        print(f'[{index+1}/5] {stage}', flush=True)
        run_stage(root, stage, config, options.master_port+index)
        if stage == 'calibration':
            finalize_calibration(root, config)
    summarize(root)


if __name__ == '__main__':
    main()
