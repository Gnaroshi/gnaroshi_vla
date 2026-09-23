"""One-command orchestration with visible progress and episode-boundary resume."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from .config import ASSETS, DEFAULT_RESULT, ROWS, SCHEMA, assigned_ids, evaluator_argv

REPO = Path(__file__).resolve().parents[4]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def sources():
    roots = [REPO / 'architectures/seer/upstream', REPO / 'architectures/seer/adapters/latentloop_freshness']
    paths = {p for root in roots for p in root.rglob('*.py')}
    # Imported helpers can affect the wrapper even when their experimental modes are off.
    for name in ('latentloop_plan_continuation', 'latentloop_horizon_regeneration', 'joint_latent_action_surrogate',
                 'latentloop_comparison', 'latentloop_segment_grid', 'latent_prediction_correction'):
        for prefix in ('methods', 'architectures/seer/adapters'):
            paths.update((REPO / prefix / name).rglob('*.py'))
    paths.add(REPO / 'tools/seer/run_latentloop_freshness.py')
    paths.add(REPO / 'architectures/seer/wrappers/lrnode/run_latentloop_freshness.sh')
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(paths) if p.is_file()}


def configure_environment():
    for name in ('LIBERO_GL_BACKEND', 'MUJOCO_GL', 'PYOPENGL_PLATFORM'):
        os.environ[name] = 'egl'
    for name in ('EVAL_CONTROL_HZ', 'EVAL_BASE_CONTROL_HZ'):
        os.environ[name] = '20'
    os.environ.update(EVAL_BASE_SETTLE_STEPS='5', EVAL_SCALE_SETTLE_STEPS_WITH_HZ='1',
                      EVAL_SCALE_MAX_STEPS_WITH_HZ='1', EVAL_ENV_HORIZON='1000')
    os.environ.update(SAVE_VIDEO='0', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1',
                      PYTHONDONTWRITEBYTECODE='1', WANDB_MODE='disabled')
    os.environ['NUMBA_CACHE_DIR'] = '/tmp/seer_freshness_numba'
    os.environ['MPLCONFIGDIR'] = '/tmp/seer_freshness_matplotlib'
    config_dir = Path('/tmp/seer_freshness_libero')
    config_dir.mkdir(exist_ok=True)
    base = Path('/home/mingyujung/private/LIBERO/libero/libero')
    locations = {'benchmark_root': str(base), 'bddl_files': str(base / 'bddl_files'),
                 'init_states': str(base / 'init_files'), 'assets': str(base / 'assets'),
                 'datasets': str(base.parent / 'datasets')}
    temporary = config_dir / f'config.{os.getpid()}.tmp'
    temporary.write_text(json.dumps(locations))
    os.replace(temporary, config_dir / 'config.yaml')
    os.environ['LIBERO_CONFIG_PATH'] = str(config_dir)
    os.environ['PATH'] = str(Path(sys.executable).parent) + ':' + os.environ.get('PATH', '')


def preflight(config):
    for name, (path, expected) in ASSETS.items():
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise FileNotFoundError(f'Missing {name}: {path}')
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f'{name} SHA256 differs: {actual}')
        print(f'[ASSET PASS] {name}: {path}', flush=True)
    libero = Path(config['libero'])
    if not (libero / 'libero/libero/bddl_files').is_dir():
        raise FileNotFoundError(f'LIBERO BDDL directory missing: {libero}')
    sys.path[:0] = [str(REPO / 'architectures/seer/upstream'), str(libero)]
    import eval_libero
    from utils.eval_utils_libero import benchmark
    previous_argv = sys.argv
    sys.argv = ['eval_libero.py'] + evaluator_argv(config, '/tmp/seer_freshness_argument_check')
    try:
        # Upstream get_parser() eagerly parses sys.argv before returning the parser.
        parsed = eval_libero.get_parser(is_eval=True).parse_args()
    finally:
        sys.argv = previous_argv
    assert parsed.action_pred_steps == 3 and parsed.sequence_length == 7
    suite = benchmark.get_benchmark_dict()['libero_10']()
    import torch
    for i in range(config['tasks']):
        task = suite.get_task(i)
        if not (libero / 'libero/libero/bddl_files' / task.problem_folder / task.bddl_file).is_file():
            raise FileNotFoundError(f'Missing task BDDL: {task.name}')
        path = libero / 'libero/libero/init_files' / task.problem_folder / task.init_states_file
        states = torch.load(path, map_location='cpu')
        if len(states) < max(config['episodes_per_task'], config['diagnostic_trials']):
            raise ValueError(f'Insufficient initial states for {task.name}: {len(states)}')
    if config['event_step'] % 4 != 1 or not 0 < config['event_step'] < config['max_steps']:
        raise ValueError('External intervention must occur on the first skipped query before the horizon')
    print('[PREFLIGHT PASS] imports, parser, checkpoints, BDDL, initial states; no training dataset required', flush=True)


def busy_gpus(devices):
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    uuids = {line.split(',')[1].strip(): line.split(',')[0].strip() for line in inventory.splitlines()}
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                        '--format=csv,noheader,nounits'], text=True)
    return [line.strip() for line in processes.splitlines()
            if uuids.get(line.split(',')[0].strip()) in devices
            and line.split(',')[1].strip() != str(os.getpid())]


def wait_for_gpus(devices):
    clean = 0
    while clean < 2:
        busy = busy_gpus(devices)
        if busy:
            clean = 0
            print('[WAIT GPU] ' + '; '.join(busy), flush=True)
            time.sleep(30)
        else:
            clean += 1
            if clean < 2:
                time.sleep(5)
    print('[GPU READY] ' + ','.join(devices), flush=True)


def verify_stage(root, stage, config, world):
    directory = root / stage
    if stage == 'smoke':
        for rank in range(world):
            path = directory / f'rank{rank}.json'
            if not path.is_file():
                return False
            result = json.loads(path.read_text())
            if (result['normal_action_max_abs_error'] != 0 or result['external_branches'] != 10
                    or result['same_input_records'] <= 0 or result.get('episode_save_resume_rows') != 11):
                raise RuntimeError(f'Incomplete smoke evidence: {path}')
        return True
    count = config['tasks'] * (config['diagnostic_trials'] if stage in ('same_input', 'external') else config['episodes_per_task'])
    for rank in range(world):
        marker = directory / f'rank{rank}_complete.json'
        if not marker.is_file():
            return False
        if json.loads(marker.read_text()) != {'stage': stage, 'ids': assigned_ids(count, world, rank)}:
            raise RuntimeError(f'Invalid completion marker: {marker}')
    for i in range(count):
        path = directory / f'episode_{i:04d}.json'
        if not path.is_file():
            return False
        value = json.loads(path.read_text())
        tensor = path.with_suffix('.pt')
        if value['eval_id'] != i or not tensor.is_file() or tensor.stat().st_size != value['artifact_bytes']:
            raise RuntimeError(f'Invalid saved episode: {path}')
    return True


def stage_progress(root, stage, config, started):
    directory = root / stage
    if stage == 'smoke':
        completed = len(list(directory.glob('rank[0-9].json')))
        print(f'[PROGRESS] smoke: {completed}/{config["world_size"]} ranks; '
              f'elapsed={(time.monotonic()-started)/60:.1f} min; details={root / "logs/smoke.log"}', flush=True)
        return
    completed = [json.loads(p.read_text()) for p in directory.glob('episode_*.json')]
    count = config['tasks'] * (config['diagnostic_trials'] if stage in ('same_input', 'external') else config['episodes_per_task'])
    sr = (100 * sum(x.get('success', False) for x in completed) / len(completed)) if completed and stage != 'external' else None
    status = f'[PROGRESS] {stage}: {len(completed)}/{count}; elapsed={(time.monotonic()-started)/60:.1f} min'
    if sr is not None:
        status += f'; SR={sr:.2f}%'
    if completed and stage not in ('external', 'smoke'):
        mean_seconds = sum(x.get('seconds', 0) for x in completed) / len(completed)
        status += f'; stage ETA~{mean_seconds*(count-len(completed))/config["world_size"]/3600:.2f} h (measured episode mean)'
    print(status, flush=True)
    for path in sorted(directory.glob('progress_rank*.json')):
        value = json.loads(path.read_text())
        print(f'  {path.stem}: task={value.get("task")} trial={value.get("trial")} '
              f'step={value.get("step")} {value.get("scene", "")} {value.get("variant", "")}', flush=True)


def run_stage(root, stage, config, port):
    world = config['world_size']
    if verify_stage(root, stage, config, world):
        print(f'[SKIP VERIFIED] {stage}', flush=True)
        return
    wait_for_gpus(config['physical_gpus'])
    requested_port = port
    for candidate in range(requested_port, requested_port + 100):
        try:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', candidate))
            port = candidate
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f'No free master port near {requested_port}')
    if port != requested_port:
        print(f'[PORT] {requested_port} occupied; using {port}', flush=True)
    (root / 'logs').mkdir(exist_ok=True)
    command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node', str(world),
               '--master_addr', '127.0.0.1', '--master_port', str(port),
               str(REPO / 'tools/seer/run_latentloop_freshness.py'), 'worker',
               '--config', str(root / 'config.json'), '--stage', stage]
    (root / 'logs' / f'{stage}_command.json').write_text(json.dumps(command, indent=2) + '\n')
    env = dict(os.environ, LOG_DIR=str(root / stage),
               PYTHONPATH=os.pathsep.join([str(REPO / 'architectures/seer/upstream'), str(REPO), config['libero']]))
    started = time.monotonic()
    with (root / 'logs' / f'{stage}.log').open('a', buffering=1) as log:
        child = subprocess.Popen(command, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                time.sleep(5)
                if time.monotonic() - started > 5 and int(time.monotonic() - started) % 30 < 5:
                    stage_progress(root, stage, config, started)
            if child.returncode or not verify_stage(root, stage, config, world):
                print((root / 'logs' / f'{stage}.log').read_text()[-10000:], flush=True)
                raise RuntimeError(f'{stage} failed (exit={child.returncode}). Saved episodes are preserved; rerun the same command.')
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait()
            raise
    print(f'[STAGE COMPLETE] {stage}; {(time.monotonic()-started)/60:.1f} min', flush=True)


def worker(options):
    config = json.loads(Path(options.config).read_text())
    locked_source = json.loads((Path(options.config).parent / 'source_hashes.json').read_text())
    if sources() != locked_source:
        raise RuntimeError('Seer source changed after campaign preflight; refusing a mixed-source evaluation')
    sys.path[:0] = [str(REPO / 'architectures/seer/upstream'), config['libero']]
    import eval_libero
    from .runtime import evaluate_loaded
    root = Path(config['result_root'])
    directory = root / options.stage
    sys.argv = ['eval_libero.py'] + evaluator_argv(config, directory)
    eval_libero.eval_one_epoch_libero_ddp = lambda args, model, image_processor, tokenizer: evaluate_loaded(
        args, model, image_processor, tokenizer, config, options.stage, directory)
    try:
        eval_libero.main()
    finally:
        import torch
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('run', 'preflight', 'worker', 'smoke'))
    parser.add_argument('--result-root', default=os.environ.get('RESULT_ROOT', DEFAULT_RESULT))
    parser.add_argument('--config')
    parser.add_argument('--stage')
    parser.add_argument('--master-port', type=int, default=int(os.environ.get('MASTER_PORT_BASE', '18100')))
    options = parser.parse_args()
    configure_environment()
    os.environ['LIBERO_GL_REQUIRE_ACTUAL'] = '1'
    if options.mode == 'worker':
        return worker(options)
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '4,5,6,7').split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
    if socket.gethostname() != 'jbrserver1':
        raise RuntimeError('This launcher is configured only for sd1/jbrserver1')
    if options.mode == 'smoke':
        if not devices or any(x not in ('4', '5', '6', '7') for x in devices):
            raise ValueError('Smoke is restricted to physical GPUs 4-7')
    elif devices != ['4', '5', '6', '7']:
        raise ValueError('The campaign requires CUDA_VISIBLE_DEVICES=4,5,6,7')
    root = Path(options.result_root).resolve()
    expected_root = Path(DEFAULT_RESULT).parent
    if not root.is_relative_to(expected_root):
        raise ValueError(f'Large results must be stored under {expected_root}')
    config = {'schema': SCHEMA, 'seed': 42, 'tasks': 10, 'episodes_per_task': 50,
              'diagnostic_trials': 2, 'max_steps': 600, 'event_step': 41, 'displacement_m': .02,
              'threads': 4, 'libero': '/home/mingyujung/private/LIBERO',
              'assets': {k: v[0] for k, v in ASSETS.items()},
              'world_size': len(devices), 'physical_gpus': devices, 'result_root': str(root)}
    preflight(config)
    if options.mode == 'preflight':
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path = root / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing campaign configuration differs; use a new semantic RESULT_ROOT')
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    source = sources()
    source_path = root / 'source_hashes.json'
    if source_path.exists() and json.loads(source_path.read_text()) != source:
        raise ValueError('Source changed since this campaign began; inspect changes before resuming')
    source_path.write_text(json.dumps(source, indent=2) + '\n')
    for command, name in [(['git', 'rev-parse', 'HEAD'], 'commit.txt'),
                          (['git', 'status', '--short'], 'git_status.txt'),
                          (['git', 'diff', '--', 'architectures/seer', 'methods'], 'working_tree.patch')]:
        if not (root / name).exists():
            (root / name).write_text(subprocess.check_output(command, cwd=REPO, text=True))
    print(f'[CAMPAIGN] results={root}', flush=True)
    print('[PLAN] smoke -> 20 same-input episodes -> 10 x 500 rollout episodes -> 20 paired external snapshots (up to 200 branches) -> report', flush=True)
    wait_for_gpus(devices)
    stages = ('smoke',) if options.mode == 'smoke' else ('smoke', 'same_input') + ROWS + ('external',)
    for index, stage in enumerate(stages):
        print(f'[{index+1}/{len(stages)}] {stage}', flush=True)
        run_stage(root, stage, config, options.master_port + index)
    if options.mode != 'smoke':
        sys.path[:0] = [str(REPO / 'architectures/seer/upstream'), config['libero']]
        from .analysis import summarize
        summarize(root)
        print(f'[DONE] {REPO / "codex_outputs/seer/freshness" / root.name / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
