"""Confirm core feature/gate controls across three fixed evaluation seeds."""
import argparse
import fcntl
import json
import os
import socket
import statistics
import subprocess
import sys
from pathlib import Path

from ..latentloop_freshness import campaign as base
from ..latentloop_freshness.config import DEFAULT_RESULT, evaluator_argv
from ..latentloop_freshness.interventions import replaced_method
from ..latentloop_freshness_controls import campaign as retained
from .control import NEW_ROW, ROWS, SEEDS, experiment_plan

REPO = base.REPO
REFERENCE = Path(DEFAULT_RESULT)
CONTROLS = REFERENCE.with_name('public33_seed42_controls_r2')
DEFAULT_ROOT = REFERENCE.with_name('public33_gate_confirmation_r1')
ENTRY = REPO / 'tools/seer/run_latentloop_freshness_confirmation.py'
RETAINED_VERIFY_STAGE = retained.verify_stage


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    result = retained.sources()
    paths = list(Path(__file__).parent.glob('*.py')) + [ENTRY,
        REPO / 'architectures/seer/wrappers/lrnode/run_latentloop_freshness_confirmation.sh']
    paths += list((REPO / 'tests/seer_freshness_confirmation').glob('*.py'))
    for path in paths:
        result[str(path.relative_to(REPO))] = base.digest(path)
    return result


def immutable_json(path, value):
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f'Existing experiment metadata differs: {path}')
    else:
        retained.atomic_json(path, value)


def verify_stage(root, stage, config):
    if stage == 'smoke':
        if not RETAINED_VERIFY_STAGE(root, stage, config):
            return False
        for rank in range(4):
            result = read(root / stage / f'rank{rank}.json')
            if result.get('fixed_cached_formula_checks') != 6 or not result.get('fixed_cached_reset_pass'):
                raise RuntimeError('New fixed-gate smoke incomplete')
        return True
    if not base.verify_stage(root, stage, config, 4):
        return False
    for i in range(500):
        value = read(root / stage / f'episode_{i:04d}.json')
        expected = {'stage':stage, 'seed':config['seed'], 'rank':i//125, 'task_id':i//50, 'trial':i%50}
        if any(value.get(k) != v for k,v in expected.items()):
            raise RuntimeError(f'Episode identity differs: {root}/{stage}/{i}')
    return True


def seed42_source(row):
    return CONTROLS if row in ('cached_feature', 'gate_fixed_residual_zero') else REFERENCE


def verify_inputs():
    retained.verify_reference()
    for name, sha in read(CONTROLS / 'source_hashes.json').items():
        if base.digest(REPO / name) != sha:
            raise RuntimeError(f'Completed controls source changed: {name}')
    config = read(CONTROLS / 'config.json')
    for path, sha in config['initial_state_hashes'].items():
        if base.digest(path) != sha:
            raise RuntimeError(f'Initial states changed: {path}')
    calibration = read(CONTROLS / 'calibration.json')
    rows = [read(CONTROLS / 'calibration' / f'episode_{i:04d}.json') for i in range(20)]
    if retained.calibration_means(rows) != calibration['gates']:
        raise RuntimeError('Frozen calibration no longer matches original episodes')
    if not retained.verify_stage(CONTROLS, 'calibration', config):
        raise RuntimeError('Calibration artifacts incomplete')
    for row in ROWS:
        if row != NEW_ROW and not verify_stage(seed42_source(row), row, config):
            raise RuntimeError(f'Reused seed42 row incomplete: {row}')
    base.preflight(config)
    return config, calibration


def worker(options):
    root = Path(options.config).parent
    config = read(options.config)
    if sources() != read(root / 'source_hashes.json'):
        raise RuntimeError('Source differs from frozen confirmation campaign')
    for path, sha in config['initial_state_hashes'].items():
        if base.digest(path) != sha:
            raise RuntimeError(f'Initial states changed: {path}')
    if read(root / 'calibration.json')['gates'] != config['fixed_gates']:
        raise RuntimeError('Frozen gate changed')
    sys.path[:0] = [str(REPO / 'architectures/seer/upstream'), config['libero']]
    import eval_libero
    from .runtime import evaluate_loaded
    sys.argv = ['eval_libero.py'] + evaluator_argv(config, root / options.stage)
    eval_libero.eval_one_epoch_libero_ddp = lambda args, model, image_processor, tokenizer: evaluate_loaded(
        args, model, image_processor, tokenizer, config, options.stage, root / options.stage)
    try:
        eval_libero.main()
    finally:
        import torch
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def summarize(root):
    results = []
    for seed in SEEDS:
        baseline_root = REFERENCE if seed == 42 else root / f'seed{seed}'
        baseline = [read(baseline_root / 'full' / f'episode_{i:04d}.json') for i in range(500)]
        for row in ROWS:
            origin = seed42_source(row) if seed == 42 and row != NEW_ROW else root / f'seed{seed}'
            config = read(origin / 'config.json')
            if not verify_stage(origin, row, config):
                raise RuntimeError('Do not summarize incomplete rows')
            values = [read(origin / row / f'episode_{i:04d}.json') for i in range(500)]
            both = [i for i in range(500) if baseline[i]['success'] and values[i]['success']]
            successes = sum(v['success'] for v in values)
            results.append({'seed':seed, 'row':row, 'source':str(origin / row), 'episodes':500,
                'successes':successes, 'sr_percent':successes/5,
                'gain_full_pp':(successes-sum(v['success'] for v in baseline))/5,
                'both_success':len(both), 'mean_steps_saved':statistics.mean(
                    baseline[i]['env_steps']-values[i]['env_steps'] for i in both) if both else None})
    aggregate = []
    for row in ROWS:
        values = [r['sr_percent'] for r in results if r['row'] == row]
        aggregate.append({'row':row, 'seed_sr_percent':values, 'mean_sr_percent':statistics.mean(values),
                          'sample_sd_pp':statistics.stdev(values), 'evaluation_seeds':list(SEEDS)})
    report = REPO / 'codex_outputs/seer/freshness' / root.name
    report.mkdir(parents=True, exist_ok=True)
    retained.atomic_json(report / 'results.json', {'rows':results, 'aggregate':aggregate})
    lines = ['# Seer freshness confirmation', '',
        'Frozen public33 + adapter39. Three evaluation seeds, not three training seeds.',
        'EGL, 500 episodes per row and seed. Calibration reused unchanged from controls_r2.',
        'Seed42 references are explicitly identified in results.json. No latency claim.', '',
        '| Row | Seed42 | Seed43 | Seed44 | Mean | Sample SD |', '|---|---:|---:|---:|---:|---:|']
    for value in aggregate:
        fields = [value['row']] + [f'{v:.1f}' for v in value['seed_sr_percent']]
        fields += [f'{value["mean_sr_percent"]:.2f}', f'{value["sample_sd_pp"]:.2f}']
        lines.append('| '+' | '.join(fields)+' |')
    (report / 'report.md').write_text('\n'.join(lines)+'\n')
    print(f'[DONE] {report / "report.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight','smoke','run','worker'))
    parser.add_argument('--config')
    parser.add_argument('--stage', choices=('smoke',)+ROWS)
    options = parser.parse_args()
    base.configure_environment()
    os.environ['LIBERO_GL_REQUIRE_ACTUAL'] = '1'
    if socket.gethostname() != 'jbrserver1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '4,5,6,7':
        raise ValueError('Use only sd1 physical GPUs 4,5,6,7')
    if options.mode == 'worker':
        return worker(options)
    root = Path(os.environ.get('RESULT_ROOT', str(DEFAULT_ROOT))).resolve()
    if root.parent != DEFAULT_ROOT.parent or root in (REFERENCE, CONTROLS):
        raise ValueError('Use a new confirmation folder on shared storage')
    original, calibration = verify_inputs()
    if options.mode == 'preflight':
        print('[PREFLIGHT PASS] 6 retained rows + 15 new rows; fixed calibration; no training', flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    immutable_json(root / 'plan.json', {'seeds':list(SEEDS), 'rows':list(ROWS),
                                       'new_evaluations':[list(row) for row in experiment_plan()],
                                       'calibration_source':str(CONTROLS / 'calibration.json')})
    locked = sources()
    immutable_json(root / 'source_hashes.json', locked)
    for name, command in (('commit.txt',['git','rev-parse','HEAD']), ('git_status.txt',['git','status','--short'])):
        if not (root / name).exists():
            (root / name).write_text(subprocess.check_output(command,cwd=REPO,text=True))
    configs = {}
    for seed in SEEDS:
        seed_root = root / f'seed{seed}'
        seed_root.mkdir(exist_ok=True)
        configs[seed] = dict(original, seed=seed, result_root=str(seed_root), fixed_gates=calibration['gates'],
            calibration_sha256=base.digest(CONTROLS / 'calibration.json'), experiment='freshness_confirmation')
        immutable_json(seed_root / 'config.json', configs[seed])
        immutable_json(seed_root / 'source_hashes.json', locked)
        immutable_json(seed_root / 'calibration.json', calibration)
    port = int(os.environ.get('MASTER_PORT_BASE','18300'))
    print(f'[PLAN] 15 x 500 new episodes; seeds=42,43,44; six seed42 rows reused; output={root}', flush=True)
    with replaced_method(retained, 'ENTRY', ENTRY), replaced_method(retained, 'verify_stage', verify_stage):
        retained.run_stage(root / 'seed42', 'smoke', configs[42], port)
        if options.mode == 'smoke':
            print('[GPU SMOKE PASS] retained formulas + fixed gate/cached residual + cache reset', flush=True)
            return
        for index, (seed, row) in enumerate(experiment_plan(), start=1):
            print(f'[{index}/15] seed={seed} row={row}; {15-index} subsequent rows', flush=True)
            retained.run_stage(root / f'seed{seed}', row, configs[seed], port+index)
    summarize(root)


if __name__ == '__main__':
    main()
