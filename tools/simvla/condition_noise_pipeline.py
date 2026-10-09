"""Fresh one-step training: gradient routing x paired-noise supervision."""
import argparse
from pathlib import Path

from tools.simvla.condition_output_split_pipeline import configuration as source_config
from tools.simvla.condition_solver_pipeline import transfer_status
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.error_compensation_common import ROOT, environment, identity, read_json, write_json
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.observation_correction_pipeline import export_arm

OUTPUT = Path('/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/condition_output_split/fresh_noise_nfe1_seed01_v1')
DEST = 'rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_condition_noise'
VARIANTS = {f'{mode}_noise{n}': (mode, n) for n in (1, 2) for mode in ('detached', 'joint')}
ARM = 'carry_base'


def configurations():
    original = source_config()
    configs = {}
    for name, (mode, count) in VARIANTS.items():
        c = {**original, 'output': str(OUTPUT/name), 'run_label': 'condition_fresh_'+name,
            'initialization': 'fresh', 'student_steps': 1, 'steps': 10000, 'sample_step_offset': 0,
            'warmup_steps': 500, 'action_gradient_mode': mode, 'action_noise_samples': count,
            'heldout_action_noise_samples': 3, 'rb2_destination': DEST+'/'+name,
            'extra_source_files': sorted(set(original['extra_source_files'] + [
                'tools/simvla/condition_noise_pipeline.py', 'tools/simvla/condition_noise_rb2.py',
                'architectures/simvla/wrappers/run_condition_noise.sh'])),
            'training_description': 'Four identical fresh initializations; carry_base only; NFE1 throughout 10K (no 3K NFE3 prefix, no native150K weights). Detached/joint action gradients x one/two paired action-noise targets. Same 2 windows per step, mixed K4/K8 unroll, LR1e-4 warmup500 cosine to0.1x. Original10 teacher/action head frozen. Two-noise arm has extra training compute, unchanged inference.',
            'evaluation_plan': 'Each model K4/K8 x500 fixed seed01 Long episodes on sd1; export independently; rb2 compiled queue after existing NFE2 and joint-gradient evaluations. Same inference work for all four arms. No SR stopping gate.'}
        prepare(c)
        write_json(Path(c['output'])/'runtime_config.json', c)
        configs[name] = c
    return configs


def jobs(configs):
    result = []
    # All four trainings are launched before any 500-episode job.
    for name, c in configs.items():
        out = Path(c['output']); config = out/'runtime_config.json'
        prefix = [c['python'], '-u', '-m']
        def add(kind, module, extra, summary, completion, deps):
            key = name+'_'+kind
            result.append(dict(id=key, cmd=prefix+[module, '--config', str(config), '--arm', ARM]+extra,
                summary=str(summary), completion=dict(identity=identity(c), **completion), deps=deps))
            return key
        smoke = add('smoke_train', 'tools.simvla.condition_output_split_train', ['--smoke'],
            out/'smoke'/ARM/'summary.json', dict(verdict='SMOKE_PASS', steps=14), [])
        env = add('smoke_env', 'tools.simvla.condition_output_split_eval', ['--smoke', '--k-c', '8'],
            out/'eval_smoke'/f'kc8_{ARM}'/'summary.json', dict(verdict='SMOKE_PASS', episodes=1), [smoke])
        trained = add('train', 'tools.simvla.condition_output_split_train', [], out/'train'/ARM/'summary.json',
            dict(verdict='TRAIN_AND_OFFLINE_COMPLETE', steps=10000), [env])
        add('export', 'tools.simvla.condition_noise_pipeline', ['--export'],
            out/'exports'/f'{ARM}.json', dict(verdict='BUNDLE_EXPORTED'), [trained])
    for k in (4, 8):
        for name, c in configs.items():
            out = Path(c['output'])
            result.append(dict(id=f'{name}_kc{k}',
                cmd=[c['python'], '-u', '-m', 'tools.simvla.condition_output_split_eval',
                    '--config', str(out/'runtime_config.json'), '--arm', ARM, '--k-c', str(k)],
                summary=str(out/'online'/f'kc{k}_{ARM}'/'summary.json'),
                completion=dict(identity=identity(c), verdict='EVALUATION_COMPLETE', episodes=500),
                deps=[name+'_train']))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true'); p.add_argument('--export', action='store_true')
    p.add_argument('--config'); p.add_argument('--arm', choices=(ARM,))
    a = p.parse_args()
    if a.export:
        export_arm(read_json(a.config), a.arm)
        return 0
    configs = configurations(); plan = jobs(configs)
    write_json(OUTPUT/'planned_jobs.json', plan)
    if a.preflight:
        print('PREFLIGHT_PASS: fresh 10K NFE1 x4, train/env smoke, export, K4/K8 x500', flush=True)
        return 0
    status = OUTPUT/'pipeline_status.json'
    write_json(status, dict(phase='running')); transfer_status(status, DEST)
    rc = 1
    try:
        rc = run_queue(OUTPUT, plan, gpus=(4,5,6,7),
            predecessor=dict(path=str(OUTPUT.parent/'fresh_initialization_sd1_seed01_v1'), lock='queue.lock'),
            environment=lambda gpu: environment(configs['detached_noise1'], gpu), cwd=ROOT, timeout=12*3600)
    finally:
        write_json(status, dict(phase='complete' if not rc else 'finished_with_failures'))
        transfer_status(status, DEST)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
