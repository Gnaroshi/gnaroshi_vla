"""Change only the final 7K student solver; reuse all completed naive3 controls."""
import argparse
from pathlib import Path
import subprocess

from methods.latentloop.modules.condition_output_split import ARMS
from tools.simvla.condition_initialization_pipeline import OUTPUT as PRIOR, jobs
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.error_compensation_common import ROOT, environment, read_json, sha, write_json
from tools.simvla.observation_correction_pipeline import export_arm
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT = PRIOR.parent / 'solver_matched_nfe1_seed01_v1'
DEST = 'rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_condition_solver'
PHASES = ('pretrained_10k', 'fresh_10k')
EXTRA = ['tools/simvla/condition_solver_pipeline.py', 'tools/simvla/condition_solver_rb2.py',
         'architectures/simvla/wrappers/run_condition_solver.sh']


def solver_paths(nfe):
    if type(nfe) is not int or nfe not in (1,2):
        raise ValueError('This continuation supports NFE1 or NFE2')
    return (PRIOR.parent / f'solver_matched_nfe{nfe}_seed01_v1',
            DEST if nfe==1 else DEST+'_nfe2')


def predecessor(nfe):
    return dict(path=str(PRIOR if nfe==1 else OUTPUT), lock='queue.lock')


def configurations(nfe=1):
    output, dest = solver_paths(nfe)
    configs = {}
    for phase in PHASES:
        original = read_json(PRIOR / phase / 'runtime_config.json')
        for arm in ARMS:
            control = read_json(PRIOR / phase / 'train' / arm / 'summary.json')
            if (control['verdict'] != 'TRAIN_AND_OFFLINE_COMPLETE'
                    or control['total_training_steps'] != 10000
                    or sha(control['checkpoint']) != control['checkpoint_sha256']):
                raise RuntimeError('Incomplete naive3 matched control')
        c = {**original, 'output': str(output / phase), 'student_steps': nfe,
             'solver_transition': f'naive3_to_naive{nfe}', 'run_label': f'condition_solver_nfe{nfe}_' + phase,
             'rb2_destination': dest + '/' + phase,
             'extra_source_files': sorted(set(original['extra_source_files'] + EXTRA)),
             'training_description': f'Reuse identical completed 3K initial models, samples 3001..10000, batch2, optimizer reset and 7K cosine. Change student naive3 to naive{nfe} only. Original10 teacher and Condition targets, trainable modules and gradient paths unchanged. Four existing naive3 10K controls reused.',
             'evaluation_plan': 'After existing rb2 queues: each final model K4/K8, 500 LIBERO-Long seed01 episodes. Compiled RTX5090 full policy latency, same H10/R5. No SR stopping gate.'}
        prepare(c)
        write_json(Path(c['output']) / 'runtime_config.json', c)
        configs[phase] = c
    return configs


def transfer_status(status, dest=DEST):
    host, remote = dest.split(':', 1)
    for attempt in range(2):
        try:
            subprocess.run(['ssh', host, 'mkdir', '-p', remote], check=True, timeout=30)
            subprocess.run(['rsync', '-a', str(status), host + ':' + remote + '/'], check=True, timeout=60)
            return
        except (subprocess.SubprocessError, OSError) as exc:
            print(f'STATUS_TRANSFER_WARNING attempt={attempt+1}: {exc}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--export', action='store_true')
    p.add_argument('--config'); p.add_argument('--arm', choices=ARMS)
    p.add_argument('--student-steps', type=int, choices=(1,2), default=1)
    a = p.parse_args()
    if a.export:
        export_arm(read_json(a.config), a.arm)
        return 0
    output, dest = solver_paths(a.student_steps)
    configs = configurations(a.student_steps)
    plan = jobs(configs, export_module='tools.simvla.condition_solver_pipeline')
    write_json(output / 'planned_jobs.json', plan)
    if a.preflight:
        print(f'CPU_PREFLIGHT_PASS: four matched 7K naive{a.student_steps} continuations, existing naive3 controls reused', flush=True)
        return 0
    status = output / 'pipeline_status.json'
    write_json(status, dict(phase='running')); transfer_status(status, dest)
    rc = 1
    try:
        rc = run_queue(output, plan, gpus=(4,5,6,7),
            predecessor=predecessor(a.student_steps),
            environment=lambda gpu: environment(configs[PHASES[0]], gpu), cwd=ROOT, timeout=12*3600)
    finally:
        rows = {}
        for phase in PHASES:
            for arm in ARMS:
                path = output / phase / 'train' / arm / 'summary.json'
                if path.exists(): rows[f'{phase}_{arm}'] = read_json(path)
        write_json(output / 'training_summary.json', dict(rows=rows, matched_naive3_root=str(PRIOR)))
        write_json(status, dict(phase='complete' if not rc else 'finished_with_failures'))
        transfer_status(status, dest)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
