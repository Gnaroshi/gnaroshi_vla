"""Four matched gradient-routing continuations with automatic rb2 handoff."""
import argparse
from pathlib import Path

import torch

from methods.latentloop.modules.condition_output_split import ARMS
from tools.simvla.condition_initialization_pipeline import jobs
from tools.simvla.condition_solver_pipeline import solver_paths, transfer_status
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.error_compensation_common import ROOT, environment, read_json, sha, write_json
from tools.simvla.observation_correction_pipeline import export_arm
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT = solver_paths(2)[0].parent / 'joint_action_gradient_seed01_v1'
DEST = 'rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_condition_gradient'
EXTRA = ['tools/simvla/condition_gradient_pipeline.py', 'tools/simvla/condition_gradient_rb2.py',
         'architectures/simvla/wrappers/run_condition_gradient.sh']


def configurations():
    configs = {}
    for nfe in (1, 2):
        phase = f'nfe{nfe}'
        control_root = solver_paths(nfe)[0] / 'pretrained_10k'
        c = read_json(control_root / 'runtime_config.json')
        controls = {}
        for arm in ARMS:
            d = read_json(control_root / 'train' / arm / 'summary.json')
            if (d['verdict'] != 'TRAIN_AND_OFFLINE_COMPLETE' or d['total_training_steps'] != 10000
                    or sha(d['checkpoint']) != d['checkpoint_sha256']):
                raise RuntimeError('Incomplete matched control: '+arm)
            p = torch.load(d['checkpoint'], map_location='cpu', weights_only=False)
            controls[arm] = dict(checkpoint_sha256=d['checkpoint_sha256'], contract=p['contract'])
        c.update(output=str(OUTPUT / phase), action_gradient_mode='joint', gradient_transition='detached_to_joint',
            matched_controls=controls, run_label='condition_joint_gradient_'+phase,
            rb2_destination=DEST+'/'+phase,
            extra_source_files=sorted(set(c['extra_source_files']+EXTRA)),
            training_description='Same pretrained 3K starts and 7K samples/optimizer as completed detached controls. NFE1 and NFE2 x both carried-condition routes. Only connect action gradients through both updaters and observation encoder; original10 teacher frozen. Four models, no new cache.',
            evaluation_plan='After existing rb2 queues: each model K4/K8, 500 paired LIBERO-Long seed01 episodes, compiled RTX5090 policy latency, H10/R5. Reuse completed controls; no SR gate.')
        prepare(c)
        write_json(Path(c['output'])/'runtime_config.json', c)
        configs[phase] = c
    return configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--export', action='store_true')
    p.add_argument('--config'); p.add_argument('--arm', choices=ARMS)
    a = p.parse_args()
    if a.export:
        export_arm(read_json(a.config), a.arm)
        return 0
    configs = configurations()
    plan = jobs(configs, export_module='tools.simvla.condition_gradient_pipeline')
    write_json(OUTPUT/'planned_jobs.json', plan)
    if a.preflight:
        print('CPU_PREFLIGHT_PASS: four joint-gradient continuations, detached controls reused', flush=True)
        return 0
    status = OUTPUT/'pipeline_status.json'
    write_json(status, dict(phase='running')); transfer_status(status, DEST)
    rc = 1
    try:
        rc = run_queue(OUTPUT, plan, gpus=(4,5,6,7),
            predecessor=dict(path=str(solver_paths(2)[0]),lock='queue.lock'),
            environment=lambda gpu:environment(configs['nfe1'],gpu), cwd=ROOT, timeout=12*3600)
    finally:
        rows = {}
        for phase in configs:
            for arm in ARMS:
                path=OUTPUT/phase/'train'/arm/'summary.json'
                if path.exists(): rows[f'{phase}_{arm}']=read_json(path)
        write_json(OUTPUT/'training_summary.json',dict(rows=rows))
        write_json(status,dict(phase='complete' if not rc else 'finished_with_failures'))
        transfer_status(status,DEST)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
