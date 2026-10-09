"""Matched continuations testing direct observation-feature supervision."""
import argparse
from pathlib import Path

from methods.latentloop.modules.condition_feature_supervision import MODES
from tools.simvla.condition_noise_pipeline import OUTPUT as PRIOR, ARM
from tools.simvla.condition_solver_pipeline import transfer_status
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.error_compensation_common import ROOT, environment, identity, read_json, sha, write_json
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.observation_correction_pipeline import export_arm

OUTPUT = PRIOR.parent/'observation_feature_nfe1_seed01_v1'
DEST = 'rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_condition_feature'


def configurations():
    source = PRIOR/'joint_noise1'
    previous = read_json(source/'runtime_config.json')
    summary_path = source/'train'/ARM/'summary.json'
    summary = read_json(summary_path)
    if (summary['verdict'] != 'TRAIN_AND_OFFLINE_COMPLETE' or summary['total_training_steps'] != 10000
            or sha(summary['checkpoint']) != summary['checkpoint_sha256']):
        raise RuntimeError('Common fresh 10K source incomplete or changed')
    configs = {}
    for mode in MODES:
        c = {**previous, 'output': str(OUTPUT/mode), 'steps': 5000, 'warmup_steps': 250,
            'sample_step_offset': 10000, 'initialization': 'continuation',
            'continuation_source_steps': 10000, 'continuation_source_action_mode': 'naive1',
            'initial_models': {ARM: dict(summary=str(summary_path), identity=summary['identity'])},
            'feature_alignment': dict(mode=mode, weight=0.05,
                target='Image groups1/2/3 only: per-token LayerNorm(z_F_current), consecutive LayerNorm difference, or mean of both MSEs',
                input='Existing 128-D observation code and token index only',
                deployment='Reader discarded; model architecture/parameters unchanged',
                gradient='Auxiliary loss to reader and observation encoder only; original targets detached; separate clip1 for model and reader'),
            'run_label': 'condition_feature_'+mode, 'rb2_destination': DEST+'/'+mode,
            'extra_source_files': sorted(set(previous['extra_source_files']+[
                'methods/latentloop/modules/condition_feature_supervision.py',
                'tools/simvla/condition_feature_pipeline.py','tools/simvla/condition_feature_rb2.py',
                'architectures/simvla/wrappers/run_condition_feature.sh'])),
            'training_description': 'Common completed fresh joint_noise1 10K checkpoint. Four matched5K continuations: no auxiliary, current original image-token features, consecutive teacher feature delta, both. Same global samples10001..15000, batch2, mixedK4/K8, NFE1, LR1e-4 cosine warmup250. Training-only reader receives observation code, never predicted/teacher condition as input. Deployment736130 parameters unchanged. No SR gate.',
            'evaluation_plan': 'sd1 and rb2: each model K4/K8 x500 paired Long seed01. Independent export then existing compiled rb2 queue. Reuse existing baseline/LB.'}
        prepare(c)
        write_json(Path(c['output'])/'runtime_config.json',c)
        configs[mode] = c
    return configs


def jobs(configs):
    result = []
    for mode,c in configs.items():
        out = Path(c['output']); prefix = [c['python'],'-u','-m']
        def add(kind,module,extra,summary,completion,deps):
            key = mode+'_'+kind
            result.append(dict(id=key,cmd=prefix+[module,'--config',str(out/'runtime_config.json'),'--arm',ARM]+extra,
                summary=str(summary),completion=dict(identity=identity(c),**completion),deps=deps))
            return key
        smoke=add('smoke_train','tools.simvla.condition_output_split_train',['--smoke'],
            out/'smoke'/ARM/'summary.json',dict(verdict='SMOKE_PASS',steps=14),[])
        env=add('smoke_env','tools.simvla.condition_output_split_eval',['--smoke','--k-c','8'],
            out/'eval_smoke'/f'kc8_{ARM}'/'summary.json',dict(verdict='SMOKE_PASS',episodes=1),[smoke])
        trained=add('train','tools.simvla.condition_output_split_train',[],out/'train'/ARM/'summary.json',
            dict(verdict='TRAIN_AND_OFFLINE_COMPLETE',steps=5000),[env])
        add('export','tools.simvla.condition_feature_pipeline',['--export'],out/'exports'/f'{ARM}.json',
            dict(verdict='BUNDLE_EXPORTED'),[trained])
    for k in (4,8):
        for mode,c in configs.items():
            out=Path(c['output'])
            result.append(dict(id=f'{mode}_kc{k}',deps=[mode+'_train'],
                cmd=[c['python'],'-u','-m','tools.simvla.condition_output_split_eval',
                    '--config',str(out/'runtime_config.json'),'--arm',ARM,'--k-c',str(k)],
                summary=str(out/'online'/f'kc{k}_{ARM}'/'summary.json'),
                completion=dict(identity=identity(c),verdict='EVALUATION_COMPLETE',episodes=500)))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    p.add_argument('--export',action='store_true');p.add_argument('--config');p.add_argument('--arm',choices=(ARM,))
    a=p.parse_args()
    if a.export:
        export_arm(read_json(a.config),a.arm)
        return 0
    configs=configurations();plan=jobs(configs);write_json(OUTPUT/'planned_jobs.json',plan)
    if a.preflight:
        print('PREFLIGHT_PASS: common10K source, four matched5K continuations, 8x500 evaluations',flush=True)
        return 0
    status=OUTPUT/'pipeline_status.json';write_json(status,dict(phase='running'));transfer_status(status,DEST)
    rc=1
    try:
        rc=run_queue(OUTPUT,plan,gpus=(4,5,6,7),predecessor=dict(path=str(PRIOR),lock='queue.lock'),
            environment=lambda gpu:environment(configs['none'],gpu),cwd=ROOT,timeout=12*3600)
    finally:
        write_json(status,dict(phase='complete' if not rc else 'finished_with_failures'));transfer_status(status,DEST)
    return rc


if __name__=='__main__':
    raise SystemExit(main())
