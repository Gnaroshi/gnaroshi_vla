"""Matched fresh/pretrained initialization; reuse the completed first 3K phase."""
import argparse
from pathlib import Path
import subprocess

from methods.latentloop.modules.condition_output_split import ARMS
from tools.simvla.condition_output_split_pipeline import OUTPUT as PRIOR, configuration as original_config
from tools.simvla.error_compensation_common import ROOT, read_json, write_json, identity, environment, sha
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.observation_correction_pipeline import export_arm
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT=PRIOR.parent/'initialization_matched_seed01_v1'
DEST='rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_condition_initialization'
PHASES=('fresh_3k','pretrained_10k','fresh_10k')
EXTRA=['tools/simvla/condition_initialization_pipeline.py',
       'tools/simvla/condition_initialization_rb2.py',
       'architectures/simvla/wrappers/run_condition_initialization.sh']


def configurations():
    original=original_config()
    previous_identity=read_json(PRIOR/'contract.json')['identity']
    configs={}
    for phase in PHASES:
        c={**original,'output':str(OUTPUT/phase),'run_label':'condition_initialization_'+phase,
            'initialization':'fresh' if phase=='fresh_3k' else 'continuation',
            'steps':3000 if phase=='fresh_3k' else 7000,
            'sample_step_offset':0 if phase=='fresh_3k' else 3000,
            'warmup_steps':150 if phase=='fresh_3k' else 350,
            'rb2_destination':DEST+'/'+phase,'initialization_group':phase.split('_')[0],
            'extra_source_files':original['extra_source_files']+EXTRA,
            'training_description':'Initialization x next-query routing. Reuse pretrained 3K phase; fresh group trains same 3K schedule and samples. All four models continue 7K with a fresh identical AdamW/warmup/cosine schedule and global sample steps 3001..10000. Original SimVLA frozen; both updaters and encoder trainable. Original10 action teacher, naive3 student. No SR stopping gate.',
            'evaluation_plan':'Finish current rb2 four 3K rows first, then four final 10K models x K4/K8 x 500 LIBERO-Long episodes, same seed01 manifest and compiled policy latency.'}
        if phase!='fresh_3k':
            source=PRIOR if phase=='pretrained_10k' else OUTPUT/'fresh_3k'
            source_id=previous_identity if phase=='pretrained_10k' else identity(configs['fresh_3k'])
            c['initial_models']={arm:dict(summary=str(source/'train'/arm/'summary.json'),identity=source_id) for arm in ARMS}
        prepare(c)
        write_json(Path(c['output'])/'runtime_config.json',c)
        configs[phase]=c
    return configs


def jobs(configs, export_module='tools.simvla.condition_initialization_pipeline'):
    plan=[]
    for phase,c in configs.items():
        out=Path(c['output']); run_id=identity(c); path=out/'runtime_config.json'
        for arm in ARMS:
            prefix=[c['python'],'-u','-m']
            def add(kind,module,extra,summary,completion,deps):
                key=f'{phase}_{kind}_{arm}'
                plan.append(dict(id=key,cmd=prefix+[module,'--config',str(path),'--arm',arm]+extra,
                    summary=str(summary),completion=dict(identity=run_id,**completion),deps=deps))
                return key
            deps=[f'fresh_3k_train_{arm}'] if phase=='fresh_10k' and 'fresh_3k' in configs else []
            smoke=add('smoke_train','tools.simvla.condition_output_split_train',['--smoke'],
                out/'smoke'/arm/'summary.json',dict(verdict='SMOKE_PASS',steps=14),deps)
            env=add('smoke_env','tools.simvla.condition_output_split_eval',['--smoke','--k-c','8'],
                out/'eval_smoke'/f'kc8_{arm}'/'summary.json',dict(verdict='SMOKE_PASS',episodes=1),[smoke])
            trained=add('train','tools.simvla.condition_output_split_train',[],out/'train'/arm/'summary.json',
                dict(verdict='TRAIN_AND_OFFLINE_COMPLETE',steps=c['steps']),[env])
            if phase!='fresh_3k':
                add('export',export_module,['--export'],
                    out/'exports'/f'{arm}.json',dict(verdict='BUNDLE_EXPORTED'),[trained])
    return plan


def summarize(configs):
    groups={'pretrained_3k':PRIOR,**{p:Path(c['output']) for p,c in configs.items()}}
    results={}
    for phase,directory in groups.items():
        for arm in ARMS:
            path=directory/'train'/arm/'summary.json'
            if path.exists(): results[f'{phase}_{arm}']=read_json(path)
    write_json(OUTPUT/'training_summary.json',dict(rows=results,
        schedule='3K + 7K with optimizer/scheduler reset at phase boundary; 150K historical pretraining belongs only to pretrained group',
        scope='Offline diagnostics and training cost; online SR and RTX5090 policy latency from queued rb2 evaluation'))


def transfer_status(status):
    host,remote=DEST.split(':',1)
    for attempt in range(2):
        try:
            subprocess.run(['ssh',host,'mkdir','-p',remote],check=True,timeout=30)
            subprocess.run(['rsync','-a',str(status),host+':'+remote+'/'],check=True,timeout=60)
            return
        except (subprocess.SubprocessError,OSError) as exc:
            print(f'STATUS_TRANSFER_WARNING attempt={attempt+1}: {exc}',flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    p.add_argument('--export',action='store_true'); p.add_argument('--config'); p.add_argument('--arm',choices=ARMS)
    a=p.parse_args()
    if a.export: export_arm(read_json(a.config),a.arm); return 0
    for arm in ARMS:
        d=read_json(PRIOR/'train'/arm/'summary.json')
        if d['verdict']!='TRAIN_AND_OFFLINE_COMPLETE' or d['steps']!=3000 or sha(d['checkpoint'])!=d['checkpoint_sha256']:
            raise RuntimeError('Prior 3K checkpoint is incomplete: '+arm)
    configs=configurations(); plan=jobs(configs)
    write_json(OUTPUT/'planned_jobs.json',plan)
    if a.preflight:
        print('CPU_PREFLIGHT_PASS: two fresh 3K trainings and four matched 7K continuations',flush=True); return 0
    status=OUTPUT/'pipeline_status.json'; write_json(status,dict(phase='running')); transfer_status(status)
    rc=1
    try:
        rc=run_queue(OUTPUT,plan,gpus=(4,5,6,7),predecessor=dict(path=str(PRIOR),lock='queue.lock'),
            environment=lambda gpu:environment({**configs['fresh_3k'],'output':str(OUTPUT/'fresh_3k')},gpu),
            cwd=ROOT,timeout=12*3600)
    finally:
        summarize(configs)
        write_json(status,dict(phase='complete' if not rc else 'finished_with_failures'))
        transfer_status(status)
    return rc


if __name__=='__main__': raise SystemExit(main())
