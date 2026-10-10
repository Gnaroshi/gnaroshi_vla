"""Matched fresh K4/K8 training with bounded versus recurrent history input."""
import argparse
from pathlib import Path
import subprocess

from tools.simvla.condition_interval_recovery import OUTPUT as PRIOR, verify_dataset
from tools.simvla.error_compensation_common import ROOT, environment, identity, read_json, write_json
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.observation_correction_pipeline import export_arm
from tools.simvla.priority_handoff import finish

OUTPUT = PRIOR.parent.parent/'bounded_history/fresh_nfe1_seed01_v1'
DEST = 'rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_bounded_history'
ARM = 'carry_base'
MODES = {f'{variant}_k{k}': (variant,k) for k in (4,8) for variant in ('bounded','recurrent')}
EXTRA = ['methods/latentloop/modules/bounded_history.py', 'tools/simvla/bounded_history_pipeline.py',
         'tools/simvla/bounded_history_rb2.py', 'tools/simvla/priority_handoff.py',
         'architectures/simvla/wrappers/run_bounded_history.sh']


def configurations():
    import torch
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
    from tools.simvla.condition_output_split_train import build_initial_model
    previous = read_json(PRIOR/'fresh_k4/runtime_config.json')
    summary = read_json(PRIOR/'fresh_k4/train'/ARM/'summary.json')
    expected = torch.load(summary['checkpoint'],map_location='cpu',weights_only=False)['contract']
    parent,payload = load_native_v0_checkpoint(previous['condition_checkpoint'],device='cpu',require_final_150k=True)
    observed = verify_dataset(previous,payload,expected)
    configs,initial = {},None
    for mode,(variant,k) in MODES.items():
        c = {**previous,'output':str(OUTPUT/mode),'bounded_history_variant':variant,
            'initialization':'fresh','steps':10000,'smoke_steps':14,'warmup_steps':500,'sample_step_offset':0,
            'training_intervals':[k],'training_condition_ages':list(range(1,k)),
            'evaluation_condition_intervals':[k],'offline_validation_intervals':[k],
            'student_steps':1,'teacher_steps':10,'freeze_condition_predictor':False,
            'run_label':'bounded_history_'+mode,'rb2_destination':DEST+'/'+mode,
            'extra_source_files':sorted(set(previous['extra_source_files']+EXTRA)),
            'student_condition_description':'Recursive single-updater outputs; bounded arm conditions candidate and gate on the fixed original refresh condition, matched recurrent arm on previous prediction.',
            'training_description':'Fresh10K, matched seed7, data, parameter count and optimizer. NFE1 student vs cached original NFE10 same-noise first5 continuous-action L1 plus0.05 condition MSE. Both losses train the same updater and CNN across all unrolled ages. No Generation Loop or extra action Condition head.',
            'evaluation_plan':'Each matched K4/K8 model: Long500 seed01 on sd1 eager and rb2 compiled. Original H10/R5, no SR stopping gate.'}
        for key in ('initial_models','continuation_source_steps','continuation_source_action_mode',
                    'interval_transition','solver_transition','teacher_transition'):
            c.pop(key,None)
        model = build_initial_model(parent,c,ARM)
        current = dict(weights_sha256=state_hash(model),parameters=sum(p.numel() for p in model.parameters()))
        if initial is not None and current!=initial:
            raise RuntimeError('Matched arms have different initialization or parameter counts')
        initial=current
        prepare(c);write_json(Path(c['output'])/'runtime_config.json',c)
        write_json(Path(c['output'])/'dataset_preflight.json',dict(verdict='ACTUAL_DATASET_AND_INITIALIZATION_PASS',**observed,**current))
        configs[mode]=c
    return configs


def jobs(configs):
    plan=[]
    for mode,(_,k) in MODES.items():
        c=configs[mode];out=Path(c['output'])
        def add(kind,module,args,summary,completion,deps):
            key=mode+'_'+kind
            plan.append(dict(id=key,cmd=[c['python'],'-u','-m',module,'--config',str(out/'runtime_config.json'),'--arm',ARM]+args,
                summary=str(summary),completion=dict(identity=identity(c),**completion),deps=deps))
            return key
        smoke=add('smoke_train','tools.simvla.condition_output_split_train',['--smoke'],out/'smoke'/ARM/'summary.json',dict(verdict='SMOKE_PASS',steps=14),[])
        env=add('smoke_env','tools.simvla.condition_output_split_eval',['--smoke','--k-c',str(k)],out/'eval_smoke'/f'kc{k}_{ARM}'/'summary.json',dict(verdict='SMOKE_PASS',episodes=1),[smoke])
        train=add('train','tools.simvla.condition_output_split_train',[],out/'train'/ARM/'summary.json',dict(verdict='TRAIN_AND_OFFLINE_COMPLETE',steps=10000),[env])
        add('export','tools.simvla.bounded_history_pipeline',['--export'],out/'exports'/f'{ARM}.json',dict(verdict='BUNDLE_EXPORTED'),[train])
    for mode,(_,k) in MODES.items():
        c=configs[mode];out=Path(c['output'])
        plan.append(dict(id=mode+'_online',cmd=[c['python'],'-u','-m','tools.simvla.condition_output_split_eval',
            '--config',str(out/'runtime_config.json'),'--arm',ARM,'--k-c',str(k)],deps=[mode+'_train'],
            summary=str(out/'online'/f'kc{k}_{ARM}'/'summary.json'),completion=dict(identity=identity(c),verdict='EVALUATION_COMPLETE',episodes=500)))
    return plan


def transfer_status(path):
    host,remote=DEST.split(':',1)
    for attempt in range(3):
        try:
            subprocess.run(['ssh',host,'mkdir','-p',remote],check=True,timeout=30)
            subprocess.run(['rsync','-a',str(path),host+':'+remote+'/'],check=True,timeout=60)
            return
        except (OSError,subprocess.SubprocessError) as exc:
            print('TRANSFER_STATUS_WARNING',attempt,str(exc),flush=True)


def summarize(configs):
    rows={}
    for mode,c in configs.items():
        for path in Path(c['output']).glob('online/*/summary.json'):
            rows[mode]=read_json(path)
    write_json(OUTPUT/'comparison_summary.json',dict(rows=rows,complete=len(rows)==4,hardware='sd1 RTX3090 eager'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--export',action='store_true')
    p.add_argument('--config');p.add_argument('--arm',choices=(ARM,));a=p.parse_args()
    if a.export:export_arm(read_json(a.config),ARM);return 0
    configs=configurations();plan=jobs(configs);write_json(OUTPUT/'planned_jobs.json',plan)
    if a.preflight:print('PREFLIGHT_PASS bounded/recurrent matched fresh K4/K8',flush=True);return 0
    status=OUTPUT/'pipeline_status.json'
    write_json(status,dict(phase='waiting_for_preserved_workers'));transfer_status(status)
    rc=1
    try:
        record=OUTPUT.parent/'priority/drain_fixed_condition.json'
        if record.exists():finish(record)
        write_json(status,dict(phase='running'));transfer_status(status)
        rc=run_queue(OUTPUT,plan,gpus=(4,5,6,7),predecessor=[],
            environment=lambda gpu:environment(configs['bounded_k4'],gpu),cwd=ROOT,timeout=24*3600)
    finally:
        summarize(configs);write_json(status,dict(phase='complete' if not rc else 'finished_with_failures'));transfer_status(status)
    return rc


if __name__=='__main__':raise SystemExit(main())
