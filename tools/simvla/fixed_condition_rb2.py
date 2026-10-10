"""Compiled evaluation of matched fixed-predictor training after pending work."""
import argparse
import os
from pathlib import Path
import sys

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT,configure,read_json,write_json,sha
from tools.simvla.compiled_policy import attach_policy
from tools.simvla.condition_interval_recovery_rb2 import OUTPUT as PRIOR,base_config as previous_config
from tools.simvla.condition_output_split_rb2 import environment,replay_factory,compiler_checker,reset_checker
from tools.simvla.condition_output_split_eval import load_payload,attach,check_policy
from tools.simvla.fixed_condition_pipeline import MODES,ARM
from tools.simvla.rollout_repair_rb2 import STORAGE,evaluate
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT=PRIOR.parent/'fixed_condition_nfe1_compiled_seed01_v1'
INCOMING=STORAGE/'incoming/simvla_fixed_condition'
ROWS={f'{m}_k{k}':(m,k) for k in (4,3,2) for m in MODES}


def base_config():
    c=previous_config()
    c.update(output=str(OUTPUT),long_rows=list(ROWS),campaign_module='tools.simvla.fixed_condition_rb2',
        extra_source_files=c['extra_source_files']+['tools/simvla/fixed_condition_pipeline.py','tools/simvla/fixed_condition_rb2.py'],
        scope='Same two-updater inference. Freeze recurrent predictor during action-head5K training versus moving predictor control. Three new checkpoints at K2/3/4,500episodes per row. Original full10 teacher, NFE1 student. No SR gate.')
    return c


def ready_spec(mode):
    d=INCOMING/mode/ARM
    if read_json(d/'READY.json')['manifest_sha256']!=sha(d/'manifest.json'):raise RuntimeError('Incomplete model transfer')
    m=read_json(d/'manifest.json')
    if m['step']!=5000 or m['parameters']!=736130 or m['arm']!=ARM or sha(d/'model.pt')!=m['checkpoint_sha256']:
        raise RuntimeError('Model specification changed')
    payload=load_payload(d/'model.pt',ARM,m['source_identity'],steps=5000,action_mode='naive1')
    c=payload['contract'];fixed=mode!='fresh_joint'
    if (c['training_intervals']!=[4,8] or c['total_training_steps']!=15000 or c['sample_step_offset']!=10000
            or c['teacher_steps']!=10 or c['seed']!=7 or c['batch_size']!=2
            or c.get('frozen_condition_predictor',False)!=fixed):
        raise RuntimeError('Training budget/objective mismatch')
    if fixed and c['future_condition_gradient']!='Frozen delta_encoder and condition_updater at every age':
        raise RuntimeError('Frozen training route mismatch')
    if mode=='parent_fixed':
        from tools.simvla.condition_deployment_pipeline import PARENT_SHA
        if c['continuation']['checkpoint_sha256']!=PARENT_SHA:raise RuntimeError('Parent changed')
    else:
        ref=read_json(STORAGE/'incoming/simvla_condition_noise/detached_noise1/carry_base/manifest.json')
        if c['continuation']['checkpoint_sha256']!=ref['checkpoint_sha256']:raise RuntimeError('Fresh parent changed')
    return dict(path=str(d/'model.pt'),sha256=m['checkpoint_sha256'],source_identity=m['source_identity'],step=5000,mode=mode,student_steps=1)


def policy_factory(replay,c,row,manifest):
    mode,k=ROWS[row];s=c['model_checkpoint']
    if sha(s['path'])!=s['sha256']:raise RuntimeError('Checkpoint changed')
    p=load_payload(s['path'],ARM,s['source_identity'],steps=5000,action_mode='naive1')
    policy=attach_policy(replay,c,'condition_naive3',manifest);policy.NFE=policy.nfe=1
    policy=attach(policy,replay.native,p,ARM,k,replay.compiler);policy.row_name=row
    return policy


def cell(c,row):
    mode,k=ROWS[row];s=ready_spec(mode)
    result=evaluate({**c,'action_mode':'condition_naive1','condition_interval':k},row,s,output=OUTPUT)
    write_json(OUTPUT/'completed'/f'{row}.json',dict(verdict='EVALUATION_COMPLETE',episodes=500,row=row,
        checkpoint_sha256=s['sha256'],result=result));summarize()


def jobs():
    return [dict(id=row,cmd=[sys.executable,'-u','-m','tools.simvla.fixed_condition_rb2','cell','--row',row],
        ready_file=str(INCOMING/mode/ARM/'READY.json'),upstream_status_file=str(INCOMING/'pipeline_status.json'),
        summary=str(OUTPUT/'completed'/f'{row}.json'),completion=dict(verdict='EVALUATION_COMPLETE',episodes=500,row=row))
        for row,(mode,k) in ROWS.items()]


def summarize():
    rows={row:read_json(OUTPUT/'completed'/f'{row}.json') for row in ROWS if (OUTPUT/'completed'/f'{row}.json').exists()}
    write_json(OUTPUT/'comparison_summary.json',dict(rows=rows,complete=len(rows)==len(ROWS),hardware='rb2 RTX5090 compiled',
        reference_comparison=str(PRIOR/'comparison_summary.json'),episodes_per_row=500,seed='seed01'))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',nargs='?',default='all',choices=('all','cell','smoke','worker'))
    p.add_argument('--preflight',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--row',choices=ROWS)
    p.add_argument('--suite',default='libero_10',choices=('libero_10',));p.add_argument('--seed',default='seed01',choices=('seed01',));a=p.parse_args()
    env=environment(0);os.environ.clear();os.environ.update(env)
    c=read_json(a.output/'runtime_config.json') if a.output else base_config()
    configure(c);sys.path.insert(0,c['libero_root']);os.environ['LIBERO_CONFIG_PATH']=c['libero_config']
    if a.command in ('smoke','worker'):
        campaign.worker(c,a.output,a.suite,a.seed,a.row,smoke=a.command=='smoke',replay_factory=replay_factory,
            policy_factory=policy_factory,policy_checker=lambda policy,row:check_policy(policy),
            compiler_checker=compiler_checker,reset_checker=reset_checker);return 0
    if a.command=='cell':cell(c,a.row);return 0
    campaign.prepare(c,OUTPUT);write_json(OUTPUT/'runtime_config.json',c);plan=jobs();write_json(OUTPUT/'planned_jobs.json',plan);summarize()
    if a.preflight:print('PREFLIGHT_PASS nine nonduplicate compiled rows after interval queue',flush=True);return 0
    rc=run_queue(OUTPUT,plan,gpus=(0,),predecessor=dict(path=str(PRIOR),lock='queue.lock'),environment=environment,cwd=ROOT,timeout=24*3600)
    summarize();return rc


if __name__=='__main__':raise SystemExit(main())
