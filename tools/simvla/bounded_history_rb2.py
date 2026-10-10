"""Compiled Long500 evaluation of matched bounded-history and recurrent models."""
import argparse
import os
from pathlib import Path
import sys

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT,configure,read_json,write_json,sha
from tools.simvla.compiled_policy import attach_policy
from tools.simvla.condition_output_split_rb2 import base_config as previous_config,environment,replay_factory,reset_checker
from tools.simvla.condition_output_split_eval import load_payload,attach,check_policy
from tools.simvla.bounded_history_pipeline import MODES,ARM,EXTRA
from tools.simvla.rollout_repair_rb2 import STORAGE,evaluate
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.priority_handoff import finish

OUTPUT=STORAGE/'results/simvla/bounded_history/fresh_nfe1_compiled_seed01_v1'
INCOMING=STORAGE/'incoming/simvla_bounded_history'


def base_config():
    c=previous_config()
    c.update(output=str(OUTPUT),long_rows=list(MODES),campaign_module='tools.simvla.bounded_history_rb2',
        extra_source_files=sorted(set(c['extra_source_files']+EXTRA)),
        scope='Fresh matched single-updater bounded/recurrent K4/K8,NFE1. Original H10/R5,Long500 seed01 paired. No SR gate.')
    return c


def ready_spec(mode):
    variant,k=MODES[mode];d=INCOMING/mode/ARM
    if read_json(d/'READY.json')['manifest_sha256']!=sha(d/'manifest.json'):raise RuntimeError('Incomplete model transfer')
    m=read_json(d/'manifest.json')
    if m['step']!=10000 or m['arm']!=ARM or sha(d/'model.pt')!=m['checkpoint_sha256']:
        raise RuntimeError('Model specification changed')
    payload=load_payload(d/'model.pt',ARM,m['source_identity'],steps=10000,action_mode='naive1')
    c=payload['contract']
    if (c.get('bounded_history_variant')!=variant or c['training_intervals']!=[k]
            or c['total_training_steps']!=10000 or c['sample_step_offset']!=0
            or c['initialization']!='fresh' or c['teacher_steps']!=10 or c['seed']!=7 or c['batch_size']!=2):
        raise RuntimeError('Training budget/objective mismatch')
    return dict(path=str(d/'model.pt'),sha256=m['checkpoint_sha256'],source_identity=m['source_identity'],step=10000,
        mode=mode,student_steps=1,parameters=m['parameters'])


def policy_factory(replay,c,row,manifest):
    _,k=MODES[row];s=c['model_checkpoint']
    if sha(s['path'])!=s['sha256']:raise RuntimeError('Checkpoint changed')
    payload=load_payload(s['path'],ARM,s['source_identity'],steps=10000,action_mode='naive1')
    policy=attach_policy(replay,c,'condition_naive3',manifest);policy.NFE=policy.nfe=1
    policy=attach(policy,replay.native,payload,ARM,k,replay.compiler);policy.row_name=row
    return policy


def compiler_checker(compiler,row):
    required={'vlm','action_transformer','observation_encoder','condition_updater'}
    missing=[key for key in required if not compiler.records.get(key,{}).get('graphs',0)]
    if missing:raise RuntimeError('Compile bypass: '+str(missing))


def cell(c,row):
    _,k=MODES[row];spec=ready_spec(row)
    result=evaluate({**c,'action_mode':'condition_naive1','condition_interval':k},row,spec,output=OUTPUT)
    write_json(OUTPUT/'completed'/f'{row}.json',dict(verdict='EVALUATION_COMPLETE',episodes=500,row=row,
        checkpoint_sha256=spec['sha256'],result=result));summarize()


def jobs():
    return [dict(id=row,cmd=[sys.executable,'-u','-m','tools.simvla.bounded_history_rb2','cell','--row',row],
        ready_file=str(INCOMING/row/ARM/'READY.json'),upstream_status_file=str(INCOMING/'pipeline_status.json'),
        summary=str(OUTPUT/'completed'/f'{row}.json'),completion=dict(verdict='EVALUATION_COMPLETE',episodes=500,row=row))
        for row in MODES]


def summarize():
    rows={row:read_json(OUTPUT/'completed'/f'{row}.json') for row in MODES if (OUTPUT/'completed'/f'{row}.json').exists()}
    write_json(OUTPUT/'comparison_summary.json',dict(rows=rows,complete=len(rows)==4,hardware='rb2 RTX5090 compiled',
        episodes_per_row=500,seed='seed01'))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',nargs='?',default='all',choices=('all','cell','smoke','worker'))
    p.add_argument('--preflight',action='store_true');p.add_argument('--output',type=Path);p.add_argument('--row',choices=MODES)
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
    if a.preflight:print('PREFLIGHT_PASS four matched compiled rows',flush=True);return 0
    record=OUTPUT.parent/'priority/drain_interval_recovery.json'
    if record.exists():finish(record)
    rc=run_queue(OUTPUT,plan,gpus=(0,),predecessor=[],environment=environment,cwd=ROOT,timeout=24*3600)
    summarize();return rc


if __name__=='__main__':raise SystemExit(main())
