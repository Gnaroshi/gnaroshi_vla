"""Evaluate repaired interval continuations and matched fresh starts."""
import argparse
import os
from pathlib import Path
import sys

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT,configure,read_json,write_json,sha
from tools.simvla.compiled_policy import attach_policy
from tools.simvla.condition_deployment_rb2 import OUTPUT as PRIOR,base_config as previous_config
from tools.simvla.condition_deployment_pipeline import PARENT_SHA
from tools.simvla.condition_interval_recovery import MODES,ARM
from tools.simvla.condition_output_split_rb2 import environment,replay_factory,compiler_checker,reset_checker
from tools.simvla.condition_output_split_eval import load_payload,attach,check_policy
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.rollout_repair_rb2 import STORAGE,evaluate

OUTPUT=PRIOR.parent/'interval_recovery_nfe1_compiled_seed01_v1'
INCOMING=STORAGE/'incoming/simvla_interval_recovery'
CONTROL=STORAGE/'incoming/simvla_condition_noise/detached_noise1/carry_base'
ROWS={mode:(mode,k) for mode,(_,k) in MODES.items()}
ROWS.update({f'fresh_mixed_k{k}':('fresh_mixed_control',k) for k in (2,3,4)})


def base_config():
    c=previous_config()
    c.update(output=str(OUTPUT),long_rows=list(ROWS),campaign_module='tools.simvla.condition_interval_recovery_rb2',
        extra_source_files=c['extra_source_files']+['tools/simvla/condition_interval_recovery.py','tools/simvla/condition_interval_recovery_rb2.py'],
        scope='Three fixed-interval5K continuations and three fresh NFE1-only10K specialists. Existing fresh mixed10K weights reused. Same8-query cache windows; prefix lengths vary. Nine Long500 seed01 compiled rows. H10/R5, no Generation Loop. No SR gate.')
    return c


def ready_spec(mode):
    directory=CONTROL if mode=='fresh_mixed_control' else INCOMING/mode/ARM
    if read_json(directory/'READY.json')['manifest_sha256']!=sha(directory/'manifest.json'):
        raise RuntimeError('Model transfer incomplete')
    m=read_json(directory/'manifest.json')
    fresh=mode.startswith('fresh');steps=10000 if fresh else 5000
    if m['step']!=steps or m['arm']!=ARM or m['parameters']!=736130 or sha(directory/'model.pt')!=m['checkpoint_sha256']:
        raise RuntimeError('Model specification changed')
    p=load_payload(directory/'model.pt',ARM,m['source_identity'],steps=steps,action_mode='naive1')
    c=p['contract'];intervals=[4,8] if mode=='fresh_mixed_control' else [MODES[mode][1]]
    if c['training_intervals']!=intervals or c['teacher_steps']!=10 or c['seed']!=7 or c['batch_size']!=2:
        raise RuntimeError('Interval/objective mismatch')
    if fresh:
        if c['initialization']!='fresh' or c['total_training_steps']!=10000 or c['sample_step_offset']!=0:
            raise RuntimeError('Fresh training budget mismatch')
    elif c['continuation']['checkpoint_sha256']!=PARENT_SHA or c['total_training_steps']!=15000:
        raise RuntimeError('Continuation parent/budget mismatch')
    if c['current_action_gradient']!='Only action_condition_updater; base and observation feature detached at its input':
        raise RuntimeError('Action gradient routing mismatch')
    return dict(path=str(directory/'model.pt'),sha256=m['checkpoint_sha256'],source_identity=m['source_identity'],
        step=steps,training_intervals=intervals,mode=mode,student_steps=1)


def policy_factory(replay,c,row,manifest):
    mode,k=ROWS[row];s=c['model_checkpoint']
    if sha(s['path'])!=s['sha256']:raise RuntimeError('Checkpoint changed')
    p=load_payload(s['path'],ARM,s['source_identity'],steps=s['step'],action_mode='naive1')
    policy=attach_policy(replay,c,'condition_naive3',manifest);policy.NFE=policy.nfe=1
    policy=attach(policy,replay.native,p,ARM,k,replay.compiler);policy.row_name=row
    return policy


def cell(c,row):
    mode,k=ROWS[row];s=ready_spec(mode)
    result=evaluate({**c,'action_mode':'condition_naive1','condition_interval':k},row,s,output=OUTPUT)
    write_json(OUTPUT/'completed'/f'{row}.json',dict(verdict='EVALUATION_COMPLETE',episodes=500,row=row,
        checkpoint_sha256=s['sha256'],result=result))
    summarize()


def jobs():
    plan=[]
    for row,(mode,k) in ROWS.items():
        directory=CONTROL if mode=='fresh_mixed_control' else INCOMING/mode/ARM
        job=dict(id=row,cmd=[sys.executable,'-u','-m','tools.simvla.condition_interval_recovery_rb2','cell','--row',row],
            ready_file=str(directory/'READY.json'),summary=str(OUTPUT/'completed'/f'{row}.json'),
            completion=dict(verdict='EVALUATION_COMPLETE',episodes=500,row=row))
        if mode!='fresh_mixed_control':job['upstream_status_file']=str(INCOMING/'pipeline_status.json')
        plan.append(job)
    return plan


def summarize():
    rows={row:read_json(OUTPUT/'completed'/f'{row}.json') for row in ROWS if (OUTPUT/'completed'/f'{row}.json').exists()}
    refs=read_json(PRIOR.parent/'bridge_interval_nfe1_compiled_seed01_v1/comparison_summary.json')
    write_json(OUTPUT/'comparison_summary.json',dict(complete=len(rows)==len(ROWS),rows=rows,
        original_parent_and_bridge=refs,hardware='rb2 RTX5090 compiled',seed='seed01',episodes_per_row=500))


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
    ready_spec('fresh_mixed_control');campaign.prepare(c,OUTPUT);write_json(OUTPUT/'runtime_config.json',c)
    plan=jobs();write_json(OUTPUT/'planned_jobs.json',plan);summarize()
    if a.preflight:print('PREFLIGHT_PASS: three recovered models, three fresh specialists, reused fresh control',flush=True);return 0
    rc=run_queue(OUTPUT,plan,gpus=(0,),predecessor=dict(path=str(PRIOR),lock='queue.lock'),
        environment=environment,cwd=ROOT,timeout=24*3600)
    summarize();return rc


if __name__=='__main__':raise SystemExit(main())
