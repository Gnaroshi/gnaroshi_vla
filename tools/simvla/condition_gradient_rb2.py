"""Complete K4/K8 evaluations of joint-gradient NFE1/NFE2 models."""
import argparse
import os
from pathlib import Path
import sys

from methods.latentloop.modules.condition_output_split import ARMS
from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT, configure, read_json, write_json, sha
from tools.simvla.compiled_policy import attach_policy
from tools.simvla.condition_solver_rb2 import solver_paths, base_config as previous_config
from tools.simvla.condition_output_split_rb2 import STORAGE, environment, replay_factory, compiler_checker, reset_checker
from tools.simvla.condition_output_split_eval import load_payload, attach, check_policy
from tools.simvla.condition_output_split_train import check_gradient_control
from tools.simvla.rollout_repair_rb2 import evaluate
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT=STORAGE/'results/simvla/condition_output_split/joint_action_gradient_compiled_seed01_v1'
INCOMING=STORAGE/'incoming/simvla_condition_gradient'
ROWS={f'nfe{nfe}_{arm}_k{k}':(nfe,arm,k) for k in (4,8) for nfe in (1,2) for arm in ARMS}


def base_config():
    c=previous_config(2)
    c.update(output=str(OUTPUT),long_rows=list(ROWS),campaign_module='tools.simvla.condition_gradient_rb2',
        extra_source_files=c['extra_source_files']+['tools/simvla/condition_gradient_rb2.py'],
        scope='Only action gradient routing changes versus matched detached controls; NFE1/2 x carried condition x K4/K8. Each row 500 paired LIBERO-Long seed01 episodes, compiled total policy latency, H10/R5; no SR gate.')
    return c


def transferred(directory,arm,nfe):
    if read_json(directory/'READY.json')['manifest_sha256']!=sha(directory/'manifest.json'):
        raise RuntimeError('Non-atomic transfer')
    m=read_json(directory/'manifest.json')
    if m['arm']!=arm or m['step']!=7000 or sha(directory/'model.pt')!=m['checkpoint_sha256']:
        raise RuntimeError('Transferred checkpoint mismatch')
    payload=load_payload(directory/'model.pt',arm,m['source_identity'],steps=7000,action_mode=f'naive{nfe}')
    return m,payload


def ready_spec(nfe,arm):
    directory=INCOMING/f'nfe{nfe}'/arm
    m,p=transferred(directory,arm,nfe)
    control_dir=solver_paths(nfe)[1]/'pretrained_10k'/arm
    control_m,control=transferred(control_dir,arm,nfe)
    check_gradient_control(p['contract'],control['contract'])
    if p['contract'].get('gradient_transition')!='detached_to_joint':
        raise RuntimeError('Missing gradient transition provenance')
    return dict(path=str(directory/'model.pt'),sha256=m['checkpoint_sha256'],arm=arm,step=7000,
        total_training_steps=10000,source_identity=m['source_identity'],student_steps=nfe,
        action_gradient_mode='joint',matched_detached_checkpoint_sha256=control_m['checkpoint_sha256'])


def policy_factory(replay,c,row,manifest):
    nfe,arm,k=ROWS[row]; spec=c['model_checkpoint']
    if spec['student_steps']!=nfe or c['student_steps']!=nfe or sha(spec['path'])!=spec['sha256']:
        raise RuntimeError('Checkpoint/deployment mismatch')
    p=load_payload(spec['path'],arm,spec['source_identity'],steps=7000,action_mode=f'naive{nfe}')
    if p['contract'].get('action_gradient_mode')!='joint':
        raise RuntimeError('Wrong gradient model')
    policy=attach_policy(replay,c,'condition_naive3',manifest)
    policy.nfe=nfe
    return attach(policy,replay.native,p,arm,k,replay.compiler)


def cell(c,row):
    nfe,arm,k=ROWS[row]; spec=ready_spec(nfe,arm)
    result=evaluate({**c,'student_steps':nfe,'action_mode':f'condition_naive{nfe}',
        'condition_interval':k},row,spec,output=OUTPUT)
    write_json(OUTPUT/'completed'/f'{row}.json',dict(verdict='EVALUATION_COMPLETE',episodes=500,
        row=row,checkpoint_sha256=spec['sha256'],result=result))


def jobs():
    return [dict(id=row,cmd=[sys.executable,'-u','-m','tools.simvla.condition_gradient_rb2','cell','--row',row],
        upstream_status_file=str(INCOMING/'pipeline_status.json'),
        ready_file=str(INCOMING/f'nfe{nfe}'/arm/'READY.json'),
        summary=str(OUTPUT/'completed'/f'{row}.json'),
        completion=dict(verdict='EVALUATION_COMPLETE',episodes=500,row=row))
        for row,(nfe,arm,k) in ROWS.items()]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('all','cell','smoke','worker'),default='all',nargs='?')
    p.add_argument('--preflight',action='store_true'); p.add_argument('--output',type=Path)
    p.add_argument('--row',choices=ROWS); p.add_argument('--suite',default='libero_10',choices=('libero_10',))
    p.add_argument('--seed',default='seed01',choices=('seed01',)); a=p.parse_args()
    env=environment(0); os.environ.clear(); os.environ.update(env)
    c=read_json(a.output/'runtime_config.json') if a.output else base_config()
    configure(c); sys.path.insert(0,c['libero_root']); os.environ['LIBERO_CONFIG_PATH']=c['libero_config']
    if a.command in ('smoke','worker'):
        campaign.worker(c,a.output,a.suite,a.seed,a.row,smoke=a.command=='smoke',
            replay_factory=replay_factory,policy_factory=policy_factory,
            policy_checker=lambda policy,row:check_policy(policy),
            compiler_checker=compiler_checker,reset_checker=reset_checker)
        return 0
    if a.command=='cell': cell(c,a.row); return 0
    campaign.prepare(c,OUTPUT); write_json(OUTPUT/'runtime_config.json',c)
    plan=jobs(); write_json(OUTPUT/'planned_jobs.json',plan)
    if a.preflight:
        print('CPU_PREFLIGHT_PASS: eight joint-gradient rows, existing controls reused',flush=True)
        return 0
    rc=run_queue(OUTPUT,plan,gpus=(0,),predecessor=dict(path=str(solver_paths(2)[0]),lock='queue.lock'),
        environment=environment,cwd=ROOT,timeout=24*3600)
    rows={r:read_json(OUTPUT/'completed'/f'{r}.json') for r in ROWS if (OUTPUT/'completed'/f'{r}.json').exists()}
    write_json(OUTPUT/'comparison_summary.json',dict(complete=len(rows)==len(ROWS),rows=rows,scope=c['scope']))
    return rc


if __name__=='__main__':
    raise SystemExit(main())
