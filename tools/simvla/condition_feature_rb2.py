"""Unchanged inference graph for models trained with disposable feature readers."""
import argparse
import os
from pathlib import Path
import sys

from methods.latentloop.modules.condition_feature_supervision import MODES
from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT,configure,read_json,write_json,sha
from tools.simvla.compiled_policy import attach_policy
from tools.simvla.condition_noise_rb2 import OUTPUT as PREDECESSOR,INCOMING as PRIOR,base_config as previous_config
from tools.simvla.condition_noise_pipeline import ARM
from tools.simvla.condition_output_split_rb2 import STORAGE,environment,replay_factory,compiler_checker,reset_checker
from tools.simvla.condition_output_split_eval import load_payload,attach,check_policy
from tools.simvla.rollout_repair_rb2 import evaluate
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT=STORAGE/'results/simvla/condition_output_split/observation_feature_nfe1_compiled_seed01_v1'
INCOMING=STORAGE/'incoming/simvla_condition_feature'
ROWS={f'{mode}_k{k}':(mode,k) for k in (4,8) for mode in MODES}


def base_config():
    c=previous_config()
    c.update(output=str(OUTPUT),long_rows=list(ROWS),campaign_module='tools.simvla.condition_feature_rb2',
        extra_source_files=sorted(set(c['extra_source_files']+['tools/simvla/condition_feature_rb2.py'])),
        scope='Common fresh10K plus matched5K: control/current original image features/consecutive delta/both. Readers discarded;736130 deployed params,NFE1,K4/K8,500 pairedLong episodes each,compiledRTX5090.')
    return c


def ready_spec(mode):
    path=INCOMING/mode/ARM
    if read_json(path/'READY.json')['manifest_sha256']!=sha(path/'manifest.json'):
        raise RuntimeError('Non-atomic transfer')
    m=read_json(path/'manifest.json')
    if m['arm']!=ARM or m['step']!=5000 or sha(path/'model.pt')!=m['checkpoint_sha256']:
        raise RuntimeError('Transferred checkpoint mismatch')
    p=load_payload(path/'model.pt',ARM,m['source_identity'],steps=5000,action_mode='naive1')
    parent=read_json(PRIOR/'joint_noise1'/ARM/'manifest.json')
    c=p['contract']
    if (c['feature_alignment']['mode']!=mode or c['total_training_steps']!=15000
            or c['continuation']['checkpoint_sha256']!=parent['checkpoint_sha256']
            or c['action_gradient_mode']!='joint' or c['action_noise_samples']!=1):
        raise RuntimeError('Wrong feature training/source contract')
    return dict(path=str(path/'model.pt'),sha256=m['checkpoint_sha256'],arm=ARM,step=5000,
        source_identity=m['source_identity'],student_steps=1,feature_mode=mode)


def policy_factory(replay,c,row,manifest):
    mode,k=ROWS[row];spec=c['model_checkpoint']
    if spec['feature_mode']!=mode or sha(spec['path'])!=spec['sha256']:
        raise RuntimeError('Wrong checkpoint')
    payload=load_payload(spec['path'],ARM,spec['source_identity'],steps=5000,action_mode='naive1')
    policy=attach_policy(replay,c,'condition_naive3',manifest);policy.nfe=1
    policy=attach(policy,replay.native,payload,ARM,k,replay.compiler)
    if any('feature_reader' in n for n,_ in policy.native_v0.named_modules()):
        raise RuntimeError('Training-only reader leaked into inference')
    return policy


def cell(c,row):
    mode,k=ROWS[row];spec=ready_spec(mode)
    result=evaluate({**c,'student_steps':1,'action_mode':'condition_naive1','condition_interval':k},row,spec,output=OUTPUT)
    write_json(OUTPUT/'completed'/f'{row}.json',dict(verdict='EVALUATION_COMPLETE',episodes=500,
        row=row,checkpoint_sha256=spec['sha256'],result=result))


def jobs():
    return [dict(id=row,cmd=[sys.executable,'-u','-m','tools.simvla.condition_feature_rb2','cell','--row',row],
        upstream_status_file=str(INCOMING/'pipeline_status.json'),ready_file=str(INCOMING/mode/ARM/'READY.json'),
        summary=str(OUTPUT/'completed'/f'{row}.json'),
        completion=dict(verdict='EVALUATION_COMPLETE',episodes=500,row=row)) for row,(mode,_) in ROWS.items()]


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('all','cell','smoke','worker'),default='all',nargs='?')
    p.add_argument('--preflight',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--row',choices=ROWS);p.add_argument('--suite',default='libero_10',choices=('libero_10',))
    p.add_argument('--seed',default='seed01',choices=('seed01',));a=p.parse_args()
    env=environment(0);os.environ.clear();os.environ.update(env)
    c=read_json(a.output/'runtime_config.json') if a.output else base_config()
    configure(c);sys.path.insert(0,c['libero_root']);os.environ['LIBERO_CONFIG_PATH']=c['libero_config']
    if a.command in ('smoke','worker'):
        campaign.worker(c,a.output,a.suite,a.seed,a.row,smoke=a.command=='smoke',
            replay_factory=replay_factory,policy_factory=policy_factory,
            policy_checker=lambda policy,row:check_policy(policy),compiler_checker=compiler_checker,reset_checker=reset_checker)
        return 0
    if a.command=='cell':
        cell(c,a.row)
        return 0
    campaign.prepare(c,OUTPUT);write_json(OUTPUT/'runtime_config.json',c)
    plan=jobs();write_json(OUTPUT/'planned_jobs.json',plan)
    if a.preflight:
        print('PREFLIGHT_PASS:8 new rows after existing fresh-noise queue; inference reader excluded',flush=True)
        return 0
    rc=run_queue(OUTPUT,plan,gpus=(0,),predecessor=dict(path=str(PREDECESSOR),lock='queue.lock'),
        environment=environment,cwd=ROOT,timeout=24*3600)
    rows={r:read_json(OUTPUT/'completed'/f'{r}.json') for r in ROWS if (OUTPUT/'completed'/f'{r}.json').exists()}
    write_json(OUTPUT/'comparison_summary.json',dict(complete=len(rows)==len(ROWS),rows=rows,scope=c['scope']))
    return rc


if __name__=='__main__':
    raise SystemExit(main())
