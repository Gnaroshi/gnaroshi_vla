"""Compiled matched-pair evaluation after the existing rb2 campaign."""
import argparse
import os
from pathlib import Path
import sys

from methods.latentloop.modules.condition_output_split import ARMS
from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT,Replay,configure,read_json,write_json,sha
from tools.simvla.compiled_policy import attach_policy,check_reset
from tools.simvla.rollout_repair_rb2 import base_config as previous_config,evaluate,STORAGE
from tools.simvla.condition_output_split_eval import load_payload,attach,check_policy
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT=STORAGE/'results/simvla/condition_output_split/mixed_k4_k8_compiled_seed01_v1'
INCOMING=STORAGE/'incoming/simvla_condition_output_split'
PREDECESSOR=STORAGE/'results/simvla/libero_pro/long_position_task_seed01_v1'
ROWS={f'{arm}_k{k}':(arm,k) for k in (4,8) for arm in ARMS}


def base_config():
    c=previous_config()
    c.update(output=str(OUTPUT),long_rows=list(ROWS),seeds=['seed01'],other_rows=[],other_suites=[],
        smoke_episodes=1,smoke_actions=41,warmup_actions=40,
        campaign_module='tools.simvla.condition_output_split_rb2',
        extra_source_files=c['extra_source_files']+['tools/simvla/condition_output_split_rb2.py',
            'tools/simvla/condition_output_split_eval.py','methods/latentloop/modules/condition_output_split.py'],
        scope='Matched two-head models: carry final action condition or carry pre-addition condition. Same native150K initialization, 3K training and naive3; each K4/K8 row uses 500 LIBERO-Long episodes, seed01, original H10/R5 and paired manifest. Compile includes both heads. No SR stopping gate.')
    return c


def ready_spec(arm):
    directory=INCOMING/arm
    if read_json(directory/'READY.json')['manifest_sha256']!=sha(directory/'manifest.json'):
        raise RuntimeError('Non-atomic transfer')
    m=read_json(directory/'manifest.json')
    if m['arm']!=arm or m['step']!=3000 or sha(directory/'model.pt')!=m['checkpoint_sha256']:
        raise RuntimeError('Transferred checkpoint mismatch')
    load_payload(directory/'model.pt',arm,m['source_identity'])
    return dict(path=str(directory/'model.pt'),sha256=m['checkpoint_sha256'],arm=arm,step=3000,
        source_identity=m['source_identity'])


def replay_factory(c,row,compiler,samples):
    return Replay(c,'condition_naive3',compiler,samples)


def policy_factory(replay,c,row,manifest):
    arm,k=ROWS[row]; spec=c['model_checkpoint']
    if sha(spec['path'])!=spec['sha256']: raise RuntimeError('Checkpoint changed')
    payload=load_payload(spec['path'],arm,spec['source_identity'])
    policy=attach_policy(replay,c,'condition_naive3',manifest)
    return attach(policy,replay.native,payload,arm,k,replay.compiler)


def compiler_checker(compiler,row):
    required={'vlm','action_transformer','observation_encoder','condition_updater','action_condition_updater'}
    missing=[key for key in required if not compiler.records.get(key,{}).get('graphs',0)]
    if missing: raise RuntimeError('Compile bypass: '+str(missing))


def reset_checker(policy):
    check_reset(policy)
    if policy._split_context is not None or policy._condition_component_calls:
        raise RuntimeError('Condition state leaked across episodes')


def cell(c,row):
    arm,k=ROWS[row]; spec=ready_spec(arm)
    result=evaluate({**c,'action_mode':'condition_naive3','condition_interval':k},row,spec,output=OUTPUT)
    write_json(OUTPUT/'completed'/f'{row}.json',dict(verdict='EVALUATION_COMPLETE',episodes=500,
        row=row,checkpoint_sha256=spec['sha256'],result=result))


def environment(gpu):
    if gpu!=0: raise ValueError('rb2 GPU must be zero')
    env=dict(os.environ)
    for key in ('GALLIUM_DRIVER','LIBGL_ALWAYS_SOFTWARE','EGL_DEVICE_ID','LP_NUM_THREADS'):
        env.pop(key,None)
    env.update(CUDA_VISIBLE_DEVICES='0',MUJOCO_EGL_DEVICE_ID='0',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
        USE_TF='0',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(ROOT))
    return env


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('all','cell','smoke','worker'),default='all',nargs='?')
    p.add_argument('--preflight',action='store_true'); p.add_argument('--output',type=Path)
    p.add_argument('--row',choices=ROWS); p.add_argument('--suite',default='libero_10',choices=('libero_10',))
    p.add_argument('--seed',default='seed01',choices=('seed01',)); a=p.parse_args()
    c=read_json(a.output/'runtime_config.json') if a.output else base_config()
    configure(c); sys.path.insert(0,c['libero_root']); os.environ['LIBERO_CONFIG_PATH']=c['libero_config']
    if a.command in ('smoke','worker'):
        campaign.worker(c,a.output,a.suite,a.seed,a.row,smoke=a.command=='smoke',replay_factory=replay_factory,
            policy_factory=policy_factory,policy_checker=lambda policy,row:check_policy(policy),
            compiler_checker=compiler_checker,reset_checker=reset_checker)
        return 0
    if a.command=='cell': cell(c,a.row); return 0
    campaign.prepare(c,OUTPUT); write_json(OUTPUT/'runtime_config.json',c)
    jobs=[dict(id=row,cmd=[sys.executable,'-u','-m','tools.simvla.condition_output_split_rb2','cell','--row',row],
        upstream_status_file=str(INCOMING/'pipeline_status.json'),
        ready_file=str(INCOMING/arm/'READY.json'),summary=str(OUTPUT/'completed'/f'{row}.json'),
        completion=dict(verdict='EVALUATION_COMPLETE',episodes=500,row=row)) for row,(arm,k) in ROWS.items()]
    write_json(OUTPUT/'planned_jobs.json',jobs)
    if a.preflight:
        print('CPU_PREFLIGHT_PASS: wait for LIBERO-PRO queue and atomic model bundles',flush=True); return 0
    rc=run_queue(OUTPUT,jobs,gpus=(0,),predecessor=dict(path=str(PREDECESSOR),lock='queue.lock'),
        environment=environment,cwd=ROOT,timeout=16*3600)
    rows={row:read_json(OUTPUT/'completed'/f'{row}.json') for row in ROWS
        if (OUTPUT/'completed'/f'{row}.json').exists()}
    write_json(OUTPUT/'comparison_summary.json',dict(complete=len(rows)==4,rows=rows,scope=c['scope']))
    return rc


if __name__=='__main__': raise SystemExit(main())
