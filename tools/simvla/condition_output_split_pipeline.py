"""Queue the matched pair after existing work; export each completed model."""
import argparse
from pathlib import Path
import subprocess

from methods.latentloop.modules.condition_output_split import ARMS
from tools.simvla.error_compensation_common import ROOT,read_json,write_json,environment,identity
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.observation_correction_pipeline import OUTPUT as SOURCE,export_arm

RESULTS=SOURCE.parents[1]
OUTPUT=RESULTS/'condition_output_split/mixed_k4_k8_seed01_v1'
PREDECESSOR=RESULTS/'condition_state_audit/frozen_observed_k8_seed01_v1'


def configuration():
    c=read_json(SOURCE/'runtime_config.json')
    c.update(output=str(OUTPUT),steps=3000,smoke_steps=14,smoke_policy_actions=41,
        training_k_c=8,
        warmup_steps=150,learning_rate=1e-4,condition_weight=0.05,
        evaluation_condition_intervals=[4,8],training_condition_ages=list(range(1,8)),
        rb2_destination='rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_condition_output_split',
        extra_source_files=['methods/latentloop/modules/condition_output_split.py',
            'tools/simvla/condition_output_split_train.py','tools/simvla/condition_output_split_eval.py',
            'tools/simvla/condition_output_split_pipeline.py','tools/simvla/condition_output_split_rb2.py',
            'architectures/simvla/wrappers/run_condition_output_split.sh'],
        training_description='Matched two-head models; same native150K initialization, zero added head output, same cache, seed, batch2 and 3K AdamW1e-4 cosine. Carry output or base only. Condition MSE on every unrolled base; final first5 continuous action L1 on original10 teacher, deployed naive3. No SR stopping gate.',
        student_condition_description='Previous predicted condition and current observation delta; input-dependent per-token extra action-condition update; only next-query routing differs.',
        evaluation_plan='sd1: two trainings, heldout geometry/action diagnostics, one K8 smoke per arm. rb2: each arm K4/K8, 500 LIBERO-Long episodes, seed01, compiled complete policy latency. Reuse existing baseline/LB; no repeated sd1 500-episode rows.')
    return c


def jobs(c,path):
    prefix=[c['python'],'-u','-m']; result=[]; run_id=identity(c)
    for arm in ARMS:
        def job(name,module,args,summary,completion,deps):
            result.append(dict(id=name,cmd=prefix+[module,'--config',str(path),'--arm',arm]+args,
                summary=str(summary),completion=dict(identity=run_id,**completion),deps=deps))
        smoke='smoke_train_'+arm
        job(smoke,'tools.simvla.condition_output_split_train',['--smoke'],OUTPUT/'smoke'/arm/'summary.json',
            dict(verdict='SMOKE_PASS',steps=14),[])
        env='smoke_env_'+arm
        job(env,'tools.simvla.condition_output_split_eval',['--smoke','--k-c','8'],
            OUTPUT/'eval_smoke'/f'kc8_{arm}'/'summary.json',dict(verdict='SMOKE_PASS',episodes=1),[smoke])
        training='train_'+arm
        job(training,'tools.simvla.condition_output_split_train',[],OUTPUT/'train'/arm/'summary.json',
            dict(verdict='TRAIN_AND_OFFLINE_COMPLETE',steps=3000),[env])
        job('export_'+arm,'tools.simvla.condition_output_split_pipeline',['--export'],
            OUTPUT/'exports'/f'{arm}.json',dict(verdict='BUNDLE_EXPORTED'),[training])
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    p.add_argument('--export',action='store_true'); p.add_argument('--config'); p.add_argument('--arm',choices=ARMS)
    a=p.parse_args()
    if a.export:
        export_arm(read_json(a.config),a.arm); return 0
    c=configuration(); prepare(c)
    path=OUTPUT/'runtime_config.json'; write_json(path,c)
    plan=jobs(c,path); write_json(OUTPUT/'planned_jobs.json',plan)
    if a.preflight:
        print('CPU_PREFLIGHT_PASS: matched pair, existing audit must release queue lock',flush=True); return 0
    rc=1
    try:
        rc=run_queue(OUTPUT,plan,gpus=(4,5,6,7),predecessor=dict(path=str(PREDECESSOR),lock='queue.lock'),
            environment=lambda gpu:environment(c,gpu),cwd=ROOT,timeout=12*3600)
    finally:
        status=OUTPUT/'pipeline_status.json'
        write_json(status,dict(phase='complete' if not rc else 'finished_with_failures'))
        host,remote=c['rb2_destination'].split(':',1)
        for attempt in range(2):
            try:
                subprocess.run(['ssh',host,'mkdir','-p',remote],check=True,timeout=30)
                subprocess.run(['rsync','-a',str(status),host+':'+remote+'/'],check=True,timeout=60)
                break
            except (subprocess.SubprocessError,OSError) as exc:
                print(f'STATUS_TRANSFER_WARNING attempt={attempt+1}: {exc}',flush=True)
    return rc


if __name__=='__main__': raise SystemExit(main())
