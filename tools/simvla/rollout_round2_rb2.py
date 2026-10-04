"""Compiled follow-up controls, then the predeclared second-round 2x2."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT,Replay,configure,read_json,write_json,sha
from tools.simvla.rollout_repair_rb2 import base_config as previous_config,make_policy,evaluate,STORAGE
from tools.simvla.trend_compiled_rb2 import reset_policy

OUTPUT=STORAGE/'results/simvla/trend_condition/rollout_round2_compiled_seed01_v1'
FIRST=STORAGE/'incoming/simvla_rollout_state_repair'
SECOND=STORAGE/'incoming/simvla_rollout_round2'
VARIANTS=('previous_fixed','previous_joint','aggregate_fixed','aggregate_joint')
ROWS={'repair_k4_generation3':(4,'ours_kc2_ng3'),
      'repair_k8_naive3':(8,'condition_naive3'),
      'repair_k8_full10':(8,'condition_nfe10'),
      **{v+'_k8':(8,'ours_kc2_ng3') for v in VARIANTS}}


def expected_counts(row,q):
    k,mode=ROWS[row]; full=(q+k-1)//k
    return dict(num_full_vlm_calls=full,num_condition_updater_calls=q-full,
        num_action_transformer_calls=q*(10 if mode=='condition_nfe10' else 3),
        num_generation_decoder_only_steps=7*q if mode=='ours_kc2_ng3' else 0,
        num_trend_head_calls=full,num_observation_encoder_calls=q-full)


def check_policy(policy,row):
    q=int(policy.metrics.counters['num_policy_queries'])
    for name,value in expected_counts(row,q).items():
        if int(policy.metrics.counters.get(name,0))!=value: raise RuntimeError(f'{row}: {name} mismatch')
    if q!=(policy.step_index+4)//5: raise RuntimeError('H10/R5 cadence changed')


def check_compiler(compiler,row):
    required={'vlm','action_transformer','trend_head','observation_encoder','condition_updater'}
    if ROWS[row][1]=='ours_kc2_ng3': required|={'action_decoder','generation_updater'}
    missing=[k for k in required if not compiler.records.get(k,{}).get('graphs',0)]
    if missing: raise RuntimeError('Compile bypass: '+str(missing))


def replay_factory(c,row,compiler,samples): return Replay(c,ROWS[row][1],compiler,samples)


def base_config():
    c=previous_config()
    c.update(output=str(OUTPUT),long_rows=list(ROWS),campaign_module='tools.simvla.rollout_round2_rb2',
        extra_source_files=c['extra_source_files']+['tools/simvla/rollout_round2_rb2.py'],
        scope='Development seed01; first three rows isolate execution changes without retraining; remaining four are matched 3K continuations of round1 repair. No final three-seed claim.')
    return c


def bundle(directory):
    if not (directory/'READY.json').exists(): return None
    if read_json(directory/'READY.json')['manifest_sha256']!=sha(directory/'manifest.json'): raise RuntimeError('Bundle not atomic')
    result=read_json(directory/'manifest.json')
    for spec in result['checkpoints'].values():
        if sha(directory/spec['file'])!=spec['sha256']: raise RuntimeError('Transfer hash mismatch')
    return result


def wait_idle():
    while True:
        text=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        memory,util=map(int,text.strip().split(','))
        if memory<600 and util<5: return
        print(f'WAIT_GPU memory={memory}MiB utilization={util}%',flush=True); time.sleep(30)


def run_all(c):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results=[]; failures=[]
        for row,(k,mode) in ROWS.items():
            directory=FIRST if row.startswith('repair_') else SECOND
            while not (ready:=bundle(directory)):
                write_json(OUTPUT/'pipeline_status.json',dict(phase='waiting_for_sd1_bundle',row=row,gpu_used=False))
                print('WAIT: sd1 second-round checkpoints; no GPU allocation',flush=True); time.sleep(60)
            key='rollout_repair' if row.startswith('repair_') else row.removesuffix('_k8')
            item=ready['checkpoints'][key]
            spec=dict(path=str(directory/item['file']),sha256=item['sha256'],arm=ready['selected_arm'],step=3000)
            try:
                write_json(OUTPUT/'pipeline_status.json',dict(phase='waiting_for_gpu',row=row))
                wait_idle()
                write_json(OUTPUT/'pipeline_status.json',dict(phase='evaluation',row=row))
                result=evaluate({**c,'action_mode':mode,'condition_interval':k},row,spec,output=OUTPUT)
                results.append(dict(row=row,**result))
            except Exception as exc:
                failures.append(dict(row=row,error=str(exc))); print(f'ROW_FAILED {row}: {exc}',flush=True)
            write_json(OUTPUT/'combined_summary.json',dict(complete=len(results)==len(ROWS),results=results,failures=failures))
        write_json(OUTPUT/'pipeline_status.json',dict(phase='complete' if not failures else 'finished_with_failures',failures=failures))
        return int(bool(failures))


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('all','preflight','smoke','worker'),default='all',nargs='?')
    p.add_argument('--output',type=Path); p.add_argument('--suite',default='libero_10',choices=('libero_10',))
    p.add_argument('--seed',default='seed01',choices=('seed01',)); p.add_argument('--row',choices=ROWS)
    a=p.parse_args(); c=read_json(a.output/'runtime_config.json') if a.output else base_config()
    configure(c); sys.path.insert(0,c['libero_root']); os.environ['LIBERO_CONFIG_PATH']=c['libero_config']
    if a.command=='preflight':
        from tools.simvla.compile_benchmark import preflight
        preflight(c)
        if not bundle(FIRST): raise RuntimeError('First-round weights missing')
        print('PREFLIGHT_PASS',flush=True); return 0
    if a.command=='all':
        try: return run_all(c)
        except BaseException as exc:
            write_json(OUTPUT/'pipeline_status.json',dict(phase='failed',error=str(exc))); raise
    campaign.worker(c,a.output,a.suite,a.seed,a.row,smoke=a.command=='smoke',
        replay_factory=replay_factory,policy_factory=make_policy,policy_checker=check_policy,
        compiler_checker=check_compiler,reset_checker=reset_policy)
    return 0


if __name__=='__main__': raise SystemExit(main())
