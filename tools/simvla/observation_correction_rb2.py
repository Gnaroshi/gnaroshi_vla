"""Consume atomic per-model bundles, then compiled LIBERO-Long K4/K8 rows."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

from methods.latentloop.modules.observation_correction import ARMS
from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import Replay,configure,read_json,write_json,sha
from tools.simvla.compiled_policy import attach_policy,check_reset
from tools.simvla.rollout_repair_rb2 import base_config as previous_config,evaluate,STORAGE
from tools.simvla.rollout_round2_rb2 import wait_idle
from tools.simvla.observation_correction_eval import load_payload,attach,check_policy

OUTPUT=STORAGE/'results/simvla/observation_correction/mixed_k4_k8_compiled_seed01_v1'
INCOMING=STORAGE/'incoming/simvla_observation_correction'
ROWS={f'{arm}_k{k}':(arm,k) for k in (8,4) for arm in ARMS}


def base_config():
    c=previous_config()
    c.update(output=str(OUTPUT),long_rows=list(ROWS),seeds=['seed01'],other_rows=[],other_suites=[],
        smoke_episodes=1,smoke_actions=41,warmup_actions=40,
        campaign_module='tools.simvla.observation_correction_rb2',
        extra_source_files=c['extra_source_files']+['tools/simvla/observation_correction_rb2.py',
            'tools/simvla/observation_correction_eval.py','methods/latentloop/modules/observation_correction.py'],
        scope='Four mixed-K4/K8 trained Condition architectures, same naive3 action solver; seed01 development, 500 episodes/row, original H10/R5 and paired manifest. Extra operations included in policy latency. No SR stopping gate.')
    return c


def ready_spec(arm):
    directory=INCOMING/arm
    if not (directory/'READY.json').exists(): return None
    if read_json(directory/'READY.json')['manifest_sha256']!=sha(directory/'manifest.json'):
        raise RuntimeError('Non-atomic transfer')
    m=read_json(directory/'manifest.json')
    if m['arm']!=arm or m['step']!=3000 or sha(directory/'model.pt')!=m['checkpoint_sha256']:
        raise RuntimeError('Transferred model differs')
    load_payload(directory/'model.pt',arm,expected_identity=m['source_identity'])
    return dict(path=str(directory/'model.pt'),sha256=m['checkpoint_sha256'],arm=arm,step=3000,
        source_identity=m['source_identity'])


def replay_factory(c,row,compiler,samples):
    return Replay(c,'condition_naive3',compiler,samples)


def policy_factory(replay,c,row,manifest):
    arm,k=ROWS[row]; spec=c['model_checkpoint']
    if sha(spec['path'])!=spec['sha256']: raise RuntimeError('Checkpoint changed')
    payload=load_payload(spec['path'],arm,expected_identity=spec['source_identity'])
    policy=attach_policy(replay,c,'condition_naive3',manifest)
    return attach(policy,replay.native,payload,arm,k,replay.compiler)


def compiler_checker(compiler,row):
    arm,_=ROWS[row]
    required={'vlm','action_transformer','observation_encoder','condition_updater'}
    if arm!='recurrent': required.add('trend_head')
    if arm=='observed_recurrent': required.update(('measurement_head','correction_gain'))
    missing=[name for name in required if not compiler.records.get(name,{}).get('graphs',0)]
    if missing: raise RuntimeError('Compile bypass: '+str(missing))


def reset_checker(policy):
    check_reset(policy)
    if policy._trend_context is not None or policy._condition_component_calls:
        raise RuntimeError('Recurrent context leaked across episodes')


def run_all(c):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        campaign.prepare(c,OUTPUT)
        remaining=list(ROWS); results=[]; failures=[]
        while remaining:
            available=[row for row in remaining if (INCOMING/ROWS[row][0]/'READY.json').exists()]
            if not available:
                write_json(OUTPUT/'pipeline_status.json',dict(phase='waiting_for_sd1_model',remaining=remaining,gpu_used=False))
                print('WAIT: first completed Condition checkpoint; no GPU allocation',flush=True)
                time.sleep(30); continue
            row=available[0]; arm,k=ROWS[row]
            try:
                spec=ready_spec(arm)
                write_json(OUTPUT/'pipeline_status.json',dict(phase='waiting_for_gpu',row=row)); wait_idle()
                write_json(OUTPUT/'pipeline_status.json',dict(phase='evaluation',row=row))
                result=evaluate({**c,'action_mode':'condition_naive3','condition_interval':k},row,spec,output=OUTPUT)
                results.append(dict(row=row,**result))
                print(f"DONE {row}: {result['successes']}/500; {result['pooled_policy_ms_per_action']:.4f} ms/action",flush=True)
            except Exception as exc:
                failures.append(dict(row=row,error=str(exc))); print(f'ROW_FAILED {row}: {exc}',flush=True)
            remaining.remove(row)
            write_json(OUTPUT/'combined_summary.json',dict(complete=len(results)==len(ROWS),results=results,
                failures=failures,remaining=remaining,scope=c['scope']))
        write_json(OUTPUT/'pipeline_status.json',dict(phase='complete' if not failures else 'finished_with_failures',failures=failures))
        return int(bool(failures))


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('all','preflight','smoke','worker'),nargs='?',default='all')
    p.add_argument('--output',type=Path); p.add_argument('--row',choices=ROWS)
    p.add_argument('--suite',default='libero_10',choices=('libero_10',)); p.add_argument('--seed',default='seed01',choices=('seed01',))
    a=p.parse_args(); c=read_json(a.output/'runtime_config.json') if a.output else base_config()
    configure(c); sys.path.insert(0,c['libero_root']); os.environ['LIBERO_CONFIG_PATH']=c['libero_config']
    if a.command=='preflight': campaign.prepare(c,OUTPUT); print('CPU_PREFLIGHT_PASS',flush=True); return 0
    if a.command=='all': return run_all(c)
    campaign.worker(c,a.output,a.suite,a.seed,a.row,smoke=a.command=='smoke',replay_factory=replay_factory,
        policy_factory=policy_factory,policy_checker=lambda policy,row:check_policy(policy),
        compiler_checker=compiler_checker,reset_checker=reset_checker)
    return 0


if __name__=='__main__': raise SystemExit(main())
