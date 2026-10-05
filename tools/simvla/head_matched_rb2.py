"""Wait for current rb2 controls, then evaluate all four matched solver models."""
import argparse
import csv
import fcntl
from functools import partial
import os
from pathlib import Path
import sys
import time

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import configure,read_json,write_json
from tools.simvla.rollout_repair_rb2 import base_config as previous_config,make_policy,evaluate,STORAGE
from tools.simvla.rollout_round2_rb2 import bundle,wait_idle
from tools.simvla import interval_followup_rb2 as controls
from tools.simvla.trend_compiled_rb2 import reset_policy

OUTPUT=STORAGE/'results/simvla/trend_condition/head_matched_compiled_seed01_v1'
PREDECESSOR=STORAGE/'results/simvla/trend_condition/post_bridge_controls_seed01_v1'
INCOMING=STORAGE/'incoming/simvla_head_matched'
ROWS={f'k{k}_{name}':(k,mode) for k in (4,8)
      for name,mode in (('learned','ours_kc2_ng3'),('naive3','condition_naive3'))}


def base_config():
    c=previous_config()
    c.update(output=str(OUTPUT),long_rows=list(ROWS),other_rows=[],other_suites=[],seeds=['seed01'],
        campaign_module='tools.simvla.head_matched_rb2',smoke_actions=41,warmup_actions=40,
        extra_source_files=c['extra_source_files']+['tools/simvla/head_matched_rb2.py',
            'tools/simvla/interval_followup_rb2.py','tools/simvla/rollout_round2_rb2.py'],
        scope='Development seed01. Four equal-budget Condition continuations from previous_joint: K4/K8 x learned3/naive3. Each evaluated with its training solver. Original10 teacher; H10/R5; 500 episodes each; no SR gate.')
    return c


def checkpoint_spec(ready,row):
    import torch
    k,mode=ROWS[row]
    settings=ready['variant_settings'][row]
    expected_mode='learned' if mode=='ours_kc2_ng3' else 'naive3'
    if (settings['generation_mode']!=expected_mode or settings['training_k_c']!=k
            or not settings['train_trend'] or ready['selected_arm']!='frozen_trend_residual'):
        raise RuntimeError('Transferred training mode/interval/architecture differs')
    item=ready['checkpoints'][row]
    if item['step']!=3000: raise RuntimeError('Unexpected training budget')
    payload=torch.load(INCOMING/item['file'],map_location='cpu',weights_only=False)
    if (payload['identity']!=ready['source_identity'] or payload['repair_variant']!=row
            or payload['contract']['training_data']!=settings or payload['step']!=3000):
        raise RuntimeError('Checkpoint payload and manifest disagree')
    return dict(path=str(INCOMING/item['file']),sha256=item['sha256'],arm=ready['selected_arm'],step=3000)


def predecessor_busy():
    with (PREDECESSOR/'pipeline.lock').open('r') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return True
    return False


def run_all(c):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while predecessor_busy():
            write_json(OUTPUT/'pipeline_status.json',dict(phase='waiting_for_predecessor',gpu_used=False,predecessor=str(PREDECESSOR)))
            print('WAIT: current K4/K5/K6 controls, including all queued rows; no GPU allocation',flush=True); time.sleep(30)
        while not (ready:=bundle(INCOMING)):
            write_json(OUTPUT/'pipeline_status.json',dict(phase='waiting_for_sd1_bundle',gpu_used=False))
            print('WAIT: four matched training checkpoints; no GPU allocation',flush=True); time.sleep(30)
        specs={row:checkpoint_spec(ready,row) for row in ROWS}
        campaign.prepare(c,OUTPUT)
        results=[]; failures=[]
        for row,(k,mode) in ROWS.items():
            try:
                write_json(OUTPUT/'pipeline_status.json',dict(phase='waiting_for_gpu',row=row)); wait_idle()
                write_json(OUTPUT/'pipeline_status.json',dict(phase='evaluation',row=row))
                result=evaluate({**c,'condition_interval':k,'action_mode':mode},row,specs[row],output=OUTPUT)
                results.append(dict(row=row,**result))
                print(f"DONE {row}: {result['successes']}/500; {result['pooled_policy_ms_per_action']:.4f} ms/action",flush=True)
            except Exception as exc:
                failures.append(dict(row=row,error=str(exc))); print(f'ROW_FAILED {row}: {exc}',flush=True)
            write_json(OUTPUT/'combined_summary.json',dict(complete=len(results)==4,results=results,failures=failures,scope=c['scope']))
            with (OUTPUT/'comparison.csv').open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['row','successes','episodes','SR_percent','policy_ms_per_action'])
                for r in results: w.writerow([r['row'],r['successes'],r['episodes'],100*r['success_rate'],r['pooled_policy_ms_per_action']])
        write_json(OUTPUT/'pipeline_status.json',dict(phase='complete' if not failures else 'finished_with_failures',failures=failures))
        return int(bool(failures))


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('all','preflight','smoke','worker'),default='all',nargs='?')
    p.add_argument('--output',type=Path); p.add_argument('--row',choices=ROWS)
    p.add_argument('--suite',default='libero_10',choices=('libero_10',)); p.add_argument('--seed',default='seed01',choices=('seed01',))
    a=p.parse_args(); c=read_json(a.output/'runtime_config.json') if a.output else base_config()
    configure(c); sys.path.insert(0,c['libero_root']); os.environ['LIBERO_CONFIG_PATH']=c['libero_config']
    if a.command=='preflight':
        campaign.prepare(c,OUTPUT)
        print('CPU_PREFLIGHT_PASS; awaiting new training weights, GPU smoke deferred',flush=True); return 0
    if a.command=='all':
        try: return run_all(c)
        except BlockingIOError: print('ALREADY_RUNNING',flush=True); return 1
        except BaseException as exc:
            write_json(OUTPUT/'pipeline_status.json',dict(phase='failed',error=str(exc))); raise
    campaign.worker(c,a.output,a.suite,a.seed,a.row,smoke=a.command=='smoke',
        replay_factory=partial(controls.replay_factory,rows=ROWS),policy_factory=make_policy,
        policy_checker=partial(controls.check_policy,rows=ROWS),
        compiler_checker=partial(controls.check_compiler,rows=ROWS),reset_checker=reset_policy)
    return 0


if __name__=='__main__': raise SystemExit(main())
