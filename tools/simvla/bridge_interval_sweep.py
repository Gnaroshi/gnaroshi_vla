"""Queue a Large Latent Bridge interval sweep after the active rb2 campaign."""
import argparse
import csv
import fcntl
from pathlib import Path
import os
import sys
import time

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT,DEFAULT_CONFIG,Replay,configure,read_json,write_json,sha
from tools.simvla.compiled_policy import attach_policy,check_reset
from tools.simvla.trend_compiled_rb2 import recover

STORAGE=Path('/home/mingyujung/private/gnaroshi_vla_storage')
OUTPUT=STORAGE/'results/simvla/latent_bridge/compiled_k1_k8_seed01_v1'
REFERENCE=STORAGE/'results/simvla/compiled_paper/three_seed_v2'
PREDECESSOR=STORAGE/'results/simvla/trend_condition/rollout_round2_compiled_seed01_v1'
ROWS=tuple(f'latent_bridge_f{k}' for k in range(1,9))


def interval(row):
    if row not in ROWS: raise ValueError(row)
    return int(row.removeprefix('latent_bridge_f'))


def expected_counts(row,q):
    k=interval(row); full=(q+k-1)//k
    return dict(num_full_vlm_calls=full,num_condition_updater_calls=q-full,
        num_latent_bridge_calls=q-full,num_action_transformer_calls=10*q,
        num_generation_decoder_only_steps=0)


def check_policy(policy,row):
    q=int(policy.metrics.counters['num_policy_queries'])
    for name,value in expected_counts(row,q).items():
        if int(policy.metrics.counters.get(name,0))!=value: raise RuntimeError(f'{row}: {name} mismatch')
    if q!=(policy.step_index+4)//5: raise RuntimeError('H10/R5 queue changed')


def check_compiler(compiler,row):
    required={'vlm','action_transformer'}
    if interval(row)>1: required.add('bridge_predict_next')
    missing=[k for k in required if not compiler.records.get(k,{}).get('graphs',0)]
    if missing: raise RuntimeError('Compile bypass: '+str(missing))
    if interval(row)==1 and compiler.records.get('bridge_predict_next',{}).get('graphs',0):
        raise RuntimeError('K1 unexpectedly executed the bridge')


def replay_factory(c,row,compiler,samples):
    return Replay(c,'latent_bridge_f2',compiler,samples)


def make_policy(replay,c,row,manifest):
    policy=attach_policy(replay,c,'latent_bridge_f2',manifest)
    policy.refresh_every=interval(row)
    policy.row_name=row
    return policy


def base_config():
    return {**read_json(DEFAULT_CONFIG),**read_json(campaign.CONFIG),
        'output':str(OUTPUT),'long_rows':list(ROWS),'other_suites':[],'other_rows':[],
        'seeds':['seed01'],'smoke_episodes':1,'smoke_actions':41,'warmup_actions':41,
        'campaign_module':'tools.simvla.bridge_interval_sweep',
        'extra_source_files':['tools/simvla/bridge_interval_sweep.py',
            'architectures/simvla/wrappers/run_trend_compiled_rb2.sh'],
        'scope':'Frozen Large Latent Bridge, f=K=1..8, original ten-step action generation, H10/R5, LIBERO-Long 500 episodes seed01. No retraining. K1 validates zero bridge calls.'}


def compatible_contract(old,new):
    issues=[]
    for name in ('artifacts','hf_assets','libero_config','libero_config_sha256','packages','gpu','options','measurement'):
        if old[name]!=new[name]: issues.append(name)
    if old['manifest_hashes'].get('libero_10/seed01')!=new['manifest_hashes'].get('libero_10/seed01'):
        issues.append('episode_manifest')
    # Worktree paths differ; compare the bytes of every recorded old source.
    old_root=next(Path(p).parents[2] for p in old['source_files'] if p.endswith('/tools/simvla/compiled_campaign.py'))
    for path,expected in old['source_files'].items():
        p=Path(path)
        translated=ROOT/p.relative_to(old_root) if p.is_relative_to(old_root) else p
        if not translated.is_file() or sha(translated)!=expected: issues.append('source:'+str(translated))
    return issues


def reference_result(c,row,contract):
    if interval(row) not in (2,3,4): return None
    directory=REFERENCE/'rows/libero_10/seed01'/row
    if not (directory/'summary.json').exists(): return None
    old=read_json(REFERENCE/'campaign_contract.json')
    if compatible_contract(old,contract): return None
    result=read_json(directory/'summary.json')
    expected_identity=campaign.digest(dict(campaign=campaign.digest(old),suite='libero_10',seed='seed01',row=row,smoke=False))
    if result.get('identity')!=expected_identity or result.get('verdict')!='EPISODES_COMPLETE' or result.get('episodes')!=500:
        raise RuntimeError('Invalid reference result identity/completion')
    records=list(csv.DictReader((directory/'outcomes.csv').open()))
    if len(records)!=500 or {(int(r['task_id']),int(r['trial_id'])) for r in records}!={(t,i) for t in range(10) for i in range(50)}:
        raise RuntimeError('Reference episode IDs differ')
    if sum(int(r['success']) for r in records)!=result['successes']: raise RuntimeError('Reference success tally differs')
    if result['timing_valid_episodes']!=500: raise RuntimeError('Reference timing incomplete')
    return dict(row=row,k=interval(row),reused=True,source=str(directory/'summary.json'),**result)


def predecessor_busy():
    path=PREDECESSOR/'pipeline.lock'
    if not path.exists(): raise RuntimeError('Expected predecessor lock file missing')
    with path.open('r') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return True
    return False


def evaluate(c,row):
    for phase in ('smoke','worker'):
        if recover(c,OUTPUT,phase,row): continue
        ok=campaign.run_child(c,OUTPUT,phase,'libero_10','seed01',row)
        if not ok and not recover(c,OUTPUT,phase,row):
            directory=OUTPUT/('smoke' if phase=='smoke' else 'rows')/'libero_10/seed01'/row
            archive=OUTPUT/'failed_attempts'/f'{row}_{phase}'
            if not archive.exists():
                archive.parent.mkdir(parents=True,exist_ok=True)
                if directory.exists(): directory.rename(archive)
                else: archive.mkdir()
                campaign.run_child(c,OUTPUT,phase,'libero_10','seed01',row)
        if not recover(c,OUTPUT,phase,row): raise RuntimeError(f'{row} {phase} incomplete')
    path=OUTPUT/'rows/libero_10/seed01'/row/'summary.json'
    return dict(row=row,k=interval(row),reused=False,source=str(path),**read_json(path))


def run_all(c):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while predecessor_busy():
            write_json(OUTPUT/'pipeline_status.json',dict(phase='waiting_for_predecessor',gpu_used=False,predecessor=str(PREDECESSOR)))
            print('WAIT: current rb2 campaign, including its queued rows; no GPU allocation',flush=True); time.sleep(30)
        contract=campaign.prepare(c,OUTPUT)
        write_json(OUTPUT/'runtime_config.json',c)
        issues=compatible_contract(read_json(REFERENCE/'campaign_contract.json'),contract)
        write_json(OUTPUT/'reference_validation.json',dict(compatible=not issues,issues=issues,
            reference_root=str(REFERENCE),policy='reuse f2/f3/f4 only when compatible, otherwise run them'))
        results=[]; failures=[]
        for row in ROWS:
            try:
                result=reference_result(c,row,contract)
                if result: print('REUSE '+row+' 500 episodes',flush=True)
                else:
                    write_json(OUTPUT/'pipeline_status.json',dict(phase='evaluation',row=row))
                    result=evaluate(c,row)
                results.append(result)
            except Exception as exc:
                failures.append(dict(row=row,error=str(exc))); print(f'ROW_FAILED {row}: {exc}',flush=True)
            write_json(OUTPUT/'combined_summary.json',dict(complete=len(results)==8,results=results,failures=failures,
                suite='libero_10',seed='seed01',episodes_per_k=500,bridge='Large',flow_steps=10))
            with (OUTPUT/'comparison.csv').open('w',newline='') as f:
                writer=csv.writer(f); writer.writerow(['K','successes','episodes','SR_percent','policy_ms_per_action','reused','source'])
                for r in results: writer.writerow([r['k'],r['successes'],r['episodes'],100*r['success_rate'],r['pooled_policy_ms_per_action'],r['reused'],r['source']])
        write_json(OUTPUT/'pipeline_status.json',dict(phase='complete' if not failures else 'finished_with_failures',failures=failures))
        return int(bool(failures))


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('all','preflight','smoke','worker'),default='all',nargs='?')
    p.add_argument('--output',type=Path); p.add_argument('--suite',default='libero_10',choices=('libero_10',))
    p.add_argument('--seed',default='seed01',choices=('seed01',)); p.add_argument('--row',choices=ROWS)
    a=p.parse_args(); c=read_json(a.output/'runtime_config.json') if a.output else base_config()
    configure(c); sys.path.insert(0,c['libero_root']); os.environ['LIBERO_CONFIG_PATH']=c['libero_config']
    if a.command=='preflight':
        contract=campaign.prepare(c,OUTPUT)
        issues=compatible_contract(read_json(REFERENCE/'campaign_contract.json'),contract)
        reusable=[r for r in ROWS if reference_result(c,r,contract)]
        write_json(OUTPUT/'preflight.json',dict(verdict='PREFLIGHT_PASS',reusable=reusable,issues=issues,new_rows=[r for r in ROWS if r not in reusable]))
        print('PREFLIGHT_PASS; reusable='+str(reusable)+'; reference issues='+str(issues),flush=True); return 0
    if a.command=='all':
        try: return run_all(c)
        except BaseException as exc:
            write_json(OUTPUT/'pipeline_status.json',dict(phase='failed',error=str(exc))); raise
    campaign.worker(c,a.output,a.suite,a.seed,a.row,smoke=a.command=='smoke',
        replay_factory=replay_factory,policy_factory=make_policy,policy_checker=check_policy,
        compiler_checker=check_compiler,reset_checker=check_reset)
    return 0


if __name__=='__main__': raise SystemExit(main())
