"""Resume-safe rb2 queue for a complete Long500 NFE1 Condition comparison."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import zipfile

from dual_loop_runtime import ROOT,atomic_json,read_json,sha256
from run_pi05_dual_loop import run_logged

STORAGE=Path(os.environ.get('PI05_STORAGE','/home/mingyujung/private/gnaroshi_vla_storage'))
OUTPUT=STORAGE/'results/openpi/condition_bridge/recurrent_nfe1_long_seed7'
FILES=('tools/openpi/condition_bridge_common.py','tools/openpi/bridge_dagger.py','tools/openpi/train_condition_bridge.py','tools/openpi/serve_condition_bridge.py',
       'tools/openpi/run_condition_bridge.py','tools/openpi/evaluate_pi05_dual_loop.py',
       'architectures/openpi/wrappers/run_condition_bridge_long.sh')


def configure():
    c=read_json(STORAGE/'results/openpi/dual_loop/method_aligned_long_seed7/config.json')
    c.pop('config_id')
    upstream=Path(os.environ['LATENT_BRIDGE_UPSTREAM_ROOT'])
    c.update(comparison_contract='condition_bridge_nfe1',run_name='pi05_recurrent_condition_vs_official_bridge',
        condition_steps=10000,extension_steps=5000,source_sha256={f:sha256(ROOT/f) for f in FILES},
        bridge_source_sha256=sha256(upstream/'scripts/pi0/pi0_bridge_kv.py'),
        bridge_loss_source_sha256=sha256(upstream/'scripts/pi0/train_pi0_bridge_kv.py'),
        bridge_upstream_commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),
        training_objective='Ours: NFE1 same-noise first5 action L1 +0.05 relative raw KV MSE, recurrent BPTT. Bridge: official delta MSE +1-cosine(next KV), teacher forcing then detached self-conditioned K8 extension.',
        data_adaptation='Same222 LIBERO-Long demonstrations,42 heldout demonstrations. Cacheless per-window targets. Bridge additionally collects300 on-policy episodes per interval4/8 with seeded fresh resets; stored simulator states must not match final50 initial states.5K mixed demonstration/on-policy refinement per stage. Fixed step budgets differ from authors epochs.',
        interface_adaptation='Official Pi0BridgeKV768x10/12heads; full local prefix including instruction tokens; original previous raw8-state and previous first7-action inputs. No zeroing of instruction KV.',
        condition_initialization='fresh shared SimVLA recurrent core with age embeddings1..7; K2/3/4 use10K checkpoint; K8 separately extends with7 recursive queries for5K steps.',
        condition_weights={'condition':0.05,'first5_action':1.0},
        generation_training='none; original action expert NFE1 for every row',integration_steps=1,
        rows=['baseline_nfe1']+[f'{m}_k{k}' for m in ('ours','bridge') for k in (2,3,4,8)],
        latency='server preprocessing+prefix or updater+KV rebuild+NFE1+postprocessing; one query boundary synchronization; component GPU events; compile warmup excluded for all rows',
        compile_policy='same compiled vision embedding and action expert; each method compiled tensor updater; prefix transformer eager for all')
    c['config_id']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    path=OUTPUT/'config.json'
    if path.exists() and read_json(path)!=c:raise RuntimeError('Source/contract changed')
    for name in ('final_manifest','split_contract'):
        if sha256(c[name])!=c[name+'_sha256']:raise RuntimeError('Dataset/evaluation contract changed')
    if sha256(Path(c['checkpoint'])/'model.safetensors')!=c['baseline_model_sha256']:raise RuntimeError('Baseline changed')
    atomic_json(path,c)
    atomic_json(OUTPUT/'source.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=c['source_sha256']))
    return c


def train_job(c,method,extension=False,smoke=False,stage=None):
    name=stage or method+('_k8' if extension else '_k4')
    out=OUTPUT/('smoke_train' if smoke else 'train')/name
    refinement=method=='bridge' and not smoke and stage!='bridge_r0'
    steps=2 if smoke else c['extension_steps'] if extension or refinement else c['condition_steps']
    cmd=[sys.executable,ROOT/'tools/openpi/train_condition_bridge.py','--config',OUTPUT/'config.json','--method',method,'--output',out,'--steps',str(steps)]
    if smoke:cmd+=['--smoke']
    if extension:cmd+=['--extension','--parent',OUTPUT/'train'/f'{method}_k4/final.pt']
    if refinement:
        if not extension:cmd+=['--parent',OUTPUT/'train/bridge_r0/final.pt']
        cmd+=['--online-data',OUTPUT/'collection'/('bridge_k8' if extension else 'bridge_k4'),
              '--collection-interval','8' if extension else '4']
    return dict(id=('smoke_' if smoke else 'train_')+name,kind='train',cmd=list(map(str,cmd)),summary=str(out/'summary.json'),
        expected=dict(complete=True,config_id=c['config_id'],steps=steps,method=method,smoke=smoke))


def evaluation_job(c,row,smoke=False):
    name=('smoke_eval_' if smoke else 'eval_')+row
    return dict(id=name,kind='eval',row=row,smoke=smoke,summary=str(OUTPUT/('smoke_eval' if smoke else 'eval')/row/'summary.json'),
        expected=dict(config_id=c['config_id'],row=row,smoke_only=smoke,**({'episodes':1} if smoke else {'complete':True,'episodes':500})))


def collection_job(c,interval):
    row=f'bridge_k{interval}'
    parent='bridge_r0' if interval==4 else 'bridge_k4'
    return dict(id='collect_'+row,kind='eval',row=row,smoke=False,collect=True,
        checkpoint=str(OUTPUT/'train'/parent/'final.pt'),
        summary=str(OUTPUT/'collection'/row/'collection_summary.json'),
        expected=dict(complete=True,episodes=300,config_id=c['config_id'],row=row,final_evaluation_initial_states_disjoint=True))


def plan(c):
    jobs=[]
    for m in ('ours','bridge'):
        jobs+=[train_job(c,m,smoke=True),evaluation_job(c,m+'_k4',smoke=True)]
    jobs.append(evaluation_job(c,'baseline_nfe1'))
    for m in ('ours','bridge'):
        if m=='bridge':jobs.extend([train_job(c,m,stage='bridge_r0'),collection_job(c,4)])
        jobs.append(train_job(c,m))
        jobs.extend(evaluation_job(c,f'{m}_k{k}') for k in (2,3,4))
    for m in ('ours','bridge'):
        if m=='bridge':jobs.append(collection_job(c,8))
        jobs.append(train_job(c,m,extension=True))
        jobs.append(evaluation_job(c,m+'_k8'))
    for job in jobs:
        name=job['id'];deps=[]
        if name.startswith('smoke_eval_'):deps=['smoke_'+job['row']]
        elif name=='train_ours_k4':deps=['smoke_eval_ours_k4']
        elif name=='train_bridge_r0':deps=['smoke_eval_bridge_k4']
        elif name=='collect_bridge_k4':deps=['train_bridge_r0']
        elif name=='train_bridge_k4':deps=['collect_bridge_k4']
        elif name=='train_ours_k8':deps=['train_ours_k4']
        elif name=='collect_bridge_k8':deps=['train_bridge_k4']
        elif name=='train_bridge_k8':deps=['collect_bridge_k8']
        elif name.startswith('eval_') and job['row']!='baseline_nfe1':
            method=job['row'].split('_')[0];suffix='k8' if job['row'].endswith('k8') else 'k4'
            deps=['train_'+method+'_'+suffix]
        job['deps']=deps
    return jobs


def complete(job):
    p=Path(job['summary'])
    if not p.exists():return False
    s=read_json(p)
    return all(s.get(k)==v for k,v in job['expected'].items())


def evaluate(job):
    from websockets.sync.client import connect
    row=job['row'];smoke=job['smoke'];out=OUTPUT/('collection' if job.get('collect') else 'smoke_eval' if smoke else 'eval')/row
    out.mkdir(parents=True,exist_ok=True)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    ready=out/'ready.json';ready.unlink(missing_ok=True)
    cmd=[sys.executable,ROOT/'tools/openpi/serve_condition_bridge.py','--config',OUTPUT/'config.json','--row',row,'--port',str(port),'--ready',ready]
    if row!='baseline_nfe1':
        method=row.split('_')[0];suffix='k8' if row.endswith('k8') else 'k4'
        cmd+=['--checkpoint',job.get('checkpoint',OUTPUT/('smoke_train' if smoke else 'train')/(method+'_'+suffix)/'final.pt')]
    with (out/'server.log').open('a') as log:
        server=subprocess.Popen(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+600
            while True:
                if server.poll() is not None:raise RuntimeError('Server failed: '+str(out/'server.log'))
                if ready.exists():
                    try:
                        with connect(f'ws://127.0.0.1:{port}',open_timeout=2) as ws:ws.recv(timeout=2)
                        break
                    except (OSError,TimeoutError):pass
                if time.monotonic()>deadline:raise TimeoutError('Server startup')
                time.sleep(2)
            client=[os.environ['PI05_CLIENT_PY'],ROOT/'tools/openpi/evaluate_pi05_dual_loop.py','--config',OUTPUT/'config.json','--row',row,'--output',out,'--port',str(port)]
            if smoke:client+=['--smoke']
            if job.get('collect'):client+=['--collect-policy-data']
            run_logged(client,OUTPUT/'logs'/(job['id']+'.log'))
        finally:
            if server.poll() is None:
                server.terminate()
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:server.kill();server.wait()


def acquire(path):
    path.parent.mkdir(parents=True,exist_ok=True);handle=path.open('a+')
    try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:handle.close();return None
    return handle


def wait_gpu(c):
    predecessor=STORAGE/'results/simvla/visual_token_transport/warm_reader_nfe1_v1/queue_status.json'
    lease_path=Path.home()/'.cache/gnaroshi_vla/gpu_leases'/socket.gethostname()/'gpu0.lock'
    while True:
        pending=predecessor.exists() and read_json(predecessor).get('phase') not in ('complete','finished_with_failures')
        pids=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        mem,util=map(int,subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        if not pending and not pids and mem<768 and util<5:
            lease=acquire(lease_path)
            if lease:
                time.sleep(3)
                if not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():return lease
                lease.close()
        atomic_json(OUTPUT/'status.json',dict(phase='gpu_wait',predecessor_pending=pending,compute_pids=pids,memory_mib=mem,config_id=c['config_id'],updated_unix=time.time()))
        print('GPU_WAIT',pending,pids or 'none',mem,flush=True);time.sleep(30)


def aggregate(c):
    rows={row:read_json(OUTPUT/'eval'/row/'summary.json') for row in c['rows'] if (OUTPUT/'eval'/row/'summary.json').exists()}
    atomic_json(OUTPUT/'combined_summary.json',dict(config_id=c['config_id'],rows=rows,complete=all(rows.get(r,{}).get('complete') for r in c['rows']),
        limitations=[c['data_adaptation'],c['interface_adaptation']]))
    with zipfile.ZipFile(OUTPUT/'results_for_chatgpt.zip','w',zipfile.ZIP_DEFLATED) as z:
        for pattern in ('*.json','train/*/*.json','eval/*/*.json','eval/*/*.csv'):
            for p in OUTPUT.glob(pattern):z.write(p,p.relative_to(OUTPUT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','all','summarize'],default='all',nargs='?');a=parser.parse_args()
    if socket.gethostname()!='jbr-TRX50' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('rb2 GPU0 only')
    OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/'logs').mkdir(exist_ok=True)
    owner=acquire(OUTPUT/'queue.lock')
    if owner is None:raise RuntimeError('Queue already active')
    c=configure();jobs=plan(c);atomic_json(OUTPUT/'plan.json',jobs)
    if a.mode=='preflight':print('PREFLIGHT_PASS',len(jobs));return
    if a.mode=='summarize':aggregate(c);return
    lease=wait_gpu(c);failed={};done=[]
    try:
        for job in jobs:
            if complete(job):done.append(job['id']);continue
            if any(d in failed for d in job['deps']):
                failed[job['id']]='technical dependency failure: '+str(job['deps'])
                continue
            for attempt in (1,2):
                atomic_json(OUTPUT/'status.json',dict(phase='running',active=job['id'],attempt=attempt,completed=done,failed=failed,total_jobs=len(jobs),updated_unix=time.time()))
                try:
                    if job['kind']=='train':run_logged(job['cmd'],OUTPUT/'logs'/(job['id']+'.log'))
                    else:evaluate(job)
                    if not complete(job):raise RuntimeError('Completion artifact missing')
                    done.append(job['id']);aggregate(c);break
                except (RuntimeError,ValueError,TimeoutError,FileNotFoundError) as e:
                    print('TECHNICAL_FAILURE',job['id'],attempt,repr(e),flush=True)
                    if attempt==2:failed[job['id']]=str(e)
            # No SR gate. A broken training artifact prevents its dependent evaluations.
        atomic_json(OUTPUT/'status.json',dict(phase='complete' if not failed else 'finished_with_failures',completed=done,failed=failed,updated_unix=time.time()))
        aggregate(c)
    finally:lease.close();owner.close()


if __name__=='__main__':main()
