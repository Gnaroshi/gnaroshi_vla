"""Frozen-prefix replacement campaign. No changes to upstream or active runs."""
import argparse
import os
from pathlib import Path
import socket
import subprocess

from tools.simvla.error_compensation_common import ROOT, CONFIG, configure, identity, read_json, write_json, environment
from tools.simvla.gpu_followup_queue import run_queue

STORAGE = Path('/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla')
OUTPUT = STORAGE/'results/simvla/prefix_residual/long_k4_k8_seed01_v1'
MODULE = 'tools.simvla.prefix_residual'


def configuration():
    return dict(read_json(CONFIG),output=str(OUTPUT),steps=3000,smoke_steps=2,smoke_policy_actions=41,
        prefix_depth=4,model=dict(dim=960,rank=128),train_windows=512,validation_windows=40,
        training_k_c=8,training_condition_ages=list(range(1,8)),evaluation_condition_intervals=[4,8],
        extra_source_files=['methods/prefix_residual/model.py','methods/prefix_residual/__init__.py',
            'architectures/simvla/wrappers/run_prefix_residual.sh'],
        student_condition_description='C0 + (Pj-P0) + learned suffix-change correction. P is original vision encoder and first4 text blocks followed by original final norm. No recursive predictions.',
        training_description='Two separate3K K4/K8 learners, same512 task-balanced train windows, same held-out episode split. Fresh NFE10 teacher labels use live evaluation preprocessing; NFE3 student. Equal initial-training-scale normalized delta-MSE and action-L1; no SR stopping gate.')


def child(c,*args):
    lease=os.environ.get('GNAROSHI_GPU_LEASE_FD')
    subprocess.run([c['python'],'-u','-m',MODULE,*args],cwd=ROOT,check=True,
        pass_fds=() if lease is None else (int(lease),))


def jobs(c):
    prefix=[c['python'],'-u','-m',MODULE]
    run_id=identity(c)
    extraction=[dict(id=f'extract_{s}',cmd=prefix+['extract','--shard',str(s)],
        summary=str(OUTPUT/'completed'/f'extract_{s}.json'),
        completion=dict(verdict='FEATURES_COMPLETE',identity=run_id,shard=s)) for s in range(4)]
    deps=[j['id'] for j in extraction]
    result=list(extraction)
    for k in (4,8):
        result.append(dict(id=f'train_k{k}',deps=deps,cmd=prefix+['train-cell','--k',str(k)],
            summary=str(OUTPUT/'completed'/f'train_k{k}.json'),
            completion=dict(verdict='TRAIN_CELL_COMPLETE',identity=run_id,k=k,steps=c['steps'])))
    for row in ('prefix_only','learned'):
        for k in (4,8):
            result.append(dict(id=f'eval_{row}_k{k}',deps=deps if row=='prefix_only' else [f'train_k{k}'],
                cmd=prefix+['eval-cell','--row',row,'--k',str(k)],
                summary=str(OUTPUT/'online'/f'kc{k}_{row}'/'summary.json'),
                completion=dict(verdict='EVALUATION_COMPLETE',identity=run_id,episodes=500)))
    return result


def summarize(c):
    rows={}
    for row in ('prefix_only','learned'):
        for k in (4,8):
            path=OUTPUT/'online'/f'kc{k}_{row}'/'summary.json'
            if path.exists(): rows[f'{row}_k{k}']=read_json(path)
    write_json(OUTPUT/'comparison_summary.json',dict(rows=rows,complete=len(rows)==4,
        protocol='sd1 RTX3090 eager, LIBERO-Long500, seed01, H10/R5/NFE3/EGL, full vision + first4 text layers every query',
        controls='Same measured prefix change, with/without learned suffix correction, independently trained K4/K8',
        latency='Whole policy.act including all prefix, vision and correction costs; exploration only',
        limitation='New architecture and corrected preprocessing jointly differ from the failed prior campaign; attribution requires matched controls above. No novelty or superiority assumed.'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('all','preflight','extract','train','train-cell','eval','eval-cell'))
    p.add_argument('--shard',type=int,choices=range(4))
    p.add_argument('--k',type=int,choices=(4,8),default=4)
    p.add_argument('--row',choices=('prefix_only','learned'),default='learned')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if socket.gethostname()!='jbrserver1': raise RuntimeError('sd1 campaign only')
    if a.command in ('all','preflight'):
        c=configuration(); configure(c)
        from tools.simvla.error_compensation_campaign import prepare
        prepare(c)
        from architectures.simvla.adapters.prefix_residual.train import make_catalog
        catalog=make_catalog(c)
        saved=OUTPUT/'catalog.json'
        if saved.exists() and read_json(saved)!=catalog: raise RuntimeError('Dataset selection changed')
        write_json(saved,catalog)
        write_json(OUTPUT/'runtime_config.json',c)
        plan=jobs(c); write_json(OUTPUT/'planned_jobs.json',dict(jobs=plan))
        print(f'CPU_PREFLIGHT_PASS jobs={len(plan)} feature_queries={len(catalog["query_ids"])}',flush=True)
        if a.command=='preflight': return 0
        try:
            return run_queue(OUTPUT,plan,gpus=(4,5,6,7),predecessor=[],
                environment=lambda g: environment(c,g),cwd=ROOT,timeout=86400)
        finally: summarize(c)
    c=read_json(OUTPUT/'runtime_config.json'); configure(c)
    if a.command=='extract':
        if a.shard is None: p.error('--shard required')
        from architectures.simvla.adapters.prefix_residual.train import extract
        extract(c,a.shard)
    elif a.command=='train':
        from architectures.simvla.adapters.prefix_residual.train import train
        train(c,a.k,a.smoke)
    elif a.command=='train-cell':
        child(c,'train','--k',str(a.k),'--smoke')
        child(c,'eval','--k',str(a.k),'--smoke')
        child(c,'train','--k',str(a.k))
        write_json(OUTPUT/'completed'/f'train_k{a.k}.json',dict(verdict='TRAIN_CELL_COMPLETE',identity=identity(c),
            k=a.k,steps=c['steps'],training=read_json(OUTPUT/'train'/f'k{a.k}'/'summary.json')))
    elif a.command=='eval':
        from tools.simvla.error_compensation_eval import run
        from architectures.simvla.adapters.prefix_residual.policy import make_policy,check_counts
        run(c,a.row,smoke=a.smoke,k_c=a.k,policy_factory=make_policy,counter_checker=check_counts)
    elif a.command=='eval-cell':
        for smoke in (True,False):
            child(c,'eval','--row',a.row,'--k',str(a.k),*(['--smoke'] if smoke else []))
        summarize(c)
    return 0


if __name__=='__main__': raise SystemExit(main())
