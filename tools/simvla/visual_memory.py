"""Three ordered Long500 tests: retain features, train retention, add spatial input."""
import argparse
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from tools.simvla import visual_midpoint as base
from tools.simvla import visual_reanchor as old_anchor
from tools.simvla import visual_short_reset as previous
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT,read_json,write_json,sha
from tools.simvla.gpu_followup_queue import run_queue

PARENT_OUTPUT=base.OUTPUT
PARENT_INITIAL=base.initial_model
OUTPUT=prior.STORAGE/'results/simvla/visual_memory/retained_spatial_nfe1_v1'
INCOMING=Path('/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_visual_memory')
ARMS=('hold','memory','spatial')
FILES=('methods/latentloop/modules/visual_memory.py','tools/simvla/visual_memory.py',
       'architectures/simvla/wrappers/run_visual_memory.sh')


def parent_checked():
    return old_anchor.PARENT_CHECKED()


def prepare():
    parent=parent_checked();previous.checked()
    c=dict(parent)
    c.update(parent_contract=str(PARENT_OUTPUT/'contract.json'),parent_sha256=sha(PARENT_OUTPUT/'contract.json'),
        source_sha256={f:sha(ROOT/f) for f in FILES},arms=list(ARMS),
        change='Keep original age4 vision tokens72x960 unchanged in context and read them at ages4..7. Memory/spatial train from same frozen K4 parent. Spatial adds current/midpoint RGB pair CNN6x6 per view to attention keys and values, preserving spatial positions.',
        comparison='Experiment1 uses completed late_every checkpoint with age4 feature reuse, no training. Experiment2 trains this retention protocol. Experiment3 trains the same protocol with spatial RGB input. Existing fresh vision, refresh-stale and one-use midpoint rows reused.',
        objective='Original cached NFE10 same-noise first5 normalized continuous action L1 plus0.05 mean original Condition scaled MSE. All late updater/CNN/attention parameters train; spatial branch trains through these same losses. Original networks and early K4 updater frozen.',
        evaluation_order=list(ARMS),training_order='Independent preparations may run concurrently; three Long500 comparisons merge in order.')
    c.pop('identity',None);c['parent_git_head']=c.pop('git_head');c['identity']=prior.digest(c)
    path=OUTPUT/'contract.json'
    if path.exists() and read_json(path)!=c:raise RuntimeError('Visual-memory contract changed')
    write_json(path,c)
    write_json(OUTPUT/'source.json',dict(git_head=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=c['source_sha256']))
    return c


def checked():
    c=read_json(OUTPUT/'contract.json');parent_checked()
    if sha(c['parent_contract'])!=c['parent_sha256']:raise RuntimeError('Parent changed')
    for f,expected in c['source_sha256'].items():
        if sha(ROOT/f)!=expected:raise RuntimeError('Source changed: '+f)
    return c


def initial_model(c,arm,native):
    import torch
    from methods.latentloop.modules.visual_memory import VisualMemoryCondition
    parent=PARENT_INITIAL(c,'late_every',native)
    torch.manual_seed(c['seed'])
    return VisualMemoryCondition(parent.early,spatial=arm=='spatial')


def configure_worker():
    from tools.simvla import visual_midpoint_features as features
    original=features.CachedFeatures
    parent=parent_checked()
    class ParentFeatures(original):
        def __init__(self,*args,**kwargs):
            super().__init__(PARENT_OUTPUT/'features',parent['identity'])
    features.CachedFeatures=ParentFeatures
    base.OUTPUT,base.INCOMING,base.checked,base.initial_model=OUTPUT,INCOMING,checked,initial_model


def model_payload(c,arm):
    if arm=='hold':return base.model_payload(parent_checked(),'late_every')
    original=base.OUTPUT,base.INCOMING
    try:
        base.OUTPUT,base.INCOMING=OUTPUT,INCOMING
        return base.model_payload(c,arm)
    finally:base.OUTPUT,base.INCOMING=original


def shard_count(arm):
    return 1 if prior.IS_RB2 else 2 if arm=='hold' else 4


def selected_manifest(manifest,shard,count):
    if not 0<=shard<count:raise ValueError('Invalid shard')
    result=dict(manifest);result['episodes']=manifest['episodes'][shard::count]
    return result


def evaluate(arm,shard,smoke=False):
    from tools.simvla import visual_midpoint_eval as evaluation
    c=checked();count=shard_count(arm)
    local=dict(c,manifest=selected_manifest(c['manifest'],shard,count))
    out=OUTPUT/'shards'/arm/f'part{shard}'
    original_write=evaluation.write_json
    def write(path,value):
        if Path(path).name=='progress.json':value=dict(value,total=len(local['manifest']['episodes']))
        original_write(path,value)
    evaluation.write_json=write
    evaluation.study=SimpleNamespace(OUTPUT=out,checked=lambda:local,initial_model=initial_model,
        model_payload=model_payload,previous=base.previous)
    evaluation.evaluate(arm,smoke=smoke)


def merge(arm):
    c=checked();records=[];parts=[];models=set()
    expected={(e['task_id'],e['trial_id']) for e in c['manifest']['episodes']}
    for shard in range(shard_count(arm)):
        p=OUTPUT/'shards'/arm/f'part{shard}'/'rows'/arm
        d=read_json(p/'summary.json')
        if d['verdict']!='EVAL_COMPLETE' or d['identity']!=c['identity']:raise RuntimeError('Incomplete shard')
        part=[read_json(f) for f in sorted((p/'episodes').glob('*.json'))]
        wanted={(e['task_id'],e['trial_id']) for e in selected_manifest(c['manifest'],shard,shard_count(arm))['episodes']}
        ids=[(e['task_id'],e['trial_id']) for e in part]
        if len(ids)!=len(set(ids)) or set(ids)!=wanted or len(ids)!=d['episodes']:raise RuntimeError('Shard episode mismatch')
        if sum(e['success'] for e in part)!=d['successes']:raise RuntimeError('Shard SR mismatch')
        if any(e['identity']!=c['identity'] for e in part):raise RuntimeError('Mixed episode contract')
        records.extend(part);parts.append(d);models.update(e['model_sha256'] for e in part)
    ids=[(e['task_id'],e['trial_id']) for e in records]
    if len(ids)!=500 or len(set(ids))!=500 or set(ids)!=expected or len(models)!=1:raise RuntimeError('Incomplete Long500')
    valid=[e for e in records if e['timing_valid']]
    summary=dict(verdict='EVAL_COMPLETE',identity=c['identity'],arm=arm,episodes=500,
        successes=sum(e['success'] for e in records),success_rate=sum(e['success'] for e in records)/500,
        policy_ms_per_action=sum(e['policy_ms_total'] for e in valid)/sum(e['episode_length'] for e in valid) if valid else None,
        timing_valid_episodes=len(valid),model_sha256=models.pop(),
        per_task={str(t):sum(e['success'] for e in records if e['task_id']==t) for t in range(10)},
        hardware=parts[0]['hardware'],shards=shard_count(arm),
        elapsed_seconds_sum=sum(d['elapsed_seconds'] for d in parts),
        timing_scope=parts[0]['timing_scope'])
    write_json(OUTPUT/'rows'/arm/'summary.json',summary)
    write_json(OUTPUT/'rows'/arm/'episode_outcomes.json',records)
    print(summary,flush=True)


def jobs(c):
    result=[]
    def add(name,args,path,verdict,deps=(),**extra):
        result.append(dict(id=name,cmd=[sys.executable,'-u','-m','tools.simvla.visual_memory']+args,
            deps=list(deps),summary=str(path),completion=dict(identity=c['identity'],verdict=verdict),**extra))
    if not prior.IS_RB2:
        for arm in ('memory','spatial'):
            add('train_smoke_'+arm,['train-smoke','--arm',arm],OUTPUT/'train_smoke'/arm/'summary.json','TRAIN_SMOKE_PASS')
            add('train_'+arm,['train','--arm',arm],OUTPUT/'train'/arm/'summary.json','TRAIN_COMPLETE',['train_smoke_'+arm])
            add('export_'+arm,['export','--arm',arm],OUTPUT/'exports'/f'{arm}.json','EXPORTED',['train_'+arm])
    for i,arm in enumerate(ARMS):
        deps=[] if i==0 else ['merge_'+ARMS[i-1]]
        if arm!='hold' and not prior.IS_RB2:deps.append('train_'+arm)
        extra=dict(ready_file=str(INCOMING/arm/'READY.json')) if prior.IS_RB2 and arm!='hold' else {}
        add('eval_smoke_'+arm,['eval-smoke','--arm',arm],OUTPUT/'shards'/arm/'part0/eval_smoke'/arm/'summary.json','EVAL_SMOKE_PASS',deps,**extra)
        for shard in range(shard_count(arm)):
            add(f'eval_{arm}_{shard}',['eval','--arm',arm,'--shard',str(shard)],
                OUTPUT/'shards'/arm/f'part{shard}'/'rows'/arm/'summary.json','EVAL_COMPLETE',['eval_smoke_'+arm],**extra)
        add('merge_'+arm,['merge','--arm',arm],OUTPUT/'rows'/arm/'summary.json','EVAL_COMPLETE',
            [f'eval_{arm}_{s}' for s in range(shard_count(arm))])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',nargs='?',default='all',
        choices=('all','preflight','train-smoke','train','export','eval-smoke','eval','merge'))
    p.add_argument('--arm',choices=ARMS);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    if a.command in ('train','train-smoke','export'):
        if a.arm=='hold':raise ValueError('hold reuses completed checkpoint')
        configure_worker()
        if a.command=='export':base.export(a.arm)
        else:base.train(a.arm,a.command=='train-smoke')
        return 0
    if a.command in ('eval','eval-smoke'):evaluate(a.arm,a.shard,a.command=='eval-smoke');return 0
    if a.command=='merge':merge(a.arm);return 0
    c=prepare();plan=jobs(c);write_json(OUTPUT/'planned_jobs.json',plan)
    print('PREFLIGHT_PASS',len(plan),'jobs',OUTPUT,flush=True)
    if a.command=='preflight':return 0
    return run_queue(OUTPUT,plan,gpus=(0,) if prior.IS_RB2 else (4,5,6,7),
        predecessor=[dict(path=str(previous.OUTPUT),lock='queue.lock',allow_when_all_assigned=True)],
        environment=prior.environment,cwd=ROOT,timeout=24*3600)


if __name__=='__main__':raise SystemExit(main())
