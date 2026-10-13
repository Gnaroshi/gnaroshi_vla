"""Matched visual-anchor training and queued Long500 evaluations."""
import argparse
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

from tools.simvla import visual_midpoint as base
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from tools.simvla.gpu_followup_queue import run_queue

PARENT_OUTPUT = base.OUTPUT
PARENT_INITIAL = base.initial_model
OUTPUT = prior.STORAGE/'results/simvla/visual_reanchor/vision_change_nfe1_v1'
INCOMING = Path('/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_visual_reanchor')
ARMS = ('raw_midpoint', 'contrast_midpoint', 'raw_late_every', 'contrast_late_every')
FILES = ('methods/latentloop/modules/visual_reanchor.py', 'tools/simvla/visual_reanchor.py',
         'architectures/simvla/wrappers/run_visual_reanchor.sh')


def PARENT_CHECKED():
    c=read_json(PARENT_OUTPUT/'contract.json');base.previous.checked()
    if sha(c['parent_contract'])!=c['parent_sha256']:
        raise RuntimeError('Original parent contract changed')
    for file,expected in c['source_sha256'].items():
        if sha(ROOT/file)!=expected:raise RuntimeError('Original source changed: '+file)
    return c


def parts(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    kind, schedule = arm.split('_', 1)
    return schedule, kind == 'contrast'


def prepare():
    parent = PARENT_CHECKED()
    c = dict(parent)
    c.update(parent_contract=str(PARENT_OUTPUT/'contract.json'),parent_sha256=sha(PARENT_OUTPUT/'contract.json'),
        arms=list(ARMS),source_sha256={f:sha(ROOT/f) for f in FILES},
        source_parent_identity=parent['identity'],
        change='At scheduled vision queries, use original refresh Condition and refresh/current CNN inputs to predict total change. Other queries remain recurrent. Contrast subtracts shared-attention readout of refresh vision from current vision. Refresh vision is captured from the original backbone connector with no additional encoder call.',
        comparison='Reuse completed recurrent-current-vision controls. Four matched10K models: direct versus contrast visual readout crossed with midpoint versus late_every schedule. Same early weights, parameters, cache, loss, optimizer and paired Long500.')
    c.pop('identity',None)
    c['parent_git_head']=c.pop('git_head')
    c['identity']=prior.digest(c)
    path=OUTPUT/'contract.json'
    if path.exists() and read_json(path)!=c:
        raise RuntimeError('Reanchor contract changed')
    write_json(path,c)
    write_json(OUTPUT/'source.json',dict(git_head=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=c['source_sha256']))
    return c


def checked():
    c=read_json(OUTPUT/'contract.json');PARENT_CHECKED()
    if sha(c['parent_contract'])!=c['parent_sha256']:
        raise RuntimeError('Parent contract changed')
    for file,expected in c['source_sha256'].items():
        if sha(ROOT/file)!=expected:raise RuntimeError('Source changed: '+file)
    return c


def initial_model(c,arm,native):
    import torch
    from methods.latentloop.modules.visual_reanchor import VisualReanchorCondition
    schedule,contrast=parts(arm)
    parent=PARENT_INITIAL(c,schedule,native)
    torch.manual_seed(c['seed'])
    return VisualReanchorCondition(parent.early,schedule,contrast)


def configure_worker():
    from tools.simvla import visual_midpoint_features as feature
    original=feature.CachedFeatures
    parent=PARENT_CHECKED()
    class Features(original):
        def __call__(self,age,images):
            import torch
            values=[]
            for ids in self.sequence['query_ids']:
                query=ids[age]
                path=feature.feature_path(PARENT_OUTPUT/'features',query)
                identity=parent['identity']
                if not path.exists():
                    path=feature.feature_path(OUTPUT/'features',query)
                    identity=checked()['identity']
                saved=torch.load(path,map_location='cpu',weights_only=False)
                if saved['identity']!=identity or saved['query_id']!=query:
                    raise RuntimeError('Feature identity mismatch')
                values.append(saved['features'])
            return torch.cat(values).to(images.device)
    feature.CachedFeatures=Features
    base.OUTPUT,base.INCOMING=OUTPUT,INCOMING
    base.checked,base.initial_model=checked,initial_model


def extract(shard):
    from tools.simvla import visual_midpoint_features as feature
    def required(data):
        return sorted({window[0] for dataset in data for window in dataset.windows
            if not feature.feature_path(PARENT_OUTPUT/'features',window[0]).exists()})
    feature.required_queries=required
    feature.extract(SimpleNamespace(checked=checked,OUTPUT=OUTPUT),shard)


def attach(policy,model,interval,compiler):
    from tools.simvla.visual_midpoint_eval import attach_visual
    policy=attach_visual(policy,model,interval,compiler)
    if not model.contrast:return policy
    # Store the connector tensor already computed by the original full refresh.
    capture=dict(enabled=False,value=None)
    def hook(module,args,output):
        if capture['enabled']:
            capture['value']=output.reshape(1,-1,model.early.condition_dim).detach()
    policy._anchor_visual_hook=policy.model.vlm.model.connector.register_forward_hook(hook)
    full,reset,provider=policy._full_refresh,policy.reset,model.feature_provider
    def refresh(self,batch,*,policy_query_index):
        capture['enabled']=True;capture['value']=None
        try:result=full(batch,policy_query_index=policy_query_index)
        finally:capture['enabled']=False
        if capture['value'] is None:
            raise RuntimeError('Original refresh connector capture failed')
        return result
    def get_features(age,images):
        if age:
            return provider(age,images)
        if capture['value'] is None:raise RuntimeError('Missing original refresh vision')
        return capture['value']
    def reset_capture(self):
        reset();capture.update(enabled=False,value=None)
    policy._full_refresh=MethodType(refresh,policy)
    policy.reset=MethodType(reset_capture,policy)
    model.feature_provider=get_features
    policy.reset()
    return policy


def evaluate(arm,smoke,initial=False):
    from tools.simvla import visual_midpoint_eval as evaluation
    configure_worker()
    evaluation.study=base
    if initial:
        def payload(c,arm):
            from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
            from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import install_exact_uint8_delta_path
            native,_=load_native_v0_checkpoint(c['model']['config']['condition_checkpoint'],device='cuda',require_final_150k=True)
            install_exact_uint8_delta_path(native)
            model=initial_model(c,arm,native)
            return dict(model=model.state_dict()),'untrained_reanchor_'+arm
        evaluation.study=SimpleNamespace(OUTPUT=OUTPUT/'initial',checked=checked,
            initial_model=initial_model,model_payload=payload,previous=base.previous)
    original=evaluation.attach_visual
    def attach_new(policy,model,interval,compiler):
        # attach() imports the original adapter; keep it available during setup.
        evaluation.attach_visual=original
        try:return attach(policy,model,interval,compiler)
        finally:evaluation.attach_visual=attach_new
    evaluation.attach_visual=attach_new
    evaluation.evaluate(arm,smoke=smoke)


def control():
    from tools.simvla import visual_midpoint_eval as evaluation
    from tools.simvla.visual_midpoint_followup import attach_stale
    c=checked();original=evaluation.attach_visual
    evaluation.attach_visual=lambda *a,**kw:attach_stale(original,*a,**kw)
    evaluation.study=SimpleNamespace(OUTPUT=OUTPUT/'control',checked=lambda:c,
        model_payload=lambda c,arm:base.model_payload(PARENT_CHECKED(),arm),
        initial_model=PARENT_INITIAL,previous=base.previous)
    evaluation.evaluate('late_every')


def jobs(c):
    plan=[]
    def add(name,args,path,verdict,deps=(),**extra):
        plan.append(dict(id=name,cmd=[sys.executable,'-u','-m','tools.simvla.visual_reanchor']+args,
            deps=list(deps),summary=str(path),completion=dict(verdict=verdict,identity=c['identity']),**extra))
    if prior.IS_RB2:
        for arm in ARMS:
            add('initial_smoke_'+arm,['initial-smoke','--arm',arm],OUTPUT/'initial/eval_smoke'/arm/'summary.json','EVAL_SMOKE_PASS')
        add('control_stale_late',['control'],OUTPUT/'control/rows/late_every/summary.json','EVAL_COMPLETE',['initial_smoke_'+arm for arm in ARMS])
    else:
        for shard in range(4):
            add('features_'+str(shard),['features','--shard',str(shard)],OUTPUT/'features'/f'shard_{shard}.json','FEATURES_COMPLETE')
        for arm in ARMS:
            add('train_smoke_'+arm,['train-smoke','--arm',arm],OUTPUT/'train_smoke'/arm/'summary.json','TRAIN_SMOKE_PASS',['features_'+str(i) for i in range(4)])
            add('initial_smoke_'+arm,['initial-smoke','--arm',arm],OUTPUT/'initial/eval_smoke'/arm/'summary.json','EVAL_SMOKE_PASS',['train_smoke_'+arm])
        for arm in ARMS:
            add('train_'+arm,['train','--arm',arm],OUTPUT/'train'/arm/'summary.json','TRAIN_COMPLETE',['initial_smoke_'+arm])
            add('export_'+arm,['export','--arm',arm],OUTPUT/'exports'/f'{arm}.json','EXPORTED',['train_'+arm])
    for arm in ARMS:
        extra=dict(ready_file=str(INCOMING/arm/'READY.json')) if prior.IS_RB2 else {}
        deps=['control_stale_late'] if prior.IS_RB2 else ['train_'+arm]
        add('eval_smoke_'+arm,['eval-smoke','--arm',arm],OUTPUT/'eval_smoke'/arm/'summary.json','EVAL_SMOKE_PASS',deps,**extra)
        add('eval_'+arm,['eval','--arm',arm],OUTPUT/'rows'/arm/'summary.json','EVAL_COMPLETE',['eval_smoke_'+arm],**extra)
    return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('command',nargs='?',default='all',
        choices=('all','preflight','features','train-smoke','train','export','initial-smoke','eval-smoke','eval','control'))
    p.add_argument('--arm',choices=ARMS);p.add_argument('--shard',type=int,choices=range(4));args=p.parse_args()
    if args.command=='features':extract(args.shard);return 0
    if args.command=='control':control();return 0
    if args.command=='initial-smoke':evaluate(args.arm,True,initial=True);return 0
    if args.command in ('eval','eval-smoke'):evaluate(args.arm,args.command=='eval-smoke');return 0
    if args.command in ('train','train-smoke','export'):
        configure_worker()
        if args.command=='export':base.export(args.arm)
        else:base.train(args.arm,args.command=='train-smoke')
        return 0
    c=prepare();plan=jobs(c)
    write_json(OUTPUT/'planned_jobs.json',plan)
    print('PREFLIGHT_PASS',len(plan),'jobs',OUTPUT,flush=True)
    if args.command=='preflight':return 0
    return run_queue(OUTPUT,plan,gpus=(0,) if prior.IS_RB2 else (4,5,6,7),
        predecessor=[dict(path=str(PARENT_OUTPUT/'followup/queue'),lock='queue.lock',allow_when_all_assigned=True)],
        environment=prior.environment,cwd=ROOT,timeout=16*3600)


if __name__=='__main__':raise SystemExit(main())
