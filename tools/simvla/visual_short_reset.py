"""Train a visual midpoint for use by the frozen, successful K4 updater."""
import argparse
from pathlib import Path
import random
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace

from tools.simvla import visual_midpoint as base
from tools.simvla import visual_reanchor as recent
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT = prior.STORAGE/'results/simvla/visual_short_reset/frozen_k4_continuation_v1'
INCOMING = Path('/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_visual_short_reset')
TRAIN_ARMS = ('instant', 'rollout')
ARMS = (*TRAIN_ARMS, 'zero_previous', 'zero_refresh')
FILES = ('methods/latentloop/modules/visual_short_reset.py', 'tools/simvla/visual_short_reset.py',
         'architectures/simvla/wrappers/run_visual_short_reset.sh')


def prepare():
    parent = recent.checked()
    c = dict(parent)
    c.update(parent_contract=str(recent.OUTPUT/'contract.json'),
        parent_sha256=sha(recent.OUTPUT/'contract.json'),
        source_sha256={f: sha(ROOT/f) for f in FILES}, arms=list(ARMS),
        change='Full backbone at age0; learned current-vision Condition at age4; frozen K4 updater with relative ages1,2,3 at physical ages5,6,7. One original vision-only call per8 queries. Early weights and original policy frozen.',
        objective='Instant: action/Condition targets at age4. Rollout: action at uniformly sampled age4..7 and Condition MSE through that age. Action gradient passes through frozen K4 transitions to the midpoint module. First5 normalized continuous L1 against original NFE10 same-noise action plus0.05 Condition MSE.',
        comparison='Two matched fresh10K trainings from the original10K K4 parent. Two zero-training controls reuse completed midpoint models and only change the following updater/relative age. Reuse completed original continuation results; no repeat baseline.')
    c.pop('identity', None)
    c['identity'] = prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Short-reset contract changed')
    write_json(path, c)
    write_json(OUTPUT/'source.json', dict(git_head=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=c['source_sha256']))
    return c


def checked():
    c=read_json(OUTPUT/'contract.json'); recent.checked()
    if sha(c['parent_contract']) != c['parent_sha256']:
        raise RuntimeError('Parent changed')
    for f,expected in c['source_sha256'].items():
        if sha(ROOT/f) != expected: raise RuntimeError('Source changed: '+f)
    return c


def initial_model(c, arm, native):
    import torch
    from methods.latentloop.modules.visual_short_reset import VisualShortReset
    parent=base.initial_model(c,'midpoint',native)
    torch.manual_seed(c['seed'])
    return VisualShortReset(parent.early, anchor_reset=arm=='zero_refresh')


def target_age(arm, step):
    if arm not in TRAIN_ARMS: raise ValueError(arm)
    return 4 if arm=='instant' else 4+(step-1)%4


def train(arm, smoke=False):
    if prior.IS_RB2: raise RuntimeError('Train on sd1 only')
    import torch
    from tqdm import trange
    from tools.simvla.error_compensation_common import configure,snapshots
    from tools.simvla.condition_output_split_train import unroll,action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from methods.latentloop.modules.trend_condition import scaled_mse
    from methods.latentloop.modules.action_aligned_joint import action_loss
    from tools.simvla.visual_midpoint_features import CachedFeatures
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch,append_jsonl
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime,lr_factor,assert_frozen
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
    c=checked();cfg=c['model']['config'];configure(cfg)
    device,native,frozen,action,data,heldout=load_runtime(cfg,snapshots(cfg))
    model=initial_model(c,arm,native).to(device).eval()
    features=CachedFeatures(base.OUTPUT/'features',base.checked()['identity']);model.feature_provider=features
    early_hash,frozen_hash=state_hash(model.early),state_hash(frozen)
    parameters=[p for p in model.parameters() if p.requires_grad]
    out=OUTPUT/('train_smoke' if smoke else 'train')/arm;out.mkdir(parents=True,exist_ok=True)
    steps=2 if smoke else c['steps']
    contract=dict(identity=c['identity'],arm=arm,steps=c['steps'],parent_sha256=c['model']['sha256'],
        data=data.contract(),heldout=heldout.contract(),batch_size=2,seed=c['seed'],
        target='Original Condition and original NFE10 first5 continuous normalized action, same initial noise',
        student_nfe=1,objective=c['objective'],sampled_ages=[4] if arm=='instant' else [4,5,6,7],
        early_hash=early_hash,initial_hash=state_hash(model),
        optimizer=dict(lr=c['learning_rate'],warmup=c['warmup'],schedule='cosine to0.1x',weight_decay=0,clip=1),
        total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in parameters))
    write_json(out/'training_contract.json',contract)
    optimizer=torch.optim.AdamW(parameters,lr=c['learning_rate'],weight_decay=0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:lr_factor(n,c['steps'],c['warmup']))
    path=out/'latest.pt';start=0;elapsed=0.
    if path.exists():
        saved=torch.load(path,map_location=device,weights_only=False)
        if saved['contract']!=contract:raise RuntimeError('Resume contract changed')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);start=saved['step'];elapsed=saved['seconds']
    began=time.monotonic();tracker=None
    if cfg.get('wandb_project') and not smoke:
        try:
            import wandb
            tracker=wandb.init(project=cfg['wandb_project'],name='visual_short_reset_'+arm,
                id=c['identity'][:12]+'_'+arm,resume='allow',config=contract,dir=str(out),settings=wandb.Settings(init_timeout=20))
        except Exception as exc:print('WANDB_WARNING',exc,flush=True)
    def save(n):
        atomic_torch_save(dict(format='simvla_visual_short_reset_v1',contract=contract,step=n,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            seconds=elapsed+time.monotonic()-began),path)
    try:
        for n in trange(start+1,steps+1,desc=arm,mininterval=2):
            rng=random.Random(c['seed']*100000+n)
            age=(4 if n==1 else 7) if smoke else target_age(arm,n)
            optimizer.zero_grad(set_to_none=True);metrics=dict(step=n,age=age,action_l1=0.,condition_mse=0.)
            for _ in range(2):
                seq=move_batch(collate_exact_teacher_sequences([data[rng.randrange(len(data))]]),device);features.bind(seq)
                s=sample_from_sequence(seq,age)
                if n<=4:
                    with torch.no_grad():
                        teacher=action.decode_action_from_condition(s['target_condition'],s['proprio'],steps=10,initial_noise=s['noise'])
                    if float((teacher-s['target_action']).abs().max())>2e-4:raise RuntimeError('Teacher/cache mismatch')
                z,bases,_=unroll(model,seq,age,8)
                loss_c=sum(scaled_mse(zj,seq['teacher_conditions'][:,j],s['anchor'],s['valid']) for j,zj in enumerate(bases))/len(bases)
                pred=action_prediction(action,z,s,steps=1)
                loss_a=action_loss(pred,action.action_space.normalize_action(s['target_action']))
                loss=loss_a+c['condition_weight']*loss_c
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                (loss/2).backward()
                metrics['action_l1']+=float(loss_a.detach())/2;metrics['condition_mse']+=float(loss_c.detach())/2
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            if not norm>0 or model.visual.output.weight.grad is None or not model.visual.output.weight.grad.abs().sum()>0:
                raise RuntimeError('Missing visual reset gradient')
            assert_frozen(frozen);assert_frozen(model.early)
            optimizer.step();scheduler.step()
            metrics.update(seconds=elapsed+time.monotonic()-began,lr=optimizer.param_groups[0]['lr'])
            if n%50==0 or n in (1,steps):
                append_jsonl(out/'metrics.jsonl',metrics);write_json(out/'progress.json',metrics)
                if tracker:tracker.log(metrics,step=n)
            if n%500==0 or n==steps:save(n)
    finally:
        if tracker:tracker.finish()
    if state_hash(frozen)!=frozen_hash or state_hash(model.early)!=early_hash:raise RuntimeError('Frozen weights changed')
    base.validate_features(c,model,action,heldout,out,smoke,features)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    write_json(out/'summary.json',dict(verdict='TRAIN_SMOKE_PASS' if smoke else 'TRAIN_COMPLETE',identity=c['identity'],
        arm=arm,checkpoint=str(path),checkpoint_sha256=sha(path),steps=steps,training_seconds=saved['seconds'],
        parent_training_steps=10000,total_training_steps=10000+steps,parameters=contract['total_parameters'],
        trainable_parameters=contract['trainable_parameters'],protected_early_unchanged=True))


def model_payload(c,arm):
    import torch
    if arm=='zero_previous':return base.model_payload(base.checked(),'midpoint')
    if arm=='zero_refresh':
        original_output,original_incoming=base.OUTPUT,base.INCOMING
        try:
            base.OUTPUT,base.INCOMING=recent.OUTPUT,recent.INCOMING
            return base.model_payload(recent.checked(),'raw_midpoint')
        finally:base.OUTPUT,base.INCOMING=original_output,original_incoming
    if prior.IS_RB2:
        p=INCOMING/arm
        if read_json(p/'READY.json')['manifest_sha256']!=sha(p/'manifest.json'):raise RuntimeError('Partial transfer')
        spec=read_json(p/'manifest.json');path=p/'model.pt'
        if spec['source_sha256']!=c['source_sha256']:raise RuntimeError('Transferred source changed')
    else:
        spec=read_json(OUTPUT/'train'/arm/'summary.json');path=Path(spec['checkpoint'])
    if sha(path)!=spec['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if (saved['format']!='simvla_visual_short_reset_v1' or saved['step']!=10000
            or saved['contract']['arm']!=arm or saved['contract']['parent_sha256']!=c['model']['sha256']):
        raise RuntimeError('Wrong training payload')
    return saved,spec['checkpoint_sha256']


def export(arm):
    original=(base.OUTPUT,base.INCOMING,base.checked)
    try:
        base.OUTPUT,base.INCOMING,base.checked=OUTPUT,INCOMING,checked
        base.export(arm)
    finally:base.OUTPUT,base.INCOMING,base.checked=original


def attach_reset(original,policy,model,interval,compiler):
    policy=original(policy,model,interval,compiler)
    update=policy._v0_update
    def update_reset(self,batch,*,age,policy_query_index):
        result=update(batch,age=age,policy_query_index=policy_query_index)
        if age>4:
            self._age_calls['late']-=1;self._age_calls['early']+=1
        return result
    policy._v0_update=MethodType(update_reset,policy)
    return policy


def check(policy,arm):
    from tools.simvla.age_specialization_eval import check as parent_check
    counts=parent_check(policy,'parent')
    midpoint=sum(t%policy.k_c==4 for t in range(counts['queries']))
    if policy._age_calls!=dict(early=counts['updater_calls']-midpoint,late=midpoint):
        raise RuntimeError('Short-interval routing mismatch')
    if policy._extra_vision_calls!=midpoint:raise RuntimeError('Vision call mismatch')
    return dict(**counts,extra_vision_calls=midpoint,short_updater_calls=policy._age_calls['early'],midpoint_updater_calls=midpoint)


def evaluate(arm,smoke=False):
    from tools.simvla import visual_midpoint_eval as evaluation
    original=evaluation.attach_visual
    evaluation.attach_visual=lambda *a,**kw:attach_reset(original,*a,**kw)
    evaluation.check=check
    evaluation.study=SimpleNamespace(OUTPUT=OUTPUT,checked=checked,initial_model=initial_model,
        model_payload=model_payload,previous=base.previous)
    evaluation.evaluate(arm,smoke=smoke)


def jobs(c):
    plan=[]
    def add(name,args,path,verdict,deps=(),**extra):
        plan.append(dict(id=name,cmd=[sys.executable,'-u','-m','tools.simvla.visual_short_reset']+args,
            deps=list(deps),summary=str(path),completion=dict(verdict=verdict,identity=c['identity']),**extra))
    if not prior.IS_RB2:
        for arm in TRAIN_ARMS:
            add('train_smoke_'+arm,['train-smoke','--arm',arm],OUTPUT/'train_smoke'/arm/'summary.json','TRAIN_SMOKE_PASS')
            add('train_'+arm,['train','--arm',arm],OUTPUT/'train'/arm/'summary.json','TRAIN_COMPLETE',['train_smoke_'+arm])
            add('export_'+arm,['export','--arm',arm],OUTPUT/'exports'/f'{arm}.json','EXPORTED',['train_'+arm])
    for arm in ARMS:
        deps=['train_'+arm] if arm in TRAIN_ARMS and not prior.IS_RB2 else []
        extra=dict(ready_file=str(INCOMING/arm/'READY.json')) if arm in TRAIN_ARMS and prior.IS_RB2 else {}
        add('eval_smoke_'+arm,['eval-smoke','--arm',arm],OUTPUT/'eval_smoke'/arm/'summary.json','EVAL_SMOKE_PASS',deps,**extra)
        add('eval_'+arm,['eval','--arm',arm],OUTPUT/'rows'/arm/'summary.json','EVAL_COMPLETE',['eval_smoke_'+arm],**extra)
    return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('command',nargs='?',default='all',
        choices=('all','preflight','train-smoke','train','export','eval-smoke','eval'))
    p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.command in ('train','train-smoke'):train(a.arm,a.command=='train-smoke');return 0
    if a.command=='export':export(a.arm);return 0
    if a.command in ('eval','eval-smoke'):evaluate(a.arm,a.command=='eval-smoke');return 0
    c=prepare();plan=jobs(c);write_json(OUTPUT/'planned_jobs.json',plan)
    print('PREFLIGHT_PASS',len(plan),'jobs',OUTPUT,flush=True)
    if a.command=='preflight':return 0
    return run_queue(OUTPUT,plan,gpus=(0,) if prior.IS_RB2 else (4,5,6,7),
        predecessor=[dict(path=str(recent.OUTPUT),lock='queue.lock',allow_when_all_assigned=True)],
        environment=prior.environment,cwd=ROOT,timeout=24*3600)


if __name__=='__main__':raise SystemExit(main())
