"""Frozen current-vision measurements within long Condition intervals."""
import argparse
from pathlib import Path
import random
import subprocess
import sys
import time

from tools.simvla import age_specialization as previous
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from tools.simvla.gpu_followup_queue import run_queue

ARMS = ('midpoint', 'late_every')
OUTPUT = prior.STORAGE/'results/simvla/visual_midpoint/current_vision_nfe1_v1'
INCOMING = Path('/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_visual_midpoint')
MODULE = 'tools.simvla.visual_midpoint'
FILES = ('methods/latentloop/modules/visual_midpoint.py',
         'tools/simvla/visual_midpoint.py', 'tools/simvla/visual_midpoint_eval.py',
         'tools/simvla/visual_midpoint_features.py',
         'architectures/simvla/wrappers/run_visual_midpoint.sh')


def prepare():
    parent = previous.checked()
    c = dict(parent_contract=str(previous.OUTPUT/'contract.json'),
        parent_sha256=sha(previous.OUTPUT/'contract.json'), model=parent['models']['recurrent'],
        source_sha256={f:sha(ROOT/f) for f in FILES}, manifest=parent['manifest'],
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        steps=10000, warmup=500, learning_rate=1e-4, condition_weight=0.05, seed=7,
        training_ages=[4,5,6,7], batch_size=2, student_nfe=1, teacher_nfe=10,
        objective=parent['objective'], arms=list(ARMS),
        change='Frozen original vision encoder and connector at age4 only or ages4..7. Current 98x960 vision tokens enter learned attention before the late updater activation. Ages1..3 unchanged.',
        comparison='Reuse completed protected recurrent K8 and original recurrent K4; new midpoint and late_every K8 Long500. New matched-runtime Latent Bridge f4 uses independent env per episode. No SR-based early stop.')
    c['identity'] = prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Visual-midpoint contract changed')
    write_json(path,c)
    return c


def checked():
    c = read_json(OUTPUT/'contract.json')
    previous.checked()
    if sha(c['parent_contract']) != c['parent_sha256']:
        raise RuntimeError('Parent contract changed')
    for f, expected in c['source_sha256'].items():
        if sha(ROOT/f) != expected:
            raise RuntimeError('Source changed: '+f)
    return c


def initial_model(c, arm, native):
    import torch
    from methods.latentloop.modules.bounded_history import BoundedHistoryCondition
    from methods.latentloop.modules.visual_midpoint import VisualMidpointCondition
    spec = c['model']
    if sha(spec['checkpoint']) != spec['sha256']:
        raise RuntimeError('K4 parent changed')
    payload = torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
    if (payload['step'] != 10000 or payload['contract']['training_intervals'] != [4]
            or payload['contract']['action_mode'] != 'naive1'
            or payload['contract']['bounded_history_variant'] != 'recurrent'):
        raise RuntimeError('Unexpected parent')
    parent = BoundedHistoryCondition(native,'recurrent')
    parent.load_state_dict(payload['model'],strict=True)
    torch.manual_seed(c['seed'])
    return VisualMidpointCondition(parent,arm)


def train(arm, smoke=False):
    if prior.IS_RB2:
        raise RuntimeError('Training is assigned to sd1')
    import torch
    from tqdm import trange
    from tools.simvla.error_compensation_common import configure, snapshots
    from tools.simvla.condition_output_split_train import unroll, action_prediction, validate
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from methods.latentloop.modules.trend_condition import scaled_mse
    from methods.latentloop.modules.action_aligned_joint import action_loss
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch, append_jsonl
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime, lr_factor, assert_frozen
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
    c=checked(); cfg=c['model']['config']; configure(cfg)
    device,native,frozen,action,data,heldout=load_runtime(cfg,snapshots(cfg))
    model=initial_model(c,arm,native).to(device).eval()
    from tools.simvla.visual_midpoint_features import CachedFeatures
    features=CachedFeatures(OUTPUT/'features',c['identity'])
    model.feature_provider=features
    early_hash=state_hash(model.early); frozen_hash=state_hash(frozen)
    parameters=[p for p in model.parameters() if p.requires_grad]
    out=OUTPUT/('train_smoke' if smoke else 'train')/arm; out.mkdir(parents=True,exist_ok=True)
    steps=2 if smoke else c['steps']
    contract=dict(identity=c['identity'],arm=arm,parent_sha256=c['model']['sha256'],
        data=data.contract(),heldout=heldout.contract(),steps=c['steps'],batch_size=2,seed=c['seed'],
        target='Cached original Condition/NFE10 same-noise normalized continuous first5 actions',
        student_nfe=1,loss=c['objective'],change=c['change'],
        optimizer=dict(name='AdamW',lr=c['learning_rate'],weight_decay=0,warmup=c['warmup'],schedule='cosine to0.1x',clip=1),
        total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in parameters),early_hash=early_hash)
    write_json(out/'training_contract.json',contract)
    optimizer=torch.optim.AdamW(parameters,lr=c['learning_rate'],weight_decay=0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:lr_factor(n,c['steps'],c['warmup']))
    path=out/'latest.pt'; start=0; elapsed=0.
    if path.exists():
        saved=torch.load(path,map_location=device,weights_only=False)
        if saved['contract'] != contract:raise RuntimeError('Resume contract changed')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);start=saved['step'];elapsed=saved['seconds']
    tracker=None
    if cfg.get('wandb_project') and not smoke:
        try:
            import wandb
            tracker=wandb.init(project=cfg['wandb_project'],name='visual_midpoint_'+arm,
                id=c['identity'][:12]+'_'+arm,resume='allow',config=contract,dir=str(out),settings=wandb.Settings(init_timeout=20))
        except Exception as exc:print('WANDB_WARNING',exc,flush=True)
    began=time.monotonic()
    def save(n):
        atomic_torch_save(dict(format='simvla_visual_midpoint_v1',contract=contract,step=n,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            seconds=elapsed+time.monotonic()-began),path)
    try:
        progress=trange(start+1,steps+1,desc=arm,mininterval=2)
        for n in progress:
            rng=random.Random(c['seed']*100000+n);age=(4 if n==1 else 7) if smoke else 4+(n-1)%4
            optimizer.zero_grad(set_to_none=True);metrics=dict(step=n,age=age,action_l1=0.,condition_mse=0.)
            for _ in range(2):
                seq=move_batch(collate_exact_teacher_sequences([data[rng.randrange(len(data))]]),device)
                features.bind(seq)
                s=sample_from_sequence(seq,age)
                if n<=4:
                    with torch.no_grad():
                        teacher=action.decode_action_from_condition(s['target_condition'],s['proprio'],steps=10,initial_noise=s['noise'])
                    difference=float((teacher-s['target_action']).abs().max())
                    if difference>2e-4:raise RuntimeError(f'Teacher/cache mismatch {difference}')
                condition,bases,_=unroll(model,seq,age,8)
                loss_c=sum(scaled_mse(base,seq['teacher_conditions'][:,j],s['anchor'],s['valid'])
                    for j,base in enumerate(bases))/len(bases)
                pred=action_prediction(action,condition,s,steps=1)
                loss_a=action_loss(pred,action.action_space.normalize_action(s['target_action']))
                loss=loss_a+c['condition_weight']*loss_c
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                (loss/2).backward()
                metrics['action_l1']+=float(loss_a.detach())/2;metrics['condition_mse']+=float(loss_c.detach())/2
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            if not norm>0:raise RuntimeError('Missing gradient')
            if model.visual.output.weight.grad is None or model.visual.output.weight.grad.abs().sum()==0:
                raise RuntimeError('Reference branch has no action/Condition gradient')
            assert_frozen(frozen);assert_frozen(model.early)
            optimizer.step();scheduler.step()
            metrics.update(seconds=elapsed+time.monotonic()-began,lr=optimizer.param_groups[0]['lr'])
            progress.set_postfix(action=f"{metrics['action_l1']:.4f}",condition=f"{metrics['condition_mse']:.4f}")
            if n%50==0 or n in (1,steps):
                append_jsonl(out/'metrics.jsonl',metrics);write_json(out/'progress.json',metrics)
                if tracker:
                    try:tracker.log(metrics,step=n)
                    except Exception as exc:print('WANDB_WARNING',exc,flush=True)
            if n%500==0 or n==steps:save(n)
    finally:
        if tracker:
            try:tracker.finish()
            except Exception:pass
    if state_hash(frozen)!=frozen_hash or state_hash(model.early)!=early_hash:
        raise RuntimeError('Frozen weights changed')
    validate_features(c,model,action,heldout,out,smoke,features)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    write_json(out/'summary.json',dict(verdict='TRAIN_SMOKE_PASS' if smoke else 'TRAIN_COMPLETE',identity=c['identity'],arm=arm,
        checkpoint=str(path),checkpoint_sha256=sha(path),steps=steps,training_seconds=saved['seconds'],
        parent_training_steps=10000,total_training_steps=10000+steps,
        parameters=contract['total_parameters'],trainable_parameters=contract['trainable_parameters'],
        protected_early_unchanged=True))


def validate_features(c,model,action,heldout,out,smoke,features):
    import torch
    from tools.simvla.condition_output_split_train import unroll,action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from methods.latentloop.modules.action_aligned_joint import action_loss
    from methods.latentloop.modules.condition_output_split import geometry
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch
    records=[]
    with torch.no_grad():
        for index in _balanced_indices(heldout.identities,limit=2 if smoke else 30,seed=c['seed']):
            seq=move_batch(collate_exact_teacher_sequences([heldout[index]]),'cuda')
            features.bind(seq)
            for age in (4,5,6,7):
                z,_,_=unroll(model,seq,age,8);s=sample_from_sequence(seq,age)
                pred=action_prediction(action,z,s,steps=1,grad=False)
                records.append(dict(window=index,age=age,
                    action_l1=float(action_loss(pred,action.action_space.normalize_action(s['target_action']))),
                    condition=geometry(z,s['target_condition'],s['valid'])))
    write_json(out/'validation.json',dict(records=records,reference='Original NFE10 same-noise actions and Condition'))


def export(arm):
    import shutil
    c=checked();s=read_json(OUTPUT/'train'/arm/'summary.json')
    if s['verdict']!='TRAIN_COMPLETE' or s['identity']!=c['identity'] or sha(s['checkpoint'])!=s['checkpoint_sha256']:
        raise RuntimeError('Unfinished or changed model')
    stage=OUTPUT/'bundles'/arm;stage.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(s['checkpoint'],stage/'model.pt')
    write_json(stage/'manifest.json',dict(**s,source_sha256=c['source_sha256']))
    write_json(stage/'READY.json',dict(manifest_sha256=sha(stage/'manifest.json')))
    dest=str(INCOMING/arm)
    subprocess.run(['ssh','rb2','mkdir','-p',dest],check=True,timeout=30)
    subprocess.run(['rsync','-a','--partial','--exclude=READY.json',str(stage)+'/',f'rb2:{dest}/'],check=True,timeout=120)
    subprocess.run(['rsync','-a',str(stage/'READY.json'),f'rb2:{dest}/'],check=True,timeout=30)
    write_json(OUTPUT/'exports'/f'{arm}.json',dict(verdict='EXPORTED',identity=c['identity']))


def model_payload(c,arm):
    import torch
    if prior.IS_RB2:
        p=INCOMING/arm
        if read_json(p/'READY.json')['manifest_sha256']!=sha(p/'manifest.json'):raise RuntimeError('Incomplete transfer')
        spec=read_json(p/'manifest.json');path=p/'model.pt'
        if spec['source_sha256']!=c['source_sha256']:raise RuntimeError('Transferred source differs')
    else:
        spec=read_json(OUTPUT/'train'/arm/'summary.json');path=Path(spec['checkpoint'])
    if sha(path)!=spec['checkpoint_sha256']:raise RuntimeError('Model checksum differs')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (payload['format']!='simvla_visual_midpoint_v1' or payload['step']!=10000
            or payload['contract']['arm']!=arm or payload['contract']['parent_sha256']!=c['model']['sha256']):
        raise RuntimeError('Training mismatch')
    return payload,spec['checkpoint_sha256']


def jobs(c):
    plan=[]
    def add(key,args,path,completion,deps=(),**extra):
        plan.append(dict(id=key,cmd=[sys.executable,'-u','-m',MODULE]+args,deps=list(deps),
            summary=str(path),completion=dict(identity=c['identity'],**completion),**extra))
    if prior.IS_RB2:
        add('profile',['profile'],OUTPUT/'profile/summary.json',dict(verdict='PROFILE_COMPLETE'))
        add('bridge_f4',['eval','--arm','bridge_f4'],OUTPUT/'rows/bridge_f4/summary.json',dict(verdict='EVAL_COMPLETE',episodes=500),['profile'])
    if not prior.IS_RB2:
        for shard in range(4):
            add('features_'+str(shard),['features','--shard',str(shard)],OUTPUT/'features'/f'shard_{shard}.json',dict(verdict='FEATURES_COMPLETE',shard=shard))
        for arm in ARMS:
            add('smoke_'+arm,['train-smoke','--arm',arm],OUTPUT/'train_smoke'/arm/'summary.json',dict(verdict='TRAIN_SMOKE_PASS',steps=2),['features_'+str(i) for i in range(4)])
        for arm in ARMS:
            add('train_'+arm,['train','--arm',arm],OUTPUT/'train'/arm/'summary.json',dict(verdict='TRAIN_COMPLETE',steps=10000),['smoke_'+arm])
        for arm in ARMS:
            add('export_'+arm,['export','--arm',arm],OUTPUT/'exports'/f'{arm}.json',dict(verdict='EXPORTED'),['train_'+arm])
    for arm in ARMS:
        extra=dict(ready_file=str(INCOMING/arm/'READY.json'),upstream_status_file=str(INCOMING/'pipeline_status.json')) if prior.IS_RB2 else {}
        deps=['bridge_f4'] if prior.IS_RB2 else ['train_'+arm]
        add('eval_smoke_'+arm,['eval-smoke','--arm',arm],OUTPUT/'eval_smoke'/arm/'summary.json',dict(verdict='EVAL_SMOKE_PASS'),deps,**extra)
        add('eval_'+arm,['eval','--arm',arm],OUTPUT/'rows'/arm/'summary.json',dict(verdict='EVAL_COMPLETE',episodes=500),['eval_smoke_'+arm],**extra)
    return plan


def publish_status(phase):
    if prior.IS_RB2:return
    p=OUTPUT/'pipeline_status.json';write_json(p,dict(phase=phase))
    try:
        subprocess.run(['ssh','rb2','mkdir','-p',str(INCOMING)],check=True,timeout=20)
        subprocess.run(['rsync','-a',str(p),f'rb2:{INCOMING}/'],check=True,timeout=30)
    except (OSError,subprocess.SubprocessError) as exc:print('STATUS_TRANSFER_WARNING',exc,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('all','preflight','features','profile','train','train-smoke','export','eval','eval-smoke'),nargs='?',default='all')
    p.add_argument('--arm',choices=(*ARMS,'bridge_f4'));p.add_argument('--shard',type=int,choices=range(4));a=p.parse_args()
    if a.command=='features':
        from tools.simvla.visual_midpoint_features import extract
        extract(sys.modules[__name__],a.shard);return 0
    if a.command=='profile':
        from tools.simvla.visual_midpoint_features import profile
        profile(sys.modules[__name__]);return 0
    if a.command in ('train','train-smoke'):train(a.arm,a.command=='train-smoke');return 0
    if a.command=='export':export(a.arm);return 0
    if a.command in ('eval','eval-smoke'):
        from tools.simvla.visual_midpoint_eval import evaluate
        evaluate(a.arm,smoke=a.command=='eval-smoke');return 0
    c=prepare();plan=jobs(c);write_json(OUTPUT/'planned_jobs.json',plan)
    print('PREFLIGHT_PASS',len(plan),'jobs',OUTPUT,flush=True)
    if a.command=='preflight':return 0
    publish_status('running');rc=1
    try:
        rc=run_queue(OUTPUT,plan,gpus=(0,) if prior.IS_RB2 else (4,5,6,7),
            predecessor=[dict(path=str(previous.OUTPUT),lock='queue.lock',allow_when_all_assigned=True)],
            environment=prior.environment,cwd=ROOT,timeout=16*3600)
    finally:publish_status('complete' if not rc else 'finished_with_failures')
    return rc


if __name__=='__main__':raise SystemExit(main())

