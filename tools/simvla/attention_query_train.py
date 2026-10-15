"""Matched query-source intervention after visual-attention joint training."""
import random
import subprocess
import time

from tools.simvla import visual_attention_train as parent
from tools.simvla import visual_readout_train as previous
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from methods.latentloop.modules.attention_query_contract import ARMS

OUTPUT = prior.STORAGE / 'results/simvla/attention_query_contract/matched_k8_nfe1_v1'
INCOMING = OUTPUT / 'unused_incoming'
FILES = ('methods/latentloop/modules/attention_query_contract.py',
         'tools/simvla/attention_query_train.py', 'tools/simvla/attention_query_campaign.py',
         'tools/simvla/attention_query_eval.py',
         'architectures/simvla/wrappers/run_attention_query_contract.sh')


def prepare():
    old = parent.checked()
    c = dict(old, parent_contract=str(parent.OUTPUT/'contract.json'),
        parent_sha256=sha(parent.OUTPUT/'contract.json'), arms=list(ARMS),
        source_sha256={f: sha(ROOT/f) for f in FILES},
        matched_parent='attention_joint fixed final 10K; no evaluation-based model selection',
        objective='Same first5 NFE10 action L1 +0.05 scaled Condition MSE + relative attention MSE. Only attention query source/target differs.',
        target_definition='Control: frozen teacher reads detached student query. Others: frozen teacher reads original previous Condition with its own frozen projection and current observation-pair code.',
        comparison='All independent10K from identical60K parent, same batches/LR/seed; K8,NFE1,H10/R5,Long500; early K4 remains frozen.',
        video_contract='Every evaluation episode, both cameras, every2 actions; numeric observations every action; compute latency excludes recording.')
    c.pop('identity', None)
    c['identity'] = prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Query contract changed')
    write_json(path, c)
    write_json(OUTPUT/'source.json', dict(git_head=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=c['source_sha256']))
    return c


def checked():
    c = read_json(OUTPUT/'contract.json'); parent.checked()
    if sha(c['parent_contract']) != c['parent_sha256']:
        raise RuntimeError('Parent contract changed')
    for f,h in c['source_sha256'].items():
        if sha(ROOT/f) != h: raise RuntimeError('Source changed: '+f)
    return c


def initial_model(c, arm, native):
    from methods.latentloop.modules.attention_query_contract import AttentionQueryCondition
    pc = parent.checked()
    model = parent.initial_model(pc, 'attention_joint', native)
    payload, _ = parent.model_payload(pc, 'attention_joint')
    model.load_state_dict(payload['model'], strict=True)
    teacher = parent.initial_model(pc, 'adapt_fresh', native)
    payload, _ = parent.model_payload(pc, 'adapt_fresh')
    teacher.load_state_dict(payload['model'], strict=True)
    return AttentionQueryCondition(model, teacher, arm)


def model_payload(c, arm):
    import torch
    summary = read_json(OUTPUT/'train'/arm/'summary.json')
    if summary['identity'] != c['identity'] or summary['verdict'] != 'TRAIN_COMPLETE':
        raise RuntimeError('Incomplete training')
    if sha(summary['checkpoint']) != summary['checkpoint_sha256']:
        raise RuntimeError('Checkpoint bytes changed')
    payload = torch.load(summary['checkpoint'],map_location='cpu',weights_only=False)
    if (payload['identity'],payload['arm'],payload['step']) != (c['identity'],arm,c['steps']):
        raise RuntimeError('Checkpoint mismatch')
    return payload,summary['checkpoint_sha256']


def unroll(model, sequence, age, interval=8):
    context = model.prepare(sequence['anchor_condition'],sequence['image_sequence'][:,0],
        sequence['proprio_sequence'][:,0],sequence['valid_mask'],sequence['group_ids'],interval)
    bases=[]
    for j in range(1,age+1):
        model.teacher_previous = (sequence['anchor_condition'] if j == 1
                                  else sequence['teacher_conditions'][:,j-2])
        output,diagnostics=model.predict(context,j,sequence['image_sequence'][:,j],sequence['proprio_sequence'][:,j])
        bases.append(diagnostics['base'])
    model.teacher_previous = None
    return output,bases,diagnostics


def train(arm, smoke=False):
    import torch
    from tqdm import trange
    from tools.simvla.error_compensation_common import configure, snapshots
    from tools.simvla.condition_output_split_train import action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from tools.simvla.visual_midpoint_features import CachedFeatures
    from methods.latentloop.modules.trend_condition import scaled_mse
    from methods.latentloop.modules.action_aligned_joint import action_loss
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch, append_jsonl
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime,lr_factor
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
    from methods.latentloop.modules.condition_output_split import geometry
    c=checked(); cfg=c['model']['config']; configure(cfg)
    device,native,frozen,action,data,heldout=load_runtime(cfg,snapshots(cfg))
    torch.manual_seed(c['seed'])
    model=initial_model(c,arm,native).to(device).eval()
    features=CachedFeatures(previous.parent.OUTPUT/'features',previous.parent.checked()['identity'])
    model.feature_provider=features
    frozen_hash,base_hash=previous.frozen_hash(model),state_hash(frozen)
    params=[p for p in model.parameters() if p.requires_grad]
    steps=2 if smoke else c['steps']; out=OUTPUT/('train_smoke' if smoke else 'train')/arm
    optimizer=torch.optim.AdamW(params,lr=c['learning_rate'],weight_decay=0)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:lr_factor(n,c['steps'],c['warmup']))
    contract=dict(identity=c['identity'],arm=arm,steps=steps,training=data.contract(),heldout=heldout.contract(),
        parent_checkpoint_sha256=parent.model_payload(parent.checked(),'attention_joint')[1],
        initial_weights_sha256=state_hash(model), trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in params),
        total_parameters=sum(p.numel() for p in model.parameters()), objective=c['objective'])
    write_json(out/'training_contract.json',contract)
    start=0; elapsed=0.; latest=out/'latest.pt'
    if latest.exists():
        p=torch.load(latest,map_location=device,weights_only=False)
        if p['contract']!=contract: raise RuntimeError('Resume changed')
        model.load_state_dict(p['model']);optimizer.load_state_dict(p['optimizer']);scheduler.load_state_dict(p['scheduler'])
        start,elapsed=p['step'],p['seconds']
    began=time.monotonic()
    for n in trange(start+1,steps+1,desc=arm,mininterval=2):
        rng=random.Random(c['seed']*100000+n);age=5+(n-1)%3
        if smoke:age=5 if n==1 else 7
        optimizer.zero_grad(set_to_none=True)
        metrics=dict(step=n,age=age,action_l1=0.,condition_mse=0.,attention_mse=0.,target_query_gap=0.)
        for _ in range(c['batch_size']):
            seq=move_batch(collate_exact_teacher_sequences([data[rng.randrange(len(data))]]),device)
            features.bind(seq);s=sample_from_sequence(seq,age)
            model.distill_enabled=True;model.attention_losses=[];model.query_errors=[]
            z,bases,_=unroll(model,seq,age)
            pred=action_prediction(action,z,s,steps=1)
            loss_a=action_loss(pred,action.action_space.normalize_action(s['target_action']))
            loss_c=sum(scaled_mse(b,seq['teacher_conditions'][:,j],s['anchor'],s['valid']) for j,b in enumerate(bases))/len(bases)
            loss_v=torch.stack(model.attention_losses).mean()
            loss=loss_a+0.05*loss_c+loss_v
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            (loss/c['batch_size']).backward()
            for key,val in [('action_l1',loss_a),('condition_mse',loss_c),('attention_mse',loss_v)]:
                metrics[key]+=float(val.detach())/c['batch_size']
            metrics['target_query_gap']+=sum(x['same_vs_original'] for x in model.query_errors)/len(model.query_errors)/c['batch_size']
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        if not norm>0:raise RuntimeError('Missing gradient')
        if any(p.grad is not None for p in frozen.parameters()) or any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError('Frozen gradient')
        optimizer.step();scheduler.step();seconds=elapsed+time.monotonic()-began
        if n%50==0 or n in (1,steps):
            metrics.update(seconds=seconds,lr=optimizer.param_groups[0]['lr'],gradient_norm=float(norm))
            append_jsonl(out/'metrics.jsonl',metrics);write_json(out/'progress.json',metrics)
        if n%500==0 or n==steps:
            atomic_torch_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                contract=contract,arm=arm,identity=c['identity'],step=n,seconds=seconds),latest)
    if previous.frozen_hash(model)!=frozen_hash or state_hash(frozen)!=base_hash:raise RuntimeError('Frozen weights changed')
    records=[]
    with torch.no_grad():
        for index in _balanced_indices(heldout.identities,limit=2 if smoke else 30,seed=c['seed']):
            seq=move_batch(collate_exact_teacher_sequences([heldout[index]]),device);features.bind(seq)
            for age in (4,5,6,7):
                model.distill_enabled=True;model.attention_losses=[];model.query_errors=[]
                z,_,_=unroll(model,seq,age);s=sample_from_sequence(seq,age)
                pred=action_prediction(action,z,s,steps=1,grad=False)
                original=action_prediction(action,s['target_condition'],s,steps=1,grad=False)
                records.append(dict(window=index,age=age,action_l1=float(action_loss(pred,action.action_space.normalize_action(s['target_action']))),
                    action_l1_vs_original_nfe1=float(action_loss(pred,original)),condition=geometry(z,s['target_condition'],s['valid']),
                    attention_query_comparisons=model.query_errors))
    model.distill_enabled=False;model.teacher_previous=None
    write_json(out/'validation.json',dict(records=records,no_selection_gate=True))
    write_json(out/'summary.json',dict(verdict='TRAIN_SMOKE_PASS' if smoke else 'TRAIN_COMPLETE',identity=c['identity'],arm=arm,
        steps=steps,checkpoint=str(latest),checkpoint_sha256=sha(latest),training_seconds=seconds,total_training_steps=70000,
        parameters=contract['total_parameters'],trainable_parameters=contract['trainable_parameters'],
        protected_frozen_parameters_unchanged=True))
