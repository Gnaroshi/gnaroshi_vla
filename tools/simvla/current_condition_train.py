"""Matched objectives for the current-action branch; frozen70K recurrence."""
import copy
import random
import time

from methods.latentloop.modules.current_condition import ARMS, install, parameter_counts, assert_frozen, frozen_hash
from tools.simvla import failure_interventions as prior
from tools.simvla import geometry_comparison_rb2 as portable
from tools.simvla import short_visual_train as base
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

OUTPUT = prior.STORAGE / 'results/simvla/current_condition/matched_targets_v1'
FILES = ('methods/latentloop/modules/current_condition.py', 'tools/simvla/current_condition_train.py',
         'tools/simvla/current_condition_campaign.py', 'tests/test_current_condition.py',
         'architectures/simvla/wrappers/run_current_condition.sh')


def prepare():
    from tools.simvla.token_interaction_campaign import parent_ledger
    c = copy.deepcopy(base.checked())
    c.pop('identity')
    bundle = portable.validate_bundle()
    origin = bundle['sources']['selected']['summary']
    c['parent_ledger'] = parent_ledger(c,bundle)
    inherited_seconds = sum(r['summary']['training_seconds'] for r in c['parent_ledger'])
    c.update(bundle=bundle, arms=list(ARMS), steps=5000, warmup=250, training_ages=[4],
        teacher_nfe=1, condition_weight=0.05, initialization={arm: dict(
            checkpoint_sha256=bundle['sources']['selected']['checkpoint_sha256'],
            parent_training_steps=origin['total_training_steps'],
            parent_training_seconds=inherited_seconds,
            total_training_steps=origin['total_training_steps']+5000,
            current_head='Copy frozen70K late updater and age4 visual reader; initial output parity') for arm in ARMS},
        objective='At age4: condition_target scaled MSE to z_t^F; action_target first5 continuous normalized action L1 to original NFE1 using identical noise; joint_target action L1 +0.05 scaled MSE.',
        gradient_routing='Only current_head updater and reader train. Frozen original action head supplies input gradients for action loss. Existing recurrent outputs and all other ages are frozen; current_head output never enters the next updater.',
        inference='At age4 reuse existing current original-vision features. Add one updater+reader and one frozen delta encoder call. Full backbone remains once per8 queries, original vision once at age4, NFE1 at every query.',
        prior_comparison='Output-splitting was tested before. This targets the measured age4 current-action-only intervention on the frozen70K visual model, with same NFE1 teacher and current original vision already available; branch starts with exact parent output.',
        no_performance_gate=True)
    c['source_sha256'].update({f: sha(ROOT/f) for f in FILES})
    c['source_sha256']['tools/simvla/token_interaction_campaign.py'] = sha(ROOT/'tools/simvla/token_interaction_campaign.py')
    c['identity'] = prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Current-condition contract changed')
    write_json(path, c)
    return c


def checked():
    c = read_json(OUTPUT/'contract.json')
    if prior.digest({k:v for k,v in c.items() if k!='identity'}) != c['identity']:
        raise RuntimeError('Contract identity changed')
    if portable.validate_bundle() != c['bundle']:
        raise RuntimeError('Parent changed')
    for f,h in c['source_sha256'].items():
        if sha(ROOT/f) != h:
            raise RuntimeError('Source changed: '+f)
    for item in c['input_sources'].values():
        base._verify_record(item)
    for item in c['parent_ledger']:
        base._verify_record(item)
    return c


def initial_model(c, arm, native):
    if arm not in ARMS:
        raise ValueError(arm)
    model = portable.construct(native)
    value, _ = portable.payload(c, 'selected_seed1')
    model.load_state_dict(value['model'], strict=True)
    return install(model)


def model_payload(c, arm, smoke=False):
    import torch
    summary = read_json(OUTPUT/('train_smoke' if smoke else 'train')/arm/'summary.json')
    steps = 2 if smoke else c['steps']
    if (summary['identity'], summary['arm'], summary['steps'], summary['verdict']) != (
            c['identity'], arm, steps, 'TRAIN_SMOKE_PASS' if smoke else 'TRAIN_COMPLETE'):
        raise RuntimeError('Incomplete training')
    if sha(summary['checkpoint']) != summary['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    value = torch.load(summary['checkpoint'], map_location='cpu', weights_only=False)
    if (value['identity'],value['arm'],value['step']) != (c['identity'],arm,steps):
        raise RuntimeError('Checkpoint metadata changed')
    if value['contract']['teacher_nfe'] != 1 or value['contract']['trainable_scope'] != 'current_head':
        raise RuntimeError('Training target/scope mismatch')
    return value, summary['checkpoint_sha256']


def loss_for(c, arm, model, action, seq, features):
    import torch
    from tools.simvla.condition_output_split_train import unroll, action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from methods.latentloop.modules.trend_condition import scaled_mse
    from methods.latentloop.modules.action_aligned_joint import action_loss
    features.bind(seq)
    s = sample_from_sequence(seq,4)
    z,_,_ = unroll(model,seq,4,8)
    lc = scaled_mse(z,s['target_condition'],s['anchor'],s['valid'])
    with torch.no_grad():
        target = action_prediction(action,s['target_condition'],s,steps=1,grad=False)
    prediction = action_prediction(action,z,s,steps=1,grad=arm!='condition_target')
    la = action_loss(prediction,target)
    loss = lc if arm=='condition_target' else la if arm=='action_target' else la+c['condition_weight']*lc
    return loss,la,lc


def train(arm, smoke=False):
    import torch
    from tools.simvla.error_compensation_common import configure
    from tools.simvla.visual_midpoint_features import CachedFeatures
    from tools.simvla.condition_output_split_train import unroll
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime,lr_factor
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch,append_jsonl
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
    from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
    c=checked();cfg=c['model']['config'];configure(cfg);torch.set_num_threads(1)
    device,native,original,action,data,heldout=load_runtime(cfg,c['snapshots'])
    model=initial_model(c,arm,native).to(device).eval()
    reference=portable.construct(native).to(device).eval()
    reference.load_state_dict(portable.payload(c,'selected_seed1')[0]['model'],strict=True)
    reference.requires_grad_(False)
    del native
    features=CachedFeatures(c['features']['root'],c['features']['identity'])
    model.feature_provider=features
    parent_hash,original_hash=frozen_hash(model),state_hash(original)
    parameters=[p for p in model.parameters() if p.requires_grad]
    steps=2 if smoke else c['steps']
    counts=parameter_counts(model)
    contract=dict(identity=c['identity'],arm=arm,steps=steps,seed=c['seed'],batch_size=c['batch_size'],
        teacher_nfe=1,student_nfe=1,training_age=4,trainable_scope='current_head',
        target=c['objective'],gradient_routing=c['gradient_routing'],
        initialization=c['initialization'][arm],data=data.contract(),heldout=heldout.contract(),
        counts=counts,initial_weights_sha256=state_hash(model),frozen_parent_sha256=parent_hash,
        frozen_original_sha256=original_hash,optimizer=dict(name='AdamW',lr=c['learning_rate'],
            weight_decay=0.,warmup=c['warmup'],schedule_steps=c['steps'],schedule='cosine to0.1x',clip=1.))
    out=OUTPUT/('train_smoke' if smoke else 'train')/arm;out.mkdir(parents=True,exist_ok=True)
    path=out/'training_contract.json'
    if path.exists() and read_json(path)!=contract:
        raise RuntimeError('Training contract changed')
    write_json(path,contract)
    optimizer=torch.optim.AdamW(parameters,lr=c['learning_rate'],weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:lr_factor(n,c['steps'],c['warmup']))
    checkpoint=out/'latest.pt';start=0;elapsed=0.
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['contract']!=contract or saved['identity']!=c['identity']:
            raise RuntimeError('Resume mismatch')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);start=saved['step'];elapsed=saved['seconds']
        if not 0<=start<=steps:
            raise RuntimeError('Resume step out of range')
    # Check both the initial action input and every carried prediction on a heldout window.
    seq=move_batch(collate_exact_teacher_sequences([heldout[0]]),device);features.bind(seq)
    reference.feature_provider=features
    with torch.no_grad():
        diffs=[]
        for age in range(1,8):
            z,_,d=unroll(model,seq,age,8);expected,_,_=unroll(reference,seq,age,8)
            carried_diff=float((d['carried']-expected).abs().max())
            output_diff=float((z-expected).abs().max())
            if carried_diff!=0 or (start==0 and output_diff!=0):
                raise RuntimeError('Initial/current versus recurrent path parity failed')
            diffs.append(dict(age=age,carried_max_abs=carried_diff,output_max_abs=output_diff))
    del reference
    write_json(out/'path_parity.json',dict(records=diffs,resume_step=start))
    began=time.monotonic()
    for step in range(start+1,steps+1):
        rng=random.Random(c['seed']*100000+step)
        indices=[rng.randrange(len(data)) for _ in range(c['batch_size'])]
        optimizer.zero_grad(set_to_none=True);metrics=dict(step=step,indices=indices,action_l1=0.,condition_mse=0.)
        for index in indices:
            seq=move_batch(collate_exact_teacher_sequences([data[index]]),device)
            loss,la,lc=loss_for(c,arm,model,action,seq,features)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            (loss/c['batch_size']).backward()
            metrics['action_l1']+=float(la.detach())/c['batch_size']
            metrics['condition_mse']+=float(lc.detach())/c['batch_size']
        assert_frozen(model,original)
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        if norm<=0:raise RuntimeError('No current-head gradient')
        if step==2:
            coverage={k:any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters())
                      for k,module in model.current_head.items()}
            if not all(coverage.values()):raise RuntimeError('Missing current-head block gradient')
            write_json(out/'gradient_audit.json',dict(coverage=coverage,frozen_parent_no_grad=True,
                frozen_original_no_grad=True,action_head_input_gradient=arm!='condition_target'))
        optimizer.step();scheduler.step()
        metrics.update(seconds=elapsed+time.monotonic()-began,lr=optimizer.param_groups[0]['lr'],gradient_norm=float(norm))
        append_jsonl(out/'progress.jsonl',metrics);write_json(out/'progress.json',metrics)
        if step%100==0 or step<=2:print(arm,metrics,flush=True)
        if step%500==0 or step==steps:
            if frozen_hash(model)!=parent_hash or state_hash(original)!=original_hash:
                raise RuntimeError('Frozen parameters changed')
            atomic_torch_save(dict(identity=c['identity'],arm=arm,step=step,contract=contract,
                model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                seconds=metrics['seconds']),checkpoint)
    optimizer.zero_grad(set_to_none=True)
    before=time.monotonic()
    base.validate(c,model,action,heldout,features,out/'validation.json',smoke)
    assert_frozen(model,original)
    if frozen_hash(model)!=parent_hash or state_hash(original)!=original_hash:
        raise RuntimeError('Frozen parameters changed after validation')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    ancestry=c['initialization'][arm]
    write_json(out/'summary.json',dict(verdict='TRAIN_SMOKE_PASS' if smoke else 'TRAIN_COMPLETE',
        identity=c['identity'],arm=arm,steps=steps,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        training_seconds=saved['seconds'],validation_seconds=time.monotonic()-before,
        parent_training_steps=ancestry['parent_training_steps'],total_training_steps=ancestry['parent_training_steps']+steps,
        prior_training_seconds=ancestry['parent_training_seconds'],
        total_training_seconds=ancestry['parent_training_seconds']+saved['seconds'],
        counts=counts,frozen_parent_unchanged=True,frozen_original_unchanged=True,no_performance_gate=True))
