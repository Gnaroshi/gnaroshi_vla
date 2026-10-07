"""Evaluation-identical teacher labels and a small suffix-change predictor."""
from functools import lru_cache
from pathlib import Path
import random
import time

import torch
from tqdm import trange

from methods.prefix_residual.model import PrefixResidual
from .prefix import FrozenPrefix
from tools.simvla.error_compensation_common import identity, read_json, write_json, sha, snapshots
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save, load_native_v0_checkpoint
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    configure_strict_torch_determinism, load_frozen_simvla, freeze_module, append_jsonl,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import make_datasets, _balanced_indices
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import _drop_unused_vlm, _load_rgb_ref
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import lr_factor, assert_frozen
from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
from architectures.simvla.adapters.latentloop.native_v0_condition_hook import build_condition_token_layout
from methods.latentloop.modules.action_aligned_joint import action_loss


def setup(c):
    configure_strict_torch_determinism(c['seed'])
    torch.set_num_threads(1)
    memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((memory-2*1024**3)/memory)


def datasets(c):
    parent, saved = load_native_v0_checkpoint(c['condition_checkpoint'], device='cpu', require_final_150k=True)
    del parent
    return make_datasets({**c, 'training_k_c': 8}, saved)


def make_catalog(c):
    train, heldout = datasets(c)
    selections = {}
    episodes = []
    for name, data, limit in (('train', train, c['train_windows']), ('heldout', heldout, c['validation_windows'])):
        indices = _balanced_indices(data.identities, limit=limit, seed=c['seed'])
        selections[name] = dict(windows=[list(data.windows[i]) for i in indices],
                                identities=[list(data.identities[i]) for i in indices], split=data.contract())
        episodes.append({tuple(data.identities[i][:2]) for i in indices})
    if episodes[0] & episodes[1]:
        raise RuntimeError('Training and held-out episodes overlap')
    query_ids = sorted({q for s in selections.values() for w in s['windows'] for q in w})
    return dict(identity=identity(c), splits=selections, query_ids=query_ids,
                teacher_source='Official checkpoint, live policy preprocessing, NFE10; previous cached labels not reused')


def query_file(c, query_id):
    import hashlib
    name = hashlib.sha256(query_id.encode()).hexdigest()
    return Path(c['output'])/'features'/name[:2]/(name+'.pt')


def extract(c, shard):
    setup(c)
    run_id = identity(c)
    from tools.simvla.error_compensation_eval import make_policy
    policy = make_policy(c, 'baseline')
    model, action = policy.model, policy.action_adapter
    before = state_hash(model)
    prefix = FrozenPrefix(model, c['prefix_depth'])
    train, _ = datasets(c)
    catalog = read_json(Path(c['output'])/'catalog.json')
    ids = catalog['query_ids'][shard::4]
    diffs, parity = [], []
    started = time.monotonic()
    with torch.no_grad():
        for i, qid in enumerate(ids):
            path = query_file(c, qid)
            if path.exists():
                old = torch.load(path, map_location='cpu', weights_only=False)
                if old['identity'] != run_id or old['query_id'] != qid:
                    raise RuntimeError('Feature provenance changed')
                diffs.append(old['previous_cache_condition_mse'])
                continue
            query = train.store.query(qid)
            meta = query['metadata']
            rgb = _load_rgb_ref(meta['raw_rgb_ref']).numpy()
            batch = policy.preprocess(rgb[0], rgb[1], query['proprio'].numpy(), meta['language_instruction'])
            prefix.start_capture()
            condition = policy.condition_adapter.encode_condition(input_ids=batch['input_ids'],
                image_input=batch['image_input'], image_mask=batch['image_mask']).float()
            feature = prefix.finish_capture()
            if i < 2:
                measured = prefix.encode(batch)
                difference = float((feature-measured).abs().max())
                parity.append(difference)
                if difference > 2e-5:
                    raise RuntimeError(f'Prefix differs from original layer output: {difference}')
            generator = torch.Generator(device='cuda').manual_seed(query['noise_seed'])
            noise = torch.randn((1, 10, 7), device='cuda', generator=generator)
            teacher = action.decode_action_from_condition(condition, batch['proprio'], steps=10,
                initial_noise=noise, return_debug=True).final_action_latent
            tokenizer = policy.processor.tokenizer
            layout = build_condition_token_layout(condition=condition, image_mask=batch['image_mask'],
                input_ids=batch['input_ids'], pad_token_id=tokenizer.pad_token_id,
                special_token_ids=tokenizer.all_special_ids)
            old = query['condition'].to(condition.device).unsqueeze(0)
            diff = float((condition-old).square().mean())
            old_action_error = None
            if i < 2:
                old_action = action.decode_action_from_condition(old, batch['proprio'], steps=10,
                    initial_noise=noise, return_debug=True).final_action_latent
                old_action_error = float(action_loss(old_action, teacher))
            record = dict(identity=run_id, query_id=qid, task_id=meta['task_id'],
                episode_id=meta['episode_id'], prefix_depth=c['prefix_depth'], total_text_layers=prefix.total_layers,
                condition=condition.cpu(), prefix=feature.cpu(), proprio=batch['proprio'].cpu(),
                noise=noise.cpu(), action=teacher.cpu(), valid=layout.valid_mask.cpu(),
                previous_cache_condition_mse=diff, previous_cache_action_l1=old_action_error)
            atomic_torch_save(record, path)
            diffs.append(diff)
            if i % 25 == 0 or i+1 == len(ids):
                print(f'FEATURES shard={shard} {i+1}/{len(ids)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    prefix.close()
    if state_hash(model) != before:
        raise RuntimeError('Frozen model mutated during extraction')
    write_json(Path(c['output'])/'completed'/f'extract_{shard}.json', dict(verdict='FEATURES_COMPLETE',
        identity=run_id, shard=shard, queries=len(ids), wall_seconds=time.monotonic()-started,
        previous_cache_condition_mse_mean=sum(diffs)/len(diffs), prefix_full_forward_max_difference=max(parity, default=0.),
        prefix_depth=c['prefix_depth'], total_text_layers=prefix.total_layers, frozen_teacher_unchanged=True))


class Features:
    def __init__(self, c):
        self.c = c
        self.run_id = identity(c)

    @lru_cache(maxsize=256)
    def cpu(self, qid):
        item = torch.load(query_file(self.c, qid), map_location='cpu', weights_only=False)
        if item['identity'] != self.run_id or item['query_id'] != qid:
            raise RuntimeError('Feature identity mismatch')
        return item

    def pair(self, window, age):
        a, b = self.cpu(window[0]), self.cpu(window[age])
        if (a['task_id'], a['episode_id']) != (b['task_id'], b['episode_id']) or not torch.equal(a['valid'], b['valid']):
            raise RuntimeError('Pair crosses episode or layout')
        keys = ('condition', 'prefix', 'proprio', 'noise', 'action', 'valid')
        return [{key: item[key].to('cuda') for key in keys} for item in (a, b)]


def predict(model, a, b, learned=True):
    return model(a['condition'], a['prefix'], b['prefix'], a['valid'], learned=learned)


def decode(action, c, b, grad=False):
    return action.decode_action_from_condition(c, b['proprio'], steps=3,
        initial_noise=b['noise'], requires_grad=grad, return_debug=True).final_action_latent


@torch.no_grad()
def validate(model, action, features, windows, k, directory, label, smoke=False):
    records = []
    for i, window in enumerate(windows[:2] if smoke else windows):
        for age in range(1, k):
            a, b = features.pair(window, age)
            reference_delta = (b['condition']-a['condition'])[a['valid']]
            for row in ('hold', 'prefix_only', 'learned', 'original_condition'):
                condition = (a['condition'] if row == 'hold' else b['condition'] if row == 'original_condition'
                             else predict(model, a, b, learned=row == 'learned'))
                delta = (condition-a['condition'])[a['valid']]
                residual = (condition-b['condition'])[a['valid']]
                records.append(dict(window=i, age=age, row=row,
                    delta_mse=float(residual.square().mean()),
                    delta_cosine=float(torch.nn.functional.cosine_similarity(delta.flatten(), reference_delta.flatten(), dim=0)),
                    delta_norm_ratio=float(delta.norm()/reference_delta.norm().clamp_min(1e-8)),
                    action_l1=float(action_loss(decode(action, condition, b), b['action']))))
    means = {row: {key: sum(r[key] for r in records if r['row']==row)/sum(r['row']==row for r in records)
                     for key in ('delta_mse','delta_cosine','delta_norm_ratio','action_l1')}
             for row in ('hold','prefix_only','learned','original_condition')}
    write_json(directory/f'heldout_{label}.json', dict(records=records, means=means,
        scope='Matched inputs, NFE3, same noise, evaluation-preprocessed NFE10 teacher; no online success inference'))
    return means


def train(c, k, smoke=False):
    setup(c)
    total = 2 if smoke else c['steps']
    directory = Path(c['output'])/('smoke_train' if smoke else 'train')/f'k{k}'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'latest.pt'
    if (directory/'summary.json').exists():
        report = read_json(directory/'summary.json')
        if report['identity'] != identity(c) or report['steps'] != total or report['checkpoint_sha256'] != sha(path):
            raise RuntimeError('Training completion provenance changed')
        return
    assets = snapshots(c)
    frozen, _, action = load_frozen_simvla(checkpoint=assets['checkpoint_snapshot'],
        norm_stats=c['norm_stats'], smolvlm_model=assets['backbone_snapshot'], device=torch.device('cuda'))
    _drop_unused_vlm(frozen)
    freeze_module(frozen)
    frozen_hash = state_hash(frozen)
    torch.manual_seed(c['seed'])
    model = PrefixResidual(**c['model']).cuda()
    initial = state_hash(model)
    features = Features(c)
    catalog = read_json(Path(c['output'])/'catalog.json')
    windows, heldout = [catalog['splits'][s]['windows'] for s in ('train','heldout')]
    # Loss units come only from training data and are fixed throughout the run.
    scale_file = directory/'loss_scales.json'
    if scale_file.exists():
        scales = read_json(scale_file)
    else:
        dsum = asum = count = 0
        with torch.no_grad():
            for w in windows[:2] if smoke else windows[:32]:
                for age in range(1, k):
                    a,b = features.pair(w,age)
                    dsum += float((b['condition']-a['condition'])[a['valid']].square().mean())
                    asum += float(action_loss(decode(action,predict(model,a,b),b),b['action']))
                    count += 1
        scales = dict(delta_mse=max(dsum/count,1e-6), action_l1=max(asum/count,1e-6),
                      source='training windows only, initial analytic prefix-change predictor', pairs=count)
        write_json(scale_file,scales)
    optimizer = torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n: lr_factor(n,total,min(150,max(1,total//10))))
    start,elapsed = 0,0.
    if path.exists():
        saved = torch.load(path,map_location='cuda',weights_only=False)
        if saved['identity'] != identity(c) or saved['k'] != k:
            raise RuntimeError('Training resume provenance changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); start=saved['steps']; elapsed=saved['training_seconds']
    else:
        validate(model,action,features,heldout,k,directory,'before',smoke)
    tracker=None
    if not smoke and c.get('wandb_project'):
        try:
            import wandb
            tracker=wandb.init(project=c['wandb_project'],name=f'prefix_residual_k{k}',
                id=identity(c)[:12]+f'_k{k}',resume='allow',dir=str(directory),
                config=dict(c,loss_scales=scales),settings=wandb.Settings(init_timeout=20))
        except Exception as exc:
            print(f'WANDB_WARNING {exc}',flush=True)
    begun=time.monotonic()
    progress=trange(start+1,total+1,desc=f'prefix residual K{k}',mininterval=2)
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        rng=random.Random(c['seed']*10000000+step)
        values=dict(action_l1=0.,delta_mse=0.)
        for _ in range(2):
            w=windows[rng.randrange(len(windows))]; age=rng.randrange(1,k)
            a,b=features.pair(w,age)
            condition=predict(model,a,b)
            delta_error=(condition-b['condition'])[a['valid']].square().mean()
            act=action_loss(decode(action,condition,b,True),b['action'])
            loss=delta_error/scales['delta_mse'] + act/scales['action_l1']
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite objective')
            (loss/2).backward()
            values['action_l1']+=float(act.detach())/2; values['delta_mse']+=float(delta_error.detach())/2
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not norm>0: raise RuntimeError('Missing update gradients')
        assert_frozen(frozen)
        optimizer.step(); scheduler.step()
        seconds=elapsed+time.monotonic()-begun
        progress.set_postfix(act=f"{values['action_l1']:.4f}",delta=f"{values['delta_mse']:.4f}")
        if step in (1,total) or step%50==0:
            record=dict(step=step,**values,training_seconds=seconds,
                lr=optimizer.param_groups[0]['lr'],estimated_remaining_seconds=(total-step)*(time.monotonic()-begun)/(step-start),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            append_jsonl(directory/'metrics.jsonl',record)
            if tracker:
                try: tracker.log(record,step=step)
                except Exception as exc: print(f'WANDB_WARNING {exc}',flush=True)
        if step==total or step%500==0:
            atomic_torch_save(dict(identity=identity(c),k=k,steps=step,model=model.state_dict(),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),training_seconds=seconds,
                model_config=c['model'],prefix_depth=c['prefix_depth'],loss_scales=scales),path)
            print(f'CHECKPOINT step={step} {path}',flush=True)
    if state_hash(frozen)!=frozen_hash or state_hash(model)==initial:
        raise RuntimeError('Teacher changed or updater never changed')
    if tracker:
        try: tracker.finish()
        except Exception: pass
    validate(model,action,features,heldout,k,directory,'after',smoke)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    write_json(directory/'summary.json',dict(verdict='TRAIN_COMPLETE',identity=identity(c),k=k,steps=total,
        checkpoint_sha256=sha(path),training_seconds=saved['training_seconds'],
        trainable_parameters=sum(p.numel() for p in model.parameters()),prefix_depth=c['prefix_depth'],
        frozen_teacher_unchanged=True,loss_scales=scales,train_split=catalog['splits']['train']['split'],
        heldout_split=catalog['splits']['heldout']['split']))
