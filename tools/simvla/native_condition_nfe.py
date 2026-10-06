"""Native K2 Condition continuation through the deployed frozen Euler solver."""
import argparse
import os
from pathlib import Path
import random
import subprocess
import time

from tools.simvla.error_compensation_common import (
    ROOT, CONFIG, configure, environment, identity, read_json, sha, snapshots, write_json,
)
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.gpu_followup_queue import run_queue

NFES = (1, 2, 3, 10)
OUTPUT = Path('/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/condition_nfe/native_matched_k2_seed01_v1')
PREDECESSOR = OUTPUT.parents[1] / 'observation_correction/mixed_k4_k8_seed01_v1'


def configuration():
    return dict(read_json(CONFIG), output=str(OUTPUT), steps=3000, smoke_steps=2,
        warmup_steps=150, save_interval=500, condition_weight=0.05,
        training_condition_ages=[1], evaluation_condition_intervals=[2],
        smoke_policy_actions=11,
        student_condition_description='Unchanged native recurrent Condition module, K2. Each updated query has one original predecessor.',
        training_description='Same final150K native initialization, same train/heldout split and two examples per step. Four independent 3K continuations through native Euler NFE1/2/3/10; first5 continuous-action L1 + 0.05 normalized Condition MSE. Frozen original-condition NFE10 teacher. No Generation module, new data, architectural change or SR gate. AdamW1e-4, 150-step warmup, cosine to0.1x.')


def condition_prediction(adapter, sequence, age):
    from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair
    if age not in (1, 3):
        raise ValueError('K2 cached windows use updated queries 1 and 3')
    previous = sequence['anchor_condition'] if age == 1 else sequence['teacher_conditions'][:, 1]
    pair = NativeV0ObservationPair(sequence['image_sequence'][:, age-1],
        sequence['image_sequence'][:, age], sequence['proprio_sequence'][:, age-1],
        sequence['proprio_sequence'][:, age])
    code = adapter.delta_encoder(pair)
    predicted = adapter.condition_updater(previous, code,
        valid_mask=sequence['valid_mask'], group_ids=sequence['group_ids'], age=1).condition
    return predicted, previous


def sample_indices(seed, step, size):
    rng = random.Random(seed * 100000 + step)
    return [rng.randrange(size) for _ in (1, 3)]


def checkpoint_path(c, nfe, smoke):
    return Path(c['output']) / ('smoke_train' if smoke else 'train') / f'nfe{nfe}' / 'latest.pt'


def load_candidate(c, nfe, smoke, device):
    import torch
    p = checkpoint_path(c, nfe, smoke)
    d = torch.load(p, map_location=device, weights_only=False)
    if (d['identity'] != identity(c) or d['nfe'] != nfe
            or d['step'] != (c['smoke_steps'] if smoke else c['steps'])):
        raise RuntimeError('Candidate identity/solver/step mismatch')
    return d


def train(c, nfe, smoke):
    import torch
    from tqdm import trange
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch, append_jsonl
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import atomic_torch_save
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime, lr_factor, assert_frozen
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement import condition_query
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
    from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
    from methods.latentloop.modules.action_aligned_joint import action_loss
    from methods.latentloop.modules.trend_condition import scaled_mse
    configure(c)
    run_id = identity(c)
    path = checkpoint_path(c, nfe, smoke)
    out = path.parent
    out.mkdir(parents=True, exist_ok=True)
    total = c['smoke_steps'] if smoke else c['steps']
    if (out/'summary.json').exists():
        load_candidate(c, nfe, smoke, 'cpu')
        return
    device, adapter, frozen, action, data, heldout = load_runtime(c, snapshots(c))
    initial, frozen_hash = state_hash(adapter), state_hash(frozen)
    adapter.requires_grad_(True)
    params = list(adapter.parameters())
    if sum(p.numel() for p in params) != 584321:
        raise RuntimeError('Native Condition parameter count changed')
    contract = dict(identity=run_id, nfe=nfe, k_c=2, steps=total, generation_loop=False,
        initial_state_sha256=initial, train=data.contract(), heldout=heldout.contract(),
        parameters=sum(p.numel() for p in params), batch_size=2,
        teacher='Same-noise original-condition native NFE10, cached actions; no demonstrations as action targets.',
        objective='First5 continuous-action L1 + 0.05 masked normalized Condition MSE',
        fresh_queries='Original Condition and this native NFE; no trainable correction on fresh queries.',
        schedule='AdamW1e-4 wd0 clip1; 150-step warmup then cosine to0.1x over3K')
    write_json(out/'training_contract.json', contract)
    optimizer = torch.optim.AdamW(params, lr=c['learning_rate'], weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda n: lr_factor(n, total, min(c['warmup_steps'], max(1, total//10))))
    start, elapsed = 0, 0.0
    if path.exists():
        d = torch.load(path, map_location=device, weights_only=False)
        if d['identity'] != run_id or d['nfe'] != nfe or d['contract'] != contract:
            raise RuntimeError('Incompatible continuation')
        adapter.load_state_dict(d['model'], strict=True)
        optimizer.load_state_dict(d['optimizer']); scheduler.load_state_dict(d['scheduler'])
        start, elapsed = d['step'], d['training_seconds']
    def sequence_at(dataset, i):
        return move_batch(collate_exact_teacher_sequences([dataset[i]]), device)
    def terms(sequence, age, grad):
        condition, previous = condition_prediction(adapter, sequence, age)
        target = sequence['teacher_conditions'][:, age-1]
        prediction = action.decode_action_from_condition(condition,
            sequence['proprio_sequence'][:, age], steps=nfe,
            initial_noise=sequence['explicit_noises'][:, age-1],
            requires_grad=grad, return_debug=True).final_action_latent
        target_action = action.action_space.normalize_action(sequence['teacher_actions'][:, age-1])
        return (action_loss(prediction, target_action),
            scaled_mse(condition, target, previous, sequence['valid_mask']), condition)
    @torch.no_grad()
    def validate(label):
        records = []
        for index in _balanced_indices(heldout.identities, limit=2 if smoke else 30, seed=c['seed']):
            sequence = sequence_at(heldout, index)
            for age in (1, 3):
                act, cond, predicted = terms(sequence, age, False)
                cos = torch.nn.functional.cosine_similarity(predicted.float(),
                    sequence['teacher_conditions'][:, age-1].float(), dim=-1)
                records.append(dict(window=index, age=age, action_l1=float(act),
                    condition_normalized_mse=float(cond),
                    condition_cosine=float(cos[sequence['valid_mask'].bool()].mean())))
        write_json(out/f'heldout_{label}.json', dict(records=records, queries=len(records),
            means={k: sum(r[k] for r in records)/len(records)
                for k in ('action_l1', 'condition_normalized_mse', 'condition_cosine')}))
    # Verify the unmodified K2 adapter and cached teacher before any optimizer step.
    if not start:
        checks = []
        with torch.no_grad():
            sequence = sequence_at(data, 0)
            for age in (1, 3):
                predicted, _ = condition_prediction(adapter, sequence, age)
                diff = float((predicted-condition_query(adapter, sequence, age).condition).abs().max())
                teacher = action.decode_action_from_condition(sequence['teacher_conditions'][:, age-1],
                    sequence['proprio_sequence'][:, age], steps=10,
                    initial_noise=sequence['explicit_noises'][:, age-1])
                teacher_diff = float((teacher-sequence['teacher_actions'][:, age-1]).abs().max())
                if diff != 0 or teacher_diff > 2e-4:
                    raise RuntimeError(f'Adapter/cache mismatch: {diff}, {teacher_diff}')
                checks.append(dict(age=age, adapter_max_diff=diff, teacher_max_diff=teacher_diff))
        write_json(out/'numerical_preflight.json', dict(verdict='PASS', checks=checks))
        validate('before')
    tracker = None
    if not smoke and start < total:
        try:
            import wandb
            tracker = wandb.init(project=c['wandb_project'], name=f'native_condition_k2_nfe{nfe}',
                id=run_id[:12]+f'_nfe{nfe}', resume='allow', config=contract,
                dir=str(out), settings=wandb.Settings(init_timeout=20))
        except Exception as exc:
            print(f'WANDB_WARNING {exc}', flush=True)
    begun = time.monotonic()
    try:
        progress = trange(start+1, total+1, desc=f'Condition K2 native NFE{nfe}', mininterval=2)
        for n in progress:
            optimizer.zero_grad(set_to_none=True)
            metrics = dict(step=n, action_l1=0., condition_mse=0.)
            for age, index in zip((1, 3), sample_indices(c['seed'], n, len(data))):
                act, cond, _ = terms(sequence_at(data, index), age, True)
                loss = act + c['condition_weight']*cond
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite objective')
                (loss/2).backward()
                metrics['action_l1'] += float(act.detach())/2
                metrics['condition_mse'] += float(cond.detach())/2
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            if not norm > 0:
                raise RuntimeError('Condition gradients missing')
            assert_frozen(frozen)
            optimizer.step(); scheduler.step()
            seconds = elapsed + time.monotonic()-begun
            metrics.update(lr=optimizer.param_groups[0]['lr'], training_seconds=seconds,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))
            progress.set_postfix(act=f"{metrics['action_l1']:.4f}", cond=f"{metrics['condition_mse']:.4f}")
            if n in (1,total) or n % c['log_interval'] == 0:
                append_jsonl(out/'metrics.jsonl', metrics)
                if tracker:
                    try: tracker.log(metrics, step=n)
                    except Exception as exc: print(f'WANDB_WARNING {exc}', flush=True)
            if n == total or n % c['save_interval'] == 0:
                atomic_torch_save(dict(format='simvla_native_condition_nfe_v1', identity=run_id,
                    nfe=nfe, step=n, contract=contract, model=adapter.state_dict(),
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    training_seconds=seconds), path)
                print(f'CHECKPOINT step={n} path={path}', flush=True)
    finally:
        if tracker:
            try: tracker.finish()
            except Exception: pass
    if state_hash(frozen) != frozen_hash or state_hash(adapter) == initial:
        raise RuntimeError('Frozen teacher changed or Condition unchanged')
    validate('after')
    saved = load_candidate(c, nfe, smoke, 'cpu')
    write_json(out/'summary.json', dict(identity=run_id, verdict='TRAIN_COMPLETE', steps=total,
        nfe=nfe, checkpoint=str(path), checkpoint_sha256=sha(path),
        training_seconds=saved['training_seconds'], parameters=584321,
        frozen_teacher_unchanged=True))


def evaluate(c, nfe, smoke):
    from tools.simvla.error_compensation_eval import make_policy, run
    def factory(config, row, *, smoke, k_c):
        policy = make_policy(config, row, smoke=smoke, k_c=k_c)
        saved = load_candidate(config, nfe, smoke, 'cuda')
        policy.native_v0.load_state_dict(saved['model'], strict=True)
        policy.native_v0.requires_grad_(False)
        return policy
    run(c, f'condition_naive{nfe}', smoke=smoke, k_c=2, policy_factory=factory)


def summarize(c):
    out = Path(c['output'])
    rows = {}
    for nfe in NFES:
        t, e = out/'train'/f'nfe{nfe}'/'summary.json', out/'online'/f'kc2_condition_naive{nfe}'/'summary.json'
        if t.exists() and e.exists():
            rows[str(nfe)] = dict(training=read_json(t), evaluation=read_json(e))
    write_json(out/'comparison_summary.json', dict(complete=len(rows)==len(NFES), rows=rows,
        comparison='Same native module, initialization, data and optimizer steps; deployed NFE differs. No Generation Loop.',
        timing_scope='RTX3090 eager screening; do not compare latency directly to rb2 compiled RTX5090.'))


def cell(c, nfe):
    lease = os.environ.get('GNAROSHI_GPU_LEASE_FD')
    for smoke in (True, False):
        for stage in ('train', 'eval'):
            p = (checkpoint_path(c, nfe, smoke).parent/'summary.json' if stage == 'train'
                else Path(c['output'])/('eval_smoke' if smoke else 'online')/f'kc2_condition_naive{nfe}'/'summary.json')
            if p.exists():
                r = read_json(p)
                if r.get('identity') != identity(c):
                    raise RuntimeError('Result identity mismatch')
                if stage == 'train': load_candidate(c, nfe, smoke, 'cpu')
                elif r.get('episodes') != (1 if smoke else 500): raise RuntimeError('Incomplete episode summary')
                continue
            cmd = [c['python'], '-u', '-m', 'tools.simvla.native_condition_nfe', stage, '--nfe', str(nfe)]
            if smoke: cmd.append('--smoke')
            rc = subprocess.run(cmd, cwd=ROOT, pass_fds=() if lease is None else (int(lease),)).returncode
            if not p.exists(): raise RuntimeError(f'{stage} NFE{nfe} failed rc={rc}')
    summarize(c)
    write_json(Path(c['output'])/'completed'/f'nfe{nfe}.json',
        dict(verdict='CELL_COMPLETE', identity=identity(c), nfe=nfe, episodes=500, steps=c['steps']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('all', 'preflight', 'cell', 'train', 'eval'))
    p.add_argument('--nfe', type=int, choices=NFES)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    child = a.command in ('cell', 'train', 'eval')
    if child and a.nfe is None: p.error('--nfe required')
    c = read_json(OUTPUT/'runtime_config.json') if child else configuration()
    configure(c)
    if a.command == 'train': train(c, a.nfe, a.smoke); return 0
    if a.command == 'eval': evaluate(c, a.nfe, a.smoke); return 0
    if a.command == 'cell': cell(c, a.nfe); return 0
    prepare(c)
    write_json(OUTPUT/'runtime_config.json', c)
    plan = [dict(id=f'nfe{nfe}', cmd=[c['python'], '-u', '-m', 'tools.simvla.native_condition_nfe', 'cell', '--nfe', str(nfe)],
        summary=str(OUTPUT/'completed'/f'nfe{nfe}.json'),
        completion=dict(verdict='CELL_COMPLETE', identity=identity(c), nfe=nfe, episodes=500, steps=c['steps'])) for nfe in NFES]
    if a.command == 'preflight':
        print('PREFLIGHT_PASS: native K2; NFE1/2/3/10; 3K training then500 episodes; GPUs4-7', flush=True)
        return 0
    rc = run_queue(OUTPUT, plan, gpus=(4,5,6,7), predecessor=PREDECESSOR,
        environment=lambda gpu: environment(c, gpu), cwd=ROOT, timeout=86400)
    summarize(c)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
