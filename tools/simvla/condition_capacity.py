"""Bounded, per-input action fitting inside/outside frozen updater output spans."""
import argparse
from pathlib import Path
import subprocess
import time

import torch

from tools.simvla.error_compensation_common import (
    ROOT, configure, digest, environment, read_json, sha, snapshots, write_json,
)
from tools.simvla.condition_initialization_pipeline import OUTPUT as TRAINING
from tools.simvla.condition_output_split_eval import load_payload
from tools.simvla.condition_output_split_train import unroll
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.rollout_state_repair import sample_from_sequence
from methods.latentloop.modules.condition_output_split import ConditionOutputSplit, geometry
from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import (
    load_runtime, assert_frozen,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash

OUTPUT = TRAINING.parents[1] / 'condition_capacity/matched_input_fit_v1'
ROWS = tuple(f'{init}_{arm}' for init in ('pretrained', 'fresh')
             for arm in ('carry_output', 'carry_base'))


def output_basis(model):
    """Include both heads and their biases; allow any coefficient in this span."""
    columns = []
    for head in (model.condition_updater, model.action_condition_updater):
        columns.extend((head.up.weight.detach().cpu().double(),
                        head.up.bias.detach().cpu().double()[:, None]))
    matrix = torch.cat(columns, dim=1)
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    rank = int((singular > singular.max() * 1e-7).sum())
    if rank == 0:
        raise RuntimeError('Empty trained output span')
    return u[:, :rank].float(), singular.tolist()


def project(value, basis):
    return value if basis is None else (value @ basis) @ basis.T


def fit_condition(start, target, decode, valid, *, basis=None, steps=24,
                  step_fraction=0.01, radius_fraction=0.25):
    """Same projected-gradient budget, tensor coordinates and radius in both arms.

    Every iteration evaluates all three step sizes, even if the first improves.
    The result is a finite-budget fit, not a certified minimum or deployable model.
    """
    if steps < 1 or not 0 < step_fraction <= radius_fraction:
        raise ValueError('Invalid optimization budget')
    start = start.detach().float()
    mask = valid.to(device=start.device, dtype=torch.bool).unsqueeze(-1)
    if mask.shape != start.shape[:-1] + (1,):
        raise ValueError('Invalid token mask')
    if basis is not None:
        basis = basis.to(start)
        if basis.ndim != 2 or basis.shape[0] != start.shape[-1]:
            raise ValueError('Invalid output basis')
    scale = start.norm(dim=-1, keepdim=True).clamp_min(1.0)
    delta = torch.zeros_like(start)
    calls = 0

    def objective(value):
        nonlocal calls
        calls += 1
        loss = (decode(value)[:, :5] - target[:, :5]).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite per-input objective')
        return loss

    with torch.no_grad():
        initial = float(objective(start))
    trace = [dict(step=0, action_l1=initial)]
    for iteration in range(steps):
        value = (start + delta).detach().requires_grad_(True)
        loss = objective(value)
        gradient, = torch.autograd.grad(loss, value)
        gradient = project(gradient * mask, basis)
        direction = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        best_loss, best_delta, chosen = float(loss.detach()), delta, 0.0
        with torch.no_grad():
            for fraction in (1.0, 0.5, 0.25):
                candidate = project(delta - fraction * step_fraction * scale * direction, basis)
                candidate = candidate * mask
                ratio = (radius_fraction * scale / candidate.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
                candidate = candidate * ratio
                score = float(objective(start + candidate))
                if score < best_loss:
                    best_loss, best_delta, chosen = score, candidate, fraction
        delta = best_delta.detach()
        trace.append(dict(step=iteration+1, action_l1=best_loss, accepted_fraction=chosen))
    if calls != 1 + 4 * steps:
        raise RuntimeError('Unequal optimization call budget')
    span_error = float((delta - project(delta, basis)).norm() / delta.norm().clamp_min(1e-8))
    if basis is not None and span_error > 1e-4:
        raise RuntimeError('Constrained fit left its output span')
    return (start + delta).detach(), dict(trace=trace, forward_calls=calls,
        backward_calls=steps, span_relative_error=span_error,
        max_relative_token_change=float((delta.norm(dim=-1, keepdim=True)/scale).max()))


def source_files():
    inherited = read_json(TRAINING/'pretrained_10k/contract.json')['source_sha256']
    files = set(inherited) | {'tools/simvla/condition_capacity.py',
        'architectures/simvla/wrappers/run_condition_capacity.sh'}
    return {name: sha(ROOT/name) for name in sorted(files)}


def prepare():
    rows = {}
    for name in ROWS:
        init, arm = name.split('_', 1)
        directory = TRAINING/(init+'_10k')
        summary = read_json(directory/'train'/arm/'summary.json')
        if summary['verdict'] != 'TRAIN_AND_OFFLINE_COMPLETE' or summary['total_training_steps'] != 10000:
            raise RuntimeError('Incomplete source training: '+name)
        if sha(summary['checkpoint']) != summary['checkpoint_sha256']:
            raise RuntimeError('Source checkpoint changed: '+name)
        rows[name] = dict(summary=summary, config=read_json(directory/'runtime_config.json'), arm=arm)
    contract = dict(rows=rows, windows=16, ages=[1, 3, 7], interval=8, student_nfe=1,
        teacher_nfe=10, steps=24, step_fraction=0.01, radius_fraction=0.25,
        source_files=source_files(), seed=rows[ROWS[0]]['config']['seed'],
        scope='Offline per-input fit; all network weights frozen; no online SR or deployable speedup.',
        objective='First-five normalized continuous action L1 against full-condition original NFE10, same noise.',
        gradient='Only current Condition values; all encoder/updater/action weights frozen. No optimized Condition enters a later query.',
        restriction='Union of both learned up-projection weight columns and biases; relaxed coefficients, not the complete neural predictor constraints.',
        optimizer='24 projected-gradient updates; three candidates evaluated each update; same per-token step and radius in both arms.',
        second_noise='Independent deterministic noise, never used to optimize Condition; original NFE10 reference recomputed.',
        limitation='Finite-budget result. An unsuccessful fit does not prove an unrepresentable action. No module training or generalization claim.')
    contract['identity'] = digest(contract)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != contract:
        raise RuntimeError('Preserve existing capacity experiment; contract changed')
    write_json(path, contract)
    return contract


def means(records):
    result = {}
    for age in (1, 3, 7):
        selected = [r for r in records if r['age'] == age]
        if not selected:
            continue
        result[str(age)] = {}
        for variant in ('predicted', 'teacher_condition', 'within_output_span', 'unrestricted'):
            result[str(age)][variant] = {metric: sum(r[variant][metric] for r in selected)/len(selected)
                for metric in ('action_l1', 'second_noise_action_l1', 'raw_mse', 'cosine')}
    return result


def worker(row, smoke=False):
    contract = read_json(OUTPUT/'contract.json')
    if contract['source_files'] != source_files():
        raise RuntimeError('Source changed since capacity preflight')
    source = contract['rows'][row]
    c = source['config']
    configure(c)
    device, parent, frozen, action, _, heldout = load_runtime(c, snapshots(c))
    summary = source['summary']
    if sha(summary['checkpoint']) != summary['checkpoint_sha256']:
        raise RuntimeError('Source checkpoint changed')
    saved = load_payload(summary['checkpoint'], source['arm'], summary['identity'], steps=7000)
    model = ConditionOutputSplit(parent, source['arm']).to(device).eval().requires_grad_(False)
    model.load_state_dict(saved['model'], strict=True)
    hashes = state_hash(frozen), state_hash(model)
    basis, singular = output_basis(model)
    basis = basis.to(device)
    indices = _balanced_indices(heldout.identities, limit=contract['windows'], seed=contract['seed'])
    out = OUTPUT/('smoke' if smoke else 'rows')/row
    write_json(out/'inputs.json', dict(identity=contract['identity'], indices=indices,
        identities=[heldout.identities[i] for i in indices], heldout=heldout.contract(),
        output_span_rank=basis.shape[1], singular_values=singular,
        checkpoint_sha256=summary['checkpoint_sha256'], gpu=torch.cuda.get_device_name(0),
        git_head=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()))
    ages = [1] if smoke else contract['ages']
    indices = indices[:1] if smoke else indices
    records = []
    began = time.monotonic()
    for index in indices:
        sequence = move_batch(collate_exact_teacher_sequences([heldout[index]]), device)
        for age in ages:
            path = out/'queries'/f'window{index}_age{age}.json'
            if path.exists():
                record = read_json(path)
                if record['identity'] != contract['identity'] or record['row'] != row:
                    raise RuntimeError('Query resume identity changed')
                records.append(record)
                continue
            s = sample_from_sequence(sequence, age)
            with torch.no_grad():
                tick = time.perf_counter()
                predicted, _, _ = unroll(model, sequence, age, 8)
                torch.cuda.synchronize()
                unroll_seconds = time.perf_counter() - tick
            def decode(z, noise=s['noise'], steps=1, grad=False):
                return action.decode_action_from_condition(z, s['proprio'], steps=steps,
                    initial_noise=noise.clone(), requires_grad=grad, return_debug=True).final_action_latent
            with torch.no_grad():
                target = decode(s['target_condition'], steps=10)
                cached = action.action_space.normalize_action(s['target_action'])
                mismatch = float((target - cached).abs().max())
                if mismatch > 2e-4:
                    raise RuntimeError(f'Teacher cache mismatch: {mismatch}')
                seed = contract['seed'] + 1000003 + 1009*index + age
                generator = torch.Generator(device=device).manual_seed(seed)
                noise2 = torch.randn(s['noise'].shape, generator=generator, device=device, dtype=s['noise'].dtype)
                target2 = decode(s['target_condition'], noise=noise2, steps=10)
            def measure(z):
                with torch.no_grad():
                    return dict(action_l1=float((decode(z)[:, :5]-target[:, :5]).abs().mean()),
                        second_noise_action_l1=float((decode(z, noise=noise2)[:, :5]-target2[:, :5]).abs().mean()),
                        **geometry(z, s['target_condition'], s['valid']))
            record = dict(identity=contract['identity'], row=row, window=index, age=age,
                second_noise_seed=seed, teacher_cache_max_abs=mismatch,
                predictor_unroll_seconds=unroll_seconds, predicted=measure(predicted),
                teacher_condition=measure(s['target_condition']))
            modes = [('within_output_span', basis), ('unrestricted', None)]
            if (index+age) % 2:
                modes.reverse()
            for name, directions in modes:
                torch.cuda.synchronize(); tick = time.perf_counter()
                fitted, details = fit_condition(predicted, target,
                    lambda z: decode(z, grad=True), s['valid'], basis=directions,
                    steps=2 if smoke else contract['steps'], step_fraction=contract['step_fraction'],
                    radius_fraction=contract['radius_fraction'])
                torch.cuda.synchronize()
                record[name] = dict(**measure(fitted), **details, optimization_seconds=time.perf_counter()-tick)
            assert_frozen(frozen, model)
            write_json(path, record)
            records.append(record)
            write_json(out/'progress.json', dict(queries=len(records), total=len(indices)*len(ages),
                seconds=time.monotonic()-began, latest={k:record[k]['action_l1'] for k in ('predicted','within_output_span','unrestricted')}))
            print(f'CAPACITY {row} query={len(records)}/{len(indices)*len(ages)} age={age} '
                  f"pred={record['predicted']['action_l1']:.5f} span={record['within_output_span']['action_l1']:.5f} "
                  f"free={record['unrestricted']['action_l1']:.5f}", flush=True)
    if hashes != (state_hash(frozen), state_hash(model)):
        raise RuntimeError('Frozen network weights changed')
    write_json(out/'summary.json', dict(identity=contract['identity'], row=row,
        verdict='SMOKE_PASS' if smoke else 'CAPACITY_DIAGNOSTIC_COMPLETE', queries=len(records),
        windows=len(indices), groups=means(records), output_span_rank=basis.shape[1],
        seconds=time.monotonic()-began, frozen_networks_unchanged=True,
        cost_scope='Offline fitting time, not policy latency; no deployable weights or online SR produced.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--row', choices=ROWS)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.row:
        worker(args.row, args.smoke)
        return 0
    contract = prepare()
    jobs = []
    for row in ROWS:
        for smoke in (True, False):
            key = row+('_smoke' if smoke else '')
            jobs.append(dict(id=key, cmd=[contract['rows'][row]['config']['python'], '-u', '-m',
                'tools.simvla.condition_capacity', '--row', row]+(['--smoke'] if smoke else []),
                deps=[] if smoke else [row+'_smoke'],
                summary=str(OUTPUT/('smoke' if smoke else 'rows')/row/'summary.json'),
                completion=dict(identity=contract['identity'], row=row, queries=1 if smoke else 48,
                    verdict='SMOKE_PASS' if smoke else 'CAPACITY_DIAGNOSTIC_COMPLETE')))
    write_json(OUTPUT/'planned_jobs.json', jobs)
    if args.preflight:
        print('CPU_PREFLIGHT_PASS: 4 frozen models, 16 shared windows x 3 ages, matched bounded input fitting')
        return 0
    code = run_queue(OUTPUT, jobs, gpus=(4,5,6,7),
        predecessor=dict(path=str(TRAINING), lock='queue.lock'),
        environment=lambda gpu: environment({**contract['rows'][ROWS[0]]['config'], 'output':str(OUTPUT)}, gpu),
        cwd=ROOT, timeout=2*3600)
    summaries = {row:read_json(OUTPUT/'rows'/row/'summary.json') for row in ROWS
                 if (OUTPUT/'rows'/row/'summary.json').exists()}
    write_json(OUTPUT/'combined_summary.json', dict(identity=contract['identity'], rows=summaries,
        verdict='COMPLETE' if len(summaries)==len(ROWS) and not code else 'INCOMPLETE'))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
