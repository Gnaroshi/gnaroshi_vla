"""Paired condition/solver interventions on heldout observations, without training."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch

from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from tools.simvla.error_compensation_common import configure, environment, identity, snapshots
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.observation_correction_pipeline import OUTPUT as PREDECESSOR
from tools.simvla.observation_correction_train import unroll
from tools.simvla.rollout_state_repair import sample_from_sequence, prediction
from methods.latentloop.modules.observation_correction import ARMS, ObservationCorrection
from methods.latentloop.modules.observed_progress import build_model
from methods.latentloop.modules.trend_condition import scaled_mse
from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices

OUTPUT = PREDECESSOR.parents[1] / 'paired_error_analysis/condition_solver_seed01_v2'
JOBS = ('previous_joint', *ARMS, 'previous_joint_student_states')


def paired_metrics(a00, a01, a10, a11):
    """A00=teacher10; A01=teacher3; A10=predicted10; A11=predicted3."""
    if any(x.shape != a00.shape for x in (a01, a10, a11)) or a00.ndim != 3 or a00.shape[1:] != (10, 7):
        raise ValueError('Expected four matched [batch,10,7] action chunks')
    parts = dict(condition=a10-a00, solver=a01-a00, interaction=a11-a10-a01+a00,
                 combined=a11-a00)
    if not all(torch.isfinite(x).all() for x in (a00, a01, a10, a11)):
        raise RuntimeError('Nonfinite paired actions')
    result = {}
    for name, error in parts.items():
        e = error[:, :5].float()
        result.update({name+'_l1': float(e.abs().mean()), name+'_mse': float(e.square().mean()),
            name+'_translation_l1': float(e[..., :3].abs().mean()),
            name+'_rotation_l1': float(e[..., 3:6].abs().mean()),
            name+'_gripper_l1': float(e[..., 6].abs().mean())})
    result['same_solver_condition_l1'] = float((a11[:, :5]-a01[:, :5]).abs().mean())
    result['combined_gripper_sign_mismatches'] = int(((a11[:, :5, 6] > 0) != (a00[:, :5, 6] > 0)).sum())
    result['interaction_identity_max_abs'] = float((parts['combined']-parts['condition']-parts['solver']-parts['interaction']).abs().max())
    return result


def query_record(metadata, actions, times, *, condition_ms, teacher_diff, latent_mse):
    # condition_mse measures action error induced by the Condition replacement.
    # Keep the latent-space normalized error separate from that action metric.
    return dict(**metadata, **paired_metrics(*actions), **times,
        condition_prediction_ms=float(condition_ms), teacher_cache_max_abs=float(teacher_diff),
        latent_condition_normalized_mse=float(latent_mse))


def configuration():
    c = read_json(PREDECESSOR/'runtime_config.json')
    c.update(output=str(OUTPUT), heldout_windows=100,
        training_description='No training. Same-input 2x2 condition/solver interventions.',
        student_condition_description='Frozen previous_joint and four completed recurrent candidates; teacher used only in diagnostic branches.')
    checkpoints = {'previous_joint': c['selected_checkpoint']}
    for arm in ARMS:
        summary = read_json(PREDECESSOR/'train'/arm/'summary.json')
        if summary['verdict'] != 'TRAIN_AND_OFFLINE_COMPLETE':
            raise RuntimeError('Training not complete: ' + arm)
        checkpoints[arm] = dict(path=summary['checkpoint'], sha256=summary['checkpoint_sha256'])
    if any(sha(p['path']) != p['sha256'] for p in checkpoints.values()):
        raise RuntimeError('Checkpoint identity changed')
    c['analysis_checkpoints'] = checkpoints
    c['analysis_protocol'] = dict(teacher_steps=10, coarse_steps=3,
        same_proprio=True, same_noise=True, frozen_weights=True, prefix=5,
        labels='Original10 continuous normalized actions, not demonstration labels',
        latency='Synchronized eager tensor computation on sd1; excludes preprocessing/environment, not paper policy latency')
    c['analysis_protocol']['metric_definitions'] = dict(
        condition_mse='MSE of (predicted Condition + original10 actions) minus (original Condition + original10 actions), first five normalized actions',
        latent_condition_normalized_mse='Condition tensor prediction MSE normalized by anchor scale')
    c['analysis_protocol']['recovery_from'] = str(OUTPUT.with_name('condition_solver_seed01_v1'))
    return c


def aggregate(records):
    def stats(rows):
        keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in ('task', 'window', 'age', 'interval')]
        return dict(queries=len(rows), metrics={k:dict(mean=float(np.mean([r[k] for r in rows])),
            p95=float(np.quantile([r[k] for r in rows], .95))) for k in keys})
    result = {'all': stats(records)}
    for source, interval in sorted({(r['source'], r['interval']) for r in records}):
        rows = [r for r in records if r['source']==source and r['interval']==interval]
        result[f'{source}/k{interval}'] = stats(rows)
        for age in sorted({r['age'] for r in rows}):
            result[f'{source}/k{interval}/age{age}'] = stats([r for r in rows if r['age']==age])
    return result


def collected_samples(c):
    # Full histories are absent here: only the absolute-anchor model is valid.
    sources = [s for s in c['record_sources'] if s['label']=='previous']
    if len(sources) != 1:
        raise RuntimeError('Ambiguous collection source')
    source = sources[0]
    seen = {}
    for marker in sorted(Path(source['path']).glob('*/task*_state*.json')):
        meta = read_json(marker)
        if meta['identity'] != source['identity']:
            raise RuntimeError('Collection identity mismatch')
        if meta.get('split') == 'train':
            continue
        p = marker.parent/meta['file']
        if sha(p) != meta['sha256']:
            raise RuntimeError('Collection payload changed')
        for record in torch.load(p, map_location='cpu', mmap=True, weights_only=False):
            m = record['metadata']
            if m['split'] != 'heldout':
                continue
            key = (m['driver'], m['task_id'], record['age'])
            if seen.get(key, 0) >= 2:
                continue
            seen[key] = seen.get(key, 0) + 1
            yield record
    if not seen:
        raise RuntimeError('No heldout collection records')


@torch.inference_mode()
def analyze(c, arm, *, smoke=False):
    configure(c)
    run_id = identity(c)
    out = (OUTPUT/'smoke' if smoke else OUTPUT)/arm
    verdict = 'PAIRED_SMOKE_PASS' if smoke else 'PAIRED_ANALYSIS_COMPLETE'
    out.mkdir(parents=True, exist_ok=True)
    dest = out/'summary.json'
    if dest.exists():
        old = read_json(dest)
        if old['identity'] != run_id:
            raise RuntimeError('Analysis identity mismatch')
        if old['verdict'] == verdict:
            return
    device, parent, frozen, action, _, heldout = load_runtime(c, snapshots(c))
    checkpoint_name = 'previous_joint' if arm == 'previous_joint_student_states' else arm
    spec = c['analysis_checkpoints'][checkpoint_name]
    if sha(spec['path']) != spec['sha256']:
        raise RuntimeError('Checkpoint changed')
    saved = torch.load(spec['path'], map_location='cpu', weights_only=False)
    if checkpoint_name == 'previous_joint':
        model = build_model(copy.deepcopy(parent), saved['arm'], max_age=7).to(device).eval()
    else:
        model = ObservationCorrection(parent, arm).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    model.requires_grad_(False)
    records, tensors = [], []
    began = time.monotonic()
    def measure(s, predicted, metadata, condition_ms):
        def decode(condition, steps):
            torch.cuda.synchronize(); tick = time.perf_counter()
            result = action.decode_action_from_condition(condition, s['proprio'], steps=steps,
                initial_noise=s['noise'].clone(), return_debug=True)
            torch.cuda.synchronize()
            if int(result.debug['iterations']) != steps:
                raise RuntimeError('Unexpected transformer count')
            return result, (time.perf_counter()-tick)*1000
        # Warm both solver paths before recording computation time.
        if not records:
            decode(s['target_condition'], 10); decode(s['target_condition'], 3)
        outputs, times = {}, {}
        for name, condition, steps in [('a00',s['target_condition'],10),('a01',s['target_condition'],3),
                                       ('a10',predicted,10),('a11',predicted,3)]:
            outputs[name], times[name+'_decode_ms'] = decode(condition, steps)
        diff = float((outputs['a00'].action-s['target_action']).abs().max())
        if diff > 2e-4:
            raise RuntimeError(f'Original teacher cache/runtime mismatch: {diff}')
        values = [outputs[key].final_action_latent for key in ('a00','a01','a10','a11')]
        r = query_record(metadata, values, times, condition_ms=condition_ms, teacher_diff=diff,
            latent_mse=scaled_mse(predicted,s['target_condition'],s['anchor'],s['valid']))
        records.append(r)
        tensors.append(torch.stack([x[0].cpu() for x in values]))
        if len(records)%20 == 0:
            write_json(out/'progress.json', dict(queries=len(records), seconds=time.monotonic()-began))
            print(f'PAIRED arm={arm} queries={len(records)}', flush=True)
    if arm == 'previous_joint_student_states':
        for i, raw in enumerate(collected_samples(c)):
            if smoke and i >= 7:
                break
            s = move_batch(raw, device)
            torch.cuda.synchronize(); tick = time.perf_counter()
            pred = prediction(model, s)
            torch.cuda.synchronize(); milliseconds = (time.perf_counter()-tick)*1000
            measure(s, pred, dict(window=i, task=raw['metadata']['task_id'], interval=8, age=raw['age'],
                source=raw['metadata']['driver']+'_collected_states'), milliseconds)
    else:
        ids = _balanced_indices(heldout.identities, limit=c['heldout_windows'], seed=c['seed'])
        if smoke:
            ids = ids[:1]
        write_json(out/'selected_windows.json',dict(indices=ids,identities=[heldout.identities[i] for i in ids]))
        for index in ids:
            seq = move_batch(collate_exact_teacher_sequences([heldout[index]]), device)
            for interval in (4, 8):
                for age in range(1, interval):
                    s = sample_from_sequence(seq, age)
                    torch.cuda.synchronize(); tick = time.perf_counter()
                    if checkpoint_name == 'previous_joint':
                        pred = prediction(model, s)
                    else:
                        pred, _ = unroll(model, seq, age, interval)
                    torch.cuda.synchronize(); milliseconds = (time.perf_counter()-tick)*1000
                    measure(s, pred, dict(window=index, interval=interval, age=age,
                        source='cached_original_states'), milliseconds)
    if not records:
        raise RuntimeError('Empty paired analysis')
    torch.save(dict(order=['teacher10','teacher3','predicted10','predicted3'], actions=torch.stack(tensors)), out/'paired_actions.pt')
    write_json(out/'query_metrics.json',records)
    write_json(dest, dict(verdict=verdict, identity=run_id, arm=arm, queries=len(records), smoke=smoke,
        groups=aggregate(records), checkpoint=spec, seconds=time.monotonic()-began,
        gpu=torch.cuda.get_device_name(0), action_tensor_sha256=sha(out/'paired_actions.pt'),
        protocol=c['analysis_protocol'],
        interpretation='Same-observation interventions, not closed-loop success attribution. Vector errors add with an interaction; absolute/MSE norms do not. Condition timing includes the entire unroll prefix for recurrent rows.'))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--arm',choices=JOBS); a=p.parse_args()
    if a.arm:
        analyze(read_json(OUTPUT/'runtime_config.json'),a.arm,smoke=a.smoke)
        return 0
    c=configuration(); OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'runtime_config.json').exists():
        if read_json(OUTPUT/'runtime_config.json') != c:
            raise RuntimeError('Prepared analysis config changed; select a new output')
        identity(c)
    prepare(c); write_json(OUTPUT/'runtime_config.json',c)
    run_id=identity(c)
    def jobs(smoke, arms):
        root = OUTPUT/'smoke' if smoke else OUTPUT
        return [dict(id=arm,cmd=[c['python'],'-u','-m',__spec__.name,'--arm',arm]+(['--smoke'] if smoke else []),
            summary=str(root/arm/'summary.json'),completion=dict(verdict='PAIRED_SMOKE_PASS' if smoke else 'PAIRED_ANALYSIS_COMPLETE',
                identity=run_id,arm=arm,smoke=smoke)) for arm in arms]
    if a.preflight:
        print('CPU_PREFLIGHT_PASS: six bounded paired analyses, no training, no new data collection',flush=True)
        return 0
    smoke_rc=run_queue(OUTPUT/'smoke',jobs(True,JOBS),gpus=(4,5,6,7),predecessor=PREDECESSOR,
        environment=lambda gpu:environment(c,gpu),cwd=ROOT)
    if a.smoke:
        return smoke_rc
    from tools.simvla.gpu_followup_queue import completed
    ready = [j['id'] for j in jobs(True,JOBS) if completed(j)]
    rc=run_queue(OUTPUT,jobs(False,ready),gpus=(4,5,6,7),predecessor=PREDECESSOR,
        environment=lambda gpu:environment(c,gpu),cwd=ROOT)
    reports={a:read_json(OUTPUT/a/'summary.json') for a in JOBS if (OUTPUT/a/'summary.json').exists()}
    write_json(OUTPUT/'comparison_summary.json',dict(complete=len(reports)==len(JOBS),rows=reports,
        inference='Action-error decomposition on matched inputs; no claim of task SR improvement'))
    return int(bool(rc or smoke_rc))


if __name__=='__main__':
    raise SystemExit(main())
