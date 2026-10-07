"""Frozen-checkpoint factorial interventions: recurrent history vs observation estimate."""
import argparse
from pathlib import Path
import time
from types import MethodType

import torch

from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from tools.simvla.error_compensation_common import configure, environment, identity, snapshots
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.error_compensation_eval import make_policy as base_policy, run as evaluate
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.observation_correction_pipeline import OUTPUT as SOURCE
from tools.simvla.observation_correction_eval import attach, check_policy, load_payload
from tools.simvla.observation_correction_train import unroll
from tools.simvla.paired_error_analysis import aggregate, paired_metrics
from tools.simvla.rollout_state_repair import sample_from_sequence
from methods.latentloop.modules.observation_correction import ObservationCorrection
from methods.latentloop.modules.trend_condition import scaled_mse
from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices

MODES = ('normal', 'teacher_history', 'teacher_measurement', 'teacher_both')
OUTPUT = SOURCE.parents[1] / 'condition_state_audit/frozen_observed_k8_seed01_v1'
DESCRIPTION = {
    'normal': '기존 예측 기억과 기존 관측 추정값을 그대로 사용',
    'teacher_history': '이전 query의 원본 Condition을 기억에 대입; 현재 관측 추정은 기존 모델',
    'teacher_measurement': '기억은 자기 예측 유지; 현재 관측 추정값만 원본 Condition으로 교체',
    'teacher_both': '이전 기억과 현재 관측 추정값 모두 원본 Condition으로 교체',
}


def intervention(model, context, age, images, proprio, mode, previous_teacher, current_teacher):
    if mode not in MODES:
        raise ValueError(mode)
    for teacher in (previous_teacher, current_teacher):
        if teacher.shape != context.anchor.shape or not torch.isfinite(teacher).all():
            raise ValueError('Invalid teacher Condition')
    if mode in ('teacher_history', 'teacher_both'):
        context.previous = torch.where(context.valid.unsqueeze(-1), previous_teacher, context.anchor)
    condition, diagnostics = model.predict(context, age, images, proprio)
    if mode in ('teacher_measurement', 'teacher_both'):
        # Freeze the gate from the learned estimate within this same history branch.
        # Recomputing the gate on the oracle would change two variables at once.
        condition = diagnostics['predicted'] + diagnostics['gain'] * (
            current_teacher - diagnostics['predicted'])
        condition = torch.where(context.valid.unsqueeze(-1), condition, context.anchor)
        context.previous = condition
    return condition, diagnostics


def configuration():
    c = read_json(SOURCE / 'runtime_config.json')
    summary = read_json(SOURCE / 'train/observed_recurrent/summary.json')
    if summary['verdict'] != 'TRAIN_AND_OFFLINE_COMPLETE':
        raise RuntimeError('Source checkpoint is incomplete')
    c.update(output=str(OUTPUT), heldout_windows=100, smoke_policy_actions=41,
        evaluation_condition_intervals=[8], extra_source_files=[
            'tools/simvla/condition_state_audit.py',
            'architectures/simvla/wrappers/run_condition_state_audit.sh'],
        audit_checkpoint=dict(path=summary['checkpoint'], sha256=summary['checkpoint_sha256'],
            source_identity=summary['identity']),
        training_description='No training. Frozen observed_recurrent checkpoint, 2x2 oracle intervention.',
        student_condition_description='Separate previous predicted memory and current independent observation estimate.',
        audit_protocol=dict(modes=DESCRIPTION, intervals_offline=[4, 8], interval_online=8,
            episodes_per_mode=500, action_nfe=3, generation_updater=False,
            oracle_gate='Learned gate unchanged by measurement substitution; previous-history changes may affect it.',
            online_teacher='Full VLM on EVERY query for every mode; discarded in normal mode.',
            interpretation='Oracle diagnosis, not a deployable acceleration method. No causality claim from latent MSE alone.',
            timing='Whole online policy.act includes shadow VLM and intervention checks; not paper speedup.',
            limitations='History replacement also changes predictor input distribution. Does not isolate action-loss-induced memory damage.'))
    if sha(c['audit_checkpoint']['path']) != c['audit_checkpoint']['sha256']:
        raise RuntimeError('Source checkpoint hash mismatch')
    return c


def payload(c):
    spec = c['audit_checkpoint']
    if sha(spec['path']) != spec['sha256']:
        raise RuntimeError('Checkpoint changed')
    return load_payload(spec['path'], 'observed_recurrent',
        expected_identity=spec['source_identity'], steps=3000)


@torch.inference_mode()
def offline(c, mode, smoke=False):
    configure(c)
    run_id = identity(c)
    out = OUTPUT / ('offline_smoke' if smoke else 'offline') / mode
    out.mkdir(parents=True, exist_ok=True)
    device, parent, frozen, action, _, heldout = load_runtime(c, snapshots(c))
    model = ObservationCorrection(parent, 'observed_recurrent').to(device).eval()
    model.load_state_dict(payload(c)['model'], strict=True)
    model.requires_grad_(False)
    indices = _balanced_indices(heldout.identities, limit=c['heldout_windows'], seed=c['seed'])
    if smoke:
        indices = indices[:1]
    write_json(out / 'windows.json', dict(indices=indices, identities=[heldout.identities[i] for i in indices]))
    records, arrays = [], []
    began = time.monotonic()
    for index in indices:
        sequence = move_batch(collate_exact_teacher_sequences([heldout[index]]), device)
        for interval in (4, 8):
            context = model.prepare(sequence['anchor_condition'], sequence['image_sequence'][:, 0],
                sequence['proprio_sequence'][:, 0], sequence['valid_mask'].bool(), sequence['group_ids'], interval)
            for age in range(1, interval):
                s = sample_from_sequence(sequence, age)
                previous = s['anchor'] if age == 1 else sequence['teacher_conditions'][:, age - 2]
                torch.cuda.synchronize(); tick = time.perf_counter()
                predicted, d = intervention(model, context, age, s['images'], s['proprio'],
                    mode, previous, s['target_condition'])
                torch.cuda.synchronize(); condition_ms = (time.perf_counter() - tick) * 1000
                if mode == 'normal' and index == indices[0]:
                    reference, _ = unroll(model, sequence, age, interval)
                    torch.testing.assert_close(predicted, reference, rtol=0, atol=0)
                outputs, times = [], {}
                for label, condition, steps in (
                    ('teacher10', s['target_condition'], 10), ('teacher3', s['target_condition'], 3),
                    ('predicted10', predicted, 10), ('predicted3', predicted, 3),
                    ('measurement3', d['measured'], 3), ('prediction3', d['predicted'], 3)):
                    torch.cuda.synchronize(); tick = time.perf_counter()
                    decoded = action.decode_action_from_condition(condition, s['proprio'], steps=steps,
                        initial_noise=s['noise'].clone(), return_debug=True)
                    torch.cuda.synchronize()
                    times[label + '_decode_ms'] = (time.perf_counter() - tick) * 1000
                    if int(decoded.debug['iterations']) != steps:
                        raise RuntimeError('Unexpected action transformer calls')
                    if label == 'teacher10':
                        difference = float((decoded.action - s['target_action']).abs().max())
                        if difference > 2e-4:
                            raise RuntimeError(f'Cache/teacher mismatch: {difference}')
                    outputs.append(decoded.final_action_latent)
                records.append(dict(window=index, interval=interval, age=age, source='cached_original_states',
                    **paired_metrics(*outputs[:4]), **times, condition_prediction_ms=condition_ms,
                    teacher_cache_max_abs=difference,
                    latent_condition_normalized_mse=float(scaled_mse(predicted, s['target_condition'], s['anchor'], s['valid'])),
                    learned_gain_mean=float(d['gain'][s['valid']].mean()),
                    measurement_action_l1=float((outputs[4][:, :5] - outputs[0][:, :5]).abs().mean()),
                    prediction_action_l1=float((outputs[5][:, :5] - outputs[0][:, :5]).abs().mean())))
                arrays.append(torch.stack([x[0].cpu() for x in outputs]))
        write_json(out / 'progress.json', dict(windows=indices.index(index)+1, queries=len(records),
            seconds=time.monotonic()-began))
        print(f'OFFLINE {mode} windows={indices.index(index)+1}/{len(indices)} queries={len(records)}', flush=True)
    torch.save(dict(order=['teacher10','teacher3','predicted10','predicted3','measurement3','prediction3'],
        actions=torch.stack(arrays)), out / 'paired_actions.pt')
    write_json(out / 'query_metrics.json', records)
    write_json(out / 'summary.json', dict(verdict='OFFLINE_COMPLETE', identity=run_id, mode=mode,
        smoke=smoke, windows=len(indices), queries=len(records), groups=aggregate(records),
        seconds=time.monotonic()-began, gpu=torch.cuda.get_device_name(0),
        checkpoint=c['audit_checkpoint'], protocol=c['audit_protocol']))


def make_policy(c, mode, *, smoke=False, k_c=8):
    policy = base_policy(c, 'condition_naive3', k_c=4)
    attach(policy, policy.native_v0, payload(c), 'observed_recurrent', k_c)
    original_full, original_reset = policy._full_refresh, policy.reset
    previous_extra = policy.extra_episode_metrics
    def reset(self):
        original_reset()
        self._audit_previous_teacher = None
        self._audit_oracle_calls = 0
        self._audit_oracle_ms = 0.0
    def full(self, batch, *, policy_query_index):
        result = original_full(batch, policy_query_index=policy_query_index)
        self._audit_previous_teacher = result[0].detach()
        return result
    def update(self, batch, *, age, policy_query_index):
        if self._audit_previous_teacher is None:
            raise RuntimeError('Missing previous-query teacher')
        torch.cuda.synchronize(); tick = time.perf_counter()
        teacher = self.condition_adapter.encode_condition(input_ids=batch['input_ids'],
            image_input=batch['image_input'], image_mask=batch['image_mask'])
        torch.cuda.synchronize()
        self._audit_oracle_ms += (time.perf_counter()-tick)*1000
        self._audit_oracle_calls += 1
        condition, _ = intervention(self.native_v0, self._trend_context, age, batch['raw_rgb'],
            batch['proprio'], mode, self._audit_previous_teacher, teacher)
        self._audit_previous_teacher = teacher.detach()
        self.metrics.counters['num_condition_updater_calls'] += 1
        self.metrics.counters['num_observation_encoder_calls'] += 2
        action, seed = self._decode(condition, batch['proprio'], policy_query_index=policy_query_index)
        self.cached_condition, self.cached_action_chunk = condition.detach(), action.detach()
        return condition, action, seed
    policy.reset = MethodType(reset, policy)
    policy._full_refresh = MethodType(full, policy)
    policy._v0_update = MethodType(update, policy)
    policy.row_name = mode
    policy.extra_episode_metrics = lambda: dict(**previous_extra(), oracle_diagnostic=True,
        oracle_extra_vlm_calls=policy._audit_oracle_calls, oracle_extra_vlm_ms=policy._audit_oracle_ms,
        intervention=DESCRIPTION[mode])
    policy.reset()
    return policy


def check_online(policy, row, calls, k):
    result = check_policy(policy)
    extra = result['queries'] - result['full_vlm']
    if (policy._audit_oracle_calls != extra or calls.get('condition', 0) != extra
            or calls.get('transformer', 0) != 3 * result['queries']):
        raise RuntimeError('Independent invocation counter mismatch')
    return dict(**result, oracle_extra_full_vlm=extra, actual_total_full_vlm=result['queries'])


def online(c, mode, smoke=False):
    evaluate(c, mode, smoke=smoke, k_c=8, policy_factory=make_policy, counter_checker=check_online)
    p = OUTPUT / ('eval_smoke' if smoke else 'online') / f'kc8_{mode}' / 'summary.json'
    summary = read_json(p)
    summary.update(oracle_diagnostic=True, new_training=False, description=DESCRIPTION[mode],
        protocol=c['audit_protocol'], latency_includes_shadow_vlm=True,
        checkpoint=c['audit_checkpoint'], candidate_training_condition_intervals=[4, 8])
    summary.pop('candidate_training_k_c', None)
    write_json(p, summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--mode', choices=MODES)
    parser.add_argument('--phase', choices=('smoke', 'offline', 'online'))
    args = parser.parse_args()
    if args.mode:
        c = read_json(OUTPUT / 'runtime_config.json')
        if args.phase == 'smoke':
            offline(c, args.mode, True)
            online(c, args.mode, True)
        elif args.phase == 'offline':
            offline(c, args.mode)
        elif args.phase == 'online':
            online(c, args.mode)
        else:
            parser.error('--mode requires --phase')
        return 0
    c = configuration()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prepare(c)
    write_json(OUTPUT / 'runtime_config.json', c)
    run_id = identity(c)
    if args.preflight:
        print('PREFLIGHT_PASS: frozen checkpoint; four diagnostic modes; 100 windows and 500 episodes each', flush=True)
        return 0
    jobs = []
    for phase in ('smoke', 'offline', 'online'):
        for mode in MODES:
            folder = 'offline' if phase == 'offline' else 'eval_smoke' if phase == 'smoke' else 'online'
            name = mode if phase == 'offline' else f'kc8_{mode}'
            completion = dict(identity=run_id, verdict='OFFLINE_COMPLETE' if phase == 'offline'
                else 'SMOKE_PASS' if phase == 'smoke' else 'EVALUATION_COMPLETE')
            if phase != 'offline':
                completion['oracle_diagnostic'] = True
            deps = [] if phase == 'smoke' else [f'smoke_{mode}'] if phase == 'offline' else [f'offline_{mode}']
            jobs.append(dict(id=f'{phase}_{mode}', deps=deps,
                cmd=[c['python'],'-u','-m',__spec__.name,'--mode',mode,'--phase',phase],
                summary=str(OUTPUT / folder / name / 'summary.json'), completion=completion))
    rc = run_queue(OUTPUT, jobs, gpus=(4,5,6,7), predecessor=[],
        environment=lambda gpu: environment(c,gpu), cwd=ROOT, timeout=86400)
    rows = {}
    for mode in MODES:
        p = OUTPUT / 'online' / f'kc8_{mode}' / 'summary.json'
        if p.exists():
            rows[mode] = read_json(p)
    write_json(OUTPUT / 'comparison_summary.json', dict(complete=len(rows)==len(MODES) and rc==0,
        rows=rows, protocol=c['audit_protocol'], new_training=False))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
