"""Frozen trend K4/K8 evaluation on the existing compiled paper protocol."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import sys
import traceback
from types import MethodType

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import DEFAULT_CONFIG, ROOT, Replay, configure, read_json, sha, write_json
from tools.simvla.compiled_policy import attach_policy, check_reset

CONFIG = ROOT / 'architectures/simvla/configs/trend_compiled_rb2.json'
ROWS = {'trend_k4': 4, 'trend_k8': 8}


def trend_payload(c):
    import torch
    spec = c['trend_checkpoint']
    if sha(spec['path']) != spec['sha256']:
        raise RuntimeError('Trend checkpoint checksum mismatch')
    payload = torch.load(spec['path'], map_location='cpu', weights_only=False)
    expected = dict(format='simvla_trend_condition_v1', arm='trend_only', step=3000, identity=spec['identity'])
    if any(payload.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Trend checkpoint provenance mismatch')
    return payload


def condition_at_age(context, age, k_c):
    if k_c not in (4, 8) or not 1 <= age < k_c:
        raise ValueError('Age outside the frozen trend evaluation window')
    return context.anchor + age * context.trend


def expected_counts(row, queries):
    full = (queries + ROWS[row] - 1) // ROWS[row]
    return dict(num_full_vlm_calls=full, num_condition_updater_calls=queries-full,
        num_action_transformer_calls=3*queries, num_generation_decoder_only_steps=7*queries,
        num_trend_head_calls=full, num_observation_encoder_calls=0)


def check_policy(policy, row):
    q = int(policy.metrics.counters['num_policy_queries'])
    for key, expected in expected_counts(row, q).items():
        if int(policy.metrics.counters.get(key, 0)) != expected:
            raise RuntimeError(f'{row}: counter {key} differs from {expected}')
    if q != (policy.step_index + 4) // 5:
        raise RuntimeError('H10/R5 action queue contract changed')


def check_compiler(compiler, row):
    required = ('vlm', 'action_transformer', 'action_decoder', 'generation_updater', 'trend_head')
    missing = [name for name in required if compiler.records.get(name, {}).get('graphs', 0) == 0]
    if missing:
        raise RuntimeError('Compile bypass: ' + str(missing))


def reset_policy(policy):
    check_reset(policy)
    if policy._trend_context is not None:
        raise RuntimeError('Trend context leaked across episodes')


def replay_factory(c, row, compiler, samples):
    return Replay(c, 'ours_kc2_ng3', compiler, samples)


def make_policy(replay, c, row, manifest):
    import torch
    from methods.latentloop.modules.trend_condition import TrendCondition
    policy = attach_policy(replay, c, 'ours_kc2_ng3', manifest)
    model = TrendCondition(replay.native, 'trend_only').to('cuda').eval().requires_grad_(False)
    model.load_state_dict(trend_payload(c)['model'], strict=True)
    if sum(p.numel() for p in model.parameters()) != 126400:
        raise RuntimeError('Unexpected trend architecture')
    model.trend_head.forward = replay.compiler.wrap('trend_head', model.trend_head.forward)
    original_full, original_reset = policy._full_refresh, policy.reset
    policy.native_v0 = model
    policy.row_name = policy.mode = row
    policy.k_c = policy.refresh_every = ROWS[row]

    def reset(self):
        original_reset()
        self._trend_context = None

    def full(self, batch, *, policy_query_index):
        condition, action, seed = original_full(batch, policy_query_index=policy_query_index)
        self._trend_context = model.prepare(condition, batch['raw_rgb'], batch['proprio'],
            self.condition_layout.valid_mask, self.condition_layout.group_ids)
        self.metrics.counters['num_trend_head_calls'] += 1
        return condition, action, seed

    def update(self, batch, *, age, policy_query_index):
        if self._trend_context is None:
            raise RuntimeError('Missing original refresh context')
        condition = condition_at_age(self._trend_context, age, self.k_c)
        self.metrics.counters['num_condition_updater_calls'] += 1
        action, seed = self._decode(condition, batch['proprio'], policy_query_index=policy_query_index)
        self.cached_condition = condition.detach()
        self.cached_action_chunk = action.detach()
        return condition, action, seed

    policy.reset = MethodType(reset, policy)
    policy._full_refresh = MethodType(full, policy)
    policy._v0_update = MethodType(update, policy)
    policy.reset()
    return policy


def prepare(c, output):
    trend_payload(c)
    previous = read_json(Path(c['reference_root']) / 'campaign_contract.json')
    for key in ('norm_stats', 'condition_checkpoint', 'generation_checkpoint'):
        if sha(c[key]) != previous['artifacts'][key]:
            raise RuntimeError('Paper comparison input differs: ' + key)
    manifest = campaign.validate_manifest(read_json(campaign.manifest_path(c, 'libero_10', 'seed01')), 'libero_10', 'seed01')
    if manifest['manifest_sha256'] != previous['manifest_hashes']['libero_10/seed01']:
        raise RuntimeError('Paper seed01 episode list differs')
    return campaign.prepare(c, output)


def summarize(c, output):
    results = []
    for row in ROWS:
        p = output / 'rows/libero_10/seed01' / row / 'summary.json'
        if p.exists():
            results.append(dict(row=row, **read_json(p), k_c=ROWS[row], n_g=3,
                training_max_age=3, evaluation_max_age=ROWS[row]-1, extrapolation=ROWS[row]>4))
    prior = read_json(Path(c['reference_root']) / 'combined_summary.json')
    references = [r for r in prior['results'] if r['suite']=='libero_10' and r['seed']=='seed01']
    report = dict(complete=len(results)==2, completed_cells=len(results), planned_cells=2,
        episodes=sum(r['episodes'] for r in results), results=results, references=references,
        reference_root=c['reference_root'], seeds=['seed01'], new_training=False,
        timing='RTX5090 compiled, CUDA-synchronized policy.act / executed actions; excludes env/render/video and compilation-contaminated episodes',
        interpretation='K4 reproduction and K8 extrapolation of the same frozen 3K slope; no slope rescaling',
        uncertainty='Single evaluation seed; compare references from seed01, not their three-seed means')
    write_json(output / 'comparison_summary.json', report)
    return report


def recover(c, output, command, row):
    contract = read_json(output / 'campaign_contract.json')
    smoke = command == 'smoke'
    identity = campaign.digest(dict(campaign=campaign.digest(contract), suite='libero_10', seed='seed01', row=row, smoke=smoke))
    manifest = read_json(output / 'manifests/libero_10/seed01/episode_manifest.json')
    specs = sorted(manifest['episodes'], key=lambda x: (-x['task_id'], x['trial_id']))
    if smoke:
        specs = specs[:c['smoke_episodes']]
    directory = output / ('smoke' if smoke else 'rows') / 'libero_10/seed01' / row
    return campaign.summarize_cell(directory, identity, [(s['task_id'], s['trial_id']) for s in specs])


def run_all(c, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'launcher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(c, output)
        failures = []
        for row in ROWS:
            for command in ('smoke', 'worker'):
                if recover(c, output, command, row):
                    continue
                ok = campaign.run_child(c, output, command, 'libero_10', 'seed01', row)
                if not ok:
                    ok = bool(recover(c, output, command, row))
                if not ok:
                    # Replay the whole row after a technical failure. Preserve
                    # partial histories; never mix a restarted global RNG stream.
                    directory = output / ('smoke' if command=='smoke' else 'rows') / 'libero_10/seed01' / row
                    archive = output / 'failed_attempts' / f'{command}_{row}'
                    if not archive.exists():
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        if directory.exists():
                            directory.rename(archive)
                        else:
                            archive.mkdir()
                        ok = campaign.run_child(c, output, command, 'libero_10', 'seed01', row)
                        ok = ok or bool(recover(c, output, command, row))
                if not ok:
                    failures.append(dict(row=row, stage=command))
                    break
                summarize(c, output)
        summary = summarize(c, output)
        write_json(output / 'status.json', dict(phase='finished', completed_cells=summary['completed_cells'], planned_cells=2, failures=failures))
        return 0 if summary['complete'] and not failures else 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('preflight', 'all', 'smoke', 'worker', 'summarize'))
    p.add_argument('--suite', default='libero_10', choices=('libero_10',))
    p.add_argument('--seed', default='seed01', choices=('seed01',))
    p.add_argument('--row', choices=ROWS, default='trend_k4')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    c = {**read_json(DEFAULT_CONFIG), **read_json(ROOT / 'architectures/simvla/configs/compile_campaign_rb2.json'), **read_json(CONFIG)}
    if args.output:
        c['output'] = str(args.output.resolve())
    configure(c)
    sys.path.insert(0, c['libero_root'])
    os.environ['LIBERO_CONFIG_PATH'] = c['libero_config']
    output = Path(c['output'])
    try:
        if args.command == 'all':
            return run_all(c, output)
        if args.command == 'preflight':
            prepare(c, output)
            print('PREFLIGHT_PASS: K4 then K8; 500 episodes each, no training', flush=True)
        elif args.command == 'summarize':
            summarize(c, output)
        else:
            campaign.worker(c, output, args.suite, args.seed, args.row, smoke=args.command=='smoke',
                replay_factory=replay_factory, policy_factory=make_policy, policy_checker=check_policy,
                compiler_checker=check_compiler, reset_checker=reset_policy)
        return 0
    except BaseException as exc:
        if args.command == 'all':
            write_json(output / 'status.json', dict(phase='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=f'{type(exc).__name__}: {exc}'))
        traceback.print_exc()
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
