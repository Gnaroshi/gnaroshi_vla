"""Descriptive summaries; do not equate teacher error or sensitivity with causality."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .config import ROWS
from .runtime import atomic_json


def summarize(root, report_directory=None):
    root = Path(root)
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    rows = {}
    for variant in ROWS:
        rows[variant] = [json.loads(p.read_text()) for p in sorted((root / variant).glob('episode_*.json'))]
    reference = {r['eval_id']: r for r in rows['full']}
    records = []
    for variant, data in rows.items():
        if not data:
            continue
        common = [(reference[r['eval_id']], r) for r in data
                  if r['success'] and reference.get(r['eval_id'], {}).get('success')]
        sr = 100. * sum(r['success'] for r in data) / len(data)
        records.append({'variant': variant, 'episodes': len(data), 'success_rate_percent': sr,
                        'gain_pp_vs_full': sr - 100. * sum(r['success'] for r in rows['full']) / len(rows['full']),
                        'both_success_n': len(common),
                        'both_success_full_mean_steps': float(np.mean([a['env_steps'] for a, _ in common])) if common else None,
                        'both_success_variant_mean_steps': float(np.mean([b['env_steps'] for _, b in common])) if common else None,
                        'both_success_mean_steps_saved': float(np.mean([a['env_steps'] - b['env_steps'] for a, b in common])) if common else None})
    if records:
        with (output / 'rollout_comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    # First average each metric within an episode, then across episodes.
    diagnostics = defaultdict(list)
    for path in sorted((root / 'same_input').glob('episode_*.pt')):
        data = torch.load(path, map_location='cpu')
        groups = defaultdict(list)
        for row in data['diagnostic_records']:
            groups[row['mode'], row['variant'], row['runtime_age']].append(row)
        for key, values in groups.items():
            metrics = {k: float(np.mean([v[k] for v in values])) for k in values[0]
                       if k not in ('timestep', 'runtime_age', 'updater_age', 'proprio_current_source_step',
                                    'proprio_previous_source_step') and isinstance(values[0][k], (int, float))}
            diagnostics[key].append(metrics)
    diagnostic_summary = [dict(mode=key[0], variant=key[1], age=key[2], episodes=len(values),
                              metrics={k: float(np.mean([v[k] for v in values])) for k in values[0]})
                          for key, values in sorted(diagnostics.items())]
    atomic_json(output / 'same_input_episode_balanced.json', diagnostic_summary)
    external, excluded, response_records = defaultdict(list), [], []
    for path in sorted((root / 'external').glob('episode_*.pt')):
        data = torch.load(path, map_location='cpu')
        if 'skip_reason' in data:
            excluded.append({k: data[k] for k in ('task_id', 'trial', 'skip_reason')})
            continue
        lookup = {(b['scene'], b['variant']): b for b in data['branches']}
        for branch in data['branches']:
            external[branch['scene'], branch['variant']].append(branch)
        for variant in ('full', 'normal', 'zero', 'stale_visual', 'hold'):
            nominal, changed = lookup['nominal', variant], lookup['displaced', variant]
            before = np.asarray(nominal['before_refresh_actions'])
            after = np.asarray(changed['before_refresh_actions'])
            first_delta = after[0] - before[0]
            full_delta = np.asarray(lookup['displaced', 'full']['before_refresh_actions'])[0] - np.asarray(lookup['nominal', 'full']['before_refresh_actions'])[0]
            denominator = np.linalg.norm(first_delta[:6]) * np.linalg.norm(full_delta[:6])
            response_records.append({'task_id': data['task_id'], 'trial': data['trial'], 'variant': variant,
                'image_change_l1': data['observation_image_l1'],
                'nominal_repeat_image_l1': data.get('nominal_repeat_image_l1'),
                'first_arm_response_l2': float(np.linalg.norm(first_delta[:6])),
                'first_gripper_response': float(first_delta[-1]),
                'response_cosine_to_full': float(np.dot(first_delta[:6], full_delta[:6]) / denominator) if denominator > 1e-12 else None,
                'nominal_success': nominal['success'], 'displaced_success': changed['success'],
                'full_policy_recovers_displacement': lookup['displaced', 'full']['success']})
    external_summary = [dict(scene=k[0], variant=k[1], eligible_snapshots=len(v),
                             success_rate_percent=100 * sum(b['success'] for b in v) / len(v)) for k, v in sorted(external.items())]
    atomic_json(output / 'external_response_and_recovery.json',
                {'summary': external_summary, 'excluded': excluded, 'paired_response_records': response_records})
    lines = ['# Seer freshness experiment results', '',
             'Frozen public33 + adapter39; seed 42 by default. No training. EGL. K=4, H=3, temporal ensembling.', '',
             '| Variant | Episodes | SR (%) | Gain vs full (pp) | Both-success n | Steps saved |',
             '|---|---:|---:|---:|---:|---:|']
    for r in records:
        saved = r['both_success_mean_steps_saved']
        lines.append(f"| {r['variant']} | {r['episodes']} | {r['success_rate_percent']:.2f} | "
                     f"{r['gain_pp_vs_full']:+.2f} | {r['both_success_n']} | {saved if saved is not None else '-'} |")
    lines.extend(['', '## Interpretation boundaries', '',
        '- Same-input metrics use baseline full-policy trajectories and identical cached observations, not separate EGL rerenders.',
        '- Teacher-input age=1, teacher-input runtime age, and recursive runtime age are separate conditions.',
        '- The legacy updater takes the FIRST state-history element. With history=7 this is q[t-6] after warmup; current_proprio is a separate intervention, not a silent fix.',
        '- d=0 removes the fused encoder vector, not only image difference. Repeated observations still pass their absolute values through the learned encoder.',
        '- Gate/residual crosses use the SAME previous latent, dt and age. They do not retrain the networks.',
        '- Both-success step savings condition on success of BOTH policies; this is descriptive and not unbiased causal time-to-completion.',
        '- EGL rollouts share task/trial IDs and seeds, but not byte-identical RGB. Same-input and external initial branches do share exact tensors.',
        '- External branches replay the same recorded action prefix, check simulator state, and share the cached latent, past action horizons and robot state at intervention.',
        '- External full policy uses the common LatentLoop prefix. Its branch recovery rate is NOT an ordinary full-policy benchmark SR.',
        '- A full-policy response is not a ground-truth optimal action. Interpret recovery, paired nominal controls, and action response together.',
        '- No diagnostic wall time is reported as method latency. No result is a multi-seed result.',
        '', 'See same_input_episode_balanced.json and external_response_and_recovery.json for measured values.',
        f'External ineligible snapshots: {len(excluded)}. No missing snapshot is imputed as success or failure.', ''])
    report_dir = (Path(report_directory) if report_directory is not None else
                  Path(__file__).resolve().parents[4] / 'codex_outputs/seer/freshness' / root.name)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'report.md').write_text('\n'.join(lines))
    atomic_json(output / 'summary.json', {'rollouts': records, 'external': external_summary})
