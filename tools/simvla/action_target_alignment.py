"""Matched NFE1-target training and removal of the second Condition updater."""
import argparse
from pathlib import Path

from tools.simvla.condition_deployment_pipeline import ARM
from tools.simvla.condition_interval_recovery import FRESH_CONTROL, verify_dataset
from tools.simvla.fixed_condition_pipeline import OUTPUT as PRIOR, checkpoint_spec, checked_payload
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.error_compensation_common import ROOT, environment, identity, read_json, write_json
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.observation_correction_pipeline import export_arm

OUTPUT = PRIOR.parent / 'action_target_alignment_nfe1_seed01_v1'
MODES = ('fixed_predictor', 'concurrent_updates')
EXTRA = ['tools/simvla/action_target_alignment.py',
         'architectures/simvla/wrappers/run_action_target_alignment.sh']


def configurations():
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    source = checkpoint_spec(FRESH_CONTROL / 'train' / ARM / 'summary.json')
    payload = checked_payload(source)
    previous = read_json(PRIOR / 'fresh_fixed/runtime_config.json')
    if payload['contract']['action_mode'] != 'naive1' or source['steps'] != 10000:
        raise RuntimeError('Expected the completed fresh NFE1-only 10K parent')
    _, cache_payload = load_native_v0_checkpoint(previous['condition_checkpoint'], device='cpu', require_final_150k=True)
    configs = {}
    for mode in MODES:
        control_name = 'fresh_fixed' if mode == 'fixed_predictor' else 'fresh_joint'
        control_config = read_json(PRIOR / control_name / 'runtime_config.json')
        # Keep all sampling and optimizer choices identical to the completed NFE10-target arm.
        c = {**control_config, 'output': str(OUTPUT / mode), 'teacher_steps': 1,
             'teacher_transition': dict(source=10, target=1),
             'freeze_condition_predictor': mode == 'fixed_predictor',
             'run_label': 'action_target_alignment_' + mode,
             'rb2_destination': 'rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_action_target_alignment/' + mode,
             'extra_source_files': sorted(set(control_config['extra_source_files'] + EXTRA)),
             'matched_original10_control': str(PRIOR / control_name),
             'training_description': 'Only action teacher changes from original10 to original1 at the same observation, proprioception and initial noise. Same fresh NFE1-only10K parent, sampled windows10001..15000, 5K AdamW, K4/K8 ages. Frozen predictor or existing detached-input concurrent learning. Original backbone/action transformer frozen. No SR gate.'}
        if checkpoint_spec(c['initial_models'][ARM]['summary'])['sha256'] != source['sha256']:
            raise RuntimeError('NFE1/NFE10 target arms have different initial weights')
        for key in ('steps', 'warmup_steps', 'sample_step_offset', 'learning_rate', 'seed',
                    'training_intervals', 'condition_weight', 'initial_models', 'cache',
                    'heldout_fraction', 'batch_size'):
            if c.get(key) != control_config.get(key):
                raise RuntimeError('Unmatched training setting: ' + key)
        observed = verify_dataset(c, cache_payload, payload['contract'])
        prepare(c)
        write_json(Path(c['output']) / 'runtime_config.json', c)
        write_json(Path(c['output']) / 'dataset_preflight.json', dict(verdict='ACTUAL_DATASET_PASS', **observed))
        configs[mode] = c
    c = {**configs['fixed_predictor'], 'output': str(OUTPUT / 'predictor_only'),
         'source_model': source, 'use_action_condition_updater': False,
         'evaluation_condition_intervals': [2, 3, 4],
         'training_description': 'No new training. Use the fresh NFE1-only10K parent encoder and first Condition updater. Pass its current output directly to frozen NFE1 action transformer and carry that same output to the next query. Second updater has zero forward calls. Compare existing full two-updater parent at each identical K.'}
    prepare(c)
    write_json(Path(c['output']) / 'runtime_config.json', c)
    configs['predictor_only'] = c
    return configs


def predictor_only_policy(c, arm, *, smoke=False, k_c=4):
    from tools.simvla.error_compensation_eval import make_policy
    from tools.simvla.condition_output_split_eval import attach
    policy = make_policy(c, 'condition_naive1', k_c=k_c)
    return attach(policy, policy.native_v0, checked_payload(c['source_model']), ARM, k_c,
                  use_action_condition_updater=False)


def jobs(configs):
    plan = []
    for mode in MODES:
        c = configs[mode]
        out = Path(c['output'])
        def add(kind, module, args, summary, completion, deps):
            key = mode + '_' + kind
            plan.append(dict(id=key, cmd=[c['python'], '-u', '-m', module, '--config', str(out / 'runtime_config.json'), '--arm', ARM] + args,
                             summary=str(summary), completion=dict(identity=identity(c), **completion), deps=deps))
            return key
        smoke = add('smoke_train', 'tools.simvla.condition_output_split_train', ['--smoke'], out / 'smoke' / ARM / 'summary.json', dict(verdict='SMOKE_PASS', steps=14), [])
        env = add('smoke_env', 'tools.simvla.condition_output_split_eval', ['--smoke', '--k-c', '4'], out / 'eval_smoke' / ('kc4_' + ARM) / 'summary.json', dict(verdict='SMOKE_PASS', episodes=1), [smoke])
        train = add('train', 'tools.simvla.condition_output_split_train', [], out / 'train' / ARM / 'summary.json', dict(verdict='TRAIN_AND_OFFLINE_COMPLETE', steps=5000), [env])
        add('export', 'tools.simvla.action_target_alignment', ['--export'], out / 'exports' / (ARM + '.json'), dict(verdict='BUNDLE_EXPORTED'), [train])
    c = configs['predictor_only']
    out = Path(c['output'])
    for smoke in (True, False):
        for k in ((4,) if smoke else (4, 3, 2)):
            name = 'predictor_only_smoke' if smoke else 'predictor_only_k' + str(k)
            plan.append(dict(id=name, cmd=[c['python'], '-u', '-m', 'tools.simvla.action_target_alignment', '--predictor-only', '--config', str(out / 'runtime_config.json'), '--k-c', str(k)] + (['--smoke'] if smoke else []),
                             summary=str(out / ('eval_smoke' if smoke else 'online') / (f'kc{k}_' + ARM) / 'summary.json'),
                             completion=dict(identity=identity(c), verdict='SMOKE_PASS' if smoke else 'EVALUATION_COMPLETE', episodes=1 if smoke else 500),
                             deps=[] if smoke else ['predictor_only_smoke']))
    for k in (4, 3, 2):
        for mode in MODES:
            c = configs[mode]
            out = Path(c['output'])
            plan.append(dict(id=f'{mode}_k{k}', cmd=[c['python'], '-u', '-m', 'tools.simvla.condition_output_split_eval', '--config', str(out / 'runtime_config.json'), '--arm', ARM, '--k-c', str(k)],
                             summary=str(out / 'online' / (f'kc{k}_' + ARM) / 'summary.json'),
                             completion=dict(identity=identity(c), verdict='EVALUATION_COMPLETE', episodes=500), deps=[mode + '_train']))
    return plan


def summarize(configs):
    rows = {}
    controls = {}
    for mode, c in configs.items():
        for p in Path(c['output']).glob('online/*/summary.json'):
            rows[mode + '/' + p.parent.name] = read_json(p)
        if mode in MODES:
            for p in Path(c['matched_original10_control']).glob('online/*/summary.json'):
                controls[mode + '/' + p.parent.name] = read_json(p)
    write_json(OUTPUT / 'comparison_summary.json', dict(complete=len(rows) == 9, rows=rows,
               original10_target_controls=controls, predictor_only_parent=str(FRESH_CONTROL),
               hardware='sd1 RTX3090 eager', scope='LIBERO-Long500 seed01,H10/R5,NFE1; complete rows only'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--predictor-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--arm', choices=(ARM,))
    parser.add_argument('--k-c', type=int, choices=(2, 3, 4), default=4)
    args = parser.parse_args()
    if args.export:
        export_arm(read_json(args.config), ARM)
        return 0
    if args.predictor_only:
        from tools.simvla.error_compensation_eval import run
        from tools.simvla.condition_output_split_eval import check_policy
        def checker(policy, row, calls, k):
            result = check_policy(policy)
            if calls.get('condition', 0) != result['condition_updater'] or calls.get('transformer', 0) != result['transformer']:
                raise RuntimeError('Independent invocation counters disagree')
            return result
        run(read_json(args.config), ARM, k_c=args.k_c, smoke=args.smoke,
            policy_factory=predictor_only_policy, counter_checker=checker)
        return 0
    configs = configurations()
    plan = jobs(configs)
    write_json(OUTPUT / 'planned_jobs.json', plan)
    summarize(configs)
    if args.preflight:
        print('PREFLIGHT_PASS: two matched5K trainings, nine Long500 rows, existing controls reused', flush=True)
        return 0
    status = OUTPUT / 'pipeline_status.json'
    write_json(status, dict(phase='waiting_for_predecessor'))
    rc = 1
    try:
        rc = run_queue(OUTPUT, plan, gpus=(4, 5, 6, 7),
                       predecessor=dict(path=str(PRIOR), lock='queue.lock', allow_when_all_assigned=True),
                       environment=lambda gpu: environment(configs['fixed_predictor'], gpu), cwd=ROOT, timeout=12 * 3600)
    finally:
        summarize(configs)
        write_json(status, dict(phase='complete' if not rc else 'finished_with_failures'))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
