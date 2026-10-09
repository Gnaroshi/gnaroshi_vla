"""Complete a paired NFE1 comparison without repeating verified 500-episode rows."""
import argparse
import csv
import os
from pathlib import Path
import sys
import subprocess
from types import MethodType

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT, Replay, configure, read_json, sha, write_json
from tools.simvla.compiled_policy import attach_policy, check_reset
from tools.simvla.condition_output_split_eval import attach, check_policy as check_ours, load_payload
from tools.simvla.condition_output_split_rb2 import environment, compiler_checker as check_ours_compiler, reset_checker as reset_ours
from tools.simvla.condition_solver_rb2 import base_config as solver_config, ready_spec
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.rollout_repair_rb2 import STORAGE, evaluate

RESULTS = STORAGE / 'results/simvla'
OUTPUT = RESULTS / 'condition_output_split/bridge_interval_nfe1_compiled_seed01_v1'
PREDECESSOR = RESULTS / 'condition_output_split/observation_feature_nfe1_compiled_seed01_v1'
OURS_SOURCE = RESULTS / 'condition_output_split/solver_matched_nfe1_compiled_seed01_v1/online/fresh_carry_base_k4'
BRIDGE_SOURCE = RESULTS / 'condition_nfe/compiled_seed01_v1'
OURS_SHA = '25d5336ce4ee4df7eae47e5cd1ea2bd6afd6f77d82262b2b0abb058a99b86938'
ROWS = {f'{method}_k{k}': (method, k) for k in (2, 3, 4) for method in ('bridge', 'ours')}
REFERENCES = {
    'bridge_k2': (BRIDGE_SOURCE, 'bridge_f2_nfe1'),
    'bridge_k3': (BRIDGE_SOURCE, 'bridge_f3_nfe1'),
    'ours_k4': (OURS_SOURCE, 'fresh_carry_base_k4'),
}


def configuration():
    c = solver_config()
    c.update(output=str(OUTPUT), long_rows=list(ROWS),
        campaign_module='tools.simvla.condition_interval_compare',
        extra_source_files=c['extra_source_files'] + ['tools/simvla/condition_interval_compare.py'],
        scope='Fixed fresh10K carry_base checkpoint (K4 464/500) vs fixed Large Latent Bridge; NFE1 for both; K/f2,3,4; paired Long500 seed01; reuse three verified rows, evaluate three missing rows. No training or Generation Loop.',
        new_training=False, student_steps=1, paired_noise_key_flow_steps=10)
    return c


def model_spec():
    spec = ready_spec('fresh', 'carry_base')
    if spec['sha256'] != OURS_SHA:
        raise RuntimeError('The requested 92.8% Ours checkpoint changed')
    return spec


def expected_counts(row, queries):
    method, k = ROWS[row]
    full = (queries + k - 1) // k
    return dict(num_full_vlm_calls=full, num_condition_updater_calls=queries-full,
        num_latent_bridge_calls=queries-full if method == 'bridge' else 0,
        num_action_condition_updater_calls=queries-full if method == 'ours' else 0,
        num_action_transformer_calls=queries, num_action_transformer_decodes=queries,
        num_generation_decoder_only_steps=0)


def replay_factory(c, row, compiler, samples):
    return Replay(c, 'latent_bridge_f2' if ROWS[row][0] == 'bridge' else 'condition_naive3', compiler, samples)


def policy_factory(replay, c, row, manifest):
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_eval import SynchronizedNaiveNFE3Policy
    method, k = ROWS[row]
    if replay.loop is not None:
        raise RuntimeError('Generation Loop loaded')
    base = 'latent_bridge_f2' if method == 'bridge' else 'condition_naive3'
    policy = attach_policy(replay, c, base, manifest)
    policy.NFE = policy.nfe = 1
    if method == 'bridge':
        policy.refresh_every = k
        policy._decode = MethodType(SynchronizedNaiveNFE3Policy._decode, policy)
    else:
        spec = c['model_checkpoint']
        if spec['sha256'] != OURS_SHA or sha(spec['path']) != OURS_SHA:
            raise RuntimeError('Ours checkpoint changed')
        payload = load_payload(spec['path'], 'carry_base', spec['source_identity'], steps=7000, action_mode='naive1')
        policy = attach(policy, replay.native, payload, 'carry_base', k, replay.compiler)
    policy.row_name = row
    if policy.flow_steps != 10:
        raise RuntimeError('Paired noise key changed')
    return policy


def check_policy(policy, row):
    q = int(policy.metrics.counters['num_policy_queries'])
    for key, value in expected_counts(row, q).items():
        if int(policy.metrics.counters.get(key, 0)) != value:
            raise RuntimeError(f'{row}: invocation mismatch {key}')
    if q != (policy.step_index+4)//5 or policy.flow_steps != 10:
        raise RuntimeError('H10/R5 or noise key changed')
    if ROWS[row][0] == 'ours':
        check_ours(policy)


def check_compiler(compiler, row):
    if ROWS[row][0] == 'ours':
        check_ours_compiler(compiler, row)
    else:
        for key in ('vlm', 'action_transformer', 'bridge_predict_next'):
            if not compiler.records.get(key, {}).get('graphs', 0):
                raise RuntimeError('Compile bypass: ' + key)
    if compiler.records.get('generation_updater', {}).get('graphs', 0):
        raise RuntimeError('Generation updater executed')


def reset_checker(policy):
    (reset_ours if hasattr(policy, '_split_context') else check_reset)(policy)


def validate_reference(directory, contract, row):
    result = read_json(directory/'summary.json')
    identity = campaign.digest(dict(campaign=campaign.digest(contract), suite='libero_10', seed='seed01', row=row, smoke=False))
    if (result.get('identity') != identity or result.get('episodes') != 500
            or result.get('verdict') != 'EPISODES_COMPLETE' or result.get('timing_valid_episodes') != 500):
        raise RuntimeError('Reference identity/completion/timing mismatch')
    with (directory/'outcomes.csv').open() as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 500 or {(int(x['task_id']), int(x['trial_id'])) for x in records} != {(t, i) for t in range(10) for i in range(50)}:
        raise RuntimeError('Reference episode IDs mismatch')
    count = sum(int(x['episode_length']) for x in records)
    successes = sum(int(x['success']) for x in records)
    ms = sum(float(x['policy_ms_total']) for x in records)
    if (successes != result['successes'] or successes/500 != result['success_rate']
            or count != result['executed_actions'] or any(x['timing_valid'] != 'True' for x in records)
            or abs(ms/count-result['pooled_policy_ms_per_action']) > 1e-9):
        raise RuntimeError('Reference aggregation mismatch')
    return result


def source_audit(old):
    """Only reviewed dispatcher/validation changes may differ from old rollouts."""
    old_root = next(Path(p).parents[2] for p in old['source_files'] if p.endswith('/tools/simvla/compiled_campaign.py'))
    allowed = {
        'methods/latentloop/modules/condition_output_split.py': ('0e9c9556635242d3faa35f5742b94f7ea283297453cbc6a2dc9121ccb8371fbf', '60481905a7cacb1bccfda7ce1ec762f898b87fcb6ba35c74b0ba9487e63fbf9a', 'interval validation expanded to2/3; existing4/8 arithmetic unchanged'),
        'tools/simvla/compiled_campaign.py': ('c849020a8c09554b3efa1716843e66175b9c56568975ff66a3c080491796cb2c', '09bdc9c8db3b7e95eeca5e9d96a490c1c1c3c0dd48e93a609a7235b99f52fe21', 'GPU lease fd inherited by subprocess; rollout and timing unchanged'),
        'tools/simvla/compiled_profile.py': ('71f12502deb5ebdd68650b0c87690773484b5b48601653a22e15b8380067df8b', '7c171a35628419c7236a368224348ae0b9b9036c5b7f899a00b0c739bb8d1b7b', 'optional profiling callbacks; not used by episode worker'),
        'tools/simvla/gpu_followup_queue.py': ('59a59864cd25d2cc549562537f49a7994f3f770e48a691c5049773ad1f7fd630', '9bc65a9d52015495a55a6044ac0f2c27918ddd7dd96fa1f32012d2647bacb5c2', 'queue ownership/lease/retry changes; not policy inference'),
    }
    historical_dispatchers = {'tools/simvla/condition_nfe_sweep.py', 'architectures/simvla/wrappers/run_condition_nfe_sweep_rb2.sh'}
    audited = []
    for name, expected in old['source_files'].items():
        p = Path(name)
        relative = str(p.relative_to(old_root)) if p.is_relative_to(old_root) else None
        current = ROOT/relative if relative else p
        if relative in historical_dispatchers:
            if sha(p) != expected:
                raise RuntimeError('Historical dispatcher changed: ' + name)
            audited.append(dict(path=name, sha256=expected, reason='retained original NFE1 dispatcher; new dispatch uses identical Euler decode and noise key'))
        elif current.is_file() and sha(current) == expected:
            continue
        elif relative in allowed and current.is_file() and (expected, sha(current)) == allowed[relative][:2]:
            audited.append(dict(path=relative, old_sha256=expected, new_sha256=sha(current), reason=allowed[relative][2]))
        else:
            raise RuntimeError('Unreviewed inference source difference: ' + str(current))
    return audited


def reuse(row, contract, spec):
    root, original_row = REFERENCES[row]
    old = read_json(root/'campaign_contract.json')
    for key in ('artifacts', 'hf_assets', 'libero_config', 'libero_config_sha256', 'gpu', 'options', 'measurement'):
        if old[key] != contract[key]:
            raise RuntimeError('Reference runtime differs: ' + key)
    if old['manifest_hashes']['libero_10/seed01'] != contract['manifest_hashes']['libero_10/seed01']:
        raise RuntimeError('Reference manifest differs')
    if ROWS[row][0] == 'ours' and old['config']['model_checkpoint'] != spec:
        raise RuntimeError('Reference checkpoint specification differs')
    audit = source_audit(old)
    package_change = audit_packages(old, contract)
    directory = root/'rows/libero_10/seed01'/original_row
    result = validate_reference(directory, old, original_row)
    for p in (directory/'episodes').glob('*.json'):
        d = read_json(p); q = int(d['counters']['num_policy_queries'])
        if any(int(d['counters'].get(key, 0)) != value for key, value in expected_counts(row, q).items()):
            raise RuntimeError('Reference invocation count differs')
    report = dict(verdict='EVALUATION_COMPLETE', row=row, episodes=500, reused=True, result=result,
        source=str(directory/'summary.json'), source_sha256=sha(directory/'summary.json'),
        outcomes_sha256=sha(directory/'outcomes.csv'), audited_source_changes=audit,
        audited_package_metadata=package_change)
    write_json(OUTPUT/'completed'/f'{row}.json', report)
    return report


def audit_packages(old, new):
    before, after = set(old['packages']), set(new['packages'])
    if before == after:
        return None
    revision = '8f1084e3132a39270c3a13ebe37270a43ece2a01'
    editable = f'-e git+https://github.com/Lifelong-Robot-Learning/LIBERO.git@{revision}#egg=libero'
    if before-after != {'libero==0.1.0'} or after-before != {editable}:
        raise RuntimeError('Unreviewed package change')
    root = Path(new['config']['libero_root'])
    if str(root) != old['config']['libero_root']:
        raise RuntimeError('Explicit LIBERO import path changed')
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise RuntimeError('LIBERO editable revision changed')
    return dict(old=sorted(before-after), new=sorted(after-before), explicit_import_root=str(root),
        git_revision=actual, initialization_sha256=sha(root/'libero/libero/__init__.py'),
        reason='Both workers prepend the same explicit LIBERO root and set the same LIBERO_CONFIG_PATH; distribution metadata changed from version to editable URL.')


def cell(c, row):
    if row in REFERENCES:
        raise RuntimeError('Verified reference must not be rerun')
    method, k = ROWS[row]
    spec = model_spec() if method == 'ours' else dict(path=c['bridge_checkpoint'], sha256=sha(c['bridge_checkpoint']), method='latent_bridge_large')
    result = evaluate({**c, 'condition_interval': k, 'student_steps': 1}, row, spec, output=OUTPUT)
    write_json(OUTPUT/'completed'/f'{row}.json', dict(verdict='EVALUATION_COMPLETE', row=row,
        episodes=500, reused=False, checkpoint_sha256=spec['sha256'], result=result,
        source=str(OUTPUT/'online'/row/'rows/libero_10/seed01'/row/'summary.json')))


def jobs():
    return [dict(id=row, cmd=[sys.executable, '-u', '-m', 'tools.simvla.condition_interval_compare', 'cell', '--row', row],
        summary=str(OUTPUT/'completed'/f'{row}.json'), completion=dict(verdict='EVALUATION_COMPLETE', episodes=500, row=row))
        for row in ROWS if row not in REFERENCES]


def summarize():
    rows = {row: read_json(OUTPUT/'completed'/f'{row}.json') for row in ROWS if (OUTPUT/'completed'/f'{row}.json').exists()}
    write_json(OUTPUT/'comparison_summary.json', dict(complete=len(rows)==6, rows=rows,
        ours_checkpoint_sha256=OURS_SHA, nfe=1, seed='seed01', episodes_per_row=500, generation_loop=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('all', 'cell', 'smoke', 'worker'), nargs='?', default='all')
    p.add_argument('--preflight', action='store_true'); p.add_argument('--output', type=Path)
    p.add_argument('--row', choices=ROWS); p.add_argument('--suite', default='libero_10', choices=('libero_10',))
    p.add_argument('--seed', default='seed01', choices=('seed01',)); args = p.parse_args()
    if args.command != 'all' and args.row is None:
        p.error('--row required')
    env = environment(0); os.environ.clear(); os.environ.update(env)
    c = read_json(args.output/'runtime_config.json') if args.output else configuration()
    configure(c); sys.path.insert(0, c['libero_root']); os.environ['LIBERO_CONFIG_PATH'] = c['libero_config']
    if args.command in ('smoke', 'worker'):
        campaign.worker(c, args.output, args.suite, args.seed, args.row, smoke=args.command=='smoke',
            replay_factory=replay_factory, policy_factory=policy_factory, policy_checker=check_policy,
            compiler_checker=check_compiler, reset_checker=reset_checker)
        return 0
    if args.command == 'cell':
        cell(c, args.row); summarize(); return 0
    contract = campaign.prepare(c, OUTPUT); write_json(OUTPUT/'runtime_config.json', c)
    spec = model_spec()
    for row in REFERENCES:
        result = reuse(row, contract, spec)
        print(f'REUSE {row}: {result["result"]["successes"]}/500', flush=True)
    plan = jobs(); write_json(OUTPUT/'planned_jobs.json', plan); summarize()
    if args.preflight:
        print('PREFLIGHT_PASS: three reused rows; three new rows; NFE1, Long500; fixed checkpoints', flush=True)
        return 0
    rc = run_queue(OUTPUT, plan, gpus=(0,), predecessor=dict(path=str(PREDECESSOR), lock='queue.lock'),
        environment=environment, cwd=ROOT, timeout=24*3600)
    summarize()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
