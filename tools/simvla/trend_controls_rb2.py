"""Missing K4 controls using the frozen trend and compiled paper protocol."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import traceback
from types import MethodType

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import DEFAULT_CONFIG, ROOT, Replay, configure, read_json, write_json
from tools.simvla.compiled_policy import attach_policy
from tools.simvla.trend_compiled_rb2 import prepare, recover, reset_policy, trend_payload

ROWS = {'trend_k4_naive3': 'condition_naive3', 'hold_k4_generation3': 'ours_kc2_ng3'}


def expected_counts(row, queries):
    hold = row == 'hold_k4_generation3'
    full = (queries + 3) // 4
    return dict(num_full_vlm_calls=full, num_condition_updater_calls=queries-full,
        num_action_transformer_calls=3*queries, num_generation_decoder_only_steps=7*queries if hold else 0,
        num_trend_head_calls=0 if hold else full, num_observation_encoder_calls=0)


def check_policy(policy, row):
    q = int(policy.metrics.counters['num_policy_queries'])
    for key, expected in expected_counts(row, q).items():
        if int(policy.metrics.counters.get(key, 0)) != expected:
            raise RuntimeError(f'{row}: {key} != {expected}')
    if q != (policy.step_index+4)//5:
        raise RuntimeError('H10/R5 queue contract changed')


def check_compiler(compiler, row):
    required = {'vlm', 'action_transformer'}
    # Naive decoder is inside the compiled ActionStep, not a decoder-only call.
    required |= ({'action_decoder', 'generation_updater'} if row == 'hold_k4_generation3' else {'trend_head'})
    missing = [n for n in required if not compiler.records.get(n, {}).get('graphs', 0)]
    if missing:
        raise RuntimeError('Compile bypass: '+str(missing))


def replay_factory(c, row, compiler, samples):
    return Replay(c, ROWS[row], compiler, samples)


def make_policy(replay, c, row, manifest):
    from methods.latentloop.modules.trend_condition import TrendCondition
    policy = attach_policy(replay, c, ROWS[row], manifest)
    model = TrendCondition(replay.native, 'trend_only').to('cuda').eval().requires_grad_(False)
    model.load_state_dict(trend_payload(c)['model'], strict=True)
    hold = row == 'hold_k4_generation3'
    if not hold:
        model.trend_head.forward = replay.compiler.wrap('trend_head', model.trend_head.forward)
    original_full, original_reset = policy._full_refresh, policy.reset
    policy.native_v0 = model
    policy.row_name = policy.mode = row
    policy.k_c = policy.refresh_every = 4

    def reset(self):
        original_reset()
        self._trend_context = None

    def full(self, batch, *, policy_query_index):
        condition, action, seed = original_full(batch, policy_query_index=policy_query_index)
        if hold:
            self._trend_context = condition.detach()
        else:
            self._trend_context = model.prepare(condition, batch['raw_rgb'], batch['proprio'],
                self.condition_layout.valid_mask, self.condition_layout.group_ids)
            self.metrics.counters['num_trend_head_calls'] += 1
        return condition, action, seed

    def update(self, batch, *, age, policy_query_index):
        if self._trend_context is None or not 1 <= age < 4:
            raise RuntimeError('Missing K4 anchor or invalid age')
        ctx = self._trend_context
        condition = ctx if hold else ctx.anchor + age*ctx.trend
        self.metrics.counters['num_condition_updater_calls'] += 1
        action, seed = self._decode(condition, batch['proprio'], policy_query_index=policy_query_index)
        self.cached_condition, self.cached_action_chunk = condition.detach(), action.detach()
        return condition, action, seed

    policy.reset, policy._full_refresh, policy._v0_update = MethodType(reset, policy), MethodType(full, policy), MethodType(update, policy)
    policy.reset()
    return policy


def summarize(c, output):
    rows = []
    for row in ROWS:
        p = output/'rows/libero_10/seed01'/row/'summary.json'
        if p.exists(): rows.append(dict(row=row, **read_json(p)))
    report = dict(complete=len(rows)==len(ROWS), results=rows, seed='seed01', episodes_per_row=500,
        k_c=4, full_action_transformer_calls_per_query=3, training='same frozen 3K K4 trend; no retraining',
        reference_root=c['reference_root'], trend_reference_root=c['trend_reference_root'],
        timing='RTX5090 compiled synchronized policy.act / executed actions; compare only same-device compiled rows')
    write_json(output/'comparison_summary.json', report)
    return report


def run_all(c, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output/'launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(c, output)
        failures=[]
        for row in ROWS:
            for command in ('smoke', 'worker'):
                if recover(c, output, command, row): continue
                ok=campaign.run_child(c, output, command, 'libero_10', 'seed01', row)
                ok=ok or bool(recover(c, output, command, row))
                if not ok:
                    directory=output/('smoke' if command=='smoke' else 'rows')/'libero_10/seed01'/row
                    archive=output/'failed_attempts'/f'{command}_{row}'
                    if not archive.exists():
                        archive.parent.mkdir(parents=True,exist_ok=True)
                        if directory.exists(): directory.rename(archive)
                        else: archive.mkdir()
                        ok=campaign.run_child(c,output,command,'libero_10','seed01',row)
                        ok=ok or bool(recover(c,output,command,row))
                if not ok:
                    failures.append(dict(row=row,phase=command))
                    break
                summarize(c,output)
        result=summarize(c,output)
        write_json(output/'status.json',dict(phase='finished',failures=failures,complete=result['complete']))
        return 0 if result['complete'] and not failures else 2


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('preflight','all','smoke','worker','summarize'))
    p.add_argument('--suite',default='libero_10',choices=('libero_10',))
    p.add_argument('--seed',default='seed01',choices=('seed01',))
    p.add_argument('--row',choices=ROWS,default='trend_k4_naive3')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    c={**read_json(DEFAULT_CONFIG),**read_json(ROOT/'architectures/simvla/configs/compile_campaign_rb2.json'),
       **read_json(ROOT/'architectures/simvla/configs/trend_compiled_rb2.json'),
       **read_json(ROOT/'architectures/simvla/configs/trend_controls_rb2.json')}
    if a.output: c['output']=str(a.output.resolve())
    configure(c)
    sys.path.insert(0,c['libero_root'])
    os.environ['LIBERO_CONFIG_PATH']=c['libero_config']
    output=Path(c['output'])
    try:
        if a.command=='all': return run_all(c,output)
        if a.command=='preflight': prepare(c,output)
        elif a.command=='summarize': summarize(c,output)
        else:
            campaign.worker(c,output,a.suite,a.seed,a.row,smoke=a.command=='smoke',
                replay_factory=replay_factory,policy_factory=make_policy,policy_checker=check_policy,
                compiler_checker=check_compiler,reset_checker=reset_policy)
        return 0
    except BaseException as exc:
        if a.command=='all': write_json(output/'status.json',dict(phase='failed',error=str(exc)))
        traceback.print_exc()
        return 130 if isinstance(exc,KeyboardInterrupt) else 1


if __name__=='__main__':
    raise SystemExit(main())
