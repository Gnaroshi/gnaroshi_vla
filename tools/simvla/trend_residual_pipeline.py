"""Frozen-trend residual training and matched K4/K8 compiled LIBERO evaluation."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import MethodType

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import DEFAULT_CONFIG, ROOT, configure, preflight, read_json, sha, stop_worker, write_json
from tools.simvla.error_compensation_common import snapshots
from tools.simvla.trend_compiled_rb2 import replay_factory, reset_policy
from tools.simvla.compiled_policy import attach_policy

CONFIG = ROOT/'architectures/simvla/configs/trend_residual_k8_rb2.json'
ROWS = {'residual_k4': (4, 'frozen_trend_residual'),
        'trend_trained_k8': (8, 'trend_only'), 'residual_k8': (8, 'frozen_trend_residual')}


def expected_counts(row, queries):
    k, arm = ROWS[row]
    full = (queries+k-1)//k
    residual = queries-full if arm == 'frozen_trend_residual' else 0
    return dict(num_full_vlm_calls=full, num_condition_updater_calls=queries-full,
        num_action_transformer_calls=3*queries, num_generation_decoder_only_steps=7*queries,
        num_trend_head_calls=full, num_residual_calls=residual, num_observation_encoder_calls=residual)


def check_policy(policy, row):
    q = int(policy.metrics.counters['num_policy_queries'])
    for name, expected in expected_counts(row, q).items():
        if int(policy.metrics.counters.get(name, 0)) != expected:
            raise RuntimeError(f'{row}: incorrect {name}, expected {expected}')
    if q != (policy.step_index+4)//5:
        raise RuntimeError('H10/R5 query cadence changed')


def check_compiler(compiler, row):
    required = ['vlm','action_transformer','action_decoder','generation_updater','trend_head']
    if ROWS[row][1] == 'frozen_trend_residual':
        required += ['observation_encoder','condition_updater']
    missing = [n for n in required if compiler.records.get(n, {}).get('graphs', 0) == 0]
    if missing:
        raise RuntimeError('Compile bypass: '+str(missing))


def make_policy(replay, c, row, manifest):
    import torch
    from methods.latentloop.modules.trend_condition import TrendCondition
    policy = attach_policy(replay, c, 'ours_kc2_ng3', manifest)
    spec = c['model_checkpoint']
    if sha(spec['path']) != spec['sha256']:
        raise RuntimeError('Evaluation checkpoint bytes changed')
    saved = torch.load(spec['path'], map_location='cpu', weights_only=False)
    k, arm = ROWS[row]
    if saved['arm'] != arm or saved['step'] != c['steps'] or saved['contract']['k_c'] != k:
        raise RuntimeError('Checkpoint arm, step or training horizon mismatch')
    model = TrendCondition(replay.native, arm, max_age=k-1).to('cuda').eval()
    model.load_state_dict(saved['model'], strict=True)
    model.requires_grad_(False)
    # replay.native already wraps the shared observation and residual modules.
    model.trend_head.forward = replay.compiler.wrap('trend_head', model.trend_head.forward)
    full_original, reset_original = policy._full_refresh, policy.reset
    policy.native_v0 = model
    policy.row_name = policy.mode = row
    policy.k_c = policy.refresh_every = k

    def reset(self):
        reset_original()
        self._trend_context = None

    def full(self, batch, *, policy_query_index):
        condition, action, seed = full_original(batch, policy_query_index=policy_query_index)
        self._trend_context = model.prepare(condition, batch['raw_rgb'], batch['proprio'],
            self.condition_layout.valid_mask, self.condition_layout.group_ids)
        self.metrics.counters['num_trend_head_calls'] += 1
        return condition, action, seed

    def update(self, batch, *, age, policy_query_index):
        condition, _ = model.predict(self._trend_context, age, batch['raw_rgb'], batch['proprio'])
        self.metrics.counters['num_condition_updater_calls'] += 1
        if arm == 'frozen_trend_residual':
            self.metrics.counters['num_residual_calls'] += 1
            self.metrics.counters['num_observation_encoder_calls'] += 1
        action, seed = self._decode(condition, batch['proprio'], policy_query_index=policy_query_index)
        self.cached_condition, self.cached_action_chunk = condition.detach(), action.detach()
        return condition, action, seed

    policy.reset = MethodType(reset, policy)
    policy._full_refresh = MethodType(full, policy)
    policy._v0_update = MethodType(update, policy)
    policy.reset()
    return policy


def configs():
    return {**read_json(DEFAULT_CONFIG), **read_json(ROOT/'architectures/simvla/configs/compile_campaign_rb2.json'),
            **read_json(CONFIG)}


def training_config(c, row):
    result = read_json(ROOT/'architectures/simvla/configs/trend_condition_sd1.json')
    for key in ('storage','python','upstream','hf_home','checkpoint_revision','condition_checkpoint',
                'generation_checkpoint','norm_stats','cache','steps','decomposition_steps','smoke_steps','heldout_windows'):
        result[key] = c[key]
    result.update(output=str(Path(c['output'])/'training'/row), training_k_c=ROWS[row][0],
        evaluation_condition_intervals=[ROWS[row][0]], wait_for_processes=[], reference_results={},
        training_condition_ages=list(range(1, ROWS[row][0])),
        training_description='750 target-fitting steps then 2250 first-five action L1 steps; separate cosine schedules; K-specific teacher windows; original and Generation frozen')
    if ROWS[row][1] == 'frozen_trend_residual':
        source = c['frozen_k4_trend']['path'] if row == 'residual_k4' else str(
            Path(c['output'])/'training/trend_trained_k8/train/trend_only/latest.pt')
        result['frozen_trend_checkpoint'] = dict(path=source, sha256=sha(source))
    return result


def prepare_training(c, row):
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import make_datasets
    t = training_config(c, row)
    _, payload = load_native_v0_checkpoint(t['condition_checkpoint'], device='cpu', require_final_150k=True)
    train, heldout = make_datasets(t, payload)
    keys = lambda ds: {(t, e) for t, e, _ in ds.identities}
    if keys(train) & keys(heldout):
        raise RuntimeError('Train/heldout episode overlap')
    for ds in (train,heldout):
        sample = ds[0]
        if sample['teacher_conditions'].shape[0] != ROWS[row][0]-1:
            raise RuntimeError('Teacher horizon mismatch')
    files = set()
    for directory in ('tools/simvla','methods/latentloop','architectures/simvla/adapters'):
        files.update((ROOT/directory).rglob('*.py'))
    contract = dict(config=t, source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},
        train=train.contract(), heldout=heldout.contract(), snapshots=snapshots(t),
        git_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        artifacts={key:sha(t[key]) for key in ('condition_checkpoint','generation_checkpoint','norm_stats')})
    contract['identity'] = campaign.digest(contract)
    output = Path(t['output'])
    path = output/'contract.json'
    if path.exists() and read_json(path) != contract and any((output/p).exists() for p in ('train','smoke')):
        raise RuntimeError('Training contract changed; preserve previous output')
    write_json(path, contract)
    write_json(output/'config.json',t)
    return t,contract


def wait_gpu(output):
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        write_json(output/'pipeline_status.json',dict(phase='waiting_for_gpu'))
        print('WAIT: GPU occupied',flush=True)
        time.sleep(30)


def run_training(c, row, smoke=False):
    t, contract = prepare_training(c,row)
    out = Path(t['output'])
    arm = ROWS[row][1]
    phase = 'smoke' if smoke else 'train'
    summary = out/phase/arm/'summary.json'
    expected = dict(identity=contract['identity'], steps=t['smoke_steps'] if smoke else t['steps'],
        verdict='SMOKE_PASS' if smoke else 'TRAIN_AND_OFFLINE_COMPLETE')
    def complete():
        return summary.exists() and all(read_json(summary).get(k)==v for k,v in expected.items())
    if complete():
        return read_json(summary)
    command = [t['python'],'-u','-m','architectures.simvla.adapters.latentloop.efficient_multirate.trend_condition_train',
        '--config',str(out/'config.json'),'--arm',arm] + (['--smoke'] if smoke else [])
    for attempt in (1,2):
        wait_gpu(Path(c['output']))
        log = out/'logs'/f'{phase}_attempt{attempt}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        print(f'START {row} {phase}: {log}',flush=True)
        began = time.monotonic()
        with log.open('a') as stream:
            process = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while process.poll() is None:
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        status = dict(phase=phase,row=row,pid=process.pid,elapsed_seconds=time.monotonic()-began)
                        metric_file = out/phase/arm/'metrics.jsonl'
                        if metric_file.exists():
                            import json
                            lines = metric_file.read_text().splitlines()
                            if lines: status['metrics'] = json.loads(lines[-1])
                        write_json(Path(c['output'])/'pipeline_status.json',status)
                        print(status,flush=True)
                        if time.monotonic()-began>7200: stop_worker(process)
            except BaseException:
                stop_worker(process)
                raise
        write_json(log.with_suffix('.status.json'),dict(returncode=process.returncode,wall_seconds=time.monotonic()-began))
        if complete():
            return read_json(summary)
        print(f'TRAINING_TECHNICAL_FAILURE {row} attempt={attempt} log={log}',flush=True)
    raise RuntimeError(f'{row} {phase} failed after bounded technical retry')


def evaluate(c, row, trained):
    output = Path(c['output'])/'online'/row
    e = {**c, 'output':str(output),'long_rows':[row],
        'model_checkpoint':dict(path=trained['checkpoint'],sha256=sha(trained['checkpoint']))}
    config = output/'runtime_config.json'
    if config.exists() and read_json(config) != e: raise RuntimeError('Evaluation config changed')
    write_json(config,e)
    campaign.prepare(e,output)
    contract = read_json(output/'campaign_contract.json')
    for command in ('smoke','worker'):
        directory = output/('smoke' if command=='smoke' else 'rows')/'libero_10/seed01'/row
        m = read_json(output/'manifests/libero_10/seed01/episode_manifest.json')
        specs = sorted(m['episodes'],key=lambda x:(-x['task_id'],x['trial_id']))
        if command=='smoke': specs=specs[:e['smoke_episodes']]
        key=campaign.digest(dict(campaign=campaign.digest(contract),suite='libero_10',seed='seed01',row=row,smoke=command=='smoke'))
        def recover():
            return campaign.summarize_cell(directory,key,[(x['task_id'],x['trial_id']) for x in specs])
        if recover(): continue
        ok=campaign.run_child(e,output,command,'libero_10','seed01',row)
        if not ok and not recover():
            # Preserve RNG histories; a failed partial row gets one clean retry.
            archive=output/'failed_attempts'/command
            if not archive.exists():
                archive.parent.mkdir(parents=True,exist_ok=True)
                if directory.exists(): directory.rename(archive)
                else: archive.mkdir()
                campaign.run_child(e,output,command,'libero_10','seed01',row)
        if not recover(): raise RuntimeError(f'{row} {command} incomplete; inspect {output}/logs')
    return read_json(directory/'summary.json')


def preflight_all(c):
    if socket.gethostname() != 'jbr-TRX50': raise RuntimeError('rb2 only')
    preflight(c)
    if sha(c['frozen_k4_trend']['path']) != c['frozen_k4_trend']['sha256']:
        raise RuntimeError('K4 reference trend changed')
    # Both window lengths are checked before starting any GPU worker.
    for row in ('residual_k4','trend_trained_k8'):
        _, contract=prepare_training(c,row)
        print(f'PREFLIGHT {row}: train={contract["train"]["windows"]} heldout={contract["heldout"]["windows"]}',flush=True)


def run_all(c):
    output=Path(c['output'])
    output.mkdir(parents=True,exist_ok=True)
    with (output/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preflight_all(c)
        results=[]
        failures=[]
        prior=read_json(Path(c['reference_root'])/'combined_summary.json')
        references=[r for r in prior['results'] if r['suite']=='libero_10' and r['seed']=='seed01'
            and r['row'] in ('baseline','naive_nfe3','condition_naive3','generation_ng3',
                            'ours_kc2_ng3','latent_bridge_f2','latent_bridge_f3','latent_bridge_f4')]
        k4=read_json(Path(c['trend_reference_root'])/'rows/libero_10/seed01/trend_k4/summary.json')
        references.append(dict(row='trend_only_k4',k_c=4,n_g=3,**k4))
        for row in ROWS:
            try:
                run_training(c,row,smoke=True)
                trained=run_training(c,row)
                result=evaluate(c,row,trained)
                results.append(dict(row=row,k_c=ROWS[row][0],n_g=3,training=trained,**result))
            except Exception as error:
                traceback.print_exc()
                failures.append(dict(row=row,error=f'{type(error).__name__}: {error}'))
            write_json(output/'combined_summary.json',dict(complete=len(results)==len(ROWS),
                results=results,failures=failures,references=references,reference_root=c['reference_root'],
                trend_reference_root=c['trend_reference_root'],evaluation_seed='seed01',
                scientific_stopping_gate=None))
        write_json(output/'pipeline_status.json',dict(phase='complete' if not failures else 'finished_with_failures',
            completed_rows=len(results),planned_rows=len(ROWS),failures=failures))
        return 0 if not failures else 2


def main():
    def interrupt(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('all','preflight','smoke','worker'))
    p.add_argument('--row',choices=ROWS,default='residual_k4')
    p.add_argument('--suite',default='libero_10',choices=('libero_10',))
    p.add_argument('--seed',default='seed01',choices=('seed01',))
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    c=read_json(args.output/'runtime_config.json') if args.output else configs()
    configure(c)
    sys.path.insert(0,c['libero_root'])
    os.environ['LIBERO_CONFIG_PATH']=c['libero_config']
    if args.command=='preflight': preflight_all(c); return 0
    if args.command=='all':
        try: return run_all(c)
        except BaseException as error:
            write_json(Path(c['output'])/'pipeline_status.json',dict(phase='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error)))
            raise
    campaign.worker(c,args.output,args.suite,args.seed,args.row,smoke=args.command=='smoke',
        replay_factory=replay_factory,policy_factory=make_policy,policy_checker=check_policy,
        compiler_checker=check_compiler,reset_checker=reset_policy)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
