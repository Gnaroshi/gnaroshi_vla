"""Recover interval training with fixed cache windows; test fresh starts."""
import argparse
from pathlib import Path
import subprocess

from tools.simvla.condition_deployment_pipeline import OUTPUT as PRIOR, ARM, PARENT_SHA
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.error_compensation_common import ROOT, environment, identity, read_json, sha, write_json
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.observation_correction_pipeline import export_arm

OUTPUT = PRIOR.parent/'interval_recovery_nfe1_seed01_v1'
DEST = 'rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_interval_recovery'
FRESH_CONTROL = PRIOR.parent/'fresh_noise_nfe1_seed01_v1/detached_noise1'
MODES = {f'{init}_k{k}': (init, k) for init in ('continue', 'fresh') for k in (2,3,4)}
EXTRA = ['tools/simvla/condition_interval_recovery.py', 'tools/simvla/condition_interval_recovery_rb2.py',
         'architectures/simvla/wrappers/run_condition_interval_recovery.sh']


def make_config(previous, mode):
    init, k = MODES[mode]
    c = {**previous, 'output': str(OUTPUT/mode), 'training_k_c': 8,
        'training_intervals': [k], 'training_condition_ages': list(range(1,k)),
        'evaluation_condition_intervals': [k], 'offline_validation_intervals': [2,3,4],
        'run_label': 'condition_interval_'+mode, 'rb2_destination': DEST+'/'+mode,
        'extra_source_files': sorted(set(previous['extra_source_files']+EXTRA)),
        'training_description': 'Cache window stays8 with original exact train/heldout membership. Training interval alone selects unroll prefix. Continue selected10K parent for5K or fresh NFE1-only10K. Same original10 teacher, detached action gradient,736130 parameters. Fresh mixed10K control reused. No SR gate.'}
    if init == 'fresh':
        c.update(initialization='fresh',steps=10000,warmup_steps=500,sample_step_offset=0)
        for key in ('interval_transition','initial_models','continuation_source_steps','continuation_source_action_mode'):
            c.pop(key,None)
    return c


def verify_dataset(c, payload, expected):
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import make_datasets
    if c['training_k_c'] != 8:
        raise RuntimeError('Cache window must remain8 for all comparisons')
    train, heldout = make_datasets(c, payload)
    observed = dict(data=train.contract(),heldout=heldout.contract())
    if any(observed[k] != expected[k] for k in observed):
        raise RuntimeError('Actual cache dataset differs from common parent')
    return observed


def configurations():
    import torch
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train import state_hash
    from tools.simvla.condition_output_split_train import build_initial_model
    previous = read_json(PRIOR/'k2/runtime_config.json')
    summary = read_json(previous['initial_models'][ARM]['summary'])
    if sha(summary['checkpoint']) != PARENT_SHA:
        raise RuntimeError('Common parent changed')
    parent_contract = torch.load(summary['checkpoint'],map_location='cpu',weights_only=False)['contract']
    parent, payload = load_native_v0_checkpoint(previous['condition_checkpoint'],device='cpu',require_final_150k=True)
    control_summary = read_json(FRESH_CONTROL/'train'/ARM/'summary.json')
    if sha(control_summary['checkpoint']) != control_summary['checkpoint_sha256']:
        raise RuntimeError('Fresh mixed control changed')
    control_payload = torch.load(control_summary['checkpoint'],map_location='cpu',weights_only=False)
    contract = control_payload['contract']
    if (contract['initialization']!='fresh' or contract['steps']!=10000
            or contract['action_mode']!='naive1' or contract['training_intervals']!=[4,8]
            or contract['action_noise_samples']!=1 or contract['action_gradient_mode']!='detached'):
        raise RuntimeError('Wrong fresh mixed control')
    observed=verify_dataset({**previous,'training_k_c':8},payload,parent_contract)
    if observed['data']!=contract['data'] or observed['heldout']!=contract['heldout']:
        raise RuntimeError('Fresh control dataset changed')
    configs={}
    for mode in MODES:
        source=read_json(PRIOR/f'k{MODES[mode][1]}'/'runtime_config.json')
        c=make_config(source,mode)
        if c['cache']!=previous['cache'] or c['condition_checkpoint']!=previous['condition_checkpoint'] or c['training_k_c']!=8:
            raise RuntimeError('Dataset-constructor inputs differ between arms')
        if MODES[mode][0]=='fresh':
            model=build_initial_model(parent,c,ARM)
            if state_hash(model)!=contract['initial_weights_sha256']:
                raise RuntimeError('Fresh initialization differs from reused control')
        prepare(c);write_json(Path(c['output'])/'runtime_config.json',c)
        write_json(Path(c['output'])/'dataset_preflight.json',dict(verdict='ACTUAL_DATASET_PASS',**observed))
        configs[mode]=c
    control={**configs['fresh_k4'],'output':str(OUTPUT/'fresh_mixed_control'),
        'training_intervals':[4,8],'evaluation_condition_intervals':[2,3,4],
        'model_checkpoint':dict(path=control_summary['checkpoint'],sha256=control_summary['checkpoint_sha256'],
            source_identity=control_summary['identity'],step=10000),
        'training_description':'No new training: reuse completed detached_noise1 fresh NFE1-only10K mixedK4/K8 control.'}
    prepare(control);write_json(Path(control['output'])/'runtime_config.json',control)
    configs['fresh_mixed_control']=control
    return configs


def jobs(configs):
    plan=[]
    for mode,(init,k) in MODES.items():
        c=configs[mode];out=Path(c['output'])
        def add(kind,module,extra,summary,completion,deps):
            key=mode+'_'+kind
            plan.append(dict(id=key,cmd=[c['python'],'-u','-m',module,'--config',str(out/'runtime_config.json'),'--arm',ARM]+extra,
                summary=str(summary),completion=dict(identity=identity(c),**completion),deps=deps))
            return key
        smoke=add('smoke_train','tools.simvla.condition_output_split_train',['--smoke'],out/'smoke'/ARM/'summary.json',dict(verdict='SMOKE_PASS',steps=14),[])
        env=add('smoke_env','tools.simvla.condition_output_split_eval',['--smoke','--k-c',str(k)],out/'eval_smoke'/f'kc{k}_{ARM}'/'summary.json',dict(verdict='SMOKE_PASS',episodes=1),[smoke])
        trained=add('train','tools.simvla.condition_output_split_train',[],out/'train'/ARM/'summary.json',dict(verdict='TRAIN_AND_OFFLINE_COMPLETE',steps=c['steps']),[env])
        add('export','tools.simvla.condition_interval_recovery',['--export'],out/'exports'/f'{ARM}.json',dict(verdict='BUNDLE_EXPORTED'),[trained])
    for mode,(_,k) in MODES.items():
        c=configs[mode];out=Path(c['output'])
        plan.append(dict(id=mode+'_online',cmd=[c['python'],'-u','-m','tools.simvla.condition_output_split_eval',
            '--config',str(out/'runtime_config.json'),'--arm',ARM,'--k-c',str(k)],deps=[mode+'_train'],
            summary=str(out/'online'/f'kc{k}_{ARM}'/'summary.json'),completion=dict(identity=identity(c),verdict='EVALUATION_COMPLETE',episodes=500)))
    c=configs['fresh_mixed_control'];out=Path(c['output'])
    for k in (2,3):
        plan.append(dict(id=f'fresh_mixed_k{k}',cmd=[c['python'],'-u','-m','tools.simvla.condition_interval_recovery',
            '--eval-control','--config',str(out/'runtime_config.json'),'--k-c',str(k)],
            summary=str(out/'online'/f'kc{k}_{ARM}'/'summary.json'),completion=dict(identity=identity(c),verdict='EVALUATION_COMPLETE',episodes=500)))
    return plan


def control_policy(c,arm,*,smoke=False,k_c=2):
    from tools.simvla.error_compensation_eval import make_policy
    from tools.simvla.condition_output_split_eval import load_payload,attach
    spec=c['model_checkpoint']
    if sha(spec['path'])!=spec['sha256']:raise RuntimeError('Control weights changed')
    p=load_payload(spec['path'],ARM,spec['source_identity'],steps=10000,action_mode='naive1')
    policy=make_policy(c,'condition_naive1',k_c=k_c)
    return attach(policy,policy.native_v0,p,ARM,k_c)


def transfer_status(path):
    host,remote=DEST.split(':',1)
    for attempt in range(2):
        try:
            subprocess.run(['ssh',host,'mkdir','-p',remote],check=True,timeout=30)
            subprocess.run(['rsync','-a',str(path),host+':'+remote+'/'],check=True,timeout=60)
            return
        except (OSError,subprocess.SubprocessError) as exc:
            print(f'STATUS_TRANSFER_WARNING {attempt+1}: {exc}',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--export',action='store_true')
    p.add_argument('--eval-control',action='store_true');p.add_argument('--config');p.add_argument('--arm',choices=(ARM,))
    p.add_argument('--k-c',type=int,choices=(2,3));a=p.parse_args()
    if a.export:export_arm(read_json(a.config),a.arm);return 0
    if a.eval_control:
        from tools.simvla.error_compensation_eval import run
        from tools.simvla.condition_output_split_eval import check_policy
        def check(policy,row,calls,k):
            result=check_policy(policy)
            if calls.get('condition',0)!=result['condition_updater'] or calls.get('transformer',0)!=result['transformer']:
                raise RuntimeError('Independent counters disagree')
            return result
        run(read_json(a.config),ARM,k_c=a.k_c,policy_factory=control_policy,counter_checker=check);return 0
    configs=configurations();plan=jobs(configs);write_json(OUTPUT/'planned_jobs.json',plan)
    if a.preflight:print('PREFLIGHT_PASS: actual8-query datasets and fresh control weights matched',flush=True);return 0
    status=OUTPUT/'pipeline_status.json';write_json(status,dict(phase='running'));transfer_status(status)
    rc=1
    try:
        rc=run_queue(OUTPUT,plan,gpus=(4,5,6,7),predecessor=dict(path=str(PRIOR),lock='queue.lock'),
            environment=lambda gpu:environment(configs['continue_k2'],gpu),cwd=ROOT,timeout=12*3600)
    finally:
        rows={}
        for mode,c in configs.items():
            for path in Path(c['output']).glob('online/*/summary.json'):rows[mode+'/'+path.parent.name]=read_json(path)
        rows['reused_fresh_mixed_k4']=read_json(FRESH_CONTROL/'online/kc4_carry_base/summary.json')
        write_json(OUTPUT/'comparison_summary.json',dict(rows=rows,hardware='sd1 RTX3090 eager'))
        write_json(status,dict(phase='complete' if not rc else 'finished_with_failures'));transfer_status(status)
    return rc


if __name__=='__main__':raise SystemExit(main())
