"""Test fixed recurrent features during action-head training, then full rollouts."""
import argparse
from pathlib import Path
import subprocess

from tools.simvla.condition_deployment_pipeline import OUTPUT as MIXED, ARM, PARENT_SHA
from tools.simvla.condition_interval_recovery import OUTPUT as PRIOR, FRESH_CONTROL, verify_dataset
from tools.simvla.error_compensation_common import ROOT,environment,identity,read_json,write_json,sha
from tools.simvla.error_compensation_campaign import prepare
from tools.simvla.observation_correction_pipeline import export_arm
from tools.simvla.gpu_followup_queue import run_queue
from tools.simvla.condition_output_split_eval import load_payload,attach,check_policy

OUTPUT=PRIOR.parent/'fixed_condition_nfe1_seed01_v1'
DEST='rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_fixed_condition'
MODES=('parent_fixed','fresh_fixed','fresh_joint')
HYBRIDS=('parent_predictor_later_head','later_predictor_parent_head')
EXTRA=['tools/simvla/fixed_condition_pipeline.py','tools/simvla/fixed_condition_rb2.py',
       'architectures/simvla/wrappers/run_fixed_condition.sh']


def checkpoint_spec(summary_path):
    s=read_json(summary_path)
    if s['verdict']!='TRAIN_AND_OFFLINE_COMPLETE' or sha(s['checkpoint'])!=s['checkpoint_sha256']:
        raise RuntimeError('Incomplete source checkpoint')
    return dict(path=s['checkpoint'],sha256=s['checkpoint_sha256'],source_identity=s['identity'],
        steps=s['steps'],summary=str(summary_path))


def checked_payload(spec):
    if sha(spec['path'])!=spec['sha256']:raise RuntimeError('Source weights changed')
    return load_payload(spec['path'],ARM,spec['source_identity'],steps=spec['steps'],action_mode='naive1')


def combine_models(base,head):
    if set(base)!=set(head):raise ValueError('Mismatched model keys')
    out={}
    for key,value in base.items():
        if value.shape!=head[key].shape or value.dtype!=head[key].dtype:
            raise ValueError('Mismatched model tensor: '+key)
        out[key]=head[key] if key.startswith('action_condition_updater.') else value
    return out


def configurations():
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    previous=read_json(MIXED/'mixed/runtime_config.json')
    parent=checkpoint_spec(previous['initial_models'][ARM]['summary'])
    later=checkpoint_spec(MIXED/'mixed/train'/ARM/'summary.json')
    fresh=checkpoint_spec(FRESH_CONTROL/'train'/ARM/'summary.json')
    if parent['sha256']!=PARENT_SHA:raise RuntimeError('Selected parent changed')
    _,cache_payload=load_native_v0_checkpoint(previous['condition_checkpoint'],device='cpu',require_final_150k=True)
    cfgs={}
    for mode in MODES:
        spec=parent if mode=='parent_fixed' else fresh
        c={**previous,'output':str(OUTPUT/mode),'training_k_c':8,'training_intervals':[4,8],
            'freeze_condition_predictor':mode!='fresh_joint','steps':5000,'warmup_steps':250,
            'sample_step_offset':10000,'initialization':'continuation',
            'continuation_source_steps':spec['steps'],'continuation_source_action_mode':'naive1',
            'initial_models':{ARM:dict(summary=spec['summary'],identity=spec['source_identity'])},
            'evaluation_condition_intervals':[2,3,4],'offline_validation_intervals':[2,3,4],
            'run_label':'fixed_condition_'+mode,'rb2_destination':DEST+'/'+mode,
            'extra_source_files':sorted(set(previous['extra_source_files']+EXTRA)),
            'training_description':'Matched5K continuation, same8-query cache and K4/K8 sampled ages10001..15000. Fixed arm freezes encoder and recurrent updater; only action_condition_updater trains on original NFE10 same-noise first5 action L1 through frozen NFE1 action transformer. Condition MSE remains diagnostic. Joint control retains prior detached-input training. All Long500 seed01, no SR gate.'}
        verify_dataset(c,cache_payload,checked_payload(spec)['contract'])
        prepare(c);write_json(Path(c['output'])/'runtime_config.json',c);cfgs[mode]=c
    for mode in HYBRIDS:
        predictor,head=(parent,later) if mode==HYBRIDS[0] else (later,parent)
        c={**cfgs['parent_fixed'],'output':str(OUTPUT/mode),'evaluation_condition_intervals':[4],
            'hybrid_predictor':predictor,'hybrid_head':head,
            'training_description':'No training: cross two existing checkpoints. Encoder and recurrent updater come from one; action Condition updater from the other. H10/R5,NFE1,K4,Long500. Original paired A/A and B/B results reused.'}
        combine_models(checked_payload(predictor)['model'],checked_payload(head)['model'])
        prepare(c);write_json(Path(c['output'])/'runtime_config.json',c);cfgs[mode]=c
    return cfgs


def hybrid_policy(c,arm,*,smoke=False,k_c=4):
    from tools.simvla.error_compensation_eval import make_policy
    base=checked_payload(c['hybrid_predictor']);head=checked_payload(c['hybrid_head'])
    p={**base,'model':combine_models(base['model'],head['model'])}
    policy=make_policy(c,'condition_naive1',k_c=k_c)
    return attach(policy,policy.native_v0,p,ARM,k_c)


def jobs(configs):
    plan=[]
    for mode in MODES:
        c=configs[mode];out=Path(c['output'])
        def add(kind,module,args,summary,completion,deps):
            key=mode+'_'+kind
            plan.append(dict(id=key,cmd=[c['python'],'-u','-m',module,'--config',str(out/'runtime_config.json'),'--arm',ARM]+args,
                summary=str(summary),completion=dict(identity=identity(c),**completion),deps=deps));return key
        smoke=add('smoke_train','tools.simvla.condition_output_split_train',['--smoke'],out/'smoke'/ARM/'summary.json',dict(verdict='SMOKE_PASS',steps=14),[])
        env=add('smoke_env','tools.simvla.condition_output_split_eval',['--smoke','--k-c','4'],out/'eval_smoke'/f'kc4_{ARM}'/'summary.json',dict(verdict='SMOKE_PASS',episodes=1),[smoke])
        train=add('train','tools.simvla.condition_output_split_train',[],out/'train'/ARM/'summary.json',dict(verdict='TRAIN_AND_OFFLINE_COMPLETE',steps=5000),[env])
        add('export','tools.simvla.fixed_condition_pipeline',['--export'],out/'exports'/f'{ARM}.json',dict(verdict='BUNDLE_EXPORTED'),[train])
    for mode in HYBRIDS:
        c=configs[mode];out=Path(c['output'])
        for smoke in (True,False):
            name=mode+('_smoke' if smoke else '_online')
            plan.append(dict(id=name,cmd=[c['python'],'-u','-m','tools.simvla.fixed_condition_pipeline',
                '--hybrid','--config',str(out/'runtime_config.json')]+(['--smoke'] if smoke else []),
                summary=str(out/('eval_smoke' if smoke else 'online')/'kc4_carry_base/summary.json'),
                completion=dict(identity=identity(c),verdict='SMOKE_PASS' if smoke else 'EVALUATION_COMPLETE',episodes=1 if smoke else 500),
                deps=[] if smoke else [mode+'_smoke']))
    for k in (4,3,2):
        for mode in MODES:
            c=configs[mode];out=Path(c['output'])
            plan.append(dict(id=f'{mode}_k{k}',cmd=[c['python'],'-u','-m','tools.simvla.condition_output_split_eval',
                '--config',str(out/'runtime_config.json'),'--arm',ARM,'--k-c',str(k)],deps=[mode+'_train'],
                summary=str(out/'online'/f'kc{k}_{ARM}'/'summary.json'),
                completion=dict(identity=identity(c),verdict='EVALUATION_COMPLETE',episodes=500)))
    return plan


def transfer_status(path):
    host,remote=DEST.split(':',1)
    for attempt in range(2):
        try:
            subprocess.run(['ssh',host,'mkdir','-p',remote],check=True,timeout=30)
            subprocess.run(['rsync','-a',str(path),host+':'+remote+'/'],check=True,timeout=60);return
        except (OSError,subprocess.SubprocessError) as exc:print('TRANSFER_STATUS_WARNING',attempt,str(exc),flush=True)


def summarize(configs):
    rows={}
    for mode,c in configs.items():
        for p in Path(c['output']).glob('online/*/summary.json'):rows[mode+'/'+p.parent.name]=read_json(p)
    write_json(OUTPUT/'comparison_summary.json',dict(rows=rows,complete=len(rows)==11,
        reused_parent_joint_control=read_json(MIXED/'comparison_summary.json'),
        recent_interval_results=read_json(PRIOR/'comparison_summary.json'),hardware='sd1 RTX3090 eager'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--export',action='store_true')
    p.add_argument('--hybrid',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--config');p.add_argument('--arm',choices=(ARM,))
    a=p.parse_args()
    if a.export:export_arm(read_json(a.config),ARM);return 0
    if a.hybrid:
        from tools.simvla.error_compensation_eval import run
        def checker(policy,row,calls,k):
            result=check_policy(policy)
            if calls.get('condition',0)!=result['condition_updater'] or calls.get('transformer',0)!=result['transformer']:
                raise RuntimeError('Invocation counters disagree')
            return result
        run(read_json(a.config),ARM,k_c=4,smoke=a.smoke,policy_factory=hybrid_policy,counter_checker=checker);return 0
    configs=configurations();plan=jobs(configs);write_json(OUTPUT/'planned_jobs.json',plan);summarize(configs)
    if a.preflight:print('PREFLIGHT_PASS fixed predictor training and checkpoint cross-combinations',flush=True);return 0
    status=OUTPUT/'pipeline_status.json';write_json(status,dict(phase='running'));transfer_status(status)
    rc=1
    try:
        rc=run_queue(OUTPUT,plan,gpus=(4,5,6,7),predecessor=dict(path=str(PRIOR),lock='queue.lock'),
            environment=lambda gpu:environment(configs['parent_fixed'],gpu),cwd=ROOT,timeout=12*3600)
    finally:
        summarize(configs);write_json(status,dict(phase='complete' if not rc else 'finished_with_failures'));transfer_status(status)
    return rc


if __name__=='__main__':raise SystemExit(main())
