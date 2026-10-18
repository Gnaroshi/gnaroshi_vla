"""Three matched trainings, complete Long500 evaluations, and rb2 confirmation."""
import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace

from methods.latentloop.modules.current_condition import ARMS, install
from tools.simvla import current_condition_train as training
from tools.simvla import short_visual_train as base
from tools.simvla import short_visual_eval as evaluation
from tools.simvla import short_visual_campaign as campaign
from tools.simvla import geometry_comparison_rb2 as portable
from tools.simvla import failure_interventions as prior
from tools.simvla import gpu_followup_queue as queue
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

MODULE='tools.simvla.current_condition_campaign'
BASE_ATTACH,BASE_CHECK=evaluation.attach_visual,evaluation.check
BUNDLE=prior.STORAGE/'transfers/simvla_current_condition'


def attach(policy,model,interval,compiler):
    policy=BASE_ATTACH(policy,model,interval,compiler)
    if compiler:
        from tools.simvla.observation_correction_eval import tensor_age_call
        compiled=compiler.wrap('current_condition_updater',model.current_head['updater'].forward)
        model.current_head['updater'].forward=lambda *a,**kw:tensor_age_call(compiled,*a,**kw)
        model.current_head['reader'].forward=compiler.wrap('current_condition_reader',model.current_head['reader'].forward)
    reset,update=policy.reset,policy._v0_update
    def reset_count(self):
        reset();self._current_head_calls=0
    def update_count(self,batch,*,age,policy_query_index):
        result=update(batch,age=age,policy_query_index=policy_query_index)
        self._current_head_calls+=int(age==4)
        return result
    policy.reset=MethodType(reset_count,policy);policy._v0_update=MethodType(update_count,policy)
    policy.reset();return policy


def check(policy):
    counts=BASE_CHECK(policy)
    expected=sum(t%policy.k_c==4 for t in range(counts['queries']))
    if policy._current_head_calls!=expected:
        raise RuntimeError('Current-head invocation mismatch')
    return dict(**counts,current_head_calls=expected,additional_delta_encoder_calls=expected)


@contextmanager
def scope():
    replace=dict(OUTPUT=training.OUTPUT,ARMS=ARMS,checked=training.checked,
        initial_model=training.initial_model,model_payload=training.model_payload)
    old={k:getattr(base,k) for k in replace};module=campaign.MODULE
    try:
        for k,v in replace.items():setattr(base,k,v)
        evaluation.attach_visual,evaluation.check=attach,check
        campaign.MODULE=MODULE
        yield
    finally:
        for k,v in old.items():setattr(base,k,v)
        evaluation.attach_visual,evaluation.check=BASE_ATTACH,BASE_CHECK
        campaign.MODULE=module


def jobs(c):
    result=[]
    with scope():
        for arm in ARMS:
            for command,suffix,deps,verdict,steps in (
                ('train-smoke','smoke',[],'TRAIN_SMOKE_PASS',2),
                ('train','train',[arm+'_runtime'],'TRAIN_COMPLETE',c['steps'])):
                result.append(dict(id=arm+'_'+suffix,
                    cmd=[sys.executable,'-u','-m',MODULE,command,'--arm',arm],deps=deps,
                    summary=str(training.OUTPUT/('train_smoke' if suffix=='smoke' else 'train')/arm/'summary.json'),
                    completion=dict(identity=c['identity'],verdict=verdict,steps=steps)))
            row=evaluation.row_name(arm,8,1)
            result.append(dict(id=arm+'_runtime',cmd=[sys.executable,'-u','-m',MODULE,'runtime-smoke','--arm',arm],
                deps=[arm+'_smoke'],summary=str(training.OUTPUT/'pretraining_runtime'/row/'part0/eval_smoke'/row/'summary.json'),
                completion=dict(identity=c['identity'],verdict='EVAL_SMOKE_PASS')))
            result+=campaign.evaluation_jobs(c,arm,8,1,[arm+'_train'])
    campaign.validate_jobs(result)
    return result


def summarize():
    c=training.checked();rows={}
    for p in (training.OUTPUT/'rows').glob('*/summary.json'):
        r=read_json(p)
        if r['identity']!=c['identity'] or r['episodes']!=500:raise RuntimeError('Mixed row')
        rows[p.parent.name]=r
    write_json(training.OUTPUT/'comparison_summary.json',dict(identity=c['identity'],rows=rows,
        complete=len(rows)==3,hardware='sd1 RTX3090 eager',
        targets='Condition only / original NFE1 action only / joint; identical current-action branch, frozen70K recurrence, 5K matched samples.',
        evaluation='Long500 K8,NFE1,H10/R5; all videos, numeric observations and goal traces; all extra calls included in synchronized policy latency.'))
    return rows


def export():
    c=training.checked();rows=summarize()
    if len(rows)!=3:raise RuntimeError('Wait for all three complete comparisons')
    selected=min(rows,key=lambda name:(-rows[name]['successes'],rows[name]['policy_ms_per_action'],name))
    arm=rows[selected]['source_arm'];_,h=training.model_payload(c,arm)
    s=read_json(training.OUTPUT/'train'/arm/'summary.json')
    BUNDLE.mkdir(parents=True,exist_ok=True)
    shutil.copy2(s['checkpoint'],BUNDLE/'selected.pt')
    bundle=dict(arm=arm,training_identity=c['identity'],checkpoint_sha256=h,training_summary=s,
        all_sd1_rows=rows,manifest=c['manifest'],source_sha256=c['source_sha256'],
        parent_checkpoint_sha256=c['bundle']['sources']['selected']['checkpoint_sha256'],
        selection='Highest sd1 seed1 SR; ties lower latency then name. Independent rb2 results must retain this selection provenance.')
    write_json(BUNDLE/'bundle.json',bundle);write_json(BUNDLE/'READY.json',dict(sha256=sha(BUNDLE/'bundle.json')))
    remote='/home/mingyujung/private/gnaroshi_vla_storage/transfers/simvla_current_condition'
    subprocess.run(['ssh','rb2','mkdir','-p',remote],check=True)
    # Publish the completion marker last so the remote queue cannot read partial files.
    subprocess.run(['rsync','-a','--exclude=READY.json',str(BUNDLE)+'/',f'rb2:{remote}/'],check=True)
    subprocess.run(['rsync','-a',str(BUNDLE/'READY.json'),f'rb2:{remote}/'],check=True)
    write_json(training.OUTPUT/'export_summary.json',dict(verdict='EXPORT_COMPLETE',identity=c['identity'],arm=arm,checkpoint_sha256=h))


def remote_bundle():
    b=read_json(BUNDLE/'bundle.json')
    if sha(BUNDLE/'bundle.json')!=read_json(BUNDLE/'READY.json')['sha256'] or sha(BUNDLE/'selected.pt')!=b['checkpoint_sha256']:
        raise RuntimeError('Incomplete or changed transfer')
    for f,h in b['source_sha256'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('Source mismatch: '+f)
    return b


def remote_contract():
    b=remote_bundle();c=copy.deepcopy(portable.checked())
    c.pop('identity');c.update(current_branch=b)
    c['identity']=prior.digest(c)
    return c


def remote_evaluate(seed_index,smoke):
    import torch
    from tools.simvla import attention_query_eval as ev
    c=remote_contract();out=prior.STORAGE/'results/simvla/current_condition/compiled_confirmation_v1'
    seed=c['manifest']['action_noise_seed_base']+seed_index-1
    c['manifest']['action_noise_seed_base']=seed
    build=ev.prior.build_policy
    def builder(mode,cfg):
        p,compiler=build(mode,cfg);p.action_noise_seed_base=seed;return p,compiler
    def payload(contract,arm):
        b=remote_bundle();value=torch.load(BUNDLE/'selected.pt',map_location='cpu',weights_only=False)
        if (value['identity'],value['arm'],value['step'])!=(b['training_identity'],b['arm'],5000):
            raise RuntimeError('Training payload mismatch')
        return value,b['checkpoint_sha256']
    base_compile_check=ev.compile_check
    def compile_check(compiler,arm):
        base_compile_check(compiler,arm)
        if compiler and any(not compiler.records.get(name,{}).get('graphs',0) for name in ('current_condition_updater','current_condition_reader')):
            raise RuntimeError('Current-head compile bypass')
    writer=ev.write_json
    def write(path,value):
        if Path(path).name in ('summary.json','runtime.json') or Path(path).parent.name=='episodes':
            value=dict(value,action_noise_seed_base=seed,selected_training_arm=c['current_branch']['arm'])
        writer(path,value)
    ev.prior.build_policy=builder;ev.write_json=write;ev.compile_check=compile_check
    ev.attach_visual=attach;ev.check=lambda policy,arm:check(policy)
    ev.study=SimpleNamespace(OUTPUT=out,checked=lambda:c,model_payload=payload,
        initial_model=lambda contract,arm,native:install(portable.construct(native)),
        previous=SimpleNamespace(OUTPUT=out/'unused_reference'))
    ev.evaluate('selected_seed'+str(seed_index),smoke=smoke)


def run_remote():
    from tools.simvla import current_condition_reference as ref
    out=prior.STORAGE/'results/simvla/current_condition/compiled_confirmation_v1'
    # Waiting consumes no GPU; acquire GPU0 only after the reference and transfer finish.
    while not (BUNDLE/'READY.json').exists():
        write_json(out/'waiting.json',dict(phase='waiting_for_sd1_selected_checkpoint',updated_unix=time.time()))
        time.sleep(30)
    c=remote_contract();write_json(out/'contract.json',c)
    plan=[]
    for seed in (1,2):
        row='selected_seed'+str(seed)
        for smoke in (True,False):
            name=row+('_smoke' if smoke else '_eval')
            plan.append(dict(id=name,cmd=[sys.executable,'-u','-m',MODULE,
                'remote-smoke' if smoke else 'remote-eval','--seed-index',str(seed)],
                deps=[] if smoke else [row+'_smoke'],
                summary=str(out/('eval_smoke' if smoke else 'rows')/row/'summary.json'),
                completion=dict(identity=c['identity'],verdict='EVAL_SMOKE_PASS' if smoke else 'EVAL_COMPLETE')))
    return queue.run_queue(out,plan,gpus=(0,),predecessor=[dict(path=str(ref.OUTPUT),lock='queue.lock')],
        environment=prior.environment,cwd=ROOT,timeout=24*3600)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('all','preflight','train','train-smoke','runtime-smoke','eval-smoke','eval','merge','export','remote','remote-smoke','remote-eval'))
    p.add_argument('--arm',choices=ARMS);p.add_argument('--interval',type=int,choices=(8,),default=8)
    p.add_argument('--seed-index',type=int,choices=(1,2),default=1);p.add_argument('--shard',type=int,choices=range(4),default=0)
    a=p.parse_args()
    if a.command.startswith('remote'):
        if not prior.IS_RB2:p.error('rb2 only')
        if a.command=='remote':return run_remote()
        remote_evaluate(a.seed_index,a.command=='remote-smoke');return 0
    if prior.IS_RB2:p.error('sd1 training/evaluation only')
    if a.command=='export':export();return 0
    if a.command in ('train','train-smoke'):
        if not a.arm:p.error('--arm required')
        training.train(a.arm,a.command=='train-smoke');return 0
    if a.command in ('runtime-smoke','eval-smoke','eval','merge'):
        if not a.arm:p.error('--arm required')
        with scope():
            if a.command=='merge':evaluation.merge(a.arm);summarize()
            else:evaluation.evaluate(a.arm,shard=a.shard,smoke=a.command!='eval',
                training_smoke=a.command=='runtime-smoke')
        return 0
    c=training.prepare();plan=jobs(c);write_json(training.OUTPUT/'planned_jobs.json',plan)
    print('PREFLIGHT_PASS',len(plan),'jobs; three matched 5K trainings + three Long500 evaluations',flush=True)
    if a.command=='preflight':return 0
    rc=queue.run_queue(training.OUTPUT,plan,gpus=(4,5,6,7),predecessor=[],environment=prior.environment,cwd=ROOT,timeout=24*3600)
    summarize()
    if rc==0:
        for attempt in range(2):
            try:export();break
            except subprocess.CalledProcessError:
                if attempt:raise
                time.sleep(15)
    return rc


if __name__=='__main__':raise SystemExit(main())
