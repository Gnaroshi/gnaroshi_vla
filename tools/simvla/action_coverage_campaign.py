"""Direct action supervision at every deployed query age, from fresh initialization."""
import argparse
import copy
from contextlib import contextmanager
import random
import subprocess
import sys
import time

from tools.simvla import short_visual_train as base
from tools.simvla import short_visual_eval as evaluation
from tools.simvla import short_visual_campaign as campaign
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

PARENT = base.OUTPUT
OUTPUT = base.prior.STORAGE/'results/simvla/action_coverage/fresh_nfe1_v1'
MODULE = 'tools.simvla.action_coverage_campaign'
ARMS = ('fresh_joint', 'fresh_all_query', 'protected_short')
FILES = ('tools/simvla/action_coverage_campaign.py',
         'architectures/simvla/wrappers/run_action_coverage.sh',
         'tests/test_action_coverage.py')
TRAIN = base.train
PAYLOAD = base.model_payload


def prepare():
    source = base.checked()
    status = read_json(PARENT/'comparison_status.json')
    if status['verdict'] != 'COMPARISON_READY':
        raise RuntimeError('Previous short-training comparison is incomplete')
    c = copy.deepcopy(source)
    c['arms'] = list(ARMS)
    c['initialization']['fresh_all_query'] = copy.deepcopy(c['initialization']['fresh_joint'])
    c['source_sha256'].update({f:sha(ROOT/f) for f in FILES})
    c['input_sources']['previous_short_contract'] = dict(path=str(PARENT/'contract.json'),sha256=sha(PARENT/'contract.json'))
    c['training_ages'] = list(range(1,8))
    c['objective'] = dict(
        fresh_joint='One terminal action target per sampled window, query age balanced over1..7. Same10K optimizer updates,batch2. First2 updates use5,7 for block-gradient audit.',
        fresh_all_query='One full unroll1..7; mean of seven current-query first5 action L1 losses plus0.05 mean Condition MSE. Fresh10K.',
        protected_short='Frozen fresh10K early; mean of four current-query action losses at4..7 in one full unroll. Late and visual train10K. Total20K including parent.')
    c['gradient_routing'] = 'All-age arms directly supervise early actions. Dense losses also propagate through earlier predicted Conditions. Protected early remains frozen; original SimVLA is frozen in all arms.'
    c['comparison'] = 'Same inference as short-training. Changes action supervision coverage, not architecture. Dense arms have more action targets and measured compute; do not claim equal FLOPs.'
    c.pop('identity',None)
    c['identity'] = base.prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path)!=c:
        raise RuntimeError('Action-coverage contract changed')
    write_json(path,c)
    write_json(OUTPUT/'source.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=c['source_sha256']))
    return c


def checked():
    c = read_json(OUTPUT/'contract.json')
    if c['identity'] != base.prior.digest({k:v for k,v in c.items() if k!='identity'}):
        raise RuntimeError('Corrupted coverage contract')
    for f,h in c['source_sha256'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('Source changed: '+f)
    for record in c['input_sources'].values():base._verify_record(record)
    for path,h in c['frozen_original_sha256'].items():
        if sha(path)!=h:raise RuntimeError('Original asset changed: '+path)
    return c


def initial_model(c, arm, native):
    if arm not in ARMS:raise ValueError(arm)
    canonical = 'protected_short' if arm=='protected_short' else 'fresh_joint'
    state = base._protected_payload(c)['model'] if canonical=='protected_short' else None
    model = base.build_model(native,canonical,seed=c['seed'],early_state=state)
    model.supervision_variant = arm
    return model


def sampled_age(arm,step,smoke=False):
    if arm not in ARMS or step<1:raise ValueError('Invalid sample')
    if step<=2:return (5,7)[step-1]
    if smoke:raise ValueError('Only two smoke updates')
    return 1+(step-3)%7 if arm=='fresh_joint' else 7


def action_ages(arm,terminal):
    if arm=='fresh_joint':return [terminal]
    if arm=='fresh_all_query':return list(range(1,terminal+1))
    if arm=='protected_short':return list(range(4,terminal+1))
    raise ValueError(arm)


def loss_for(c,model,action,seq,age,features):
    from tools.simvla.condition_output_split_train import unroll,action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from methods.latentloop.modules.trend_condition import scaled_mse
    from methods.latentloop.modules.action_aligned_joint import action_loss
    features.bind(seq)
    _,predictions,_ = unroll(model,seq,age,8)
    ages = action_ages(model.supervision_variant,age)
    losses=[]
    for j in ages:
        sample=sample_from_sequence(seq,j)
        prediction=action_prediction(action,predictions[j-1],sample,steps=1)
        losses.append(action_loss(prediction,action.action_space.normalize_action(sample['target_action'])))
    loss_a=sum(losses)/len(losses)
    sample=sample_from_sequence(seq,age)
    loss_c=sum(scaled_mse(z,seq['teacher_conditions'][:,j],sample['anchor'],sample['valid'])
               for j,z in enumerate(predictions))/len(predictions)
    return loss_a+c['condition_weight']*loss_c,loss_a,loss_c


@contextmanager
def scope(arm):
    replacements=dict(OUTPUT=OUTPUT,ARMS=ARMS,checked=checked,initial_model=initial_model,loss_for=loss_for,
        sampling_step=lambda c,n,smoke=False:(random.Random(c['seed']*100000+n),sampled_age(arm,n,smoke)))
    old={k:getattr(base,k) for k in replacements}
    module=campaign.MODULE
    try:
        for k,v in replacements.items():setattr(base,k,v)
        campaign.MODULE=MODULE
        yield
    finally:
        for k,v in old.items():setattr(base,k,v)
        campaign.MODULE=module


def jobs(c):
    result=[]
    with scope(None):
        for arm in ARMS:
            local=campaign.study_jobs(c,arm)
            if arm=='fresh_all_query':
                k4=campaign.evaluation_jobs(c,arm,4,1,['train_'+arm])
                next(j for j in local if j['id'].endswith('k8_seed1_smoke'))['deps']=[k4[-1]['id']]
                local+=k4
            result+=local
    campaign.validate_jobs(result)
    return result


def summarize():
    c=checked()
    rows={p.parent.name:read_json(p) for p in (OUTPUT/'rows').glob('*/summary.json')}
    for row in rows.values():
        if row['identity']!=c['identity'] or row['episodes']!=500:raise RuntimeError('Incomplete row')
    result=dict(identity=c['identity'],rows=rows,training={a:read_json(OUTPUT/'train'/a/'summary.json')
        for a in ARMS if (OUTPUT/'train'/a/'summary.json').exists()},
        previous_short_comparison=str(PARENT/'comparison_summary.json'),updated_unix=time.time())
    write_json(OUTPUT/'comparison_summary.json',result)
    return result


def run():
    owner=campaign.queue.acquire_lock(OUTPUT/'campaign.lock')
    if owner is None:raise RuntimeError('Coverage campaign already running')
    try:
        c=prepare();plan=jobs(c)
        write_json(OUTPUT/'planned_jobs.json',plan)
        campaign.queue.run_queue(OUTPUT/'primary',plan,gpus=(4,5,6,7),predecessor=[],
            environment=base.prior.environment,cwd=ROOT,timeout=12*3600)
        result=summarize()
        candidates={a:result['rows'][f'{a}_k8_seed1'] for a in ARMS
                    if f'{a}_k8_seed1' in result['rows']}
        state=read_json(OUTPUT/'primary/queue_status.json')
        if candidates:
            selected=campaign.select_short(candidates)
            write_json(OUTPUT/'selection.json',dict(arm=selected,seed01=candidates[selected]))
            with scope(selected):followup=campaign.evaluation_jobs(c,selected,8,2)
            write_json(OUTPUT/'followup_jobs.json',followup)
            campaign.queue.run_queue(OUTPUT/'confirmation',followup,gpus=(4,5,6,7),predecessor=[],
                environment=base.prior.environment,cwd=ROOT,timeout=12*3600)
            state2=read_json(OUTPUT/'confirmation/queue_status.json')
            state['failed'].update(state2['failed'])
        else:state['failed']['confirmation']='No complete checkpoint'
        summarize()
        write_json(OUTPUT/'queue_status.json',dict(phase='finished_with_failures' if state['failed'] else 'complete',
            failed=state['failed'],updated_unix=time.time()))
        return int(bool(state['failed']))
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('all','preflight','train','train-smoke','runtime-smoke','eval-smoke','eval','merge','summarize'))
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--interval',type=int,choices=(4,8),default=8)
    p.add_argument('--seed-index',type=int,choices=(1,2),default=1)
    p.add_argument('--shard',type=int,choices=range(4),default=0)
    a=p.parse_args()
    if base.prior.IS_RB2:p.error('sd1 training campaign')
    if a.command=='all':return run()
    if a.command=='preflight':
        c=prepare();print('PREFLIGHT_PASS',len(jobs(c)),'jobs +6 confirmation');return 0
    if a.command=='summarize':print(summarize());return 0
    if not a.arm:p.error('--arm required')
    with scope(a.arm):
        if a.command in ('train','train-smoke'):TRAIN(a.arm,smoke=a.command=='train-smoke')
        elif a.command=='merge':evaluation.merge(a.arm,a.interval,a.seed_index)
        else:evaluation.evaluate(a.arm,a.interval,a.seed_index,a.shard,
            smoke=a.command!='eval',training_smoke=a.command=='runtime-smoke',bounded=a.command=='runtime-smoke')
    return 0


if __name__=='__main__':raise SystemExit(main())
