"""Equal-cost original-Condition interventions on current and subsequent queries."""
import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace

from methods.latentloop.modules.condition_routing import ARMS, install
from tools.simvla import failure_interventions as prior
from tools.simvla import geometry_comparison_rb2 as portable
from tools.simvla import gpu_followup_queue as queue
from tools.simvla import short_visual_train as base
from tools.simvla import short_visual_eval as evaluation
from tools.simvla import short_visual_campaign as campaign
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

OUTPUT = prior.STORAGE / 'results/simvla/condition_routing/original_midpoint_v1'
MODULE = 'tools.simvla.condition_routing_campaign'
FILES = ('methods/latentloop/modules/condition_routing.py',
         'tools/simvla/condition_routing_campaign.py', 'tests/test_condition_routing.py',
         'architectures/simvla/wrappers/run_condition_routing.sh')
BASE_ATTACH, BASE_CHECK = evaluation.attach_visual, evaluation.check


def prepare():
    if prior.IS_RB2:
        raise RuntimeError('Intervention campaign is assigned to sd1')
    parent, bundle = base.checked(), portable.validate_bundle()
    c = dict(model=parent['model'], bundle=bundle, manifest=bundle['manifest'],
        source_sha256=dict(parent['source_sha256']), arms=list(ARMS),
        target_age=4, new_training=False,
        protocol='Frozen70K K8,NFE1,H10/R5,Long500,independent env per episode,EGL.',
        intervention='At age4 evaluate original backbone once extra. Route its Condition to current action, next updater, or both. Retain all learned computations and vision calls in every arm.',
        cost='All arms have2 full backbones per8 queries,1 extra original vision,8 NFE1 actions. Include every call in policy latency. These are causal controls with higher cost than the unmodified K8.',
        comparison='Existing fullK4 resets every4 queries; these arms retain late updater, age counter, images and visual memory. Original target is available only at age4.',
        offline_windows=128)
    c['source_sha256'].update({f:sha(ROOT/f) for f in FILES})
    c['identity'] = prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Condition routing contract changed')
    write_json(path,c)
    write_json(OUTPUT/'source.json',dict(commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), files=c['source_sha256']))
    return c


def checked():
    c=read_json(OUTPUT/'contract.json')
    if prior.digest({k:v for k,v in c.items() if k!='identity'}) != c['identity']:
        raise RuntimeError('Contract identity changed')
    if portable.validate_bundle() != c['bundle']:
        raise RuntimeError('Frozen checkpoint bundle changed')
    for f,h in c['source_sha256'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('Source changed: '+f)
    return c


def initial_model(c,arm,native):
    return install(portable.construct(native),arm)


def payload(c,arm,smoke=False):
    return portable.payload(c,'selected_seed1')


def attach(policy,model,interval,compiler):
    policy=BASE_ATTACH(policy,model,interval,compiler)
    previous_update,previous_reset=policy._v0_update,policy.reset
    encode=policy.condition_adapter.encode_condition
    def counted_encode(*args,**kwargs):
        policy._observed_backbone_calls+=1
        return encode(*args,**kwargs)
    def reset(self):
        previous_reset()
        self._routing_calls=0
        self._observed_backbone_calls=0
        model.routing_reference=None
    def update(self,batch,*,age,policy_query_index):
        if age==4:
            model.routing_reference=self.condition_adapter.encode_condition(
                input_ids=batch['input_ids'], image_input=batch['image_input'],
                image_mask=batch['image_mask']).detach()
            self._routing_calls+=1
        return previous_update(batch,age=age,policy_query_index=policy_query_index)
    policy.condition_adapter.encode_condition=counted_encode
    policy.reset=MethodType(reset,policy)
    policy._v0_update=MethodType(update,policy)
    policy.reset()
    return policy


def check(policy):
    counts=BASE_CHECK(policy)
    expected=sum(t%policy.k_c==4 for t in range(counts['queries']))
    total=counts['backbone_calls']+expected
    if policy._routing_calls!=expected or policy._observed_backbone_calls!=total:
        raise RuntimeError('Actual original backbone calls differ from intervention contract')
    return dict(**counts,intervention_backbone_calls=expected,total_backbone_calls=total)


@contextmanager
def scope():
    replacements=dict(OUTPUT=OUTPUT,ARMS=ARMS,checked=checked,initial_model=initial_model,model_payload=payload)
    old={k:getattr(base,k) for k in replacements}
    old_module=campaign.MODULE
    try:
        for k,v in replacements.items():setattr(base,k,v)
        evaluation.attach_visual,evaluation.check=attach,check
        campaign.MODULE=MODULE
        yield
    finally:
        for k,v in old.items():setattr(base,k,v)
        evaluation.attach_visual,evaluation.check=BASE_ATTACH,BASE_CHECK
        campaign.MODULE=old_module


def offline():
    import torch
    from tools.simvla import visual_midpoint
    from tools.simvla.error_compensation_common import configure,snapshots
    from tools.simvla.visual_midpoint_features import CachedFeatures
    from tools.simvla.condition_output_split_train import action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from tools.simvla.readout_causality import metric
    from methods.latentloop.modules.action_aligned_joint import action_loss
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
    c=checked();cfg=c['model']['config'];configure(cfg);torch.set_num_threads(1)
    device,native,_,action,_,heldout=load_runtime(cfg,snapshots(cfg))
    value,checksum=payload(c,'both')
    features=CachedFeatures(visual_midpoint.OUTPUT/'features',visual_midpoint.checked()['identity'])
    models={arm:(portable.construct(native) if arm=='control' else initial_model(c,arm,native)).to(device).eval()
            for arm in ('control',)+ARMS}
    for model in models.values():
        model.load_state_dict(value['model'],strict=True);model.requires_grad_(False);model.feature_provider=features
    indices=_balanced_indices(heldout.identities,limit=c['offline_windows'],seed=7)
    records=[];began=time.monotonic()
    with torch.no_grad():
        for index in indices:
            seq=move_batch(collate_exact_teacher_sequences([heldout[index]]),device);features.bind(seq)
            first=sample_from_sequence(seq,1)
            contexts={arm:m.prepare(first['anchor'],first['anchor_images'],first['anchor_proprio'],
                first['valid'],first['groups'],8) for arm,m in models.items()}
            for age in range(1,8):
                s=sample_from_sequence(seq,age)
                original_action=action_prediction(action,s['target_condition'],s,steps=1,grad=False)
                target=action.action_space.normalize_action(s['target_action'])
                for arm,model in models.items():
                    if age==4 and arm!='control':model.routing_reference=s['target_condition']
                    z,_=model.predict(contexts[arm],age,s['images'],s['proprio'])
                    pred=action_prediction(action,z,s,steps=1,grad=False)
                    records.append(dict(window=index,arm=arm,age=age,
                        action_l1_vs_original10=float(action_loss(pred,target)),
                        action_l1_vs_original1=float(action_loss(pred,original_action)),
                        original1_l1_vs_original10=float(action_loss(original_action,target)),
                        **metric(z,s['target_condition'],s['valid'])))
            if len(records)%(28*8)==0:print('OFFLINE',len(records)//28,'/',len(indices),flush=True)
    metrics=('action_l1_vs_original10','action_l1_vs_original1','condition_mse','condition_cosine')
    means={arm:{str(age):{key:sum(r[key] for r in records if r['arm']==arm and r['age']==age)/len(indices)
            for key in metrics} for age in range(1,8)} for arm in models}
    # Exact same-input checks distinguish the two uses of the original Condition.
    for index in indices:
        row={(r['arm'],r['age']):r for r in records if r['window']==index}
        for arm in ARMS:
            for age in range(1,4):
                if any(row[arm,age][k]!=row['control',age][k] for k in metrics):
                    raise RuntimeError('Intervention changed a preceding query')
        for age in range(5,8):
            if any(row['current_action',age][k]!=row['control',age][k] for k in metrics):
                raise RuntimeError('Current-action intervention changed carried Condition')
        for arm in ('current_action','both'):
            if row[arm,4]['action_l1_vs_original1']!=0:
                raise RuntimeError('Original Condition did not reproduce original NFE1 action')
    write_json(OUTPUT/'offline/summary.json',dict(verdict='OFFLINE_COMPLETE',identity=c['identity'],
        windows=len(indices),records=records,by_age=means,checkpoint_sha256=checksum,
        elapsed_seconds=time.monotonic()-began,same_input_route_checks_pass=True,
        reference='Identical heldout observations/noise. Action targets original NFE1 and original NFE10; Condition target current z_t^F.'))


def jobs(c):
    result=[dict(id='offline',cmd=[sys.executable,'-u','-m',MODULE,'offline'],deps=[],
        summary=str(OUTPUT/'offline/summary.json'),completion=dict(verdict='OFFLINE_COMPLETE',identity=c['identity']))]
    with scope():
        for arm in ARMS:
            result+=campaign.evaluation_jobs(c,arm,8,1)
    campaign.validate_jobs(result)
    return result


def summarize():
    c=checked()
    rows={p.parent.name:read_json(p) for p in (OUTPUT/'rows').glob('*/summary.json')}
    records={name:read_json(OUTPUT/'rows'/name/'episode_outcomes.json') for name in rows}
    paired={}
    for a in records:
        for b in records:
            if a>=b:continue
            left={(r['task_id'],r['trial_id']):r['success'] for r in records[a]}
            right={(r['task_id'],r['trial_id']):r['success'] for r in records[b]}
            paired[a+'__'+b]=dict(a_only=sum(left[k] and not right[k] for k in left),
                b_only=sum(right[k] and not left[k] for k in left))
    write_json(OUTPUT/'comparison_summary.json',dict(identity=c['identity'],rows=rows,paired=paired,
        complete=len(rows)==3,new_training=False,
        interpretation='Equal additional original backbone cost. Compare current-action versus next-prediction routing; both may retain late-updater distribution mismatch. All original calls included in latency.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('all','preflight','offline','eval-smoke','eval','merge','summarize'))
    p.add_argument('--arm',choices=ARMS);p.add_argument('--interval',type=int,choices=(8,),default=8)
    p.add_argument('--seed-index',type=int,choices=(1,),default=1)
    p.add_argument('--shard',type=int,choices=range(4),default=0)
    a=p.parse_args()
    if prior.IS_RB2:p.error('sd1 only')
    if a.command=='offline':offline();return 0
    if a.command=='summarize':summarize();return 0
    if a.command in ('eval','eval-smoke','merge'):
        if not a.arm:p.error('--arm required')
        with scope():
            if a.command=='merge':evaluation.merge(a.arm);summarize()
            else:evaluation.evaluate(a.arm,shard=a.shard,smoke=a.command=='eval-smoke')
        return 0
    c=prepare();plan=jobs(c);write_json(OUTPUT/'planned_jobs.json',plan)
    print('PREFLIGHT_PASS',len(plan),'jobs; three Long500 interventions plus paired offline attribution',flush=True)
    if a.command=='preflight':return 0
    owner=queue.acquire_lock(OUTPUT/'campaign.lock')
    if owner is None:raise RuntimeError('Condition routing campaign already running')
    failed={}
    try:
        first=[j for j in plan if j['id']=='offline' or j['id'].endswith('_smoke')]
        groups=[('smoke_and_offline',first)]
        groups += [(arm,[j for j in plan if j['id'].startswith(arm+'_')]) for arm in ARMS]
        for name,local in groups:
            write_json(OUTPUT/'queue_status.json',dict(phase='running',active_phase=name,failed=failed,
                detail=str(OUTPUT/'queues'/name/'queue_status.json'),updated_unix=time.time()))
            queue.run_queue(OUTPUT/'queues'/name,local,gpus=(4,5,6,7),predecessor=[],
                environment=prior.environment,cwd=ROOT,timeout=12*3600)
            failed.update(read_json(OUTPUT/'queues'/name/'queue_status.json')['failed'])
            summarize()
        write_json(OUTPUT/'queue_status.json',dict(phase='complete' if not failed else 'finished_with_failures',
            failed=failed,pending=[],active={},updated_unix=time.time()))
        return int(bool(failed))
    finally:owner.close()


if __name__=='__main__':raise SystemExit(main())
