"""Same-checkpoint content/selector interventions with explicit oracle cost."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace

from tools.simvla import failure_interventions as prior
from tools.simvla import gpu_followup_queue as queue
from tools.simvla import geometry_comparison_rb2 as portable
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from tools.simvla.attention_query_eval import attach_visual as ORIGINAL_ATTACH
from methods.latentloop.modules.readout_intervention import (
    MODES, ReadoutIntervention, needs_previous_reference)

MODULE = 'tools.simvla.readout_causality'
OUTPUT = prior.STORAGE/'results/simvla/readout_causality/selected_k8_nfe1_v1'
ARMS = MODES[1:]
FILES = ('methods/latentloop/modules/readout_intervention.py',
         'tools/simvla/readout_causality.py', 'tests/test_readout_causality.py',
         'architectures/simvla/wrappers/run_readout_causality.sh')


def prepare():
    bundle = portable.validate_bundle()
    if prior.IS_RB2:
        parent = portable.checked()
        cfg = copy.deepcopy(parent['model']['config'])
    else:
        from tools.simvla import geometry_target_train
        parent = geometry_target_train.checked()
        cfg = copy.deepcopy(parent['model']['config'])
    c = dict(bundle=bundle, model=dict(config=cfg, sha256=bundle['parent_fresh_sha256']),
        manifest=bundle['manifest'], source_sha256=dict(parent['source_sha256']),
        selected=bundle['sources']['selected'], modes=list(MODES),
        intervention='Frozen70K; current vision KV and/or original previous Condition only in visual attention query. Predicted Condition remains updater/residual input.',
        reference_calls='Selector modes: full original backbone on ages3..6, used only for next-query attention. Explicitly counted and included in latency.',
        hardware='rb2 RTX5090 compiled' if prior.IS_RB2 else 'sd1 RTX3090 eager',
        parent_output=str(portable.OUTPUT), protocol='Long500,H10/R5,NFE1,K8,seed01,EGL')
    c['source_sha256'].update({f:sha(ROOT/f) for f in FILES})
    c['identity'] = prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Readout intervention contract changed')
    write_json(path, c)
    write_json(OUTPUT/'source.json', dict(commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), files=c['source_sha256']))
    return c


def checked():
    c = read_json(OUTPUT/'contract.json')
    if c['identity'] != prior.digest({k:v for k,v in c.items() if k != 'identity'}):
        raise RuntimeError('Contract identity mismatch')
    if portable.validate_bundle() != c['bundle']:
        raise RuntimeError('Checkpoint bundle changed')
    for f,h in c['source_sha256'].items():
        if sha(ROOT/f) != h:
            raise RuntimeError('Source changed: '+f)
    return c


def payload(c, arm):
    return portable.payload(c, 'selected_seed1')


def initial_model(c, arm, native):
    return ReadoutIntervention(native, arm)


def attach(policy, model, interval, compiler):
    policy = ORIGINAL_ATTACH(policy, model, interval, compiler)
    old_reset, old_update = policy.reset, policy._v0_update
    def reset(self):
        old_reset()
        self._selector_backbone_calls = 0
        model.original_previous = model.original_previous_age = None
    def update(self, batch, *, age, policy_query_index):
        result = old_update(batch, age=age, policy_query_index=policy_query_index)
        if needs_previous_reference(model.intervention_mode, self.k_c, age):
            model.original_previous = self.condition_adapter.encode_condition(
                input_ids=batch['input_ids'], image_input=batch['image_input'],
                image_mask=batch['image_mask']).detach()
            model.original_previous_age = age
            self._selector_backbone_calls += 1
        return result
    policy.reset = MethodType(reset, policy)
    policy._v0_update = MethodType(update, policy)
    policy.reset()
    return policy


def check(policy, arm):
    from tools.simvla.age_specialization_eval import check as base_check
    counts = base_check(policy, 'protected')
    model = policy._age_model
    expected_vision = sum(model.uses_vision(t % policy.k_c) for t in range(counts['queries']))
    expected_selector = sum(needs_previous_reference(arm,policy.k_c,t % policy.k_c)
                            for t in range(counts['queries']))
    if policy._extra_vision_calls != expected_vision or policy._selector_backbone_calls != expected_selector:
        raise RuntimeError('Intervention call count mismatch')
    return dict(**counts, extra_vision_calls=expected_vision,
        diagnostic_backbone_calls=expected_selector,
        total_backbone_calls=counts['backbone_calls']+expected_selector)


def evaluate(arm, smoke=False):
    from tools.simvla import attention_query_eval as evaluation
    c = checked()
    evaluation.study = SimpleNamespace(OUTPUT=OUTPUT, checked=lambda:c, model_payload=payload,
        initial_model=initial_model, previous=SimpleNamespace(OUTPUT=OUTPUT/'unused'))
    evaluation.attach_visual = attach
    evaluation.check = check
    evaluation.evaluate(arm, smoke=smoke)
    if not smoke:
        summarize()


def metric(prediction, target, valid):
    import torch
    mask = valid.unsqueeze(-1).expand_as(prediction)
    mse = float((prediction-target).square().masked_select(mask).mean())
    cos = torch.nn.functional.cosine_similarity(prediction, target, dim=-1)
    return dict(condition_mse=mse, condition_cosine=float(cos.masked_select(valid).mean()))


def offline(arm, smoke=False):
    import torch
    from tools.simvla import visual_midpoint
    from tools.simvla.error_compensation_common import configure, snapshots
    from tools.simvla.visual_midpoint_features import CachedFeatures
    from tools.simvla.condition_output_split_train import action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from methods.latentloop.modules.action_aligned_joint import action_loss
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices
    c=checked();cfg=c['model']['config'];configure(cfg)
    torch.set_num_threads(1)
    device,native,_,action,_,heldout=load_runtime(cfg,snapshots(cfg))
    value,checksum=payload(c,arm)
    model=initial_model(c,arm,native).to(device).eval()
    model.load_state_dict(value['model'],strict=True);model.requires_grad_(False)
    features=CachedFeatures(visual_midpoint.OUTPUT/'features',visual_midpoint.checked()['identity'])
    model.feature_provider=features
    indices=_balanced_indices(heldout.identities,limit=1 if smoke else 30,seed=7)
    records=[];started=time.monotonic()
    reference_model=None
    if arm=='control':
        reference_model=portable.construct(native).to(device).eval()
        reference_model.load_state_dict(value['model'],strict=True)
        reference_model.feature_provider=features
    with torch.no_grad():
        for index in indices:
            seq=move_batch(collate_exact_teacher_sequences([heldout[index]]),device)
            features.bind(seq)
            sample=sample_from_sequence(seq,1)
            context=model.prepare(sample['anchor'],sample['anchor_images'],sample['anchor_proprio'],
                                  sample['valid'],sample['groups'],8)
            if reference_model is not None:
                reference_context=reference_model.prepare(sample['anchor'],sample['anchor_images'],
                    sample['anchor_proprio'],sample['valid'],sample['groups'],8)
            for age in range(1,8):
                sample=sample_from_sequence(seq,age)
                if age>=4:
                    model.original_previous=seq['teacher_conditions'][:,age-2]
                    model.original_previous_age=age-1
                z,_=model.predict(context,age,sample['images'],sample['proprio'])
                parity=None
                if reference_model is not None:
                    expected,_=reference_model.predict(reference_context,age,sample['images'],sample['proprio'])
                    parity=float((expected-z).abs().max())
                    if parity!=0:
                        raise RuntimeError('Control implementation differs from completed70K model')
                prediction=action_prediction(action,z,sample,steps=1,grad=False)
                original_one=action_prediction(action,seq['teacher_conditions'][:,age-1],sample,steps=1,grad=False)
                target=action.action_space.normalize_action(sample['target_action'])
                records.append(dict(window=index,age=age,action_l1=float(action_loss(prediction,target)),
                    action_l1_vs_original1=float(action_loss(prediction,original_one)),
                    original1_l1_vs_original10=float(action_loss(original_one,target)),
                    control_max_diff=parity,**metric(z,seq['teacher_conditions'][:,age-1],sample['valid'])))
    output=dict(verdict='OFFLINE_SMOKE_PASS' if smoke else 'OFFLINE_COMPLETE',identity=c['identity'],
        arm=arm,model_sha256=checksum,windows=len(indices),records=records,
        elapsed_seconds=time.monotonic()-started,
        scope='Same heldout observations and originalNFE10 first5 targets; teacher previous query supplied from cache. No deployment latency claim.')
    write_json(OUTPUT/('offline_smoke' if smoke else 'offline')/arm/'summary.json',output)
    print(output['verdict'],arm,len(indices),flush=True)


def offline_summary():
    c=checked();data={}
    for arm in MODES:
        path=OUTPUT/'offline'/arm/'summary.json'
        if path.exists():
            value=read_json(path)
            if value['identity']!=c['identity'] or value['windows']!=30:
                raise RuntimeError('Offline result identity/size mismatch')
            records=value['records']
            data[arm]=dict(by_age={str(age):{metric:sum(r[metric] for r in records if r['age']==age)/30
                for metric in ('action_l1','action_l1_vs_original1','original1_l1_vs_original10',
                               'condition_mse','condition_cosine')} for age in range(1,8)},
                windows=sorted({r['window'] for r in records}),source=str(path))
    if data and any(v['windows']!=next(iter(data.values()))['windows'] for v in data.values()):
        raise RuntimeError('Unpaired heldout windows')
    write_json(OUTPUT/'offline_comparison.json',dict(identity=c['identity'],complete=len(data)==4,rows=data))


def summarize():
    c=checked();reference=read_json(portable.OUTPUT/'rows/selected_seed1/summary.json')
    refdir=portable.OUTPUT/'rows/selected_seed1/episodes'
    expected={(x['task_id'],x['trial_id']) for x in c['manifest']['episodes']}
    ref={(d['task_id'],d['trial_id']):d for d in (read_json(p) for p in refdir.glob('*.json'))}
    if (set(ref)!=expected or len(ref)!=500 or len(expected)!=500
            or any(d['model_sha256']!=c['selected']['checkpoint_sha256'] for d in ref.values())
            or reference['successes']!=sum(d['success'] for d in ref.values())):
        raise RuntimeError('Incomplete completed control')
    rows={}
    for arm in ARMS:
        path=OUTPUT/'rows'/arm/'summary.json'
        if not path.exists():continue
        s=read_json(path)
        data=[read_json(p) for p in (path.parent/'episodes').glob('*.json')]
        if (s['identity']!=c['identity'] or s['episodes']!=500 or len(data)!=500
                or {(d['task_id'],d['trial_id']) for d in data}!=expected
                or any(d['identity']!=c['identity'] or d['model_sha256']!=c['selected']['checkpoint_sha256'] for d in data)):
            raise RuntimeError('Mixed/incomplete intervention episodes')
        s['paired_with_control']=dict(
            intervention_only=sum(d['success'] and not ref[d['task_id'],d['trial_id']]['success'] for d in data),
            control_only=sum(not d['success'] and ref[d['task_id'],d['trial_id']]['success'] for d in data))
        s['failure_progress']=failure_progress(data)
        rows[arm]=s
    interaction=None
    if len(rows)==3:
        interaction=(rows['both']['success_rate']-rows['current_content']['success_rate']
                     -rows['original_selector']['success_rate']+reference['success_rate'])
    write_json(OUTPUT/'comparison_summary.json',dict(identity=c['identity'],complete=len(rows)==3,
        reference=reference,reference_failure_progress=failure_progress(list(ref.values())),rows=rows,
        descriptive_success_interaction=interaction,
        timing_scope='All added original calls included. Selector/both are diagnostic interventions with extra backbone calls; report actual call counts and diagnostic runtime.'))


def failure_progress(records):
    result=dict(failed=0,failed_after_new_goal=0,new_goal_lost_at_end=0)
    for record in records:
        if record['success']:continue
        result['failed']+=1
        trace=record.get('query_goal_progress',[])
        if not trace:raise RuntimeError('Missing goal trace')
        initial=trace[0]['satisfied'];final=record['terminal_goal_progress']['satisfied']
        achieved=[not initial[i] and any(s['satisfied'][i] for s in trace) for i in range(len(initial))]
        result['failed_after_new_goal']+=int(any(achieved))
        result['new_goal_lost_at_end']+=int(any(v and not final[i] for i,v in enumerate(achieved)))
    return result


def jobs(c):
    def job(name,command,arm,deps=(),smoke=False):
        folder=('eval_smoke' if smoke else 'rows') if command in ('smoke','eval') else ('offline_smoke' if smoke else 'offline')
        verdict={'smoke':'EVAL_SMOKE_PASS','eval':'EVAL_COMPLETE',
                 'offline-smoke':'OFFLINE_SMOKE_PASS','offline':'OFFLINE_COMPLETE'}[command]
        return dict(id=name,cmd=[sys.executable,'-u','-m',MODULE,command,'--arm',arm],deps=list(deps),
            summary=str(OUTPUT/folder/arm/'summary.json'),completion=dict(identity=c['identity'],verdict=verdict,
                **(dict(episodes=500) if command=='eval' else {})))
    result=[]
    if prior.IS_RB2:
        for arm in ARMS:
            result.extend([job(arm+'_smoke','smoke',arm,smoke=True),
                           job(arm+'_eval','eval',arm,[arm+'_smoke'])])
    else:
        for arm in MODES:
            result.extend([job(arm+'_offline_smoke','offline-smoke',arm,smoke=True),
                           job(arm+'_offline','offline',arm,[arm+'_offline_smoke'])])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('all','preflight','smoke','eval','offline','offline-smoke'))
    p.add_argument('--arm',choices=MODES)
    a=p.parse_args()
    if a.command in ('all','preflight'):
        c=prepare();plan=jobs(c)
        write_json(OUTPUT/'planned_jobs.json',plan)
        if a.command=='preflight':print('PREFLIGHT_PASS',len(plan));return 0
        predecessor=([dict(path=str(portable.OUTPUT),lock='queue.lock')] if prior.IS_RB2 else
            [dict(path=str(prior.STORAGE/'results/simvla/action_coverage/fresh_nfe1_v1'),lock='campaign.lock')])
        result=queue.run_queue(OUTPUT,plan,gpus=(0,) if prior.IS_RB2 else (4,5,6,7),
            predecessor=predecessor,environment=prior.environment,cwd=ROOT,timeout=8*3600)
        if prior.IS_RB2:summarize()
        else:offline_summary()
        return result
    if not a.arm:p.error('--arm required')
    if a.command in ('smoke','eval'):
        if not prior.IS_RB2 or a.arm=='control':p.error('Online: three rb2 interventions only')
        evaluate(a.arm,a.command=='smoke')
    else:
        if prior.IS_RB2:p.error('Heldout cache remains on sd1')
        offline(a.arm,a.command=='offline-smoke')
    return 0


if __name__=='__main__':raise SystemExit(main())
