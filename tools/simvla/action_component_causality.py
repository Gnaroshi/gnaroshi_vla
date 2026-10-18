"""Localize the measured age4 original-action benefit before another training run."""
import argparse
import copy
from contextlib import contextmanager
import subprocess
import sys
import time
from types import MethodType

from tools.simvla import condition_routing_campaign as routing
from tools.simvla import current_condition_train as predecessor
from tools.simvla import failure_interventions as prior
from tools.simvla import geometry_comparison_rb2 as portable
from tools.simvla import gpu_followup_queue as queue
from tools.simvla import short_visual_train as base
from tools.simvla import short_visual_eval as evaluation
from tools.simvla import short_visual_campaign as campaign
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

ARMS = ('gripper_midpoint', 'pose_midpoint', 'late_actions')
OUTPUT = prior.STORAGE / 'results/simvla/action_component_causality/original_action_v1'
MODULE = 'tools.simvla.action_component_causality'
FILES = ('tools/simvla/action_component_causality.py', 'tests/test_action_component_causality.py',
         'architectures/simvla/wrappers/run_action_component_causality.sh')
BASE_ATTACH, BASE_CHECK = evaluation.attach_visual, evaluation.check


def intervention_ages(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return (5, 6, 7) if arm == 'late_actions' else (4,)


def merge_action(predicted, original, arm):
    if arm not in ARMS or predicted.shape != original.shape or predicted.shape[-1] != 7:
        raise ValueError('Expected matching seven-component action chunks and a supported arm')
    result = predicted.clone()
    if arm == 'gripper_midpoint':
        result[..., 6:] = original[..., 6:]
    elif arm == 'pose_midpoint':
        result[..., :6] = original[..., :6]
    else:
        result.copy_(original)
    return result


def prepare():
    if prior.IS_RB2:
        raise RuntimeError('Causal controls are assigned to sd1')
    parent = routing.checked()
    c = copy.deepcopy(parent)
    c.pop('identity')
    c.update(arms=list(ARMS), new_training=False, training_seconds=0,
        intervention='Frozen70K K8/NFE1. Original action replaces gripper only atage4, pose only atage4, or all seven coordinates atages5/6/7. Predicted Condition remains in the recurrent context.',
        cost='Retain every parent call; additionally run original backbone and NFE1 at each intervention. First two arms add1+1 calls per8 queries; late_actions adds3+3. Count all work in synchronized policy latency.',
        comparison='Reuse completed original-Condition age4/current-action control and unmodified70K. This decomposes the changed action coordinates and tests the remaining late temporal coverage; prior gripper losses and midpoint recurrent-reset training are not repeated.',
        limitation='Action component hybrids may disrupt coordination. Their effects are interventions, not an additive attribution of failure or a deployable efficiency result.',
        predecessor=str(predecessor.OUTPUT), reference_identity=parent['identity'])
    c['source_sha256'].update({f: sha(ROOT/f) for f in FILES})
    c['identity'] = prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Action component contract changed')
    write_json(path, c)
    write_json(OUTPUT/'source.json', dict(commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), files=c['source_sha256']))
    return c


def checked():
    c = read_json(OUTPUT/'contract.json')
    if prior.digest({k:v for k,v in c.items() if k != 'identity'}) != c['identity']:
        raise RuntimeError('Contract digest changed')
    if portable.validate_bundle() != c['bundle']:
        raise RuntimeError('Frozen70K input bundle changed')
    for f, h in c['source_sha256'].items():
        if sha(ROOT/f) != h:
            raise RuntimeError('Source changed: ' + f)
    return c


def initial_model(c, arm, native):
    intervention_ages(arm)
    model = portable.construct(native)
    model.component_arm = arm
    previous_prepare = model.prepare
    def prepare_model(self, anchor, images, proprio, valid, groups, interval=8):
        return None if interval == 1 else previous_prepare(anchor, images, proprio, valid, groups, interval)
    model.prepare = MethodType(prepare_model, model)
    return model


def payload(c, arm, smoke=False):
    return portable.payload(c, 'selected_seed1')


def install_action_intervention(policy, model):
    import torch
    previous_update, previous_reset = policy._v0_update, policy.reset
    encode = policy.condition_adapter.encode_condition
    decode = policy.action_adapter.decode_action_from_condition
    arm = model.component_arm
    def counted_encode(*args, **kwargs):
        policy._observed_backbone_calls += 1
        return encode(*args, **kwargs)
    def reset(self):
        previous_reset()
        self._component_calls = self._observed_backbone_calls = self._reference_action_calls = 0
        self._component_by_age = {}
    def update(self, batch, *, age, policy_query_index):
        z, chunk, seed = previous_update(batch, age=age, policy_query_index=policy_query_index)
        if age not in intervention_ages(arm):
            return z, chunk, seed
        original_z = self.condition_adapter.encode_condition(input_ids=batch['input_ids'],
            image_input=batch['image_input'], image_mask=batch['image_mask'])
        noise, original_seed = self._paired_initial_noise(original_z, batch['proprio'], policy_query_index)
        if original_seed != seed:
            raise RuntimeError('Original and predicted actions used different initial noise seeds')
        original_chunk = decode(original_z, batch['proprio'], steps=1, initial_noise=noise)
        result = merge_action(chunk, original_chunk, arm)
        if not torch.isfinite(result).all():
            raise RuntimeError('Nonfinite intervened action')
        # Only the queued action changes. Both Condition caches retain the parent prediction.
        if not torch.equal(self._age_context.previous, z) or not torch.equal(self.cached_condition, z):
            raise RuntimeError('Parent recurrent/cached Condition changed')
        self.cached_action_chunk = result.detach()
        self._component_calls += 1
        self._reference_action_calls += 1
        stats = self._component_by_age.setdefault(str(age), dict(queries=0, pose_abs_sum=0.,
            pose_elements=0, gripper_different=0, gripper_elements=0))
        stats['queries'] += 1
        stats['pose_abs_sum'] += float((chunk[..., :5, :6]-original_chunk[..., :5, :6]).abs().sum())
        stats['pose_elements'] += chunk[..., :5, :6].numel()
        stats['gripper_different'] += int((chunk[..., :5, 6] != original_chunk[..., :5, 6]).sum())
        stats['gripper_elements'] += chunk[..., :5, 6].numel()
        return z, result, seed
    policy.condition_adapter.encode_condition = counted_encode
    policy.reset, policy._v0_update = MethodType(reset, policy), MethodType(update, policy)
    policy.reset()
    return policy


def attach(policy, model, interval, compiler):
    return install_action_intervention(BASE_ATTACH(policy, model, interval, compiler), model)


def check(policy):
    counts = BASE_CHECK(policy)
    expected = sum(t % policy.k_c in intervention_ages(policy._age_model.component_arm)
                   for t in range(counts['queries']))
    total_backbone = counts['backbone_calls'] + expected
    if (policy._component_calls != expected or policy._reference_action_calls != expected
            or policy._observed_backbone_calls != total_backbone):
        raise RuntimeError('Observed intervention calls differ from the contract')
    return dict(**counts, intervention_calls=expected, total_backbone_calls=total_backbone,
        total_action_transformer_calls=counts['action_transformer_calls']+expected,
        action_component_differences=copy.deepcopy(policy._component_by_age))


@contextmanager
def scope():
    replacements = dict(OUTPUT=OUTPUT, ARMS=ARMS, checked=checked, initial_model=initial_model, model_payload=payload)
    old = {k:getattr(base, k) for k in replacements}
    old_module = campaign.MODULE
    try:
        for k, v in replacements.items():
            setattr(base, k, v)
        evaluation.attach_visual, evaluation.check = attach, check
        campaign.MODULE = MODULE
        yield
    finally:
        for k, v in old.items():
            setattr(base, k, v)
        evaluation.attach_visual, evaluation.check = BASE_ATTACH, BASE_CHECK
        campaign.MODULE = old_module


def jobs(c):
    with scope():
        plan = [job for arm in ARMS for job in campaign.evaluation_jobs(c, arm, 8, 1)]
    campaign.validate_jobs(plan)
    return plan


def summarize():
    c = checked()
    rows = {p.parent.name:read_json(p) for p in (OUTPUT/'rows').glob('*/summary.json')}
    references = {}
    for arm in ('current_action', 'next_prediction', 'both'):
        path = routing.OUTPUT/'rows'/f'{arm}_k8_seed1'/'summary.json'
        if path.exists():
            references[arm] = dict(path=str(path), sha256=sha(path), result=read_json(path))
    paired = {}
    reference_path = routing.OUTPUT/'rows/current_action_k8_seed1/episode_outcomes.json'
    if reference_path.exists():
        reference = {(r['task_id'],r['trial_id']):r for r in read_json(reference_path)}
        for row in rows:
            records = read_json(OUTPUT/'rows'/row/'episode_outcomes.json')
            if len(records) != 500 or set(reference) != {(r['task_id'],r['trial_id']) for r in records}:
                raise RuntimeError('Paired outcomes do not cover the same500 episodes')
            if any(r['model_sha256'] != reference[r['task_id'],r['trial_id']]['model_sha256']
                   or r['action_noise_seed_base'] != reference[r['task_id'],r['trial_id']]['action_noise_seed_base']
                   for r in records):
                raise RuntimeError('Reference checkpoint or noise mismatch')
            paired[row] = dict(intervention_only=sum(r['success'] and not reference[r['task_id'],r['trial_id']]['success'] for r in records),
                original_age4_only=sum(not r['success'] and reference[r['task_id'],r['trial_id']]['success'] for r in records))
    write_json(OUTPUT/'comparison_summary.json', dict(identity=c['identity'], rows=rows,
        references=references, paired_vs_original_age4=paired, complete=len(rows)==3,
        training_seconds=0, interpretation=c['limitation'], updated_unix=time.time()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('all', 'preflight', 'eval-smoke', 'eval', 'merge', 'summarize'))
    p.add_argument('--arm', choices=ARMS)
    p.add_argument('--interval', type=int, choices=(8,), default=8)
    p.add_argument('--seed-index', type=int, choices=(1,), default=1)
    p.add_argument('--shard', type=int, choices=range(4), default=0)
    a = p.parse_args()
    if prior.IS_RB2:
        p.error('sd1 only; retain rb2 for compiled confirmation of current branch')
    if a.command == 'summarize':
        summarize(); return 0
    if a.command in ('eval-smoke', 'eval', 'merge'):
        if not a.arm:
            p.error('--arm required')
        with scope():
            if a.command == 'merge':
                evaluation.merge(a.arm); summarize()
            else:
                evaluation.evaluate(a.arm, shard=a.shard, smoke=a.command=='eval-smoke')
        return 0
    c = prepare(); plan = jobs(c)
    write_json(OUTPUT/'planned_jobs.json', plan)
    print('PREFLIGHT_PASS 3x500 causal episodes, no retraining, parent70K reused', flush=True)
    if a.command == 'preflight':
        return 0
    owner = queue.acquire_lock(OUTPUT/'campaign.lock')
    if owner is None:
        raise RuntimeError('Action-component campaign already running')
    failures = {}
    predecessor_spec = [dict(path=str(predecessor.OUTPUT), lock='queue.lock')]
    try:
        for arm in ARMS:
            local = [j for j in plan if j['id'].startswith(arm+'_')]
            write_json(OUTPUT/'queue_status.json', dict(phase='running', active_arm=arm,
                failed=failures, detail=str(OUTPUT/'queues'/arm/'queue_status.json'), updated_unix=time.time()))
            queue.run_queue(OUTPUT/'queues'/arm, local, gpus=(4,5,6,7), predecessor=predecessor_spec,
                environment=prior.environment, cwd=ROOT, timeout=12*3600)
            failures.update(read_json(OUTPUT/'queues'/arm/'queue_status.json')['failed'])
            summarize()
        write_json(OUTPUT/'queue_status.json', dict(phase='finished_with_failures' if failures else 'complete',
            failed=failures, pending=[], active={}, updated_unix=time.time()))
        return int(bool(failures))
    finally:
        owner.close()


if __name__ == '__main__':
    raise SystemExit(main())
