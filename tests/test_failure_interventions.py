from collections import Counter, deque
from types import SimpleNamespace

import pytest
import torch

from tools.simvla.failure_interventions import (
    ARMS, attach_recorder, execution_plan, select_cases, select_queries,
)


def test_four_arms_and_original_continuation():
    for arm in ARMS:
        assert execution_plan(2, 8, 3, arm, 'isolate') == dict(full=False, nfe=1, replace_condition=False)
        at = execution_plan(3, 8, 3, arm, 'isolate')
        assert at['nfe'] == (10 if arm.endswith('_10') else 1)
        assert at['replace_condition'] == arm.startswith('original_')
        assert execution_plan(4, 8, 3, arm, 'isolate') == dict(full=True, nfe=10, replace_condition=False)
        assert execution_plan(4, 8, 3, arm, 'continue') == dict(full=False, nfe=1, replace_condition=False)
        assert execution_plan(8, 8, 3, arm, 'continue')['full']


def test_case_selection_identical_ids_unique_and_explained():
    modes = ('bounded_k4', 'recurrent_k4', 'bounded_k8', 'recurrent_k8')
    outcomes = {m: {(t,r): (r!=2 if m.endswith('4') else r==3) for t in (6,8,9) for r in range(5)} for m in modes}
    cases = select_cases(outcomes, (6,8,9))
    assert len(cases) == 6
    assert [(c['task_id'],c['trial_id']) for c in cases] == [(t,r) for t in (6,8,9) for r in (0,2)]
    assert {c['selection'] for c in cases} == {'K4_success_K8_failure','K4_failure'}


@pytest.mark.parametrize('k', (4,8))
def test_branch_locations_non_refresh_unique_and_with_remaining_future(k):
    trace = [dict(query=t, object_displacement=float(t==5), goals=[t>=4,t>=28]) for t in range(40)]
    points = select_queries(trace,k)
    assert len(points) == 2
    assert len({p['query'] for p in points}) == 2
    assert all(0<p['query']<39 and p['query']%k for p in points)


class Policy:
    def __init__(self):
        self.k_c=4
        self.nfe=1
        self.condition_adapter=SimpleNamespace(encode_condition=lambda **kw: torch.full((1,2,3), 2.))
        self.reset()

    def reset(self):
        self.metrics=SimpleNamespace(counters=Counter())
        self.query_index=0
        self.action_queue=deque()
        self.query_trace=[]

    def _decode(self, condition, proprio, *, policy_query_index):
        self.metrics.counters['num_action_transformer_calls']+=self.nfe
        return torch.ones(1,10,7)*condition.mean(), 123+policy_query_index

    def _full_refresh(self, batch, *, policy_query_index):
        self.metrics.counters['num_full_vlm_calls']+=1
        z=torch.ones(1,2,3)
        a,s=self._decode(z,batch['proprio'],policy_query_index=policy_query_index)
        return z,a,s

    def _v0_update(self,batch,*,age,policy_query_index):
        self.metrics.counters['num_condition_updater_calls']+=1
        z=torch.ones(1,2,3)*.5
        a,s=self._decode(z,batch['proprio'],policy_query_index=policy_query_index)
        return z,a,s


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('continuation', ('isolate','continue'))
def test_counters_and_action_queue(arm,continuation):
    p=attach_recorder(Policy())
    p._intervention=dict(branch=1,arm=arm,continuation=continuation)
    batch=dict(input_ids=None,image_input=None,image_mask=None,proprio=None)
    for t in range(9):
        p._refill_action_queue(batch)
        assert len(p.action_queue)==5
        assert p._pending_record['query']==t
        if t==1 and arm.startswith('original_'):
            assert p._pending_record['condition'].mean()==2
            assert p._pending_record['predicted'].mean()==.5
    assert p.metrics.counters['num_policy_queries']==9
    p.reset()
    assert not p.metrics.counters and p._pending_record is None


def test_short_capture_cannot_silently_supply_missing_branch():
    assert select_queries([dict(query=0)],8)==[]
