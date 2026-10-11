from collections import Counter

import pytest

from tools.simvla import failure_interventions as prior
from tools.simvla.failure_continuations import execution_plan,window_end


@pytest.mark.parametrize('branch',(1,7,9,15,25))
@pytest.mark.parametrize('axis',('condition','solver'))
@pytest.mark.parametrize('phase',('suffix','window'))
def test_prefix_unchanged_and_two_axes_separate(branch,axis,phase):
    k=8
    for t in range(branch+1):
        assert execution_plan(t,k,branch,'predicted_1',f'{phase}_{axis}')==dict(full=t%k==0,nfe=1,replace_condition=False)
    changed=execution_plan(branch+1,k,branch,'predicted_1',f'{phase}_{axis}')
    assert changed['full']==(axis=='condition' or (branch+1)%k==0)
    assert changed['nfe']==(10 if axis=='solver' else 1)


@pytest.mark.parametrize('branch',(1,7,9,15,25))
def test_window_resume_has_consecutive_valid_ages(branch):
    k=8
    end=window_end(branch,k)
    assert end%k==0 and k<=end-branch-1<=2*k-2
    previous_age=0
    for t in range(end+2*k):
        plan=execution_plan(t,k,branch,'predicted_1','window_condition')
        if plan['full']:
            previous_age=0
        else:
            age=t%k
            assert age==previous_age+1
            previous_age=age
        if t>=end:
            assert plan==dict(full=t%k==0,nfe=1,replace_condition=False)


def test_existing_controls_remain_exact():
    for continuation in prior.STAGES:
        for arm in prior.ARMS:
            for t in range(20):
                assert execution_plan(t,8,3,arm,continuation)==prior.execution_plan(t,8,3,arm,continuation)


def test_suffix_usage_separates_condition_and_solver_cost():
    calls={}
    for axis in ('condition','solver'):
        plans=[execution_plan(t,8,7,'predicted_1','suffix_'+axis) for t in range(24)]
        calls[axis]=Counter(backbone=sum(p['full'] for p in plans),nfe=sum(p['nfe'] for p in plans))
    assert calls['condition']==dict(backbone=17,nfe=24)
    assert calls['solver']==dict(backbone=3,nfe=168)
