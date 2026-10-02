from types import SimpleNamespace

import pytest
import torch

from tools.simvla.trend_followup_eval import ROWS, condition_at_age, expected_counts, check_counts
from tools.simvla.trend_followup_campaign import jobs


@pytest.mark.parametrize('k_c',(4,6,8))
@pytest.mark.parametrize('hold',(False,True))
def test_linear_prediction_is_absolute_causal_and_age_bounded(k_c,hold):
    anchor=torch.randn(2,5,12)
    trend=torch.randn_like(anchor)
    context=SimpleNamespace(anchor=anchor.clone(),trend=trend.clone())
    for age in range(1,k_c):
        predicted,residual=condition_at_age(context,age,k_c,hold)
        torch.testing.assert_close(predicted,anchor if hold else anchor+age*trend)
        assert torch.count_nonzero(residual)==0
    torch.testing.assert_close(context.anchor,anchor,atol=0,rtol=0)
    for invalid in (0,k_c):
        with pytest.raises(ValueError): condition_at_age(context,invalid,k_c,hold)


@pytest.mark.parametrize('row',ROWS)
def test_counts_cover_refresh_boundary(row):
    k=ROWS[row]['k_c']
    for q in (1,k-1,k,k+1,2*k+1):
        counts=expected_counts(row,q)
        assert counts['full_vlm']==(q+k-1)//k
        assert counts['transformer']==3*q
        policy=SimpleNamespace(step_index=q*5,_trend_counts={
            'trend':counts['trend'],'observation':0},metrics=SimpleNamespace(counters={
                'num_policy_queries':q,'num_full_vlm_calls':counts['full_vlm'],
                'num_condition_updater_calls':counts['lightweight_conditions']}))
        assert check_counts(policy,row,counts,k)==counts
        with pytest.raises(RuntimeError): check_counts(policy,row,dict(counts,transformer=0),k)


def test_eval_only_queue_and_smoke_cover_all_ages():
    c=dict(python='/python',output='/result')
    for smoke in (False,True):
        plan=jobs(c,'/config',smoke)
        assert len(plan)==5
        assert len({j['id'] for j in plan})==5
        assert all(j['id'].startswith('eval_') and not j['deps'] for j in plan)
        assert all(('--smoke' in j['cmd'])==smoke for j in plan)
    assert 41//5>=8
