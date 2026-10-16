import pytest


def test_two_studies_and_selected_confirmation(monkeypatch,tmp_path):
    from tools.simvla import short_visual_campaign as c
    monkeypatch.setattr(c.training,'OUTPUT',tmp_path)
    contract=dict(identity='fixed',manifest=dict(action_noise_seed_base=10))
    first=c.study_jobs(contract,'fresh_joint')
    second=c.study_jobs(contract,'protected_short')
    assert len(first)==15 and len(second)==9
    c.validate_jobs(first+second)
    assert sum(j['completion'].get('episodes',0) for j in first if j['id'].endswith(('_0','_1','_2','_3')))==1000
    assert sum(j['completion'].get('episodes',0) for j in second if j['id'].endswith(('_0','_1','_2','_3')))==500
    assert not any('k4' in j['id'] for j in second)
    for arm,plan in [('fresh_joint',first),('protected_short',second)]:
        assert next(j for j in plan if j['id']=='train_'+arm)['deps']==['runtime_'+arm]
    confirmation=c.evaluation_jobs(contract,'fresh_joint',8,2)
    c.validate_jobs(confirmation)
    assert len(confirmation)==6
    for job in confirmation:
        if job['id'].endswith(('_0','_1','_2','_3')):
            assert job['completion']['action_noise_seed_base']==11


def test_selection_only_complete_k8():
    from tools.simvla.short_visual_campaign import select_short
    def row(n,ms,k=8):
        return dict(verdict='EVAL_COMPLETE',episodes=500,interval=k,successes=n,policy_ms_per_action=ms)
    assert select_short(dict(a=row(400,10),b=row(401,15),c=row(500,1,4)))=='b'
    assert select_short(dict(a=row(401,10),b=row(401,15)))=='a'
    with pytest.raises(RuntimeError):select_short({})


def test_duplicate_or_missing_dependencies_rejected():
    from tools.simvla.short_visual_campaign import validate_jobs
    with pytest.raises(RuntimeError):
        validate_jobs([dict(id='a',summary='a',deps=['missing'])])
    with pytest.raises(RuntimeError):
        validate_jobs([dict(id='a',summary='same',deps=[]),dict(id='b',summary='same',deps=[])])


def test_reference_seed_override_preserves_original_checkpoint_contract(tmp_path):
    from types import SimpleNamespace
    from tools.simvla.short_visual_reference_repair import runtime_seed_builder
    from tools.simvla.compile_benchmark import write_json
    original=dict(output=str(tmp_path),action_noise_seed_base=10,other='fixed')
    write_json(tmp_path/'contract.json',dict(config=original))
    def factory(mode,config):
        assert config==original
        return SimpleNamespace(action_noise_seed_base=10),None
    build=runtime_seed_builder(factory)
    policy,_=build('recurrent_k4',dict(original,action_noise_seed_base=11))
    assert policy.action_noise_seed_base==11
    assert original['action_noise_seed_base']==10
    with pytest.raises(RuntimeError):
        build('recurrent_k4',dict(original,other='changed'))
