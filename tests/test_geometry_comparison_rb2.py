from tools.simvla import geometry_comparison_rb2 as c
from tools.simvla.short_visual_campaign import validate_jobs


def test_four_full_rows_unique_and_dependency_safe(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUTPUT',tmp_path)
    plan=c.plan(dict(identity='fixed'))
    validate_jobs(plan)
    assert len(plan)==8
    assert [j['id'] for j in plan if j['completion'].get('episodes')==500]==[
        'selected_seed1_eval','control_seed1_eval','selected_seed2_eval','control_seed2_eval']
    for j in plan:
        assert j['cmd'][3]==c.MODULE
        if j['id'].endswith('_eval'):assert j['deps']==[j['id'].removesuffix('_eval')+'_smoke']
