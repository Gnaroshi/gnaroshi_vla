"""Portable frozen70K comparison, with the existing compiled rb2 evaluation path."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from tools.simvla import failure_interventions as prior
from tools.simvla import gpu_followup_queue as queue
from tools.simvla.compile_benchmark import ROOT,read_json,write_json,sha

MODULE='tools.simvla.geometry_comparison_rb2'
OUTPUT=prior.STORAGE/'results/simvla/geometry_target_compiled/matched_k8_nfe1_v1'
BUNDLE=prior.STORAGE/'transfers/simvla_geometry_comparison'
ROWS=('selected_seed1','control_seed1','selected_seed2','control_seed2')


def export():
    from tools.simvla import geometry_target_train as selected
    from tools.simvla import local_geometry_train as control
    sources={}
    for name,provider,arm in [('selected',selected,'mean_original_target'),('control',control,'mean_pool_control')]:
        c=provider.checked();_,h=provider.model_payload(c,arm)
        summary=read_json(provider.OUTPUT/'train'/arm/'summary.json')
        BUNDLE.mkdir(parents=True,exist_ok=True)
        shutil.copy2(summary['checkpoint'],BUNDLE/(name+'.pt'))
        write_json(BUNDLE/(name+'_origin.json'),c)
        sources[name]=dict(arm=arm,identity=c['identity'],checkpoint=name+'.pt',checkpoint_sha256=h,
            origin=name+'_origin.json',origin_sha256=sha(BUNDLE/(name+'_origin.json')),
            summary=summary,completed_sd1=read_json(provider.OUTPUT/'rows'/arm/'summary.json'))
    c=selected.checked();cfg=c['model']['config']
    manifest=dict(sources=sources,manifest=c['manifest'],parent_fresh_sha256=c['model']['sha256'],
        checkpoint_revision=cfg['checkpoint_revision'],norm_sha256=sha(cfg['norm_stats']),
        metadata_checkpoint_sha256=sha(cfg['condition_checkpoint']),
        source_sha256={f:sha(ROOT/f) for f in dict.fromkeys(c['source_sha256'])},
        export_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(BUNDLE/'bundle.json',manifest)
    write_json(BUNDLE/'READY.json',dict(sha256=sha(BUNDLE/'bundle.json')))
    print('EXPORT_PASS',BUNDLE,flush=True)


def construct(native):
    from methods.latentloop.modules.short_visual_training import build_model
    from methods.latentloop.modules.visual_midpoint import VisualMidpointCondition
    from methods.latentloop.modules.attention_query_contract import AttentionQueryCondition
    parent=build_model(native,'fresh_joint',seed=7)
    teacher=VisualMidpointCondition(parent.early,'late_every')
    return AttentionQueryCondition(parent,teacher,'original_query_target')


def validate_bundle():
    if sha(BUNDLE/'bundle.json')!=read_json(BUNDLE/'READY.json')['sha256']:raise RuntimeError('Incomplete transfer')
    b=read_json(BUNDLE/'bundle.json')
    for s in b['sources'].values():
        if sha(BUNDLE/s['checkpoint'])!=s['checkpoint_sha256']:raise RuntimeError('Checkpoint bytes changed')
        if sha(BUNDLE/s['origin'])!=s['origin_sha256']:raise RuntimeError('Training provenance changed')
    for f,h in b['source_sha256'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('Training source differs: '+f)
    return b


def prepare():
    b=validate_bundle();cfg=prior.config('recurrent_k4')
    if cfg['checkpoint_revision']!=b['checkpoint_revision']:raise RuntimeError('Official checkpoint revision changed')
    if sha(cfg['norm_stats'])!=b['norm_sha256']:raise RuntimeError('Normalization changed')
    if sha(cfg['condition_checkpoint'])!=b['metadata_checkpoint_sha256']:raise RuntimeError('Runtime shape checkpoint changed')
    if cfg['model_checkpoint']['sha256']!=b['parent_fresh_sha256']:raise RuntimeError('K4 reference changed')
    paths=list(b['source_sha256'])+['tools/simvla/geometry_comparison_rb2.py','tools/simvla/nvidia_user_runtime.py',
        'tools/simvla/attention_query_eval.py','tools/simvla/short_visual_train.py',
        'methods/latentloop/modules/short_visual_training.py']
    c=dict(bundle=b,model=dict(config=cfg,sha256=b['parent_fresh_sha256']),manifest=b['manifest'],
        source_sha256={f:sha(ROOT/f) for f in dict.fromkeys(paths)},
        rows=list(ROWS),hardware='rb2 RTX5090 compiled',protocol='Long500,H10/R5,K8,NFE1,EGL,all videos; frozen checkpoints and two action-noise seeds')
    c['identity']=prior.digest(c)
    path=OUTPUT/'contract.json'
    if path.exists() and read_json(path)!=c:raise RuntimeError('Remote contract changed')
    write_json(path,c);return c


def checked():
    c=read_json(OUTPUT/'contract.json');b=validate_bundle()
    if c['identity']!=prior.digest({k:v for k,v in c.items() if k!='identity'}):raise RuntimeError('Runtime contract identity changed')
    if b!=c['bundle']:raise RuntimeError('Bundle changed')
    for f,h in c['source_sha256'].items():
        if sha(ROOT/f)!=h:raise RuntimeError('Runtime source changed: '+f)
    return c


def payload(c,row):
    import torch
    s=c['bundle']['sources'][row.split('_seed')[0]]
    value=torch.load(BUNDLE/s['checkpoint'],map_location='cpu',weights_only=False)
    if (value['identity'],value['arm'],value['step'])!=(s['identity'],s['arm'],10000):
        raise RuntimeError('Training checkpoint metadata mismatch')
    return value,s['checkpoint_sha256']


def evaluate(row,smoke=False):
    from tools.simvla import attention_query_eval as evaluation
    c=copy.deepcopy(checked());seed=c['manifest']['action_noise_seed_base']+int(row[-1])-1
    c['manifest']['action_noise_seed_base']=seed
    original=evaluation.prior.build_policy
    def builder(mode,cfg):
        policy,compiler=original(mode,cfg);policy.action_noise_seed_base=seed
        return policy,compiler
    writer=evaluation.write_json
    def write(path,value):
        if Path(path).name in ('summary.json','runtime.json') or Path(path).parent.name=='episodes':
            value=dict(value,action_noise_seed_base=seed)
        writer(path,value)
    evaluation.prior.build_policy=builder;evaluation.write_json=write
    evaluation.study=SimpleNamespace(OUTPUT=OUTPUT,checked=lambda:c,model_payload=payload,
        initial_model=lambda c,arm,native:construct(native),previous=SimpleNamespace(OUTPUT=OUTPUT/'unused_reference'))
    evaluation.evaluate(row,smoke=smoke)
    if not smoke:summarize()


def summarize():
    c=checked();rows={}
    expected={(x['task_id'],x['trial_id']) for x in c['manifest']['episodes']}
    if len(expected)!=500:raise RuntimeError('Expected Long500')
    for row in ROWS:
        path=OUTPUT/'rows'/row/'summary.json'
        if not path.exists():continue
        summary=read_json(path)
        records=[read_json(p) for p in (path.parent/'episodes').glob('*.json')]
        seed=c['manifest']['action_noise_seed_base']+int(row[-1])-1
        checksum=c['bundle']['sources'][row.split('_seed')[0]]['checkpoint_sha256']
        if (summary['identity']!=c['identity'] or summary['episodes']!=500 or len(records)!=500
                or {(r['task_id'],r['trial_id']) for r in records}!=expected
                or any(r['identity']!=c['identity'] or r['action_noise_seed_base']!=seed or r['model_sha256']!=checksum for r in records)
                or sum(r['success'] for r in records)!=summary['successes']):
            raise RuntimeError('Completed row contract mismatch: '+row)
        rows[row]=summary
    write_json(OUTPUT/'comparison_summary.json',dict(identity=c['identity'],complete=len(rows)==len(ROWS),
        rows=rows,scope='Same rb2 compiled runtime; two action-noise seeds, identical trained checkpoints. Existing other-server success/latency are separate references.'))


def parity():
    if prior.IS_RB2:raise RuntimeError('Constructor equivalence is checked on sd1')
    import torch
    from tools.simvla import geometry_target_train as source
    from tools.simvla.error_compensation_common import configure,snapshots
    from tools.simvla.visual_midpoint_features import CachedFeatures
    from tools.simvla.condition_output_split_train import unroll
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch
    c=source.checked();cfg=c['model']['config'];configure(cfg)
    device,native,_,_,_,heldout=load_runtime(cfg,snapshots(cfg))
    value,_=source.model_payload(c,'mean_original_target')
    original=source.initial_model(c,'mean_original_target',native).to(device).eval()
    portable=construct(native).to(device).eval()
    for model in (original,portable):model.load_state_dict(value['model'],strict=True);model.requires_grad_(False)
    from tools.simvla import visual_midpoint
    features=CachedFeatures(visual_midpoint.OUTPUT/'features',visual_midpoint.checked()['identity'])
    original.feature_provider=portable.feature_provider=features
    records=[]
    with torch.no_grad():
        for index in (0,1):
            seq=move_batch(collate_exact_teacher_sequences([heldout[index]]),device);features.bind(seq)
            for age in range(1,8):
                left=unroll(original,seq,age,8)[0];right=unroll(portable,seq,age,8)[0]
                diff=float((left-right).abs().max());records.append(dict(window=index,age=age,max_abs=diff))
                if diff!=0:raise RuntimeError('Portable constructor changed computation')
    write_json(BUNDLE/'constructor_parity.json',dict(verdict='CONSTRUCTOR_PARITY_PASS',records=records,
        checkpoint_sha256=sha(BUNDLE/'selected.pt')))
    print('CONSTRUCTOR_PARITY_PASS',flush=True)


def plan(c):
    result=[]
    for row in ROWS:
        for smoke in (True,False):
            name=row+('_smoke' if smoke else '_eval')
            result.append(dict(id=name,cmd=[sys.executable,'-u','-m',MODULE,'smoke' if smoke else 'eval','--row',row],
                deps=[] if smoke else [row+'_smoke'],summary=str(OUTPUT/('eval_smoke' if smoke else 'rows')/row/'summary.json'),
                completion=dict(identity=c['identity'],verdict='EVAL_SMOKE_PASS' if smoke else 'EVAL_COMPLETE',
                    **({} if smoke else dict(episodes=500)))))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('export','parity','preflight','all','smoke','eval'))
    p.add_argument('--row',choices=ROWS);a=p.parse_args()
    if a.command=='export':export();return 0
    if a.command=='parity':parity();return 0
    if not prior.IS_RB2:p.error('rb2 evaluation only')
    if a.command in ('smoke','eval'):
        if not a.row:p.error('--row required')
        evaluate(a.row,a.command=='smoke');return 0
    c=prepare()
    evidence=read_json(BUNDLE/'constructor_parity.json')
    if evidence['verdict']!='CONSTRUCTOR_PARITY_PASS' or evidence['checkpoint_sha256']!=c['bundle']['sources']['selected']['checkpoint_sha256']:
        raise RuntimeError('Constructor parity missing')
    if a.command=='preflight':print('PREFLIGHT_PASS compiled four Long500 rows');return 0
    result=queue.run_queue(OUTPUT,plan(c),gpus=(0,),predecessor=[],environment=prior.environment,cwd=ROOT,timeout=12*3600)
    summarize();return result


if __name__=='__main__':raise SystemExit(main())
