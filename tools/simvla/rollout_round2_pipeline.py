"""Separate visited-state coverage from a fixed anchor-trend constraint."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

from tools.simvla.error_compensation_common import ROOT,read_json,write_json,sha,identity
from tools.simvla.error_compensation_campaign import prepare,campaign

PRIOR=Path('/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/trend_condition/rollout_state_repair_seed01_v1')
VARIANTS=('previous_fixed','previous_joint','aggregate_fixed','aggregate_joint')


def settings():
    return {v:dict(train_trend=v.endswith('joint'),driver='student',
        sources=['previous','current'] if v.startswith('aggregate') else ['previous']) for v in VARIANTS}


def configuration():
    c=read_json(PRIOR/'runtime_config.json')
    completed=read_json(PRIOR/'campaign_complete.json')
    if completed['verdict']!='COMPLETE' or completed['failed']: raise RuntimeError('First repair campaign incomplete')
    ckpt=PRIOR/'train/rollout_repair/latest.pt'
    summary=read_json(PRIOR/'train/rollout_repair/summary.json')
    if sha(ckpt)!=summary['checkpoint_sha256']: raise RuntimeError('First-round checkpoint changed')
    c.update(output=str(PRIOR.parent/'rollout_round2_seed01_v1'),predecessor=str(PRIOR),
        selected_checkpoint=dict(path=str(ckpt),sha256=sha(ckpt),arm='frozen_trend_residual',step=3000),
        variant_settings=settings(),steps=3000,smoke_steps=14,
        rb2_destination='rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_rollout_round2',
        training_description='Matched 2x2: previous vs 50/50 previous+current student states; frozen vs trained b. Same 3K initialization, 50% old original cache, batch2, action L1, cosine LR. Original/Generation frozen. Development seed01; no SR stopping gate.')
    source_identity=read_json(PRIOR/'contract.json')['identity']
    for smoke in (False,True):
        c['record_sources_smoke' if smoke else 'record_sources']=[dict(
            path=str(PRIOR/('collection_smoke' if smoke else 'collection')),identity=source_identity,label='previous')]
    return c


def jobs(c,config,smoke):
    out=Path(c['output']); extra=['--smoke'] if smoke else []
    prefix=[c['python'],'-u','-m','tools.simvla.rollout_state_repair']
    result=[]
    for shard in ([0] if smoke else range(4)):
        result.append(dict(id=f'collect_student_{shard}',deps=[],cmd=prefix+['collect','--config',str(config),'--shard',str(shard)]+extra,
            expected_verdict='COLLECTION_COMPLETE',summary=str(out/('collection_smoke' if smoke else 'collection')/'student'/f'shard{shard}_summary.json')))
    deps=[j['id'] for j in result]
    for v in VARIANTS:
        result.append(dict(id='train_'+v,deps=deps,cmd=prefix+['train','--config',str(config),'--variant',v]+extra,
            summary=str(out/('smoke' if smoke else 'train')/v/'summary.json')))
    if not smoke:
        result.append(dict(id='export_rb2',deps=['train_'+v for v in VARIANTS],
            cmd=[c['python'],'-u','-m','tools.simvla.rollout_round2_pipeline','--export-config',str(config)],
            expected_verdict='BUNDLE_EXPORTED',summary=str(out/'export_summary.json')))
    for k in ([8] if smoke else (8,4)):
        for v in VARIANTS:
            result.append(dict(id=f'eval_kc{k}_{v}',deps=['train_'+v],
                cmd=prefix+['eval','--config',str(config),'--variant',v,'--k-c',str(k)]+extra,
                summary=str(out/('eval_smoke' if smoke else 'online')/f'kc{k}_{v}'/'summary.json')))
    return result


def summarize(c):
    out=Path(c['output']); rows={}
    for k in (8,4):
        for v in VARIANTS:
            path=out/'online'/f'kc{k}_{v}'/'summary.json'
            if path.exists(): rows[f'kc{k}_{v}']=read_json(path)
    write_json(out/'comparison_summary.json',dict(complete=len(rows)==8,rows=rows,
        variant_settings=settings(),development_seed_reused=True,added_inference_operations=0))


def export(c,variants=VARIANTS):
    out=Path(c['output']); bundle=out/'rb2_bundle'; bundle.mkdir(exist_ok=True)
    manifest=dict(selected_arm='frozen_trend_residual',checkpoints={},source_identity=identity(c),variant_settings=c.get('variant_settings',settings()))
    for v in variants:
        source=out/'train'/v/'latest.pt'; target=bundle/(v+'.pt')
        if target.exists():
            if sha(target)!=sha(source): raise RuntimeError('Export changed')
        else: os.link(source,target)
        manifest['checkpoints'][v]=dict(file=target.name,sha256=sha(target),step=3000)
    write_json(bundle/'manifest.json',manifest)
    write_json(bundle/'READY.json',dict(manifest_sha256=sha(bundle/'manifest.json')))
    destination=c['rb2_destination']; host,path=destination.split(':',1)
    for attempt in range(3):
        try:
            subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'mkdir','-p',path],check=True,timeout=60)
            remote=['-e','ssh -o BatchMode=yes -o ConnectTimeout=10']
            subprocess.run(['rsync','-a','--partial',*remote,'--exclude=READY.json',str(bundle)+'/',destination+'/'],check=True,timeout=600)
            subprocess.run(['rsync','-a',*remote,str(bundle/'READY.json'),destination+'/'],check=True,timeout=60)
            write_json(out/'export_summary.json',dict(identity=identity(c),verdict='BUNDLE_EXPORTED',manifest=manifest)); return
        except (OSError,subprocess.SubprocessError):
            if attempt==2: raise
            time.sleep(30)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--export-config'); p.add_argument('--preflight',action='store_true'); a=p.parse_args()
    if a.export_config: export(read_json(a.export_config)); return 0
    c=configuration(); out=Path(c['output']); out.mkdir(parents=True,exist_ok=True)
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            prepare(c); path=out/'runtime_config.json'; write_json(path,c)
            if a.preflight: print('PREFLIGHT_PASS: predecessor, fixed collection split, source hashes',flush=True); return 0
            for smoke in (True,False):
                write_json(out/'pipeline_status.json',dict(phase='smoke' if smoke else 'collect_train_evaluate'))
                if campaign(c,path,smoke,job_builder=jobs,summarizer=summarize): raise RuntimeError('Technical failure; inspect status.json and logs')
            write_json(out/'pipeline_status.json',dict(phase='complete')); return 0
        except BaseException as exc:
            write_json(out/'pipeline_status.json',dict(phase='failed',error=str(exc))); raise


if __name__=='__main__': raise SystemExit(main())
