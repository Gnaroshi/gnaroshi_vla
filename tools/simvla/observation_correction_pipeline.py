"""Four independent sd1 trainings, per-model transfer, and full K4/K8 evaluation."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

from methods.latentloop.modules.observation_correction import ARMS
from tools.simvla.head_matched_pipeline import configuration as previous_config
from tools.simvla.error_compensation_campaign import prepare, campaign
from tools.simvla.error_compensation_common import read_json, write_json, sha, identity, ROOT

OUTPUT=Path('/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/observation_correction/mixed_k4_k8_seed01_v1')
DEST='rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_observation_correction'


def configuration():
    c=previous_config()
    c.update(output=str(OUTPUT), variant_settings={}, steps=3000, smoke_steps=14,
        condition_weight=0.05, measurement_weight=0.05, rb2_destination=DEST,
        training_description='Four causal Condition architectures. Same source, two full-cache sequences per optimizer step, mixed K4/K8 ages, 3K AdamW1e-4 cosine, frozen original10 teacher, train/deploy naive3. One self-unroll and one same-sequence frozen-source previous-state intervention. Condition MSE weight0.05 for every arm; independent measurement auxiliary0.05 only for observed arm. Seed01 development; 500 episodes/row; no SR stopping gate.',
        student_condition_description='Recurrent current-observation updates, optional refresh trend, midpoint slope re-estimation, or independent current-observation correction.')
    return c


def export_arm(c, arm):
    out=Path(c['output']); summary=read_json(out/'train'/arm/'summary.json')
    if summary['identity']!=identity(c) or summary['verdict']!='TRAIN_AND_OFFLINE_COMPLETE':
        raise RuntimeError('Unfinished training export')
    ckpt=Path(summary['checkpoint'])
    if sha(ckpt)!=summary['checkpoint_sha256']: raise RuntimeError('Checkpoint hash changed')
    staging=out/'rb2_bundle'/arm; staging.mkdir(parents=True,exist_ok=True)
    target=staging/'model.pt'
    if not target.exists(): os.link(ckpt,target)
    if sha(target)!=summary['checkpoint_sha256']: raise RuntimeError('Staged checkpoint changed')
    manifest=dict(arm=arm,source_identity=identity(c),checkpoint_sha256=summary['checkpoint_sha256'],
        step=c['steps'],training_seconds=summary['training_seconds'],parameters=summary['parameters'],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(staging/'manifest.json',manifest)
    write_json(staging/'READY.json',dict(manifest_sha256=sha(staging/'manifest.json')))
    host,remote=c['rb2_destination'].split(':',1)
    for attempt in range(3):
        try:
            subprocess.run(['ssh',host,'mkdir','-p',remote+'/'+arm],check=True)
            subprocess.run(['rsync','-a','--partial','--exclude=READY.json',str(staging)+'/',host+':'+remote+'/'+arm+'/'],check=True)
            subprocess.run(['rsync','-a',str(staging/'READY.json'),host+':'+remote+'/'+arm+'/'],check=True)
            write_json(out/'exports'/f'{arm}.json',dict(identity=identity(c),verdict='BUNDLE_EXPORTED',**manifest))
            return
        except subprocess.CalledProcessError:
            if attempt==2: raise
            time.sleep(20)


def jobs(c,config,smoke):
    out=Path(c['output']); extra=['--smoke'] if smoke else []
    prefix=[c['python'],'-u','-m']
    jobs=[dict(id='train_'+arm,deps=[],cmd=prefix+['tools.simvla.observation_correction_train',
        '--config',str(config),'--arm',arm]+extra,
        summary=str(out/('smoke' if smoke else 'train')/arm/'summary.json')) for arm in ARMS]
    if not smoke:
        for arm in ARMS:
            jobs.append(dict(id='export_'+arm,deps=['train_'+arm],cmd=prefix+[
                'tools.simvla.observation_correction_pipeline','--export',arm,'--config',str(config)],
                summary=str(out/'exports'/f'{arm}.json'),expected_verdict='BUNDLE_EXPORTED'))
    for k in (8,4):
        for arm in ARMS:
            jobs.append(dict(id=f'eval_kc{k}_{arm}',deps=['train_'+arm],cmd=prefix+[
                'tools.simvla.observation_correction_eval','--config',str(config),'--arm',arm,'--k-c',str(k)]+extra,
                summary=str(out/('eval_smoke' if smoke else 'online')/f'kc{k}_{arm}'/'summary.json')))
    return jobs


def summarize(c):
    out=Path(c['output']); rows={}
    for k in (4,8):
        for arm in ARMS:
            p=out/'online'/f'kc{k}_{arm}'/'summary.json'
            if p.exists(): rows[f'kc{k}_{arm}']=read_json(p)
    write_json(out/'comparison_summary.json',dict(complete=len(rows)==8,rows=rows,
        scope='Development seed01, 500 episodes/row; same action solver; eager sd1 latency reported separately from compiled rb2.',
        training_seconds={a:read_json(out/'train'/a/'summary.json')['training_seconds']
            for a in ARMS if (out/'train'/a/'summary.json').exists()}))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true')
    p.add_argument('--smoke-only',action='store_true'); p.add_argument('--export',choices=ARMS); p.add_argument('--config')
    a=p.parse_args()
    if a.export: export_arm(read_json(a.config),a.export); return 0
    c=configuration(); OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            prepare(c); config=OUTPUT/'runtime_config.json'; write_json(config,c)
            if a.preflight: print('PREFLIGHT_PASS: GPU4..7, 4 models, K4/K8, 500 episodes each',flush=True); return 0
            for smoke in ([True] if a.smoke_only else [True,False]):
                write_json(OUTPUT/'pipeline_status.json',dict(phase='smoke' if smoke else 'train_evaluate_export'))
                if campaign(c,config,smoke,job_builder=jobs,summarizer=summarize):
                    raise RuntimeError('Independent jobs finished; technical failures listed in status.json')
            write_json(OUTPUT/'pipeline_status.json',dict(phase='smoke_complete' if a.smoke_only else 'complete'))
            return 0
        except BaseException as exc:
            write_json(OUTPUT/'pipeline_status.json',dict(phase='failed',error=str(exc))); raise


if __name__=='__main__': raise SystemExit(main())
