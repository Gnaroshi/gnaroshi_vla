"""Four matched Condition continuations: K4/K8 x learned/coarse action solver."""
import argparse
import fcntl
from pathlib import Path

from tools.simvla.error_compensation_campaign import prepare, campaign
from tools.simvla.error_compensation_common import read_json, write_json, sha
from tools.simvla.rollout_round2_pipeline import export

ROOT_RESULTS=Path('/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/trend_condition')
PRIOR=ROOT_RESULTS/'rollout_round2_seed01_v1'
OUTPUT=ROOT_RESULTS/'head_matched_seed01_v1'
VARIANTS={f'k{k}_{mode}':(k,mode) for k in (4,8) for mode in ('learned','naive3')}


def settings():
    return {v:dict(train_trend=True,driver='student',sources=['previous'],
                   training_k_c=k,generation_mode=mode) for v,(k,mode) in VARIANTS.items()}


def configuration():
    if read_json(PRIOR/'pipeline_status.json')['phase']!='complete':
        raise RuntimeError('Second-round campaign is not complete')
    c=read_json(PRIOR/'runtime_config.json')
    ckpt=PRIOR/'train/previous_joint/latest.pt'
    summary=read_json(PRIOR/'train/previous_joint/summary.json')
    expected='f010bbb00ebed8a3bfef7f0a768a6b50f34e5650abf9624c88f27d75c011823e'
    if sha(ckpt)!=expected or summary['checkpoint_sha256']!=expected:
        raise RuntimeError('Common initialization changed')
    c.update(output=str(OUTPUT),predecessor=str(PRIOR),variant_settings=settings(),
        selected_checkpoint=dict(path=str(ckpt),sha256=expected,arm='frozen_trend_residual',step=3000),
        steps=3000,smoke_steps=14,evaluation_condition_intervals=[4,8],
        rb2_destination='rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_head_matched',
        training_description='Same previous_joint initialization, b+E trainable, same 50% original cache and 50% previous student states, batch2, 3K, AdamW1e-4 cosine to0.1x. K4 ages1..3/K8 ages1..7; each mode trained through its deployed learned3 or naive3 solver. Teacher always original10. Frozen backbone/action/generation. No new collection or SR gate; seed01 development only.')
    for smoke in (False,True):
        key='record_sources_smoke' if smoke else 'record_sources'
        previous=[s for s in c[key] if s['label']=='previous']
        c[key]=previous+[dict(path=str(PRIOR/('collection_smoke' if smoke else 'collection')),
            identity=read_json(PRIOR/'contract.json')['identity'],label='current')]
    return c


def jobs(c,config,smoke):
    out=Path(c['output']); extra=['--smoke'] if smoke else []
    prefix=[c['python'],'-u','-m','tools.simvla.rollout_state_repair']
    result=[dict(id='train_'+v,deps=[],cmd=prefix+['train','--config',str(config),'--variant',v]+extra,
        summary=str(out/('smoke' if smoke else 'train')/v/'summary.json')) for v in VARIANTS]
    if not smoke:
        result.append(dict(id='export_rb2',deps=['train_'+v for v in VARIANTS],
            cmd=[c['python'],'-u','-m','tools.simvla.head_matched_pipeline','--export-config',str(config)],
            expected_verdict='BUNDLE_EXPORTED',summary=str(out/'export_summary.json')))
    for v,(k,_) in VARIANTS.items():
        result.append(dict(id=f'eval_kc{k}_{v}',deps=['train_'+v],
            cmd=prefix+['eval','--config',str(config),'--variant',v,'--k-c',str(k)]+extra,
            summary=str(out/('eval_smoke' if smoke else 'online')/f'kc{k}_{v}'/'summary.json')))
    return result


def summarize(c):
    out=Path(c['output']); rows={}
    for v,(k,_) in VARIANTS.items():
        p=out/'online'/f'kc{k}_{v}'/'summary.json'
        if p.exists(): rows[v]=read_json(p)
    write_json(out/'comparison_summary.json',dict(complete=len(rows)==4,rows=rows,
        variant_settings=settings(),development_seed_reused=True,
        question='Does the learned Generation advantage persist when both Condition models train through the solver used at evaluation?'))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true'); p.add_argument('--export-config')
    a=p.parse_args()
    if a.export_config: export(read_json(a.export_config),variants=VARIANTS); return 0
    c=configuration(); OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            prepare(c); config=OUTPUT/'runtime_config.json'; write_json(config,c)
            if a.preflight: print('PREFLIGHT_PASS: four matched head/interval trainings; no GPU allocation',flush=True); return 0
            for smoke in (True,False):
                write_json(OUTPUT/'pipeline_status.json',dict(phase='smoke' if smoke else 'train_evaluate_export'))
                if campaign(c,config,smoke,job_builder=jobs,summarizer=summarize):
                    raise RuntimeError('Technical failure; inspect per-job logs')
            write_json(OUTPUT/'pipeline_status.json',dict(phase='complete')); return 0
        except BaseException as exc:
            write_json(OUTPUT/'pipeline_status.json',dict(phase='failed',error=str(exc))); raise


if __name__=='__main__': raise SystemExit(main())
