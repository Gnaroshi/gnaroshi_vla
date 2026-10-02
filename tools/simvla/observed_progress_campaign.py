"""Matched progress/residual efficacy on sd1; uses the existing four-GPU queue."""
import argparse
import fcntl
from pathlib import Path

from tools.simvla.error_compensation_common import ROOT, read_json, write_json, sha
from tools.simvla.error_compensation_campaign import prepare, campaign
from tools.simvla.error_compensation_eval import run
from tools.simvla.trend_condition_eval import make_policy, check_counts

ARMS = ('frozen_trend_residual', 'progress_only', 'progress_residual', 'progress_spatial')


def jobs(c, config, smoke):
    extra = ['--smoke'] if smoke else []
    plan = []
    for arm in ARMS:
        plan.append(dict(id='train_'+arm, deps=[], cmd=[c['python'],'-m',
            'architectures.simvla.adapters.latentloop.efficient_multirate.trend_condition_train',
            '--config',str(config),'--arm',arm]+extra,
            summary=str(Path(c['output'])/('smoke' if smoke else 'train')/arm/'summary.json')))
    for k in ([8] if smoke else c['evaluation_condition_intervals']):
        for arm in ARMS:
            deps = ['train_'+arm]
            if not smoke and k == 4:
                deps.append('eval_kc8_'+arm)
            plan.append(dict(id=f'eval_kc{k}_{arm}', deps=deps, cmd=[c['python'],'-m',
                'tools.simvla.observed_progress_campaign','--config',str(config),'--eval-row',arm,
                '--k-c',str(k)]+extra,
                summary=str(Path(c['output'])/('eval_smoke' if smoke else 'online')/f'kc{k}_{arm}'/'summary.json')))
    if not smoke:
        plan.append(dict(id='offline_oracle',deps=[],cmd=[c['python'],'-m',
            'tools.simvla.progress_oracle_diagnostic','--config',str(config),
            '--output',str(Path(c['output'])/'oracle')],
            summary=str(Path(c['output'])/'oracle/summary.json'),
            expected_verdict='OFFLINE_ORACLE_COMPLETE'))
    return plan


def summarize(c):
    out=Path(c['output'])
    rows={}
    for k in c['evaluation_condition_intervals']:
        for arm in ARMS:
            f=out/'online'/f'kc{k}_{arm}'/'summary.json'
            if f.exists(): rows[f'kc{k}_{arm}']=read_json(f)
    train={a:read_json(out/'train'/a/'summary.json') for a in ARMS if (out/'train'/a/'summary.json').exists()}
    write_json(out/'comparison_summary.json',dict(complete=len(rows)==8,rows=rows,training=train,
        scientific_stopping_gate=None,training_k_c=8,evaluation_k_c=[8,4],
        interpretation='Matched 6K control separates training budget, observation-dependent progress, orthogonal correction and spatial input. K4 is transfer of K8-trained models. RTX3090 eager times are not RTX5090 paper times.'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config')
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--eval-row',choices=ARMS)
    p.add_argument('--k-c',type=int,choices=(4,8),default=8)
    a=p.parse_args()
    c=(read_json(a.config) if a.config else {**read_json(ROOT/'architectures/simvla/configs/trend_condition_sd1.json'),
        **read_json(ROOT/'architectures/simvla/configs/observed_progress_sd1.json')})
    out=Path(c['output'])
    if a.eval_row:
        run(c,a.eval_row,smoke=a.smoke,k_c=a.k_c,policy_factory=make_policy,counter_checker=check_counts)
        return 0
    if sha(c['frozen_trend_checkpoint']['path'])!=c['frozen_trend_checkpoint']['sha256']:
        raise RuntimeError('Frozen K8 b changed')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(c)
        config=out/'runtime_config.json'
        write_json(config,c)
        if a.prepare: return 0
        try:
            for smoke in ([True] if a.smoke else [True,False]):
                write_json(out/'pipeline_status.json',dict(phase='smoke' if smoke else 'train_and_evaluate',gpu_pool=[4,5,6,7]))
                rc=campaign(c,config,smoke,job_builder=jobs,summarizer=summarize)
                if rc:
                    write_json(out/'pipeline_status.json',dict(phase='technical_failure',returncode=rc))
                    return rc
            write_json(out/'pipeline_status.json',dict(phase='smoke_pass' if a.smoke else 'complete'))
            return 0
        except BaseException as exc:
            write_json(out/'pipeline_status.json',dict(phase='interrupted_or_error',error=str(exc)))
            raise


if __name__=='__main__':
    raise SystemExit(main())
