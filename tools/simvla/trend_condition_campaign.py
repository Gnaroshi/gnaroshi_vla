"""Four matched pilots followed by paired 500-episode evaluations."""
import argparse
import csv
import fcntl
from pathlib import Path

from methods.latentloop.modules.trend_condition import ARMS
from tools.simvla.error_compensation_common import ROOT, read_json, write_json
from tools.simvla.error_compensation_campaign import campaign, prepare
from tools.simvla.action_aligned_campaign import process_matches
import time

CONFIG=ROOT/'architectures/simvla/configs/trend_condition_sd1.json'


def jobs(c,config,smoke):
    extra=['--smoke'] if smoke else []
    plan=[]
    for arm in ARMS:
        plan.append(dict(id='train_'+arm,deps=[],cmd=[c['python'],'-m',
            'architectures.simvla.adapters.latentloop.efficient_multirate.trend_condition_train',
            '--config',str(config),'--arm',arm]+extra,
            summary=str(Path(c['output'])/('smoke' if smoke else 'train')/arm/'summary.json')))
    for arm in ARMS:
        plan.append(dict(id='eval_kc4_'+arm,deps=['train_'+arm],cmd=[c['python'],'-m',
            'tools.simvla.trend_condition_eval','--config',str(config),'--row',arm,'--k-c','4']+extra,
            summary=str(Path(c['output'])/('eval_smoke' if smoke else 'online')/('kc4_'+arm)/'summary.json')))
    return plan


def summarize(c):
    out=Path(c['output'])
    rows={a:read_json(out/'online'/('kc4_'+a)/'summary.json') for a in ARMS
        if (out/'online'/('kc4_'+a)/'summary.json').exists()}
    references={}
    for label,path in c.get('reference_results',{}).items():
        if Path(path).is_file(): references[label]=dict(source=path,**read_json(path))
    write_json(out/'comparison_summary.json',dict(complete=len(rows)==len(ARMS),rows=rows,
        unavailable=[a for a in ARMS if a not in rows],references=references,
        timing='sd1 RTX3090 eager CUDA-synchronized policy.act / executed actions; concurrent CPU/render load, not rb2 paper timing',
        note='All new arms 750 decomposition +2250 executed-action steps. Historical controls have different training budgets; report separately. No SR stopping gate.'))
    with (out/'sr_latency.csv').open('w',newline='') as stream:
        fields=['method','episodes','successes','success_rate','policy_ms_per_action','gpu','compile','paper_latency']
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for name,row in {**references,**rows}.items():
            writer.writerow(dict(method=name,**{k:row.get(k) for k in fields[1:]}))


def run_all(c,config):
    out=Path(c['output'])
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=out/'pipeline_status.json'
    try:
        prepare(c)
        while True:
            waiting=[r for r in c.get('wait_for_processes',[]) if process_matches(r)]
            if not waiting: break
            write_json(status,dict(phase='WAITING_FOR_EXISTING_SIMVLA_CAMPAIGN',waiting=waiting,
                gpu_pool=[4,5,6,7],gpu_validation='PENDING',gpu_jobs_started=False))
            print('WAIT: preceding SimVLA campaign owns GPU4-7; checking again in 60s.',flush=True)
            time.sleep(60)
        for smoke,phase in [(True,'GPU_SMOKE'),(False,'TRAIN_AND_EVALUATE')]:
            write_json(status,dict(phase=phase,gpu_pool=[4,5,6,7]))
            rc=campaign(c,config,smoke,job_builder=jobs,summarizer=summarize)
            if rc:
                write_json(status,dict(phase=phase,verdict='TECHNICAL_FAILURE',returncode=rc))
                return rc
        write_json(status,dict(phase='COMPLETE',verdict='COMPLETE'))
        return 0
    except KeyboardInterrupt:
        write_json(status,dict(phase='INTERRUPTED'))
        return 130
    except Exception as error:
        write_json(status,dict(phase='TECHNICAL_ERROR',error=f'{type(error).__name__}: {error}'))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(CONFIG))
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    c=read_json(a.config)
    if a.prepare: prepare(c)
    elif a.smoke: raise SystemExit(campaign(c,Path(a.config).resolve(),True,job_builder=jobs,summarizer=summarize))
    else: raise SystemExit(run_all(c,Path(a.config).resolve()))
