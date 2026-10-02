"""Evaluation-only queue: one frozen checkpoint, five nonduplicate contrasts."""
import argparse
import fcntl
from pathlib import Path
import time

from tools.simvla.error_compensation_common import ROOT, read_json, write_json
from tools.simvla.error_compensation_campaign import campaign, prepare
from tools.simvla.action_aligned_campaign import process_matches
from tools.simvla.trend_followup_eval import ROWS, load_checkpoint

CONFIG=ROOT/'architectures/simvla/configs/trend_followup_sd1.json'


def jobs(c, config, smoke):
    plan=[]
    for row,spec in ROWS.items():
        key=f"kc{spec['k_c']}_{row}"
        plan.append(dict(id='eval_'+key,deps=[],cmd=[c['python'],'-m',
            'tools.simvla.trend_followup_eval','--config',str(config),'--row',row]
            + (['--smoke'] if smoke else []),
            summary=str(Path(c['output'])/('eval_smoke' if smoke else 'online')/key/'summary.json')))
    return plan


def summarize(c):
    out=Path(c['output'])
    rows={}
    for row,spec in ROWS.items():
        key=f"kc{spec['k_c']}_{row}"
        p=out/'online'/key/'summary.json'
        if p.is_file():
            rows[key]=dict(**read_json(p),condition_mode='hold' if spec['hold'] else 'trend_only',
                generation_mode=spec['generation'],extrapolation=spec['k_c']>4 and not spec['hold'])
    references={name:dict(source=path,**read_json(path)) for name,path in c['reference_results'].items()}
    write_json(out/'comparison_summary.json',dict(complete=len(rows)==len(ROWS),rows=rows,
        references=references,training='Frozen existing 3K checkpoint; no additional training',
        timing='sd1 RTX3090 eager, shared CPU load; policy wall ms / executed actions; not paper latency'))


def run_all(c,config):
    out=Path(c['output'])
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=out/'pipeline_status.json'
    try:
        load_checkpoint(c,'trend_only')
        prepare(c)
        while True:
            waiting=[r for r in c.get('wait_for_processes',[]) if process_matches(r)]
            if not waiting:
                break
            write_json(status,dict(phase='WAITING_FOR_EXISTING_GPU_CAMPAIGN',waiting=waiting,
                gpu_pool=[4,5,6,7],gpu_jobs_started=False,gpu_validation='PENDING'))
            print('WAIT: existing Seer campaign owns GPU4-7; no SimVLA GPU use. Checking in 60s.',flush=True)
            time.sleep(60)
        for smoke,phase in [(True,'GPU_SMOKE'),(False,'EVALUATE_500_EACH')]:
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
    except Exception as exc:
        write_json(status,dict(phase='TECHNICAL_ERROR',error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',default=str(CONFIG))
    a=p.parse_args()
    raise SystemExit(run_all(read_json(a.config),Path(a.config).resolve()))
