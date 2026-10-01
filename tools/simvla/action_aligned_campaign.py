"""Four-GPU Condition/Generation parameter-factorization experiment."""
import argparse
from pathlib import Path
import time

from methods.latentloop.modules.action_aligned_joint import ARMS
from tools.simvla.error_compensation_common import ROOT, read_json, write_json
from tools.simvla.error_compensation_campaign import campaign, prepare

CONFIG = ROOT / "architectures/simvla/configs/action_aligned_joint_sd1.json"


def process_matches(record, proc_root=Path('/proc')):
    try:
        fields = (proc_root / str(record['pid']) / 'stat').read_text().rsplit(')',1)[1].split()
        return fields[0] != 'Z' and fields[19] == str(record['start_ticks'])
    except FileNotFoundError:
        return False


def run_all(c, config):
    out = Path(c['output'])
    status = out / 'pipeline_status.json'
    try:
        while True:
            waiting = [r for r in c.get('wait_for_processes',[]) if process_matches(r)]
            if not waiting: break
            write_json(status, dict(phase='WAITING_FOR_EXISTING_GPU_JOB', waiting=waiting,
                gpu_pool=[4,5,6,7], gpu_validation='PENDING', gpu_jobs_started=False))
            print('WAIT: existing GPU4-7 campaign is still running; no SimVLA GPU use. Checking again in 60 seconds.', flush=True)
            time.sleep(60)
        for smoke, phase in [(True,'GPU_SMOKE'), (False,'TRAIN_AND_EVALUATE')]:
            write_json(status, dict(phase=phase, gpu_pool=[4,5,6,7]))
            rc = campaign(c, config, smoke, job_builder=jobs, summarizer=summarize)
            if rc:
                write_json(status, dict(phase=phase, verdict='FAILED', returncode=rc))
                return rc
        write_json(status, dict(phase='COMPLETE', verdict='COMPLETE'))
        return 0
    except KeyboardInterrupt:
        write_json(status, dict(phase='INTERRUPTED', verdict='INTERRUPTED'))
        return 130
    except Exception as error:
        write_json(status, dict(phase='TECHNICAL_ERROR', error=f'{type(error).__name__}: {error}'))
        raise


def jobs(c, config, smoke):
    prefix = [c["python"], "-m"]
    extra = ["--smoke"] if smoke else []
    result = []
    for arm in ARMS:
        result.append(dict(id="train_"+arm, deps=[], cmd=prefix+[
            "architectures.simvla.adapters.latentloop.efficient_multirate.action_aligned_train",
            "--config", str(config), "--arm", arm]+extra,
            summary=str(Path(c["output"]) / ("smoke" if smoke else "train") / arm / "summary.json")))
    for k in c["evaluation_condition_intervals"]:
        for arm in ARMS:
            result.append(dict(id=f"eval_kc{k}_{arm}", deps=["train_"+arm], cmd=prefix+[
                "tools.simvla.action_aligned_eval", "--config", str(config), "--row", arm,
                "--k-c", str(k)]+extra, summary=str(Path(c["output"]) /
                ("eval_smoke" if smoke else "online") / f"kc{k}_{arm}" / "summary.json")))
    return result


def summarize(c):
    out = Path(c["output"])
    keys = [f"kc{k}_{a}" for k in c["evaluation_condition_intervals"] for a in ARMS]
    reports = {r: read_json(out / "online" / r / "summary.json") for r in keys
        if (out / "online" / r / "summary.json").exists()}
    write_json(out / "comparison_summary.json", dict(complete=len(reports)==len(keys),
        rows=reports, unavailable=[r for r in keys if r not in reports],
        interpretation="Does final executed-action supervision help Condition, Generation, or their joint adaptation? Architecture and inference compute unchanged; no superiority assumed."))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--prepare", action="store_true")
    a = p.parse_args()
    c = read_json(a.config)
    if a.prepare: prepare(c)
    elif a.smoke: raise SystemExit(campaign(c, Path(a.config).resolve(), True,
        job_builder=jobs, summarizer=summarize))
    else: raise SystemExit(run_all(c, Path(a.config).resolve()))
