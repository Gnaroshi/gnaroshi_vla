"""Four-GPU Condition/Generation parameter-factorization experiment."""
import argparse
from pathlib import Path

from methods.latentloop.modules.action_aligned_joint import ARMS
from tools.simvla.error_compensation_common import ROOT, read_json, write_json
from tools.simvla.error_compensation_campaign import campaign, prepare

CONFIG = ROOT / "architectures/simvla/configs/action_aligned_joint_sd1.json"


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
    else: raise SystemExit(campaign(c, Path(a.config).resolve(), a.smoke,
        job_builder=jobs, summarizer=summarize))
