"""Four-GPU dependency queue. Resume training/episodes and report real failures."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from tools.simvla.error_compensation_common import (
    ROOT, CONFIG, ARMS, ROWS, configure, digest, environment, read_json, sha, snapshots, write_json,
)


def prepare(c):
    if socket.gethostname() != "jbrserver1":
        raise RuntimeError("This efficacy campaign is restricted to sd1")
    configure(c)
    from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_contracts import FROZEN_CONDITION_CHECKPOINT_SHA256
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
        FROZEN_GENERATION_CHECKPOINT_SHA256, FROZEN_NORM_STATS_SHA256, FROZEN_EXACT_CACHE_MANIFEST_SHA256,
    )
    import yaml
    import torch
    import mujoco
    import transformers
    output = Path(c["output"])
    expected = {"condition_checkpoint": FROZEN_CONDITION_CHECKPOINT_SHA256,
        "generation_checkpoint": FROZEN_GENERATION_CHECKPOINT_SHA256, "norm_stats": FROZEN_NORM_STATS_SHA256,
        "cache_manifest": FROZEN_EXACT_CACHE_MANIFEST_SHA256}
    inputs = {**{k: c[k] for k in expected if k != "cache_manifest"},
        "cache_manifest": str(Path(c["cache"]) / "manifest.json")}
    hashes = {k: sha(p) for k, p in inputs.items()}
    if hashes != expected:
        raise RuntimeError(f"Input hash mismatch: {hashes}")
    assets = snapshots(c)
    if any(not Path(p).is_dir() for p in assets.values()):
        raise FileNotFoundError(assets)
    libero = Path(c["upstream"]) / "evaluation/libero/LIBERO/libero/libero"
    settings = {"benchmark_root": str(libero), "assets": str(libero / "assets"),
        "bddl_files": str(libero / "bddl_files"), "init_states": str(libero / "init_files"),
        "datasets": str(libero / "datasets")}
    for key in ("assets", "bddl_files", "init_states"):
        if not Path(settings[key]).is_dir(): raise FileNotFoundError(settings[key])
    config_dir = output / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump(settings))
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    files = [ROOT / p for p in tracked if p.endswith(".py") and p.startswith(("tools/simvla/", "methods/latentloop/", "architectures/simvla/adapters/", "architectures/simvla/wrappers/"))]
    files += list(ROOT.glob("tools/simvla/error_compensation*.py"))
    files += [ROOT / "architectures/simvla/adapters/latentloop/efficient_multirate/error_compensation_train.py"]
    contract = {"config": c, "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
        "artifacts": hashes, "snapshots": assets,
        "upstream_sources": {str(p): sha(p) for p in (Path(c["upstream"]) / "models").glob("*.py")},
        "libero_config": settings, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {"torch": torch.__version__, "mujoco": mujoco.__version__, "transformers": transformers.__version__,
            "python": sys.version, "hostname": socket.gethostname()},
        "protocol": {"suite": "libero_10", "trials_per_task": 50, "tasks": 10, "action_horizon": 10,
            "execution_horizon": 5, "wait_steps": 10, "max_policy_actions": 900, "renderer": "egl",
            "reset": "explicit seed 7 at each episode; fixed task/trial state; paired noise per query",
            "timing": "sd1 eager exploration only; no rb2 paper timing claim",
            "training_condition_ages": c["training_condition_ages"],
            "evaluation_condition_intervals": c["evaluation_condition_intervals"],
            "student_condition": c.get("student_condition_description", "recursive predictions through ages 1,2,3; teacher-recorded observations"),
            "training": c.get("training_description", "three matched 5k arms; fixed Condition/backbone/decoder; NO success or MSE stopping gate")}}
    contract["identity"] = digest(contract)
    dest = output / "contract.json"
    if dest.exists() and read_json(dest) != contract:
        if (output / "train").exists() or (output / "online").exists():
            raise RuntimeError("Production contract changed; choose a new semantic run ID")
    write_json(dest, contract)
    write_json(output / "episode_manifest.json", {"identity": contract["identity"], **contract["protocol"],
        "evaluation_seed": c["evaluation_seed"], "action_noise_seed_base": c["action_noise_seed_base"],
        "episodes": [{"task_id": task, "trial_id": trial, "init_state_index": trial} for task in range(9, -1, -1) for trial in range(50)]})
    print("PREPARE_PASS: inputs, source, package versions, 500-episode paired manifest", flush=True)


def jobs(c, config, smoke):
    prefix = [c["python"], "-m"]
    result = []
    for arm in ARMS:
        result.append({"id": "train_" + arm, "deps": [], "cmd": prefix + [
            "architectures.simvla.adapters.latentloop.efficient_multirate.error_compensation_train",
            "--config", str(config), "--arm", arm] + (["--smoke"] if smoke else []),
            "summary": str(Path(c["output"]) / ("smoke" if smoke else "train") / arm / "summary.json")})
    for k_c in c["evaluation_condition_intervals"]:
        for row in ROWS:
            key = f"kc{k_c}_{row}"
            result.append({"id": "eval_" + key, "deps": ["train_" + row] if row in ARMS else [],
                "cmd": prefix + ["tools.simvla.error_compensation_eval", "--config", str(config),
                    "--row", row, "--k-c", str(k_c)] + (["--smoke"] if smoke else []),
                "summary": str(Path(c["output"]) / ("eval_smoke" if smoke else "online") / key / "summary.json")})
    return result


def job_complete(job, run_identity, smoke, steps, smoke_steps=3):
    path = Path(job["summary"])
    if not path.exists():
        return False
    report = read_json(path)
    if report.get("identity") != run_identity:
        raise RuntimeError(f"Incompatible result: {path}")
    training = job["id"].startswith("train_")
    verdict = "SMOKE_PASS" if smoke else ("TRAIN_AND_OFFLINE_COMPLETE" if training else "EVALUATION_COMPLETE")
    key = "steps" if training else "episodes"
    expected = (smoke_steps if smoke else steps) if training else (1 if smoke else 500)
    return report.get("verdict") == verdict and report.get(key) == expected


def idle(gpu):
    line = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    memory, utilization = map(int, line.strip().split(","))
    return memory < 512 and utilization < 5


def summarize(c):
    output = Path(c["output"])
    keys = [f"kc{k}_{r}" for k in c["evaluation_condition_intervals"] for r in ROWS]
    reports = {r: read_json(output / "online" / r / "summary.json") for r in keys
        if (output / "online" / r / "summary.json").exists()}
    write_json(output / "comparison_summary.json", {"complete": len(reports) == len(keys),
        "rows": reports, "unavailable": [r for r in keys if r not in reports],
        "interpretation": "Matched recursive Condition inputs: true_condition vs same_condition tests the oracle target; true_condition vs no_code tests shared observation features. condition_full10 removes Generation approximation. No superiority assumed."})


def campaign(c, config, smoke, *, job_builder=jobs, summarizer=summarize):
    output = Path(c["output"])
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "campaign.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepare(c)
    plan = job_builder(c, config, smoke)
    run_identity = read_json(output / "contract.json")["identity"]
    completed = {j["id"] for j in plan if job_complete(j, run_identity, smoke, c["steps"], c.get("smoke_steps", 3))}
    active, failed, retries = {}, {}, {}
    logfile = output / ("smoke_logs" if smoke else "logs")
    logfile.mkdir(exist_ok=True)
    def stop(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        last_status = 0
        while len(completed) + len(failed) < len(plan):
            for gpu, (job, process, stream) in list(active.items()):
                if process.poll() is None: continue
                stream.close()
                del active[gpu]
                ok = process.returncode == 0 and job_complete(job, run_identity, smoke, c["steps"], c.get("smoke_steps", 3))
                if ok:
                    completed.add(job["id"])
                    print(f"DONE gpu={gpu} {job['id']}", flush=True)
                else:
                    retries[job["id"]] = retries.get(job["id"], 0) + 1
                    if retries[job["id"]] > 1: failed[job["id"]] = process.returncode or "completion validation failed"
                    print(f"JOB_ERROR {job['id']} rc={process.returncode} retry={retries[job['id']]} log={logfile / (job['id'] + '.log')}", flush=True)
            for job in plan:
                if job["id"] not in completed and any(d in failed for d in job["deps"]):
                    failed[job["id"]] = "dependency failed"
            active_ids = {v[0]["id"] for v in active.values()}
            for gpu in (4, 5, 6, 7):
                if gpu in active or not idle(gpu): continue
                pending = next((j for j in plan if j["id"] not in completed | set(failed) | active_ids
                    and all(d in completed for d in j["deps"])), None)
                if pending is None: continue
                stream = (logfile / (pending["id"] + ".log")).open("a", buffering=1)
                process = subprocess.Popen(pending["cmd"], cwd=ROOT, env=environment(c, gpu),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (pending, process, stream)
                active_ids.add(pending["id"])
                print(f"START gpu={gpu} pid={process.pid} {pending['id']}", flush=True)
            if time.monotonic() - last_status > 60:
                status = {"active": {str(g): {"job": j["id"], "pid": p.pid} for g, (j, p, _) in active.items()},
                    "completed": sorted(completed), "failed": failed, "total_jobs": len(plan)}
                write_json(output / ("smoke_status.json" if smoke else "status.json"), status)
                print(f"STATUS {len(completed)}/{len(plan)} done; active={status['active']}; failures={failed}", flush=True)
                if not smoke: summarizer(c)
                last_status = time.monotonic()
            time.sleep(5)
    finally:
        for _, process, stream in active.values():
            if process.poll() is None: os.killpg(process.pid, signal.SIGTERM)
        for _, process, stream in active.values():
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stream.close()
        write_json(output / ("smoke_status.json" if smoke else "status.json"),
            {"active": {}, "completed": sorted(completed), "failed": failed,
                "total_jobs": len(plan), "unfinished": sorted({j["id"] for j in plan} - completed - set(failed))})
    if not smoke: summarizer(c)
    write_json(output / ("smoke_complete.json" if smoke else "campaign_complete.json"),
        {"verdict": "COMPLETE" if not failed else "INCOMPLETE", "completed": sorted(completed), "failed": failed})
    return 1 if failed else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    c = read_json(a.config)
    if a.prepare: prepare(c)
    else: raise SystemExit(campaign(c, Path(a.config).resolve(), a.smoke))
