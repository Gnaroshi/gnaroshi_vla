"""Check trained Doll artifacts, then evaluate only missing coupled non-Long rows."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

STORAGE = Path("/home/mingyujung/private/gnaroshi_vla_storage")
DRIVER = Path(__file__).resolve().parents[2]
FIXED = Path("/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_paper_grid_seed02")
UPSTREAM = Path("/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream")
RESULTS = STORAGE / "results/simvla"
OUTPUT = RESULTS / "paper_completion/coupled_nonlong_three_seed_v1"
REAL = RESULTS / "real_world/stackcupanddoll_v2_corrected"
INPUTS = STORAGE / "artifacts/simvla/fixed_2x2_inputs_v1"
PROVENANCE = RESULTS / "paper_followup/three_seed_long500_primary_v1/provenance"
COUPLED = RESULTS / "coupled_condition_generation/kc2_ng3_real_cj_projection10k_seed02_v1/train/projection_10k/checkpoints/coupled_generation_step_010000.pt"
ROW = "condition_kc2_ng3_coupled"
SUITES = ("libero_spatial", "libero_object", "libero_goal")
SEEDS = ("seed01", "seed02", "seed03")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def records(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def episode_ids(rows):
    return {(int(r["task_id"]), int(r["trial_id"])) for r in rows}


def build_plan():
    cells = []
    for suite in SUITES:
        for seed in SEEDS:
            reference = RESULTS / (
                "paper_nonlong_seed01_primary_v1" if seed == "seed01"
                else "paper_completion/three_seed_5090_egl_v1/nonlong"
            )
            manifest = reference / "manifests" / suite / seed / "episode_manifest.json"
            data = read(manifest)
            expected = {(task, trial) for task in range(10) for trial in range(50)}
            if (data["suite"] != suite or data["inference_seed_replica"] != seed
                    or data["max_policy_actions"] != 800
                    or data["action_horizon"] != 10 or data["execution_horizon"] != 5
                    or data["flow_steps"] != 10 or len(data["episodes"]) != 500
                    or episode_ids(data["episodes"]) != expected):
                raise ValueError(f"Reference contract mismatch: {manifest}")
            parity = reference / "gates" / suite / seed / "fixed_2x2_parity.json"
            gate = read(parity)
            if gate["verdict"] != "FIXED_2X2_PARITY_PASS" or gate["manifest_sha256"] != data["manifest_sha256"]:
                raise ValueError(f"Reference parity mismatch: {parity}")
            registry = read(reference / "summary/selected_matrix_summary.json")
            matches = [c for c in registry["cell_reports"].values()
                       if c["suite"] == suite and c["inference_seed"] == seed
                       and c["row"] == "full_nfe10"]
            if len(matches) != 1 or matches[0]["manifest_sha256"] != data["manifest_sha256"]:
                raise ValueError(f"Baseline registry mismatch: {reference}")
            baseline = Path(matches[0]["metrics_path"])
            baseline_rows = records(baseline)
            if len(baseline_rows) != 500 or episode_ids(baseline_rows) != expected:
                raise ValueError(f"Baseline episode IDs mismatch: {baseline}")
            cells.append(dict(suite=suite, seed=seed, manifest=str(manifest),
                digest=data["manifest_sha256"], parity=str(parity),
                baseline=str(baseline), output=str(OUTPUT / "rows" / suite / seed / ROW)))
    return cells


def command(cell):
    return ["bash", str(FIXED / "architectures/simvla/wrappers/run_fixed_2x2_single_gpu_row.sh"),
        "--row", ROW, "--output", cell["output"], "--manifest", cell["manifest"],
        "--manifest-sha256", cell["digest"], "--bundle-root", str(INPUTS / "generation_bundle"),
        "--condition-checkpoint", str(INPUTS / "condition/native_v0_step_150000.pt"),
        "--coupled-generation-checkpoint", str(COUPLED),
        "--source-lock", str(PROVENANCE / "fixed_eval_source_lock.json"),
        "--control-manifest", str(PROVENANCE / "control_manifest.json"),
        "--parity-gate", cell["parity"], "--physical-gpu-id", "0",
        "--classification", "RB2_CONFIRMATORY_EGL", "--inference-seed", cell["seed"],
        "--task-ids", "0,1,2,3,4,5,6,7,8,9", "--episodes-per-task-limit", "50"]


def validate_cell(cell):
    directory = Path(cell["output"]) / "merged"
    summary = read(directory / "row_summary.json")
    rows = records(directory / "episode_metrics.csv")
    baseline = records(cell["baseline"])
    successes = sum(r["success"].lower() in {"true", "1"} for r in rows)
    if (len(rows) != 500 or episode_ids(rows) != episode_ids(baseline)
            or summary["row"] != ROW or summary["inference_seed"] != cell["seed"]
            or summary["manifest_sha256"] != cell["digest"]
            or summary["episodes"] != 500 or summary["successes"] != successes
            or summary["verdict"] != "FIXED_2X2_ROW_PASS"
            or summary["generation_checkpoint_sha256"] != hashlib.sha256(COUPLED.read_bytes()).hexdigest()
            or summary["paper_runtime_match"] is not True):
        raise ValueError(f"Completed row validation failed: {directory}")
    return dict(suite=cell["suite"], seed=cell["seed"], successes=successes,
        episodes=500, success_rate_percent=100 * successes / 500,
        latency_ms_per_action=summary["latency_per_executed_action_ms"],
        source=str(directory / "row_summary.json"))


def run(cmd, cwd, env, log, progress=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(cmd), flush=True)
    with log.open("a") as handle:
        process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        print(f"pid={process.pid} log={log}", flush=True)
        try:
            while process.poll() is None:
                time.sleep(30)
                message = f"RUNNING pid={process.pid}; log={log}"
                if progress is not None and progress.exists():
                    try:
                        rows = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
                        rows = [r for r in rows if "success" in r]
                        successes = sum(r["success"] in (True, 1, "true", "True") for r in rows)
                        message = f"PROGRESS {len(rows)}/500 success={successes}/{len(rows)} log={log}"
                    except (ValueError, OSError):
                        pass
                print(message, flush=True)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return process.returncode


def wait_gpu():
    while True:
        processes = subprocess.check_output(["nvidia-smi", "-i", "0",
            "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        free = int(subprocess.check_output(["nvidia-smi", "-i", "0",
            "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        if not processes and free >= 28000:
            return
        print(f"WAIT_GPU free_mib={free} compute_pids={processes!r}", flush=True)
        time.sleep(60)


def environment():
    env = dict(os.environ)
    env.update(PYTHONPATH=f"{FIXED}:{UPSTREAM}:{STORAGE}/datasets/LIBERO",
        SIMVLA_UPSTREAM_ROOT=str(UPSTREAM), SIMVLA_FIXED_2X2_ROOT=str(FIXED),
        SIMVLA_FIXED_2X2_PYTHON=sys.executable, SIMVLA_FIXED_2X2_RUN="1",
        SIMVLA_LIBERO_ROOT=str(STORAGE / "datasets/LIBERO"),
        LIBERO_ROOT=str(STORAGE / "datasets/LIBERO"),
        LIBERO_CONFIG_PATH=str(RESULTS / "reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/runtime/libero_config"),
        HF_HOME=str(STORAGE / "cache/simvla/huggingface"), HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        CUDA_VISIBLE_DEVICES="0", PYTHONDONTWRITEBYTECODE="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return env


def preflight(cells, env):
    for path in (FIXED, COUPLED, INPUTS / "condition/native_v0_step_150000.pt",
                 REAL / "deployment_bundle_v4/deployment_manifest.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    long_result = read(RESULTS / "paper_followup/three_seed_long500_primary_v1/rows/seed01/condition_kc2_ng3_coupled/merged/row_summary.json")
    if hashlib.sha256(COUPLED.read_bytes()).hexdigest() != long_result["generation_checkpoint_sha256"]:
        raise ValueError("Coupled checkpoint differs from the completed Long result")
    code = """
from types import SimpleNamespace
from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval import _verify_provenance
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import validate_manifest_identity
from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_contracts import FROZEN_CONDITION_SOURCE_SHA256, FROZEN_CONDITION_CHECKPOINT_SHA256
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import FROZEN_GENERATION_SOURCE_SHA256, FROZEN_GENERATION_CHECKPOINT_SHA256
import json, sys
d=json.load(open(sys.argv[1]))
p=_verify_provenance(SimpleNamespace(**d["provenance_args"]))
assert p["verdict"] == "FROZEN_PROVENANCE_PASS", p
for c in d["cells"]:
    m=json.load(open(c["manifest"]))
    assert validate_manifest_identity(m, expected_manifest_sha256=c["digest"])["verdict"] == "EPISODE_MANIFEST_PASS"
    g=json.load(open(c["parity"]))
    assert g["condition_source_combined_sha256"] == FROZEN_CONDITION_SOURCE_SHA256
    assert g["generation_source_combined_sha256"] == FROZEN_GENERATION_SOURCE_SHA256
    assert g["condition_checkpoint_sha256"] == FROZEN_CONDITION_CHECKPOINT_SHA256
    assert g["generation_checkpoint_sha256"] == FROZEN_GENERATION_CHECKPOINT_SHA256
print("COUPLED_NONLONG_CPU_PREFLIGHT_PASS cells=9 episodes=4500")
"""
    plan = dict(cells=cells, commands=[command(c) for c in cells],
        provenance_args=dict(bundle_root=str(INPUTS / "generation_bundle"),
            condition_checkpoint=str(INPUTS / "condition/native_v0_step_150000.pt"),
            fixed_2x2_source_lock=str(PROVENANCE / "fixed_eval_source_lock.json"),
            control_manifest=str(PROVENANCE / "control_manifest.json"),
            classification="RB2_CONFIRMATORY_EGL"))
    write(OUTPUT / "plan.json", plan)
    subprocess.run([sys.executable, "-c", code, str(OUTPUT / "plan.json")],
                   cwd=FIXED, env={**env, "CUDA_VISIBLE_DEVICES": ""}, check=True)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "all", "doll-only", "coupled-only"), default="preflight", nargs="?")
    args = parser.parse_args()
    if args.mode != "preflight" and os.environ.get("SIMVLA_COMPLETED_FOLLOWUP_RUN") != "1":
        raise ValueError("Set SIMVLA_COMPLETED_FOLLOWUP_RUN=1 to enable GPU work")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = environment()
        cells = build_plan()
        preflight(cells, env)
        if args.mode == "preflight":
            return
        failures, completed = [], []
        if args.mode in {"all", "doll-only"}:
            real_env = {**env, "PYTHONPATH": str(DRIVER), "SIMVLA_REAL_PYTHON": sys.executable,
                "SIMVLA_REAL_CUDA_DEVICE": "0", "SIMVLA_REAL_LOG_ROOT": str(REAL / "deployment_checks")}
            for method in ("baseline", "condition_loop", "latentloop"):
                wait_gpu()
                rc = run(["bash", str(DRIVER / "architectures/simvla/wrappers/deploy_latentloop_real.sh"),
                    "artifact-preflight", "--manifest", str(REAL / "deployment_bundle_v4/deployment_manifest.json"),
                    "--method", method], DRIVER, real_env, OUTPUT / f"logs/doll_{method}.log")
                if rc:
                    failures.append(f"doll:{method}:rc={rc}")
        if args.mode in {"all", "coupled-only"}:
            for cell in cells:
                try:
                    completed.append(validate_cell(cell))
                    print("REUSE", cell["suite"], cell["seed"], flush=True)
                    continue
                except (FileNotFoundError, ValueError, KeyError):
                    pass
                output = Path(cell["output"])
                if output.exists():
                    recovery = [sys.executable, "-m",
                        "architectures.simvla.adapters.latentloop.efficient_multirate.row_postprocess_recovery",
                        "--row", ROW, "--shard", str(output / "shard_rank0_tasks_0_9"),
                        "--merged", str(output / "merged"), "--expected-manifest-sha256", cell["digest"],
                        "--generation-checkpoint", str(COUPLED)]
                    run(recovery, FIXED, {**env, "CUDA_VISIBLE_DEVICES": ""},
                        OUTPUT / f"logs/recovery_{cell['suite']}_{cell['seed']}.log")
                else:
                    wait_gpu()
                    run(command(cell), FIXED, env, OUTPUT / f"logs/{cell['suite']}_{cell['seed']}.log",
                        output / "shard_rank0_tasks_0_9/progress.jsonl")
                try:
                    completed.append(validate_cell(cell))
                except (FileNotFoundError, ValueError, KeyError) as exc:
                    failures.append(f"{cell['suite']}:{cell['seed']}:{exc}")
                write(OUTPUT / "progress.json", dict(completed=completed, failures=failures))
        suites = {}
        for suite in SUITES:
            rows = [c for c in completed if c["suite"] == suite]
            if len(rows) == 3:
                suites[suite] = dict(success_rate_percent=statistics.mean(r["success_rate_percent"] for r in rows),
                    latency_ms_per_action=statistics.mean(r["latency_ms_per_action"] for r in rows))
        result = dict(verdict="FOLLOWUP_COMPLETE" if not failures else "FOLLOWUP_INCOMPLETE",
                      mode=args.mode, failures=failures, completed=completed, suites=suites,
                      robot_commands_issued=0, baseline_rerun=False, training_run=False)
        write(OUTPUT / "summary.json", result)
        print(json.dumps(result, indent=2), flush=True)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    def interrupted(*_):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    main()
