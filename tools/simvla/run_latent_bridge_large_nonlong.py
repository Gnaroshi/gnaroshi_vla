"""Resume the frozen Large bridge on nine existing non-Long episode manifests."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

STORAGE = Path("/home/mingyujung/private/gnaroshi_vla_storage")
RESULTS = STORAGE / "results/simvla"
ROOT = Path(__file__).resolve().parents[2]
LB = Path("/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_latent_bridge_matched")
UPSTREAM = Path("/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream")
OUTPUT = RESULTS / "latent_bridge/large_nonlong_three_seed_v1"
LONG = RESULTS / "latent_bridge/libero10_ours_matched_full_small_v1/full_183m_matched_eval"
BRIDGE = RESULTS / "latent_bridge/libero10_feature_bridge_paper_v4/r1_train/best.pt"
DOLL = RESULTS / "real_world/stackcupanddoll_v2_corrected"
SUITES = ("libero_spatial", "libero_object", "libero_goal")
SEEDS = ("seed01", "seed02", "seed03")
ROWS = tuple(f"latent_bridge_f{f}" for f in (2, 3, 4))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identities(rows):
    return {(int(r["task_id"]), int(r["trial_id"])) for r in rows}


def build_plan():
    cells = []
    for seed in SEEDS:
        reference = RESULTS / ("paper_nonlong_seed01_primary_v1" if seed == "seed01"
            else "paper_completion/three_seed_5090_egl_v1/nonlong")
        registry = read(reference / "summary/selected_matrix_summary.json")
        for suite in SUITES:
            manifest = reference / "manifests" / suite / seed / "episode_manifest.json"
            data = read(manifest)
            expected = {(t, i) for t in range(10) for i in range(50)}
            if (data["suite"] != suite or data["inference_seed_replica"] != seed
                    or data["max_policy_actions"] != 800 or len(data["episodes"]) != 500
                    or identities(data["episodes"]) != expected):
                raise ValueError(f"Invalid existing episode manifest: {manifest}")
            references = {}
            for row in ("full_nfe10", "condition_kc2_ng3"):
                matches = [c for c in registry["cell_reports"].values()
                    if c["suite"] == suite and c["inference_seed"] == seed and c["row"] == row]
                if len(matches) != 1 or matches[0]["manifest_sha256"] != data["manifest_sha256"]:
                    raise ValueError(f"Missing matching reference: {suite}/{seed}/{row}")
                report = matches[0]
                with Path(report["metrics_path"]).open(newline="") as handle:
                    episodes = list(csv.DictReader(handle))
                if len(episodes) != 500 or identities(episodes) != expected:
                    raise ValueError(f"Reference episodes incomplete: {report['metrics_path']}")
                references[row] = report
            cells.append(dict(suite=suite, seed=seed, manifest=str(manifest),
                manifest_sha256=data["manifest_sha256"], references=references,
                output=str(OUTPUT / "rows" / suite / seed)))
    return cells


def options(cell):
    return ["--output", cell["output"], "--resume-output",
        "--checkpoint", "YuankaiLuo/SimVLA-LIBERO",
        "--checkpoint-revision", "93dc4d90b0596c652ad2840ad743c62b9c4473fb",
        "--smolvlm-model", "HuggingFaceTB/SmolVLM-500M-Instruct",
        "--norm-stats", str(UPSTREAM / "norm_stats/libero_norm.json"),
        "--bridge-checkpoint", str(BRIDGE), "--reference-manifest", cell["manifest"],
        "--rows", *ROWS, "--bridge-precision", "bf16", "--compile-bridge",
        "--suite", cell["suite"], "--num-trials", "50", "--max-tasks", "10",
        "--max-policy-steps", "800", "--device", "cuda"]


def environment(cell):
    env = dict(os.environ)
    for name in ("EGL_DEVICE_ID", "GALLIUM_DRIVER", "LIBGL_ALWAYS_SOFTWARE"):
        env.pop(name, None)
    env.update(PYTHONPATH=f"{LB}:{UPSTREAM}:{STORAGE}/datasets/LIBERO",
        SIMVLA_UPSTREAM_ROOT=str(UPSTREAM), LATENT_BRIDGE_UPSTREAM_ROOT=str(LB / "architectures/latent_bridge/upstream"),
        LIBERO_ROOT=str(STORAGE / "datasets/LIBERO"),
        LIBERO_CONFIG_PATH=str(RESULTS / "reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/runtime/libero_config"),
        HF_HOME=str(STORAGE / "cache/simvla/huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="0", MUJOCO_EGL_DEVICE_ID="0", SIMVLA_LATENT_BRIDGE_EVAL_RUN="1",
        PYTHON=sys.executable, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", NUMBA_CACHE_DIR="/tmp/numba_cache",
        MPLCONFIGDIR=f"/tmp/matplotlib-{os.getuid()}")
    env.update({k: str(v) for k, v in read(cell["manifest"])["renderer"].items()})
    return env


def check_cell(index):
    from architectures.simvla.adapters.latent_bridge.eval import parser, _load_reference_manifest, _configure_paths
    from architectures.simvla.adapters.latent_bridge.provenance import simvla_latent_bridge_integration_manifest
    cell = read(OUTPUT / "plan.json")["cells"][index]
    args = parser().parse_args(options(cell))
    _, audit = _load_reference_manifest(args)
    _configure_paths()
    previous = read(LONG / cell["seed"] / "environment_metadata.json")
    current = simvla_latent_bridge_integration_manifest()
    if current["combined_sha256"] != previous["simvla_latent_bridge_integration"]["combined_sha256"]:
        raise ValueError("Bridge implementation differs from the completed Large Long evaluation")
    if index == 0:
        from architectures.simvla.adapters.latent_bridge.checkpoint import load_bridge_checkpoint
        bridge, payload = load_bridge_checkpoint(BRIDGE, device="cpu")
        if bridge.parameter_audit()["total"] != 183622848:
            raise ValueError("Not the completed Large configuration")
        if not payload["runtime_integration_compatibility"]["passed"]:
            raise ValueError("Checkpoint integration compatibility failed")
        identity = payload["provenance"]["training_data_identity"]
        if identity["checkpoint"] != args.checkpoint or identity["norm_stats_sha256"] != audit["norm_stats_sha256"]:
            raise ValueError("Bridge training base/norm differs from evaluation")
    print(f"MANIFEST_CPU_PASS {cell['suite']} {cell['seed']} episodes={audit['episodes']}", flush=True)


def metadata_ok(cell, meta):
    data = read(cell["manifest"])
    previous = read(LONG / cell["seed"] / "environment_metadata.json")
    expected = dict(suite=cell["suite"], rows=list(ROWS), trials_per_task=50,
        max_policy_actions=800, action_horizon=10, execution_horizon=5, flow_steps=10,
        num_wait_steps=10, client_resize_size=224, bridge_checkpoint=str(BRIDGE),
        evaluation_seed=data["determinism_seed"], environment_seed=data["environment_seed"],
        action_noise_seed_base=data["action_noise_seed_base"], task_ids=data["task_iteration_order"]["rank0"])
    for key in ("checkpoint", "checkpoint_revision", "norm_stats_sha256", "runtime_stack", "bridge_parameter_audit"):
        expected[key] = previous[key]
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Evaluation identity mismatch: {cell['output']}")
    if (meta["reference_manifest"]["canonical_manifest_sha256"] != cell["manifest_sha256"]
            or not meta["reference_manifest"]["protocol_exact_match"]
            or not meta["reference_manifest"]["renderer_exact_match"]
            or meta["simvla_latent_bridge_integration"]["combined_sha256"] != previous["simvla_latent_bridge_integration"]["combined_sha256"]
            or meta["determinism"] != previous["determinism"]):
        raise ValueError("Evaluation metadata is not on the existing matched comparison axis")


def validate_episodes(episodes, cell, row):
    expected = {(int(e["task_id"]), int(e["trial_id"])): e for e in read(cell["manifest"])["episodes"]}
    if len(episodes) != len(identities(episodes)) or not identities(episodes).issubset(expected):
        raise ValueError("Duplicate or unexpected episode identity")
    for e in episodes:
        spec = expected[(int(e["task_id"]), int(e["trial_id"]))]
        if e["row"] != row or any(e[k] != spec[k] for k in ("init_state_index", "environment_seed")):
            raise ValueError("Episode seed/initial state/row mismatch")
        if type(e["success"]) is not bool or not 1 <= e["episode_length"] <= 800:
            raise ValueError("Invalid episode outcome")
        for key in ("latency_per_executed_action_ms", "vlm_latency_total_ms", "bridge_latency_total_ms", "action_latency_total_ms"):
            if not math.isfinite(e[key]) or e[key] < 0:
                raise ValueError(f"Invalid measurement: {key}")
        if e["num_action_transformer_calls"] != 10 * e["num_policy_queries"]:
            raise ValueError("Latent Bridge must preserve ten action-transformer calls per query")


def recover(cell):
    directory = Path(cell["output"])
    if not directory.exists():
        return []
    if not (directory / "environment_metadata.json").exists():
        if any(p.stat().st_size for p in directory.glob("*/progress.jsonl")):
            raise ValueError("Episode records exist without provenance metadata")
        # Loading/compilation can be interrupted before the evaluator writes metadata.
        # Preserve that failed startup directory without making it a resumable run.
        number = 1
        backup = directory.with_name(directory.name + f"_startup_interrupted_{number}")
        while backup.exists():
            number += 1
            backup = directory.with_name(directory.name + f"_startup_interrupted_{number}")
        directory.rename(backup)
        print(f"PRESERVED_INTERRUPTED_STARTUP {backup}", flush=True)
        return []
    metadata_ok(cell, read(directory / "environment_metadata.json"))
    completed = []
    for row in ROWS:
        progress = directory / row / "progress.jsonl"
        if not progress.exists():
            continue
        episodes = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
        validate_episodes(episodes, cell, row)
        if len(episodes) < 500:
            continue
        f = int(row[-1])
        summary = dict(episodes=500, successes=sum(e["success"] for e in episodes),
            success_rate=statistics.mean(e["success"] for e in episodes),
            latency_per_executed_action_ms=statistics.mean(e["latency_per_executed_action_ms"] for e in episodes),
            full_vlm_calls=sum(e["num_full_vlm_calls"] for e in episodes),
            bridge_calls=sum(e["num_condition_updater_calls"] for e in episodes),
            refresh_every=f, expected_full_vlm_call_saving=1 - 1 / f)
        # Rebuild derived files only; immutable episode records remain the evidence.
        with (directory / row / "episode_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
            writer.writeheader()
            writer.writerows(episodes)
        write(directory / row / "summary.json", summary)
        completed.append(dict(suite=cell["suite"], seed=cell["seed"], row=row,
            manifest_sha256=cell["manifest_sha256"], summary=summary, source=str(progress)))
    if len(completed) == 3:
        write(directory / "comparison_summary.json", dict(verdict="SIMVLA_LATENT_BRIDGE_EVAL_COMPLETE",
            summaries={r["row"]: r["summary"] for r in completed}, paired_action_noise=True,
            paired_episode_identity="suite+task_id+trial_id+init_state_index+environment_seed",
            reference_manifest_exact=True))
    return completed


def doll_ready():
    exit_path = DOLL / "logs/pipeline.exit_code"
    if not exit_path.exists():
        return False
    if exit_path.read_text().strip() != "0":
        raise RuntimeError("Doll training exited unsuccessfully; not treating idle GPU as completion")
    bundle = DOLL / "deployment_bundle_v4/bundle_inventory.json"
    return bundle.exists() and read(bundle)["verdict"] == "REAL_SIMVLA_DEPLOYMENT_BUNDLE_PASS"


def wait_ready():
    stable = 0
    while True:
        ready = doll_ready()
        pids = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
        free = int(subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        stable = stable + 1 if ready and not pids and free >= 28000 else 0
        if stable >= 2:
            return
        print(f"WAIT doll_complete={ready} gpu0_free_mib={free} compute_pids={pids!r}", flush=True)
        time.sleep(30)


def run_cell(cell):
    log = OUTPUT / "logs" / f"{cell['suite']}_{cell['seed']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["bash", str(LB / "architectures/simvla/wrappers/simvla_latent_bridge_eval.sh"), *options(cell)]
    print(f"START {cell['suite']} {cell['seed']} f=2,3,4 log={log}", flush=True)
    with log.open("a") as handle:
        process = subprocess.Popen(cmd, cwd=LB, env=environment(cell), stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True)
        write(OUTPUT / "active_job.json", dict(pid=process.pid, command=cmd, cell=cell, started_unix=time.time()))
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    counts = []
                    for row in ROWS:
                        path = Path(cell["output"]) / row / "progress.jsonl"
                        lines = path.read_text().splitlines() if path.exists() else []
                        samples = []
                        for line in lines:
                            try:
                                samples.append(json.loads(line))
                            except json.JSONDecodeError:
                                break  # A writer may currently be appending its last record.
                        counts.append(f"f={row[-1]} {len(samples)}/500 success={sum(e['success'] for e in samples)}/{len(samples)}")
                    print(f"PROGRESS {cell['suite']} {cell['seed']} " + " | ".join(counts), flush=True)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    write(OUTPUT / "active_job.json", dict(pid=process.pid, returncode=process.returncode, finished_unix=time.time()))
    return process.returncode


def summarize(completed, failures):
    aggregates = []
    for suite in SUITES:
        for row in ROWS:
            cells = [c for c in completed if c["suite"] == suite and c["row"] == row]
            if len(cells) == 3:
                aggregates.append(dict(suite=suite, row=row, episodes=1500,
                    successes=sum(c["summary"]["successes"] for c in cells),
                    success_rate_percent=100 * statistics.mean(c["summary"]["success_rate"] for c in cells),
                    latency_per_executed_action_ms=statistics.mean(c["summary"]["latency_per_executed_action_ms"] for c in cells)))
    write(OUTPUT / "summary.json", dict(verdict="LARGE_NONLONG_COMPLETE" if len(completed) == 27 and not failures else "LARGE_NONLONG_INCOMPLETE",
        completed=completed, failures=failures, suite_aggregates=aggregates,
        expected_rows=27, expected_episodes=13500, baseline_rerun=False, long_rerun=False,
        training_run=False, coupled_ours_run=False, small_bridge_run=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "all"), nargs="?", default="preflight")
    parser.add_argument("--check-cell", type=int)
    args = parser.parse_args()
    if args.check_cell is not None:
        check_cell(args.check_cell)
        return
    if args.mode == "all" and os.environ.get("SIMVLA_LB_LARGE_NONLONG_RUN") != "1":
        raise RuntimeError("Set SIMVLA_LB_LARGE_NONLONG_RUN=1 to enable evaluation")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cells = build_plan()
        plan = dict(cells=cells, bridge_checkpoint=str(BRIDGE), bridge_sha256=sha(BRIDGE),
            commands=[["bash", str(LB / "architectures/simvla/wrappers/simvla_latent_bridge_eval.sh"), *options(c)] for c in cells])
        path = OUTPUT / "plan.json"
        if path.exists() and read(path) != plan:
            raise ValueError("Inputs changed since the first launch; refusing to mix results")
        write(path, plan)
        def git(directory, *arguments):
            return subprocess.check_output(["git", "-C", str(directory), *arguments], text=True).strip()
        gpu = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
        if "5090" not in gpu:
            raise RuntimeError(f"This launcher is for rb2 RTX5090 only: {gpu}")
        from importlib.metadata import version
        write(OUTPUT / "launcher_metadata.json", dict(host=platform.node(), python=sys.version,
            launcher_commit=git(ROOT, "rev-parse", "HEAD"), launcher_status=git(ROOT, "status", "--short"),
            evaluator_commit=git(LB, "rev-parse", "HEAD"), evaluator_status=git(LB, "status", "--short"),
            gpu=gpu, packages={name: version(name) for name in ("torch", "transformers", "mujoco", "numpy")},
            training="reuse frozen Long-trained Large; no per-suite adaptation", started_unix=time.time()))
        for index, cell in enumerate(cells):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--check-cell", str(index)],
                cwd=LB, env={**environment(cell), "CUDA_VISIBLE_DEVICES": ""}, check=True)
            recover(cell)
        print("LARGE_NONLONG_PREFLIGHT_PASS cells=9 rows=27 episodes=13500 GPU_rollouts=0", flush=True)
        if args.mode == "preflight":
            return
        completed, failures = [], []
        for cell in cells:
            try:
                recovered = recover(cell)
                if len(recovered) < 3:
                    wait_ready()
                    rc = run_cell(cell)
                    recovered = recover(cell)
                    if len(recovered) < 3:
                        raise RuntimeError(f"rc={rc}, completed rows={len(recovered)}/3; inspect logs")
                    if rc:
                        print("RECOVERED completed episode records after nonzero process exit", flush=True)
                completed.extend(recovered)
                print(f"COMPLETE {cell['suite']} {cell['seed']} total_rows={len(completed)}/27", flush=True)
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
                failures.append(dict(suite=cell["suite"], seed=cell["seed"], error=str(exc)))
                print(f"CELL_FAILED {cell['suite']} {cell['seed']}: {exc}; continuing independent cells", flush=True)
            summarize(completed, failures)
        if failures:
            raise SystemExit(1)
        print(f"LARGE_NONLONG_COMPLETE summary={OUTPUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    def interrupt(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    main()
