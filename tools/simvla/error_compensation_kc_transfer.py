"""Evaluate fixed K_C=2-trained candidates at K_C=3/4 without retraining."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import MethodType

from tools.simvla.error_compensation_common import (
    ROOT, configure, digest, environment, identity, read_json, sha, write_json,
)

CONFIG = ROOT / "architectures/simvla/configs/error_compensation_kc_transfer_sd1.json"
ROWS = ("condition_naive3", "parent", "true_condition", "true_condition_no_code")
CANDIDATES = ROWS[2:]


def candidate_query(row, k_c, query):
    if row not in ROWS or k_c not in (2, 3, 4) or query < 0:
        raise ValueError((row, k_c, query))
    return row in CANDIDATES and query % k_c != 0


def expected_counts(row, k_c, actions):
    if row not in ROWS or k_c not in (2, 3, 4) or actions < 1:
        raise ValueError((row, k_c, actions))
    q = (actions + 4) // 5
    full = (q + k_c - 1) // k_c
    return {"queries": q, "full_vlm": full, "condition": q - full,
            "transformer": 3 * q, "generation": 0 if row == "condition_naive3" else 7 * q}


def prepare(config, path):
    if socket.gethostname() != "jbrserver1":
        raise RuntimeError("Only sd1 is authorized for this campaign")
    if config["gpu_pool"] != [4, 5, 6, 7] or config["condition_intervals"] != [3, 4]:
        raise ValueError("This campaign requires GPUs 4..7 and K_C=3,4")
    if tuple(config["rows"]) != ROWS:
        raise ValueError("Unexpected experiment rows")
    source = read_json(ROOT / config["source_config"])
    configure(source)
    source_id = identity(source)
    source_out = Path(source["output"])
    if read_json(source_out / "campaign_complete.json")["verdict"] != "COMPLETE":
        raise RuntimeError("Finish the previous campaign before transfer evaluation")
    source_contract = read_json(source_out / "contract.json")
    artifacts = {name: sha(source[name]) for name in ("condition_checkpoint", "generation_checkpoint", "norm_stats")}
    artifacts["cache_manifest"] = sha(Path(source["cache"]) / "manifest.json")
    if artifacts != source_contract["artifacts"]:
        raise RuntimeError("Original model, normalization or cache identity changed")
    for p, expected in source_contract["upstream_sources"].items():
        if sha(p) != expected:
            raise RuntimeError(f"Upstream source changed: {p}")
    checkpoints = {}
    for row in CANDIDATES:
        result = read_json(source_out / "train" / row / "summary.json")
        if result["identity"] != source_id or result["steps"] != 5000:
            raise RuntimeError(f"Invalid trained candidate: {row}")
        p = source_out / "train" / row / "latest.pt"
        checkpoints[row] = {"path": str(p), "sha256": sha(p)}
    manifest = read_json(source_out / "episode_manifest.json")
    expected = {(task, trial) for task in range(10) for trial in range(50)}
    if len(manifest["episodes"]) != 500 or {
        (s["task_id"], s["trial_id"]) for s in manifest["episodes"]
    } != expected or manifest["identity"] != source_id:
        raise RuntimeError("Expected the original paired 10-task x 50-trial manifest")
    source_files = [Path(__file__), ROOT / "architectures/simvla/wrappers/run_error_compensation_kc_transfer_sd1.sh"]
    contract = {"config": config, "source_config": source, "source_identity": source_id,
        "candidate_checkpoints": checkpoints, "source_manifest_sha256": sha(source_out / "episode_manifest.json"),
        "new_sources": {str(p.relative_to(ROOT)): sha(p) for p in source_files},
        "config_sha256": sha(path),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol": {**source_contract["protocol"], "condition_intervals": [3, 4],
            "candidate_training_k_c": 2, "additional_training_steps": 0,
            "n_g": 3, "full_indices": [0, 4, 8], "candidate_queries": "query_index % k_c != 0",
            "purpose": "longer-condition transfer, not a model trained at K_C=3/4",
            "baseline": "reuse prior K_C=1 reference; no repeated baseline episodes",
            "candidate_selection": "fixed final 5K checkpoints; no performance stopping gate"}}
    contract["identity"] = digest(contract)
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    # Workers verify the same metadata concurrently; do not race on a shared .tmp.
    with (out / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = {"contract.json": contract, "episode_manifest.json": {
            **manifest, "identity": contract["identity"], "source_identity": source_id}}
        for filename, value in metadata.items():
            destination = out / filename
            if destination.exists():
                if read_json(destination) != value:
                    raise RuntimeError("Transfer contract changed; use a new run directory")
            else:
                write_json(destination, value)
    return source, contract


def make_policy(source, row, k_c):
    from tools.simvla.error_compensation_eval import make_policy as original_policy
    from tools.simvla.compile_runtime import ActionStep

    policy = original_policy(source, row)
    policy.k_c = policy.refresh_every = k_c
    if row == "condition_naive3":
        return policy
    # Preserve the parent/candidate weights, mask convention and noise from K_C=2.
    # Only refresh scheduling changes, including which queries use the candidate.
    step = ActionStep(policy.model.transformer).eval()
    parent_loop = policy._experiment_loops[0]
    def decode(self, condition, proprio, *, policy_query_index):
        noise, seed = self._paired_initial_noise(condition, proprio, policy_query_index)
        normalized = self.action_adapter.normalize_proprio(proprio)
        updated = candidate_query(row, k_c, policy_query_index)
        loop = self._experiment_loops[1] if updated else parent_loop
        code = condition.new_zeros(condition.shape[0], parent_loop.updater.condition_code_dim)
        if updated and row != "true_condition_no_code":
            if self._condition_code is None:
                raise RuntimeError("Missing current observation-change features")
            code = self._condition_code
        trace = loop(noise, full_step=lambda x, t: step(condition, x, normalized, t),
            full_step_indices=(0, 4, 8), proprio=normalized, condition=condition,
            condition_valid_mask=self.condition_layout.valid_mask if updated else None,
            condition_change_code=code)
        self.metrics.counters["num_action_transformer_calls"] += 3
        self.metrics.counters["num_action_transformer_decodes"] += 1
        self.metrics.counters["num_generation_decoder_only_steps"] += 7
        return self.action_adapter.action_space.postprocess(trace.final_noisy_action), seed
    policy._decode = MethodType(decode, policy)
    return policy


def validate_episode(saved, run_id, row, k_c, spec):
    for key, expected in {"identity": run_id, "row": row, "k_c": k_c,
        "task_id": spec["task_id"], "trial_id": spec["trial_id"]}.items():
        if saved.get(key) != expected:
            raise RuntimeError(f"Mismatched saved episode {key}")
    if not 1 <= saved["episode_length"] <= 900:
        raise RuntimeError("Invalid completed episode length")
    if saved["counters"] != expected_counts(row, k_c, saved["episode_length"]):
        raise RuntimeError("Saved episode invocation mismatch")


def evaluate(policy, source, contract, row, k_c, directory, *, smoke=False):
    import numpy as np
    import torch
    from libero.libero import benchmark
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
        build_env_obs, get_libero_env, save_episode_video, video_frame_from_obs,
    )
    run_id = contract["identity"]
    manifest = read_json(Path(contract["config"]["output"]) / "episode_manifest.json")
    specs = manifest["episodes"][:1] if smoke else manifest["episodes"]
    directory.mkdir(parents=True, exist_ok=True)
    done, handles, calls = [], [], Counter()
    def add(module, name):
        handles.append(module.register_forward_pre_hook(lambda _m, _i: calls.update([name])))
    add(policy.model.transformer.blocks[0], "transformer")
    add(policy.native_v0.condition_updater, "condition")
    if row != "condition_naive3":
        for loop in policy._experiment_loops:
            add(loop.updater, "generation")
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    env, last_task = None, None
    try:
        for spec in specs:
            task, trial = spec["task_id"], spec["trial_id"]
            path = directory / "episodes" / f"task{task}_trial{trial}.json"
            if path.exists() and not smoke:
                saved = read_json(path)
                validate_episode(saved, run_id, row, k_c, spec)
                done.append(saved)
                continue
            configure_strict_torch_determinism(source["evaluation_seed"])
            if task != last_task:
                if env is not None:
                    env.close()
                env, prompt = get_libero_env(suite.get_task(task), 256, 7)
                states = suite.get_task_init_states(task)
                last_task = task
            env.seed(7)
            env.reset()
            obs = env.set_init_state(states[trial])
            for _ in range(10):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
            policy.reset()
            policy.task_id, policy.trial_id = task, trial
            if hasattr(policy, "_condition_code"):
                policy._condition_code = None
            calls.clear()
            frames, timing = [], []
            started = last_progress = time.monotonic()
            success = False
            for index in range(26 if smoke else 900):
                inputs = build_env_obs(obs)
                if not smoke and trial == 0 and task in (9, 4) and index % 2 == 0:
                    frames.append(video_frame_from_obs(obs))
                torch.cuda.synchronize()
                tick = time.perf_counter()
                step = policy.act(*inputs, prompt)
                torch.cuda.synchronize()
                timing.append((time.perf_counter() - tick) * 1000)
                if not np.isfinite(step.action).all():
                    raise RuntimeError("Nonfinite action")
                obs, _, success, _ = env.step(step.action.tolist())
                if time.monotonic() - last_progress > 30:
                    progress = {"task": task, "trial": trial, "actions": index + 1,
                        "completed": len(done), "total": len(specs), "successes": sum(r["success"] for r in done)}
                    write_json(directory / "progress.json", progress)
                    print(f"PROGRESS kc{k_c}/{row} {progress}", flush=True)
                    last_progress = time.monotonic()
                if success:
                    break
            expected = expected_counts(row, k_c, index + 1)
            observed = {"queries": int(policy.metrics.counters["num_policy_queries"]),
                "full_vlm": int(policy.metrics.counters["num_full_vlm_calls"]),
                **{k: calls.get(k, 0) for k in ("condition", "transformer", "generation")}}
            if observed != expected:
                raise RuntimeError(f"Invocation mismatch: {observed} != {expected}")
            ages = sorted({int(t["age"]) for t in policy.query_trace})
            if smoke and ages != list(range(k_c)):
                raise RuntimeError(f"Smoke did not cover all condition ages: {ages}")
            result = {"identity": run_id, "row": row, "k_c": k_c, "task_id": task, "trial_id": trial,
                "success": bool(success), "episode_length": index + 1, "counters": observed,
                "condition_ages_seen": ages, "policy_ms_total": sum(timing),
                "wall_seconds": time.monotonic() - started}
            validate_episode(result, run_id, row, k_c, spec)
            write_json(path, result)
            done.append(result)
            if frames:
                try:
                    save_episode_video(frames, directory / f"task{task}_trial{trial}.mp4", fps=10)
                except Exception as error:
                    write_json(directory / f"task{task}_video_error.json", {"error": str(error)})
            print(f"kc{k_c}/{row}: {len(done)}/{len(specs)} success={sum(r['success'] for r in done)}/{len(done)} task={task} trial={trial}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        if env is not None:
            env.close()
    write_json(directory / "summary.json", {"verdict": "SMOKE_PASS" if smoke else "EVALUATION_COMPLETE",
        "identity": run_id, "row": row, "k_c": k_c, "episodes": len(done),
        "successes": sum(r["success"] for r in done), "success_rate": sum(r["success"] for r in done) / len(done),
        "policy_ms_per_action": sum(r["policy_ms_total"] for r in done) / sum(r["episode_length"] for r in done),
        "paper_latency": False, "compile": False, "gpu": torch.cuda.get_device_name(0),
        "timing_scope": "eager policy.act, outer CUDA sync, invocation hooks; sd1 transfer only",
        "per_task_successes": {str(t): sum(r["success"] for r in done if r["task_id"] == t) for t in range(10)},
        "candidate_training_k_c": 2, "additional_training_steps": 0})


def run_worker(config, path, row, k_c, smoke_only):
    source, contract = prepare(config, path)
    import torch
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    torch.set_num_threads(1)
    configure(source)
    configure_strict_torch_determinism(source["evaluation_seed"])
    key = f"kc{k_c}_{row}"
    out = Path(config["output"])
    (out / "locks").mkdir(exist_ok=True)
    with (out / "locks" / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with torch.inference_mode():
            policy = make_policy(source, row, k_c)
            smoke = out / "smoke" / key
            if not (smoke / "summary.json").exists():
                evaluate(policy, source, contract, row, k_c, smoke, smoke=True)
            elif read_json(smoke / "summary.json")["identity"] != contract["identity"]:
                raise RuntimeError("Stale smoke result")
            if not smoke_only:
                evaluate(policy, source, contract, row, k_c, out / "online" / key)


def summary_valid(path, run_id):
    if not path.exists():
        return False
    saved = read_json(path)
    if saved.get("identity") != run_id:
        raise RuntimeError(f"Stale summary: {path}")
    return saved.get("verdict") == "EVALUATION_COMPLETE" and saved.get("episodes") == 500


def campaign(config, path):
    from tools.simvla.error_compensation_campaign import idle
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source, contract = prepare(config, path)
    plan = [(k, r) for k in config["condition_intervals"] for r in config["rows"]]
    key = lambda job: f"kc{job[0]}_{job[1]}"
    complete = {key(j) for j in plan if summary_valid(out / "online" / key(j) / "summary.json", contract["identity"])}
    failed, active, retries = {}, {}, {}
    (out / "logs").mkdir(exist_ok=True)
    def report():
        write_json(out / "status.json", {"identity": contract["identity"], "total_jobs": len(plan),
            "completed": sorted(complete), "failed": failed,
            "active": {str(g): {"job": key(j), "pid": p.pid} for g, (j, p, _) in active.items()}})
        rows = {k: read_json(out / "online" / k / "summary.json") for k in sorted(complete)}
        write_json(out / "comparison_summary.json", {"identity": contract["identity"], "rows": rows,
            "complete": len(complete) == len(plan), "unavailable": [key(j) for j in plan if key(j) not in complete],
            "candidate_training_k_c": 2, "additional_training_steps": 0,
            "reference_kc2": str(Path(source["output"]) / "comparison_summary.json")})
    def stop(_sig, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    interrupted = False
    try:
        last = 0
        while len(complete) + len(failed) < len(plan):
            for gpu, (job, process, stream) in list(active.items()):
                if process.poll() is None:
                    continue
                stream.close()
                del active[gpu]
                name = key(job)
                if process.returncode == 0 and summary_valid(out / "online" / name / "summary.json", contract["identity"]):
                    complete.add(name)
                    print(f"DONE gpu={gpu} {name}", flush=True)
                else:
                    retries[name] = retries.get(name, 0) + 1
                    if retries[name] > 1:
                        failed[name] = process.returncode
                    print(f"JOB_ERROR {name} rc={process.returncode} retry={retries[name]}", flush=True)
            for gpu in config["gpu_pool"]:
                if gpu in active or not idle(gpu):
                    continue
                busy = {key(v[0]) for v in active.values()}
                job = next((j for j in plan if key(j) not in complete | set(failed) | busy), None)
                if job is None:
                    continue
                name = key(job)
                stream = (out / "logs" / (name + ".log")).open("a", buffering=1)
                env = environment(source, gpu)
                env.update(WANDB_MODE="offline", NUMBA_CACHE_DIR=str(out / "runtime/numba"),
                    MPLCONFIGDIR=str(out / "runtime/matplotlib"))
                cmd = [source["python"], "-u", "-m", "tools.simvla.error_compensation_kc_transfer",
                    "--config", str(path), "--row", job[1], "--k-c", str(job[0])]
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (job, p, stream)
                print(f"START gpu={gpu} pid={p.pid} {name}", flush=True)
            if time.monotonic() - last > 60:
                report()
                print(f"STATUS {len(complete)}/{len(plan)} done, running={[key(j) for j, _, _ in active.values()]}, failed={failed}", flush=True)
                last = time.monotonic()
            time.sleep(5)
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        for _, p, _ in active.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for _, p, stream in active.values():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            stream.close()
        active.clear()
        report()
        write_json(out / "campaign_complete.json", {"verdict": "COMPLETE" if len(complete) == len(plan) else "INCOMPLETE",
            "identity": contract["identity"], "completed": sorted(complete), "failed": failed, "interrupted": interrupted})
    return 0 if len(complete) == len(plan) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--row", choices=ROWS)
    parser.add_argument("--k-c", type=int, choices=(3, 4))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    path = Path(args.config).resolve()
    config = read_json(path)
    if args.prepare:
        _, contract = prepare(config, path)
        print("PREPARE_PASS", contract["identity"], flush=True)
    elif args.row:
        if args.k_c is None:
            parser.error("--row requires --k-c")
        run_worker(config, path, args.row, args.k_c, args.smoke)
    else:
        if args.smoke or args.k_c:
            parser.error("--smoke/--k-c requires --row")
        return campaign(config, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
