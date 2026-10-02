"""Compiled SimVLA LIBERO campaign with immutable inputs and row recovery."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import traceback

from tools.simvla.compile_benchmark import (
    DEFAULT_CONFIG, ROOT, Replay, configure, model_asset_identity, preflight,
    read_json, sha, source_identity, stop_worker, verify_recorded_inputs, write_json,
)
from tools.simvla.compiled_policy import attach_policy, base_row, check_policy, check_reset

CONFIG = ROOT / "architectures/simvla/configs/compile_campaign_rb2.json"
SEEDS = {"seed01": (20260815, 6828326409295398833), "seed02": (20260816, 6828326409295398834),
         "seed03": (20260817, 6828326409295398835)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def manifest_path(c, suite, seed):
    return Path(c["manifest_root"]) / suite / seed / "episode_manifest.json"


def validate_manifest(m, suite, seed):
    payload = {k: v for k, v in m.items() if k != "manifest_sha256"}
    if digest(payload) != m["manifest_sha256"]:
        raise RuntimeError("Episode manifest hash changed")
    expected = {"suite": suite, "determinism_seed": SEEDS[seed][0],
        "action_noise_seed_base": SEEDS[seed][1], "environment_seed": 7,
        "action_horizon": 10, "execution_horizon": 5, "flow_steps": 10,
        "num_wait_steps": 10, "client_resize_size": 224, "model_image_size": 384,
        "environment_resolution": 256, "max_policy_actions": 900 if suite == "libero_10" else 800}
    for k, v in expected.items():
        if m.get(k) != v:
            raise RuntimeError(f"Manifest mismatch {k}: {m.get(k)} != {v}")
    pairs = [(x["task_id"], x["trial_id"]) for x in m["episodes"]]
    if len(pairs) != 500 or set(pairs) != {(t, r) for t in range(10) for r in range(50)}:
        raise RuntimeError("Exactly 10 tasks x 50 trials are required")
    if any(x["init_state_index"] != x["trial_id"] or x["suite"] != suite for x in m["episodes"]):
        raise RuntimeError("Episode initial-state assignment changed")
    return m


def jobs(c):
    result = []
    for suite, rows in [("libero_10", c["long_rows"])] + [(s, c["other_rows"]) for s in c["other_suites"]]:
        for seed in c["seeds"]:
            for row in rows:
                result.append((suite, seed, row))
    return result


def sources(c):
    result = source_identity(c)
    extra = [ROOT / "tools/simvla/compiled_campaign.py", ROOT / "tools/simvla/compiled_policy.py",
        ROOT / "tools/simvla/compiled_profile.py",
        CONFIG, DEFAULT_CONFIG, ROOT / "architectures/simvla/wrappers/run_compiled_paper_rb2.sh"]
    extra.extend(ROOT / p for p in c.get("extra_source_files", []))
    for directory in (ROOT / "architectures/simvla/wrappers/dcld_eval", ROOT / "architectures/simvla/adapters", ROOT / "methods"):
        extra.extend(directory.rglob("*.py"))
    for p in extra:
        result[str(p)] = sha(p)
    return result


def prepare(c, output):
    preflight(c)
    verify_recorded_inputs(Path(c["recorded_input"]))
    if socket.gethostname() != "jbr-TRX50":
        raise RuntimeError("This campaign is restricted to rb2")
    import yaml
    libero_config = Path(c["libero_config"]) / "config.yaml"
    settings = yaml.safe_load(libero_config.read_text())
    for key in ("assets", "bddl_files", "init_states"):
        if not Path(settings[key]).is_dir():
            raise RuntimeError(f"LIBERO {key} directory is missing")
    manifests = {f"{s}/{seed}": validate_manifest(read_json(manifest_path(c, s, seed)), s, seed)
        for s in ["libero_10"] + c["other_suites"] for seed in c["seeds"]}
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import FROZEN_GENERATION_CHECKPOINT_SHA256, FROZEN_NORM_STATS_SHA256
    from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_contracts import FROZEN_CONDITION_CHECKPOINT_SHA256
    expected = {"condition_checkpoint": FROZEN_CONDITION_CHECKPOINT_SHA256,
        "generation_checkpoint": FROZEN_GENERATION_CHECKPOINT_SHA256, "norm_stats": FROZEN_NORM_STATS_SHA256}
    artifacts = {key: sha(c[key]) for key in (*expected, "bridge_checkpoint")}
    if any(artifacts[key] != value for key, value in expected.items()):
        raise RuntimeError("Frozen checkpoint or norm stats changed")
    contract = {"config": c, "source_files": sources(c), "artifacts": artifacts,
        "hf_assets": model_asset_identity(c), "libero_config": settings,
        "libero_config_sha256": sha(libero_config),
        "manifest_hashes": {k: v["manifest_sha256"] for k, v in manifests.items()},
        "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip(),
        "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "measurement": "CUDA-synchronized policy.act incl. CPU preprocessing and action transfer; env/render/video excluded; outer sync only",
        "options": {"max_autotune": True, "cudagraphs": False, "dynamic": False,
            "base_dtype": "float32", "bridge_dtype": "bfloat16"},
        "episodes_per_cell": 500, "episodes_total": len(jobs(c)) * 500,
        "new_training": False, "success_threshold": None}
    path = output / "campaign_contract.json"
    if path.exists() and read_json(path) != contract:
        raise RuntimeError("Campaign contract changed; preserve this output and use a new run name")
    write_json(path, contract)
    for key, m in manifests.items():
        write_json(output / "manifests" / key / "episode_manifest.json", m)
    return contract


def move(value):
    import torch
    if torch.is_tensor(value): return value.cuda()
    if isinstance(value, list): return [move(x) for x in value]
    if isinstance(value, dict): return {k: move(v) for k, v in value.items()}
    return value


def check_compiler(compiler, row):
    from tools.simvla.compile_checks import required_components
    missing = [x for x in required_components(base_row(row)) if compiler.records.get(x, {}).get("graphs", 0) == 0]
    if missing:
        raise RuntimeError("Compile bypass: " + str(missing))


def warmup(policy, obs, prompt, compiler, row, *, actions=40,
           policy_checker=check_policy, compiler_checker=check_compiler, reset_checker=check_reset):
    import torch
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import build_env_obs
    images = build_env_obs(obs)
    started = time.perf_counter()
    for cycle in range(2):
        reset_checker(policy)
        for _ in range(actions):
            policy.act(*images, prompt)
        torch.cuda.synchronize()
        print(f"WARMUP row={row} cycle={cycle+1}/2 graphs={compiler.graph_count()}", flush=True)
    compiler_checker(compiler, row)
    policy_checker(policy, row)
    reset_checker(policy)
    return time.perf_counter() - started


def summarize_cell(directory, identity, expected):
    values = [read_json(p) for p in sorted((directory / "episodes").glob("*.json"))]
    keys = {(v["task_id"], v["trial_id"]) for v in values}
    if keys != set(expected) or len(values) != len(expected):
        return None
    if any(v["identity"] != identity for v in values):
        raise RuntimeError("Mixed experiment identity in episode results")
    successes = sum(v["success"] for v in values)
    valid = [v for v in values if v["timing_valid"]]
    action_count = sum(v["episode_length"] for v in valid)
    report = {"verdict": "EPISODES_COMPLETE", "identity": identity, "episodes": len(values),
        "successes": successes, "success_rate": successes / len(values),
        "timing_valid_episodes": len(valid), "executed_actions": action_count,
        "pooled_policy_ms_per_action": sum(v["policy_ms_total"] for v in valid) / action_count if action_count else None,
        "mean_episode_policy_ms_per_action": statistics.mean(v["policy_ms_total"] / v["episode_length"] for v in valid) if valid else None,
        "per_task": {str(t): {"episodes": sum(v["task_id"] == t for v in values), "successes": sum(v["success"] for v in values if v["task_id"] == t)} for t in sorted({v["task_id"] for v in values})},
        "timing_scope": "policy.act; excludes env/render/video; compilation-contaminated episodes excluded from timing, never from SR"}
    write_json(directory / "summary.json", report)
    with (directory / "outcomes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "trial_id", "success", "episode_length", "policy_ms_total", "timing_valid"])
        writer.writeheader()
        writer.writerows({k: v[k] for k in writer.fieldnames} for v in values)
    return report


def worker(c, output, suite_name, seed, row, *, smoke=False,
           replay_factory=Replay, policy_factory=attach_policy,
           policy_checker=check_policy, compiler_checker=check_compiler, reset_checker=check_reset):
    import numpy as np
    import torch
    from libero.libero import benchmark
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import build_env_obs, get_libero_env, video_frame_from_obs, save_episode_video
    from tools.simvla.compile_runtime import Compiler
    from tools.simvla.compile_audit import WarningCapture
    import logging

    contract = read_json(output / "campaign_contract.json")
    if sources(c) != contract["source_files"]:
        raise RuntimeError("Source changed after campaign preparation")
    m = validate_manifest(read_json(output / "manifests" / suite_name / seed / "episode_manifest.json"), suite_name, seed)
    identity = digest({"campaign": digest(contract), "suite": suite_name, "seed": seed, "row": row, "smoke": smoke})
    directory = output / ("smoke" if smoke else "rows") / suite_name / seed / row
    specs = sorted(m["episodes"], key=lambda x: (-x["task_id"], x["trial_id"]))
    if smoke: specs = specs[:c.get("smoke_episodes", 2)]
    expected = [(x["task_id"], x["trial_id"]) for x in specs]
    if summarize_cell(directory, identity, expected):
        print(f"RECOVERED_COMPLETE row={row} episodes={len(specs)}", flush=True)
        return
    # A partial trajectory is not a resumable RNG state. Preserve failed
    # attempts separately instead of merging different rollout histories.
    if list((directory / "episodes").glob("*.json")):
        raise RuntimeError("Partial row preserved; requires explicit recovery, not silent episode skipping")
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    deterministic = configure_strict_torch_determinism(m["determinism_seed"])
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total - 2 * 1024**3) / total)
    verify_recorded_inputs(Path(c["recorded_input"]))
    sample = move(torch.load(Path(c["recorded_input"]) / "recorded_inputs.pt", map_location="cpu", weights_only=False)[:1])
    capture = WarningCapture()
    for name in ("torch._dynamo", "torch._inductor"): logging.getLogger(name).addHandler(capture)
    compiler = Compiler(True)
    with torch.inference_mode():
        replay = replay_factory(c, base_row(row), compiler, sample)
        policy = policy_factory(replay, c, row, m)
        suite = benchmark.get_benchmark_dict()[suite_name]()
        completed = successes = saved_videos = 0
        cell_start = time.monotonic()
        observations = []
        for task_id in sorted({x["task_id"] for x in specs}, reverse=True):
            env, prompt = get_libero_env(suite.get_task(task_id), 256, 7)
            states = suite.get_task_init_states(task_id)
            task_specs = [x for x in specs if x["task_id"] == task_id]
            try:
                for position, spec in enumerate(task_specs):
                    policy.task_id, policy.trial_id = task_id, spec["trial_id"]
                    env.reset()
                    obs = env.set_init_state(states[spec["init_state_index"]])
                    for _ in range(10): obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                    if position == 0:
                        write_json(directory / "progress.json", {"phase": "warmup", "task": task_id, "completed": completed, "total": len(specs)})
                        seconds = warmup(policy, obs, prompt, compiler, row,
                            actions=c.get("warmup_actions", 40), policy_checker=policy_checker,
                            compiler_checker=compiler_checker, reset_checker=reset_checker)
                        write_json(directory / f"warmup_task{task_id}.json", {"seconds_excluded": seconds, "compiler": compiler.records})
                    reset_checker(policy)
                    times, frames, actions = [], [], []
                    graphs_before = compiler.graph_count()
                    success = False
                    started = time.monotonic()
                    last_progress = started
                    limit = c["smoke_actions"] if smoke else m["max_policy_actions"]
                    for index in range(limit):
                        images = build_env_obs(obs)
                        if len(observations) < 8 and not policy.action_queue:
                            observations.append((*[x.copy() for x in images], prompt))
                        if not smoke and saved_videos < c["videos_per_row"] and index % 2 == 0:
                            frames.append(video_frame_from_obs(obs))
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        step = policy.act(*images, prompt)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter() - tick) * 1000)
                        if not np.isfinite(step.action).all(): raise RuntimeError("Non-finite action")
                        obs, _, done, _ = env.step(step.action.tolist())
                        actions.append(step.action.copy())
                        if time.monotonic() - last_progress > 30:
                            write_json(directory / "progress.json", {"phase": "rollout", "task": task_id, "trial": spec["trial_id"], "action": index+1, "completed": completed, "successes": successes, "total": len(specs)})
                            last_progress = time.monotonic()
                        if done:
                            success = True
                            break
                    policy_checker(policy, row)
                    compiler_checker(compiler, row)
                    fallback = any(any(t in msg for t in ("hit config.recompile_limit", "hit config.cache_size_limit", "WON'T CONVERT")) for msg in capture.messages)
                    timing_valid = compiler.graph_count() == graphs_before and not fallback
                    result = {"identity": identity, "suite": suite_name, "seed": seed, "row": row,
                        "task_id": task_id, "trial_id": spec["trial_id"], "success": int(success),
                        "episode_length": len(actions), "policy_ms_total": sum(times), "policy_ms": times,
                        "elapsed_seconds": time.monotonic()-started, "timing_valid": timing_valid,
                        "graphs_before": graphs_before, "graphs_after": compiler.graph_count(),
                        "counters": dict(policy.metrics.counters), "action_sha256": hashlib.sha256(np.stack(actions).tobytes()).hexdigest()}
                    write_json(directory / "episodes" / f"task{task_id}_trial{spec['trial_id']:02d}.json", result)
                    if not smoke:
                        np.savez_compressed(directory / f"task{task_id}_trial{spec['trial_id']:02d}_actions.npz", actions=np.stack(actions))
                    if frames:
                        try:
                            save_episode_video(frames, directory / "videos" / f"task{task_id}_trial{spec['trial_id']:02d}.mp4", fps=10)
                            saved_videos += 1
                        except Exception as exc:
                            write_json(directory / "video_error.json", {"error": str(exc)})
                            saved_videos = c["videos_per_row"]
                    completed += 1
                    successes += int(success)
                    progress = {"phase": "rollout", "completed": completed, "successes": successes, "total": len(specs),
                        "eta_seconds": (time.monotonic()-cell_start)/completed*(len(specs)-completed)}
                    write_json(directory / "progress.json", progress)
                    print(f"EPISODE row={row} suite={suite_name} seed={seed} {completed}/{len(specs)} successes={successes} last={int(success)} eta_s={progress['eta_seconds']:.0f}", flush=True)
            finally:
                env.close()
        if hasattr(policy, "close"): policy.close()
        torch.save(observations, directory / "observations.pt")
        write_json(directory / "runtime.json", {"compiler": compiler.records, "warnings": capture.messages, "determinism": deterministic,
            "libero_module": str(Path(benchmark.__file__).resolve()), "identity": identity})
        summarize_cell(directory, identity, expected)


def aggregate(c, output):
    rows = []
    for suite, seed, row in jobs(c):
        path = output / "rows" / suite / seed / row / "summary.json"
        if path.exists(): rows.append({"suite": suite, "seed": seed, "row": row, **read_json(path)})
    groups = []
    for suite, row in sorted({(v["suite"], v["row"]) for v in rows}):
        group = [v for v in rows if v["suite"] == suite and v["row"] == row]
        complete = {v["seed"] for v in group} == set(c["seeds"])
        groups.append({"suite": suite, "row": row, "completed_seeds": len(group),
            "three_seed_complete": complete,
            "success_rate_mean": statistics.mean(v["success_rate"] for v in group) if complete else None,
            "success_rate_sample_std": statistics.stdev(v["success_rate"] for v in group) if complete else None,
            "successes": sum(v["successes"] for v in group), "episodes": sum(v["episodes"] for v in group)})
    write_json(output / "combined_summary.json", {"completed_cells": len(rows), "planned_cells": len(jobs(c)),
        "episodes": sum(x["episodes"] for x in rows), "three_seed_results": groups, "results": rows})
    return len(rows)


def run_child(c, output, command, suite, seed, row):
    while subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        write_json(output / "status.json", {"phase": "waiting_for_gpu", "row": row})
        print("GPU occupied; waiting without starting a GPU worker", flush=True)
        time.sleep(30)
    directory = output / {"smoke": "smoke", "worker": "rows", "profile": "latency"}[command] / suite / seed / row
    log_path = output / "logs" / f"{command}_{suite}_{seed}_{row}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {command} {suite}/{seed}/{row} log={log_path}", flush=True)
    with log_path.open("a") as log:
        entry = (["-m", c["campaign_module"]] if c.get("campaign_module")
                 else [str(Path(__file__).resolve())])
        process = subprocess.Popen([sys.executable, "-u", *entry, command,
            "--suite", suite, "--seed", seed, "--row", row, "--output", str(output)], stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, "PYTHONHASHSEED": str(SEEDS[seed][0])})
        started = time.monotonic()
        try:
            while process.poll() is None:
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    progress = read_json(directory / "progress.json") if (directory / "progress.json").exists() else {"phase": "loading_or_compiling"}
                    status = {"phase": command, "suite": suite, "seed": seed, "row": row, "pid": process.pid, "elapsed_seconds": time.monotonic()-started, "progress": progress}
                    write_json(output / "status.json", status)
                    print(json.dumps(status), flush=True)
                    if time.monotonic()-started > c["worker_timeout_seconds"]:
                        stop_worker(process)
        except BaseException:
            stop_worker(process)
            raise
    write_json(log_path.with_suffix(".status.json"), {"exit_code": process.returncode, "elapsed_seconds": time.monotonic()-started})
    print(f"FINISH {command} {suite}/{seed}/{row} rc={process.returncode}", flush=True)
    return process.returncode == 0


def run_all(c, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "launcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(c, output)
        failures = []
        eligible = set()
        for row in c["long_rows"]:
            if run_child(c, output, "smoke", "libero_10", "seed01", row): eligible.add(row)
            else: failures.append({"stage": "smoke", "row": row})
        for suite, seed, row in jobs(c):
            if row not in eligible: continue
            if not run_child(c, output, "worker", suite, seed, row):
                # All rollouts can be intact even if final serialization failed.
                contract = read_json(output / "campaign_contract.json")
                identity = digest({"campaign": digest(contract), "suite": suite, "seed": seed, "row": row, "smoke": False})
                recovered = summarize_cell(output / "rows" / suite / seed / row, identity,
                    [(t, r) for t in range(10) for r in range(50)])
                if recovered:
                    print(f"POSTPROCESS_RECOVERED {suite}/{seed}/{row}", flush=True)
                else:
                    failures.append({"stage": "online", "suite": suite, "seed": seed, "row": row})
            aggregate(c, output)
            write_json(output / "failures.json", failures)
        for row in c["long_rows"]:
            if row in eligible and not run_child(c, output, "profile", "libero_10", "seed01", row):
                failures.append({"stage": "latency", "row": row})
        completed = aggregate(c, output)
        write_json(output / "status.json", {"phase": "finished", "completed_cells": completed, "planned_cells": len(jobs(c)), "failures": failures})
        return completed == len(jobs(c)) and not failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("preflight", "all", "smoke", "worker", "profile", "summarize"))
    p.add_argument("--suite", default="libero_10")
    p.add_argument("--seed", default="seed01", choices=tuple(SEEDS))
    p.add_argument("--row", default="baseline")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    c = {**read_json(DEFAULT_CONFIG), **read_json(CONFIG)}
    if args.output: c["output"] = str(args.output.resolve())
    configure(c)
    sys.path.insert(0, c["libero_root"])
    os.environ["LIBERO_CONFIG_PATH"] = c["libero_config"]
    output = Path(c["output"])
    if args.row not in c["long_rows"]: raise ValueError("Unknown row")
    try:
        if args.command == "preflight":
            contract = prepare(c, output)
            print(f"PREFLIGHT_PASS cells={len(jobs(c))} episodes={contract['episodes_total']}")
        elif args.command == "all": return 0 if run_all(c, output) else 2
        elif args.command == "summarize": aggregate(c, output)
        elif args.command == "profile":
            from tools.simvla.compiled_profile import profile
            profile(c, output, args.suite, args.seed, args.row)
        else: worker(c, output, args.suite, args.seed, args.row, smoke=args.command == "smoke")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
