"""Wait for existing OpenPI campaign, then run one bounded SimVLA pilot."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.simvla.condition_mechanism_pipeline import configure, digest, provenance, signature, write_json


def process_records(proc_root=Path("/proc")):
    records = []
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            cmd = (path / "cmdline").read_bytes().decode(errors="replace").replace("\0", " ").strip()
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            records.append({"pid": int(path.name), "start_ticks": fields[19],
                            "state": fields[0], "command": cmd})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return records


def openpi_blockers(records):
    markers = ("tools/openpi/run_pi05_dual_loop.py", "wrappers/run_pi05_dual_loop_long.sh")
    return [r for r in records if r["pid"] != os.getpid() and r["state"] != "Z"
            and any(marker in r["command"] for marker in markers)]


def remaining_blockers(records, captured):
    identities = {(r["pid"], r["start_ticks"]) for r in captured}
    current = {r["pid"]: r for r in openpi_blockers(records)}
    current.update({r["pid"]: r for r in records if r["state"] != "Z"
                    and (r["pid"], r["start_ticks"]) in identities})
    return list(current.values())


def gpu_status(c):
    lines = subprocess.check_output(["nvidia-smi", "-i", str(c["physical_gpu"]),
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
    memory = subprocess.check_output(["nvidia-smi", "-i", str(c["physical_gpu"]),
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
    return {"compute_pids": [int(p.strip()) for p in lines.splitlines() if p.strip()], "free_mib": int(memory)}


def wait_idle(c, output):
    snapshot = output / "metadata" / "wait_dependencies.json"
    if snapshot.exists():
        captured = json.loads(snapshot.read_text())["processes"]
    else:
        captured = openpi_blockers(process_records())
        write_json(snapshot, {"processes": captured, "time": time.time(),
            "policy": "wait entire OpenPI parent pipeline plus GPU idle; never kill another process"})
    stable = 0
    while True:
        try:
            blockers = remaining_blockers(process_records(), captured)
            gpu = gpu_status(c)
            idle = not blockers and not gpu["compute_pids"] and gpu["free_mib"] >= c["minimum_free_mib"]
            stable = stable + 1 if idle else 0
            state = "WAITING_OPENPI" if blockers else "WAITING_GPU"
            if stable >= c["idle_samples"]:
                state = "GPU_READY"
            record = {"state": state, "time": time.time(), "openpi_processes": blockers,
                **gpu, "idle_samples": stable, "required_idle_samples": c["idle_samples"]}
            write_json(output / "status.json", record)
            print(f"{state} openpi_pids={[p['pid'] for p in blockers]} "
                  f"gpu_pids={gpu['compute_pids']} free_mib={gpu['free_mib']} stable={stable}", flush=True)
            if state == "GPU_READY":
                return
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            stable = 0
            write_json(output / "status.json", {"state": "WAITING_STATUS_UNAVAILABLE", "error": str(exc), "time": time.time()})
            print(f"WAITING_STATUS_UNAVAILABLE {exc}", flush=True)
        time.sleep(c["poll_seconds"])


def preflight(c, output, *, bind=True):
    if platform.node() != c["host"] or c["host"] != "jbr-TRX50" or c["physical_gpu"] != 0:
        raise RuntimeError("This launcher is restricted to rb2 GPU0")
    if not 0 < c["warmup_steps"] < c["steps"] or c["steps"] != 10000:
        raise ValueError("Expected the approved bounded 10K horizon")
    if c["idle_samples"] < 2 or c["batch_size"] < 1:
        raise ValueError("Invalid queue or batch configuration")
    for name in ("python", "upstream", "cache", "condition_checkpoint", "norm_stats", "hf_home"):
        if not Path(c[name]).exists():
            raise FileNotFoundError(f"{name}: {c[name]}")
    disk = os.statvfs(c["storage"])
    if disk.f_bavail * disk.f_frsize < 5 * 1024**3:
        raise RuntimeError("At least 5 GiB free disk required; no production cache will be built")
    configure(c)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import make_datasets
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import validate_exact_cache
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement import condition_query
    from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import install_exact_uint8_delta_path
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import models_for_seed
    torch.set_num_threads(1)
    p = provenance(c)
    own_paths = [Path(__file__), ROOT / "methods/latentloop/modules/shared_refinement.py",
        ROOT / "architectures/simvla/adapters/latentloop/efficient_multirate/shared_refinement.py",
        ROOT / "architectures/simvla/adapters/latentloop/efficient_multirate/shared_refinement_train.py",
        ROOT / "architectures/simvla/wrappers/run_shared_refinement.sh"]
    p["source_sha256"].update({str(path.relative_to(ROOT)): digest(path) for path in own_paths})
    identity = signature(p)
    previous = output / "metadata" / ("contract.json" if bind else "preflight_preview.json")
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if bind and git_status.strip():
        raise RuntimeError("This dedicated experiment worktree must be committed before launch")
    if bind and previous.exists() and json.loads(previous.read_text())["identity"] != identity:
        raise RuntimeError("Existing run source/config/input differs. Preserve it and choose a new output")
    cache = validate_exact_cache(c["cache"], verify_checksums=False)
    if not cache["passed"]:
        raise RuntimeError(json.dumps(cache))
    cache_manifest = json.loads((Path(c["cache"]) / "manifest.json").read_text())
    if cache_manifest["norm_stats_sha256"] != p["input_sha256"]["norm_stats"]:
        raise RuntimeError("Teacher cache normalization differs from frozen SimVLA")
    if cache_manifest["checkpoint"] != c["checkpoint"] or cache_manifest["flow_steps"] != 10:
        raise RuntimeError("Teacher cache is not the original native-10 policy")
    adapter, payload = load_native_v0_checkpoint(c["condition_checkpoint"], device="cpu", require_final_150k=True)
    install_exact_uint8_delta_path(adapter)
    train, heldout = make_datasets(c, payload)
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
    # Check all referenced HDF5 paths, then actually read every task's first
    # train/heldout window. This catches moved mounts and camera/key mismatches.
    image_paths = {train.store.query(window[0])["metadata"]["raw_rgb_ref"]["hdf5_path"]
                   for window in train.store.manifest["windows"]}
    for path in image_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Cache source image file: {path}")
    checked = {}
    for label, dataset in (("train", train), ("heldout", heldout)):
        first = {}
        for index, value in enumerate(dataset.identities):
            first.setdefault(value[0], index)
        checked[label] = []
        for task, index in sorted(first.items()):
            sequence = collate_exact_teacher_sequences([dataset[index]])
            for age in (1, 2, 3):
                context = condition_query(adapter, sequence, age)
                if context.token_code.shape[1:] != (122, 65) or not bool(torch.isfinite(context.condition).all()):
                    raise RuntimeError("Condition feature contract failed")
            checked[label].append(task)
    models = models_for_seed(c["seed"], torch.device("cpu"))
    write_json(previous, {"identity": identity, **p})
    write_json(output / "metadata" / "preflight.json", {"verdict": "CPU_PREFLIGHT_PASS", "identity": identity,
        "checked_task_ids": checked, "source_hdf5_files": len(image_paths),
        "train_windows": len(train), "heldout_windows": len(heldout),
        "parameters": {k: sum(v.numel() for v in m.parameters()) for k, m in models.items()},
        "gpu_started": False})
    print("CPU_PREFLIGHT_PASS (no CUDA allocation)", flush=True)
    return identity, p


def run_worker(c, output, identity, p):
    configure(c)
    from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import (
        load_runtime, gpu_contract_smoke, run_training, evaluate,
    )
    write_json(output / "status.json", {"state": "GPU_CONTRACT_SMOKE", "time": time.time()})
    runtime = load_runtime(c, p)
    smoke = output / "gpu_contract_smoke.json"
    if not smoke.exists():
        gpu_contract_smoke(c, output, runtime, write_json)
    write_json(output / "status.json", {"state": "TRAINING", "time": time.time()})
    models = run_training(c, output, identity, runtime, write_json)
    write_json(output / "status.json", {"state": "OFFLINE_COMPARISON", "time": time.time()})
    report = evaluate(c, output, identity, runtime, models, write_json)
    write_json(output / "status.json", {"state": "COMPLETE", "time": time.time(), "verdict": report["verdict"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "architectures/simvla/configs/shared_refinement_rb2.json"))
    parser.add_argument("mode", choices=("preflight", "wait-and-train", "worker"))
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    output = Path(c["output"])
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "worker":
        p = json.loads((output / "metadata" / "contract.json").read_text())
        identity = p.pop("identity")
        if signature(p) != identity:
            raise RuntimeError("Corrupt run provenance")
        run_worker(c, output, identity, p)
        return 0
    lock = (output / "pipeline.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        _, p = preflight(c, output, bind=args.mode != "preflight")
        if args.mode == "preflight":
            return 0
        for attempt in range(2):
            wait_idle(c, output)
            # Use a fresh interpreter: CPU preflight must never initialize this
            # worker with CUDA_VISIBLE_DEVICES=''. Child creates CUDA only now.
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
            log = output / f"worker_attempt{attempt + 1}.log"
            print(f"WORKER_START attempt={attempt + 1} log={log}", flush=True)
            with log.open("ab", buffering=0) as handle:
                child = subprocess.Popen([c["python"], "-u", str(Path(__file__).resolve()),
                    "--config", str(Path(args.config).resolve()), "worker"], cwd=ROOT, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                try:
                    while data := child.stdout.read1(4096):
                        handle.write(data)
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    returncode = child.wait()
                except BaseException:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                    raise
                finally:
                    child.stdout.close()
            if returncode == 0:
                print(f"COMPLETE results={output}", flush=True)
                return 0
            print(f"WORKER_FAILED rc={returncode} log={log}", flush=True)
            if attempt == 0:
                print("One resume attempt from the last complete optimizer checkpoint", flush=True)
        raise RuntimeError("Both bounded worker attempts failed; inspect worker logs")
    except BaseException as exc:
        write_json(output / "status.json", {"state": "FAILED", "error": str(exc), "time": time.time()})
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
