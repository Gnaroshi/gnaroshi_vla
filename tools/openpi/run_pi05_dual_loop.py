#!/usr/bin/env python3
"""Resumable single-GPU pi0.5 dual-loop training/evaluation coordinator."""

import argparse
import fcntl
import hashlib
import json
import netrc
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import zipfile

from dual_loop_runtime import ROOT, UPSTREAM, atomic_json, read_json, sha256

ROWS = ("baseline", "condition_k2", "generation_ng3", "dual_k2_ng3", "naive_nfe3")


def configure(storage, output):
    old = storage / "results/openpi/latentloop/contracts/pi05_v0_mode_b_rb2_seed42_30k"
    source = read_json(old / "source_lock_v2.json")
    checkpoint = Path(source["checkpoint"]["directory"])
    condition = storage / "results/openpi/latentloop/cacheless_streaming/train/pi05_v0_mode_b_rb2_seed42_30k/checkpoints/best.pt"
    for path, expected in ((checkpoint / "model.safetensors", source["checkpoint"]["model_sha256"]),
                           (checkpoint / "assets/physical-intelligence/libero/norm_stats.json", source["normalization"]["sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"baseline asset differs: {path}")
    if not (storage / "datasets/lerobot/physical-intelligence/libero/meta/info.json").is_file():
        raise FileNotFoundError("local LIBERO LeRobot data is missing")
    sources = {}
    for root in (ROOT / "architectures/openpi/adapters/latentloop", ROOT / "methods/latentloop",
                 ROOT / "methods/variable_time_latentloop", ROOT / "tools/openpi"):
        for path in sorted(root.rglob("*.py")):
            sources[str(path.relative_to(ROOT))] = sha256(path)
    sources["architectures/openpi/wrappers/run_pi05_dual_loop_long.sh"] = sha256(
        ROOT / "architectures/openpi/wrappers/run_pi05_dual_loop_long.sh")
    for rel in ("src/openpi/models_pytorch/pi0_pytorch.py", "src/openpi/models_pytorch/gemma_pytorch.py",
                "src/openpi/training/config.py", "src/openpi/policies/policy_config.py"):
        sources["upstream/" + rel] = sha256(UPSTREAM / rel)
    config = {
        "run_name": "pi05_dual_loop_long_seed7", "checkpoint": str(checkpoint),
        "baseline_model_sha256": source["checkpoint"]["model_sha256"],
        "normalization_sha256": source["normalization"]["sha256"],
        "condition_checkpoint": str(condition), "condition_checkpoint_sha256": sha256(condition),
        "split_contract": str(old / "protocol/pi05_split_contract_v2.json"),
        "final_manifest": str(old / "protocol/pi05_final_evaluation_manifest_v2.json"),
        "train_seed": 42, "noise_seed": 7, "train_steps": 10000, "validation_queries": 20,
        "learning_rate": {"peak": 1e-4, "warmup_steps": 200, "final": 1e-5, "schedule": "cosine_10k"},
        "training_objective": "layer_normalized_hidden_MSE + relative_velocity_MSE_at_student_x",
        "condition_training": "reuse_frozen_previous_V0_best; no additional condition optimization",
        "generation_training": "50% exact prefix, 50% frozen condition age1; separate updater, not joint finetuning",
        "H": 10, "R": 5, "integration_steps": 10, "generation_anchors_zero_based": [0, 4, 7],
        "rows": list(ROWS), "episodes_per_row": 500, "source_sha256": sources,
        "wandb_mode": os.environ.get("WANDB_MODE", "online"),
    }
    for field in ("split_contract", "final_manifest"):
        config[field + "_sha256"] = sha256(config[field])
    config["config_id"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    target = output / "config.json"
    if target.exists() and read_json(target) != config:
        raise RuntimeError("source/config changed since this run was created; set PI05_DUAL_OUTPUT to a new semantic run directory")
    atomic_json(target, config)
    atomic_json(output / "source_manifest.json", sources)
    import importlib.metadata
    import torch
    atomic_json(output / "environment_metadata.json", {
        "hostname": socket.gethostname(), "python": sys.version,
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"], text=True),
        "source_git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "deterministic_algorithms": True, "TF32": False, "renderer": "egl"})
    if config["wandb_mode"] == "online":
        try:
            auth = netrc.netrc().authenticators("api.wandb.ai")
        except (FileNotFoundError, netrc.NetrcParseError):
            auth = None
        if not (os.environ.get("WANDB_API_KEY") or auth):
            raise RuntimeError("W&B online needs login first; alternatively explicitly set WANDB_MODE=disabled")
    return config


def wait_gpu():
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    stable = 0
    while stable < 2:
        text = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        free = int(subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        stable = stable + 1 if not text and free >= 27000 else 0
        if stable < 2:
            print(f"GPU_WAIT gpu={gpu} free_MiB={free} active_compute_pids={text or 'none'}", flush=True)
            time.sleep(15 if stable else 60)


def run_logged(command, logfile):
    print("RUN " + " ".join(map(str, command)), flush=True)
    with Path(logfile).open("ab") as log:
        process = subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   bufsize=0)
        try:
            while True:
                chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            code = process.wait()
            if code:
                raise RuntimeError(f"phase failed rc={code}; log={logfile}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def eval_rows(config_path, generation, output, rows, smoke=False):
    from websockets.sync.client import connect
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ready = output / "server_ready.json"
    ready.unlink(missing_ok=True)
    server_cmd = [sys.executable, ROOT / "tools/openpi/serve_pi05_dual_loop.py", "--config", config_path,
                  "--generation", generation, "--port", str(port), "--ready", ready]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "server.log").open("a") as log:
        server = subprocess.Popen([str(x) for x in server_cmd], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"policy server failed; inspect {output / 'server.log'}")
                if ready.exists():
                    try:
                        with connect(f"ws://127.0.0.1:{port}", open_timeout=2) as connection:
                            connection.recv(timeout=2)
                            break
                    except (OSError, TimeoutError):
                        pass
                if time.monotonic() > deadline:
                    raise TimeoutError("policy server did not become ready within 10 minutes")
                time.sleep(2)
            for row in rows:
                target = output / row
                if (target / "summary.json").exists() and read_json(target / "summary.json").get("complete"):
                    print(f"RESUME_SKIP complete row={row}", flush=True)
                    continue
                command = [os.environ["PI05_CLIENT_PY"], ROOT / "tools/openpi/evaluate_pi05_dual_loop.py",
                           "--config", config_path, "--row", row, "--output", target, "--port", str(port)]
                if smoke:
                    command.append("--smoke")
                run_logged(command, output / (row + ".log"))
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            ready.unlink(missing_ok=True)


def aggregate(output):
    summaries = {row: read_json(output / "eval" / row / "summary.json") for row in ROWS}
    if not all(s.get("complete") and s["episodes"] == 500 for s in summaries.values()):
        raise RuntimeError("not all five rows have 500 completed episodes")
    atomic_json(output / "combined_summary.json", summaries)
    baseline_ms = summaries["baseline"]["policy_ms_per_actual_action"]
    lines = ["# pi0.5 LIBERO-Long 두 Loop 결과", "", "각 행 10 tasks x 50 trials, seed 7. rb2 RTX5090.",
             "Condition은 기존 V0 checkpoint를 고정하고 Generation만 10K 학습했다. 두 모듈 공동학습은 아니다.",
             "학습 손실은 새 이식의 설계 선택이며 원 논문의 공식 pi0.5 설정이 아니다.", "",
             "| 구성 | 성공 | 성공률 | policy ms/실행 action | baseline 대비 가속 |",
             "|---|---:|---:|---:|---:|"]
    for row, s in summaries.items():
        latency = s["policy_ms_per_actual_action"]
        lines.append(f"| {row} | {s['successes']}/500 | {100*s['success_rate']:.2f}% | {latency:.3f} | {baseline_ms/latency:.3f}x |")
    lines += ["", "policy latency는 서버 전처리, 모델, 후처리를 포함한다. 환경 step 및 websocket 전송은 제외한다.",
              "구성요소 latency와 client roundtrip은 각 summary.json에 별도로 기록했다.",
              "이는 한 seed의 신규 평가이며 이전 4-suite 평균과 직접 동일시하지 않는다."]
    (output / "report_ko.md").write_text("\n".join(lines) + "\n")
    with zipfile.ZipFile(output / "results_for_chatgpt.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for pattern in ("*.json", "*.md", "train/*.json", "eval/*/*.json", "eval/*/*.csv"):
            for path in output.glob(pattern):
                archive.write(path, path.relative_to(output))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "verify", "all", "summarize"))
    args = p.parse_args()
    storage = Path(os.environ["PI05_STORAGE"])
    output = Path(os.environ.get("PI05_DUAL_OUTPUT", storage / "results/openpi/dual_loop/libero_long_seed7"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    lock = (output / "pipeline.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another dual-loop launcher already owns this output")
    status = output / "status.json"
    phase = "prepare"
    try:
        config = configure(storage, output)
        config_path = output / "config.json"
        print(f"PREFLIGHT_PASS config={config['config_id']} output={output}", flush=True)
        if args.mode == "prepare":
            return
        if args.mode == "summarize":
            aggregate(output)
            return
        if args.mode == "all" and all((output / "eval" / row / "summary.json").exists()
                and read_json(output / "eval" / row / "summary.json").get("complete") for row in ROWS):
            aggregate(output)
            atomic_json(status, {"phase": "complete", "state": "complete", "config_id": config["config_id"]})
            return
        phase = "gpu_wait"
        atomic_json(status, {"phase": phase, "state": "running"})
        wait_gpu()
        verification = output / "verification.json"
        if not verification.exists():
            phase = "bounded_verification"
            atomic_json(status, {"phase": phase, "state": "running"})
            run_logged([sys.executable, ROOT / "tools/openpi/train_pi05_generation.py", "--config", config_path,
                        "--output", output / "smoke_train", "--steps", "2", "--smoke"], output / "logs/verify_train.log")
            eval_rows(config_path, output / "smoke_train/best.pt", output / "smoke_eval", ROWS, smoke=True)
            atomic_json(verification, {"pass": True, "config_id": config["config_id"],
                                      "scope": "2 training steps + 10 environment steps per row; no SR claim"})
        if args.mode == "verify":
            atomic_json(status, {"phase": "verified", "state": "complete"})
            return
        phase = "generation_train"
        atomic_json(status, {"phase": phase, "state": "running"})
        train_summary = output / "train/summary.json"
        if not train_summary.exists() or not read_json(train_summary).get("complete"):
            run_logged([sys.executable, ROOT / "tools/openpi/train_pi05_generation.py", "--config", config_path,
                        "--output", output / "train", "--steps", str(config["train_steps"])], output / "logs/train.log")
        phase = "libero_long_eval"
        atomic_json(status, {"phase": phase, "state": "running"})
        eval_rows(config_path, output / "train/best.pt", output / "eval", ROWS)
        phase = "summarize"
        aggregate(output)
        atomic_json(status, {"phase": "complete", "state": "complete", "config_id": config["config_id"]})
        print(f"COMPLETE report={output / 'report_ko.md'}", flush=True)
    except BaseException as error:
        atomic_json(status, {"phase": phase, "state": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                             "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
