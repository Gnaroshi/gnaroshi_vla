"""Fail-closed sd1 stage graph for S50/S150 Condition stability alignment."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    validate_exact_cache,
)
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    BUNDLE_SCHEMA,
    SD1_STAGES,
    AtomicStageState,
    atomic_write_json,
    condition_only_2k_continuation,
    free_gpu_pairs,
    gpu_is_free,
    load_json,
    select_condition_only_parent,
    sha256_file,
)


WORKTREE = Path(
    os.environ.get(
        "SIMVLA_STABILITY_WORKTREE",
        "/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_stability_alignment",
    )
).resolve()
UPSTREAM = Path(
    os.environ.get(
        "SIMVLA_UPSTREAM_ROOT",
        "/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream",
    )
).resolve()
PYTHON = Path(
    os.environ.get(
        "SIMVLA_STABILITY_PYTHON",
        "/home/mingyujung/miniconda3/envs/simvla_libero/bin/python",
    )
).resolve()
RESULT_ROOT = Path(
    os.environ.get(
        "SIMVLA_STABILITY_RESULT_ROOT",
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/"
        "results/simvla/stability_alignment/condition_only_v2",
    )
).resolve()
TRANSFER_ROOT = Path(
    os.environ.get(
        "SIMVLA_STABILITY_TRANSFER_ROOT",
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/"
        "transfers/simvla_stability_aligned_selected",
    )
).resolve()
CACHE = Path(
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/"
    "simvla/latentloop/simvla_efficient_coupled_multirate_latentloop_sigfix_v1/"
    "03_exact_teacher_cache"
).resolve()
CONDITION = {
    "S50": Path(
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/"
        "simvla/latentloop/correct_native_v0_seed20260815_v1/08_train_150k/"
        "checkpoints/native_v0_step_050000.pt"
    ).resolve(),
    "S150": Path(
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/"
        "simvla/latentloop/correct_native_v0_seed20260815_v1/08_train_150k/"
        "checkpoints/native_v0_step_150000.pt"
    ).resolve(),
}
GENERATION = Path(
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/artifacts/"
    "simvla/generation_eval_bundle_20260824_v1/checkpoint/generation_step_030000.pt"
).resolve()
NORM = (UPSTREAM / "norm_stats/libero_norm.json").resolve()
MODULE = "architectures.simvla.adapters.latentloop.stability_alignment.trainer"
LONG_MODULE = "architectures.simvla.adapters.latentloop.stability_alignment.long_span"
SHORT_BRANCHES = ("S50", "S150")
TWO_K_TERMINAL_VERDICTS = (
    "STABILITY_2K_GATE_PASS",
    "STABILITY_2K_GATE_FAIL",
)
TEN_K_TERMINAL_VERDICTS = (
    "STABILITY_10K_GATE_PASS",
    "STABILITY_10K_GATE_FAIL",
)


@dataclass
class Job:
    name: str
    command: list[str]
    success_file: Path
    success_verdicts: tuple[str, ...]
    log: Path
    expected_fields: dict[str, Any] | None = None


@dataclass
class RunningJob:
    job: Job
    pair: tuple[int, int]
    process: subprocess.Popen[str]
    handle: Any


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _parse_pool(raw: str) -> tuple[int, ...]:
    result = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if len(result) < 2 or len(set(result)) != len(result):
        raise ValueError("SD1_GPU_POOL needs at least two unique IDs")
    return result


def _gpu_snapshot() -> dict[int, dict[str, Any]]:
    rows = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        text=True,
    ).splitlines()
    snapshot: dict[int, dict[str, Any]] = {}
    uuid_to_index: dict[str, int] = {}
    for row in rows:
        index, uuid, memory, utilization = [value.strip() for value in row.split(",")]
        gpu = int(index)
        snapshot[gpu] = {
            "memory_used_mib": int(memory),
            "utilization_percent": int(utilization),
            "compute_pids": [],
            "uuid": uuid,
        }
        uuid_to_index[uuid] = gpu
    applications = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    for row in applications.stdout.splitlines():
        values = [value.strip() for value in row.split(",")]
        if len(values) != 2 or values[0] not in uuid_to_index:
            continue
        snapshot[uuid_to_index[values[0]]]["compute_pids"].append(int(values[1]))
    for payload in snapshot.values():
        payload["free"] = gpu_is_free(**{key: payload[key] for key in ("memory_used_mib", "utilization_percent", "compute_pids")})
    return snapshot


def _active_audit() -> dict[str, Any]:
    snapshot = _gpu_snapshot()
    tmux = subprocess.run(
        ("tmux", "list-sessions", "-F", "#{session_name}|#{session_windows}"),
        text=True,
        capture_output=True,
        check=False,
    )
    processes = subprocess.check_output(("ps", "-eo", "pid=,args="), text=True)
    worktree_users = [
        line.strip()
        for line in processes.splitlines()
        if str(WORKTREE) in line
        and "stability_alignment.sd1_pipeline" not in line
        and "ps -eo" not in line
    ]
    return {
        "hostname": socket.gethostname(),
        "timestamp": _timestamp(),
        "gpu_snapshot": snapshot,
        "tmux_sessions": tmux.stdout.splitlines(),
        "source_worktree": str(WORKTREE),
        "source_head": subprocess.check_output(
            ("git", "-C", str(WORKTREE), "rev-parse", "HEAD"), text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ("git", "-C", str(WORKTREE), "status", "--short"), text=True
        ).splitlines(),
        "other_processes_using_worktree": worktree_users,
        "active_jobs_altered": False,
    }


def _common(branch: str, output: Path) -> list[str]:
    return [
        "--output",
        str(output),
        "--cache",
        str(CACHE),
        "--condition-parent",
        str(CONDITION[branch]),
        "--generation-parent",
        str(GENERATION),
        "--norm-stats",
        str(NORM),
        "--split-seed",
        "20260822",
        "--seed",
        "20260825",
        "--num-workers",
        "2",
    ]


def _torchrun(
    command: str,
    args: Sequence[str],
    port: int,
    *,
    module: str = MODULE,
) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        f"--master_port={int(port)}",
        "-m",
        module,
        command,
        *args,
    ]


def _completed(job: Job) -> bool:
    if not job.success_file.is_file():
        return False
    payload = load_json(job.success_file)
    if str(payload.get("verdict")) not in set(job.success_verdicts):
        return False
    for key, value in (job.expected_fields or {}).items():
        observed = payload.get(key)
        if key == "optimizer_step":
            if observed is None or int(observed) < int(value):
                return False
        elif observed != value:
            return False
    return True


def _gate_branch_decision(
    gate_paths: Mapping[str, Path],
    *,
    pass_verdict: str,
    fail_verdict: str,
) -> dict[str, Any]:
    """Separate completed gate evaluations from branches approved to continue."""
    outcomes: dict[str, dict[str, Any]] = {}
    passing: list[str] = []
    continuing: list[str] = []
    trend_warning: list[str] = []
    stopped: list[str] = []
    for branch, path in gate_paths.items():
        payload = load_json(path)
        verdict = str(payload.get("verdict"))
        if verdict not in {pass_verdict, fail_verdict}:
            raise RuntimeError(f"non-terminal {branch} gate verdict: {verdict}")
        gate_passed = bool(payload.get("gate", {}).get("passed"))
        if gate_passed != (verdict == pass_verdict):
            raise RuntimeError(
                f"inconsistent {branch} gate payload: verdict={verdict}, "
                f"gate.passed={gate_passed}"
            )
        continuation = condition_only_2k_continuation(payload)
        safety_passed = bool(continuation["passed"])
        trend_passed = bool(continuation["stability_trend_passed"])
        continue_condition_only = safety_passed
        outcomes[branch] = {
            "verdict": verdict,
            "passed": gate_passed,
            "safety_passed": safety_passed,
            "stability_trend_passed": trend_passed,
            "continue_condition_only": continue_condition_only,
            "condition_only_fallback": continuation["condition_only_fallback"],
            "optimizer_step": payload.get("optimizer_step"),
            "path": str(path),
        }
        if gate_passed:
            passing.append(branch)
        if continue_condition_only:
            continuing.append(branch)
            if not trend_passed:
                trend_warning.append(branch)
        else:
            stopped.append(branch)
    return {
        "verdict": (
            "SAFE_CONDITION_ONLY_CONTINUATION"
            if continuing
            else "NO_SAFE_BRANCH_SURVIVED"
        ),
        "passing_branches": passing,
        "continuing_branches": continuing,
        "trend_warning_branches": trend_warning,
        "stopped_branches": stopped,
        "branches": outcomes,
    }


def _select_short_span_summary(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"verdict": "NO_SHORT_SPAN_BRANCH_SELECTED", "selected": None}
    if len(summaries) == 1:
        summary = summaries[0]
        if not bool(summary.get("gate_passed")):
            return {"verdict": "NO_SHORT_SPAN_BRANCH_SELECTED", "selected": None}
        branch = str(summary["branch"])
        return {
            "verdict": f"{branch}_SELECTED_ONLY_2K_SURVIVOR",
            "selected": branch,
        }
    if len(summaries) != 2:
        raise ValueError(f"expected one or two short-span summaries, got {len(summaries)}")
    by_branch = {str(summary["branch"]): summary for summary in summaries}
    if set(by_branch) != set(SHORT_BRANCHES):
        raise ValueError(f"unexpected short-span branches: {sorted(by_branch)}")
    return select_condition_only_parent(by_branch["S50"], by_branch["S150"])


def _job_environment(pair: tuple[int, int]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "SIMVLA_GPU_IDS": ",".join(map(str, pair)),
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, pair)),
            "SIMVLA_UPSTREAM_ROOT": str(UPSTREAM),
            "PYTHONPATH": f"{WORKTREE}:{UPSTREAM}:{environment.get('PYTHONPATH', '')}",
            "HF_HOME": "/home/mingyujung/private/gnaroshi_vla/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTHONHASHSEED": "20260825",
            "WANDB_MODE": environment.get("WANDB_MODE", "online"),
        }
    )
    return environment


def _run_jobs(jobs: Sequence[Job], pool: Sequence[int], stage_dir: Path) -> None:
    pending = [job for job in jobs if not _completed(job)]
    if not pending:
        return
    running: list[RunningJob] = []
    stage_dir.mkdir(parents=True, exist_ok=True)
    while pending or running:
        finished: list[RunningJob] = []
        for active in running:
            rc = active.process.poll()
            if rc is None:
                continue
            active.handle.close()
            finished.append(active)
            atomic_write_json(
                stage_dir / f"{active.job.name}.process.json",
                {
                    "name": active.job.name,
                    "pair": list(active.pair),
                    "pid": active.process.pid,
                    "exit_code": rc,
                    "finished": _timestamp(),
                    "log": str(active.job.log),
                },
            )
            if rc != 0 or not _completed(active.job):
                raise RuntimeError(
                    f"job failed or completion verdict missing: {active.job.name}; log={active.job.log}"
                )
        for active in finished:
            running.remove(active)
        if pending:
            snapshot = _gpu_snapshot()
            busy = [gpu for gpu in pool if not snapshot.get(gpu, {}).get("free", False)]
            pairs = free_gpu_pairs(
                pool,
                busy,
                running_pairs=[active.pair for active in running],
                max_simultaneous_pairs=2,
            )
            for pair in pairs:
                if not pending:
                    break
                job = pending.pop(0)
                job.log.parent.mkdir(parents=True, exist_ok=True)
                handle = job.log.open("a", encoding="utf-8")
                handle.write(f"[{_timestamp()}] launch pair={pair} command={' '.join(job.command)}\n")
                handle.flush()
                process = subprocess.Popen(
                    job.command,
                    cwd=WORKTREE,
                    env=_job_environment(pair),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                active = RunningJob(job, pair, process, handle)
                running.append(active)
                atomic_write_json(
                    stage_dir / f"{job.name}.process.json",
                    {
                        "name": job.name,
                        "pair": list(pair),
                        "pid": process.pid,
                        "started": _timestamp(),
                        "log": str(job.log),
                    },
                )
        if pending or running:
            time.sleep(30)


def _branch_summary(branch: str, offline: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    by_age = offline["aggregate"]["by_age"]
    state = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
    parameters = sum(value.numel() for value in state["condition_state_dict"].values()) + sum(
        value.numel() for value in state["generation_state_dict"].values()
    )
    return {
        "branch": branch,
        "gate_passed": bool(offline["gate"]["passed"]),
        "age2_recursive_first_r_mean": by_age["2"]["candidate_recursive_first_r"]["mean"],
        "age3_recursive_first_r_mean": by_age["3"]["candidate_recursive_first_r"]["mean"],
        "age3_recursive_first_r_p95": by_age["3"]["candidate_recursive_first_r"]["p95"],
        "age3_gripper_tail": by_age["3"]["candidate_gripper_sign_mismatch"]["p99"],
        "exact_ng3_error": sum(
            by_age[str(age)]["candidate_exact_ng3_first_r"]["mean"] for age in (1, 2, 3)
        ) / 3.0,
        "parameter_count": parameters,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def _write_comparison(rows: Sequence[dict[str, Any]]) -> None:
    path = RESULT_ROOT / "s50_s150_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _continuation_allowed(offline: dict[str, Any]) -> bool:
    metrics = offline["aggregate"]["gate_metrics"]
    return bool(
        offline["gate"]["passed"]
        and float(metrics["stability_slope"]) < 0.0
        and bool(metrics["no_p99_or_gripper_collapse"])
    )


def _write_bundle_source_payload(destination: Path) -> None:
    source_root = destination / "source"
    source_root.mkdir()
    package = WORKTREE / "architectures/simvla/adapters/latentloop/stability_alignment"
    shutil.copytree(package, source_root / "stability_alignment")
    for relative in (
        "architectures/simvla/adapters/latentloop/efficient_multirate/fixed_2x2_eval.py",
        "architectures/simvla/adapters/latentloop/efficient_multirate/generation_control_eval.py",
    ):
        target = source_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WORKTREE / relative, target)
    patch_parts = [
        subprocess.check_output(
            ("git", "-C", str(WORKTREE), "diff", "--binary", "HEAD")
        )
    ]
    untracked = subprocess.check_output(
        (
            "git",
            "-C",
            str(WORKTREE),
            "ls-files",
            "--others",
            "--exclude-standard",
        ),
        text=True,
    ).splitlines()
    for relative in untracked:
        completed = subprocess.run(
            ("git", "diff", "--no-index", "--binary", "/dev/null", relative),
            cwd=WORKTREE,
            capture_output=True,
            check=False,
        )
        if completed.returncode not in {0, 1}:
            raise RuntimeError(f"failed to create source patch for {relative}")
        patch_parts.append(completed.stdout)
    (destination / "source.patch").write_bytes(b"".join(patch_parts))


def _export_bundle(
    *,
    selected: str,
    checkpoint: Path,
    offline: dict[str, Any],
    offline_path: Path,
    common_weights: Path,
) -> Path:
    destination = TRANSFER_ROOT / "short_span"
    if destination.exists():
        ready = destination / "READY_SHORT_FOR_RB2.json"
        if ready.is_file() and load_json(ready).get("checkpoint_sha256") == sha256_file(checkpoint):
            return destination
        raise FileExistsError(f"refusing incompatible existing bundle: {destination}")
    destination.mkdir(parents=True)
    shutil.copy2(checkpoint, destination / "stability_aligned_selected.pt")
    shutil.copy2(NORM, destination / "libero_norm.json")
    shutil.copy2(common_weights, destination / "stability_alignment_loss_weights.json")
    shutil.copy2(offline_path, destination / "offline_gate.json")
    shutil.copy2(
        RESULT_ROOT / "selected_short_span_checkpoint.json",
        destination / "selected_short_span_checkpoint.json",
    )
    train_root = checkpoint.parents[1]
    for name in ("source_lock.json", "training_contract.json", "determinism.json"):
        source = train_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / name)
    checkpoint_payload = __import__("torch").load(
        checkpoint, map_location="cpu", weights_only=False
    )
    atomic_write_json(
        destination / "parent_identity.json",
        checkpoint_payload.get("parent_identity", {}),
    )
    _write_bundle_source_payload(destination)
    readiness = offline.get("readiness", {})
    if not any(
        readiness.get(name, {}).get("passed", False) for name in ("kc3", "kc4")
    ):
        raise RuntimeError("selected checkpoint is not ready for K_C=3 or K_C=4")
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    manifest = {
        str(path.relative_to(destination)): sha256_file(path) for path in files
    }
    atomic_write_json(destination / "SHA256_MANIFEST.json", manifest)
    ready = {
        "schema_version": BUNDLE_SCHEMA,
        "verdict": "READY_SHORT_FOR_RB2",
        "selected_branch": selected,
        "checkpoint": "stability_aligned_selected.pt",
        "checkpoint_sha256": sha256_file(destination / "stability_aligned_selected.pt"),
        "optimizer_step": int(checkpoint_payload["optimizer_step"]),
        "kc3_offline_ready": bool(readiness.get("kc3", {}).get("passed")),
        "kc4_offline_ready": bool(readiness.get("kc4", {}).get("passed")),
        "generation_ng3_preserved": True,
        "n_g": 3,
        "full_generation_indices": [0, 4, 8],
    }
    atomic_write_json(destination / "READY_SHORT_FOR_RB2.json", ready)
    return destination


def _export_long_bundle(
    *,
    checkpoint: Path,
    offline_path: Path,
    common_weights: Path,
    catalog: Path,
    short_selection: Path,
) -> Path:
    destination = TRANSFER_ROOT / "long_span"
    checkpoint_sha = sha256_file(checkpoint)
    if destination.exists():
        ready = destination / "READY_KC8_FOR_RB2.json"
        if ready.is_file() and load_json(ready).get("checkpoint_sha256") == checkpoint_sha:
            return destination
        raise FileExistsError(f"refusing incompatible existing long bundle: {destination}")
    offline = load_json(offline_path)
    if offline.get("verdict") != "KC8_OFFLINE_READY" or not offline.get("passed"):
        raise RuntimeError("long bundle requires KC8_OFFLINE_READY")
    destination.mkdir(parents=True)
    checkpoint_name = "stability_aligned_kc8.pt"
    shutil.copy2(checkpoint, destination / checkpoint_name)
    shutil.copy2(NORM, destination / "libero_norm.json")
    shutil.copy2(common_weights, destination / "stability_alignment_loss_weights.json")
    shutil.copy2(offline_path, destination / "offline_gate.json")
    shutil.copy2(catalog, destination / "q0_q7_catalog.json")
    shutil.copy2(short_selection, destination / "selected_short_span_checkpoint.json")
    train_root = checkpoint.parents[1]
    for name in ("source_lock.json", "training_contract.json", "determinism.json"):
        source = train_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination / name)
    checkpoint_payload = __import__("torch").load(
        checkpoint, map_location="cpu", weights_only=False
    )
    parent_identity = dict(checkpoint_payload.get("parent_identity", {}))
    parent_identity.update(
        {
            "generation_parent": str(GENERATION),
            "generation_parent_sha256": sha256_file(GENERATION),
        }
    )
    atomic_write_json(destination / "parent_identity.json", parent_identity)
    _write_bundle_source_payload(destination)
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    atomic_write_json(
        destination / "SHA256_MANIFEST.json",
        {str(path.relative_to(destination)): sha256_file(path) for path in files},
    )
    ready = {
        "schema_version": BUNDLE_SCHEMA,
        "verdict": "READY_KC8_FOR_RB2",
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha,
        "kc8_offline_ready": True,
        "generation_ng3_preserved": True,
        "n_g": 3,
        "full_generation_indices": [0, 4, 8],
        "condition_ages": list(range(1, 8)),
        "short_parent_sha256": offline["short_parent_sha256"],
    }
    atomic_write_json(destination / "READY_KC8_FOR_RB2.json", ready)
    return destination


def _publish_bundle(
    bundle: Path,
    *,
    span: str,
    readiness_name: str,
) -> dict[str, Any]:
    """Push a ready bundle over the verified sd1 -> rb2 SSH direction."""

    destination = os.environ.get(
        "RB2_BUNDLE_DESTINATION",
        "rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/"
        "simvla_stability_aligned_selected",
    )
    if ":" not in destination:
        raise ValueError("RB2_BUNDLE_DESTINATION must be host:/absolute/path")
    host, root = destination.split(":", 1)
    if not host or not root.startswith("/"):
        raise ValueError("invalid RB2_BUNDLE_DESTINATION")
    remote_span = f"{root.rstrip('/')}/{span}"
    ready = load_json(bundle / readiness_name)
    checkpoint_sha = str(ready["checkpoint_sha256"])
    checkpoint_name = str(ready["checkpoint"])
    existing = subprocess.run(
        (
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "test",
            "-f",
            f"{remote_span}/{readiness_name}",
        ),
        check=False,
    ).returncode == 0
    if existing:
        observed = subprocess.check_output(
            (
                "ssh",
                "-o",
                "BatchMode=yes",
                host,
                "sha256sum",
                f"{remote_span}/{checkpoint_name}",
            ),
            text=True,
        ).split()[0]
        if observed != checkpoint_sha:
            raise FileExistsError(
                f"refusing incompatible rb2 {span} bundle: {remote_span}"
            )
        return {
            "verdict": f"RB2_{span.upper()}_BUNDLE_ALREADY_PRESENT",
            "destination": destination,
            "checkpoint_sha256": checkpoint_sha,
        }
    remote_exists = subprocess.run(
        ("ssh", "-o", "BatchMode=yes", host, "test", "-e", remote_span),
        check=False,
    ).returncode == 0
    if remote_exists:
        raise FileExistsError(
            f"refusing rb2 {span} destination without compatible readiness: {remote_span}"
        )
    remote_partial = f"{root.rstrip('/')}/.{span}.partial-{checkpoint_sha[:12]}"
    subprocess.run(
        (
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "mkdir",
            "-p",
            root,
            remote_partial,
        ),
        check=True,
    )
    subprocess.run(
        ("rsync", "-a", "--partial", f"{bundle}/", f"{host}:{remote_partial}/"),
        check=True,
    )
    observed = subprocess.check_output(
        (
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "sha256sum",
            f"{remote_partial}/{checkpoint_name}",
        ),
        text=True,
    ).split()[0]
    if observed != checkpoint_sha:
        raise RuntimeError("rb2 bundle transfer checksum mismatch")
    subprocess.run(
        ("ssh", "-o", "BatchMode=yes", host, "mv", remote_partial, remote_span),
        check=True,
    )
    observed = subprocess.check_output(
        (
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "sha256sum",
            f"{remote_span}/{checkpoint_name}",
        ),
        text=True,
    ).split()[0]
    if observed != checkpoint_sha:
        raise RuntimeError("rb2 final bundle checksum mismatch")
    return {
        "verdict": f"RB2_{span.upper()}_BUNDLE_PUBLISHED",
        "destination": destination,
        "checkpoint_sha256": checkpoint_sha,
    }


def _publish_short_bundle(bundle: Path) -> dict[str, Any]:
    return _publish_bundle(
        bundle,
        span="short_span",
        readiness_name="READY_SHORT_FOR_RB2.json",
    )


def _publish_long_bundle(bundle: Path) -> dict[str, Any]:
    return _publish_bundle(
        bundle,
        span="long_span",
        readiness_name="READY_KC8_FOR_RB2.json",
    )


def _publish_long_terminal(*, verdict: str, reason: str, stage: str) -> dict[str, Any]:
    local = TRANSFER_ROOT / "long_span/LONG_SPAN_TERMINAL.json"
    payload = {
        "schema_version": BUNDLE_SCHEMA,
        "verdict": str(verdict),
        "reason": str(reason),
        "stage": str(stage),
        "timestamp": _timestamp(),
        "kc8_offline_ready": False,
    }
    atomic_write_json(local, payload)
    destination = os.environ.get(
        "RB2_BUNDLE_DESTINATION",
        "rb2:/home/mingyujung/private/gnaroshi_vla_storage/incoming/"
        "simvla_stability_aligned_selected",
    )
    if ":" not in destination:
        raise ValueError("RB2_BUNDLE_DESTINATION must be host:/absolute/path")
    host, root = destination.split(":", 1)
    remote_span = f"{root.rstrip('/')}/long_span"
    if subprocess.run(
        (
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "test",
            "-f",
            f"{remote_span}/READY_KC8_FOR_RB2.json",
        ),
        check=False,
    ).returncode == 0:
        return {
            "verdict": "RB2_KC8_READY_ALREADY_PRESENT",
            "destination": destination,
        }
    subprocess.run(
        ("ssh", "-o", "BatchMode=yes", host, "mkdir", "-p", remote_span),
        check=True,
    )
    remote_partial = f"{remote_span}/.LONG_SPAN_TERMINAL.partial"
    subprocess.run(
        ("rsync", "-a", str(local), f"{host}:{remote_partial}"),
        check=True,
    )
    subprocess.run(
        (
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "mv",
            remote_partial,
            f"{remote_span}/LONG_SPAN_TERMINAL.json",
        ),
        check=True,
    )
    return {
        "verdict": "RB2_LONG_SPAN_TERMINAL_PUBLISHED",
        "destination": destination,
        "terminal": payload,
    }


def _wait_for_rb2_kc4_gate(
    *,
    checkpoint_sha256: str,
    poll_seconds: int = 120,
) -> dict[str, Any]:
    remote = os.environ.get(
        "RB2_SELECTION_REMOTE",
        "rb2:/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/"
        "stability_alignment/condition_only_rb2_v2/final_simvla_method_selection.json",
    )
    if ":" not in remote:
        raise ValueError("RB2_SELECTION_REMOTE must be host:/absolute/file")
    host, path = remote.split(":", 1)
    if not host or not path.startswith("/"):
        raise ValueError("invalid RB2_SELECTION_REMOTE")
    timeout_hours = float(os.environ.get("SIMVLA_RB2_GATE_TIMEOUT_HOURS", "168"))
    deadline = time.monotonic() + 3600.0 * timeout_hours
    while True:
        completed = subprocess.run(
            ("ssh", "-o", "BatchMode=yes", host, "cat", path),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode == 0:
            payload = json.loads(completed.stdout)
            if payload.get("manifest_sha256") != (
                "9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48"
            ):
                raise RuntimeError("rb2 K_C=4 gate uses another episode manifest")
            observed_sha = payload.get("checkpoint_sha256_by_k_c", {}).get("4")
            if observed_sha != checkpoint_sha256:
                raise RuntimeError("rb2 K_C=4 gate uses another selected checkpoint")
            acceptance = payload.get("learned_ng3_acceptance", {}).get("4")
            if not isinstance(acceptance, dict):
                raise RuntimeError("rb2 result lacks learned K_C=4,N_G=3 acceptance")
            return {
                "verdict": (
                    "RB2_KC4_ONLINE_GATE_PASS"
                    if acceptance.get("passed")
                    else "RB2_KC4_ONLINE_GATE_FAIL"
                ),
                "passed": bool(acceptance.get("passed")),
                "remote": remote,
                "checkpoint_sha256": observed_sha,
                "acceptance": acceptance,
            }
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for rb2 K_C=4 gate: {remote}")
        print(
            f"[{_timestamp()}] rb2 K_C=4 online result not ready; "
            f"waiting {int(poll_seconds)}s without GPU use",
            flush=True,
        )
        time.sleep(int(poll_seconds))


def _final_report(state: AtomicStageState, error: str | None = None) -> None:
    selected_path = RESULT_ROOT / "selected_short_span_checkpoint.json"
    selected = load_json(selected_path) if selected_path.is_file() else None
    long_path = RESULT_ROOT / "selected_long_span_checkpoint.json"
    selected_long = load_json(long_path) if long_path.is_file() else None
    lines = [
        "# SimVLA Stability Alignment Campaign",
        "",
        f"- timestamp: `{_timestamp()}`",
        f"- hostname: `{socket.gethostname()}`",
        f"- worktree: `{WORKTREE}`",
        f"- result root: `{RESULT_ROOT}`",
        "- N_G=3: retained as the validated Generation Loop component",
        "- naive NFE=3: mandatory mechanism control; it does not invalidate N_G=3",
        "- active jobs altered: no",
        "- additional inference seeds launched: no",
        "- git add/commit/push: none",
        f"- selected short-span branch: `{selected.get('selected') if selected else 'PENDING'}`",
        f"- selected long-span checkpoint: `{selected_long.get('verdict') if selected_long else 'CONDITIONAL_PENDING'}`",
        f"- pipeline error: `{error or 'none'}`",
        "",
        "## Stage State",
        "",
        "```json",
        json.dumps(state.payload, indent=2, sort_keys=True),
        "```",
    ]
    (RESULT_ROOT / "final_simvla_stability_alignment_campaign_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    state = AtomicStageState(RESULT_ROOT / "pipeline_state.json", SD1_STAGES)
    lock_handle = (RESULT_ROOT / "pipeline.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another sd1 stability pipeline is already active")
    current_stage = "S0"
    try:
        state.set("S0", "RUNNING", started=_timestamp())
        audit = _active_audit()
        atomic_write_json(RESULT_ROOT / "active_process_audit.json", audit)
        if audit["other_processes_using_worktree"]:
            raise RuntimeError("dedicated worktree is already used by another process")
        state.set("S0", "PASSED", finished=_timestamp())

        current_stage = "S1"
        state.require_passed("S0")
        state.set("S1", "RUNNING", started=_timestamp())
        for path in (*CONDITION.values(), GENERATION, NORM, CACHE / "manifest.json"):
            if not path.is_file():
                raise FileNotFoundError(path)
        cache = validate_exact_cache(CACHE, verify_checksums=False)
        if not cache["passed"] or int(cache["windows"]) != 6_525:
            raise RuntimeError(f"exact cache validation failed: {cache}")
        atomic_write_json(RESULT_ROOT / "exact_cache_validation.json", cache)
        state.set("S1", "PASSED", finished=_timestamp())

        current_stage = "S2"
        state.require_passed("S1")
        state.set("S2", "RUNNING", started=_timestamp())
        calibration_jobs = []
        for index, branch in enumerate(SHORT_BRANCHES):
            output = RESULT_ROOT / branch.lower() / "calibration"
            calibration_jobs.append(
                Job(
                    f"{branch.lower()}_calibration",
                    _torchrun("calibrate", _common(branch, output), 29620 + index),
                    output / "stability_alignment_loss_weights.json",
                    ("STABILITY_LOSS_CALIBRATION_COMPLETE",),
                    RESULT_ROOT / "logs" / f"{branch.lower()}_calibration.log",
                )
            )
        _run_jobs(calibration_jobs, args.gpu_pool, RESULT_ROOT / "process_state" / "S2")
        common_weights = RESULT_ROOT / "stability_alignment_loss_weights.json"
        subprocess.run(
            [
                str(PYTHON),
                "-m",
                MODULE,
                "merge-calibrations",
                "--s50",
                str(calibration_jobs[0].success_file),
                "--s150",
                str(calibration_jobs[1].success_file),
                "--output",
                str(common_weights),
            ],
            cwd=WORKTREE,
            env={**os.environ, "PYTHONPATH": f"{WORKTREE}:{UPSTREAM}:{os.environ.get('PYTHONPATH','')}"},
            check=True,
        )
        state.set("S2", "PASSED", finished=_timestamp())

        current_stage = "S3"
        state.require_passed("S2")
        state.set("S3", "RUNNING", started=_timestamp())
        benchmark_jobs = []
        for index, branch in enumerate(SHORT_BRANCHES):
            output = RESULT_ROOT / branch.lower() / "benchmark_500"
            command = _common(branch, output) + [
                "--loss-weights",
                str(common_weights),
                "--stop-step",
                "500",
                "--benchmark-steps",
                "500",
                "--practical-budget-hours",
                str(args.practical_budget_hours),
            ]
            benchmark_jobs.append(
                Job(
                    f"{branch.lower()}_benchmark",
                    _torchrun("benchmark", command, 29630 + index),
                    output / "speed_benchmark.json",
                    ("STABILITY_500_STEP_BENCHMARK_COMPLETE",),
                    RESULT_ROOT / "logs" / f"{branch.lower()}_benchmark.log",
                )
            )
        _run_jobs(benchmark_jobs, args.gpu_pool, RESULT_ROOT / "process_state" / "S3")
        projections = [load_json(job.success_file) for job in benchmark_jobs]
        if not all(value["within_practical_budget"] for value in projections):
            raise RuntimeError(f"projected 10K exceeds practical budget: {projections}")
        if not all(
            value.get("numerical_stability", {}).get("passed")
            for value in projections
        ):
            raise RuntimeError(
                f"500-step numerical stability gate failed: {projections}"
            )
        atomic_write_json(
            RESULT_ROOT / "projected_10k_time.json",
            {"S50": projections[0], "S150": projections[1]},
        )
        state.set("S3", "PASSED", finished=_timestamp())

        current_stage = "S4"
        state.require_passed("S3")
        state.set("S4", "RUNNING", started=_timestamp())
        train_2k_jobs = []
        for index, branch in enumerate(SHORT_BRANCHES):
            output = RESULT_ROOT / branch.lower() / "train"
            command = _common(branch, output) + [
                "--loss-weights",
                str(common_weights),
                "--stop-step",
                "2000",
                "--save-interval",
                "2000",
                "--wandb-project",
                "gnaroshi-simvla-stability-alignment",
                "--wandb-name",
                f"simvla_stability_{branch.lower()}",
            ]
            train_2k_jobs.append(
                Job(
                    f"{branch.lower()}_train_2k",
                    _torchrun("train", command, 29640 + index),
                    output / "run_summary.json",
                    ("STABILITY_TRAINING_SEGMENT_COMPLETE",),
                    RESULT_ROOT / "logs" / f"{branch.lower()}_train_2k.log",
                    {"optimizer_step": 2_000},
                )
            )
        _run_jobs(train_2k_jobs, args.gpu_pool, RESULT_ROOT / "process_state" / "S4")
        state.set("S4", "PASSED", finished=_timestamp())

        current_stage = "S5"
        state.require_passed("S4")
        state.set("S5", "RUNNING", started=_timestamp())
        gate_2k_jobs = []
        for index, branch in enumerate(SHORT_BRANCHES):
            output = RESULT_ROOT / branch.lower() / "offline_2k"
            checkpoint = RESULT_ROOT / branch.lower() / "train/checkpoints/stability_step_002000.pt"
            command = _common(branch, output) + [
                "--candidate",
                str(checkpoint),
                "--offline-split",
                "checkpoint_validation",
            ]
            gate_2k_jobs.append(
                Job(
                    f"{branch.lower()}_offline_2k",
                    _torchrun("offline", command, 29650 + index),
                    output / "offline_gate.json",
                    TWO_K_TERMINAL_VERDICTS,
                    RESULT_ROOT / "logs" / f"{branch.lower()}_offline_2k.log",
                )
            )
        _run_jobs(gate_2k_jobs, args.gpu_pool, RESULT_ROOT / "process_state" / "S5")
        two_k_decision = _gate_branch_decision(
            {
                branch: job.success_file
                for branch, job in zip(SHORT_BRANCHES, gate_2k_jobs)
            },
            pass_verdict="STABILITY_2K_GATE_PASS",
            fail_verdict="STABILITY_2K_GATE_FAIL",
        )
        atomic_write_json(RESULT_ROOT / "s5_branch_decision.json", two_k_decision)
        surviving_branches = tuple(two_k_decision["continuing_branches"])
        if not surviving_branches:
            raise RuntimeError("no S50/S150 branch passed the 2K safety gate")
        state.set(
            "S5",
            "PASSED",
            finished=_timestamp(),
            continuing_branches=list(surviving_branches),
            trend_warning_branches=two_k_decision["trend_warning_branches"],
            stopped_branches=two_k_decision["stopped_branches"],
        )

        current_stage = "S6"
        state.require_passed("S5")
        state.set("S6", "RUNNING", started=_timestamp())
        train_10k_jobs = []
        for index, branch in enumerate(surviving_branches):
            output = RESULT_ROOT / branch.lower() / "train"
            checkpoint = output / "checkpoints/stability_step_002000.pt"
            gate = RESULT_ROOT / branch.lower() / "offline_2k/offline_gate.json"
            command = _common(branch, output) + [
                "--loss-weights",
                str(common_weights),
                "--resume",
                str(checkpoint),
                "--safety-gate",
                str(gate),
                "--stop-step",
                "10000",
                "--save-interval",
                "2000",
                "--wandb-project",
                "gnaroshi-simvla-stability-alignment",
                "--wandb-name",
                f"simvla_stability_{branch.lower()}",
            ]
            train_10k_jobs.append(
                Job(
                    f"{branch.lower()}_train_10k",
                    _torchrun("train", command, 29660 + index),
                    output / "run_summary.json",
                    ("STABILITY_TRAINING_SEGMENT_COMPLETE",),
                    RESULT_ROOT / "logs" / f"{branch.lower()}_train_10k.log",
                    {"optimizer_step": 10_000},
                )
            )
        _run_jobs(train_10k_jobs, args.gpu_pool, RESULT_ROOT / "process_state" / "S6")
        state.set("S6", "PASSED", finished=_timestamp())

        current_stage = "S7"
        state.require_passed("S6")
        state.set("S7", "RUNNING", started=_timestamp())
        gate_10k_jobs = []
        for index, branch in enumerate(surviving_branches):
            output = RESULT_ROOT / branch.lower() / "offline_10k"
            checkpoint = RESULT_ROOT / branch.lower() / "train/checkpoints/stability_step_010000.pt"
            command = _common(branch, output) + [
                "--candidate",
                str(checkpoint),
                "--offline-split",
                "final_offline",
            ]
            gate_10k_jobs.append(
                Job(
                    f"{branch.lower()}_offline_10k",
                    _torchrun("offline", command, 29670 + index),
                    output / "offline_gate.json",
                    TEN_K_TERMINAL_VERDICTS,
                    RESULT_ROOT / "logs" / f"{branch.lower()}_offline_10k.log",
                )
            )
        _run_jobs(gate_10k_jobs, args.gpu_pool, RESULT_ROOT / "process_state" / "S7")
        offline_10k = {
            branch: load_json(job.success_file)
            for branch, job in zip(surviving_branches, gate_10k_jobs)
        }
        summaries = [
            _branch_summary(
                branch,
                offline_10k[branch],
                RESULT_ROOT / branch.lower() / "train/checkpoints/stability_step_010000.pt",
            )
            for branch in surviving_branches
        ]
        _write_comparison(summaries)
        selection = _select_short_span_summary(summaries)
        if selection["selected"] is None:
            raise RuntimeError("no surviving S50/S150 branch passed the 10K gate")
        selected = str(selection["selected"])
        summary_by_branch = {str(summary["branch"]): summary for summary in summaries}
        selected_summary = summary_by_branch[selected]
        selection.update(
            {
                "checkpoint": selected_summary["checkpoint"],
                "checkpoint_sha256": selected_summary["checkpoint_sha256"],
                "S50": summary_by_branch.get(
                    "S50",
                    {
                        "branch": "S50",
                        "gate_passed": False,
                        "status": "DROPPED_AT_2K",
                        "offline_2k": two_k_decision["branches"]["S50"],
                    },
                ),
                "S150": summary_by_branch.get(
                    "S150",
                    {
                        "branch": "S150",
                        "gate_passed": False,
                        "status": "DROPPED_AT_2K",
                        "offline_2k": two_k_decision["branches"]["S150"],
                    },
                ),
            }
        )
        atomic_write_json(RESULT_ROOT / "selected_short_span_checkpoint.json", selection)
        selected_offline = offline_10k[selected]
        primary_checkpoint = Path(selected_summary["checkpoint"])
        primary_offline_path = (
            RESULT_ROOT / selected.lower() / "offline_10k/offline_gate.json"
        )
        bundle = _export_bundle(
            selected=selected,
            checkpoint=primary_checkpoint,
            offline=selected_offline,
            offline_path=primary_offline_path,
            common_weights=common_weights,
        )
        publication = _publish_short_bundle(bundle)
        atomic_write_json(RESULT_ROOT / "rb2_short_bundle_publication.json", publication)
        state.set(
            "S7",
            "PASSED",
            finished=_timestamp(),
            selected=selected,
            primary_optimizer_step=10_000,
            bundle=str(bundle),
            rb2_publication=publication,
        )

        current_stage = "S8"
        state.require_passed("S7")
        state.set("S8", "RUNNING", started=_timestamp())
        final_checkpoint = primary_checkpoint
        continuation_attempted = False
        if _continuation_allowed(selected_offline):
            continuation_attempted = True
            output = RESULT_ROOT / selected.lower() / "train"
            command = _common(selected, output) + [
                "--loss-weights",
                str(common_weights),
                "--resume",
                str(primary_checkpoint),
                "--stop-step",
                "30000",
                "--save-interval",
                "5000",
                "--wandb-project",
                "gnaroshi-simvla-stability-alignment",
                "--wandb-name",
                f"simvla_stability_{selected.lower()}",
            ]
            continuation = Job(
                f"{selected.lower()}_train_30k",
                _torchrun("train", command, 29680),
                output / "run_summary.json",
                ("STABILITY_TRAINING_SEGMENT_COMPLETE",),
                RESULT_ROOT / "logs" / f"{selected.lower()}_train_30k.log",
                {"optimizer_step": 30_000},
            )
            _run_jobs([continuation], args.gpu_pool, RESULT_ROOT / "process_state" / "S8")
            final_checkpoint = output / "checkpoints/stability_step_030000.pt"
            state.set("S8", "PASSED", finished=_timestamp(), continued_to_30k=True)
        else:
            state.set("S8", "SKIPPED", finished=_timestamp(), reason="10K continuation criteria not met")

        current_stage = "S9"
        state.require_passed("S7")
        state.set("S9", "RUNNING", started=_timestamp())
        final_output = RESULT_ROOT / selected.lower() / "offline_final"
        final_job = Job(
            f"{selected.lower()}_offline_final",
            _torchrun(
                "offline",
                _common(selected, final_output)
                + ["--candidate", str(final_checkpoint), "--offline-split", "final_offline"],
                29690,
            ),
            final_output / "offline_gate.json",
            TEN_K_TERMINAL_VERDICTS,
            RESULT_ROOT / "logs" / f"{selected.lower()}_offline_final.log",
        )
        _run_jobs([final_job], args.gpu_pool, RESULT_ROOT / "process_state" / "S9")
        evaluated_final_offline = load_json(final_job.success_file)
        final_metrics = final_output / "recursive_stability_metrics.csv"
        retained_step = int(evaluated_final_offline["optimizer_step"])
        if not evaluated_final_offline["gate"]["passed"]:
            if not continuation_attempted:
                raise RuntimeError("selected 10K checkpoint failed its deterministic final recheck")
            atomic_write_json(
                RESULT_ROOT / "rejected_30k_continuation.json",
                {
                    "verdict": "STABILITY_30K_REJECTED_10K_RETAINED",
                    "rejected_checkpoint": str(final_checkpoint),
                    "rejected_checkpoint_sha256": sha256_file(final_checkpoint),
                    "rejected_offline_gate": str(final_job.success_file),
                    "retained_checkpoint": str(primary_checkpoint),
                    "retained_checkpoint_sha256": sha256_file(primary_checkpoint),
                    "retained_offline_gate": str(
                        RESULT_ROOT / selected.lower() / "offline_10k/offline_gate.json"
                    ),
                },
            )
            final_checkpoint = primary_checkpoint
            final_offline = selected_offline
            final_metrics = (
                RESULT_ROOT
                / selected.lower()
                / "offline_10k/recursive_stability_metrics.csv"
            )
            retained_step = 10_000
        else:
            final_offline = evaluated_final_offline
        shutil.copy2(
            final_metrics,
            RESULT_ROOT / "recursive_stability_metrics.csv",
        )
        selection = load_json(RESULT_ROOT / "selected_short_span_checkpoint.json")
        selection.update(
            {
                "final_checkpoint": str(primary_checkpoint),
                "final_checkpoint_sha256": sha256_file(primary_checkpoint),
                "final_optimizer_step": 10_000,
                "kc3_readiness": selected_offline["readiness"].get("kc3"),
                "kc4_readiness": selected_offline["readiness"].get("kc4"),
                "optional_continuation_attempted": continuation_attempted,
                "optional_continuation_checkpoint": str(final_checkpoint),
                "optional_continuation_checkpoint_sha256": sha256_file(final_checkpoint),
                "optional_continuation_optimizer_step": final_offline["optimizer_step"],
                "optional_continuation_gate_passed": bool(final_offline["gate"]["passed"]),
            }
        )
        atomic_write_json(RESULT_ROOT / "selected_short_span_checkpoint.json", selection)
        state.set(
            "S9",
            "PASSED",
            finished=_timestamp(),
            retained_step=retained_step,
            continuation_attempted=continuation_attempted,
        )

        current_stage = "S10"
        state.require_passed("S9")
        state.set("S10", "RUNNING", started=_timestamp())
        bundle = _export_bundle(
            selected=selected,
            checkpoint=primary_checkpoint,
            offline=selected_offline,
            offline_path=primary_offline_path,
            common_weights=common_weights,
        )
        publication = _publish_short_bundle(bundle)
        atomic_write_json(RESULT_ROOT / "rb2_short_bundle_publication.json", publication)
        state.set(
            "S10",
            "PASSED",
            finished=_timestamp(),
            bundle=str(bundle),
            rb2_publication=publication,
        )

        current_stage = "S11"
        if not selected_offline["readiness"].get("kc4", {}).get("passed"):
            state.set("S11", "SKIPPED", reason="K_C=4 offline gate did not pass")
            state.set("S12", "SKIPPED", reason="K_C=4 offline prerequisite failed")
            state.set("S13", "SKIPPED", reason="no KC8_OFFLINE_READY checkpoint")
            state.set("S14", "PASSED", finished=_timestamp())
            _final_report(state)
            return 0
        state.set("S11", "RUNNING", started=_timestamp())
        rb2_gate = _wait_for_rb2_kc4_gate(
            checkpoint_sha256=sha256_file(primary_checkpoint)
        )
        atomic_write_json(RESULT_ROOT / "rb2_kc4_online_gate.json", rb2_gate)
        if not rb2_gate["passed"]:
            state.set("S11", "BLOCKED", reason="K_C=4 learned N_G=3 online gate failed")
            state.set("S12", "SKIPPED", reason="K_C=4 online prerequisite failed")
            state.set("S13", "SKIPPED", reason="no KC8_OFFLINE_READY checkpoint")
            state.set("S14", "PASSED", finished=_timestamp())
            _final_report(state)
            return 0
        catalog = RESULT_ROOT / "long_span/q0_q7_catalog.json"
        subprocess.run(
            (
                str(PYTHON),
                "-m",
                LONG_MODULE,
                "catalog",
                "--output",
                str(catalog),
                "--cache",
                str(CACHE),
                "--split-seed",
                "20260822",
            ),
            cwd=WORKTREE,
            env={
                **os.environ,
                "PYTHONPATH": f"{WORKTREE}:{UPSTREAM}:{os.environ.get('PYTHONPATH', '')}",
            },
            check=True,
        )
        catalog_payload = load_json(catalog)
        if (
            catalog_payload.get("verdict") != "Q0_Q7_INDEX_READY"
            or catalog_payload.get("query_tensors_duplicated") is not False
        ):
            raise RuntimeError("q0-q7 catalog contract failed")
        state.set(
            "S11",
            "PASSED",
            finished=_timestamp(),
            sequences=int(catalog_payload["sequences"]),
        )

        current_stage = "S12"
        state.require_passed("S11")
        state.set("S12", "RUNNING", started=_timestamp())
        long_train = RESULT_ROOT / "long_span/train"
        long_common = [
            "--cache",
            str(CACHE),
            "--short-parent",
            str(primary_checkpoint),
            "--norm-stats",
            str(NORM),
            "--split-seed",
            "20260822",
            "--seed",
            "20260825",
            "--num-workers",
            "2",
        ]
        train_10k = Job(
            "long_span_train_10k",
            _torchrun(
                "train",
                [
                    "--output",
                    str(long_train),
                    *long_common,
                    "--loss-weights",
                    str(common_weights),
                    "--stop-step",
                    "10000",
                    "--save-interval",
                    "2000",
                ],
                29700,
                module=LONG_MODULE,
            ),
            long_train / "run_summary.json",
            ("STABILITY_LONG_TRAINING_SEGMENT_COMPLETE",),
            RESULT_ROOT / "logs/long_span_train_10k.log",
            {"optimizer_step": 10_000},
        )
        _run_jobs([train_10k], args.gpu_pool, RESULT_ROOT / "process_state/S12_train_10k")
        checkpoint_10k = long_train / "checkpoints/stability_long_step_010000.pt"
        offline_10k_output = RESULT_ROOT / "long_span/offline_10k"
        offline_10k_job = Job(
            "long_span_offline_10k",
            _torchrun(
                "offline",
                [
                    "--output",
                    str(offline_10k_output),
                    *long_common,
                    "--candidate",
                    str(checkpoint_10k),
                ],
                29710,
                module=LONG_MODULE,
            ),
            offline_10k_output / "offline_gate.json",
            ("KC8_OFFLINE_READY", "KC8_OFFLINE_BLOCKED"),
            RESULT_ROOT / "logs/long_span_offline_10k.log",
        )
        _run_jobs([offline_10k_job], args.gpu_pool, RESULT_ROOT / "process_state/S12_offline_10k")
        offline_10k = load_json(offline_10k_job.success_file)
        if not offline_10k.get("passed"):
            atomic_write_json(
                RESULT_ROOT / "selected_long_span_checkpoint.json",
                {
                    "verdict": "NO_KC8_OFFLINE_READY_CHECKPOINT",
                    "checkpoint_10k": str(checkpoint_10k),
                    "checkpoint_10k_sha256": sha256_file(checkpoint_10k),
                    "offline_gate": str(offline_10k_job.success_file),
                },
            )
            terminal = _publish_long_terminal(
                verdict="KC8_OFFLINE_BLOCKED",
                reason="10K long-span offline gate failed",
                stage="S12",
            )
            atomic_write_json(RESULT_ROOT / "rb2_long_terminal_publication.json", terminal)
            state.set("S12", "BLOCKED", reason="10K long-span offline gate failed")
            state.set("S13", "SKIPPED", reason="no KC8_OFFLINE_READY checkpoint")
            state.set("S14", "PASSED", finished=_timestamp())
            _final_report(state)
            return 0

        long_checkpoint = checkpoint_10k
        long_offline_path = offline_10k_job.success_file
        selected_step = 10_000
        continuation_attempted = False
        continuation_retained = False
        if offline_10k.get("continuation_to_30k_allowed"):
            continuation_attempted = True
            train_30k = Job(
                "long_span_train_30k",
                _torchrun(
                    "train",
                    [
                        "--output",
                        str(long_train),
                        *long_common,
                        "--loss-weights",
                        str(common_weights),
                        "--resume",
                        str(checkpoint_10k),
                        "--stop-step",
                        "30000",
                        "--save-interval",
                        "5000",
                    ],
                    29720,
                    module=LONG_MODULE,
                ),
                long_train / "run_summary.json",
                ("STABILITY_LONG_TRAINING_SEGMENT_COMPLETE",),
                RESULT_ROOT / "logs/long_span_train_30k.log",
                {"optimizer_step": 30_000},
            )
            _run_jobs([train_30k], args.gpu_pool, RESULT_ROOT / "process_state/S12_train_30k")
            checkpoint_30k = long_train / "checkpoints/stability_long_step_030000.pt"
            offline_30k_output = RESULT_ROOT / "long_span/offline_30k"
            offline_30k_job = Job(
                "long_span_offline_30k",
                _torchrun(
                    "offline",
                    [
                        "--output",
                        str(offline_30k_output),
                        *long_common,
                        "--candidate",
                        str(checkpoint_30k),
                    ],
                    29730,
                    module=LONG_MODULE,
                ),
                offline_30k_output / "offline_gate.json",
                ("KC8_OFFLINE_READY", "KC8_OFFLINE_BLOCKED"),
                RESULT_ROOT / "logs/long_span_offline_30k.log",
            )
            _run_jobs([offline_30k_job], args.gpu_pool, RESULT_ROOT / "process_state/S12_offline_30k")
            offline_30k = load_json(offline_30k_job.success_file)
            if offline_30k.get("passed"):
                long_checkpoint = checkpoint_30k
                long_offline_path = offline_30k_job.success_file
                selected_step = 30_000
                continuation_retained = True
        long_selection = {
            "verdict": f"KC8_{selected_step // 1000}K_SELECTED",
            "checkpoint": str(long_checkpoint),
            "checkpoint_sha256": sha256_file(long_checkpoint),
            "optimizer_step": selected_step,
            "short_parent": str(primary_checkpoint),
            "short_parent_sha256": sha256_file(primary_checkpoint),
            "offline_gate": str(long_offline_path),
            "continuation_attempted": continuation_attempted,
            "continuation_retained": continuation_retained,
            "selection_basis": "offline age-1..7 gate only; no online SR used",
        }
        atomic_write_json(RESULT_ROOT / "selected_long_span_checkpoint.json", long_selection)
        state.set(
            "S12",
            "PASSED",
            finished=_timestamp(),
            selected_step=selected_step,
            checkpoint_sha256=long_selection["checkpoint_sha256"],
        )

        current_stage = "S13"
        state.require_passed("S12")
        state.set("S13", "RUNNING", started=_timestamp())
        long_bundle = _export_long_bundle(
            checkpoint=long_checkpoint,
            offline_path=long_offline_path,
            common_weights=common_weights,
            catalog=catalog,
            short_selection=RESULT_ROOT / "selected_short_span_checkpoint.json",
        )
        long_publication = _publish_long_bundle(long_bundle)
        atomic_write_json(
            RESULT_ROOT / "rb2_long_bundle_publication.json", long_publication
        )
        state.set(
            "S13",
            "PASSED",
            finished=_timestamp(),
            bundle=str(long_bundle),
            rb2_publication=long_publication,
        )
        state.set("S14", "PASSED", finished=_timestamp())
        _final_report(state)
        return 0
    except Exception as exc:
        if current_stage in {"S11", "S12", "S13"}:
            gate_path = RESULT_ROOT / "rb2_kc4_online_gate.json"
            try:
                gate_payload = load_json(gate_path) if gate_path.is_file() else {}
                if gate_payload.get("passed"):
                    terminal = _publish_long_terminal(
                        verdict="KC8_PIPELINE_FAILED",
                        reason=f"{type(exc).__name__}: {exc}",
                        stage=current_stage,
                    )
                    atomic_write_json(
                        RESULT_ROOT / "rb2_long_terminal_publication.json", terminal
                    )
            except Exception as terminal_exc:
                atomic_write_json(
                    RESULT_ROOT / "rb2_long_terminal_publication_failed.json",
                    {
                        "verdict": "RB2_LONG_TERMINAL_PUBLICATION_FAILED",
                        "original_error": f"{type(exc).__name__}: {exc}",
                        "terminal_error": f"{type(terminal_exc).__name__}: {terminal_exc}",
                    },
                )
        try:
            state.set(current_stage, "FAILED", finished=_timestamp(), error=str(exc))
        except Exception:
            pass
        _final_report(state, error=f"{type(exc).__name__}: {exc}")
        (RESULT_ROOT / "pipeline_traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        print(f"SD1_STABILITY_PIPELINE_FAILED stage={current_stage}: {exc}", file=sys.stderr)
        return 1
    finally:
        lock_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu-pool",
        type=_parse_pool,
        default=_parse_pool(os.environ.get("SD1_GPU_POOL", "4,5,6,7")),
    )
    parser.add_argument("--practical-budget-hours", type=float, default=12.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
