"""Sequential rb2 frontier controls and selected-checkpoint EGL evaluation."""

from __future__ import annotations

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
from pathlib import Path
from typing import Any

from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    BUNDLE_SCHEMA,
    AtomicStageState,
    atomic_write_json,
    load_json,
    sha256_file,
)


WORKTREE = Path(
    os.environ.get(
        "SIMVLA_STABILITY_WORKTREE",
        "/home/mingyujung/private/gnaroshi_vla_worktrees/simvla_stability_alignment",
    )
).resolve()
PYTHON = Path(
    "/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/"
    "libero_mujoco237/bin/python"
).resolve()
UPSTREAM = Path(
    "/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream"
).resolve()
STORAGE = Path("/home/mingyujung/private/gnaroshi_vla_storage").resolve()
RESULT_ROOT = STORAGE / "results/simvla/stability_alignment/condition_only_rb2_v2"
IMPORT_ROOT = STORAGE / "artifacts/simvla/stability_aligned_selected"
COMPLETED_FRONTIER_EVIDENCE = (
    STORAGE / "artifacts/simvla/stability_alignment/completed_frontier_sd1_seed02"
)
COMPLETED_FRONTIER_LOCK = (
    WORKTREE
    / "architectures/simvla/adapters/latentloop/stability_alignment/"
    "completed_frontier_evidence.json"
)
MANIFEST = (
    STORAGE
    / "results/simvla/latentloop/generation_loop_ng2_rb2_v1/online/"
    "step_030000_long500_egl_seed02_v1/episode_manifest.json"
)
MANIFEST_SHA256 = "9e652bf2027652717b409bb25b4c9bcf8fcfbd3791560b2c81137c0c3daaca48"
BASELINE = (
    STORAGE
    / "results/simvla/latentloop/generation_loop_ng2_rb2_v1/online/"
    "step_030000_long500_egl_seed02_v1/baseline_k1/row_summary.json"
)
PARENT_BASELINES = {
    "baseline_k1": BASELINE,
    "condition_kc2_ng10": (
        STORAGE
        / "results/simvla/fixed_2x2/kc2_ng3_seed02_v1/"
        "condition_kc2_ng10/merged/row_summary.json"
    ),
    "condition_kc2_ng3": (
        STORAGE
        / "results/simvla/fixed_2x2/kc2_ng3_seed02_v1/"
        "condition_kc2_ng3/merged/row_summary.json"
    ),
    "condition_kc3_ng3": (
        STORAGE
        / "results/simvla/coupled_condition_generation/"
        "kc3_ng3_real_cj_projection10k_seed02_v1/online/"
        "condition_kc3_ng3/merged/row_summary.json"
    ),
}
SELECTED_EVALUATOR = (
    "architectures.simvla.adapters.latentloop.stability_alignment.online_eval"
)
RB2_STAGES = tuple(f"R{index}" for index in range(13))


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _gpu_snapshot() -> dict[str, Any]:
    row = subprocess.check_output(
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ),
        text=True,
    ).strip()
    applications = subprocess.run(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ),
        text=True,
        capture_output=True,
        check=False,
    ).stdout.splitlines()
    index, name, total, used, utilization = [value.strip() for value in row.split(",")]
    return {
        "index": int(index),
        "name": name,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "utilization_percent": int(utilization),
        "compute_processes": applications,
        "free": not applications and int(used) <= 1024 and int(utilization) <= 5,
    }


def _audit() -> dict[str, Any]:
    tmux = subprocess.run(
        ("tmux", "list-sessions", "-F", "#{session_name}|#{session_windows}|#{session_attached}"),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "hostname": socket.gethostname(),
        "timestamp": _timestamp(),
        "gpu": _gpu_snapshot(),
        "tmux_sessions": tmux.stdout.splitlines(),
        "source_worktree": str(WORKTREE),
        "source_head": subprocess.check_output(
            ("git", "-C", str(WORKTREE), "rev-parse", "HEAD"), text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ("git", "-C", str(WORKTREE), "status", "--short"), text=True
        ).splitlines(),
        "renderer": "egl",
        "active_jobs_altered": False,
    }


def _wait_for_gpu() -> None:
    while True:
        snapshot = _gpu_snapshot()
        if snapshot["free"]:
            return
        print(
            f"[{_timestamp()}] rb2 GPU busy; processes={snapshot['compute_processes']}; waiting 60s",
            flush=True,
        )
        time.sleep(60)


def _environment() -> dict[str, str]:
    manifest = load_json(MANIFEST)
    renderer = manifest["renderer"]
    environment = dict(os.environ)
    environment.update(
        {
            "SIMVLA_UPSTREAM_ROOT": str(UPSTREAM),
            "SIMVLA_LIBERO_ROOT": str(STORAGE / "datasets/LIBERO"),
            "LIBERO_CONFIG_PATH": str(
                STORAGE
                / "results/simvla/reproduction/official_ckpt_mujoco237_official_norm_seed7_n50_r2/"
                "runtime/libero_config"
            ),
            "PYTHONPATH": ":".join(
                (
                    str(WORKTREE),
                    str(UPSTREAM),
                    str(STORAGE / "datasets/LIBERO"),
                    environment.get("PYTHONPATH", ""),
                )
            ),
            "HF_HOME": str(STORAGE / "cache/simvla/huggingface"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": "0",
            "CUDA_VISIBLE_DEVICES": "0",
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "CUBLAS_WORKSPACE_CONFIG": str(renderer["CUBLAS_WORKSPACE_CONFIG"]),
            "CUDA_DEVICE_MAX_CONNECTIONS": str(renderer["CUDA_DEVICE_MAX_CONNECTIONS"]),
            "PYTHONHASHSEED": str(renderer["PYTHONHASHSEED"]),
            "SIMVLA_RENDER_AXIS": str(renderer["SIMVLA_RENDER_AXIS"]),
        }
    )
    environment.pop("GALLIUM_DRIVER", None)
    environment.pop("LIBGL_ALWAYS_SOFTWARE", None)
    return environment


def _remote_parts(remote: str) -> tuple[str, str]:
    if ":" not in remote:
        raise ValueError("SD1_BUNDLE_REMOTE must be host:/absolute/path")
    host, path = remote.split(":", 1)
    if not host or not path.startswith("/"):
        raise ValueError("invalid SD1_BUNDLE_REMOTE")
    return host, path.rstrip("/")


def remote_ready(remote: str, readiness: str = "READY_SHORT_FOR_RB2.json") -> bool:
    span = "long_span" if readiness.startswith("READY_KC8") else "short_span"
    if remote.startswith("/"):
        return (Path(remote) / span / readiness).is_file()
    host, path = _remote_parts(remote)
    completed = subprocess.run(
        ("ssh", "-o", "BatchMode=yes", host, "test", "-f", f"{path}/{span}/{readiness}"),
        check=False,
    )
    return completed.returncode == 0


def remote_payload(
    remote: str,
    *,
    span: str,
    filename: str,
) -> dict[str, Any] | None:
    if remote.startswith("/"):
        path = Path(remote) / span / filename
        return load_json(path) if path.is_file() else None
    host, root = _remote_parts(remote)
    completed = subprocess.run(
        ("ssh", "-o", "BatchMode=yes", host, "cat", f"{root}/{span}/{filename}"),
        text=True,
        capture_output=True,
        check=False,
    )
    return json.loads(completed.stdout) if completed.returncode == 0 else None


def _validate_bundle(bundle: Path, readiness_name: str) -> dict[str, Any]:
    ready = load_json(bundle / readiness_name)
    expected_verdict = (
        "READY_KC8_FOR_RB2" if readiness_name.startswith("READY_KC8") else "READY_SHORT_FOR_RB2"
    )
    if ready.get("schema_version") != BUNDLE_SCHEMA or ready.get("verdict") != expected_verdict:
        raise RuntimeError("imported bundle readiness contract failed")
    manifest = load_json(bundle / "SHA256_MANIFEST.json")
    failures = {
        relative: {
            "expected": digest,
            "observed": sha256_file(bundle / relative) if (bundle / relative).is_file() else None,
        }
        for relative, digest in manifest.items()
        if not (bundle / relative).is_file() or sha256_file(bundle / relative) != digest
    }
    if failures:
        raise RuntimeError(f"imported bundle hash mismatch: {failures}")
    checkpoint = bundle / ready["checkpoint"]
    if sha256_file(checkpoint) != ready["checkpoint_sha256"]:
        raise RuntimeError("imported checkpoint hash mismatch")
    return ready


def _import_short_bundle(remote: str) -> tuple[Path, dict[str, Any]]:
    destination = IMPORT_ROOT / "short_span"
    if destination.is_dir() and (destination / "READY_SHORT_FOR_RB2.json").is_file():
        return destination, _validate_bundle(destination, "READY_SHORT_FOR_RB2.json")
    partial = IMPORT_ROOT / ".short_span.partial"
    partial.mkdir(parents=True, exist_ok=True)
    if remote.startswith("/"):
        source = Path(remote) / "short_span"
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, partial, dirs_exist_ok=True)
    else:
        host, path = _remote_parts(remote)
        subprocess.run(
            ("rsync", "-a", "--partial", f"{host}:{path}/short_span/", f"{partial}/"),
            check=True,
        )
    ready = _validate_bundle(partial, "READY_SHORT_FOR_RB2.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing incompatible bundle destination: {destination}")
    os.replace(partial, destination)
    return destination, ready


def _import_long_bundle(remote: str) -> tuple[Path, dict[str, Any]]:
    destination = IMPORT_ROOT / "long_span"
    readiness = "READY_KC8_FOR_RB2.json"
    if destination.is_dir() and (destination / readiness).is_file():
        return destination, _validate_bundle(destination, readiness)
    partial = IMPORT_ROOT / ".long_span.partial"
    partial.mkdir(parents=True, exist_ok=True)
    if remote.startswith("/"):
        source = Path(remote) / "long_span"
        if not source.is_dir():
            raise FileNotFoundError(source)
        shutil.copytree(source, partial, dirs_exist_ok=True)
    else:
        host, path = _remote_parts(remote)
        subprocess.run(
            ("rsync", "-a", "--partial", f"{host}:{path}/long_span/", f"{partial}/"),
            check=True,
        )
    ready = _validate_bundle(partial, readiness)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing incompatible long bundle: {destination}")
    os.replace(partial, destination)
    return destination, ready


def _validate_completed_frontier_evidence() -> dict[str, Any]:
    """Lock completed sd1 rows without treating them as rb2 confirmatory rows."""

    lock = load_json(COMPLETED_FRONTIER_LOCK)
    if lock.get("schema_version") != "simvla_completed_frontier_evidence_v1":
        raise RuntimeError("completed frontier evidence schema changed")
    if (
        lock.get("classification") != "HOST_LOCAL_EGL_DIAGNOSTIC"
        or lock.get("manifest_sha256") != MANIFEST_SHA256
        or lock.get("inference_seed") != "seed02"
        or lock.get("rerun_allowed") is not False
    ):
        raise RuntimeError("completed frontier evidence axis changed")
    observed_rows: list[dict[str, Any]] = []
    for row, expected in lock["rows"].items():
        status_path = COMPLETED_FRONTIER_EVIDENCE / expected["status_file"]
        summary_path = COMPLETED_FRONTIER_EVIDENCE / expected["summary_file"]
        if sha256_file(status_path) != expected["status_sha256"]:
            raise RuntimeError(f"completed status hash changed: {row}")
        if status_path.read_text(encoding="utf-8").strip() != "exit_code=0":
            raise RuntimeError(f"completed row status is not zero: {row}")
        if sha256_file(summary_path) != expected["summary_sha256"]:
            raise RuntimeError(f"completed summary hash changed: {row}")
        payload = load_json(summary_path)
        checks = (
            payload.get("row") == row,
            payload.get("verdict") == "KC_FRONTIER_ROW_PASS",
            int(payload.get("episodes", -1)) == 500,
            int(payload.get("successes", -1)) == int(expected["successes"]),
            abs(float(payload.get("success_rate", -1.0)) - float(expected["success_rate"]))
            <= 1e-12,
            payload.get("manifest_sha256") == MANIFEST_SHA256,
        )
        if not all(checks):
            raise RuntimeError(f"completed row content changed: {row}")
        observed_rows.append(payload)
    aggregate_path = COMPLETED_FRONTIER_EVIDENCE / lock["aggregate_file"]
    paired_path = COMPLETED_FRONTIER_EVIDENCE / lock["paired_outcomes_file"]
    launcher_path = COMPLETED_FRONTIER_EVIDENCE / lock["launcher_status_file"]
    if sha256_file(aggregate_path) != lock["aggregate_sha256"]:
        raise RuntimeError("completed frontier aggregate hash changed")
    if sha256_file(paired_path) != lock["paired_outcomes_sha256"]:
        raise RuntimeError("completed frontier paired outcomes hash changed")
    if sha256_file(launcher_path) != lock["launcher_status_sha256"]:
        raise RuntimeError("historical launcher status hash changed")
    if not launcher_path.read_text(encoding="utf-8").startswith("exit_code=1"):
        raise RuntimeError("historical aggregation failure provenance changed")
    aggregate = load_json(aggregate_path)
    if (
        aggregate.get("classification") != "HOST_LOCAL_EGL_DIAGNOSTIC"
        or aggregate.get("manifest_sha256") != MANIFEST_SHA256
        or int(aggregate.get("episodes_per_row", -1)) != 500
    ):
        raise RuntimeError("completed frontier aggregate axis changed")
    with paired_path.open(newline="", encoding="utf-8") as handle:
        paired_rows = list(csv.DictReader(handle))
    if len(paired_rows) != 500:
        raise RuntimeError("completed paired outcome table is not 500 episodes")
    return {
        "verdict": "COMPLETED_FRONTIER_EVIDENCE_LOCKED",
        "classification": lock["classification"],
        "source_host": lock["source_host"],
        "manifest_sha256": MANIFEST_SHA256,
        "rows": observed_rows,
        "learned_vs_naive": lock["learned_vs_naive"],
        "launcher_exit_interpretation": lock["launcher_exit_interpretation"],
        "rerun_allowed": False,
    }


def _audit_parent_baselines() -> dict[str, Any]:
    """Require compatible same-host parents without scheduling duplicate rows."""

    rows: dict[str, Any] = {}
    missing: list[str] = []
    incompatible: list[str] = []
    for name, path in PARENT_BASELINES.items():
        if not path.is_file():
            missing.append(name)
            continue
        payload = load_json(path)
        if (
            int(payload.get("episodes", -1)) != 500
            or payload.get("manifest_sha256") != MANIFEST_SHA256
        ):
            incompatible.append(name)
            continue
        rows[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "row": payload,
        }
    if missing or incompatible:
        raise RuntimeError(
            "same-host parent audit failed; no automatic replacement will be run: "
            f"missing={missing}, incompatible={incompatible}"
        )
    return {
        "verdict": "SAME_HOST_PARENT_BASELINES_REUSED",
        "manifest_sha256": MANIFEST_SHA256,
        "missing": [],
        "incompatible": [],
        "new_rows_launched": [],
        "rows": rows,
    }


def _selected_output(k_c: int, mode: str) -> Path:
    span = "selected_long_span" if int(k_c) == 8 else "selected_short_span"
    return RESULT_ROOT / span / f"kc{k_c}_{mode}"


def _run_selected_row(bundle: Path, k_c: int, mode: str) -> dict[str, Any]:
    output = _selected_output(k_c, mode)
    summary = output / "row_summary.json"
    if summary.is_file():
        payload = load_json(summary)
        if (
            payload.get("verdict") == "RB2_STABILITY_ROW_COMPLETE"
            and int(payload.get("episodes", -1)) == 500
            and payload.get("manifest_sha256") == MANIFEST_SHA256
        ):
            return payload
        raise RuntimeError(f"incompatible selected row exists: {output}")
    _wait_for_gpu()
    subprocess.run(
        (
            str(PYTHON),
            "-m",
            SELECTED_EVALUATOR,
            "--output",
            str(output),
            "--bundle",
            str(bundle),
            "--manifest",
            str(MANIFEST),
            "--k-c",
            str(k_c),
            "--generation-mode",
            mode,
            "--physical-gpu-id",
            "0",
        ),
        cwd=WORKTREE,
        env=_environment(),
        check=True,
    )
    return load_json(summary)


def _write_frontier_summary(
    completed: dict[str, Any], parents: dict[str, Any]
) -> None:
    atomic_write_json(
        RESULT_ROOT / "rb2_frontier_control_summary.json",
        {
            "verdict": "RB2_FRONTIER_QUEUE_CLEARED_BY_LOCKED_EVIDENCE",
            "manifest_sha256": MANIFEST_SHA256,
            "inference_seed": "seed02",
            "additional_inference_seed": False,
            "completed_frontier_evidence": completed,
            "same_host_parent_audit": parents,
            "do_not_repeat_rows": sorted(completed["rows"], key=lambda row: row["row"]),
            "frontier_rows_remaining_in_execution_queue": [],
        },
    )


def _acceptance(
    *, baseline: dict[str, Any], row: dict[str, Any], old_success_rate: float | None
) -> dict[str, Any]:
    k_c = int(row["k_c"])
    rate = float(row["success_rate"])
    baseline_rate = float(baseline["success_rate"])
    tolerance = {2: 0.01, 3: 0.01, 4: 0.02, 8: 0.04}[k_c]
    task_drops = {
        task: float(baseline["per_task"][task]["success_rate"])
        - float(row["per_task_success_rate"][task])
        for task in baseline["per_task"]
    }
    task_drop_limit = 0.20 if k_c == 8 else 0.10
    checks: dict[str, bool] = {
        "within_baseline_tolerance": rate >= baseline_rate - tolerance,
        (
            "no_catastrophic_task_drop_over_20pp"
            if k_c == 8
            else "no_task_drop_over_10pp"
        ): max(task_drops.values()) <= task_drop_limit + 1e-12,
        "no_catastrophic_gripper_tail": float(row["switch_disagreement_p95"])
        <= max(
            2.0 * float(baseline["switch_disagreement_p95"]),
            float(baseline["switch_disagreement_p95"]) + 0.10,
        ),
    }
    if k_c == 3 and old_success_rate is not None:
        checks["old_kc3_improved_by_1pp"] = rate >= old_success_rate + 0.01 - 1e-12
    if k_c == 4 and old_success_rate is not None:
        checks["old_kc4_improved_by_5pp"] = rate >= old_success_rate + 0.05 - 1e-12
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_success_rate": baseline_rate,
        "success_rate": rate,
        "max_task_drop": max(task_drops.values()),
        "full_vlm_call_reduction_expected": 1.0 - 1.0 / k_c,
    }


def _baseline_with_tasks() -> dict[str, Any]:
    baseline = load_json(BASELINE)
    episode_csv = BASELINE.parent / "shard_rank0_tasks_0_9/episode_metrics.csv"
    with episode_csv.open(newline="", encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    if len(episodes) != 500:
        raise RuntimeError("same-host baseline episode table is not 500 episodes")
    baseline["per_task"] = {
        str(task): {
            "episodes": 50,
            "successes": sum(
                str(row["success"]).lower() in {"1", "true"}
                for row in episodes
                if int(row["task_id"]) == task
            ),
        }
        for task in range(10)
    }
    for value in baseline["per_task"].values():
        value["success_rate"] = value["successes"] / value["episodes"]
    switch = sorted(float(row["switch_disagreement"]) for row in episodes)
    baseline["switch_disagreement_p95"] = switch[int(round(0.95 * (len(switch) - 1)))]
    return baseline


def _aggregate_selected(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = _baseline_with_tasks()
    old = {3: 467 / 500, 4: 438 / 500}
    learned: dict[int, dict[str, Any]] = {}
    table: list[dict[str, Any]] = []
    for row in rows:
        acceptance = None
        if row["generation_mode"] == "learned_ng3":
            acceptance = _acceptance(
                baseline=baseline,
                row=row,
                old_success_rate=old.get(int(row["k_c"])),
            )
            learned[int(row["k_c"])] = acceptance
        table.append({**row, "online_acceptance": acceptance})
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = (
        "row",
        "k_c",
        "generation_mode",
        "episodes",
        "successes",
        "success_rate",
        "mean_policy_query_latency_ms",
        "mean_vlm_latency_ms",
        "mean_condition_latency_ms",
        "mean_generation_latency_ms",
        "mean_gripper_switches_per_episode",
        "switch_disagreement_p95",
        "manifest_sha256",
        "checkpoint_sha256",
    )
    for filename, selected_table in (
        ("rb2_kc2_kc3_kc4_results.csv", [row for row in table if int(row["k_c"]) <= 4]),
        ("rb2_kc8_results.csv", [row for row in table if int(row["k_c"]) == 8]),
    ):
        with (RESULT_ROOT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected_table)
    selected = next(
        (k_c for k_c in (8, 4, 3, 2) if learned.get(k_c, {}).get("passed")),
        None,
    )
    result = {
        "verdict": "SIMVLA_STABILITY_METHOD_SELECTED" if selected else "NO_ONLINE_OPERATING_POINT_PASSED",
        "selected_k_c": selected,
        "selected_n_g": 3 if selected else None,
        "manifest_sha256": MANIFEST_SHA256,
        "checkpoint_sha256_by_k_c": {
            str(k_c): next(
                str(row["checkpoint_sha256"])
                for row in table
                if int(row["k_c"]) == k_c
            )
            for k_c in sorted({int(row["k_c"]) for row in table})
        },
        "baseline": baseline,
        "learned_ng3_acceptance": learned,
        "naive_nfe3_is_mechanism_control": True,
        "generation_ng3_retained": True,
        "additional_inference_seed": False,
    }
    atomic_write_json(RESULT_ROOT / "final_simvla_method_selection.json", result)
    return result


def _final_report(state: AtomicStageState, error: str | None = None) -> None:
    selection = RESULT_ROOT / "final_simvla_method_selection.json"
    lines = [
        "# rb2 SimVLA Stability Alignment Evaluation",
        "",
        f"- timestamp: `{_timestamp()}`",
        f"- hostname: `{socket.gethostname()}`",
        "- renderer: `EGL`",
        f"- manifest SHA-256: `{MANIFEST_SHA256}`",
        "- inference axis: `seed02` only",
        "- N_G=3: retained as validated component",
        "- naive NFE=3: mechanism control",
        "- corrected completed frontier: locked HOST_LOCAL_EGL_DIAGNOSTIC evidence",
        "- completed frontier rows rerun: `no`",
        "- selected-checkpoint K_C=2 queue: learned N_G=3 only",
        "- selected-checkpoint K_C>=3 queue: full NFE=10 + naive NFE=3 + learned N_G=3",
        "- active jobs altered: `no`",
        "- additional inference seed launched: `no`",
        f"- final selection: `{selection if selection.is_file() else 'PENDING'}`",
        f"- error: `{error or 'none'}`",
        "",
        "```json",
        json.dumps(state.payload, indent=2, sort_keys=True),
        "```",
    ]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "final_rb2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = RESULT_ROOT / "pipeline.lock"
    lock_handle = lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"another rb2 stability pipeline owns {lock}") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()}\n")
    lock_handle.flush()
    state = AtomicStageState(RESULT_ROOT / "pipeline_state.json", RB2_STAGES)
    stage = "R0"
    try:
        for path in (
            PYTHON,
            WORKTREE,
            UPSTREAM,
            MANIFEST,
            BASELINE,
            COMPLETED_FRONTIER_EVIDENCE,
            COMPLETED_FRONTIER_LOCK,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        if load_json(MANIFEST).get("manifest_sha256") != MANIFEST_SHA256:
            raise RuntimeError("rb2 seed02 manifest identity changed")
        state.set("R0", "RUNNING", started=_timestamp())
        atomic_write_json(RESULT_ROOT / "current_rb2_state.json", _audit())
        state.set("R0", "PASSED", finished=_timestamp())

        stage = "R1"
        state.set("R1", "RUNNING", started=_timestamp())
        completed_frontier = _validate_completed_frontier_evidence()
        parent_baselines = _audit_parent_baselines()
        _write_frontier_summary(completed_frontier, parent_baselines)
        state.set(
            "R1",
            "PASSED",
            finished=_timestamp(),
            completed_frontier_rows=len(completed_frontier["rows"]),
            same_host_parent_rows=len(parent_baselines["rows"]),
            rows_launched=0,
        )

        remote = os.environ.get("SD1_BUNDLE_REMOTE", "")
        if not remote:
            raise ValueError("SD1_BUNDLE_REMOTE is required")
        stage = "R2"
        state.set(
            "R2",
            "SKIPPED",
            reason="condition_kc2_naive_nfe3 is locked completed evidence",
        )
        stage = "R3"
        state.set(
            "R3",
            "SKIPPED",
            reason="condition_kc3_naive_nfe3 is locked completed evidence",
        )

        stage = "R4"
        state.set(
            "R4",
            "SKIPPED",
            reason=(
                "condition_kc2_naive_nfe2 and condition_kc2_ng2 are locked "
                "completed evidence"
            ),
        )

        stage = "R5"
        state.set("R5", "RUNNING", started=_timestamp())
        while not remote_ready(remote):
            print(f"[{_timestamp()}] short sd1 bundle not ready; waiting 120s without GPU use", flush=True)
            time.sleep(120)
        state.set("R5", "PASSED", finished=_timestamp())

        stage = "R6"
        state.set("R6", "RUNNING", started=_timestamp())
        bundle, ready = _import_short_bundle(remote)
        state.set("R6", "PASSED", finished=_timestamp(), checkpoint_sha256=ready["checkpoint_sha256"])

        selected_rows: list[dict[str, Any]] = []
        stage = "R7"
        state.set("R7", "RUNNING", started=_timestamp())
        selected_rows.append(_run_selected_row(bundle, 2, "learned_ng3"))
        state.set("R7", "PASSED", finished=_timestamp())

        stage = "R8"
        if ready.get("kc3_offline_ready"):
            state.set("R8", "RUNNING", started=_timestamp())
            for mode in ("nfe10", "naive_nfe3", "learned_ng3"):
                selected_rows.append(_run_selected_row(bundle, 3, mode))
            state.set("R8", "PASSED", finished=_timestamp())
        else:
            state.set("R8", "SKIPPED", reason="KC3_OFFLINE_READY absent")

        stage = "R9"
        if ready.get("kc4_offline_ready"):
            state.set("R9", "RUNNING", started=_timestamp())
            for mode in ("nfe10", "naive_nfe3", "learned_ng3"):
                selected_rows.append(_run_selected_row(bundle, 4, mode))
            state.set("R9", "PASSED", finished=_timestamp())
        else:
            state.set("R9", "SKIPPED", reason="KC4_OFFLINE_READY absent")

        preliminary = _aggregate_selected(selected_rows)
        stage = "R10"
        kc4_passed = bool(
            preliminary.get("learned_ng3_acceptance", {}).get(4, {}).get("passed")
        )
        if kc4_passed:
            state.set("R10", "RUNNING", started=_timestamp())
            long_blocked: dict[str, Any] | None = None
            while not remote_ready(remote, "READY_KC8_FOR_RB2.json"):
                long_blocked = remote_payload(
                    remote,
                    span="long_span",
                    filename="LONG_SPAN_TERMINAL.json",
                )
                if long_blocked is not None:
                    break
                print(
                    f"[{_timestamp()}] K_C=4 passed; long sd1 bundle not ready; "
                    "waiting 120s without GPU use",
                    flush=True,
                )
                time.sleep(120)
            if long_blocked is not None:
                state.set(
                    "R10",
                    "SKIPPED",
                    finished=_timestamp(),
                    reason="sd1 long-span pipeline terminated without KC8 readiness",
                    sd1_terminal=long_blocked,
                )
                state.set("R11", "SKIPPED", reason="KC8_OFFLINE_READY absent")
            else:
                long_bundle, long_ready = _import_long_bundle(remote)
                state.set(
                    "R10",
                    "PASSED",
                    finished=_timestamp(),
                    checkpoint_sha256=long_ready["checkpoint_sha256"],
                )
                stage = "R11"
                state.set("R11", "RUNNING", started=_timestamp())
                for mode in ("nfe10", "naive_nfe3", "learned_ng3"):
                    selected_rows.append(_run_selected_row(long_bundle, 8, mode))
                state.set("R11", "PASSED", finished=_timestamp())
        else:
            state.set(
                "R10",
                "SKIPPED",
                reason="K_C=4 learned N_G=3 online gate did not pass",
            )
            state.set("R11", "SKIPPED", reason="long-span prerequisite failed")
        stage = "R12"
        state.set("R12", "RUNNING", started=_timestamp())
        _aggregate_selected(selected_rows)
        state.set("R12", "PASSED", finished=_timestamp())
        _final_report(state)
        return 0
    except Exception as exc:
        try:
            state.set(stage, "FAILED", finished=_timestamp(), error=str(exc))
        except Exception:
            pass
        _final_report(state, error=f"{type(exc).__name__}: {exc}")
        (RESULT_ROOT / "pipeline_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"RB2_STABILITY_PIPELINE_FAILED stage={stage}: {exc}", file=sys.stderr)
        return 1
    finally:
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(run())
