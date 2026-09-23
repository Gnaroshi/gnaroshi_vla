"""One-command sd1 pipeline for bounded diagnostics and gated V3 training."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    BUNDLE_SCHEMA,
    atomic_write_json,
    canonical_sha256,
    free_gpu_pairs,
    gpu_is_free,
    load_json,
    sha256_file,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_contracts import (
    v3_stage_graph,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_runtime import (
    DEFAULT_CACHE,
    DEFAULT_CONDITION_50K,
    DEFAULT_CONDITION_150K,
    DEFAULT_GENERATION_30K,
    DEFAULT_NORM,
)


WORKTREE = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_WORKTREE",
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
        "SIMVLA_STABILITY_V3_PYTHON",
        "/home/mingyujung/miniconda3/envs/simvla_libero/bin/python",
    )
).resolve()
RESULT_ROOT = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_RESULT_ROOT",
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/"
        "results/simvla/stability_alignment/recurrence_v3_bounded_pilot_v3",
    )
).resolve()
DIAGNOSTIC_REUSE_ROOT = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_DIAGNOSTIC_REUSE_ROOT",
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/"
        "results/simvla/stability_alignment/recurrence_v3",
    )
).resolve()
TRANSFER_ROOT = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_TRANSFER_ROOT",
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/"
        "transfers/simvla_stability_v3_selected",
    )
).resolve()
REPORT_ROOT = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V3_REPORT_ROOT",
        "/home/mingyujung/private/gnaroshi_vla/codex_outputs/"
        "simvla_stability_v3_bounded_pilot_v3_runtime",
    )
).resolve()
CONDITION = {
    "S50": Path(os.environ.get("SIMVLA_CONDITION_50K", DEFAULT_CONDITION_50K)).resolve(),
    "S150": Path(
        os.environ.get("SIMVLA_CONDITION_150K", DEFAULT_CONDITION_150K)
    ).resolve(),
}
CACHE = Path(os.environ.get("SIMVLA_STABILITY_V3_CACHE", DEFAULT_CACHE)).resolve()
GENERATION = Path(
    os.environ.get("SIMVLA_STABILITY_V3_GENERATION", DEFAULT_GENERATION_30K)
).resolve()
NORM = Path(os.environ.get("SIMVLA_STABILITY_V3_NORM", DEFAULT_NORM)).resolve()
V2_ROOT = Path(
    os.environ.get(
        "SIMVLA_STABILITY_V2_RESULT_ROOT",
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/"
        "results/simvla/stability_alignment/condition_only_v2",
    )
).resolve()
DIAGNOSTICS_MODULE = (
    "architectures.simvla.adapters.latentloop.stability_alignment.v3_diagnostics"
)
PREPARE_MODULE = (
    "architectures.simvla.adapters.latentloop.stability_alignment.v3_prepare"
)
TRAINER_MODULE = (
    "architectures.simvla.adapters.latentloop.stability_alignment.v3_trainer"
)


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    success_file: Path
    validator: Callable[[Mapping[str, Any]], bool]
    output_root: Path
    preserve_existing: bool = False


@dataclass
class RunningJob:
    job: Job
    pair: tuple[int, int]
    process: subprocess.Popen[str]
    handle: Any


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _gpu_pool() -> tuple[int, ...]:
    raw = os.environ.get("SIMVLA_STABILITY_V3_GPU_POOL", "4,5,6,7")
    pool = tuple(int(value) for value in raw.split(",") if value.strip())
    if len(pool) < 2 or len(pool) != len(set(pool)):
        raise ValueError("SIMVLA_STABILITY_V3_GPU_POOL needs unique GPU IDs")
    return pool


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
            "uuid": uuid,
            "memory_used_mib": int(memory),
            "utilization_percent": int(utilization),
            "compute_pids": [],
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
        fields = [value.strip() for value in row.split(",")]
        if len(fields) == 2 and fields[0] in uuid_to_index:
            snapshot[uuid_to_index[fields[0]]]["compute_pids"].append(int(fields[1]))
    for payload in snapshot.values():
        payload["free"] = gpu_is_free(
            memory_used_mib=payload["memory_used_mib"],
            utilization_percent=payload["utilization_percent"],
            compute_pids=payload["compute_pids"],
        )
    return snapshot


def _r150_training_enabled(pool: Sequence[int]) -> bool:
    mode = os.environ.get("SIMVLA_STABILITY_V3_ENABLE_R150", "auto")
    if mode == "0":
        return False
    if mode == "1":
        return True
    if mode != "auto":
        raise ValueError("SIMVLA_STABILITY_V3_ENABLE_R150 must be auto, 0, or 1")
    snapshot = _gpu_snapshot()
    busy = [gpu for gpu, payload in snapshot.items() if not payload["free"]]
    available = free_gpu_pairs(
        pool,
        busy,
        running_pairs=(),
        max_simultaneous_pairs=max(1, len(pool) // 2),
    )
    return len(available) >= 2


def _environment(pair: tuple[int, int]) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "SIMVLA_GPU_IDS": ",".join(map(str, pair)),
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, pair)),
            "SIMVLA_UPSTREAM_ROOT": str(UPSTREAM),
            "PYTHONPATH": f"{WORKTREE}:{UPSTREAM}:{environment.get('PYTHONPATH', '')}",
            "HF_HOME": environment.get(
                "HF_HOME", "/home/mingyujung/private/gnaroshi_vla/.cache/huggingface"
            ),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTHONHASHSEED": "20260825",
            "WANDB_MODE": environment.get("WANDB_MODE", "online"),
        }
    )
    return environment


def _torchrun(module: str, command: str, args: Sequence[str], port: int) -> tuple[str, ...]:
    return (
        str(PYTHON),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        f"--master_port={int(port)}",
        "-m",
        module,
        command,
        *tuple(args),
    )


def _common(output: Path, condition_parent: Path) -> list[str]:
    return [
        "--repository",
        str(WORKTREE),
        "--output",
        str(output),
        "--cache",
        str(CACHE),
        "--condition-parent",
        str(condition_parent),
        "--generation-parent",
        str(GENERATION),
        "--norm-stats",
        str(NORM),
        "--split-seed",
        "20260822",
        "--seed",
        "20260825",
    ]


def _complete(job: Job) -> bool:
    if not job.success_file.is_file():
        return False
    try:
        return bool(job.validator(load_json(job.success_file)))
    except Exception:
        return False


def _reuse_completed_v2_diagnostics() -> None:
    """Copy immutable diagnostics so the policy revision does not rerun them."""

    if RESULT_ROOT == DIAGNOSTIC_REUSE_ROOT:
        return
    reusable = (
        "diagnostics/v2_checkpoint_sweep",
        "diagnostics/v2_gradient_audit",
    )
    for relative in reusable:
        source = DIAGNOSTIC_REUSE_ROOT / relative
        destination = RESULT_ROOT / relative
        if destination.exists():
            continue
        if not source.is_dir():
            raise FileNotFoundError(f"reusable V2 diagnostic missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
    atomic_write_json(
        RESULT_ROOT / "diagnostic_reuse_contract.json",
        {
            "schema_version": "simvla_stability_v3_diagnostic_reuse_v1",
            "source": str(DIAGNOSTIC_REUSE_ROOT),
            "destination": str(RESULT_ROOT),
            "reused": list(reusable),
            "reason": "calibration policy revision does not alter immutable V2 diagnostics",
        },
    )


def _run_jobs(jobs: Sequence[Job], *, stage: str, pool: Sequence[int]) -> None:
    pending = [job for job in jobs if not _complete(job)]
    if not pending:
        return
    log_root = RESULT_ROOT / "logs"
    process_root = RESULT_ROOT / "process_state" / stage
    log_root.mkdir(parents=True, exist_ok=True)
    process_root.mkdir(parents=True, exist_ok=True)
    running: list[RunningJob] = []
    while pending or running:
        for active in tuple(running):
            rc = active.process.poll()
            if rc is None:
                continue
            active.handle.close()
            running.remove(active)
            atomic_write_json(
                process_root / f"{active.job.name}.json",
                {
                    "name": active.job.name,
                    "pair": list(active.pair),
                    "returncode": int(rc),
                    "finished_at": _timestamp(),
                    "success_file": str(active.job.success_file),
                },
            )
            if rc != 0 or not _complete(active.job):
                raise RuntimeError(
                    f"job failed: {active.job.name}; log={log_root / (active.job.name + '.log')}"
                )
        if pending:
            snapshot = _gpu_snapshot()
            atomic_write_json(RESULT_ROOT / "gpu_wait_state.json", snapshot)
            busy = [gpu for gpu, payload in snapshot.items() if not payload["free"]]
            available = free_gpu_pairs(
                pool,
                busy,
                running_pairs=[active.pair for active in running],
                max_simultaneous_pairs=max(1, len(pool) // 2),
            )
            while pending and available:
                pair = available[0]
                available = available[1:]
                job = pending.pop(0)
                if (
                    job.output_root.exists()
                    and not job.preserve_existing
                    and not _complete(job)
                ):
                    failed_root = RESULT_ROOT / "failed_attempts"
                    failed_root.mkdir(parents=True, exist_ok=True)
                    destination = failed_root / (
                        f"{job.name}_{time.strftime('%Y%m%d_%H%M%S')}"
                    )
                    shutil.move(str(job.output_root), destination)
                log = log_root / f"{job.name}.log"
                handle = log.open("a", encoding="utf-8")
                handle.write(
                    f"[{_timestamp()}] launch pair={pair} command={' '.join(job.command)}\n"
                )
                handle.flush()
                process = subprocess.Popen(
                    job.command,
                    cwd=WORKTREE,
                    env=_environment(pair),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running.append(RunningJob(job, pair, process, handle))
                print(
                    f"[{_timestamp()}] {stage}: launched {job.name} on GPUs {pair}",
                    flush=True,
                )
        if pending or running:
            time.sleep(30)


def _run_cpu(command: Sequence[str], *, name: str) -> None:
    log = RESULT_ROOT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(
            tuple(command),
            cwd=WORKTREE,
            env={
                **os.environ,
                "PYTHONPATH": f"{WORKTREE}:{UPSTREAM}:{os.environ.get('PYTHONPATH', '')}",
            },
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def _checkpoint_list(branch: str) -> str:
    paths = [
        V2_ROOT / branch.lower() / "train/checkpoints" / f"stability_step_{step:06d}.pt"
        for step in (2_000, 4_000, 6_000, 8_000, 10_000)
    ]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(f"immutable {branch} v2 checkpoint set is incomplete")
    return ",".join(map(str, paths))


def _write_merged_gradient_csv(paths: Mapping[str, Path], destination: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in paths.values():
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_v2_objective_audit(
    sweeps: Mapping[str, Mapping[str, Any]],
    gradients: Mapping[str, Mapping[str, Any]],
    destination: Path,
) -> None:
    lines = [
        "# Stability V2 Objective Audit",
        "",
        "## Immutable outcome",
        "",
        "S50 and S150 both reached 10K and failed the strict offline gate. No v2 "
        "checkpoint, metric file, or training artifact was modified by this audit.",
        "",
        "## Checkpoint sweep",
        "",
    ]
    for branch in sweeps:
        sweep = sweeps[branch]
        selection = sweep["validation_selection"]
        final = sweep["final_offline"]["corrected_scientific_gate"]
        lines.append(
            f"- {branch}: validation selected {selection['selected_step']} steps; "
            f"corrected final-offline verdict `{final['verdict']}`."
        )
    lines.extend(
        [
            "",
            "## Confirmed defects",
            "",
            "1. Historical recurrence stability was not the dominant gradient despite "
            "being the mechanism target.",
            "2. Parent preservation received a much larger realized gradient and raw-loss "
            "share than its nominal 1% role.",
            "3. With unique batch size 1 and three ages, the old top-10% CVaR reduced to "
            "the single maximum age loss, not a dataset-level tail objective.",
            "4. The event index included cross-query gripper transitions, while the old "
            "loss supervised only within-query transitions.",
            "5. The old teacher-forced target was emitted by the same trainable updater, "
            "so the target moved during optimization.",
            "6. Historical offline candidate joint actions used real change codes while "
            "training and deployment used zero 128-D codes. The new sweep uses zero code; "
            "historical rows remain immutable.",
            "",
            "## Measured weighted gradient shares",
            "",
        ]
    )
    for branch in gradients:
        shares = gradients[branch]["actual_weighted_gradient_shares"]
        lines.append(f"### {branch}")
        lines.append("")
        for name, value in shares.items():
            lines.append(f"- `{name}`: {100.0 * float(value):.3f}%")
        lines.append("")
        lines.append("Per-age raw diagnostics from the same fixed batches:")
        lines.append("")
        for name, payload in gradients[branch]["per_age_raw_metrics"].items():
            lines.append(
                f"- `{name}`: mean={float(payload['mean']):.8g}, "
                f"p95={float(payload['p95']):.8g}."
            )
        lines.append("")
    lines.extend(
        [
            "## V3 correction",
            "",
            "V3 removes parent preservation and batch-local CVaR, freezes the parent "
            "teacher, supervises age-2/3 recurrence gain, uses explicit source-locked "
            "hard pools, and calibrates weights by Condition-updater gradient contribution.",
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_method_contract(destination: Path) -> None:
    destination.write_text(
        """# SimVLA Stability Alignment V3 Contract

V3 does not change the Condition-updater architecture. It warm-starts from the
existing Condition 50K or 150K checkpoint and freezes the validated Generation
N_G=3 updater.

- Student: recursive Condition path only.
- Teacher: frozen parent updater with exact previous Condition.
- Primary ages: age 2 (weight 1) and age 3 (weight 2).
- Primary loss: input-error-normalized recurrence gain above a frozen parent
  median threshold measured before optimizer step 0.
- Sampling: 70% base, 15% gripper-transition, 15% frozen-parent top-10% tail.
- Gripper supervision: continuous, sign, within-query switch, and cross-query
  executed-boundary switch.
- Gradient targets: recurrence 35%, frozen N_G=3 25%, exact+frozen teacher 20%,
  gripper 10%, rotating full NFE10 5%, explicit hard sequence 5%.
- Pairwise cosine below -0.30 and nonpositive weighted-total alignment are
  retained as diagnostics, not pre-training scientific failures. Finite
  gradient calibration and declared contribution shares approve only a
  500-step bounded pilot; the measured multi-metric gate decides continuation.
- Removed: parent-preservation loss and batch-local top-10% CVaR.
- Scheduler: fixed 30K horizon from step 0; bounded stops at 500, 2K, 5K,
  and 10K.
- Export: only a strict final-offline gate-passing checkpoint.
""",
        encoding="utf-8",
    )


def _diagnostic_jobs(port_base: int, branches: Sequence[str]) -> list[Job]:
    jobs: list[Job] = []
    for offset, branch in enumerate(branches):
        output = RESULT_ROOT / "diagnostics/v2_checkpoint_sweep" / branch.lower()
        jobs.append(
            Job(
                name=f"v2_sweep_{branch.lower()}",
                command=_torchrun(
                    DIAGNOSTICS_MODULE,
                    "checkpoint-sweep",
                    [
                        *_common(output, CONDITION[branch]),
                        "--branch",
                        branch,
                        "--checkpoints",
                        _checkpoint_list(branch),
                    ],
                    port_base + offset,
                ),
                success_file=output / "checkpoint_sweep_summary.json",
                validator=lambda payload: payload.get("verdict")
                == "V2_CHECKPOINT_SWEEP_COMPLETE",
                output_root=output,
            )
        )
    return jobs


def _gradient_jobs(port_base: int, branches: Sequence[str]) -> list[Job]:
    jobs: list[Job] = []
    common_weights = V2_ROOT / "stability_alignment_loss_weights.json"
    for offset, branch in enumerate(branches):
        sweep_root = RESULT_ROOT / "diagnostics/v2_checkpoint_sweep" / branch.lower()
        selected = load_json(sweep_root / "checkpoint_sweep_summary.json")[
            "validation_selection"
        ]["selected_checkpoint"]
        output = RESULT_ROOT / "diagnostics/v2_gradient_audit" / branch.lower()
        jobs.append(
            Job(
                name=f"v2_gradient_{branch.lower()}",
                command=_torchrun(
                    DIAGNOSTICS_MODULE,
                    "gradient-audit",
                    [
                        *_common(output, CONDITION[branch]),
                        "--branch",
                        branch,
                        "--candidate",
                        selected,
                        "--loss-weights",
                        str(common_weights),
                        "--audit-batches",
                        "64",
                    ],
                    port_base + offset,
                ),
                success_file=output / "gradient_audit_summary.json",
                validator=lambda payload: payload.get("verdict")
                == "STABILITY_V2_GRADIENT_AUDIT_COMPLETE",
                output_root=output,
            )
        )
    return jobs


def _prepare_jobs(
    command: str, port_base: int, branches: Sequence[str]
) -> list[Job]:
    jobs: list[Job] = []
    for offset, branch in enumerate(branches):
        parent_key = "S50" if branch == "R50" else "S150"
        pool_root = RESULT_ROOT / "v3_preparation" / branch.lower() / "hard_pool"
        calibration_root = RESULT_ROOT / "v3_preparation" / branch.lower() / "calibration"
        if command == "build-pools":
            output = pool_root
            extra: list[str] = []
            success = output / "pool_preparation_summary.json"
            verdict = "STABILITY_V3_HARD_POOLS_READY"
        else:
            output = calibration_root
            extra = [
                "--hard-pool",
                str(pool_root / "stability_v3_hard_pool_contract.json"),
                "--audit-batches",
                "64",
            ]
            success = output / "stability_v3_loss_weights.json"
            verdict = {
                "STABILITY_V3_WEIGHTS_APPROVED",
                "STABILITY_V3_BOUNDED_PILOT_APPROVED_WITH_CONFLICT_WARNING",
                "STABILITY_V3_BOUNDED_PILOT_APPROVED_WITH_ALIGNMENT_WARNING",
                "STABILITY_V3_WEIGHTS_BLOCKED",
            }
        verdicts = {verdict} if isinstance(verdict, str) else verdict
        jobs.append(
            Job(
                name=f"v3_{command.replace('-', '_')}_{branch.lower()}",
                command=_torchrun(
                    PREPARE_MODULE,
                    command,
                    [*_common(output, CONDITION[parent_key]), *extra],
                    port_base + offset,
                ),
                success_file=success,
                validator=lambda payload, expected=verdicts: payload.get("verdict")
                in expected,
                output_root=output,
            )
        )
    return jobs


def _train_job(branch: str, stop_step: int, resume: Path | None, port: int) -> Job:
    parent_key = "S50" if branch == "R50" else "S150"
    train_root = RESULT_ROOT / "training" / branch.lower()
    pool_root = RESULT_ROOT / "v3_preparation" / branch.lower() / "hard_pool"
    weights_root = RESULT_ROOT / "v3_preparation" / branch.lower() / "calibration"
    args = [
        *_common(train_root, CONDITION[parent_key]),
        "--branch",
        branch,
        "--hard-pool",
        str(pool_root / "stability_v3_hard_pool_contract.json"),
        "--loss-weights",
        str(weights_root / "stability_v3_loss_weights.json"),
        "--stop-step",
        str(stop_step),
        "--wandb-name",
        f"simvla_stability_v3_{branch.lower()}",
    ]
    if stop_step == 500:
        args.extend(("--audit-interval", "29"))
    elif stop_step == 2_000 and resume is not None:
        args.extend(("--audit-interval", "79"))
    if resume is not None:
        args.extend(("--resume", str(resume)))
    summary = train_root / f"run_summary_step_{stop_step:06d}.json"
    step_label = "500" if stop_step == 500 else f"{stop_step // 1000}k"
    return Job(
        name=f"v3_train_{branch.lower()}_{step_label}",
        command=_torchrun(TRAINER_MODULE, "train", args, port),
        success_file=summary,
        validator=lambda payload, step=stop_step: payload.get("verdict")
        in {
            "STABILITY_V3_TRAINING_SEGMENT_COMPLETE",
            "STABILITY_V3_TRAINING_SEGMENT_COMPLETE_WITH_AUDIT_WARNING",
        }
        and int(payload.get("optimizer_step", -1)) == step,
        output_root=train_root,
        preserve_existing=resume is not None,
    )


def _offline_job(
    branch: str,
    step: int,
    split: str,
    previous: Path | None,
    port: int,
) -> Job:
    parent_key = "S50" if branch == "R50" else "S150"
    train_root = RESULT_ROOT / "training" / branch.lower()
    suffix = "validation" if split == "checkpoint_validation" else "final_offline"
    step_label = "500" if step == 500 else f"{step // 1000}k"
    output = RESULT_ROOT / "offline" / branch.lower() / f"{step_label}_{suffix}"
    checkpoint = train_root / "checkpoints" / f"stability_v3_step_{step:06d}.pt"
    args = [
        *_common(output, CONDITION[parent_key]),
        "--branch",
        branch,
        "--candidate",
        str(checkpoint),
        "--split",
        split,
    ]
    if previous is not None:
        args.extend(("--previous-gate", str(previous)))
    expected = (
        {"STABILITY_V3_GATE_PASS", "STABILITY_V3_GATE_FAIL"}
        if split == "final_offline" or step == 10_000
        else {
            f"STABILITY_V3_{step_label.upper()}_CONTINUE",
            f"STABILITY_V3_{step_label.upper()}_STOP",
        }
    )
    return Job(
        name=f"v3_offline_{branch.lower()}_{step_label}_{suffix}",
        command=_torchrun(TRAINER_MODULE, "offline", args, port),
        success_file=output / "offline_gate.json",
        validator=lambda payload, verdicts=expected: payload.get("verdict") in verdicts,
        output_root=output,
    )


def _passing(job: Job) -> bool:
    payload = load_json(job.success_file)
    return bool(payload.get("passed"))


def _stage_survivors(
    *,
    step: int,
    train_jobs: Mapping[str, Job],
    offline_jobs: Mapping[str, Job],
) -> list[str]:
    decisions: dict[str, Any] = {}
    survivors: list[str] = []
    for branch in train_jobs:
        training = load_json(train_jobs[branch].success_file)
        offline = load_json(offline_jobs[branch].success_file)
        audit = dict(training["moving_window_audit"])
        checks = {
            "training_segment_complete": training.get("verdict")
            in {
                "STABILITY_V3_TRAINING_SEGMENT_COMPLETE",
                "STABILITY_V3_TRAINING_SEGMENT_COMPLETE_WITH_AUDIT_WARNING",
            },
            "moving_window_numerical_safety": bool(
                audit.get("hard_safety_passed")
            ),
            "offline_multi_metric_gate": bool(offline.get("passed")),
        }
        passed = all(checks.values())
        if passed:
            survivors.append(branch)
        decisions[branch] = {
            "passed": passed,
            "checks": checks,
            "gradient_share_target_warning": not bool(
                audit.get("target_balance_passed")
            ),
            "moving_window_audit": audit,
            "offline_gate_verdict": offline.get("verdict"),
            "offline_gate": str(offline_jobs[branch].success_file),
            "training_summary": str(train_jobs[branch].success_file),
        }
    output = RESULT_ROOT / "stage_decisions" / f"step_{int(step):06d}.json"
    atomic_write_json(
        output,
        {
            "schema_version": "simvla_stability_v3_combined_stage_decision_v1",
            "optimizer_step": int(step),
            "policy": (
                "gradient-share targets are warnings; continuation requires the "
                "offline multi-metric gate and moving-window numerical safety"
            ),
            "branches": decisions,
            "survivors": survivors,
        },
    )
    return survivors


def _transfer_to_rb2(remote: str) -> None:
    payload = subprocess.run(
        (
            "rsync",
            "-a",
            "--exclude=READY_SHORT_FOR_RB2.json",
            f"{TRANSFER_ROOT}/",
            remote,
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    ready = (
        subprocess.run(
            (
                "rsync",
                "-a",
                str(TRANSFER_ROOT / "READY_SHORT_FOR_RB2.json"),
                remote,
            ),
            text=True,
            capture_output=True,
            check=False,
        )
        if payload.returncode == 0
        else None
    )
    atomic_write_json(
        TRANSFER_ROOT / "rb2_transfer_status.json",
        {
            "destination": remote,
            "payload_returncode": int(payload.returncode),
            "readiness_returncode": int(ready.returncode) if ready else None,
            "returncode": int(
                payload.returncode or (ready.returncode if ready else 1)
            ),
            "payload_stdout": payload.stdout,
            "payload_stderr": payload.stderr,
            "readiness_stdout": ready.stdout if ready else "",
            "readiness_stderr": ready.stderr if ready else "payload transfer failed",
            "readiness_sent_last": ready is not None,
            "scientific_bundle_remains_valid": True,
        },
    )


def _export(branch: str, gate_path: Path) -> dict[str, Any]:
    checkpoint = (
        RESULT_ROOT
        / "training"
        / branch.lower()
        / "checkpoints/stability_v3_step_010000.pt"
    )
    if not checkpoint.is_file() or not load_json(gate_path).get("passed"):
        raise RuntimeError("refusing to export a non-passing V3 checkpoint")
    if TRANSFER_ROOT.exists():
        ready = TRANSFER_ROOT / "READY_SHORT_FOR_RB2.json"
        if ready.is_file():
            existing = load_json(ready)
            if existing.get("checkpoint_sha256") == sha256_file(checkpoint):
                remote = os.environ.get("RB2_V3_BUNDLE_DESTINATION")
                if remote:
                    _transfer_to_rb2(remote)
                return existing
        raise FileExistsError(f"incompatible V3 transfer root exists: {TRANSFER_ROOT}")
    TRANSFER_ROOT.mkdir(parents=True)
    checkpoint_name = "stability_v3_selected_step_010000.pt"
    shutil.copy2(checkpoint, TRANSFER_ROOT / checkpoint_name)
    shutil.copy2(NORM, TRANSFER_ROOT / "libero_norm.json")
    shutil.copy2(gate_path, TRANSFER_ROOT / "final_offline_gate.json")
    train_root = RESULT_ROOT / "training" / branch.lower()
    shutil.copy2(train_root / "source_lock.json", TRANSFER_ROOT / "source_lock.json")
    shutil.copy2(
        train_root / "training_contract.json", TRANSFER_ROOT / "training_contract.json"
    )
    ready = {
        "schema_version": BUNDLE_SCHEMA,
        "verdict": "READY_SHORT_FOR_RB2",
        "method_version": "stability_v3",
        "selected_branch": branch,
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": sha256_file(TRANSFER_ROOT / checkpoint_name),
        "offline_gate_passed": True,
        "offline_gate_verdict": "STABILITY_V3_GATE_PASS",
        "kc3_offline_ready": True,
        "kc4_offline_ready": True,
        "kc8_offline_ready": False,
        "kc8_blocked_until_kc4_online_passes": True,
        "generation_change_code": "zero_128d",
        "files": {
            relative: sha256_file(TRANSFER_ROOT / relative)
            for relative in (
                checkpoint_name,
                "libero_norm.json",
                "final_offline_gate.json",
                "source_lock.json",
                "training_contract.json",
            )
        },
    }
    ready["combined_sha256"] = canonical_sha256(ready)
    atomic_write_json(TRANSFER_ROOT / "READY_SHORT_FOR_RB2.json", ready)
    atomic_write_json(TRANSFER_ROOT / "SHA256_MANIFEST.json", ready["files"])
    remote = os.environ.get("RB2_V3_BUNDLE_DESTINATION")
    if remote:
        _transfer_to_rb2(remote)
    return ready


def _write_final_report(
    *,
    surviving: Sequence[str],
    final_gates: Mapping[str, Path],
    exported: Mapping[str, Any] | None,
) -> None:
    rb2_status = "UNCHANGED_RUNNING_OR_EXTERNAL"
    selected_v2: dict[str, Any] = {}
    for historical in ("S50", "S150"):
        path = (
            RESULT_ROOT
            / "diagnostics/v2_checkpoint_sweep"
            / historical.lower()
            / "checkpoint_sweep_summary.json"
        )
        if path.is_file():
            selected_v2[historical] = load_json(path)["validation_selection"]
    branch_status: dict[str, str] = {}
    calibration_conflicts: dict[str, Any] = {}
    for branch in ("R50", "R150"):
        calibration = (
            RESULT_ROOT
            / "v3_preparation"
            / branch.lower()
            / "calibration/stability_v3_loss_weights.json"
        )
        candidates = sorted(
            (RESULT_ROOT / "offline" / branch.lower()).glob("*/offline_gate.json")
        )
        if candidates:
            branch_status[branch] = str(load_json(candidates[-1])["verdict"])
        elif calibration.is_file():
            payload = load_json(calibration)
            branch_status[branch] = str(payload["verdict"])
            calibration_conflicts[branch] = payload.get("severe_conflicts", {})
        else:
            branch_status[branch] = "NOT_STARTED"
    lines = [
        "# Final SimVLA Stability V3 Preparation Report",
        "",
        "## Immutable evidence",
        "",
        "- S50 v2: 10K strict offline failure; immutable.",
        "- S150 v2: 10K strict offline failure; immutable.",
        "- No v2 30K continuation was run.",
        "- No active job was stopped, restarted, or modified by this pipeline.",
        "- No git add, commit, or push was performed.",
        "- Confirmed defects: gradient underweighting of recurrence, overweighted "
        "parent preservation, batch-local max masquerading as CVaR, missing "
        "cross-query gripper supervision, moving teacher targets, and the historical "
        "candidate-only change-code evaluation mismatch.",
        "",
        "## V2 validation-selected checkpoints",
        "",
        *[
            f"- {branch}: step {payload['selected_step']} at `{payload['selected_checkpoint']}`."
            for branch, payload in selected_v2.items()
        ],
        "",
        "## V3 result",
        "",
        f"- Branches reaching strict final-offline: {', '.join(surviving) or 'none'}.",
        f"- R50 approval status: `{branch_status['R50']}`.",
        f"- R150 approval status: `{branch_status['R150']}`.",
    ]
    for branch, conflicts in calibration_conflicts.items():
        if conflicts:
            lines.append(f"- {branch} calibration conflicts: `{conflicts}`.")
    for branch, path in final_gates.items():
        gate = load_json(path)
        lines.append(f"- {branch}: `{gate['verdict']}` at `{path}`.")
    lines.extend(
        [
            f"- Exported branch: `{exported.get('selected_branch') if exported else 'none'}`.",
            f"- Existing rb2 S50 diagnostic status: `{rb2_status}`; classification remains DIAGNOSTIC_ONLY.",
            f"- K_C=3/4 readiness: `{'READY_FOR_GATED_RB2' if exported else 'BLOCKED_OFFLINE'}`.",
            "- K_C=8 remains blocked until an approved K_C=4 online row passes.",
            "",
            "The exact v2 objective defects and measured shares are recorded in "
            "`stability_v2_objective_audit.md`.",
        ]
    )
    (REPORT_ROOT / "final_simvla_stability_v3_preparation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run() -> dict[str, Any]:
    if os.environ.get("SIMVLA_STABILITY_V3_RUN") != "1":
        raise RuntimeError("set SIMVLA_STABILITY_V3_RUN=1 to approve this pipeline")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    pool = _gpu_pool()
    r150_enabled = _r150_training_enabled(pool)
    historical_branches = ("S50", "S150") if r150_enabled else ("S50",)
    v3_branches = ("R50", "R150") if r150_enabled else ("R50",)
    for path in (WORKTREE, PYTHON, CACHE / "manifest.json", GENERATION, NORM, *CONDITION.values()):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    _reuse_completed_v2_diagnostics()
    stage_graph = v3_stage_graph()
    atomic_write_json(RESULT_ROOT / "r50_r150_stage_graph.json", stage_graph)
    atomic_write_json(REPORT_ROOT / "r50_r150_stage_graph.json", stage_graph)
    _write_method_contract(REPORT_ROOT / "stability_v3_method_contract.md")
    atomic_write_json(
        RESULT_ROOT / "run_contract.json",
        {
            "schema_version": "simvla_stability_v3_pipeline_contract_v1",
            "gpu_pool": list(pool),
            "source_worktree": str(WORKTREE),
            "result_root": str(RESULT_ROOT),
            "v2_retraining": False,
            "automatic_30k": False,
            "r150_enabled": r150_enabled,
            "historical_diagnostic_branches": list(historical_branches),
            "v3_training_branches": list(v3_branches),
            "calibration_policy": "alignment_warning_then_500_step_measured_gate_v2",
            "bounded_pilot_steps": 500,
            "active_job_altered": False,
            "git_operation": False,
        },
    )
    print("[V1] immutable-v2 checkpoint sweeps", flush=True)
    _run_jobs(
        _diagnostic_jobs(29720, historical_branches),
        stage="V1_SWEEP",
        pool=pool,
    )
    sweep_merge = RESULT_ROOT / "diagnostics/v2_checkpoint_sweep/merged"
    if r150_enabled:
        if not (sweep_merge / "merged_sweep_summary.json").is_file():
            _run_cpu(
                (
                    str(PYTHON),
                    "-m",
                    DIAGNOSTICS_MODULE,
                    "merge",
                    "--output",
                    str(sweep_merge),
                    "--s50",
                    str(RESULT_ROOT / "diagnostics/v2_checkpoint_sweep/s50"),
                    "--s150",
                    str(RESULT_ROOT / "diagnostics/v2_checkpoint_sweep/s150"),
                ),
                name="v2_sweep_merge",
            )
        sweep_csv = sweep_merge / "stability_v2_checkpoint_sweep.csv"
    else:
        sweep_csv = (
            RESULT_ROOT
            / "diagnostics/v2_checkpoint_sweep/s50/stability_v2_checkpoint_sweep.csv"
        )
    shutil.copy2(
        sweep_csv,
        RESULT_ROOT / "stability_v2_checkpoint_sweep.csv",
    )
    shutil.copy2(
        sweep_csv,
        REPORT_ROOT / "stability_v2_checkpoint_sweep.csv",
    )
    print("[V2] fixed 64-batch historical gradient audits", flush=True)
    _run_jobs(
        _gradient_jobs(29730, historical_branches),
        stage="V2_GRADIENT",
        pool=pool,
    )
    gradient_csvs = {
        branch: RESULT_ROOT
        / "diagnostics/v2_gradient_audit"
        / branch.lower()
        / "stability_v2_gradient_cosine.csv"
        for branch in historical_branches
    }
    _write_merged_gradient_csv(
        gradient_csvs, RESULT_ROOT / "stability_v2_gradient_cosine.csv"
    )
    shutil.copy2(
        RESULT_ROOT / "stability_v2_gradient_cosine.csv",
        REPORT_ROOT / "stability_v2_gradient_cosine.csv",
    )
    sweep_summaries = {
        branch: load_json(
            RESULT_ROOT
            / "diagnostics/v2_checkpoint_sweep"
            / branch.lower()
            / "checkpoint_sweep_summary.json"
        )
        for branch in historical_branches
    }
    gradient_summaries = {
        branch: load_json(
            RESULT_ROOT
            / "diagnostics/v2_gradient_audit"
            / branch.lower()
            / "gradient_audit_summary.json"
        )
        for branch in historical_branches
    }
    _write_v2_objective_audit(
        sweep_summaries,
        gradient_summaries,
        REPORT_ROOT / "stability_v2_objective_audit.md",
    )
    print("[V3] frozen-parent hard pools and gamma", flush=True)
    _run_jobs(
        _prepare_jobs("build-pools", 29740, v3_branches),
        stage="V3_POOLS",
        pool=pool,
    )
    print("[V4] gradient-contribution calibration", flush=True)
    _run_jobs(
        _prepare_jobs("calibrate", 29750, v3_branches),
        stage="V4_CALIBRATION",
        pool=pool,
    )
    shutil.copy2(
        RESULT_ROOT
        / "v3_preparation/r50/hard_pool/stability_v3_hard_pool_contract.json",
        RESULT_ROOT / "stability_v3_hard_pool_contract.json",
    )
    shutil.copy2(
        RESULT_ROOT / "stability_v3_hard_pool_contract.json",
        REPORT_ROOT / "stability_v3_hard_pool_contract.json",
    )
    shutil.copy2(
        RESULT_ROOT
        / "v3_preparation/r50/calibration/stability_v3_loss_weights.json",
        RESULT_ROOT / "stability_v3_loss_weights.json",
    )
    shutil.copy2(
        RESULT_ROOT / "stability_v3_loss_weights.json",
        REPORT_ROOT / "stability_v3_loss_weights.json",
    )
    calibration_results = {
        branch: load_json(
            RESULT_ROOT
            / "v3_preparation"
            / branch.lower()
            / "calibration/stability_v3_loss_weights.json"
        )
        for branch in v3_branches
    }
    branches = [
        branch
        for branch, payload in calibration_results.items()
        if payload.get("approved_for_bounded_pilot")
    ]
    if not branches:
        _write_final_report(surviving=(), final_gates={}, exported=None)
        return {
            "verdict": "STABILITY_V3_STOPPED_AT_CALIBRATION",
            "exported": False,
            "blocked_calibrations": {
                branch: payload["severe_conflicts"]
                for branch, payload in calibration_results.items()
            },
        }
    print(f"[V5] 0->500 bounded pilot branches={branches}", flush=True)
    train_500 = {
        branch: _train_job(branch, 500, None, 29760 + index)
        for index, branch in enumerate(branches)
    }
    _run_jobs(list(train_500.values()), stage="V5_TRAIN_500", pool=pool)
    offline_500 = {
        branch: _offline_job(
            branch, 500, "checkpoint_validation", None, 29770 + index
        )
        for index, branch in enumerate(branches)
    }
    _run_jobs(list(offline_500.values()), stage="V6_GATE_500", pool=pool)
    branches = _stage_survivors(
        step=500, train_jobs=train_500, offline_jobs=offline_500
    )
    if not branches:
        _write_final_report(surviving=(), final_gates={}, exported=None)
        return {"verdict": "STABILITY_V3_STOPPED_AT_500", "exported": False}
    print(f"[V7] 500->2K branches={branches}", flush=True)
    train_2k: dict[str, Job] = {}
    for index, branch in enumerate(branches):
        resume = (
            RESULT_ROOT
            / "training"
            / branch.lower()
            / "checkpoints/stability_v3_step_000500.pt"
        )
        train_2k[branch] = _train_job(branch, 2_000, resume, 29780 + index)
    _run_jobs(list(train_2k.values()), stage="V7_TRAIN_2K", pool=pool)
    offline_2k: dict[str, Job] = {}
    for index, branch in enumerate(branches):
        previous = (
            RESULT_ROOT
            / "offline"
            / branch.lower()
            / "500_validation/offline_gate.json"
        )
        offline_2k[branch] = _offline_job(
            branch,
            2_000,
            "checkpoint_validation",
            previous,
            29790 + index,
        )
    _run_jobs(list(offline_2k.values()), stage="V8_GATE_2K", pool=pool)
    branches = _stage_survivors(
        step=2_000, train_jobs=train_2k, offline_jobs=offline_2k
    )
    if not branches:
        _write_final_report(surviving=(), final_gates={}, exported=None)
        return {"verdict": "STABILITY_V3_STOPPED_AT_2K", "exported": False}
    print(f"[V10] 2K->5K branches={branches}", flush=True)
    train_5k: dict[str, Job] = {}
    for index, branch in enumerate(branches):
        resume = RESULT_ROOT / "training" / branch.lower() / "checkpoints/stability_v3_step_002000.pt"
        train_5k[branch] = _train_job(branch, 5_000, resume, 29800 + index)
    _run_jobs(list(train_5k.values()), stage="V10_TRAIN_5K", pool=pool)
    offline_5k: dict[str, Job] = {}
    for index, branch in enumerate(branches):
        previous = RESULT_ROOT / "offline" / branch.lower() / "2k_validation/offline_gate.json"
        offline_5k[branch] = _offline_job(
            branch, 5_000, "checkpoint_validation", previous, 29810 + index
        )
    _run_jobs(list(offline_5k.values()), stage="V11_GATE_5K", pool=pool)
    branches = _stage_survivors(
        step=5_000, train_jobs=train_5k, offline_jobs=offline_5k
    )
    if not branches:
        _write_final_report(surviving=(), final_gates={}, exported=None)
        return {"verdict": "STABILITY_V3_STOPPED_AT_5K", "exported": False}
    print(f"[V12] 5K->10K branches={branches}", flush=True)
    train_10k: dict[str, Job] = {}
    for index, branch in enumerate(branches):
        resume = RESULT_ROOT / "training" / branch.lower() / "checkpoints/stability_v3_step_005000.pt"
        train_10k[branch] = _train_job(branch, 10_000, resume, 29820 + index)
    _run_jobs(list(train_10k.values()), stage="V12_TRAIN_10K", pool=pool)
    validation_10k = {
        branch: _offline_job(
            branch, 10_000, "checkpoint_validation", None, 29830 + index
        )
        for index, branch in enumerate(branches)
    }
    _run_jobs(list(validation_10k.values()), stage="V13_VALIDATE_10K", pool=pool)
    branches = _stage_survivors(
        step=10_000, train_jobs=train_10k, offline_jobs=validation_10k
    )
    if not branches:
        _write_final_report(surviving=(), final_gates={}, exported=None)
        return {"verdict": "STABILITY_V3_STRICT_VALIDATION_FAIL", "exported": False}
    final_jobs = [
        _offline_job(branch, 10_000, "final_offline", None, 29840 + index)
        for index, branch in enumerate(branches)
    ]
    _run_jobs(final_jobs, stage="V13_FINAL_OFFLINE", pool=pool)
    final_gates = {
        job.name.split("_")[2].upper(): job.success_file for job in final_jobs
    }
    survivors = [branch for branch, path in final_gates.items() if load_json(path)["passed"]]
    exported = None
    if survivors:
        selected = "R50" if "R50" in survivors else survivors[0]
        exported = _export(selected, final_gates[selected])
    _write_final_report(
        surviving=survivors, final_gates=final_gates, exported=exported
    )
    return {
        "verdict": (
            "STABILITY_V3_GATE_PASSING_BUNDLE_EXPORTED"
            if exported
            else "STABILITY_V3_FINAL_OFFLINE_FAIL"
        ),
        "exported": bool(exported),
        "surviving_branches": survivors,
        "transfer_root": str(TRANSFER_ROOT) if exported else None,
    }


def main() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = RESULT_ROOT / "pipeline.lock"
    try:
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run()
            atomic_write_json(RESULT_ROOT / "pipeline_summary.json", result)
            stale_failure = RESULT_ROOT / "pipeline_failure.json"
            if stale_failure.is_file():
                historical = RESULT_ROOT / "historical_failures"
                historical.mkdir(parents=True, exist_ok=True)
                shutil.move(
                    str(stale_failure),
                    historical
                    / f"pipeline_failure_{time.strftime('%Y%m%d_%H%M%S')}.json",
                )
    except Exception as error:
        atomic_write_json(
            RESULT_ROOT / "pipeline_failure.json",
            {
                "verdict": "STABILITY_V3_PIPELINE_RUNTIME_FAILURE",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "active_job_altered": False,
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
