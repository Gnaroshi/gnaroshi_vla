"""Fail-closed two-GPU selection and launch metadata for multirate SimVLA."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    atomic_write_json,
)


def parse_gpu_ids(value: str | None) -> tuple[int, int]:
    if value is None:
        raise ValueError("SIMVLA_GPU_IDS=<gpu_a>,<gpu_b> is required")
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError("SIMVLA_GPU_IDS must contain exactly two comma-separated IDs")
    try:
        selected = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("SIMVLA_GPU_IDS must contain integer physical GPU IDs") from exc
    if len(set(selected)) != 2 or any(value < 0 for value in selected):
        raise ValueError("SIMVLA_GPU_IDS must contain two distinct non-negative IDs")
    return selected[0], selected[1]


def _run_nvidia_smi(arguments: list[str]) -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _gpu_metadata(gpu_id: int) -> dict[str, Any]:
    rows = _run_nvidia_smi(
        [
            "-i",
            str(gpu_id),
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if len(rows) != 1:
        raise RuntimeError(f"expected one metadata row for GPU {gpu_id}, got {rows}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"unexpected metadata row for GPU {gpu_id}: {rows[0]}")
    return {
        "physical_id": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "driver_version": fields[3],
        "memory_total_mib": int(fields[4]),
    }


def _compute_processes(gpu_id: int) -> list[dict[str, Any]]:
    rows = _run_nvidia_smi(
        [
            "-i",
            str(gpu_id),
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    processes: list[dict[str, Any]] = []
    for row in rows:
        if row.lower().startswith("no running"):
            continue
        fields = [field.strip() for field in row.split(",", 2)]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected process row for GPU {gpu_id}: {row}")
        processes.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_gpu_memory_mib": int(fields[2]),
            }
        )
    return processes


def require_two_idle_gpus(
    value: str | None,
    *,
    output: str | Path | None,
    require_absent_output: bool,
) -> dict[str, Any]:
    selected = parse_gpu_ids(value)
    metadata = [_gpu_metadata(gpu_id) for gpu_id in selected]
    processes = {str(gpu_id): _compute_processes(gpu_id) for gpu_id in selected}
    occupied = {gpu_id: processes[str(gpu_id)] for gpu_id in selected if processes[str(gpu_id)]}
    if occupied:
        raise RuntimeError(f"selected GPUs are not idle: {occupied}")
    output_path = Path(output).expanduser().resolve() if output else None
    if require_absent_output and output_path is not None and output_path.exists():
        raise FileExistsError(f"refusing existing output root: {output_path}")

    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
    except Exception as exc:  # pragma: no cover - environment diagnostic
        torch_version = f"UNAVAILABLE:{type(exc).__name__}"
        torch_cuda = None
    return {
        "verdict": "TWO_SELECTED_GPUS_IDLE",
        "selected_physical_gpu_ids": list(selected),
        "cuda_visible_devices_for_child": ",".join(str(value) for value in selected),
        "selected_gpu_metadata": metadata,
        "selected_gpu_processes": processes,
        "torch_version": torch_version,
        "torch_cuda_version": torch_cuda,
        "output": str(output_path) if output_path else None,
        "output_must_not_preexist": bool(require_absent_output),
        "unselected_gpus_inspected": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", default=os.environ.get("SIMVLA_GPU_IDS"))
    parser.add_argument("--output", default="")
    parser.add_argument("--require-absent-output", action="store_true")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    payload = require_two_idle_gpus(
        args.gpu_ids,
        output=args.output or None,
        require_absent_output=args.require_absent_output,
    )
    atomic_write_json(args.json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
