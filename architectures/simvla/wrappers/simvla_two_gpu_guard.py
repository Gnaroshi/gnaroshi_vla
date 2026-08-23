#!/usr/bin/env python3
"""Guard exactly two user-selected idle physical GPUs for SimVLA V0."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def configured_forbidden_gpu_ids() -> set[int]:
    """Return optional site-local exclusions without hard-coding transient occupancy."""

    value = os.environ.get("SIMVLA_FORBIDDEN_GPU_IDS", "")
    if not value.strip():
        return set()
    try:
        identifiers = {int(part.strip()) for part in value.split(",") if part.strip()}
    except ValueError as exc:
        raise ValueError("SIMVLA_FORBIDDEN_GPU_IDS must contain integer physical GPU IDs") from exc
    if any(identifier < 0 for identifier in identifiers):
        raise ValueError("SIMVLA_FORBIDDEN_GPU_IDS must contain non-negative GPU IDs")
    return identifiers


def parse_selected_gpu_ids(value: str | None) -> tuple[int, int]:
    if value is None:
        raise ValueError("SIMVLA_GPU_IDS=<gpu_a>,<gpu_b> is required")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("SIMVLA_GPU_IDS must contain exactly two comma-separated IDs")
    try:
        identifiers = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("SIMVLA_GPU_IDS must contain integer physical GPU IDs") from exc
    if len(set(identifiers)) != 2:
        raise ValueError("SIMVLA_GPU_IDS must contain two distinct GPU IDs")
    forbidden = set(identifiers) & configured_forbidden_gpu_ids()
    if forbidden:
        raise ValueError(f"selected GPUs are forbidden by SIMVLA_FORBIDDEN_GPU_IDS: {sorted(forbidden)}")
    if any(identifier < 0 for identifier in identifiers):
        raise ValueError("GPU IDs must be non-negative")
    return identifiers[0], identifiers[1]


def query_selected_gpu_processes(gpu_id: int) -> list[dict[str, Any]]:
    """Inspect one selected physical GPU only."""
    command = [
        "nvidia-smi",
        "-i",
        str(gpu_id),
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no running"):
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi process row for GPU {gpu_id}: {line}")
        rows.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_gpu_memory_mib": int(fields[2]),
            }
        )
    return rows


def validate_two_idle_gpus(
    value: str | None,
    *,
    output: str | Path | None = None,
    require_empty_output: bool = False,
) -> dict[str, Any]:
    selected = parse_selected_gpu_ids(value)
    processes = {str(identifier): query_selected_gpu_processes(identifier) for identifier in selected}
    occupied = {identifier: processes[str(identifier)] for identifier in selected if processes[str(identifier)]}
    if occupied:
        raise RuntimeError(f"selected GPUs are not idle: {occupied}")
    output_path = Path(output).expanduser().resolve() if output else None
    if require_empty_output and output_path is not None and output_path.exists():
        raise FileExistsError(f"refusing existing output path: {output_path}")
    return {
        "verdict": "TWO_SELECTED_GPUS_IDLE",
        "selected_physical_gpu_ids": list(selected),
        "configured_forbidden_gpu_ids_not_inspected": sorted(configured_forbidden_gpu_ids()),
        "selected_gpu_processes": processes,
        "output": str(output_path) if output_path else None,
        "output_must_not_preexist": bool(require_empty_output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids", default=os.environ.get("SIMVLA_GPU_IDS"))
    parser.add_argument("--output", default="")
    parser.add_argument("--require-empty-output", action="store_true")
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    result = validate_two_idle_gpus(
        args.gpu_ids,
        output=args.output or None,
        require_empty_output=args.require_empty_output,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
