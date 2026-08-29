#!/usr/bin/env python3
"""Guard one or two idle physical GPUs for Generation Loop screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from architectures.simvla.wrappers.simvla_two_gpu_guard import (
    configured_forbidden_gpu_ids,
    query_selected_gpu_processes,
)


def parse_generation_gpu_ids(value: str | None) -> tuple[int, ...]:
    if value is None:
        raise ValueError("SIMVLA_GPU_IDS=<gpu> or <gpu_a>,<gpu_b> is required")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) not in (1, 2):
        raise ValueError("Generation Loop requires one or two comma-separated GPU IDs")
    try:
        identifiers = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("SIMVLA_GPU_IDS must contain integer physical GPU IDs") from exc
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("SIMVLA_GPU_IDS must contain distinct GPU IDs")
    if any(identifier < 0 for identifier in identifiers):
        raise ValueError("GPU IDs must be non-negative")
    forbidden = set(identifiers) & configured_forbidden_gpu_ids()
    if forbidden:
        raise ValueError(
            f"selected GPUs are forbidden by SIMVLA_FORBIDDEN_GPU_IDS: {sorted(forbidden)}"
        )
    return identifiers


def validate_generation_gpus(
    value: str | None,
    *,
    output: str | Path | None = None,
    require_empty_output: bool = False,
) -> dict[str, Any]:
    selected = parse_generation_gpu_ids(value)
    processes = {
        str(identifier): query_selected_gpu_processes(identifier)
        for identifier in selected
    }
    occupied = {
        identifier: processes[str(identifier)]
        for identifier in selected
        if processes[str(identifier)]
    }
    if occupied:
        raise RuntimeError(f"selected GPUs are not idle: {occupied}")
    output_path = Path(output).expanduser().resolve() if output else None
    if require_empty_output and output_path is not None and output_path.exists():
        raise FileExistsError(f"refusing existing output path: {output_path}")
    return {
        "verdict": "GENERATION_SELECTED_GPUS_IDLE",
        "selected_physical_gpu_ids": list(selected),
        "world_size": len(selected),
        "local_batch_for_global_batch_two": 2 // len(selected),
        "global_unique_batch": 2,
        "configured_forbidden_gpu_ids_not_inspected": sorted(
            configured_forbidden_gpu_ids()
        ),
        "selected_gpu_processes": processes,
        "output": str(output_path) if output_path else None,
        "output_must_not_preexist": bool(require_empty_output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-ids")
    parser.add_argument("--output", default="")
    parser.add_argument("--require-empty-output", action="store_true")
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    result = validate_generation_gpus(
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
