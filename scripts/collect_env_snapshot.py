#!/usr/bin/env python3
"""Collect a lightweight environment snapshot for a VLA experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": "", "stderr": str(exc)}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def torch_info() -> str:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on selected env.
        return f"torch_importable: false\nerror: {exc}\n"

    lines = [
        "torch_importable: true",
        f"torch_version: {getattr(torch, '__version__', 'unknown')}",
        f"cuda_available: {torch.cuda.is_available()}",
        f"torch_cuda_version: {getattr(torch.version, 'cuda', None)}",
    ]
    try:
        lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                lines.append(f"cuda_device_{idx}: {torch.cuda.get_device_name(idx)}")
    except Exception as exc:  # pragma: no cover - hardware dependent.
        lines.append(f"cuda_query_error: {exc}")
    return "\n".join(lines) + "\n"


def build_snapshot(env_id: str) -> dict[str, object]:
    return {
        "env_id": env_id,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "conda_executable": shutil.which("conda"),
        "pip_executable": shutil.which("pip"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="unknown", help="Logical environment id.")
    parser.add_argument("--output-dir", help="Directory where snapshot files are written.")
    parser.add_argument("--include-conda", action="store_true", help="Write conda list if conda is available.")
    parser.add_argument("--include-pip", action="store_true", help="Write pip freeze if pip is available.")
    args = parser.parse_args()

    snapshot = build_snapshot(args.env_id)
    if not args.output_dir:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_text(output_dir / "python.txt", json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    write_text(output_dir / "platform.txt", platform.platform() + "\n")
    write_text(output_dir / "torch_cuda.txt", torch_info())

    if args.include_pip:
        result = run_command([sys.executable, "-m", "pip", "freeze"])
        write_text(output_dir / "pip_freeze.txt", result["stdout"] or result["stderr"])

    if args.include_conda and shutil.which("conda"):
        result = run_command(["conda", "list"])
        write_text(output_dir / "conda_list.txt", result["stdout"] or result["stderr"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

