#!/usr/bin/env python3
"""Verify key runtime versions and an actual OSMesa LIBERO context."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--libero-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    expected = contract["runtime"]
    actual = {
        "python": platform.python_version(),
        "torch": package_version("torch"),
        "numpy": package_version("numpy"),
        "mujoco": package_version("mujoco"),
        "robosuite": package_version("robosuite"),
        "libero": package_version("libero"),
        "PyOpenGL": package_version("PyOpenGL"),
    }
    mismatches = {name: {"expected": expected[name], "actual": value} for name, value in actual.items() if value != expected[name]}
    if mismatches:
        raise RuntimeError(f"runtime version mismatch: {mismatches}")

    os.environ.update(
        LIBERO_GL_BACKEND="osmesa",
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        LIBERO_GL_REQUIRE_ACTUAL="1",
    )
    sys.path[:0] = [str(args.repo_root / "architectures/seer/upstream"), str(args.libero_path)]
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from utils.eval_utils_libero import get_renderer_backend_metadata, verify_renderer_backend

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = suite.get_task(0)
    bddl = args.libero_path / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=64, camera_widths=64,
        render_gpu_device_id=0, control_freq=20, horizon=1000,
    )
    try:
        env.reset()
        renderer = verify_renderer_backend(env, 0)
    finally:
        env.close()
    if renderer.get("backend_classification") != expected["renderer_classification"]:
        raise RuntimeError(f"renderer classification mismatch: {renderer}")
    if expected["renderer_substring"].lower() not in renderer.get("actual_gl_renderer", "").lower():
        raise RuntimeError(f"renderer identity mismatch: {renderer}")
    payload = {
        "schema_version": 1,
        "status": "S18_RUNTIME_GATE_PASS",
        "python_executable": sys.executable,
        "versions": actual,
        "renderer": renderer,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("S18_RUNTIME_GATE_PASS")


if __name__ == "__main__":
    main()
