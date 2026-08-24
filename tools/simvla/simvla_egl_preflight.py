#!/usr/bin/env python3
"""Fail-closed GPU EGL and one-step LIBERO preflight for SimVLA evaluation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(
    os.environ.get("SIMVLA_UPSTREAM_ROOT", ROOT / "architectures" / "simvla" / "upstream")
).expanduser().resolve()
LIBERO_ROOT = UPSTREAM / "evaluation" / "libero" / "LIBERO"
for candidate in (ROOT, UPSTREAM, LIBERO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (  # noqa: E402
    atomic_write_json,
    validate_egl_identity,
)


def _decode_gl_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _environment_snapshot() -> dict[str, str | None]:
    names = (
        "CUDA_VISIBLE_DEVICES",
        "MUJOCO_GL",
        "PYOPENGL_PLATFORM",
        "MUJOCO_EGL_DEVICE_ID",
        "GALLIUM_DRIVER",
        "LIBGL_ALWAYS_SOFTWARE",
        "EGL_DEVICE_ID",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED",
        "SIMVLA_RENDER_AXIS",
    )
    return {name: os.environ.get(name) for name in names}


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing preflight output: {output}")

    environment = _environment_snapshot()
    result: dict[str, Any] = {
        "verdict": "EGL_PREFLIGHT_FAIL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "physical_gpu_id": int(args.gpu_id),
        "renderer_backend": "egl",
        "environment": environment,
        "upstream_root": str(UPSTREAM),
        "libero_root": str(LIBERO_ROOT),
        "libero_reset_pass": False,
        "libero_step_pass": False,
    }
    try:
        expected_visible = str(int(args.gpu_id))
        if environment["CUDA_VISIBLE_DEVICES"] != expected_visible:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES must be exactly the requested physical GPU: "
                f"{environment['CUDA_VISIBLE_DEVICES']!r} != {expected_visible!r}"
            )
        if environment["MUJOCO_GL"] != "egl":
            raise RuntimeError("MUJOCO_GL must equal egl before importing MuJoCo")
        if environment["PYOPENGL_PLATFORM"] != "egl":
            raise RuntimeError("PYOPENGL_PLATFORM must equal egl before importing OpenGL")
        if environment["MUJOCO_EGL_DEVICE_ID"] != expected_visible:
            raise RuntimeError(
                "MUJOCO_EGL_DEVICE_ID must select the requested physical GPU: "
                f"{environment['MUJOCO_EGL_DEVICE_ID']!r} != {expected_visible!r}"
            )
        if not LIBERO_ROOT.is_dir():
            raise FileNotFoundError(f"LIBERO root not found: {LIBERO_ROOT}")

        import mujoco
        import torch
        from OpenGL import GL

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        result.update(
            {
                "mujoco_version": str(mujoco.__version__),
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "visible_cuda_device_count": int(torch.cuda.device_count()),
                "visible_cuda_device_name": torch.cuda.get_device_name(0),
            }
        )
        if torch.cuda.device_count() != 1:
            raise RuntimeError("preflight requires exactly one visible CUDA device")

        context = mujoco.GLContext(64, 64)
        try:
            context.make_current()
            gl_vendor = _decode_gl_string(GL.glGetString(GL.GL_VENDOR))
            gl_renderer = _decode_gl_string(GL.glGetString(GL.GL_RENDERER))
            gl_version = _decode_gl_string(GL.glGetString(GL.GL_VERSION))
        finally:
            context.free()
        result.update(
            {
                "gl_vendor": gl_vendor,
                "gl_renderer": gl_renderer,
                "gl_version": gl_version,
            }
        )
        identity = validate_egl_identity(
            environment=environment,
            gl_vendor=gl_vendor,
            gl_renderer=gl_renderer,
        )
        result["egl_identity"] = identity
        if identity["verdict"] != "EGL_PREFLIGHT_PASS":
            raise RuntimeError("; ".join(identity["failures"]))

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[args.suite]()
        task = suite.get_task(int(args.task_id))
        bddl = (
            Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=int(args.resolution),
            camera_widths=int(args.resolution),
        )
        try:
            env.seed(int(args.environment_seed))
            env.reset()
            result["libero_reset_pass"] = True
            initial_states = suite.get_task_init_states(int(args.task_id))
            observation = env.set_init_state(initial_states[0])
            observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
            required_keys = {
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            }
            missing = sorted(required_keys - set(observation))
            if missing:
                raise RuntimeError(f"LIBERO step observation is missing keys: {missing}")
            result["libero_step_pass"] = True
            result["libero_task"] = {
                "suite": args.suite,
                "task_id": int(args.task_id),
                "bddl_file": str(bddl),
                "environment_seed": int(args.environment_seed),
                "resolution": int(args.resolution),
            }
        finally:
            env.close()

        result["verdict"] = "EGL_PREFLIGHT_PASS"
        result["failures"] = []
    except Exception as exc:
        result["verdict"] = "EGL_PREFLIGHT_FAIL"
        result["failures"] = [f"{type(exc).__name__}: {exc}"]
        result["traceback"] = traceback.format_exc()
    atomic_write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    result = run_preflight(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "EGL_PREFLIGHT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
