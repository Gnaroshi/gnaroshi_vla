#!/usr/bin/env python3
"""Fail closed unless one selected GPU renders and steps LIBERO through EGL."""

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

from architectures.simvla.adapters.latentloop.native_v0_runtime import write_json  # noqa: E402


SOFTWARE_TOKENS = ("llvmpipe", "softpipe", "software rasterizer", "swrast", "osmesa")


def _decode(value: Any) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _environment() -> dict[str, str | None]:
    return {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "MUJOCO_GL",
            "PYOPENGL_PLATFORM",
            "MUJOCO_EGL_DEVICE_ID",
            "GALLIUM_DRIVER",
            "LIBGL_ALWAYS_SOFTWARE",
            "SIMVLA_RENDER_AXIS",
        )
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing preflight output: {output}")
    environment = _environment()
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
        expected = str(int(args.gpu_id))
        if environment["CUDA_VISIBLE_DEVICES"] != expected:
            raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly the requested GPU")
        if environment["MUJOCO_GL"] != "egl" or environment["PYOPENGL_PLATFORM"] != "egl":
            raise RuntimeError("MUJOCO_GL and PYOPENGL_PLATFORM must both equal egl")
        if environment["MUJOCO_EGL_DEVICE_ID"] != expected:
            raise RuntimeError("MUJOCO_EGL_DEVICE_ID must equal the requested physical GPU")
        if environment["GALLIUM_DRIVER"] or str(environment["LIBGL_ALWAYS_SOFTWARE"] or "").lower() in {"1", "true", "yes"}:
            raise RuntimeError("software-rendering environment variables are forbidden")
        if not LIBERO_ROOT.is_dir():
            raise FileNotFoundError(f"LIBERO root not found: {LIBERO_ROOT}")

        import mujoco
        import torch
        from OpenGL import GL

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("preflight requires exactly one visible CUDA device")
        result.update(
            {
                "mujoco_version": str(mujoco.__version__),
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda),
                "visible_cuda_device_count": int(torch.cuda.device_count()),
                "visible_cuda_device_name": torch.cuda.get_device_name(0),
            }
        )
        context = mujoco.GLContext(64, 64)
        try:
            context.make_current()
            vendor = _decode(GL.glGetString(GL.GL_VENDOR))
            renderer = _decode(GL.glGetString(GL.GL_RENDERER))
            version = _decode(GL.glGetString(GL.GL_VERSION))
        finally:
            context.free()
        combined = f"{vendor} {renderer}".lower()
        if not vendor or not renderer or any(token in combined for token in SOFTWARE_TOKENS):
            raise RuntimeError(f"non-GPU GL identity: vendor={vendor!r}, renderer={renderer!r}")
        result.update({"gl_vendor": vendor, "gl_renderer": renderer, "gl_version": version})

        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[args.suite]()
        task = suite.get_task(int(args.task_id))
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=int(args.resolution),
            camera_widths=int(args.resolution),
        )
        try:
            env.seed(int(args.environment_seed))
            env.reset()
            result["libero_reset_pass"] = True
            states = suite.get_task_init_states(int(args.task_id))
            observation = env.set_init_state(states[0])
            observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
            required = {
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            }
            missing = sorted(required - set(observation))
            if missing:
                raise RuntimeError(f"LIBERO observation is missing keys: {missing}")
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
        result["failures"] = [f"{type(exc).__name__}: {exc}"]
        result["traceback"] = traceback.format_exc()
    write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "EGL_PREFLIGHT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
