"""Dependency, CUDA and display checks without constructing robot interfaces."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_PACKAGES = {
    "torch": "2.6.0+cu124",
    "torchvision": "0.21.0+cu124",
    "transformers": "4.57.3",
    "numpy": "1.26.3",
    "Pillow": "12.0.0",
    "pyrealsense2": "2.57.7.10387",
    "ur-rtde": "1.6.3",
}
IMPORTS = (
    "torch", "torchvision", "transformers", "h5py", "scipy", "PIL.ImageTk",
    "cv2", "safetensors", "einops", "fastapi", "uvicorn", "json_numpy",
    "pyrealsense2", "rtde_control", "rtde_receive", "tkinter",
    "architectures.simvla.adapters.latentloop_real_deploy.controller",
    "architectures.simvla.adapters.latentloop_real_deploy.deploy_gui",
)


def _probe(code: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=90,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return {"passed": result.returncode == 0, "exit_code": result.returncode,
            "stdout": result.stdout[-12000:], "stderr": result.stderr[-12000:]}


def _video_probe() -> dict[str, Any]:
    executable = shutil.which("ffmpeg")
    if executable is None:
        return {"passed": False, "error": "ffmpeg is not on PATH"}
    with tempfile.TemporaryDirectory(prefix="simvla_video_probe_") as directory:
        output = Path(directory) / "probe.mp4"
        result = subprocess.run(
            [executable, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
             "rgb24", "-s", "64x64", "-r", "10", "-i", "-", "-an", "-c:v",
             "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(output)],
            input=bytes(64 * 64 * 3 * 2), capture_output=True, timeout=30,
        )
        size = output.stat().st_size if output.is_file() else 0
        return {"passed": result.returncode == 0 and size > 0,
                "exit_code": result.returncode, "ffmpeg": executable,
                "size_bytes": size, "stderr": result.stderr.decode(errors="replace")[-4000:]}


def inspect_environment(*, require_cuda: bool, require_gui: bool) -> dict[str, Any]:
    failures: list[str] = []
    versions: dict[str, str | None] = {}
    if sys.version_info[:2] != (3, 10):
        failures.append("Python 3.10 is required by the tested real deployment environment")
    for name, expected in EXPECTED_PACKAGES.items():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        if versions[name] != expected:
            failures.append(f"{name}: observed={versions[name]} expected={expected}")
    checks = {}
    code = "import importlib; " + "; ".join(
        f"importlib.import_module({name!r})" for name in IMPORTS
    ) + "; print('DEPENDENCY_IMPORT_PASS')"
    probes = {"imports": lambda: _probe(code), "video_encoder": _video_probe}
    if require_cuda:
        probes["cuda_bf16"] = lambda: _probe(
            "import torch,json; assert torch.cuda.is_available(); "
            "assert torch.cuda.device_count()==1, 'Expose one physical GPU'; "
            "assert torch.cuda.is_bf16_supported(); "
            "x=torch.ones((32,32),device='cuda',dtype=torch.bfloat16); "
            "assert torch.isfinite(x@x).all(); torch.cuda.synchronize(); "
            "print(json.dumps({'gpu':torch.cuda.get_device_name(0),"
            "'free_bytes':torch.cuda.mem_get_info()[0]}))"
        )
    if require_gui:
        probes["tk_display"] = lambda: _probe(
            "import tkinter as tk; root=tk.Tk(); root.withdraw(); "
            "root.update_idletasks(); root.destroy(); print('TK_DISPLAY_PASS')"
        )
    for label, run in probes.items():
        try:
            checks[label] = run()
        except Exception as error:
            checks[label] = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        if not checks[label]["passed"]:
            failures.append(f"{label} failed")
    return {
        "verdict": "REAL_ENVIRONMENT_PASS" if not failures else "REAL_ENVIRONMENT_FAIL",
        "python": sys.executable, "python_version": platform.python_version(),
        "versions": versions, "checks": checks, "failures": failures,
        "cuda_checked": require_cuda, "gui_checked": require_gui,
        "display": os.environ.get("DISPLAY"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "robot_interfaces_constructed": False, "camera_streams_started": False,
    }
