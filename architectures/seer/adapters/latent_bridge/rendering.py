"""Renderer contract shared by Latent Bridge collection and evaluation."""

from __future__ import annotations

import os


SUPPORTED_RENDERERS = {"egl", "osmesa"}


def configured_renderer_backend() -> str:
    """Return the renderer selected before LIBERO imports occur."""

    backend = os.environ.get(
        "LIBERO_GL_BACKEND",
        os.environ.get("MUJOCO_GL", os.environ.get("PYOPENGL_PLATFORM", "osmesa")),
    ).strip().lower()
    if backend not in SUPPORTED_RENDERERS:
        raise ValueError(
            f"unsupported LIBERO renderer {backend!r}; expected one of "
            f"{sorted(SUPPORTED_RENDERERS)}"
        )
    values = {
        "LIBERO_GL_BACKEND": os.environ.get("LIBERO_GL_BACKEND", "").strip().lower(),
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "").strip().lower(),
        "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM", "").strip().lower(),
    }
    conflicts = {name: value for name, value in values.items() if value and value != backend}
    if conflicts:
        raise RuntimeError(
            f"conflicting renderer environment for selected backend {backend!r}: {conflicts}"
        )
    return backend


def renderer_gpu_device_id(local_device_id: int) -> int:
    """Map a torch local rank to the physical device required by old robosuite EGL."""

    if configured_renderer_backend() != "egl":
        return int(local_device_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return int(local_device_id)
    try:
        physical_ids = [int(item.strip()) for item in visible.split(",")]
    except ValueError as exc:
        raise RuntimeError(
            "this robosuite EGL backend requires numeric CUDA_VISIBLE_DEVICES entries; "
            f"got {visible!r}"
        ) from exc
    if not 0 <= int(local_device_id) < len(physical_ids):
        raise RuntimeError(
            f"local render device {local_device_id} is outside CUDA_VISIBLE_DEVICES={visible!r}"
        )
    return physical_ids[int(local_device_id)]
