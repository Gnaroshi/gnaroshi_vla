#!/usr/bin/env python3
"""Render fixed LIBERO-Object initial states and record image statistics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def _decode_gl_string(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _image_record(array: np.ndarray) -> dict:
    rgb = np.asarray(array, dtype=np.uint8)
    gray = rgb.astype(np.float32).mean(axis=-1)
    return {
        "shape": list(rgb.shape),
        "mean_rgb": [float(value) for value in rgb.mean(axis=(0, 1))],
        "mean_intensity": float(gray.mean()),
        "dark_pixel_fraction_lt_32": float((gray < 32.0).mean()),
        "sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-gpu-device-id", type=int, required=True)
    parser.add_argument("--expected-mujoco", default="3.3.2")
    parser.add_argument("--minimum-primary-mean", type=float, default=125.0)
    args = parser.parse_args()

    actual_mujoco = importlib.metadata.version("mujoco")
    if actual_mujoco != args.expected_mujoco:
        raise RuntimeError(
            f"MuJoCo mismatch: expected {args.expected_mujoco}, got {actual_mujoco}"
        )
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    from OpenGL.GL import GL_RENDERER, GL_VENDOR, GL_VERSION, glGetString
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    records = []
    renderer = None
    for task_id in range(suite.get_num_tasks()):
        task = suite.get_task(task_id)
        bddl = (
            args.libero_path
            / "libero/libero/bddl_files"
            / task.problem_folder
            / task.bddl_file
        )
        init_path = (
            args.libero_path
            / "libero/libero/init_files"
            / task.problem_folder
            / task.init_states_file
        )
        init_state = torch.load(init_path)[0]
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
            render_gpu_device_id=args.render_gpu_device_id,
            control_freq=20,
            horizon=1000,
        )
        try:
            env.reset()
            if renderer is None:
                context = env.sim._render_context_offscreen.gl_ctx
                context.make_current()
                renderer = {
                    "vendor": _decode_gl_string(glGetString(GL_VENDOR)),
                    "renderer": _decode_gl_string(glGetString(GL_RENDERER)),
                    "version": _decode_gl_string(glGetString(GL_VERSION)),
                    "render_gpu_device_id": args.render_gpu_device_id,
                }
            obs = env.set_init_state(init_state)
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
            task_record = {"task_id": task_id, "task_name": task.name}
            for camera_key, short_name in (
                ("agentview_image", "primary"),
                ("robot0_eye_in_hand_image", "wrist"),
            ):
                rgb = np.flipud(np.asarray(obs[camera_key], dtype=np.uint8))
                Image.fromarray(rgb).save(
                    args.output_dir / f"task{task_id:02d}_{short_name}.png"
                )
                task_record[short_name] = _image_record(rgb)
            records.append(task_record)
        finally:
            env.close()

    primary_mean = float(
        np.mean([record["primary"]["mean_intensity"] for record in records])
    )
    wrist_mean = float(
        np.mean([record["wrist"]["mean_intensity"] for record in records])
    )
    payload = {
        "status": "PASS" if primary_mean >= args.minimum_primary_mean else "FAIL",
        "versions": {
            "mujoco": actual_mujoco,
            "robosuite": importlib.metadata.version("robosuite"),
            "libero": importlib.metadata.version("libero"),
        },
        "renderer_backend": "egl",
        "renderer": renderer,
        "suite": "libero_object",
        "aggregate": {
            "primary_mean_intensity": primary_mean,
            "wrist_mean_intensity": wrist_mean,
            "minimum_primary_mean": args.minimum_primary_mean,
        },
        "records": records,
    }
    output = args.output_dir / "render_metrics.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError(
            "LIBERO-Object render is darker than the locked MuJoCo 3.3.2 threshold: "
            f"primary_mean={primary_mean:.3f}, minimum={args.minimum_primary_mean:.3f}"
        )


if __name__ == "__main__":
    main()
