#!/usr/bin/env python3
"""Reset and step one real LIBERO-Long environment under a selected renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from architectures.seer.adapters.latent_bridge.rendering import renderer_gpu_device_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--renderer", choices=("egl", "osmesa"), required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    os.environ["LIBERO_GL_BACKEND"] = args.renderer
    os.environ["PYOPENGL_PLATFORM"] = args.renderer
    os.environ["MUJOCO_GL"] = args.renderer
    sys.path.insert(0, args.libero_path)
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(0)
    bddl = Path(args.libero_path) / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
    init_path = Path(args.libero_path) / "libero/libero/init_files" / task.problem_folder / task.init_states_file
    init_states = torch.load(init_path)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=args.image_size, camera_widths=args.image_size,
        render_gpu_device_id=renderer_gpu_device_id(0), control_freq=20, horizon=620,
    )
    try:
        env.reset()
        env.seed(42)
        obs = env.set_init_state(init_states[0])
        next_obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        images = {
            "primary": np.asarray(obs["agentview_image"]),
            "wrist": np.asarray(obs["robot0_eye_in_hand_image"]),
            "next_primary": np.asarray(next_obs["agentview_image"]),
        }
        expected_shape = (args.image_size, args.image_size, 3)
        if any(image.shape != expected_shape for image in images.values()):
            raise RuntimeError(f"unexpected renderer image shapes: {[v.shape for v in images.values()]}")
        if any(float(image.std()) == 0.0 for image in images.values()):
            raise RuntimeError("renderer returned a constant image")
        payload = {
            "status": "PASS",
            "suite": args.suite,
            "renderer": args.renderer,
            "image_size": args.image_size,
            "task": task.name,
            "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest(),
            "init_states_sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
            "images": {
                name: {"shape": list(value.shape), "mean": float(value.mean()), "std": float(value.std())}
                for name, value in images.items()
            },
        }
    finally:
        env.close()
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
