#!/usr/bin/env python3
"""Saved-batch equivalence diagnostic for official SimVLA vs wrapper full path.

This script intentionally does not run DCLD K-sweeps. It records a small set of
official-style LIBERO policy-query batches and compares:

1. official client resize vs wrapper resize
2. official server tensors vs wrapper tensors
3. model.generate_actions(...) vs condition/decode adapter on identical tensors

The official generate_actions API does not expose initial noise, so this script
patches only the matching torch.randn(shape=(B, num_actions, dim_action)) call
inside generate_actions to inject a saved initial_noise tensor.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
LIBERO_EVAL = UPSTREAM / "evaluation" / "libero"
LIBERO_ROOT = LIBERO_EVAL / "LIBERO"

for path in [ROOT, UPSTREAM, LIBERO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter, SimVLAConditionAdapter  # noqa: E402
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (  # noqa: E402
    build_env_obs,
    command_output,
    get_libero_env,
    resize_with_pad_uint8,
    stable_seed,
    tensor_action_diff,
)
from models.modeling_smolvlm_vla import SmolVLMVLA  # noqa: E402
from models.processing_smolvlm_vla import SmolVLMVLAProcessor  # noqa: E402

try:
    from openpi_client import image_tools

    HAS_OPENPI_CLIENT = True
except Exception:
    image_tools = None
    HAS_OPENPI_CLIENT = False


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def array_diff(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    row: dict[str, Any] = {
        "shape_a": list(a_arr.shape),
        "shape_b": list(b_arr.shape),
        "dtype_a": str(a_arr.dtype),
        "dtype_b": str(b_arr.dtype),
        "shape_match": tuple(a_arr.shape) == tuple(b_arr.shape),
        "dtype_match": str(a_arr.dtype) == str(b_arr.dtype),
        "min_a": float(a_arr.min()) if a_arr.size else None,
        "max_a": float(a_arr.max()) if a_arr.size else None,
        "min_b": float(b_arr.min()) if b_arr.size else None,
        "max_b": float(b_arr.max()) if b_arr.size else None,
    }
    if tuple(a_arr.shape) != tuple(b_arr.shape):
        row.update(
            {
                "exact_equal": False,
                "allclose_1e_5": False,
                "mean_abs_diff": None,
                "max_abs_diff": None,
            }
        )
        return row
    diff = np.abs(a_arr.astype(np.float64) - b_arr.astype(np.float64))
    row.update(
        {
            "exact_equal": bool(np.array_equal(a_arr, b_arr)),
            "allclose_1e_5": bool(np.allclose(a_arr, b_arr, atol=1e-5, rtol=1e-5)),
            "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
            "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        }
    )
    return row


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    return tensor_action_diff(a.detach(), b.detach())


def official_client_resize(image: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return np.ascontiguousarray(image.astype(np.uint8))
    if not HAS_OPENPI_CLIENT:
        raise RuntimeError("openpi_client.image_tools is unavailable")
    resized = image_tools.resize_with_pad(image, size, size)
    return np.ascontiguousarray(image_tools.convert_to_uint8(resized))


def server_preprocess_images(image0: np.ndarray, image1: np.ndarray, image_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Local copy of official serve_smolvlm_libero.preprocess_images."""

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    img0_t = transform(Image.fromarray(image0.astype(np.uint8)))
    img1_t = transform(Image.fromarray(image1.astype(np.uint8)))
    padding = torch.zeros_like(img0_t)
    images = torch.stack([img0_t, img1_t, padding], dim=0).unsqueeze(0).to(device)
    image_mask = torch.tensor([[True, True, False]], device=device)
    return images, image_mask


def wrapper_preprocess(
    image0: np.ndarray,
    image1: np.ndarray,
    proprio: np.ndarray,
    prompt: str,
    *,
    processor: Any,
    image_size: int,
    client_resize_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor | np.ndarray]:
    if client_resize_size > 0:
        image0 = resize_with_pad_uint8(image0, client_resize_size)
        image1 = resize_with_pad_uint8(image1, client_resize_size)
    image_input, image_mask = server_preprocess_images(image0, image1, image_size, device)
    lang = processor.encode_language([prompt])
    input_ids = lang["input_ids"].to(device)
    proprio_t = torch.as_tensor(proprio, dtype=torch.float32, device=device).reshape(1, -1)[:, :8]
    return {
        "resized_image": image0,
        "resized_wrist_image": image1,
        "image_input": image_input,
        "image_mask": image_mask,
        "input_ids": input_ids,
        "proprio": proprio_t,
    }


@contextlib.contextmanager
def patch_randn_for_initial_noise(initial_noise: torch.Tensor) -> Iterator[dict[str, Any]]:
    original_randn = torch.randn
    state = {"count": 0, "shapes": []}
    expected_shape = tuple(initial_noise.shape)

    def patched_randn(*size: Any, **kwargs: Any) -> torch.Tensor:
        if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)):
            shape = tuple(size[0])
        else:
            shape = tuple(size)
        state["shapes"].append(list(shape))
        if shape == expected_shape:
            state["count"] += 1
            device = kwargs.get("device", initial_noise.device)
            dtype = kwargs.get("dtype", initial_noise.dtype)
            return initial_noise.to(device=device, dtype=dtype).clone()
        return original_randn(*size, **kwargs)

    torch.randn = patched_randn
    try:
        yield state
    finally:
        torch.randn = original_randn


def collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import mujoco
    except Exception:
        mujoco = None
    try:
        import robosuite
    except Exception:
        robosuite = None

    cuda_name = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(torch.device(args.device))

    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "hostname": socket.gethostname(),
        "root": str(ROOT),
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": cuda_name,
        "torch_version": torch.__version__,
        "mujoco_version": getattr(mujoco, "__version__", None) if mujoco else package_version("mujoco"),
        "robosuite_version": getattr(robosuite, "__version__", None) if robosuite else package_version("robosuite"),
        "libero_path": str(LIBERO_ROOT),
        "checkpoint": args.checkpoint,
        "norm_stats": args.norm_stats,
        "root_git_branch": command_output(["git", "branch", "--show-current"], cwd=ROOT),
        "root_git_head": command_output(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "root_git_status_short": command_output(["git", "status", "--short"], cwd=ROOT),
        "simvla_upstream_git_head": command_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], cwd=ROOT),
        "simvla_upstream_git_status_short": command_output(["git", "-C", str(UPSTREAM), "status", "--short"], cwd=ROOT),
        "openpi_client_available": HAS_OPENPI_CLIENT,
        "openpi_client_image_tools": None if not HAS_OPENPI_CLIENT else str(getattr(image_tools, "__file__", "")),
    }


def write_static_audits(out: Path) -> None:
    write_text(
        out / "official_action_path_audit.md",
        "\n".join(
            [
                "# Official SimVLA Action Path Audit",
                "",
                "- client file: `architectures/simvla/upstream/evaluation/libero/libero_client.py`",
                "- server file: `architectures/simvla/upstream/evaluation/libero/serve_smolvlm_libero.py`",
                "- model file: `architectures/simvla/upstream/models/modeling_smolvlm_vla.py`",
                "- official WebSocket client reads rotated `obs['image']` and `obs['wrist_image']`.",
                "- client applies `openpi_client.image_tools.resize_with_pad(image, 224, 224)` and then `convert_to_uint8`.",
                "- client sends `observation/image`, `observation/wrist_image`, `observation/state`, and `prompt`.",
                "- server converts received uint8 arrays through PIL, resizes to `(384, 384)`, `ToTensor`, and ImageNet normalization.",
                "- server stacks `[agentview, wrist, zero_padding]` and uses `image_mask=[[True, True, False]]`.",
                "- prompt is encoded through `SmolVLMVLAProcessor.encode_language([prompt])`.",
                "- state vector is `[robot0_eef_pos(3), quat2axisangle(robot0_eef_quat)(3), robot0_gripper_qpos(2)]`.",
                "- action path is `model.generate_actions(input_ids, image_input, image_mask, proprio, steps=10)`.",
                "- `generate_actions` samples `torch.randn(B, num_actions, dim_action)` internally; it has no explicit initial-noise argument.",
                "- action chunk is postprocessed by `model.action_space.postprocess(x_t)` and the client executes the first `replan_steps=5` actions.",
            ]
        ),
    )
    write_text(
        out / "wrapper_full_path_audit.md",
        "\n".join(
            [
                "# Wrapper Full Path Audit",
                "",
                "- wrapper file: `architectures/simvla/wrappers/simvla_dcld_eval.py`",
                "- condition adapter: `architectures/simvla/adapters/dcld/simvla_condition_adapter.py`",
                "- action adapter: `architectures/simvla/adapters/dcld/simvla_action_adapter.py`",
                "- wrapper builds env observation with the same rotated agentview/wrist images and the same 8D state construction.",
                "- wrapper full path currently uses local `resize_with_pad_uint8(image, 224)` before the same PIL/384/ImageNet transform.",
                "- prompt is encoded with `SmolVLMVLAProcessor.encode_language([prompt])`.",
                "- condition extraction is `model.forward_vlm_efficient(image_input, image_mask, input_ids)['vlm_features']`.",
                "- action decoding is `SimVLAActionAdapter.decode_action_from_condition(condition, proprio, steps=flow_steps, initial_noise=...)`.",
                "- for full K=1 rows DCLD is inactive; no DCLD update or fast encoder call should occur.",
                "- action queue executes the first `replan_steps=5` actions from each 10-step chunk.",
            ]
        ),
    )
    write_text(
        out / "eval_semantics_comparison.md",
        "\n".join(
            [
                "# Eval Semantics Comparison",
                "",
                "- task order used here: `official_reverse` -> task ids `[9, 8, ..., 0]` for `libero_10`.",
                "- wait steps: `10` dummy actions before policy control.",
                "- max policy steps: wrapper calibration uses `900` for `libero_10`, matching official max-step setting.",
                "- replan steps: `5`; a policy query returns 10 actions but only the first 5 are queued/executed before replanning.",
                "- client resize size: `224`, then server/model transform size: `384`.",
                "- this diagnostic records saved policy-query batches only; it does not run DCLD K-sweep or full benchmark evaluation.",
            ]
        ),
    )


def make_initial_noise(
    *,
    seed_base: int,
    suite: str,
    task_id: int,
    trial_id: int,
    query_index: int,
    flow_steps: int,
    num_actions: int,
    dim_action: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    seed = stable_seed(seed_base, suite, task_id, trial_id, query_index, flow_steps, num_actions, dim_action)
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return torch.randn((1, num_actions, dim_action), device=device, dtype=dtype, generator=generator), seed


def save_batch(
    path: Path,
    *,
    raw_image0: np.ndarray,
    raw_image1: np.ndarray,
    official_batch: dict[str, Any],
    wrapper_batch: dict[str, Any],
    initial_noise: torch.Tensor,
    official_action: torch.Tensor,
    wrapper_action: torch.Tensor,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "metadata": metadata,
        "raw_image0": raw_image0,
        "raw_image1": raw_image1,
        "official_resized_image": official_batch["resized_image"],
        "official_resized_wrist_image": official_batch["resized_wrist_image"],
        "wrapper_resized_image": wrapper_batch["resized_image"],
        "wrapper_resized_wrist_image": wrapper_batch["resized_wrist_image"],
        "official_image_input": official_batch["image_input"].detach().cpu(),
        "wrapper_image_input": wrapper_batch["image_input"].detach().cpu(),
        "official_image_mask": official_batch["image_mask"].detach().cpu(),
        "wrapper_image_mask": wrapper_batch["image_mask"].detach().cpu(),
        "official_input_ids": official_batch["input_ids"].detach().cpu(),
        "wrapper_input_ids": wrapper_batch["input_ids"].detach().cpu(),
        "official_proprio": official_batch["proprio"].detach().cpu(),
        "wrapper_proprio": wrapper_batch["proprio"].detach().cpu(),
        "initial_noise": initial_noise.detach().cpu(),
        "official_action": official_action.detach().cpu(),
        "wrapper_action": wrapper_action.detach().cpu(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "path": str(path),
        "metadata": metadata,
        "shapes": {
            key: list(value.shape)
            for key, value in payload.items()
            if hasattr(value, "shape")
        },
    }


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_static_audits(out)
    write_json(out / "environment_metadata.json", collect_environment(args))

    from libero.libero import benchmark

    os.environ.setdefault("LIBERO_ROOT", str(LIBERO_ROOT))
    device = torch.device(args.device)
    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device)
    model.eval()
    if args.norm_stats and Path(args.norm_stats).exists():
        model.action_space.load_norm_stats(args.norm_stats)
    for param in model.parameters():
        param.requires_grad_(False)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)
    condition_adapter = SimVLAConditionAdapter(model)
    action_adapter = SimVLAActionAdapter(model)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    requested_task_ids = [int(x) for x in args.task_ids.split(",") if x]

    resize_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    shape_manifest: list[dict[str, Any]] = []
    saved_dir = out / "saved_official_style_batches"
    saved_dir.mkdir(parents=True, exist_ok=True)

    for task_id in requested_task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, args.resolution, args.seed)
        try:
            env.reset()
            obs = env.set_init_state(initial_states[args.trial_id % len(initial_states)])
            for _ in range(args.num_wait_steps):
                obs, _reward, _done, _info = env.step([0.0] * 6 + [-1.0])

            query_index = 0
            while query_index < args.queries_per_task:
                raw_image0, raw_image1, state = build_env_obs(obs)

                official_image0 = official_client_resize(raw_image0, args.client_resize_size)
                official_image1 = official_client_resize(raw_image1, args.client_resize_size)
                wrapper_image0 = resize_with_pad_uint8(raw_image0, args.client_resize_size)
                wrapper_image1 = resize_with_pad_uint8(raw_image1, args.client_resize_size)

                for view_name, off_img, wrap_img in [
                    ("agentview", official_image0, wrapper_image0),
                    ("wrist", official_image1, wrapper_image1),
                ]:
                    row = {
                        "suite": args.suite,
                        "task_id": task_id,
                        "trial_id": args.trial_id,
                        "policy_query_index": query_index,
                        "view": view_name,
                    }
                    row.update(array_diff(off_img, wrap_img))
                    resize_rows.append(row)

                official_image_input, official_image_mask = server_preprocess_images(official_image0, official_image1, args.image_size, device)
                lang = processor.encode_language([task_description])
                official_input_ids = lang["input_ids"].to(device)
                official_proprio = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, -1)[:, :8]
                official_batch = {
                    "resized_image": official_image0,
                    "resized_wrist_image": official_image1,
                    "image_input": official_image_input,
                    "image_mask": official_image_mask,
                    "input_ids": official_input_ids,
                    "proprio": official_proprio,
                }
                wrapper_batch = wrapper_preprocess(
                    raw_image0,
                    raw_image1,
                    state,
                    task_description,
                    processor=processor,
                    image_size=args.image_size,
                    client_resize_size=args.client_resize_size,
                    device=device,
                )

                state_row = {
                    "suite": args.suite,
                    "task_id": task_id,
                    "trial_id": args.trial_id,
                    "policy_query_index": query_index,
                    "state_dim": int(state.shape[0]),
                    "state_dtype": str(state.dtype),
                    "state_values": json.dumps([float(x) for x in state.tolist()]),
                    "official_wrapper_state_allclose_1e_8": bool(
                        torch.allclose(official_batch["proprio"], wrapper_batch["proprio"], atol=1e-8, rtol=1e-8)
                    ),
                }
                state_rows.append(state_row)

                initial_noise, action_seed = make_initial_noise(
                    seed_base=args.action_noise_seed_base,
                    suite=args.suite,
                    task_id=task_id,
                    trial_id=args.trial_id,
                    query_index=query_index,
                    flow_steps=args.flow_steps,
                    num_actions=action_adapter.num_actions,
                    dim_action=action_adapter.dim_action,
                    device=device,
                    dtype=official_batch["proprio"].dtype,
                )

                with torch.no_grad(), patch_randn_for_initial_noise(initial_noise) as patch_state:
                    official_action = model.generate_actions(
                        input_ids=official_batch["input_ids"],
                        image_input=official_batch["image_input"],
                        image_mask=official_batch["image_mask"],
                        proprio=official_batch["proprio"],
                        steps=args.flow_steps,
                    )
                with torch.no_grad():
                    wrapper_condition = condition_adapter.encode_condition(
                        input_ids=wrapper_batch["input_ids"],
                        image_input=wrapper_batch["image_input"],
                        image_mask=wrapper_batch["image_mask"],
                    )
                    wrapper_out = action_adapter.decode_action_from_condition(
                        wrapper_condition,
                        wrapper_batch["proprio"],
                        steps=args.flow_steps,
                        initial_noise=initial_noise,
                        return_debug=True,
                    )
                    wrapper_action = wrapper_out.action
                    official_condition = model.forward_vlm_efficient(
                        official_batch["image_input"],
                        official_batch["image_mask"],
                        official_batch["input_ids"],
                    )["vlm_features"]

                record_id = f"task{task_id:02d}_trial{args.trial_id:02d}_query{query_index:03d}"
                metadata = {
                    "record_id": record_id,
                    "suite": args.suite,
                    "task_id": task_id,
                    "trial_id": args.trial_id,
                    "policy_query_index": query_index,
                    "task_description": task_description,
                    "action_noise_seed": int(action_seed),
                    "randn_patch_count": int(patch_state["count"]),
                    "randn_patch_shapes": patch_state["shapes"],
                    "replan_steps": args.replan_steps,
                    "flow_steps": args.flow_steps,
                    "client_resize_size": args.client_resize_size,
                    "image_size": args.image_size,
                }
                saved = save_batch(
                    saved_dir / f"{record_id}.pt",
                    raw_image0=raw_image0,
                    raw_image1=raw_image1,
                    official_batch=official_batch,
                    wrapper_batch=wrapper_batch,
                    initial_noise=initial_noise,
                    official_action=official_action,
                    wrapper_action=wrapper_action,
                    metadata=metadata,
                )
                manifest.append({"record_id": record_id, **metadata, "path": saved["path"]})
                shape_manifest.append(saved)

                action_row: dict[str, Any] = {
                    "record_id": record_id,
                    "suite": args.suite,
                    "task_id": task_id,
                    "trial_id": args.trial_id,
                    "policy_query_index": query_index,
                    "action_noise_seed": int(action_seed),
                    "randn_patch_count": int(patch_state["count"]),
                    "official_wrapper_image_input_allclose_1e_5": bool(
                        torch.allclose(official_batch["image_input"], wrapper_batch["image_input"], atol=1e-5, rtol=1e-5)
                    ),
                    "official_wrapper_input_ids_equal": bool(
                        torch.equal(official_batch["input_ids"], wrapper_batch["input_ids"])
                    ),
                    "official_wrapper_image_mask_equal": bool(
                        torch.equal(official_batch["image_mask"], wrapper_batch["image_mask"])
                    ),
                    "official_wrapper_proprio_allclose_1e_8": bool(
                        torch.allclose(official_batch["proprio"], wrapper_batch["proprio"], atol=1e-8, rtol=1e-8)
                    ),
                }
                action_diff = tensor_diff(official_action, wrapper_action)
                action_row.update({f"action_{key}": value for key, value in action_diff.items()})
                condition_diff = tensor_diff(official_condition, wrapper_condition)
                action_row.update({f"condition_{key}": value for key, value in condition_diff.items()})
                action_rows.append(action_row)

                # Step with the official full action chunk to collect more real observations.
                for action_idx in range(min(args.replan_steps, official_action.shape[1])):
                    obs, _reward, done, _info = env.step(official_action[0, action_idx].detach().cpu().numpy().tolist())
                    if done:
                        break
                query_index += 1
                if done:
                    break
        finally:
            env.close()

    csv_write(out / "resize_byte_comparison.csv", resize_rows)
    csv_write(out / "state_vector_comparison.csv", state_rows)
    csv_write(out / "official_vs_wrapper_action_diff.csv", action_rows)
    write_json(out / "saved_batch_manifest.json", manifest)
    write_json(out / "saved_batch_shapes.json", shape_manifest)

    resize_exact = bool(resize_rows) and all(bool(row.get("exact_equal")) for row in resize_rows)
    tensors_equal = bool(action_rows) and all(
        bool(row.get("official_wrapper_image_input_allclose_1e_5"))
        and bool(row.get("official_wrapper_input_ids_equal"))
        and bool(row.get("official_wrapper_image_mask_equal"))
        and bool(row.get("official_wrapper_proprio_allclose_1e_8"))
        for row in action_rows
    )
    actions_equal = bool(action_rows) and all(bool(row.get("action_allclose_1e_5")) for row in action_rows)
    conditions_equal = bool(action_rows) and all(bool(row.get("condition_allclose_1e_5")) for row in action_rows)
    randn_patch_ok = bool(action_rows) and all(int(row.get("randn_patch_count", 0)) == 1 for row in action_rows)

    summary = {
        "num_resize_rows": len(resize_rows),
        "resize_exact_equal_all": resize_exact,
        "num_saved_batches": len(manifest),
        "num_action_rows": len(action_rows),
        "tensors_equal_all": tensors_equal,
        "conditions_allclose_1e_5_all": conditions_equal,
        "actions_allclose_1e_5_all": actions_equal,
        "randn_patch_count_one_all": randn_patch_ok,
        "max_action_diff": None if not action_rows else max(float(row.get("action_max_abs_diff") or 0.0) for row in action_rows),
        "max_condition_diff": None if not action_rows else max(float(row.get("condition_max_abs_diff") or 0.0) for row in action_rows),
    }
    if not HAS_OPENPI_CLIENT:
        verdict = "WRAPPER_OFFICIAL_EQUIVALENCE_INCONCLUSIVE"
    elif tensors_equal and actions_equal and randn_patch_ok:
        verdict = "WRAPPER_OFFICIAL_EQUIVALENCE_PASSED"
    else:
        verdict = "WRAPPER_OFFICIAL_EQUIVALENCE_FAILED_NEEDS_PATCH"
    summary["verdict"] = verdict
    write_json(out / "official_vs_wrapper_action_diff_summary.json", summary)

    write_text(
        out / "resize_byte_comparison_summary.md",
        "\n".join(
            [
                "# Resize Byte Comparison Summary",
                "",
                f"- rows: `{len(resize_rows)}`",
                f"- openpi_client_available: `{HAS_OPENPI_CLIENT}`",
                f"- all_exact_equal: `{resize_exact}`",
                f"- max_abs_diff_max: `{None if not resize_rows else max(float(row.get('max_abs_diff') or 0.0) for row in resize_rows)}`",
                "",
                "If this fails, the wrapper's local resize is not byte-equivalent to the official WebSocket client resize.",
            ]
        ),
    )
    write_text(
        out / "state_vector_comparison_report.md",
        "\n".join(
            [
                "# State Vector Comparison",
                "",
                f"- rows: `{len(state_rows)}`",
                f"- all official/wrapper proprio equal: `{all(bool(row.get('official_wrapper_state_allclose_1e_8')) for row in state_rows) if state_rows else False}`",
                "- construction: `[robot0_eef_pos, quat2axisangle(robot0_eef_quat), robot0_gripper_qpos]`",
            ]
        ),
    )
    write_text(
        out / "official_vs_wrapper_action_diff_report.md",
        "\n".join(
            [
                "# Official vs Wrapper Action Diff Report",
                "",
                f"- saved_batches: `{len(manifest)}`",
                f"- tensors_equal_all: `{tensors_equal}`",
                f"- conditions_allclose_1e_5_all: `{conditions_equal}`",
                f"- actions_allclose_1e_5_all: `{actions_equal}`",
                f"- randn_patch_count_one_all: `{randn_patch_ok}`",
                f"- max_action_diff: `{summary['max_action_diff']}`",
                f"- max_condition_diff: `{summary['max_condition_diff']}`",
                f"- verdict: `{verdict}`",
                "",
                "The official path is `model.generate_actions(...)`; the wrapper path is `forward_vlm_efficient(...)` plus `SimVLAActionAdapter.decode_action_from_condition(...)`.",
            ]
        ),
    )

    recommendations: list[str] = []
    if not resize_exact:
        recommendations.append(
            "Patch wrapper preprocessing to call `openpi_client.image_tools.resize_with_pad` + `convert_to_uint8` for official WebSocket parity, or explicitly document non-official resize."
        )
    if resize_exact and not actions_equal:
        recommendations.append(
            "Investigate action adapter loop/proprio normalization/postprocess parity; resize is not the blocking factor."
        )
    if not randn_patch_ok:
        recommendations.append(
            "The official generate_actions random-noise interception did not trigger exactly once; add a real explicit initial_noise argument upstream or rerun with a stricter hook."
        )
    if not recommendations:
        recommendations.append("No patch recommendation from saved-batch equivalence; proceed only after benchmark-level calibration is accepted.")
    write_text(out / "patch_recommendations.md", "\n".join(["# Patch Recommendations", "", *[f"- {item}" for item in recommendations]]))

    rerun_command = f"""cd {ROOT}
conda activate simvla_libero
export CUDA_VISIBLE_DEVICES=4
export HF_HOME={ROOT}/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR=/tmp/numba_cache
export MPLCONFIGDIR=/tmp/matplotlib-${{USER}}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python architectures/simvla/wrappers/simvla_official_vs_wrapper_equivalence.py \\
  --output {out} \\
  --checkpoint {args.checkpoint} \\
  --norm-stats {args.norm_stats} \\
  --suite {args.suite} \\
  --task-ids {args.task_ids} \\
  --queries-per-task {args.queries_per_task} \\
  --device {args.device}
"""
    write_text(out / "rerun_command.md", "```bash\n" + rerun_command.rstrip() + "\n```")

    write_text(
        out / "final_official_vs_wrapper_equivalence_report.md",
        "\n".join(
            [
                "# Final Official vs Wrapper Equivalence Report",
                "",
                f"- verdict: `{verdict}`",
                f"- official upstream direct libero_10 reference: `94/100`",
                f"- previous wrapper K1 unpaired calibration: `baseline_full_k1=90/100`, `ours_full_k1=94/100`",
                f"- saved batches: `{len(manifest)}`",
                f"- resize_exact_equal_all: `{resize_exact}`",
                f"- tensors_equal_all: `{tensors_equal}`",
                f"- conditions_allclose_1e_5_all: `{conditions_equal}`",
                f"- actions_allclose_1e_5_all: `{actions_equal}`",
                f"- randn_patch_count_one_all: `{randn_patch_ok}`",
                f"- max_action_diff: `{summary['max_action_diff']}`",
                f"- max_condition_diff: `{summary['max_condition_diff']}`",
                "",
                "## Interpretation",
                "",
                "This diagnostic isolates single policy-query computation from benchmark rollout stochasticity. It does not prove end-to-end benchmark calibration by itself.",
                "If resize or tensor parity fails, the wrapper cannot yet be treated as the official SimVLA path.",
                "If saved-batch parity passes but rollout score remains lower, the remaining issue is benchmark/runtime semantics rather than the action-head replacement path.",
                "",
                "## DCLD K Sweep Status",
                "",
                f"- allowed_by_this_diagnostic: `{verdict == 'WRAPPER_OFFICIAL_EQUIVALENCE_PASSED'}`",
                "- note: benchmark-level calibration should still be resolved before making paper claims against the official 94/100 `libero_10` baseline.",
            ]
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--smolvlm-model-path", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--norm-stats", type=str, default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--suite", type=str, default="libero_10")
    parser.add_argument("--task-ids", type=str, default="8,9,2")
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--queries-per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--client-resize-size", type=int, default=224)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--action-noise-seed-base", type=int, default=20260708)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
