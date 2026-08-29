#!/usr/bin/env python3
"""Real-batch SimVLA condition-hook equivalence check.

This script loads one real LIBERO batch through the official SimVLA dataloader
and compares the official generate_actions path against decoding from the
precomputed ``enc["vlm_features"]`` condition latent.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter  # noqa: E402
from datasets import create_smolvlm_dataloader  # noqa: E402
from models.modeling_smolvlm_vla import SmolVLMVLA  # noqa: E402
from models.processing_smolvlm_vla import SmolVLMVLAProcessor  # noqa: E402


def tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach()
    stats: dict[str, Any] = {
        "shape": list(y.shape),
        "dtype": str(y.dtype),
        "device": str(y.device),
    }
    if y.numel() == 0:
        return stats
    if y.dtype == torch.bool:
        stats.update({"true_count": int(y.sum().item()), "false_count": int((~y).sum().item())})
        return stats
    if not torch.is_floating_point(y):
        y = y.float()
    else:
        y = y.float()
    stats.update(
        {
            "min": float(y.min().item()),
            "max": float(y.max().item()),
            "mean": float(y.mean().item()),
            "std": float(y.std(unbiased=False).item()) if y.numel() > 1 else 0.0,
            "norm": float(y.norm().item()),
        }
    )
    return stats


def numpy_stats(x: np.ndarray) -> dict[str, Any]:
    y = np.asarray(x)
    stats: dict[str, Any] = {"shape": list(y.shape), "dtype": str(y.dtype)}
    if y.size == 0:
        return stats
    yf = y.astype(np.float32)
    stats.update(
        {
            "min": float(yf.min()),
            "max": float(yf.max()),
            "mean": float(yf.mean()),
            "std": float(yf.std()),
        }
    )
    return stats


def describe_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return tensor_stats(value)
    if isinstance(value, np.ndarray):
        return numpy_stats(value)
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, str) for v in value):
            return {"type": type(value).__name__, "len": len(value), "preview": list(value[:3])}
        return {"type": type(value).__name__, "len": len(value)}
    return {"type": type(value).__name__, "repr": repr(value)[:500]}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_md(path: Path, title: str, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("\n".join(lines).rstrip() + "\n")


def load_raw_hdf5_preview(metas_path: Path) -> dict[str, Any]:
    import h5py

    with metas_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    first = meta["datalist"][0]
    h5_path = Path(first["path"] if isinstance(first, dict) else first)
    if not h5_path.is_absolute():
        h5_path = UPSTREAM / h5_path
    task = first.get("task", "") if isinstance(first, dict) else ""
    out: dict[str, Any] = {"hdf5_path": str(h5_path), "task": task}
    with h5py.File(h5_path, "r") as h5:
        demo_key = sorted(h5["data"].keys())[0]
        demo = h5["data"][demo_key]
        agent = np.asarray(demo["obs/agentview_rgb"][0])
        wrist = np.asarray(demo["obs/eye_in_hand_rgb"][0])
        state = np.concatenate(
            [
                np.asarray(demo["obs/ee_pos"][0]),
                np.asarray(demo["obs/ee_ori"][0]),
                np.asarray(demo["obs/gripper_states"][0]),
            ]
        )
        action = np.asarray(demo["actions"][0])
        out.update(
            {
                "demo_key": demo_key,
                "agentview_rgb": numpy_stats(agent),
                "eye_in_hand_rgb": numpy_stats(wrist),
                "raw_state_euler_concat": numpy_stats(state),
                "action": numpy_stats(action),
            }
        )
    return out


@contextmanager
def deterministic_rng(seed: int, device: torch.device):
    devices: list[int] = []
    if device.type == "cuda":
        devices = [device.index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield


def to_device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "image_input": batch["image_input"].to(device),
        "image_mask": batch["image_mask"].to(device),
        "proprio": batch["proprio"].to(device),
    }


def infer_image_input_kind(stats: dict[str, Any]) -> str:
    mn = float(stats.get("min", 0.0))
    mx = float(stats.get("max", 0.0))
    if mn < -0.1 or mx > 1.5:
        return "imagenet_normalized_preprocessed_tensor"
    if 0.0 <= mn and mx <= 1.0:
        return "float_rgb_0_1_or_padded_tensor"
    if 0.0 <= mn and mx <= 255.0:
        return "raw_or_uint8_like_rgb"
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_id", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--smolvlm_model_path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--metas_path", type=Path, default=UPSTREAM / "datasets" / "metas" / "libero_train.json")
    parser.add_argument("--norm_stats_path", type=Path, default=UPSTREAM / "norm_stats" / "libero_norm.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    os.chdir(UPSTREAM)
    dataloader = create_smolvlm_dataloader(
        batch_size=1,
        metas_path=str(args.metas_path),
        num_actions=args.steps,
        training=False,
        action_mode="libero_joint",
        num_workers=0,
        image_size=args.image_size,
    )
    batch = next(iter(dataloader))
    os.chdir(old_cwd)
    raw_preview = load_raw_hdf5_preview(args.metas_path)

    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)
    lang = processor.encode_language(batch["language_instruction"])
    model_batch_cpu = {
        "image_input": batch["image_input"],
        "image_mask": batch["image_mask"],
        "proprio": batch["proprio"],
        "action": batch.get("action"),
        "input_ids": lang["input_ids"],
    }

    batch_audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metas_path": str(args.metas_path),
        "raw_hdf5_preview": raw_preview,
        "batch_keys": sorted(list(batch.keys()) + ["input_ids"]),
        "language_instruction": describe_value(batch.get("language_instruction")),
        "tensors": {k: describe_value(v) for k, v in model_batch_cpu.items() if v is not None},
    }
    image_kind = infer_image_input_kind(batch_audit["tensors"]["image_input"])
    batch_audit["image_input_kind_inferred"] = image_kind
    write_json(out / "real_batch_format.json", batch_audit)
    write_md(
        out / "real_batch_format_audit.md",
        "Real Batch Format Audit",
        [
            f"- metas_path: `{args.metas_path}`",
            f"- batch keys: `{batch_audit['batch_keys']}`",
            f"- language preview: `{batch_audit['language_instruction'].get('preview')}`",
            f"- image_input: `{batch_audit['tensors']['image_input']}`",
            f"- inferred image_input kind: `{image_kind}`",
            f"- image_mask: `{batch_audit['tensors']['image_mask']}`",
            f"- input_ids: `{batch_audit['tensors']['input_ids']}`",
            f"- proprio: `{batch_audit['tensors']['proprio']}`",
            f"- action: `{batch_audit['tensors'].get('action')}`",
            f"- raw hdf5 preview: `{raw_preview}`",
        ],
    )

    device = torch.device(args.device)
    model = SmolVLMVLA.from_pretrained(args.checkpoint_id)
    model.to(device)
    model.eval()
    if args.norm_stats_path.exists():
        model.action_space.load_norm_stats(str(args.norm_stats_path))

    model_batch = to_device_batch({**model_batch_cpu, "input_ids": lang["input_ids"]}, device)
    action_adapter = SimVLAActionAdapter(model)

    with torch.no_grad(), deterministic_rng(args.seed, device):
        full_action = model.generate_actions(
            input_ids=model_batch["input_ids"],
            image_input=model_batch["image_input"],
            image_mask=model_batch["image_mask"],
            proprio=model_batch["proprio"],
            steps=args.steps,
        )

    with torch.no_grad(), deterministic_rng(args.seed, device):
        enc = model.forward_vlm_efficient(
            model_batch["image_input"],
            model_batch["image_mask"],
            model_batch["input_ids"],
        )
        condition = enc["vlm_features"]
        action_from_condition = action_adapter.decode_action_from_condition(
            condition,
            model_batch["proprio"],
            steps=args.steps,
            deterministic=False,
        )

    diff = (full_action - action_from_condition).detach().float()
    diff_rows = []
    flat_full = full_action.detach().cpu().float().reshape(-1)
    flat_hook = action_from_condition.detach().cpu().float().reshape(-1)
    flat_diff = diff.detach().cpu().reshape(-1)
    for i in range(flat_diff.numel()):
        diff_rows.append({"index": i, "full_action": float(flat_full[i]), "action_from_condition": float(flat_hook[i]), "abs_diff": float(abs(flat_diff[i]))})
    with (out / "condition_action_diff.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "full_action", "action_from_condition", "abs_diff"])
        writer.writeheader()
        writer.writerows(diff_rows)

    equivalence = {
        "checkpoint_id": args.checkpoint_id,
        "norm_stats_path": str(args.norm_stats_path),
        "seed": args.seed,
        "flow_steps": args.steps,
        "determinism_method": "torch.random.fork_rng + torch.manual_seed before both paths; upstream generate_actions does not accept explicit initial_noise",
        "condition": tensor_stats(condition),
        "full_action": tensor_stats(full_action),
        "action_from_condition": tensor_stats(action_from_condition),
        "action_diff": {
            "mean_abs_diff": float(diff.abs().mean().item()),
            "max_abs_diff": float(diff.abs().max().item()),
            "l2_diff": float(diff.flatten(start_dim=1).norm(dim=-1).mean().item()),
            "allclose_1e_5": bool(torch.allclose(full_action, action_from_condition, atol=1e-5, rtol=1e-5)),
            "allclose_1e_4": bool(torch.allclose(full_action, action_from_condition, atol=1e-4, rtol=1e-4)),
            "allclose_1e_3": bool(torch.allclose(full_action, action_from_condition, atol=1e-3, rtol=1e-3)),
            "allclose_1e_2": bool(torch.allclose(full_action, action_from_condition, atol=1e-2, rtol=1e-2)),
        },
    }
    write_json(out / "condition_hook_equivalence_real.json", equivalence)
    write_md(
        out / "condition_hook_equivalence_real.md",
        "Condition Hook Equivalence Real Check",
        [
            f"- checkpoint_id: `{args.checkpoint_id}`",
            f"- norm_stats_path: `{args.norm_stats_path}`",
            f"- seed: `{args.seed}`",
            f"- flow steps: `{args.steps}`",
            f"- condition stats: `{equivalence['condition']}`",
            f"- full_action stats: `{equivalence['full_action']}`",
            f"- action_from_condition stats: `{equivalence['action_from_condition']}`",
            f"- action diff: `{equivalence['action_diff']}`",
            "- deterministic limitation: upstream `generate_actions` does not accept explicit `initial_noise`; RNG state is reset before both paths.",
        ],
    )

    recommendation_lines = [
        f"- observed `image_input` kind: `{image_kind}`",
        "- recommendation: use raw RGB from LIBERO observation for DCLD visual delta, or explicitly unnormalize SimVLA `image_input` before feeding FastVisualDeltaEncoder.",
        "- reason: SimVLA `image_input` is ImageNet-normalized after resize/ToTensor; FastVisualDeltaEncoder currently expects uint8-like or 0..1 RGB and may divide normalized values by 255 when max > 2.",
        "- eval current image source: websocket observation keys `observation/image` and `observation/wrist_image` before SimVLA normalization.",
        "- eval previous/key image source: cache the previous full-refresh raw front/wrist RGB observations in the DCLD eval wrapper.",
        "- teacher-cache current/key images: read `obs/agentview_rgb` and `obs/eye_in_hand_rgb` from LIBERO HDF5 at key/current frame indices, applying the same 180-degree rotation convention as `LiberoHDF5Handler` if matching SimVLA training view orientation.",
        "- camera representation: two real cameras, agentview/front and wrist/eye-in-hand; SimVLA pads a third zero view and marks it false in `image_mask`.",
        "- value range: raw HDF5 RGB is uint8 0..255; if converted to float for DCLD use 0..1 without ImageNet normalization unless a dedicated unnormalize path is implemented.",
    ]
    write_md(out / "dcld_visual_input_recommendation.md", "DCLD Visual Input Recommendation", recommendation_lines)

    allclose_good = equivalence["action_diff"]["allclose_1e_5"] or equivalence["action_diff"]["max_abs_diff"] <= 1e-4
    final_lines = [
        f"1. Real batch loading succeeded: `True`",
        f"2. Condition hook equivalence passed: `{allclose_good}`",
        "3. Exact condition tensor: `enc[\"vlm_features\"]` from `model.forward_vlm_efficient`.",
        f"4. Exact action diff numbers: `{equivalence['action_diff']}`",
        "5. FastVisualDeltaEncoder visual input: raw RGB is recommended; preprocessed `image_input` requires unnormalize/adapter conversion.",
        "6. gate_bias added: `not checked by this script`; see gate_bias_patch_report.md after code patch.",
        "7. distill-only trainer skeleton exists: `not checked by this script`; see distill_trainer_implementation_report.md after code patch.",
        "8. upstream SimVLA files modified: `False` by this script.",
        "9. files modified by this script: report files only.",
        "10. remaining blockers: gate_bias patch, distill trainer skeleton, teacher cache generation, full eval wiring.",
    ]
    write_md(out / "final_real_condition_hook_report.md", "Final Real Condition Hook Report", final_lines)
    print(json.dumps({"output_dir": str(out), "action_diff": equivalence["action_diff"], "passed": allclose_good}, indent=2))
    return 0 if allclose_good else 2


if __name__ == "__main__":
    raise SystemExit(main())
