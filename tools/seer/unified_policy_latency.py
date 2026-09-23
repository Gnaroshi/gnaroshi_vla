#!/usr/bin/env python3
"""Unified model-path latency benchmark for Seer efficiency methods.

The benchmark keeps one Seer instance and one set of cached, preprocessed
LIBERO inputs resident on a single GPU. It switches only the method path and
measures complete action schedules with synchronized CUDA wall time and CUDA
events. Simulator stepping, rendering, disk I/O, and host preprocessing are
outside the timing boundary.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "architectures/seer/upstream"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: str | Path, expected: str, label: str) -> str:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {label}: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected={expected}, actual={actual}, path={path}"
        )
    return actual


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _git_state(path: Path) -> dict:
    return {
        "root": str(path.resolve()),
        "commit": subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "-C", str(path), "branch", "--show-current"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"], text=True
            ).strip()
        ),
    }


@contextlib.contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def _load_model(args, device: torch.device):
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    with _working_directory(UPSTREAM):
        from models.seer_model import SeerAgent

        model = SeerAgent(
            finetune_type="libero_finetune",
            clip_device=str(device),
            vit_checkpoint_path=args.vit_checkpoint,
            sequence_length=7,
            num_resampler_query=6,
            num_obs_token_per_image=9,
            obs_pred=True,
            atten_only_obs=False,
            attn_robot_proprio_state=False,
            atten_goal=False,
            atten_goal_state=False,
            mask_l_obs_ratio=0.0,
            calvin_input_image_size=224,
            patch_size=16,
            action_pred_steps=3,
            transformer_layers=24,
            hidden_dim=384,
            transformer_heads=12,
            phase="evaluate",
            gripper_width=True,
            use_lrnode_latent_update=1,
            lrnode_hidden_dim=256,
            lrnode_motion_dim=128,
            lrnode_fast_encoder_type="diffcnn",
            lrnode_gate_init_bias=-4.0,
            vla_cache_mode="off",
        )

    base_payload = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" not in base_payload:
        raise KeyError("Seer checkpoint lacks model_state_dict")
    base_state = _strip_module_prefix(base_payload["model_state_dict"])
    base_load = model.load_state_dict(base_state, strict=False)
    disallowed_missing = [
        key
        for key in base_load.missing_keys
        if not key.startswith(
            (
                "vision_encoder.",
                "clip_model.",
                "attention_mask",
                "image_decoder_position_embedding",
                "lrnode_delta_encoder.",
                "lrnode_dynamics.",
            )
        )
    ]
    if disallowed_missing or base_load.unexpected_keys:
        raise RuntimeError(
            "base checkpoint load contract failed: "
            f"missing={disallowed_missing}, unexpected={base_load.unexpected_keys}"
        )

    adapter_payload = torch.load(args.latentloop_checkpoint, map_location="cpu")
    if "model_state_dict" not in adapter_payload:
        raise KeyError("LatentLoop checkpoint lacks model_state_dict")
    raw_adapter = adapter_payload["model_state_dict"]
    allowed_raw_prefixes = (
        "module.lrnode_delta_encoder.",
        "module.lrnode_dynamics.",
        "lrnode_delta_encoder.",
        "lrnode_dynamics.",
    )
    invalid = sorted(key for key in raw_adapter if not key.startswith(allowed_raw_prefixes))
    if invalid:
        raise RuntimeError(f"LatentLoop checkpoint contains non-adapter keys: {invalid[:8]}")
    adapter_state = _strip_module_prefix(raw_adapter)
    model_state = model.state_dict()
    missing_adapter_targets = sorted(set(adapter_state) - set(model_state))
    shape_mismatches = sorted(
        key
        for key, value in adapter_state.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
    )
    if missing_adapter_targets or shape_mismatches:
        raise RuntimeError(
            "LatentLoop adapter/model mismatch: "
            f"missing_targets={missing_adapter_targets}, shape_mismatches={shape_mismatches}"
        )
    adapter_load = model.load_state_dict(adapter_state, strict=False)
    if adapter_load.unexpected_keys:
        raise RuntimeError(f"unexpected LatentLoop keys: {adapter_load.unexpected_keys}")
    loaded_state = model.state_dict()
    for key, expected in adapter_state.items():
        if not torch.equal(loaded_state[key].cpu(), expected.cpu()):
            raise RuntimeError(f"LatentLoop tensor did not load exactly: {key}")

    model.float()
    model.vision_encoder.bfloat16()
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    model.profile_full_action_head = False
    model._init_model_type()
    return model, {
        "base_loaded_tensor_count": len(base_state),
        "base_allowed_missing_keys": list(base_load.missing_keys),
        "adapter_tensor_count": len(adapter_state),
        "adapter_numel": int(sum(value.numel() for value in adapter_state.values())),
        "adapter_unexpected_keys": list(adapter_load.unexpected_keys),
    }


def _load_shifted_inputs(args, model, device: torch.device, count: int):
    with _working_directory(UPSTREAM):
        import clip
        from utils.data_utils import get_libero_finetune_dataset

        dataset_args = SimpleNamespace(
            rgb_pad=-1,
            gripper_pad=-1,
            traj_cons=True,
            text_aug=False,
            multi_step_action=1,
            root_dir=args.dataset_root,
            image_primary_size=224,
            image_wrist_size=224,
            window_size=7,
            dif_ws=False,
            min_window_size=7,
            max_window_size=7,
            primary_mode="image_primary",
            small_size=0,
            gripper_width=True,
            load_libero_file="h5",
            batch_size=1,
            world_size=1,
            workers=1,
            rank=0,
            seed=args.seed,
        )
        data = get_libero_finetune_dataset(
            dataset_args, model.image_processor, clip, epoch=0
        )
        model_dtype = next(model.action_decoder.parameters()).dtype
        inputs = []
        audits = []
        for sample_index in range(args.sample_start, args.sample_start + count):
            sample = data.dataset[sample_index]
            batch = data.dataset.collator([sample])
            images_primary, text_tokens, _, images_wrist, states, _ = batch
            input_states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
            item = {
                "image_primary": images_primary[:, :7].to(device=device, dtype=model_dtype),
                "image_wrist": images_wrist[:, :7].to(device=device, dtype=model_dtype),
                "state": input_states[:, :7].to(device=device, dtype=model_dtype),
                "text_token": text_tokens.to(device).unsqueeze(1).repeat(1, 7, 1),
                "action": torch.zeros(1, 7, 7, device=device, dtype=model_dtype),
            }
            inputs.append(item)
            audits.append(
                {
                    "sample_index": sample_index,
                    "episode_id": str(sample["episode_id"]),
                    "language": str(sample["lang"]),
                    "model_input_shapes": {key: list(value.shape) for key, value in item.items()},
                    "model_input_dtypes": {key: str(value.dtype) for key, value in item.items()},
                    "model_input_sha256": {
                        key: _tensor_sha256(value) for key, value in item.items()
                    },
                }
            )

    episode_ids = {entry["episode_id"] for entry in audits}
    languages = {entry["language"] for entry in audits}
    if len(episode_ids) != 1 or len(languages) != 1:
        raise RuntimeError(
            f"selected samples cross episode/instruction boundaries: episodes={episode_ids}, languages={languages}"
        )
    overlap_errors = []
    for previous, current in zip(inputs, inputs[1:]):
        for key in ("image_primary", "image_wrist", "state", "text_token"):
            left = previous[key][:, 1:]
            right = current[key][:, :-1]
            if not torch.equal(left, right):
                overlap_errors.append(key)
    if overlap_errors:
        raise RuntimeError(
            "cached inputs are not exact one-step shifted windows; mismatched tensors="
            f"{sorted(set(overlap_errors))}"
        )
    return inputs, {
        "selection": audits,
        "same_episode": True,
        "same_instruction": True,
        "exact_shifted_window_overlap": True,
        "augmentation": "disabled (rgb_pad=-1, gripper_pad=-1)",
    }


def _model_call(model, item):
    return model(**item, return_action_latent=True)


@dataclass
class BenchmarkMethod:
    name: str
    group: str
    actions_per_call: int
    full_forwards_per_call: int
    lightweight_updates_per_call: int
    run: Callable[[], object]
    prepare: Callable[[], None] = lambda: None
    cleanup: Callable[[], None] = lambda: None


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _measure_one(method: BenchmarkMethod) -> tuple[float, float]:
    method.prepare()
    try:
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        started = time.perf_counter()
        method.run()
        end_event.record()
        end_event.synchronize()
        wall_ms = (time.perf_counter() - started) * 1000.0
        gpu_ms = float(start_event.elapsed_time(end_event))
    finally:
        method.cleanup()
    return wall_ms / method.actions_per_call, gpu_ms / method.actions_per_call


def _build_vla_config(mode: str):
    from models.vla_cache import build_seer_vla_cache_config

    return build_seer_vla_cache_config(
        mode=mode,
        pruning_layers="2,6,9,11",
        reference_attention_layer=15,
        similarity_threshold=0.996,
        positive_growth_factor=0.55,
        transformer_layers=24,
        sequence_length=7,
        num_resampler_query=6,
        num_obs_token_per_image=9,
        obs_pred=True,
        action_pred_steps=3,
    )


def _build_methods(args, model, inputs, bridge, compiled_bridge, bridge_payload):
    from architectures.seer.adapters.latent_bridge.hooks import SeerBoundaryCapture
    from architectures.seer.adapters.latent_bridge.layout import SeerTokenLayout

    off_config = _build_vla_config("off")
    matched_config = _build_vla_config("matched_full")
    reuse_config = _build_vla_config("reuse")
    layout = SeerTokenLayout.from_model(model)
    capture_layers = (
        ()
        if bridge.config.stable_layer == "ln_f"
        else (int(bridge.config.stable_layer.removeprefix("block_")),)
    )

    def prepare_off():
        model.vla_cache_config = off_config
        model.reset_vla_cache_state()

    def run_baseline():
        output = None
        for item in inputs:
            output = _model_call(model, item)
        return output

    methods = {
        "seer_full_k1": BenchmarkMethod(
            name="seer_full_k1",
            group="schedule",
            actions_per_call=len(inputs),
            full_forwards_per_call=len(inputs),
            lightweight_updates_per_call=0,
            prepare=prepare_off,
            run=run_baseline,
        )
    }

    for interval in args.k_values:
        def run_latentloop(k=interval):
            output = _model_call(model, inputs[0])
            latent = output["action_latent"][:, -1].detach()
            decoded = None
            for index in range(1, k):
                previous = inputs[index - 1]
                current = inputs[index]
                latent = model.lrnode_predict_next_latent(
                    z_prev=latent,
                    key_image_primary=previous["image_primary"][:, -1],
                    key_image_wrist=previous["image_wrist"][:, -1],
                    cur_image_primary=current["image_primary"][:, -1],
                    cur_image_wrist=current["image_wrist"][:, -1],
                    q_key=previous["state"][:, -1],
                    q_cur=current["state"][:, -1],
                    dt=1.0,
                    age=float(index),
                )
                decoded = model.decode_action_from_latent(latent)
            return output if decoded is None else decoded

        methods[f"latentloop_k{interval}"] = BenchmarkMethod(
            name=f"latentloop_k{interval}",
            group="schedule",
            actions_per_call=interval,
            full_forwards_per_call=1,
            lightweight_updates_per_call=interval - 1,
            prepare=prepare_off,
            run=run_latentloop,
        )

    bridge_capture = {"value": None}

    def prepare_bridge():
        prepare_off()
        capture = SeerBoundaryCapture(model, layer_indices=capture_layers)
        capture.__enter__()
        bridge_capture["value"] = capture

    def cleanup_bridge():
        capture = bridge_capture.pop("value", None)
        if capture is not None:
            capture.__exit__(None, None, None)

    def bridge_schedule(bridge_module):
        capture = bridge_capture["value"]
        capture.clear()
        output = _model_call(model, inputs[0])
        capture.require_complete()
        latent = output["action_latent"][:, -1].detach()
        if bridge.config.stable_layer == "ln_f":
            flat = capture.final_output
        else:
            flat = capture.layer_outputs[bridge.config.stable_layer]
        stable = layout.select(
            flat, timestep=-1, group=bridge.config.stable_token_group
        ).detach()
        sequence = torch.cat(
            [output["arm_pred_action"][:, -1], output["gripper_pred_action"][:, -1]],
            dim=-1,
        )
        previous_action = sequence[:, 0].detach()
        decoded = None
        for index in range(1, args.bridge_k):
            current_state = inputs[index]["state"][:, -1].detach()
            bridge_inputs = tuple(
                value.to(dtype=torch.bfloat16)
                for value in (latent, stable, current_state, previous_action)
            )
            latent = bridge_inputs[0] + bridge_module(*bridge_inputs)
            latent = latent.to(dtype=next(model.action_decoder.parameters()).dtype)
            arm, gripper = model.decode_action_from_latent(latent)
            decoded = (arm, gripper)
            previous_action = torch.cat([arm, gripper], dim=-1)[:, 0].detach()
            latent = latent.detach()
        return decoded

    methods[f"latent_bridge_large_k{args.bridge_k}_eager"] = BenchmarkMethod(
        name=f"latent_bridge_large_k{args.bridge_k}_eager",
        group="schedule_control",
        actions_per_call=args.bridge_k,
        full_forwards_per_call=1,
        lightweight_updates_per_call=args.bridge_k - 1,
        prepare=prepare_bridge,
        cleanup=cleanup_bridge,
        run=lambda: bridge_schedule(bridge),
    )
    methods[f"latent_bridge_large_k{args.bridge_k}_compiled"] = BenchmarkMethod(
        name=f"latent_bridge_large_k{args.bridge_k}_compiled",
        group="schedule",
        actions_per_call=args.bridge_k,
        full_forwards_per_call=1,
        lightweight_updates_per_call=args.bridge_k - 1,
        prepare=prepare_bridge,
        cleanup=cleanup_bridge,
        run=lambda: bridge_schedule(compiled_bridge),
    )

    def vla_method():
        output = None
        for item in inputs:
            output = _model_call(model, item)
        return output

    for name, config in (
        ("vla_cache_indexed_full", matched_config),
        ("vla_cache_reuse", reuse_config),
    ):
        def prepare_vla(cfg=config):
            model.vla_cache_config = cfg
            model.reset_vla_cache_state()

        methods[name] = BenchmarkMethod(
            name=name,
            group="schedule" if name.endswith("reuse") else "schedule_control",
            actions_per_call=len(inputs),
            full_forwards_per_call=len(inputs),
            lightweight_updates_per_call=0,
            prepare=prepare_vla,
            run=vla_method,
        )

    prepare_off()
    with SeerBoundaryCapture(model, layer_indices=capture_layers) as capture:
        seed_output = _model_call(model, inputs[0])
        capture.require_complete()
        seed_latent = seed_output["action_latent"][:, -1].detach()
        stable_flat = (
            capture.final_output
            if bridge.config.stable_layer == "ln_f"
            else capture.layer_outputs[bridge.config.stable_layer]
        )
        seed_stable = layout.select(
            stable_flat, timestep=-1, group=bridge.config.stable_token_group
        ).detach()
    seed_sequence = torch.cat(
        [seed_output["arm_pred_action"][:, -1], seed_output["gripper_pred_action"][:, -1]],
        dim=-1,
    )
    seed_action = seed_sequence[:, 0].detach()
    seed_state = inputs[1]["state"][:, -1].detach()
    fixed_bridge_inputs = tuple(
        value.to(dtype=torch.bfloat16)
        for value in (seed_latent, seed_stable, seed_state, seed_action)
    )

    def run_action_head():
        return model.decode_action_from_latent(seed_latent)

    def run_latentloop_skip():
        latent = model.lrnode_predict_next_latent(
            z_prev=seed_latent,
            key_image_primary=inputs[0]["image_primary"][:, -1],
            key_image_wrist=inputs[0]["image_wrist"][:, -1],
            cur_image_primary=inputs[1]["image_primary"][:, -1],
            cur_image_wrist=inputs[1]["image_wrist"][:, -1],
            q_key=inputs[0]["state"][:, -1],
            q_cur=inputs[1]["state"][:, -1],
            dt=1.0,
            age=1.0,
        )
        return model.decode_action_from_latent(latent)

    def run_bridge_skip(module):
        latent = fixed_bridge_inputs[0] + module(*fixed_bridge_inputs)
        return model.decode_action_from_latent(
            latent.to(dtype=next(model.action_decoder.parameters()).dtype)
        )

    methods.update(
        {
            "component_shared_action_head": BenchmarkMethod(
                name="component_shared_action_head",
                group="component",
                actions_per_call=1,
                full_forwards_per_call=0,
                lightweight_updates_per_call=0,
                prepare=prepare_off,
                run=run_action_head,
            ),
            "component_latentloop_skip_age1": BenchmarkMethod(
                name="component_latentloop_skip_age1",
                group="component",
                actions_per_call=1,
                full_forwards_per_call=0,
                lightweight_updates_per_call=1,
                prepare=prepare_off,
                run=run_latentloop_skip,
            ),
            "component_latent_bridge_skip_eager": BenchmarkMethod(
                name="component_latent_bridge_skip_eager",
                group="component",
                actions_per_call=1,
                full_forwards_per_call=0,
                lightweight_updates_per_call=1,
                prepare=prepare_off,
                run=lambda: run_bridge_skip(bridge),
            ),
            "component_latent_bridge_skip_compiled": BenchmarkMethod(
                name="component_latent_bridge_skip_compiled",
                group="component",
                actions_per_call=1,
                full_forwards_per_call=0,
                lightweight_updates_per_call=1,
                prepare=prepare_off,
                run=lambda: run_bridge_skip(compiled_bridge),
            ),
        }
    )
    return methods, {
        "bridge_stage": bridge_payload["stage"],
        "bridge_epoch": int(bridge_payload["epoch"]),
        "bridge_preset": bridge.config.preset,
        "bridge_parameters": int(sum(parameter.numel() for parameter in bridge.parameters())),
        "bridge_compile": "torch.compile(mode=max-autotune), warmup excluded",
        "bridge_previous_action_for_timing": (
            "first token of the previously decoded action horizon; temporal ensembling "
            "is intentionally outside this pure model-path timing boundary"
        ),
        "latentloop_parameters": int(
            sum(parameter.numel() for parameter in model.lrnode_delta_encoder.parameters())
            + sum(parameter.numel() for parameter in model.lrnode_dynamics.parameters())
        ),
    }


def _parse_k_values(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed or sorted(set(parsed)) != list(parsed) or min(parsed) < 2:
        raise argparse.ArgumentTypeError("K values must be unique ascending integers >= 2")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--latentloop-checkpoint", required=True)
    parser.add_argument("--latentloop-sha256", required=True)
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--bridge-sha256", required=True)
    parser.add_argument("--bridge-source-root", required=True)
    parser.add_argument("--bridge-source-commit", required=True)
    parser.add_argument("--vit-checkpoint", required=True)
    parser.add_argument("--vit-sha256", required=True)
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--clip-sha256", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", default="libero_10_converted")
    parser.add_argument("--libero-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--k-values", type=_parse_k_values, default=_parse_k_values("2,3,4,5,6,7,8"))
    parser.add_argument("--bridge-k", type=int, default=4)
    parser.add_argument("--warmup-cycles", type=int, default=5)
    parser.add_argument("--measured-cycles", type=int, default=60)
    parser.add_argument("--progress-interval", type=int, default=5)
    parser.add_argument("--barrier-dir", default="")
    parser.add_argument("--barrier-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"each replicate must see exactly one GPU, found {torch.cuda.device_count()}"
        )
    if max(args.k_values) < args.bridge_k:
        raise ValueError("input horizon must cover bridge K")
    if args.warmup_cycles < 1 or args.measured_cycles < 2:
        raise ValueError("warmup must be >=1 and measured cycles >=2")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    bridge_source = Path(args.bridge_source_root).resolve()
    for source in (str(REPO_ROOT), str(bridge_source)):
        if source not in sys.path:
            sys.path.insert(0, source)
    bridge_git = _git_state(bridge_source)
    bridge_source_files = [
        bridge_source / "architectures/seer/adapters/latent_bridge/bridge.py",
        bridge_source / "architectures/seer/adapters/latent_bridge/checkpoint.py",
        bridge_source / "architectures/seer/adapters/latent_bridge/hooks.py",
        bridge_source / "architectures/seer/adapters/latent_bridge/layout.py",
    ]
    bridge_relative_files = [str(path.relative_to(bridge_source)) for path in bridge_source_files]
    bridge_source_dirty = subprocess.run(
        ["git", "-C", str(bridge_source), "diff", "--quiet", "HEAD", "--", *bridge_relative_files],
        check=False,
    ).returncode != 0
    if bridge_git["commit"] != args.bridge_source_commit or bridge_source_dirty:
        raise RuntimeError(
            "Latent Bridge benchmark source must match the locked commit: "
            f"expected={args.bridge_source_commit}, actual={bridge_git}, "
            f"benchmark_source_dirty={bridge_source_dirty}"
        )

    assets = {
        "seer_checkpoint": _require_hash(
            args.checkpoint, args.checkpoint_sha256, "Seer checkpoint"
        ),
        "latentloop_checkpoint": _require_hash(
            args.latentloop_checkpoint, args.latentloop_sha256, "LatentLoop checkpoint"
        ),
        "latent_bridge_checkpoint": _require_hash(
            args.bridge_checkpoint, args.bridge_sha256, "Latent Bridge checkpoint"
        ),
        "vit_checkpoint": _require_hash(
            args.vit_checkpoint, args.vit_sha256, "ViT-MAE checkpoint"
        ),
        "clip_checkpoint": _require_hash(
            args.clip_checkpoint, args.clip_sha256, "CLIP checkpoint"
        ),
    }
    expected_clip = Path.home() / ".cache/clip/ViT-B-32.pt"
    if Path(args.clip_checkpoint).resolve() != expected_clip.resolve():
        raise RuntimeError(
            "Seer resolves ViT-B/32 through ~/.cache/clip; "
            f"expected --clip-checkpoint={expected_clip}, got {args.clip_checkpoint}"
        )
    metadata = Path(args.dataset_root) / args.dataset_name / "meta_info.h5"
    if not metadata.is_file():
        raise FileNotFoundError(f"invalid converted dataset root; missing {metadata}")
    if not (Path(args.libero_path) / "libero/libero").is_dir():
        raise FileNotFoundError(f"invalid LIBERO root: {args.libero_path}")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if "RTX 3090" not in properties.name:
        raise RuntimeError(f"paper latency benchmark requires RTX 3090, got {properties.name}")
    _seed_everything(args.seed)
    model, load_audit = _load_model(args, device)
    inputs, input_audit = _load_shifted_inputs(
        args, model, device, max(args.k_values)
    )

    from architectures.seer.adapters.latent_bridge.checkpoint import load_bridge_checkpoint

    bridge, bridge_payload = load_bridge_checkpoint(args.bridge_checkpoint, map_location="cpu")
    if bridge.config.preset != "full":
        raise RuntimeError(f"expected Latent Bridge Large/full, got {bridge.config.preset}")
    bridge = bridge.to(device=device, dtype=torch.bfloat16).eval()
    bridge.requires_grad_(False)
    compiled_bridge = torch.compile(bridge, mode="max-autotune")

    methods, method_audit = _build_methods(
        args, model, inputs, bridge, compiled_bridge, bridge_payload
    )
    with torch.inference_mode():
        print(
            f"[{args.replicate_id}] warmup: {len(methods)} methods x {args.warmup_cycles}",
            flush=True,
        )
        for method in methods.values():
            for _ in range(args.warmup_cycles):
                _measure_one(method)

        if args.barrier_dir:
            barrier_dir = Path(args.barrier_dir)
            if not barrier_dir.is_dir():
                raise FileNotFoundError(f"barrier directory is missing: {barrier_dir}")
            ready_path = barrier_dir / f"{args.replicate_id}.ready"
            ready_path.write_text(f"pid={os.getpid()}\n", encoding="utf-8")
            deadline = time.monotonic() + args.barrier_timeout_seconds
            print(f"[{args.replicate_id}] warmup complete; waiting at measurement barrier", flush=True)
            while not (barrier_dir / "release").is_file():
                if (barrier_dir / "abort").is_file():
                    raise RuntimeError("measurement barrier was aborted by launcher")
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for measurement barrier release")
                time.sleep(0.25)
            print(f"[{args.replicate_id}] measurement barrier released", flush=True)

        wall_samples = {name: [] for name in methods}
        gpu_samples = {name: [] for name in methods}
        if not args.replicate_id.startswith("gpu") or not args.replicate_id[3:].isdigit():
            raise ValueError("replicate-id must have the form gpu<physical-index>")
        rng = random.Random(args.seed + int(args.replicate_id[3:]))
        names = list(methods)
        started = time.time()
        for cycle in range(args.measured_cycles):
            order = names.copy()
            rng.shuffle(order)
            for name in order:
                wall_ms, gpu_ms = _measure_one(methods[name])
                wall_samples[name].append(wall_ms)
                gpu_samples[name].append(gpu_ms)
            if (cycle + 1) % args.progress_interval == 0 or cycle + 1 == args.measured_cycles:
                elapsed = time.time() - started
                eta = elapsed / (cycle + 1) * (args.measured_cycles - cycle - 1)
                print(
                    f"[{args.replicate_id}] measured {cycle + 1}/{args.measured_cycles}; "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

        model.vla_cache_config = _build_vla_config("reuse")
        model.transformer_backbone.reset_vla_cache_statistics()
        model.reset_vla_cache_state()
        for item in inputs:
            _model_call(model, item)
        vla_reuse_stats = model.get_vla_cache_stats()
        if int(vla_reuse_stats["actual_kv_reuse_calls"]) <= 0:
            raise RuntimeError("VLA-Cache diagnostic completed without actual K/V reuse")

        model.vla_cache_config = _build_vla_config("matched_full")
        model.transformer_backbone.reset_vla_cache_statistics()
        model.reset_vla_cache_state()
        for item in inputs:
            _model_call(model, item)
        vla_matched_stats = model.get_vla_cache_stats()
        if int(vla_matched_stats["actual_kv_reuse_calls"]) != 0:
            raise RuntimeError("VLA-Cache indexed-full control unexpectedly reused K/V")

    source_files = [
        REPO_ROOT / "architectures/seer/upstream/models/seer_model.py",
        REPO_ROOT / "architectures/seer/upstream/models/gpt2.py",
        REPO_ROOT / "architectures/seer/upstream/models/lrnode_modules.py",
        REPO_ROOT / "architectures/seer/upstream/models/vla_cache.py",
        Path(__file__).resolve(),
        *bridge_source_files,
    ]
    results = {}
    baseline_mean = _summary(wall_samples["seer_full_k1"])["mean"]
    for name, method in methods.items():
        wall = _summary(wall_samples[name])
        gpu = _summary(gpu_samples[name])
        results[name] = {
            "group": method.group,
            "actions_per_measured_call": method.actions_per_call,
            "full_forwards_per_measured_call": method.full_forwards_per_call,
            "lightweight_updates_per_measured_call": method.lightweight_updates_per_call,
            "full_forward_fraction": method.full_forwards_per_call / method.actions_per_call,
            "wall_ms_per_action": wall,
            "cuda_event_ms_per_action": gpu,
            "speedup_vs_same_process_seer_wall": baseline_mean / wall["mean"],
            "raw_wall_ms_per_action": wall_samples[name],
            "raw_cuda_event_ms_per_action": gpu_samples[name],
        }

    payload = {
        "status": "PASS",
        "replicate_id": args.replicate_id,
        "scope": "Seer only",
        "timing_contract": {
            "primary_metric": "CUDA-synchronized wall-clock model-path milliseconds per executed action",
            "secondary_metric": "CUDA event milliseconds per executed action",
            "included": [
                "Seer image/text/state encoding and action head on full queries",
                "method-specific cache/update logic",
                "shared Seer action head on LatentLoop and Latent Bridge skip steps",
                "one full refresh plus K-1 lightweight updates for periodic K methods",
            ],
            "excluded": [
                "LIBERO simulator and renderer",
                "disk I/O and dataset collation",
                "host image preprocessing and device transfer",
                "temporal ensembling",
                "torch.compile compilation and all warmup cycles",
            ],
            "input": "eight exact consecutive, overlapping, unaugmented converted-LIBERO windows cached on GPU",
            "interleaving": "deterministically shuffled method order in every measured cycle",
            "bridge_runtime": "BF16 Large bridge; eager control and production max-autotune compiled path both reported",
            "seer_runtime": "eager FP32 with BF16 vision encoder, matching Seer evaluation precision",
            "renderer_dependency": "none; no simulator or renderer is constructed",
        },
        "hardware": {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [properties.major, properties.minor],
            "visible_gpu_count": torch.cuda.device_count(),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "source": {
            "unified_worktree": _git_state(REPO_ROOT),
            "latent_bridge_worktree": bridge_git,
            "file_sha256": {str(path): _sha256(path) for path in source_files},
        },
        "assets": {
            "paths": {
                "seer_checkpoint": args.checkpoint,
                "latentloop_checkpoint": args.latentloop_checkpoint,
                "latent_bridge_checkpoint": args.bridge_checkpoint,
                "vit_checkpoint": args.vit_checkpoint,
                "clip_checkpoint": args.clip_checkpoint,
                "dataset_metadata": str(metadata),
            },
            "sha256": assets,
        },
        "model_load_audit": load_audit,
        "input_audit": input_audit,
        "method_audit": method_audit,
        "benchmark": {
            "seed": args.seed,
            "sample_start": args.sample_start,
            "warmup_cycles_excluded": args.warmup_cycles,
            "measured_cycles": args.measured_cycles,
            "k_values": list(args.k_values),
            "bridge_k": args.bridge_k,
        },
        "results": results,
        "vla_cache_runtime_validation": {
            "reuse": vla_reuse_stats,
            "indexed_full": vla_matched_stats,
        },
    }
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(f"[{args.replicate_id}] PASS: {output}", flush=True)


if __name__ == "__main__":
    main()
