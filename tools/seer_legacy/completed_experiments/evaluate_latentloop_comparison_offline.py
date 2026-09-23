#!/usr/bin/env python3
"""Evaluate one comparison adapter on deterministic held-out Seer teacher tuples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F


_PATH_ARGUMENT_NAMES = (
    "repo_root",
    "source_lock",
    "teacher",
    "canonical_adapter",
    "adapter_checkpoint",
    "dataset_root",
    "vit_checkpoint",
    "libero_path",
    "output",
)
_LIBERO_DATASET_NAME = "libero_10_converted"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: float("nan") for key in ("mean", "p50", "p90", "p95", "p99", "max")} | {"count": 0}
    q = np.quantile(array, [0.50, 0.90, 0.95, 0.99])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(q[0]),
        "p90": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
        "max": float(array.max()),
    }


def _strip_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    raw = checkpoint.get("model_state_dict", checkpoint)
    return {
        key.removeprefix("module."): value
        for key, value in raw.items()
    }


def _load_matching(model: torch.nn.Module, path: Path, prefixes: tuple[str, ...]) -> int:
    state = {
        key: value
        for key, value in _strip_state(path).items()
        if key.startswith(prefixes)
    }
    if not state:
        raise RuntimeError(f"No keys with prefixes={prefixes} in {path}")
    model.load_state_dict(state, strict=False)
    return len(state)


def _per_sample(value: torch.Tensor) -> list[float]:
    return value.detach().float().reshape(value.shape[0], -1).mean(dim=1).cpu().tolist()


def _per_sample_rms(value: torch.Tensor) -> list[float]:
    flat = value.detach().float().reshape(value.shape[0], -1)
    return flat.square().mean(dim=1).sqrt().cpu().tolist()


def _parse_legacy_seer_runtime_args(
    get_parser: Any,
    save_checkpoint_path: Path,
) -> argparse.Namespace:
    """Build Seer args without letting its legacy parser consume this tool's CLI.

    ``utils.arguments_utils.get_parser`` parses ``sys.argv`` before returning
    its parser. Temporarily provide the required Seer arguments, restore the
    offline evaluator CLI, and then parse the same explicit argument list.
    """

    seer_args = [
        "--save_checkpoint_path",
        str(save_checkpoint_path),
        "--phase",
        "finetune",
    ]
    outer_argv = sys.argv
    try:
        sys.argv = ["seer_offline_runtime", *seer_args]
        parser = get_parser()
    finally:
        sys.argv = outer_argv
    return parser.parse_args(seer_args)


def _run_with_seer_upstream_cwd(
    args: argparse.Namespace,
    callback: Callable[[argparse.Namespace], dict[str, Any]],
) -> dict[str, Any]:
    """Run the evaluator where Seer's legacy relative assets are resolvable.

    Seer's LIBERO dataset loads ``./data_info/*.json`` relative to the process
    working directory. Resolve every CLI path before changing directory, keep
    the upstream cwd through DataLoader iteration, and always restore the
    caller's cwd.
    """

    for name in _PATH_ARGUMENT_NAMES:
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())

    upstream = args.repo_root / "architectures/seer/upstream"
    data_info = upstream / f"data_info/{_LIBERO_DATASET_NAME}.json"
    if not data_info.is_file():
        raise FileNotFoundError(f"Missing Seer LIBERO data-info file: {data_info}")
    runtime_dataset = args.dataset_root / _LIBERO_DATASET_NAME
    for required in (runtime_dataset / "episodes", runtime_dataset / "meta_info.h5"):
        if not required.exists():
            raise FileNotFoundError(
                "Invalid Seer ROOT_DIR contract; expected runtime dataset asset: "
                f"{required}"
            )

    previous_cwd = Path.cwd()
    os.chdir(upstream)
    try:
        return callback(args)
    finally:
        os.chdir(previous_cwd)


def _build_runtime(args: argparse.Namespace):
    repo_root = args.repo_root.resolve()
    upstream = repo_root / "architectures/seer/upstream"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(upstream))
    import clip  # noqa: PLC0415
    from architectures.seer.adapters.latentloop_comparison import (  # noqa: PLC0415
        attach_comparison_adapter,
    )
    from train import _apply_precision_policy, _build_seer_agent_from_args  # noqa: PLC0415
    from utils.arguments_utils import get_parser  # noqa: PLC0415
    from utils.data_utils import get_libero_finetune_dataset  # noqa: PLC0415

    parser_args = _parse_legacy_seer_runtime_args(
        get_parser,
        args.output.parent,
    )
    overrides = {
        "finetune_type": "libero_finetune",
        "root_dir": str(args.dataset_root),
        "vit_checkpoint_path": str(args.vit_checkpoint),
        "libero_path": str(args.libero_path),
        "world_size": 1,
        "rank": 0,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "rgb_pad": -1,
        "gripper_pad": -1,
        "traj_cons": True,
        "sequence_length": 7,
        "window_size": 10,
        "future_steps": 3,
        "action_pred_steps": 3,
        "num_resampler_query": 6,
        "transformer_layers": 24,
        "obs_pred": True,
        "gripper_width": True,
        "precision": "fp32",
        "bf16_module": "vision_encoder",
        "use_lrnode_latent_update": 1,
        "lrnode_hidden_dim": 256,
        "lrnode_motion_dim": 128,
        "lrnode_fast_encoder_type": "diffcnn",
        "lrnode_gate_init_bias": -4.0,
        "lrnode_use_post_layernorm": 0,
        "latentloop_plan_adapter_mode": args.mode,
        "latentloop_plan_adapter_hidden_dim": 0,
        "latentloop_plan_parameter_match_tolerance": 0.05,
        "latentloop_comparison_protocol": 1,
        "latentloop_comparison_split_role": "validation",
        "latentloop_comparison_validation_fraction": args.validation_fraction,
        "latentloop_comparison_validation_seed": args.validation_seed,
        "seed": args.seed,
    }
    for name, value in overrides.items():
        setattr(parser_args, name, value)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = _build_seer_agent_from_args(parser_args, device)
    adapter_report = attach_comparison_adapter(model, parser_args)
    base_state = {
        key: value
        for key, value in _strip_state(args.teacher).items()
        if not key.startswith(("lrnode_", "latentloop_plan_adapter."))
    }
    model.load_state_dict(base_state, strict=False)
    if args.adapter_checkpoint is not None:
        _load_matching(
            model,
            args.adapter_checkpoint,
            ("latentloop_plan_adapter.",),
        )
    if args.canonical_adapter is not None:
        _load_matching(
            model,
            args.canonical_adapter,
            ("lrnode_delta_encoder.", "lrnode_dynamics."),
        )
    model = _apply_precision_policy(model, parser_args).to(device)
    model._init_model_type()
    model.eval()
    model.requires_grad_(False)
    dataset = get_libero_finetune_dataset(
        parser_args, model.image_processor, clip, epoch=0, floor=True
    )
    return parser_args, model, dataset, adapter_report, device


def _evaluate_from_seer_upstream(args: argparse.Namespace) -> dict[str, Any]:
    """Run the held-out teacher-tuple gate for one baseline state."""

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    source_lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    if source_lock.get("status") != "PASS":
        raise RuntimeError("Offline validation requires a PASS source lock")
    locked_inputs = source_lock["training_inputs"]
    if args.dataset_root != Path(locked_inputs["dataset_root"]).resolve():
        raise RuntimeError("Dataset root does not match the source lock")
    vit_sha256 = _sha256(args.vit_checkpoint)
    if vit_sha256 != locked_inputs["vit_checkpoint"]["sha256"]:
        raise RuntimeError("ViT checkpoint does not match the source lock")
    teacher_sha256 = _sha256(args.teacher)
    if teacher_sha256 != source_lock["primary_identity"]["teacher"]["sha256"]:
        raise RuntimeError("Teacher checkpoint does not match the source lock")
    canonical_sha256 = (
        _sha256(args.canonical_adapter) if args.canonical_adapter else None
    )
    if canonical_sha256 is not None and canonical_sha256 != source_lock[
        "primary_identity"
    ]["adapter"]["sha256"]:
        raise RuntimeError("Canonical adapter does not match the source lock")
    parser_args, model, data, adapter_report, device = _build_runtime(args)
    adapter = model.latentloop_plan_adapter
    metrics: dict[str, list[float]] = defaultdict(list)
    offset_counts: dict[int, int] = defaultdict(int)
    nan_or_inf = False
    batches = 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    data.set_epoch(0)
    for batch_index, batch in enumerate(data.dataloader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        images_primary = batch[0].to(device, dtype=torch.float32)
        images_wrist = batch[3].to(device, dtype=torch.float32)
        states = batch[4].to(device, dtype=torch.float32)
        actions = batch[2].to(device, dtype=torch.float32)
        text = batch[1].to(device).unsqueeze(1).repeat(1, 10, 1)
        states = torch.cat((states[..., :6], states[..., -2:]), dim=-1)
        actions = actions.clone()
        actions[..., 6:] = (actions[..., 6:] + 1) // 2
        offset = 1 if args.mode == "action_correction" else (batch_index % 3) + 1
        selected = 6
        current_stop = 7 + offset
        with torch.no_grad():
            torch.manual_seed(args.seed + batch_index * 17)
            anchor_output = model(
                image_primary=images_primary[:, :7],
                image_wrist=images_wrist[:, :7],
                state=states[:, :7],
                text_token=text[:, :7],
                action=actions[:, :7],
                return_action_latent=True,
                lrnode_compute_loss=False,
            )
            torch.manual_seed(args.seed + batch_index * 17 + offset)
            teacher_output = model(
                image_primary=images_primary[:, offset:current_stop],
                image_wrist=images_wrist[:, offset:current_stop],
                state=states[:, offset:current_stop],
                text_token=text[:, offset:current_stop],
                action=actions[:, offset:current_stop],
                return_action_latent=True,
                lrnode_compute_loss=False,
            )
            z_anchor = anchor_output["action_latent"][:, selected]
            z_teacher = teacher_output["action_latent"][:, selected]
            anchor_action = model.decode_action_diagnostics_from_latent(z_anchor)
            teacher_action = model.decode_action_diagnostics_from_latent(z_teacher)

            if args.canonical_adapter is not None:
                canonical_latent = z_anchor
                no_observation_latent = z_anchor
                for step_offset in range(1, offset + 1):
                    step_feature = model.lrnode_delta_encoder(
                        [
                            images_primary[:, selected + step_offset - 1],
                            images_wrist[:, selected + step_offset - 1],
                        ],
                        [
                            images_primary[:, selected + step_offset],
                            images_wrist[:, selected + step_offset],
                        ],
                        q_key=states[:, selected + step_offset - 1],
                        q_cur=states[:, selected + step_offset],
                    )
                    canonical_latent = model.lrnode_apply_dynamics(
                        canonical_latent,
                        step_feature,
                        dt=1.0,
                        age=float(step_offset),
                    )
                    no_observation_latent = model.lrnode_apply_dynamics(
                        no_observation_latent,
                        torch.zeros_like(step_feature),
                        dt=1.0,
                        age=float(step_offset),
                    )
                canonical_action = model.decode_action_diagnostics_from_latent(
                    canonical_latent
                )
                no_observation_action = model.decode_action_diagnostics_from_latent(
                    no_observation_latent
                )
                target_continuous = torch.cat(
                    (
                        teacher_action["arm"],
                        teacher_action["gripper_probability"],
                    ),
                    dim=-1,
                )
                metrics["canonical_latent_mse"].extend(
                    _per_sample((canonical_latent - z_teacher).square())
                )
                metrics["canonical_action_l1"].extend(
                    _per_sample(
                        (
                            torch.cat(
                                (
                                    canonical_action["arm"],
                                    canonical_action["gripper_probability"],
                                ),
                                dim=-1,
                            )
                            - target_continuous
                        ).abs()
                    )
                )
                metrics["no_observation_latent_mse"].extend(
                    _per_sample((no_observation_latent - z_teacher).square())
                )
                metrics["no_observation_action_l1"].extend(
                    _per_sample(
                        (
                            torch.cat(
                                (
                                    no_observation_action["arm"],
                                    no_observation_action["gripper_probability"],
                                ),
                                dim=-1,
                            )
                            - target_continuous
                        ).abs()
                    )
                )

            if args.mode == "action_correction":
                feature = adapter.encode_delta(
                    images_primary[:, selected],
                    images_wrist[:, selected],
                    images_primary[:, selected + 1],
                    images_wrist[:, selected + 1],
                    states[:, selected],
                    states[:, selected + 1],
                )
                output = adapter.forward_from_feature(
                    anchor_action["arm"],
                    anchor_action["gripper_logit"],
                    feature,
                    age=1.0,
                )
                predicted_arm = output.arm
                predicted_logit = output.gripper_logit
                predicted_probability = output.gripper_probability
                metrics["raw_arm_smooth_l1"].extend(
                    _per_sample(F.smooth_l1_loss(predicted_arm, teacher_action["arm"], reduction="none"))
                )
                metrics["raw_gripper_bce"].extend(
                    _per_sample(F.binary_cross_entropy_with_logits(
                        predicted_logit,
                        teacher_action["gripper_probability"],
                        reduction="none",
                    ))
                )
                metrics["raw_executed_arm_smooth_l1"].extend(
                    _per_sample(F.smooth_l1_loss(
                        predicted_arm[:, 0], teacher_action["arm"][:, 0], reduction="none"
                    ))
                )
                arm_reg = _per_sample(output.arm_residual.square())
                grip_reg = _per_sample(output.gripper_logit_residual.square())
                metrics["raw_residual_l2"].extend(
                    left + right for left, right in zip(arm_reg, grip_reg)
                )
                hold_arm = torch.cat(
                    (anchor_action["arm"][:, 1:], anchor_action["arm"][:, -1:]), dim=1
                )
                hold_logit = torch.cat(
                    (
                        anchor_action["gripper_logit"][:, 1:],
                        anchor_action["gripper_logit"][:, -1:],
                    ),
                    dim=1,
                )
                metrics["hold_action_l1"].extend(
                    _per_sample((torch.cat((hold_arm, torch.sigmoid(hold_logit)), -1) - torch.cat((teacher_action["arm"], teacher_action["gripper_probability"]), -1)).abs())
                )
            else:
                feature = adapter.encode_anchor_to_current(
                    images_primary[:, selected],
                    images_wrist[:, selected],
                    images_primary[:, selected + offset],
                    images_wrist[:, selected + offset],
                    states[:, selected],
                    states[:, selected + offset],
                )
                output = adapter.forward_from_feature(z_anchor, feature, age=float(offset))
                predicted = model.decode_action_diagnostics_from_latent(output.latent)
                predicted_arm = predicted["arm"]
                predicted_logit = predicted["gripper_logit"]
                predicted_probability = predicted["gripper_probability"]
                metrics["raw_latent_mse"].extend(
                    _per_sample((output.latent - z_teacher).square())
                )
                metrics["raw_action_l1"].extend(
                    _per_sample((torch.cat((predicted_arm, predicted_probability), -1) - torch.cat((teacher_action["arm"], teacher_action["gripper_probability"]), -1)).abs())
                )
                metrics["raw_smooth_mse"].extend(
                    _per_sample((output.latent - z_anchor).square())
                )
                metrics["hold_anchor_latent_mse"].extend(
                    _per_sample((z_anchor - z_teacher).square())
                )
                cosine = F.cosine_similarity(
                    output.latent.float().reshape(output.latent.shape[0], -1),
                    z_teacher.float().reshape(z_teacher.shape[0], -1),
                    dim=-1,
                )
                metrics[f"offset_{offset}_latent_cosine"].extend(cosine.cpu().tolist())
                metrics[f"offset_{offset}_latent_mse"].extend(
                    _per_sample((output.latent - z_teacher).square())
                )

            predicted_action = torch.cat((predicted_arm, predicted_probability), dim=-1)
            target_action = torch.cat(
                (teacher_action["arm"], teacher_action["gripper_probability"]), dim=-1
            )
            error = predicted_action - target_action
            metrics["action_horizon_l1"].extend(_per_sample(error.abs()))
            metrics["action_horizon_l2"].extend(_per_sample_rms(error))
            metrics["first_token_l1"].extend(_per_sample(error[:, :1].abs()))
            metrics["first_token_l2"].extend(_per_sample_rms(error[:, :1]))
            metrics["translation_l1"].extend(_per_sample(error[..., :3].abs()))
            metrics["rotation_l1"].extend(_per_sample(error[..., 3:6].abs()))
            metrics["gripper_probability_l1"].extend(
                _per_sample((predicted_probability - teacher_action["gripper_probability"]).abs())
            )
            metrics["predicted_gripper_probability"].extend(
                predicted_probability.detach().float().reshape(-1).cpu().tolist()
            )
            all_values = [predicted_action, feature]
            if args.mode == "anchor_bridge":
                all_values.append(output.latent)
            nan_or_inf = nan_or_inf or any(
                not torch.isfinite(value).all().item() for value in all_values
            )
        offset_counts[offset] += int(images_primary.shape[0])
        batches += 1

    if not batches:
        raise RuntimeError("Offline evaluator processed zero validation batches")
    summaries = {name: _distribution(values) for name, values in sorted(metrics.items())}
    if args.mode == "action_correction":
        validation_total = (
            summaries["raw_arm_smooth_l1"]["mean"]
            + summaries["raw_gripper_bce"]["mean"]
        )
        reference_improved = summaries["action_horizon_l1"]["mean"] < summaries["hold_action_l1"]["mean"]
    else:
        validation_total = summaries["raw_latent_mse"]["mean"] + summaries["raw_action_l1"]["mean"]
        reference_improved = summaries["raw_latent_mse"]["mean"] < summaries["hold_anchor_latent_mse"]["mean"]
    gripper_values = np.asarray(
        metrics["predicted_gripper_probability"], dtype=np.float64
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "latentloop_q1_q2_offline_validation_v1",
        "mode": args.mode,
        "source_lock": str(args.source_lock.resolve()),
        "source_lock_sha256": _sha256(args.source_lock),
        "teacher": str(args.teacher.resolve()),
        "teacher_sha256": teacher_sha256,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()) if args.adapter_checkpoint else None,
        "adapter_checkpoint_sha256": (
            _sha256(args.adapter_checkpoint) if args.adapter_checkpoint else None
        ),
        "canonical_adapter": str(args.canonical_adapter.resolve()) if args.canonical_adapter else None,
        "canonical_adapter_sha256": canonical_sha256,
        "vit_checkpoint": str(args.vit_checkpoint),
        "vit_checkpoint_sha256": vit_sha256,
        "dataset_root": str(args.dataset_root),
        "runtime_dataset_path": str(args.dataset_root / _LIBERO_DATASET_NAME),
        "validation_split": data.split_manifest,
        "adapter_parameter_report": adapter_report,
        "batches": batches,
        "examples": int(sum(offset_counts.values())),
        "offset_counts": {str(key): value for key, value in sorted(offset_counts.items())},
        "metrics": summaries,
        "selection_metric": "validation_total_loss",
        "selection_metric_value": float(validation_total),
        "gates": {
            "finite": not nan_or_inf,
            "improves_corresponding_hold_reference": bool(reference_improved),
            "gripper_noncollapsed_proxy": bool(
                np.isfinite(gripper_values).all()
                and float(gripper_values.std()) > 1e-4
                and 0.01 < float(gripper_values.mean()) < 0.99
            ),
            "p99_finite": all(math.isfinite(float(row["p99"])) for row in summaries.values()),
        },
    }
    result["gates"]["pass"] = all(result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    reloaded = json.loads(args.output.read_text(encoding="utf-8"))
    if reloaded["mode"] != args.mode or reloaded["examples"] != result["examples"]:
        raise RuntimeError("Offline trace serialization/reload validation failed")
    return result


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve CLI paths and evaluate under Seer's required upstream cwd."""

    return _run_with_seer_upstream_cwd(args, _evaluate_from_seer_upstream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--canonical-adapter", type=Path)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--mode", choices=["action_correction", "anchor_bridge"], required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vit-checkpoint", type=Path, required=True)
    parser.add_argument("--libero-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--validation-seed", type=int, default=20260805)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
