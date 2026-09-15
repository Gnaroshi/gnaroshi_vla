"""Frozen, paired Condition updater interventions. No optimizer or Generation loop."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    ExactTeacherSequenceDataset, collate_exact_teacher_sequences,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.latent_fidelity_analysis import (
    _balanced_indices, masked_condition_metrics,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch
from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair


VARIANTS = (
    "hold", "zero_feature", "full_update", "stale_images", "stale_encoder_proprio",
    "repeated_observation", "full_stale_head_proprio", "zero_stale_head_proprio",
    "full_gate_zero_residual", "zero_gate_full_residual", "constant_update", "age_only_update",
)
PRIMARY = ("hold", "zero_feature", "full_update")
REGIMES = ("shared_teacher_previous", "recursive")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_hash(value: torch.Tensor | np.ndarray) -> str:
    array = value.detach().cpu().contiguous().numpy() if torch.is_tensor(value) else np.asarray(value)
    header = json.dumps([str(array.dtype), list(array.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()


def completed_unit(path: Path, identity: str) -> dict | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result.get("identity") != identity or result.get("complete") is not True:
        raise RuntimeError(f"Incompatible/incomplete result, use a separate output: {path}")
    return result


def observation_codes(adapter: Any, sequence: dict, age: int) -> dict[str, torch.Tensor]:
    pi, ci = sequence["image_sequence"][:, age - 1], sequence["image_sequence"][:, age]
    pq, cq = sequence["proprio_sequence"][:, age - 1], sequence["proprio_sequence"][:, age]
    pairs = {
        "full": (ci, cq), "stale_images": (pi, cq),
        "stale_encoder_proprio": (ci, pq), "repeated_observation": (pi, pq),
    }
    codes = {
        name: adapter.delta_encoder(NativeV0ObservationPair(pi, images, pq, proprio))
        for name, (images, proprio) in pairs.items()
    }
    codes["zero"] = torch.zeros_like(codes["full"])
    return codes


def intervention(
    adapter: Any, previous: torch.Tensor, codes: dict, mask: torch.Tensor,
    groups: torch.Tensor, age: int, variant: str, means: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if variant not in VARIANTS:
        raise ValueError(variant)
    if variant == "hold":
        return previous, previous.new_zeros((*previous.shape[:2], 1)), torch.zeros_like(previous)
    if variant in ("constant_update", "age_only_update"):
        if means is None:
            raise ValueError("train-only correction means are required")
        offset = means.mean(0) if variant == "constant_update" else means[age - 1]
        residual = offset.unsqueeze(0).expand_as(previous)
        gate = previous.new_ones((*previous.shape[:2], 1))
    else:
        key = {
            "zero_feature": "zero", "zero_stale_head_proprio": "zero",
            "stale_images": "stale_images", "stale_encoder_proprio": "stale_encoder_proprio",
            "repeated_observation": "repeated_observation",
        }.get(variant, "full")
        update = adapter.condition_updater(
            previous, codes[key], valid_mask=mask, group_ids=groups, age=age,
        )
        gate, residual = update.gate, update.residual
        if variant in ("full_gate_zero_residual", "zero_gate_full_residual"):
            zero = adapter.condition_updater(
                previous, codes["zero"], valid_mask=mask, group_ids=groups, age=age,
            )
            if variant == "full_gate_zero_residual":
                residual = zero.residual
            else:
                gate = zero.gate
    valid = mask.bool().unsqueeze(-1)
    gate = torch.where(valid, gate, torch.zeros_like(gate))
    residual = torch.where(valid, residual, torch.zeros_like(residual))
    return torch.where(valid, previous + gate * residual, previous), gate, residual


def action_metrics(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    p, r = prediction.float(), reference.float()
    d = (p - r).abs()
    if p.shape != r.shape or p.shape[-2:] != (10, 7):
        raise ValueError("expected matching H=10, action_dim=7 chunks")
    return {
        "first5_action_l1": float(d[:, :5].mean()),
        "full_chunk_action_l1": float(d.mean()),
        "translation_l1": float(d[:, :5, :3].mean()),
        "rotation_l1": float(d[:, :5, 3:6].mean()),
        "continuous_gripper_l1": float(d[:, :5, 6].mean()),
        "gripper_sign_disagreement": float(((p[:, :5, 6] > 0) != (r[:, :5, 6] > 0)).float().mean()),
        "first5_action_cosine": float(F.cosine_similarity(p[:, :5].flatten(1), r[:, :5].flatten(1)).mean()),
    }


def load_sequence(dataset: Any, index: int, device: torch.device) -> dict:
    return move_batch(collate_exact_teacher_sequences([dataset[index]]), device)


def make_datasets(config: dict, checkpoint_payload: dict) -> tuple[Any, Any]:
    train_config = checkpoint_payload["training_config"]
    kwargs = {"heldout_fraction": train_config["heldout_fraction"], "split_seed": train_config["split_seed"]}
    train = ExactTeacherSequenceDataset(config["cache"], split="train", **kwargs)
    heldout = ExactTeacherSequenceDataset(config["cache"], split="heldout", **kwargs)
    expected = train_config["dataset_splits"]
    if train.split_sha256 != expected["train_split_sha256"] or heldout.split_sha256 != expected["heldout_split_sha256"]:
        raise RuntimeError("Cache split does not match the trained Condition checkpoint")
    return train, heldout


@torch.no_grad()
def train_only_means(adapter: Any, train: Any, selected: list[int], device: torch.device) -> torch.Tensor:
    sums = counts = None
    for index in tqdm(selected, desc="학습 구간의 고정 보정량 집계", mininterval=1.0):
        s = load_sequence(train, index, device)
        previous = s["anchor_condition"]
        if sums is None:
            sums = previous.new_zeros((3, previous.shape[1], previous.shape[2]))
            counts = previous.new_zeros((3, previous.shape[1], 1))
        mask = s["valid_mask"].unsqueeze(-1)
        for age in (1, 2, 3):
            zero = previous.new_zeros((previous.shape[0], adapter.delta_dim))
            update = adapter.condition_updater(previous, zero, valid_mask=s["valid_mask"], group_ids=s["group_ids"], age=age)
            sums[age - 1] += ((update.condition - previous) * mask).sum(0)
            counts[age - 1] += mask.sum(0)
            previous = update.condition
    if sums is None:
        raise RuntimeError("No calibration windows")
    return sums / counts.clamp_min(1)


@torch.no_grad()
def run_offline(config: dict, output: Path, identity: str, adapter: Any, payload: dict, action: Any) -> dict:
    started = time.monotonic()
    device = next(adapter.parameters()).device
    train, heldout = make_datasets(config, payload)
    train_indices = _balanced_indices(train.identities, limit=config["calibration_windows"], seed=config["analysis_seed"])
    selected = _balanced_indices(heldout.identities, limit=config["heldout_windows"], seed=config["analysis_seed"])
    selection = {
        "identity": identity, "train": [train.identities[i] for i in train_indices],
        "heldout": [heldout.identities[i] for i in selected],
        "train_split": train.contract(), "heldout_split": heldout.contract(),
        "constant_control": "mean zero-feature update from training episodes only; no gradient fitting",
    }
    write_json(output / "selection.json", selection)
    means_file = output / "train_only_means.pt"
    if means_file.exists():
        saved = torch.load(means_file, map_location=device, weights_only=False)
        if saved["identity"] != identity:
            raise RuntimeError("Correction means identity changed")
        means = saved["means"]
    else:
        means = train_only_means(adapter, train, train_indices, device)
        temporary = means_file.with_suffix(".tmp")
        torch.save({"identity": identity, "means": means.cpu()}, temporary)
        temporary.replace(means_file)
    progress = tqdm(selected, desc="동일 입력·경로·재귀 분석", mininterval=1.0)
    all_rows = []
    for index in progress:
        path = output / "units" / f"window_{index:05d}.json"
        saved = completed_unit(path, identity)
        if saved is not None:
            all_rows.extend(saved["rows"])
            continue
        unit_started = time.monotonic()
        s = load_sequence(heldout, index, device)
        exact = [s["anchor_condition"], *s["teacher_conditions"].unbind(1)]
        recursive = {v: exact[0] for v in VARIANTS}
        rows = []
        for age in (1, 2, 3):
            codes = observation_codes(adapter, s, age)
            noise, proprio = s["explicit_noises"][:, age - 1], s["proprio_sequence"][:, age]
            reference = action.decode_action_from_condition(exact[age], proprio, steps=10, initial_noise=noise.clone(), requires_grad=False)
            for regime in REGIMES:
                for variant in VARIANTS:
                    previous = exact[age - 1] if regime == "shared_teacher_previous" else recursive[variant]
                    updater_age = 1 if regime == "shared_teacher_previous" else age
                    candidate, gate, residual = intervention(adapter, previous, codes, s["valid_mask"], s["group_ids"], updater_age, variant, means)
                    # Updater age=1 mirrors a local K_C=2 query. Recursive age is 1/2/3.
                    head_q = s["proprio_sequence"][:, age - 1] if "stale_head" in variant else proprio
                    prediction = action.decode_action_from_condition(candidate, head_q, steps=10, initial_noise=noise.clone(), requires_grad=False)
                    valid = s["valid_mask"].bool()
                    metrics = masked_condition_metrics(candidate, exact[age], previous, valid)[0]
                    metrics["diagnostic_token_standardized_mse"] = metrics.pop("condition_normalized_mse")
                    # Also measure at the actual learned LayerNorm, not just arbitrary standardization.
                    normalized_error = adapter.condition_updater.norm(candidate) - adapter.condition_updater.norm(exact[age])
                    metrics["updater_layernorm_mse"] = float(normalized_error[valid].square().mean())
                    projected_error = action.model.transformer.vlm_proj(candidate) - action.model.transformer.vlm_proj(exact[age])
                    metrics["action_input_projection_mse"] = float(projected_error[valid].square().mean())
                    row = {
                        "dataset_index": index, "task_id": int(s["task_id"][0]),
                        "episode_id": s["episode_id"][0], "anchor_query_index": int(s["anchor_query_index"][0]),
                        "regime": regime, "age": age, "updater_age": updater_age,
                        "window_action_offset": 5 * age, "actions_since_last_full": 5 * updater_age, "variant": variant,
                        "previous_condition_sha256": tensor_hash(previous), "noise_sha256": tensor_hash(noise),
                        "head_proprio_sha256": tensor_hash(head_q),
                        "gate_mean": float(gate[valid].mean()),
                        "residual_rms": float(residual[valid].square().mean().sqrt()),
                        "update_rms": float((candidate - previous)[valid].square().mean().sqrt()),
                        **metrics, **action_metrics(prediction, reference),
                    }
                    if not all(np.isfinite(v) for v in row.values() if isinstance(v, float)):
                        raise RuntimeError(f"Nonfinite metrics: window={index} {variant}")
                    rows.append(row)
                    if regime == "recursive":
                        recursive[variant] = candidate
        saved = {"identity": identity, "complete": True, "rows": rows, "seconds": time.monotonic() - unit_started}
        write_json(path, saved)
        all_rows.extend(rows)
    result = {"identity": identity, "complete": True, "windows": len(selected), "rows": len(all_rows),
              "flow_steps": 10, "generation_updater_used": False, "training_run": False,
              "seconds_this_invocation": time.monotonic() - started}
    write_json(output / "summary.json", result)
    return result
