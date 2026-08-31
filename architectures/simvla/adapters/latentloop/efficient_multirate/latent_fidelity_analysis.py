"""Paired mechanistic analysis for the SimVLA Condition and Generation loops.

This module is intentionally an offline analysis.  It reuses the frozen exact
teacher cache, frozen Condition updater, and frozen coupled Generation updater;
it does not train parameters or step a LIBERO environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    GENERATION_SCHEDULES,
    sha256_file,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    COUPLED_CHECKPOINT_SCHEMA,
    audit_projection_only_state,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import (
    install_exact_uint8_delta_path,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    ExactTeacherSequenceDataset,
    _drop_unused_vlm,
    collate_exact_teacher_sequences,
    validate_exact_cache,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    load_generation_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_hidden import (
    full_generation_step_with_hidden,
)
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    load_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.native_v0_condition_hook import (
    extract_action_condition,
)
from architectures.simvla.adapters.latentloop.native_v0_prepare import (
    _official_training_image_inputs,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
    move_batch,
    write_json,
)
from methods.latentloop.modules.native_simvla_v0 import (
    NativeSimVLAV0,
    NativeV0ObservationPair,
)
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop


OBSERVATION_VARIANTS = (
    "ours_full_observation",
    "hold_previous_condition",
    "stale_observation",
    "vision_only",
    "proprio_only",
    "shuffled_current_same_task",
)
CONDITION_METRICS = (
    "condition_mse",
    "condition_normalized_mse",
    "condition_cosine",
    "condition_delta_cosine",
    "condition_delta_norm_ratio",
    "gate_mean",
    "residual_rms",
)
ACTION_METRICS = (
    "nfe10_first5_l1",
    "nfe10_full_chunk_l1",
    "nfe10_translation_l1",
    "nfe10_rotation_l1",
    "nfe10_gripper_l1",
)
GENERATION_METHODS = (
    "original_nfe10",
    "naive_nfe3",
    "ours_generation_ng3",
    "coupled_generation_ng3",
)
GATE_ABLATIONS = (
    "learned_gate",
    "fixed_sample_mean_gate",
    "unit_gate",
    "hold_condition",
)


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _masked_vectors(value: torch.Tensor, mask: torch.Tensor, index: int) -> torch.Tensor:
    return value[index].float()[mask[index].bool()]


def masked_condition_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    previous: torch.Tensor,
    valid_mask: torch.Tensor,
) -> list[dict[str, float]]:
    """Return per-sample latent and latent-delta fidelity metrics."""

    rows: list[dict[str, float]] = []
    for index in range(prediction.shape[0]):
        pred = _masked_vectors(prediction, valid_mask, index)
        ref = _masked_vectors(target, valid_mask, index)
        prev = _masked_vectors(previous, valid_mask, index)
        pred_norm = F.layer_norm(pred, (pred.shape[-1],))
        ref_norm = F.layer_norm(ref, (ref.shape[-1],))
        pred_delta = (pred - prev).reshape(-1)
        ref_delta = (ref - prev).reshape(-1)
        ref_delta_norm = ref_delta.norm().clamp_min(1e-12)
        rows.append(
            {
                "condition_mse": float((pred - ref).square().mean().item()),
                "condition_normalized_mse": float(
                    (pred_norm - ref_norm).square().mean().item()
                ),
                "condition_cosine": float(
                    F.cosine_similarity(pred, ref, dim=-1).mean().item()
                ),
                "condition_delta_cosine": float(
                    F.cosine_similarity(
                        pred_delta.unsqueeze(0), ref_delta.unsqueeze(0), dim=-1
                    ).item()
                ),
                "condition_delta_norm_ratio": float(
                    (pred_delta.norm() / ref_delta_norm).item()
                ),
            }
        )
    return rows


def _action_metrics(
    prediction: torch.Tensor, target: torch.Tensor, *, prefix: str
) -> list[dict[str, float]]:
    difference = (prediction.float() - target.float()).abs()
    first5 = difference[:, :5]
    rows: list[dict[str, float]] = []
    for index in range(prediction.shape[0]):
        rows.append(
            {
                f"{prefix}_first5_l1": float(first5[index].mean().item()),
                f"{prefix}_full_chunk_l1": float(difference[index].mean().item()),
                f"{prefix}_translation_l1": float(first5[index, :, :3].mean().item()),
                f"{prefix}_rotation_l1": float(first5[index, :, 3:6].mean().item()),
                f"{prefix}_gripper_l1": float(first5[index, :, 6:].mean().item()),
            }
        )
    return rows


def deterministic_task_partner_indices(
    identities: Sequence[tuple[int, str, int]],
) -> dict[int, int]:
    """Pair every sequence with another sequence from the same task."""

    by_task: dict[int, list[int]] = defaultdict(list)
    for index, identity in enumerate(identities):
        by_task[int(identity[0])].append(index)
    result: dict[int, int] = {}
    for task_indices in by_task.values():
        if len(task_indices) < 2:
            raise RuntimeError("same-task observation shuffle requires at least two sequences")
        ordered = sorted(task_indices, key=lambda index: identities[index])
        for offset, index in enumerate(ordered):
            result[index] = ordered[(offset + 1) % len(ordered)]
    return result


def _balanced_indices(
    identities: Sequence[tuple[int, str, int]], *, limit: int, seed: int
) -> list[int]:
    by_task: dict[int, list[int]] = defaultdict(list)
    for index, identity in enumerate(identities):
        by_task[int(identity[0])].append(index)
    generator = np.random.default_rng(seed)
    for values in by_task.values():
        generator.shuffle(values)
    selected: list[int] = []
    tasks = sorted(by_task)
    cursor = 0
    target = len(identities) if int(limit) <= 0 else min(int(limit), len(identities))
    while len(selected) < target:
        made_progress = False
        for task in tasks:
            values = by_task[task]
            if cursor < len(values):
                selected.append(values[cursor])
                made_progress = True
                if len(selected) == target:
                    break
        if not made_progress:
            break
        cursor += 1
    return selected


def _variant_pair(
    sequence: dict[str, Any],
    partner: dict[str, Any],
    *,
    age: int,
    variant: str,
) -> NativeV0ObservationPair | None:
    previous_images = sequence["image_sequence"][:, age - 1]
    current_images = sequence["image_sequence"][:, age]
    previous_proprio = sequence["proprio_sequence"][:, age - 1]
    current_proprio = sequence["proprio_sequence"][:, age]
    if variant == "hold_previous_condition":
        return None
    if variant == "stale_observation":
        current_images = previous_images
        current_proprio = previous_proprio
    elif variant == "vision_only":
        current_proprio = previous_proprio
    elif variant == "proprio_only":
        current_images = previous_images
    elif variant == "shuffled_current_same_task":
        current_images = partner["image_sequence"][:, age]
        current_proprio = partner["proprio_sequence"][:, age]
    elif variant != "ours_full_observation":
        raise ValueError(f"unknown observation variant: {variant}")
    return NativeV0ObservationPair(
        previous_images=previous_images,
        current_images=current_images,
        previous_proprio=previous_proprio,
        current_proprio=current_proprio,
    )


@torch.no_grad()
def _condition_update(
    adapter: NativeSimVLAV0,
    previous: torch.Tensor,
    sequence: dict[str, Any],
    partner: dict[str, Any],
    *,
    age: int,
    updater_age: int,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pair = _variant_pair(sequence, partner, age=age, variant=variant)
    batch = previous.shape[0]
    if pair is None:
        return (
            previous,
            previous.new_zeros((batch, adapter.delta_dim)),
            previous.new_zeros((batch, previous.shape[1], 1)),
            previous.new_zeros(previous.shape),
        )
    code = adapter.delta_encoder(pair)
    update = adapter.condition_updater(
        previous,
        code,
        valid_mask=sequence["valid_mask"],
        group_ids=sequence["group_ids"],
        age=updater_age,
    )
    return update.condition, code, update.gate, update.residual


def _observation_change(sequence: dict[str, Any], age: int) -> list[dict[str, float]]:
    previous_images = sequence["image_sequence"][:, age - 1].float().div(255.0)
    current_images = sequence["image_sequence"][:, age].float().div(255.0)
    previous_proprio = sequence["proprio_sequence"][:, age - 1].float()
    current_proprio = sequence["proprio_sequence"][:, age].float()
    visual = (current_images - previous_images).abs().flatten(1).mean(dim=1)
    proprio = (current_proprio - previous_proprio).norm(dim=-1)
    return [
        {
            "visual_change_l1": float(visual[index].item()),
            "proprio_change_l2": float(proprio[index].item()),
        }
        for index in range(visual.shape[0])
    ]


def assign_change_quartiles(rows: list[dict[str, Any]]) -> None:
    """Assign a joint empirical visual/proprio change quartile in-place."""

    anchors = {
        (int(row["dataset_index"]), int(row["age"])): (
            float(row["visual_change_l1"]), float(row["proprio_change_l2"])
        )
        for row in rows
        if row["regime"] == "kc2_local" and row["variant"] == "ours_full_observation"
    }
    keys = sorted(anchors)
    if not keys:
        return

    def ranks(position: int) -> dict[tuple[int, int], float]:
        ordered = sorted(keys, key=lambda key: (anchors[key][position], key))
        denominator = max(1, len(ordered) - 1)
        return {key: rank / denominator for rank, key in enumerate(ordered)}

    visual_rank = ranks(0)
    proprio_rank = ranks(1)
    quartiles = {
        key: min(4, int(((visual_rank[key] + proprio_rank[key]) * 0.5) * 4) + 1)
        for key in keys
    }
    for row in rows:
        row["observation_change_quartile"] = quartiles.get(
            (int(row["dataset_index"]), int(row["age"])), ""
        )


def valid_generation_schedules(
    *, exact_steps: int = 3, total_steps: int = 10, max_skipped_age: int = 3
) -> tuple[tuple[int, ...], ...]:
    """Enumerate schedules supported by the frozen age-1/2/3 updater."""

    if exact_steps < 1 or total_steps < exact_steps or max_skipped_age < 0:
        raise ValueError("invalid generation schedule bounds")
    schedules: list[tuple[int, ...]] = []
    for suffix in combinations(range(1, total_steps), exact_steps - 1):
        schedule = (0, *suffix)
        age = 0
        supported = True
        full = set(schedule)
        for step in range(total_steps):
            if step in full:
                age = 0
            else:
                age += 1
                if age > max_skipped_age:
                    supported = False
                    break
        if supported:
            schedules.append(schedule)
    return tuple(schedules)


def gate_ablation_conditions(
    previous: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build causal gate controls while preserving the learned residual."""

    mask = valid_mask.bool().unsqueeze(-1)
    weight = mask.to(dtype=gate.dtype)
    sample_mean = (gate * weight).sum(dim=1, keepdim=True) / weight.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)

    def apply(selected_gate: torch.Tensor) -> torch.Tensor:
        candidate = previous + selected_gate * residual
        return torch.where(mask, candidate, previous)

    return {
        "learned_gate": apply(gate),
        "fixed_sample_mean_gate": apply(sample_mean),
        "unit_gate": apply(torch.ones_like(gate)),
        "hold_condition": previous,
    }


@torch.no_grad()
def _generation_trace(
    *,
    loop: SimVLAGenerationLoop,
    transformer: Any,
    action_space: Any,
    condition: torch.Tensor,
    valid_mask: torch.Tensor,
    normalized_proprio: torch.Tensor,
    noise: torch.Tensor,
    code: torch.Tensor,
    full_step_indices: Sequence[int] = GENERATION_SCHEDULES[3],
) -> tuple[Any, torch.Tensor]:
    def full_step(
        noisy_action: torch.Tensor, tau: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = full_generation_step_with_hidden(
            transformer,
            condition=condition,
            noisy_action=noisy_action,
            proprio=normalized_proprio,
            tau=tau,
            dt=-0.1,
        )
        return output.action_hidden, output.velocity

    trace = loop(
        noise,
        full_step=full_step,
        full_step_indices=full_step_indices,
        proprio=normalized_proprio,
        condition=condition,
        condition_valid_mask=valid_mask,
        condition_change_code=code,
    )
    return trace, action_space.postprocess(trace.final_noisy_action)


def _method_fidelity_rows(
    *,
    method: str,
    prediction: torch.Tensor,
    original: torch.Tensor,
    teacher: torch.Tensor,
    dataset_indices: Sequence[int],
    query_age: int,
    schedule: Sequence[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    versus_original = _action_metrics(prediction, original, prefix="vs_original")
    versus_teacher = _action_metrics(prediction, teacher, prefix="vs_teacher")
    rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    schedule_text = "" if schedule is None else ",".join(map(str, schedule))
    for local, dataset_index in enumerate(dataset_indices):
        flat_prediction = prediction[local].float().reshape(1, -1)
        flat_original = original[local].float().reshape(1, -1)
        first_prediction = prediction[local, :5].float().reshape(1, -1)
        first_original = original[local, :5].float().reshape(1, -1)
        rows.append(
            {
                "dataset_index": int(dataset_index),
                "query_age": int(query_age),
                "method": method,
                "schedule": schedule_text,
                "vs_original_full_cosine": float(
                    F.cosine_similarity(flat_prediction, flat_original).item()
                ),
                "vs_original_first5_cosine": float(
                    F.cosine_similarity(first_prediction, first_original).item()
                ),
                **versus_original[local],
                **versus_teacher[local],
            }
        )
        for action_step in range(prediction.shape[1]):
            predicted_step = prediction[local, action_step].float()
            original_step = original[local, action_step].float()
            step_rows.append(
                {
                    "dataset_index": int(dataset_index),
                    "query_age": int(query_age),
                    "method": method,
                    "schedule": schedule_text,
                    "action_step": int(action_step),
                    "vs_original_l1": float(
                        (predicted_step - original_step).abs().mean().item()
                    ),
                    "vs_original_cosine": float(
                        F.cosine_similarity(
                            predicted_step.unsqueeze(0), original_step.unsqueeze(0)
                        ).item()
                    ),
                }
            )
    return rows, step_rows


@torch.no_grad()
def _hidden_fidelity_rows(
    *,
    mode: str,
    trace: Any,
    transformer: Any,
    condition: torch.Tensor,
    normalized_proprio: torch.Tensor,
    dataset_indices: Sequence[int],
    query_age: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, step in enumerate(trace.skipped_step_indices):
        exact = full_generation_step_with_hidden(
            transformer,
            condition=condition,
            noisy_action=trace.skipped_noisy_actions[offset],
            proprio=normalized_proprio,
            tau=trace.skipped_times[offset],
            dt=-0.1,
        )
        predicted_hidden = trace.predicted_hidden[offset].float()
        exact_hidden = exact.action_hidden.float()
        predicted_velocity = trace.predicted_velocity[offset].float()
        exact_velocity = exact.velocity.float()
        hidden_normalized_mse = (
            F.layer_norm(predicted_hidden, (predicted_hidden.shape[-1],))
            - F.layer_norm(exact_hidden, (exact_hidden.shape[-1],))
        ).square().flatten(1).mean(dim=1)
        hidden_cosine = F.cosine_similarity(
            predicted_hidden, exact_hidden, dim=-1
        ).mean(dim=1)
        velocity_l1 = (predicted_velocity - exact_velocity).abs().flatten(1).mean(dim=1)
        velocity_cosine = F.cosine_similarity(
            predicted_velocity, exact_velocity, dim=-1
        ).mean(dim=1)
        norm_ratio = predicted_hidden.flatten(1).norm(dim=1) / exact_hidden.flatten(1).norm(
            dim=1
        ).clamp_min(1e-12)
        for index, dataset_index in enumerate(dataset_indices):
            rows.append(
                {
                    "dataset_index": int(dataset_index),
                    "query_age": int(query_age),
                    "code_mode": mode,
                    "flow_step": int(step),
                    "skipped_age": int(trace.skipped_ages[offset]),
                    "hidden_normalized_mse": float(hidden_normalized_mse[index].item()),
                    "hidden_cosine": float(hidden_cosine[index].item()),
                    "hidden_norm_ratio": float(norm_ratio[index].item()),
                    "velocity_l1": float(velocity_l1[index].item()),
                    "velocity_cosine": float(velocity_cosine[index].item()),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty analysis table: {path.name}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_summary(
    rows: list[dict[str, Any]], *, group_fields: Sequence[str], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)
    result: list[dict[str, Any]] = []
    for key, selected in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        entry: dict[str, Any] = dict(zip(group_fields, key))
        entry["rows"] = len(selected)
        for metric in metrics:
            values = [row[metric] for row in selected if metric in row and row[metric] != ""]
            if values:
                entry[metric] = _summary(values)
        result.append(entry)
    return result


def _timing_summary(values: Sequence[float]) -> dict[str, float | int]:
    result = _summary(values)
    result["minimum"] = float(np.min(np.asarray(values, dtype=np.float64)))
    result["maximum"] = float(np.max(np.asarray(values, dtype=np.float64)))
    return result


@torch.no_grad()
def _benchmark_cuda(
    label: str,
    function: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(repeats):
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return (
        {"component": label, "clock": "cuda_event", **_timing_summary(samples)},
        [
            {"component": label, "clock": "cuda_event", "repeat": index, "milliseconds": value}
            for index, value in enumerate(samples)
        ],
    )


def _benchmark_cpu(
    label: str,
    function: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for _ in range(warmup):
        function()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return (
        {"component": label, "clock": "perf_counter", **_timing_summary(samples)},
        [
            {"component": label, "clock": "perf_counter", "repeat": index, "milliseconds": value}
            for index, value in enumerate(samples)
        ],
    )


@torch.no_grad()
def _latency_decomposition(
    *,
    model: Any,
    processor: Any,
    action_adapter: Any,
    condition_adapter: NativeSimVLAV0,
    generation_loop: SimVLAGenerationLoop,
    generation_updater: Any,
    sample: dict[str, Any],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure batch-one module latency with explicit CUDA synchronization."""

    def preprocess() -> dict[str, torch.Tensor]:
        result = _official_training_image_inputs(
            sample["image_sequence"][1],
            image_size=int(getattr(processor, "image_size", 384)),
            num_views=int(getattr(processor, "num_views", 3)),
        )
        result.update(processor.encode_language([sample["language_instruction"]]))
        return result

    processed_cpu = preprocess()
    processed = {key: value.to(device) for key, value in processed_cpu.items()}
    tokenizer = processor.tokenizer

    def encode_condition() -> Any:
        return extract_action_condition(
            model,
            input_ids=processed["input_ids"],
            image_input=processed["image_input"],
            image_mask=processed["image_mask"],
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            special_token_ids=getattr(tokenizer, "all_special_ids", ()),
        )

    previous = sample["anchor_condition"].unsqueeze(0).to(device)
    condition = sample["teacher_conditions"][0].unsqueeze(0).to(device)
    valid_mask = sample["valid_mask"].unsqueeze(0).to(device)
    group_ids = sample["group_ids"].unsqueeze(0).to(device)
    proprio = sample["proprio_sequence"][1].unsqueeze(0).to(device)
    normalized_proprio = action_adapter.normalize_proprio(proprio)
    noise = sample["explicit_noises"][0].unsqueeze(0).to(device)
    pair = NativeV0ObservationPair(
        previous_images=sample["image_sequence"][0].unsqueeze(0).to(device),
        current_images=sample["image_sequence"][1].unsqueeze(0).to(device),
        previous_proprio=sample["proprio_sequence"][0].unsqueeze(0).to(device),
        current_proprio=proprio,
    )
    code = condition_adapter.delta_encoder(pair)

    def delta_encoder() -> Any:
        return condition_adapter.delta_encoder(pair)

    def condition_updater() -> Any:
        return condition_adapter.condition_updater(
            previous, code, valid_mask=valid_mask, group_ids=group_ids, age=1
        )

    zero_code = noise.new_zeros((1, int(generation_updater.condition_code_dim)))

    def original_nfe10() -> Any:
        return action_adapter.decode_action_from_condition(
            condition, proprio, steps=10, initial_noise=noise, requires_grad=False
        )

    def naive_nfe3() -> Any:
        return action_adapter.decode_action_from_condition(
            condition, proprio, steps=3, initial_noise=noise, requires_grad=False
        )

    def generation_ng3() -> Any:
        return _generation_trace(
            loop=generation_loop,
            transformer=model.transformer,
            action_space=action_adapter.action_space,
            condition=condition,
            valid_mask=valid_mask,
            normalized_proprio=normalized_proprio,
            noise=noise,
            code=zero_code,
            full_step_indices=GENERATION_SCHEDULES[3],
        )

    tau_before = noise.new_ones((1,))
    first = full_generation_step_with_hidden(
        model.transformer,
        condition=condition,
        noisy_action=noise,
        proprio=normalized_proprio,
        tau=tau_before,
        dt=-0.1,
    )
    noisy_after = noise - 0.1 * first.velocity
    tau_after = noise.new_full((1,), 0.9)

    def full_action_transformer_step() -> Any:
        return model.transformer(
            vlm_features=condition,
            action_with_noise=noise,
            proprio=normalized_proprio,
            t=tau_before,
        )

    def generation_hidden_updater() -> Any:
        return generation_updater(
            first.action_hidden,
            noise,
            noisy_after,
            tau_before=tau_before,
            tau_after=tau_after,
            proprio=normalized_proprio,
            condition_change_code=zero_code,
            condition=condition,
            condition_valid_mask=valid_mask,
            generator_age=1,
        )

    update = generation_hidden_updater()

    def frozen_action_decoder() -> Any:
        return model.transformer.action_decoder(update.hidden)

    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    cpu_summary, cpu_rows = _benchmark_cpu(
        "image_language_preprocess", preprocess, warmup=warmup, repeats=repeats
    )
    summaries.append(cpu_summary)
    rows.extend(cpu_rows)
    for label, function in (
        ("vlm_condition_encoder", encode_condition),
        ("delta_encoder", delta_encoder),
        ("condition_updater", condition_updater),
        ("full_action_transformer_step", full_action_transformer_step),
        ("generation_hidden_updater", generation_hidden_updater),
        ("frozen_action_decoder", frozen_action_decoder),
        ("original_action_generation_nfe10", original_nfe10),
        ("naive_action_generation_nfe3", naive_nfe3),
        ("ours_action_generation_ng3", generation_ng3),
    ):
        summary, component_rows = _benchmark_cuda(
            label, function, warmup=warmup, repeats=repeats
        )
        summaries.append(summary)
        rows.extend(component_rows)
    return summaries, rows


def _artifact_contract(
    *,
    coupled_payload: dict[str, Any],
    parent_checkpoint: str | Path,
    condition_checkpoint: str | Path,
    norm_stats: str | Path,
    cache: str | Path,
) -> dict[str, Any]:
    source = dict(coupled_payload["source_lock"])
    observed = {
        "parent_generation_checkpoint_sha256": sha256_file(parent_checkpoint),
        "condition_checkpoint_sha256": sha256_file(condition_checkpoint),
        "norm_stats_sha256": sha256_file(norm_stats),
        "exact_cache_manifest_sha256": sha256_file(Path(cache) / "manifest.json"),
    }
    checks = {key: source.get(key) == value for key, value in observed.items()}
    return {
        "verdict": "FROZEN_ARTIFACT_IDENTITY_PASS" if all(checks.values()) else "FROZEN_ARTIFACT_IDENTITY_FAIL",
        "checks": checks,
        "expected": {key: source.get(key) for key in observed},
        "observed": observed,
        "note": "Source commit may advance for analysis code; frozen artifact hashes must not.",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal --physical-gpu")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    determinism = configure_strict_torch_determinism(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    cache_validation = validate_exact_cache(args.cache, verify_checksums=args.verify_cache_checksums)
    if cache_validation["verdict"] != "EXACT_TEACHER_CACHE_VALID":
        raise RuntimeError(json.dumps(cache_validation, indent=2, sort_keys=True))
    condition_adapter, condition_payload = load_native_v0_checkpoint(
        args.condition_checkpoint, device=device, require_final_150k=True
    )
    freeze_module(condition_adapter)
    condition_adapter.eval()
    delta_path = install_exact_uint8_delta_path(condition_adapter)
    parent_updater, parent_payload = load_generation_checkpoint(
        args.parent_generation_checkpoint, device=device
    )
    coupled_updater, coupled_payload = load_generation_checkpoint(
        args.coupled_generation_checkpoint, device=device
    )
    if coupled_payload.get("training_config", {}).get("schema_version") != COUPLED_CHECKPOINT_SCHEMA:
        raise RuntimeError("coupled checkpoint schema mismatch")
    artifact_contract = _artifact_contract(
        coupled_payload=coupled_payload,
        parent_checkpoint=args.parent_generation_checkpoint,
        condition_checkpoint=args.condition_checkpoint,
        norm_stats=args.norm_stats,
        cache=args.cache,
    )
    if artifact_contract["verdict"] != "FROZEN_ARTIFACT_IDENTITY_PASS":
        raise RuntimeError(json.dumps(artifact_contract, indent=2, sort_keys=True))
    projection_audit = audit_projection_only_state(parent_updater, coupled_updater)
    if projection_audit["verdict"] != "PROJECTION_ONLY_STATE_PASS":
        raise RuntimeError(json.dumps(projection_audit, indent=2, sort_keys=True))
    freeze_module(parent_updater)
    freeze_module(coupled_updater)

    frozen_model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    freeze_module(frozen_model)
    parent_loop = SimVLAGenerationLoop(
        parent_updater, frozen_model.transformer.action_decoder
    ).to(device).eval()
    coupled_loop = SimVLAGenerationLoop(
        coupled_updater, frozen_model.transformer.action_decoder
    ).to(device).eval()
    dataset = ExactTeacherSequenceDataset(
        args.cache,
        split="heldout",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    selected = _balanced_indices(
        dataset.identities, limit=args.condition_windows, seed=args.seed
    )
    if not selected:
        raise RuntimeError("analysis selected no heldout windows")
    latency_summary, latency_rows = _latency_decomposition(
        model=frozen_model,
        processor=processor,
        action_adapter=action_adapter,
        condition_adapter=condition_adapter,
        generation_loop=parent_loop,
        generation_updater=parent_updater,
        sample=dataset[selected[0]],
        device=device,
        warmup=args.latency_warmup,
        repeats=args.latency_repeats,
    )
    del processor
    dropped = _drop_unused_vlm(frozen_model)
    partners = deterministic_task_partner_indices(dataset.identities)
    kc2_keys = [(index, age) for index in selected for age in (1, 3)]
    action_keys = set(kc2_keys[: min(args.action_queries, len(kc2_keys))])
    generation_keys = set(kc2_keys[: min(args.generation_queries, len(kc2_keys))])
    schedule_keys = set(kc2_keys[: min(args.schedule_queries, len(kc2_keys))])
    candidate_schedules = valid_generation_schedules()
    condition_rows: list[dict[str, Any]] = []
    hidden_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    generation_method_rows: list[dict[str, Any]] = []
    action_step_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    row_lookup: dict[tuple[int, int, str], dict[str, Any]] = {}
    gate_row_lookup: dict[tuple[int, int, str], dict[str, Any]] = {}

    with torch.no_grad():
        for start in tqdm(
            range(0, len(selected), args.batch_size),
            desc="SimVLA latent fidelity",
            dynamic_ncols=True,
        ):
            batch_indices = selected[start : start + args.batch_size]
            sequence = move_batch(
                collate_exact_teacher_sequences([dataset[index] for index in batch_indices]),
                device,
            )
            partner = move_batch(
                collate_exact_teacher_sequences(
                    [dataset[partners[index]] for index in batch_indices]
                ),
                device,
            )
            exact_conditions = (
                sequence["anchor_condition"],
                sequence["teacher_conditions"][:, 0],
                sequence["teacher_conditions"][:, 1],
                sequence["teacher_conditions"][:, 2],
            )

            recursive = {
                variant: sequence["anchor_condition"] for variant in OBSERVATION_VARIANTS
            }
            for age in (1, 2, 3):
                changes = _observation_change(sequence, age)
                for variant in OBSERVATION_VARIANTS:
                    prediction, code, gate, residual = _condition_update(
                        condition_adapter,
                        recursive[variant],
                        sequence,
                        partner,
                        age=age,
                        updater_age=age,
                        variant=variant,
                    )
                    metrics = masked_condition_metrics(
                        prediction,
                        exact_conditions[age],
                        recursive[variant],
                        sequence["valid_mask"],
                    )
                    for local, dataset_index in enumerate(batch_indices):
                        condition_rows.append(
                            {
                                "dataset_index": dataset_index,
                                "task_id": int(sequence["task_id"][local].item()),
                                "episode_id": sequence["episode_id"][local],
                                "anchor_query_index": int(
                                    sequence["anchor_query_index"][local].item()
                                ),
                                "regime": "k4_recursive",
                                "age": age,
                                "variant": variant,
                                **changes[local],
                                **metrics[local],
                                "gate_mean": float(
                                    gate[local, :, 0]
                                    .float()[sequence["valid_mask"][local].bool()]
                                    .mean()
                                    .item()
                                ),
                                "residual_rms": float(
                                    residual[local]
                                    .float()[sequence["valid_mask"][local].bool()]
                                    .square()
                                    .mean()
                                    .sqrt()
                                    .item()
                                ),
                                "code_norm": float(code[local].float().norm().item()),
                            }
                        )
                    recursive[variant] = prediction

            for age in (1, 3):
                previous_index = age - 1
                previous = exact_conditions[previous_index]
                target = exact_conditions[age]
                changes = _observation_change(sequence, age)
                variant_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                learned_gate_state: tuple[torch.Tensor, torch.Tensor] | None = None
                for variant in OBSERVATION_VARIANTS:
                    prediction, code, gate, residual = _condition_update(
                        condition_adapter,
                        previous,
                        sequence,
                        partner,
                        age=age,
                        updater_age=1,
                        variant=variant,
                    )
                    variant_outputs[variant] = (prediction, code)
                    if variant == "ours_full_observation":
                        learned_gate_state = (residual, gate)
                    metrics = masked_condition_metrics(
                        prediction, target, previous, sequence["valid_mask"]
                    )
                    for local, dataset_index in enumerate(batch_indices):
                        row = {
                            "dataset_index": dataset_index,
                            "task_id": int(sequence["task_id"][local].item()),
                            "episode_id": sequence["episode_id"][local],
                            "anchor_query_index": int(
                                sequence["anchor_query_index"][local].item()
                            ),
                            "regime": "kc2_local",
                            "age": age,
                            "variant": variant,
                            **changes[local],
                            **metrics[local],
                            "gate_mean": float(
                                gate[local, :, 0]
                                .float()[sequence["valid_mask"][local].bool()]
                                .mean()
                                .item()
                            ),
                            "residual_rms": float(
                                residual[local]
                                .float()[sequence["valid_mask"][local].bool()]
                                .square()
                                .mean()
                                .sqrt()
                                .item()
                            ),
                            "code_norm": float(code[local].float().norm().item()),
                        }
                        condition_rows.append(row)
                        row_lookup[(dataset_index, age, variant)] = row

                if learned_gate_state is None:
                    raise RuntimeError("learned gate state was not captured")
                residual, learned_gate = learned_gate_state
                gate_candidates = gate_ablation_conditions(
                    previous,
                    residual,
                    learned_gate,
                    sequence["valid_mask"],
                )
                for gate_mode, candidate in gate_candidates.items():
                    metrics = masked_condition_metrics(
                        candidate, target, previous, sequence["valid_mask"]
                    )
                    for local, dataset_index in enumerate(batch_indices):
                        row = {
                            "dataset_index": int(dataset_index),
                            "task_id": int(sequence["task_id"][local].item()),
                            "episode_id": sequence["episode_id"][local],
                            "anchor_query_index": int(
                                sequence["anchor_query_index"][local].item()
                            ),
                            "age": int(age),
                            "gate_mode": gate_mode,
                            "learned_gate_mean": float(
                                learned_gate[local, :, 0]
                                .float()[sequence["valid_mask"][local].bool()]
                                .mean()
                                .item()
                            ),
                            **metrics[local],
                        }
                        gate_rows.append(row)
                        gate_row_lookup[(dataset_index, age, gate_mode)] = row

                action_positions = [
                    local
                    for local, dataset_index in enumerate(batch_indices)
                    if (dataset_index, age) in action_keys
                ]
                if not action_positions:
                    continue
                position = torch.tensor(action_positions, device=device, dtype=torch.long)
                noise = sequence["explicit_noises"].index_select(0, position)[:, age - 1]
                proprio = sequence["proprio_sequence"].index_select(0, position)[:, age]
                target_action = sequence["teacher_actions"].index_select(0, position)[:, age - 1]
                selected_dataset_indices = [batch_indices[value] for value in action_positions]
                for variant, (prediction, _code) in variant_outputs.items():
                    predicted_action = action_adapter.decode_action_from_condition(
                        prediction.index_select(0, position),
                        proprio,
                        steps=10,
                        initial_noise=noise,
                        requires_grad=False,
                    )
                    action_metrics = _action_metrics(
                        predicted_action, target_action, prefix="nfe10"
                    )
                    for local, dataset_index in enumerate(selected_dataset_indices):
                        row_lookup[(dataset_index, age, variant)].update(action_metrics[local])

                for gate_mode, candidate in gate_candidates.items():
                    gate_action = action_adapter.decode_action_from_condition(
                        candidate.index_select(0, position),
                        proprio,
                        steps=10,
                        initial_noise=noise,
                        requires_grad=False,
                    )
                    gate_metrics = _action_metrics(
                        gate_action, target_action, prefix="nfe10"
                    )
                    for local, dataset_index in enumerate(selected_dataset_indices):
                        gate_row_lookup[(dataset_index, age, gate_mode)].update(
                            gate_metrics[local]
                        )

                generation_positions = [
                    local
                    for local, dataset_index in enumerate(selected_dataset_indices)
                    if (dataset_index, age) in generation_keys
                ]
                if not generation_positions:
                    continue
                generation_position = torch.tensor(
                    generation_positions, device=device, dtype=torch.long
                )
                actual_condition, actual_code = variant_outputs["ours_full_observation"]
                actual_condition = actual_condition.index_select(0, position).index_select(
                    0, generation_position
                )
                actual_code = actual_code.index_select(0, position).index_select(
                    0, generation_position
                )
                generation_proprio = proprio.index_select(0, generation_position)
                normalized_proprio = action_adapter.normalize_proprio(generation_proprio)
                generation_noise = noise.index_select(0, generation_position)
                generation_target = target_action.index_select(0, generation_position)
                generation_dataset_indices = [
                    selected_dataset_indices[value] for value in generation_positions
                ]
                local_oracle = action_adapter.decode_action_from_condition(
                    actual_condition,
                    generation_proprio,
                    steps=10,
                    initial_noise=generation_noise,
                    requires_grad=False,
                )
                generation_valid_mask = sequence["valid_mask"].index_select(
                    0, position
                ).index_select(0, generation_position)
                zero_code = torch.zeros_like(actual_code)
                naive_action = action_adapter.decode_action_from_condition(
                    actual_condition,
                    generation_proprio,
                    steps=3,
                    initial_noise=generation_noise,
                    requires_grad=False,
                )
                parent_trace, parent_generated = _generation_trace(
                    loop=parent_loop,
                    transformer=frozen_model.transformer,
                    action_space=action_adapter.action_space,
                    condition=actual_condition,
                    valid_mask=generation_valid_mask,
                    normalized_proprio=normalized_proprio,
                    noise=generation_noise,
                    code=zero_code,
                )
                coupled_trace, coupled_generated = _generation_trace(
                    loop=coupled_loop,
                    transformer=frozen_model.transformer,
                    action_space=action_adapter.action_space,
                    condition=actual_condition,
                    valid_mask=generation_valid_mask,
                    normalized_proprio=normalized_proprio,
                    noise=generation_noise,
                    code=actual_code,
                )
                for method, generated, schedule in (
                    ("original_nfe10", local_oracle, tuple(range(10))),
                    ("naive_nfe3", naive_action, None),
                    (
                        "ours_generation_ng3",
                        parent_generated,
                        GENERATION_SCHEDULES[3],
                    ),
                    (
                        "coupled_generation_ng3",
                        coupled_generated,
                        GENERATION_SCHEDULES[3],
                    ),
                ):
                    method_rows, step_rows = _method_fidelity_rows(
                        method=method,
                        prediction=generated,
                        original=local_oracle,
                        teacher=generation_target,
                        dataset_indices=generation_dataset_indices,
                        query_age=age,
                        schedule=schedule,
                    )
                    generation_method_rows.extend(method_rows)
                    action_step_rows.extend(step_rows)

                hidden_rows.extend(
                    _hidden_fidelity_rows(
                        mode="parent_zero_observation_code",
                        trace=parent_trace,
                        transformer=frozen_model.transformer,
                        condition=actual_condition,
                        normalized_proprio=normalized_proprio,
                        dataset_indices=generation_dataset_indices,
                        query_age=age,
                    )
                )
                hidden_rows.extend(
                    _hidden_fidelity_rows(
                        mode="coupled_real_observation_code",
                        trace=coupled_trace,
                        transformer=frozen_model.transformer,
                        condition=actual_condition,
                        normalized_proprio=normalized_proprio,
                        dataset_indices=generation_dataset_indices,
                        query_age=age,
                    )
                )

                schedule_positions = [
                    local
                    for local, dataset_index in enumerate(generation_dataset_indices)
                    if (dataset_index, age) in schedule_keys
                ]
                if schedule_positions:
                    schedule_position = torch.tensor(
                        schedule_positions, device=device, dtype=torch.long
                    )
                    schedule_condition = actual_condition.index_select(
                        0, schedule_position
                    )
                    schedule_normalized_proprio = normalized_proprio.index_select(
                        0, schedule_position
                    )
                    schedule_noise = generation_noise.index_select(0, schedule_position)
                    schedule_target = generation_target.index_select(0, schedule_position)
                    schedule_oracle = local_oracle.index_select(0, schedule_position)
                    schedule_valid_mask = generation_valid_mask.index_select(
                        0, schedule_position
                    )
                    schedule_code = zero_code.index_select(0, schedule_position)
                    schedule_dataset_indices = [
                        generation_dataset_indices[value] for value in schedule_positions
                    ]
                    for schedule in candidate_schedules:
                        _, scheduled_action = _generation_trace(
                            loop=parent_loop,
                            transformer=frozen_model.transformer,
                            action_space=action_adapter.action_space,
                            condition=schedule_condition,
                            valid_mask=schedule_valid_mask,
                            normalized_proprio=schedule_normalized_proprio,
                            noise=schedule_noise,
                            code=schedule_code,
                            full_step_indices=schedule,
                        )
                        method_rows, _ = _method_fidelity_rows(
                            method="ours_generation_ng3_schedule",
                            prediction=scheduled_action,
                            original=schedule_oracle,
                            teacher=schedule_target,
                            dataset_indices=schedule_dataset_indices,
                            query_age=age,
                            schedule=schedule,
                        )
                        schedule_rows.extend(method_rows)

    assign_change_quartiles(condition_rows)
    _write_csv(output / "condition_fidelity_rows.csv", condition_rows)
    _write_csv(output / "generation_hidden_fidelity_rows.csv", hidden_rows)
    _write_csv(output / "gate_ablation_rows.csv", gate_rows)
    _write_csv(output / "generation_method_fidelity_rows.csv", generation_method_rows)
    _write_csv(output / "generation_action_step_rows.csv", action_step_rows)
    _write_csv(output / "generation_schedule_rows.csv", schedule_rows)
    _write_csv(output / "latency_samples.csv", latency_rows)
    condition_summary = _group_summary(
        condition_rows,
        group_fields=("regime", "age", "variant"),
        metrics=CONDITION_METRICS + ACTION_METRICS,
    )
    observation_strata = _group_summary(
        [row for row in condition_rows if row["regime"] == "kc2_local"],
        group_fields=("age", "observation_change_quartile", "variant"),
        metrics=("condition_normalized_mse", "condition_cosine", "nfe10_first5_l1"),
    )
    generation_summary = _group_summary(
        hidden_rows,
        group_fields=("code_mode", "flow_step", "skipped_age"),
        metrics=(
            "hidden_normalized_mse",
            "hidden_cosine",
            "hidden_norm_ratio",
            "velocity_l1",
            "velocity_cosine",
        ),
    )
    gate_summary = _group_summary(
        gate_rows,
        group_fields=("age", "gate_mode"),
        metrics=(
            "condition_normalized_mse",
            "condition_cosine",
            "condition_delta_cosine",
            "nfe10_first5_l1",
            "nfe10_full_chunk_l1",
        ),
    )
    generation_method_summary = _group_summary(
        generation_method_rows,
        group_fields=("method",),
        metrics=(
            "vs_original_first5_l1",
            "vs_original_full_chunk_l1",
            "vs_original_translation_l1",
            "vs_original_rotation_l1",
            "vs_original_gripper_l1",
            "vs_original_first5_cosine",
            "vs_original_full_cosine",
            "vs_teacher_first5_l1",
        ),
    )
    action_step_summary = _group_summary(
        action_step_rows,
        group_fields=("method", "action_step"),
        metrics=("vs_original_l1", "vs_original_cosine"),
    )
    schedule_summary = _group_summary(
        schedule_rows,
        group_fields=("schedule",),
        metrics=(
            "vs_original_first5_l1",
            "vs_original_full_chunk_l1",
            "vs_original_first5_cosine",
            "vs_teacher_first5_l1",
        ),
    )
    all_numeric = [
        float(value)
        for row in (
            condition_rows
            + hidden_rows
            + gate_rows
            + generation_method_rows
            + action_step_rows
            + schedule_rows
            + latency_rows
        )
        for value in row.values()
        if isinstance(value, (float, int)) and not isinstance(value, bool)
    ]
    result = {
        "verdict": "LATENT_FIDELITY_ANALYSIS_COMPLETE",
        "paper_analysis_candidate": True,
        "training_run": False,
        "environment_rollout_run": False,
        "scientific_questions": {
            "latent_approximation": "Exact teacher latent and same-noise downstream action fidelity.",
            "observation_importance": "Paired current/stale/partial/same-task-shuffled observation counterfactuals.",
            "generation_approximation": "Skipped action-transformer hidden state versus a local exact call at the same flow state.",
            "naive_control": "Same-condition and same-noise direct Original NFE10, Naive NFE3, and learned N_G=3 action comparison.",
            "gate_causality": "Inference-only learned/fixed/unit/hold gate controls with the same learned residual.",
            "latency": "Batch-one RTX 5090 module timings with CUDA-event synchronization.",
        },
        "dataset": {
            "heldout_windows_available": len(dataset),
            "heldout_windows_analyzed": len(selected),
            "action_queries_requested": args.action_queries,
            "action_queries_analyzed": len(action_keys),
            "generation_queries_requested": args.generation_queries,
            "generation_queries_analyzed": len(generation_keys),
            "schedule_queries_requested": args.schedule_queries,
            "schedule_queries_analyzed": len(schedule_keys),
            "split_sha256": dataset.split_sha256,
            "split_seed": args.split_seed,
        },
        "checkpoints": {
            "condition_step": int(condition_payload["global_optimizer_step"]),
            "parent_generation_step": int(parent_payload["optimizer_step"]),
            "coupled_projection_step": int(coupled_payload["optimizer_step"]),
        },
        "artifact_contract": artifact_contract,
        "projection_only_state_audit": projection_audit,
        "cache_validation": cache_validation,
        "delta_path": delta_path,
        "dropped_frozen_components": dropped,
        "determinism": determinism,
        "all_numeric_metrics_finite": all(math.isfinite(value) for value in all_numeric),
        "condition_summary": condition_summary,
        "observation_change_strata": observation_strata,
        "generation_hidden_summary": generation_summary,
        "gate_ablation_summary": gate_summary,
        "generation_method_summary": generation_method_summary,
        "generation_action_step_summary": action_step_summary,
        "generation_schedule_summary": schedule_summary,
        "valid_generation_schedules": [list(value) for value in candidate_schedules],
        "latency_summary": latency_summary,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "git_commit": subprocess.check_output(
            ("git", "-C", str(Path(__file__).resolve().parents[5]), "rev-parse", "HEAD"),
            text=True,
        ).strip(),
    }
    write_json(output / "latent_fidelity_analysis.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--condition-checkpoint", required=True)
    parser.add_argument("--parent-generation-checkpoint", required=True)
    parser.add_argument("--coupled-generation-checkpoint", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--condition-windows", type=int, default=0)
    parser.add_argument("--action-queries", type=int, default=512)
    parser.add_argument("--generation-queries", type=int, default=256)
    parser.add_argument("--schedule-queries", type=int, default=128)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--verify-cache-checksums", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
