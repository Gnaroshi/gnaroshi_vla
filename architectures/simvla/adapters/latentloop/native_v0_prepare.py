"""Bounded pre-training gates for corrected native SimVLA V0."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from architectures.simvla.adapters.latentloop.native_v0_checkpoint import NativeV0Config
from architectures.simvla.adapters.latentloop.native_v0_condition_hook import (
    GROUP_NAMES,
    extract_action_condition,
    run_same_noise_k1_parity,
    source_hook_audit,
)
from architectures.simvla.adapters.latentloop.native_v0_dataset import (
    NativeV0SequenceDataset,
    collate_native_v0_sequences,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    UPSTREAM,
    cached_batch_token_layout,
    configure_strict_torch_determinism,
    load_frozen_simvla,
    move_batch,
    native_v0_source_manifest,
    require_gate,
    require_new_output,
    write_json,
)
from architectures.simvla.adapters.latentloop.native_v0_training_cache import (
    build_training_cache,
    first_training_query_record,
    load_training_manifest,
    validate_training_cache,
)
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from methods.latentloop.training.native_simvla_v0 import (
    NativeV0LossWeights,
    decode_age_conditions,
    flattened_gradients,
    native_v0_raw_losses,
    weighted_native_v0_loss,
)


def command_build_training_cache(args: argparse.Namespace) -> dict[str, Any]:
    return build_training_cache(
        output=args.output,
        teacher_cache=args.teacher_cache,
        dataset_root=args.dataset_root,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        action_noise_seed_base=args.action_noise_seed_base,
    )


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _tokenizer_metadata(processor: Any) -> tuple[int | None, Sequence[int]]:
    tokenizer = processor.tokenizer
    return getattr(tokenizer, "pad_token_id", None), getattr(tokenizer, "all_special_ids", ())


def _official_training_image_inputs(
    raw_rgb: torch.Tensor,
    *,
    image_size: int = 384,
    num_views: int = 3,
) -> dict[str, torch.Tensor]:
    """Reproduce the no-augmentation SimVLA training image transform exactly."""

    if raw_rgb.ndim != 4 or raw_rgb.shape[-1] != 3:
        raise ValueError(f"raw_rgb must have shape [V,H,W,3], got {tuple(raw_rgb.shape)}")
    if raw_rgb.shape[0] > num_views:
        raise ValueError(f"raw_rgb has {raw_rgb.shape[0]} views, but num_views={num_views}")
    image_transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                inplace=True,
            ),
        ]
    )
    views = [
        image_transform(Image.fromarray(image.cpu().numpy().astype(np.uint8)))
        for image in raw_rgb
    ]
    if not views:
        raise ValueError("raw_rgb must contain at least one camera view")
    padding = views[0].new_zeros((num_views - len(views), *views[0].shape))
    image_input = torch.cat((torch.stack(views, dim=0), padding), dim=0).unsqueeze(0)
    image_mask = torch.zeros((1, num_views), dtype=torch.bool)
    image_mask[:, : len(views)] = True
    return {"image_input": image_input, "image_mask": image_mask}


def _processed_cache_record(record: dict[str, Any], processor: Any, device: torch.device) -> dict[str, torch.Tensor]:
    # The compact cache was produced by the official training dataloader. Keep
    # this parity check on that source path rather than the optimized tensor path.
    processed = _official_training_image_inputs(
        record["raw_rgb"],
        image_size=int(getattr(processor, "image_size", 384)),
        num_views=int(getattr(processor, "num_views", 3)),
    )
    processed.update(processor.encode_language([record["language_instruction"]]))
    return {key: value.to(device) for key, value in processed.items()}


def _split_contracts(args: argparse.Namespace) -> dict[str, Any]:
    train = NativeV0SequenceDataset(
        args.cache,
        split="train",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    heldout = NativeV0SequenceDataset(
        args.cache,
        split="heldout",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    return {
        "split_seed": int(args.split_seed),
        "heldout_fraction": float(args.heldout_fraction),
        "train_split_sha256": train.split_sha256,
        "heldout_split_sha256": heldout.split_sha256,
        "train_sequences": len(train),
        "heldout_sequences": len(heldout),
    }


def command_audit(args: argparse.Namespace) -> dict[str, Any]:
    output = require_new_output(args.output)
    source = native_v0_source_manifest(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        cache=args.cache,
    )
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.transformer_smolvlm import SmolVLMActionTransformer

    hook = source_hook_audit(SmolVLMVLA, SmolVLMActionTransformer)
    cache_validation = validate_training_cache(args.cache, verify_sequence_hashes=True)
    cache_manifest = load_training_manifest(args.cache)
    cache_compatibility = {
        "checkpoint_identifier_matches": cache_manifest.get("checkpoint") == args.checkpoint,
        "checkpoint_revision_matches": (
            cache_manifest.get("checkpoint_revision") == source["checkpoint"].get("revision")
        ),
        "simvla_upstream_commit_matches": (
            cache_manifest.get("simvla_upstream_commit") == source["simvla_upstream_commit"]
        ),
        "selected_physical_gpu_ids_match": (
            cache_manifest.get("selected_physical_gpu_ids")
            == source["selected_physical_gpu_ids"]
        ),
        "official_training_demonstrations_only": (
            cache_manifest.get("data_role") == "official_libero_training_demonstrations"
        ),
        "final_eval_episode_overlap_false": (
            cache_manifest.get("metadata", {}).get("train_test_episode_overlap") is False
        ),
    }
    source_lock_complete = all(
        not str(source[key]).startswith("ERROR:")
        for key in ("simvla_upstream_commit", "libero_commit")
    )
    passed = (
        hook["verdict"] == "SOURCE_EXACT_PRE_VLM_PROJ"
        and cache_validation["passed"]
        and all(cache_compatibility.values())
        and source_lock_complete
    )
    result = {
        "verdict": "SOURCE_AUDIT_PASS" if passed else "SOURCE_AUDIT_FAIL",
        "source_combined_sha256": source["combined_sha256"],
        "source": source,
        "condition_hook": hook,
        "cache_validation": cache_validation,
        "cache_compatibility": cache_compatibility,
        "source_lock_complete": source_lock_complete,
        "dataset_splits": _split_contracts(args),
        "contract": {
            "action_horizon": 10,
            "execution_horizon": 5,
            "fixed_k": 4,
            "refresh_queries": [0, 4],
            "update_ages": [1, 2, 3],
            "updater_inputs": ["previous_condition", "previous/current_all_camera_images", "previous/current_proprioception", "age"],
            "executed_action_input": False,
            "action_correction": False,
            "v1_v2": False,
        },
    }
    write_json(output / "source_audit.json", result)
    write_json(output / "source_lock.json", source)
    return result


def command_parity(args: argparse.Namespace) -> dict[str, Any]:
    output = require_new_output(args.output)
    source = native_v0_source_manifest(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        cache=args.cache,
    )
    require_gate(
        args.source_gate,
        verdicts=("SOURCE_AUDIT_PASS",),
        source_combined_sha256=source["combined_sha256"],
    )
    determinism = configure_strict_torch_determinism(args.seed)
    device = _device(args.device)
    model, processor, _ = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    record = first_training_query_record(args.cache)
    processed = _processed_cache_record(record, processor, device)
    pad_id, special_ids = _tokenizer_metadata(processor)
    with torch.no_grad():
        extracted = extract_action_condition(
            model,
            input_ids=processed["input_ids"],
            image_input=processed["image_input"],
            image_mask=processed["image_mask"],
            pad_token_id=pad_id,
            special_token_ids=special_ids,
        )
        parity = run_same_noise_k1_parity(
            model,
            input_ids=processed["input_ids"],
            image_input=processed["image_input"],
            image_mask=processed["image_mask"],
            proprio=record["proprio"].unsqueeze(0).to(device),
            initial_noise=record["initial_noise"].unsqueeze(0).to(device),
            steps=10,
            pad_token_id=pad_id,
            special_token_ids=special_ids,
        )
    cached = record["full_condition"].unsqueeze(0).to(device)
    cache_difference = (extracted.condition.float() - cached.float()).abs()
    cache_allclose = bool(torch.allclose(extracted.condition.float(), cached.float(), atol=1e-5, rtol=1e-5))
    passed = parity["verdict"] == "K1_HOOK_PARITY_PASS" and cache_allclose
    result = {
        **parity,
        "verdict": "K1_HOOK_PARITY_PASS" if passed else "K1_HOOK_PARITY_FAIL",
        "source_combined_sha256": source["combined_sha256"],
        "cache_condition_reencode": {
            "allclose_1e_5": cache_allclose,
            "max_abs": float(cache_difference.max().item()),
            "mean_abs": float(cache_difference.mean().item()),
            "image_preprocessing_contract": "official_training_no_augmentation_bicubic_384",
            "cached_condition_source": "official_training_teacher_cache",
        },
        "checkpoint": args.checkpoint,
        "norm_stats": str(Path(args.norm_stats).resolve()),
        "cache": str(Path(args.cache).resolve()),
        "determinism": determinism,
        "dataset_splits": _split_contracts(args),
    }
    write_json(output / "k1_parity.json", result)
    write_json(output / "source_lock.json", source)
    return result


def _quantiles(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _effective_rank(matrix: torch.Tensor) -> dict[str, float | int]:
    if matrix.numel() == 0:
        return {"rows": 0, "columns": 0, "entropy_effective_rank": 0.0, "rank90": 0, "rank95": 0}
    matrix = matrix.float()
    if matrix.shape[0] > 4096:
        indices = torch.linspace(0, matrix.shape[0] - 1, 4096).long()
        matrix = matrix[indices]
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(torch.finfo(energy.dtype).eps)
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum()
    cumulative = probability.cumsum(0)
    return {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "entropy_effective_rank": float(entropy.exp().item()),
        "rank90": int((cumulative < 0.90).sum().item() + 1),
        "rank95": int((cumulative < 0.95).sum().item() + 1),
    }


def command_token_analysis(args: argparse.Namespace) -> dict[str, Any]:
    output = require_new_output(args.output)
    source = native_v0_source_manifest(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        cache=args.cache,
    )
    require_gate(
        args.parity_gate,
        verdicts=("K1_HOOK_PARITY_PASS",),
        source_combined_sha256=source["combined_sha256"],
    )
    configure_strict_torch_determinism(args.seed)
    dataset = NativeV0SequenceDataset(
        args.cache,
        split="train",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    device = _device(args.device)
    _, processor, decoder = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    raw_rows: list[dict[str, Any]] = []
    delta_matrices: dict[int, list[torch.Tensor]] = {group: [] for group in GROUP_NAMES}
    full_valid_delta_matrices: list[torch.Tensor] = []
    count = min(int(args.max_sequences), len(dataset))
    for dataset_index in range(count):
        batch = move_batch(collate_native_v0_sequences([dataset[dataset_index]]), device)
        layout = cached_batch_token_layout(
            condition=batch["anchor_condition"],
            language_instructions=batch["language_instruction"],
            processor=processor,
        )
        previous_conditions = [batch["anchor_condition"], batch["teacher_conditions"][:, 0], batch["teacher_conditions"][:, 1]]
        for offset, age in enumerate((1, 2, 3)):
            previous = previous_conditions[offset]
            current = batch["teacher_conditions"][:, offset]
            proprio = batch["proprio_sequence"][:, age]
            noise = batch["explicit_noises"][:, offset]
            full_valid_delta_matrices.append(
                (current - previous)[layout.valid_mask].detach().cpu()
            )
            with torch.no_grad():
                reference_action = decoder.decode_action_from_condition(
                    current,
                    proprio,
                    steps=10,
                    initial_noise=noise,
                )
            for group_id, group_name in GROUP_NAMES.items():
                group_mask = layout.valid_mask & (layout.group_ids == group_id)
                if not bool(group_mask.any()):
                    continue
                token_delta = current - previous
                selected = token_delta[group_mask]
                delta_matrices[group_id].append(selected.detach().cpu())
                previous_selected = previous[group_mask]
                current_selected = current[group_mask]
                cosine = F.cosine_similarity(previous_selected.float(), current_selected.float(), dim=-1)
                replaced = current.clone()
                replaced[group_mask] = previous[group_mask]
                with torch.no_grad():
                    replacement_action = decoder.decode_action_from_condition(
                        replaced,
                        proprio,
                        steps=10,
                        initial_noise=noise,
                    )
                action_difference = (replacement_action - reference_action).abs()
                first5 = action_difference[:, :5]
                raw_rows.append(
                    {
                        "dataset_index": dataset_index,
                        "suite": "libero_10_training_cache",
                        "task_id": int(batch["task_id"][0].item()),
                        "episode_id": batch["episode_id"][0],
                        "anchor_query_index": int(batch["anchor_query_index"][0].item()),
                        "age": age,
                        "group_id": group_id,
                        "group": group_name,
                        "valid_tokens": int(group_mask.sum().item()),
                        "condition_change_norm_mean": float(selected.float().norm(dim=-1).mean().item()),
                        "condition_cosine_mean": float(cosine.mean().item()),
                        "first5_action_l1": float(first5.mean().item()),
                        "full_chunk_action_l1": float(action_difference.mean().item()),
                        "translation_l1": float(first5[..., :3].mean().item()),
                        "rotation_l1": float(first5[..., 3:6].mean().item()),
                        "continuous_gripper_l1": float(first5[..., 6:].mean().item()),
                        "previous_proprio": batch["proprio_sequence"][0, age - 1].detach().cpu().tolist(),
                        "current_proprio": batch["proprio_sequence"][0, age].detach().cpu().tolist(),
                    }
                )

    raw_path = output / "token_action_sensitivity.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault((int(row["age"]), int(row["group_id"])), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    metric_names = (
        "condition_change_norm_mean",
        "condition_cosine_mean",
        "first5_action_l1",
        "full_chunk_action_l1",
        "translation_l1",
        "rotation_l1",
        "continuous_gripper_l1",
    )
    for (age, group_id), rows in sorted(grouped.items()):
        row: dict[str, Any] = {"age": age, "group_id": group_id, "group": GROUP_NAMES[group_id], "samples": len(rows)}
        for metric in metric_names:
            for statistic, value in _quantiles([float(item[metric]) for item in rows]).items():
                row[f"{metric}_{statistic}"] = value
        summary_rows.append(row)
    csv_path = output / "simvla_condition_token_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    effective_ranks = {
        GROUP_NAMES[group]: _effective_rank(torch.cat(values, dim=0))
        for group, values in delta_matrices.items()
        if values
    }
    effective_ranks["full_valid_tokens"] = _effective_rank(
        torch.cat(full_valid_delta_matrices, dim=0)
    )
    all_pairs = [
        (float(row["condition_change_norm_mean"]), float(row["first5_action_l1"]))
        for row in raw_rows
    ]
    correlation = float(np.corrcoef(np.asarray(all_pairs).T)[0, 1]) if len(all_pairs) > 1 else math.nan
    finite = all(
        math.isfinite(float(value))
        for row in raw_rows
        for key, value in row.items()
        if key.endswith("_l1") or key.endswith("_mean")
    )
    verdict = "TOKEN_ANALYSIS_PASS" if raw_rows and finite else "TOKEN_ANALYSIS_FAIL"
    mask_contract = {
        "verdict": verdict,
        "source_combined_sha256": source["combined_sha256"],
        "decision": "update_all_source_valid_tokens_copy_batch_padding",
        "group_names": GROUP_NAMES,
        "valid_language_pad_ids_remain_updateable": True,
        "padding_group_id": 0,
        "online_success_rate_used_for_decision": False,
        "training_split_only": True,
        "sequences": count,
        "train_split_sha256": dataset.split_sha256,
    }
    write_json(output / "simvla_condition_update_mask_contract.json", mask_contract)
    analysis = {
        "verdict": verdict,
        "source_combined_sha256": source["combined_sha256"],
        "bounded_training_sequences": count,
        "raw_rows": len(raw_rows),
        "effective_ranks": effective_ranks,
        "condition_delta_vs_first5_sensitivity_pearson": correlation,
        "full_jacobian_computed": False,
        "final_libero_long_episodes_used": False,
        "mask_decision": mask_contract["decision"],
        "dataset_splits": _split_contracts(args),
    }
    write_json(output / "token_analysis_gate.json", analysis)
    markdown = [
        "# SimVLA condition token analysis",
        "",
        f"- Verdict: `{verdict}`",
        f"- Training-only native-R5 sequences: `{count}`",
        f"- Group/age sensitivity records: `{len(raw_rows)}`",
        "- Replacement keeps current state/noise and substitutes exactly one token group from the previous full condition.",
        "- Default mask updates every source-valid token and copies only true batch padding.",
        "- No online success rate and no full Jacobian were used.",
        "",
        "Effective ranks:",
        "```json",
        json.dumps(effective_ranks, indent=2, sort_keys=True),
        "```",
    ]
    (output / "simvla_condition_token_analysis.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    write_json(output / "source_lock.json", source)
    return analysis


def command_parameters(args: argparse.Namespace) -> dict[str, Any]:
    output = require_new_output(args.output)
    source = native_v0_source_manifest(checkpoint=args.checkpoint, norm_stats=args.norm_stats, cache=args.cache)
    require_gate(args.analysis_gate, verdicts=("TOKEN_ANALYSIS_PASS",), source_combined_sha256=source["combined_sha256"])
    primary = NativeV0Config(rank_dim=64)
    primary.validate_primary()
    model = primary.build()
    audit = model.parameter_audit()
    rank96 = NativeV0Config(rank_dim=96).build().parameter_audit()
    passed = bool(audit["under_hard_cap_1000000"] and audit["in_target_range_500000_1000000"])
    result = {
        "verdict": "PARAMETER_AUDIT_PASS" if passed else "PARAMETER_AUDIT_FAIL",
        "source_combined_sha256": source["combined_sha256"],
        "primary_rank64": audit,
        "disabled_rank96_template": {**rank96, "enabled": False, "training_started": False},
        "capacity_sweep": False,
        "dataset_splits": _split_contracts(args),
    }
    write_json(output / "simvla_v0_parameter_audit.json", result)
    return result


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def command_mode_ab(args: argparse.Namespace) -> dict[str, Any]:
    output = require_new_output(args.output)
    source = native_v0_source_manifest(checkpoint=args.checkpoint, norm_stats=args.norm_stats, cache=args.cache)
    require_gate(args.parameter_gate, verdicts=("PARAMETER_AUDIT_PASS",), source_combined_sha256=source["combined_sha256"])
    configure_strict_torch_determinism(args.seed)
    dataset = NativeV0SequenceDataset(args.cache, split="train", heldout_fraction=args.heldout_fraction, split_seed=args.split_seed)
    device = _device(args.device)
    frozen_model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )

    def decoder(condition: torch.Tensor, proprio: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return action_adapter.decode_action_from_condition(
            condition,
            proprio,
            steps=10,
            initial_noise=noise,
            requires_grad=True,
        )

    seed_model = NativeV0Config().build()
    initial_state = seed_model.state_dict()
    rows: list[dict[str, Any]] = []
    count = min(int(args.sequences), len(dataset))
    for dataset_index in range(count):
        batch = move_batch(collate_native_v0_sequences([dataset[dataset_index]]), device)
        layout = cached_batch_token_layout(condition=batch["anchor_condition"], language_instructions=batch["language_instruction"], processor=processor)
        mode_results: dict[str, dict[str, Any]] = {}
        for mode in ("A", "B"):
            adapter = NativeV0Config().build().to(device)
            adapter.load_state_dict(initial_state, strict=True)
            adapter.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _synchronize(device)
            started = time.perf_counter()
            unroll = adapter(
                batch["anchor_condition"],
                batch["image_sequence"],
                batch["proprio_sequence"],
                valid_mask=layout.valid_mask,
                group_ids=layout.group_ids,
            )
            actions = decode_age_conditions(
                decoder,
                unroll.conditions,
                tuple(batch["proprio_sequence"][:, age] for age in (1, 2, 3)),
                tuple(batch["explicit_noises"][:, index] for index in range(3)),
                mode=mode,
            )
            full_conditions = tuple(
                batch["teacher_conditions"][:, index].detach() for index in range(3)
            )
            full_actions = decode_age_conditions(
                lambda condition, proprio, noise: action_adapter.decode_action_from_condition(
                    condition,
                    proprio,
                    steps=10,
                    initial_noise=noise,
                    requires_grad=False,
                ),
                full_conditions,
                tuple(batch["proprio_sequence"][:, age] for age in (1, 2, 3)),
                tuple(batch["explicit_noises"][:, index] for index in range(3)),
                mode=mode,
            )
            raw = native_v0_raw_losses(
                unroll=unroll,
                teacher_conditions=full_conditions,
                predicted_actions=actions,
                teacher_actions=full_actions,
                valid_mask=layout.valid_mask,
            )
            total, _ = weighted_native_v0_loss(raw, NativeV0LossWeights(1, 1, 1, 1, 1e-3))
            total.backward()
            _synchronize(device)
            elapsed = time.perf_counter() - started
            mode_results[mode] = {
                "total": float(total.detach().item()),
                "first5": float(raw["first5_action_l1"].detach().item()),
                "actions": [action.detach().cpu() for action in actions],
                "gradient": flattened_gradients(adapter).cpu(),
                "elapsed_seconds": elapsed,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                "device_total_memory_bytes": (
                    int(torch.cuda.get_device_properties(device).total_memory)
                    if device.type == "cuda"
                    else 0
                ),
                "finite": bool(torch.isfinite(total).item() and torch.isfinite(flattened_gradients(adapter)).all().item()),
            }
            del adapter
        gradient_a = mode_results["A"]["gradient"]
        gradient_b = mode_results["B"]["gradient"]
        gradient_cosine = float(F.cosine_similarity(gradient_a.unsqueeze(0), gradient_b.unsqueeze(0)).item())
        gradient_relative = float((gradient_a - gradient_b).norm().item() / max(gradient_a.norm().item(), gradient_b.norm().item(), 1e-12))
        action_max = max(
            float((left - right).abs().max().item())
            for left, right in zip(mode_results["A"]["actions"], mode_results["B"]["actions"])
        )
        rows.append(
            {
                "dataset_index": dataset_index,
                "total_loss_relative_difference": _relative_difference(mode_results["A"]["total"], mode_results["B"]["total"]),
                "first5_loss_relative_difference": _relative_difference(mode_results["A"]["first5"], mode_results["B"]["first5"]),
                "gradient_cosine": gradient_cosine,
                "gradient_relative_error": gradient_relative,
                "action_max_abs_difference": action_max,
                "mode_a_seconds": mode_results["A"]["elapsed_seconds"],
                "mode_b_seconds": mode_results["B"]["elapsed_seconds"],
                "speedup": mode_results["A"]["elapsed_seconds"] / mode_results["B"]["elapsed_seconds"],
                "mode_a_peak_vram_bytes": mode_results["A"]["peak_vram_bytes"],
                "mode_b_peak_vram_bytes": mode_results["B"]["peak_vram_bytes"],
                "finite": bool(mode_results["A"]["finite"] and mode_results["B"]["finite"]),
                "ages": [1, 2, 3],
            }
        )
    aggregate = {
        "max_total_loss_relative_difference": max(row["total_loss_relative_difference"] for row in rows),
        "max_first5_loss_relative_difference": max(row["first5_loss_relative_difference"] for row in rows),
        "min_gradient_cosine": min(row["gradient_cosine"] for row in rows),
        "max_gradient_relative_error": max(row["gradient_relative_error"] for row in rows),
        "median_speedup": float(np.median([row["speedup"] for row in rows])),
        "max_mode_b_peak_vram_bytes": max(row["mode_b_peak_vram_bytes"] for row in rows),
        "device_total_memory_bytes": int(mode_results["B"]["device_total_memory_bytes"]),
        "mode_b_peak_vram_fits": bool(
            device.type != "cuda"
            or max(row["mode_b_peak_vram_bytes"] for row in rows)
            < int(mode_results["B"]["device_total_memory_bytes"])
        ),
        "all_finite": all(row["finite"] for row in rows),
        "all_ages_represented": all(row["ages"] == [1, 2, 3] for row in rows),
    }
    passed = bool(
        aggregate["max_total_loss_relative_difference"] <= 0.005
        and aggregate["max_first5_loss_relative_difference"] <= 0.005
        and aggregate["min_gradient_cosine"] >= 0.999
        and aggregate["max_gradient_relative_error"] <= 0.01
        and aggregate["median_speedup"] >= 1.5
        and aggregate["all_finite"]
        and aggregate["all_ages_represented"]
        and aggregate["mode_b_peak_vram_fits"]
    )
    result = {
        "verdict": "MODE_B_LOCAL_PASS" if passed else "MODE_B_LOCAL_FAIL",
        "source_combined_sha256": source["combined_sha256"],
        "sequences": count,
        "aggregate": aggregate,
        "rows": rows,
        "frozen_base_gradients": all(parameter.grad is None for parameter in frozen_model.parameters()),
        "dataset_splits": _split_contracts(args),
    }
    # Tensor payloads were removed from rows before serialization.
    write_json(output / "mode_ab_local.json", result)
    return result


def command_mode_ab_decide(args: argparse.Namespace) -> dict[str, Any]:
    output = require_new_output(args.output)
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.local_report]
    if len(reports) != 2:
        raise ValueError("exactly two per-GPU Mode A/B reports are required")
    hashes = {report.get("source_combined_sha256") for report in reports}
    if len(hashes) != 1:
        raise RuntimeError("Mode A/B reports use different source locks")
    split_hashes = {
        json.dumps(report.get("dataset_splits"), sort_keys=True) for report in reports
    }
    if len(split_hashes) != 1:
        raise RuntimeError("Mode A/B reports use different train/held-out splits")
    passed = all(report.get("verdict") == "MODE_B_LOCAL_PASS" for report in reports)
    result = {
        "verdict": "MODE_B_APPROVED" if passed else "MODE_A_REQUIRED",
        "source_combined_sha256": hashes.pop(),
        "scientific_training_mode": "B" if passed else "A",
        "two_selected_gpu_reports": [str(Path(path).resolve()) for path in args.local_report],
        "mode_b_changes_recursion_or_loss": False,
        "dataset_splits": reports[0]["dataset_splits"],
    }
    write_json(output / "mode_ab_decision.json", result)
    return result


def command_calibrate(args: argparse.Namespace) -> dict[str, Any]:
    output = require_new_output(args.output)
    source = native_v0_source_manifest(checkpoint=args.checkpoint, norm_stats=args.norm_stats, cache=args.cache)
    configure_strict_torch_determinism(args.seed)
    mode_gate = require_gate(args.mode_gate, verdicts=("MODE_B_APPROVED", "MODE_A_REQUIRED"), source_combined_sha256=source["combined_sha256"])
    mode = str(mode_gate["scientific_training_mode"])
    dataset = NativeV0SequenceDataset(args.cache, split="train", heldout_fraction=args.heldout_fraction, split_seed=args.split_seed)
    device = _device(args.device)
    _, processor, action_adapter = load_frozen_simvla(checkpoint=args.checkpoint, norm_stats=args.norm_stats, smolvlm_model=args.smolvlm_model, device=device)
    adapter = NativeV0Config().build().to(device).train()

    def decoder(condition: torch.Tensor, proprio: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return action_adapter.decode_action_from_condition(condition, proprio, steps=10, initial_noise=noise, requires_grad=False)

    metrics: dict[str, list[float]] = {}
    count = min(int(args.sequences), len(dataset))
    for index in range(count):
        batch = move_batch(collate_native_v0_sequences([dataset[index]]), device)
        layout = cached_batch_token_layout(condition=batch["anchor_condition"], language_instructions=batch["language_instruction"], processor=processor)
        with torch.no_grad():
            unroll = adapter(batch["anchor_condition"], batch["image_sequence"], batch["proprio_sequence"], valid_mask=layout.valid_mask, group_ids=layout.group_ids)
            actions = decode_age_conditions(
                decoder,
                unroll.conditions,
                tuple(batch["proprio_sequence"][:, age] for age in (1, 2, 3)),
                tuple(batch["explicit_noises"][:, age - 1] for age in (1, 2, 3)),
                mode=mode,
            )
            full_conditions = tuple(
                batch["teacher_conditions"][:, age - 1].detach() for age in (1, 2, 3)
            )
            full_actions = decode_age_conditions(
                decoder,
                full_conditions,
                tuple(batch["proprio_sequence"][:, age] for age in (1, 2, 3)),
                tuple(batch["explicit_noises"][:, age - 1] for age in (1, 2, 3)),
                mode=mode,
            )
            raw = native_v0_raw_losses(
                unroll=unroll,
                teacher_conditions=full_conditions,
                predicted_actions=actions,
                teacher_actions=full_actions,
                valid_mask=layout.valid_mask,
            )
        for name, value in raw.items():
            metrics.setdefault(name, []).append(float(value.item()))
    summary = {name: _quantiles(values) for name, values in metrics.items()}
    result = {
        "verdict": "LOSS_SCALE_CALIBRATION_COMPLETE",
        "source_combined_sha256": source["combined_sha256"],
        "mode": mode,
        "sequences": count,
        "raw_loss_summary": summary,
        "weights_approved": False,
        "dataset_splits": _split_contracts(args),
        "note": "A human must explicitly create an approved weight file; calibration does not approve or start training.",
    }
    write_json(output / "loss_scale_calibration.json", result)
    write_json(
        output / "approved_loss_weights.template.json",
        {
            "approved_by_user": False,
            "source_combined_sha256": source["combined_sha256"],
            "train_split_sha256": dataset.split_sha256,
            "condition": None,
            "first5_action": None,
            "full_chunk_action": None,
            "continuous_gripper": None,
            "update_regularization": None,
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    compact = subparsers.add_parser("build-training-cache")
    compact.add_argument("--output", required=True)
    compact.add_argument("--teacher-cache", required=True)
    compact.add_argument("--dataset-root", required=True)
    compact.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    compact.add_argument("--norm-stats", required=True)
    compact.add_argument("--action-noise-seed-base", type=int, default=20260822)
    compact.set_defaults(handler=command_build_training_cache)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", required=True)
    common.add_argument("--cache", required=True)
    common.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    common.add_argument("--norm-stats", required=True)
    common.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    common.add_argument("--device", default="cuda:0")
    common.add_argument("--heldout-fraction", type=float, default=0.2)
    common.add_argument("--split-seed", type=int, default=20260822)
    common.add_argument("--seed", type=int, default=20260815)

    audit = subparsers.add_parser("audit", parents=[common])
    audit.set_defaults(handler=command_audit)
    parity = subparsers.add_parser("parity", parents=[common])
    parity.add_argument("--source-gate", required=True)
    parity.set_defaults(handler=command_parity)
    analysis = subparsers.add_parser("token-analysis", parents=[common])
    analysis.add_argument("--parity-gate", required=True)
    analysis.add_argument("--max-sequences", type=int, default=16)
    analysis.set_defaults(handler=command_token_analysis)
    parameters = subparsers.add_parser("parameters", parents=[common])
    parameters.add_argument("--analysis-gate", required=True)
    parameters.set_defaults(handler=command_parameters)
    mode = subparsers.add_parser("mode-ab", parents=[common])
    mode.add_argument("--parameter-gate", required=True)
    mode.add_argument("--sequences", type=int, default=4)
    mode.set_defaults(handler=command_mode_ab)
    decision = subparsers.add_parser("mode-ab-decide")
    decision.add_argument("--output", required=True)
    decision.add_argument("--local-report", action="append", required=True)
    decision.set_defaults(handler=command_mode_ab_decide)
    calibrate = subparsers.add_parser("calibrate", parents=[common])
    calibrate.add_argument("--mode-gate", required=True)
    calibrate.add_argument("--sequences", type=int, default=16)
    calibrate.set_defaults(handler=command_calibrate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = args.handler(args)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
