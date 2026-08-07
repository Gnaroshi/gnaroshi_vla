"""Held-out query-cache evaluation for LatentLoop and matched baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
for path in (ROOT, UPSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter  # noqa: E402
from architectures.simvla.adapters.latentloop.checkpoint import (  # noqa: E402
    freeze_module,
    load_adapter_checkpoint,
)
from architectures.simvla.adapters.latentloop.source_lock import (  # noqa: E402
    collect_source_lock,
    require_empty_output,
)
from methods.latentloop.eval import distribution_summary  # noqa: E402
from methods.latentloop.training import (  # noqa: E402
    QueryCacheDataset,
    collate_query_records,
    deterministic_episode_split_indices,
    normalized_condition_mse,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _parse_adapter_specs(values: list[str]) -> dict[str, Path]:
    specs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--adapter values must be VARIANT=CHECKPOINT")
        variant, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        specs[variant] = path
    return specs


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _action_errors(prediction: Tensor, target: Tensor, lengths: Tensor) -> dict[str, Tensor]:
    difference = prediction - target
    horizon = prediction.shape[1]
    mask = torch.arange(horizon, device=prediction.device).unsqueeze(0) < lengths.unsqueeze(1)
    expanded = mask.unsqueeze(-1).expand_as(prediction)
    full_flat = difference.flatten(start_dim=1)
    prefix_values = difference[expanded].reshape(prediction.shape[0], -1)
    predicted_switch = (prediction[:, 1:, 6] >= 0) != (prediction[:, :-1, 6] >= 0)
    target_switch = (target[:, 1:, 6] >= 0) != (target[:, :-1, 6] >= 0)
    return {
        "chunk_l1": difference.abs().flatten(start_dim=1).mean(dim=1),
        "chunk_l2": torch.linalg.vector_norm(full_flat, dim=1),
        "prefix_l1": prefix_values.abs().mean(dim=1),
        "prefix_l2": torch.linalg.vector_norm(prefix_values, dim=1),
        "gripper_switch_classification_accuracy": (
            predicted_switch == target_switch
        ).float().mean(dim=1),
    }


def _paired_bootstrap_ci(differences: list[float], seed: int, samples: int = 5000) -> list[float | None]:
    if not differences:
        return [None, None]
    values = torch.tensor(differences, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0,
        len(values),
        (samples, len(values)),
        generator=generator,
    )
    means = values[indices].mean(dim=1)
    return [float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate hold and supplied checkpoints on a deterministic held-out split."""

    from models.modeling_smolvlm_vla import SmolVLMVLA

    if args.progress_interval < 1:
        raise ValueError("--progress-interval must be positive")
    output = require_empty_output(args.output)
    source_lock = collect_source_lock(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    _write_json(output / "source_lock.json", source_lock)
    dataset = QueryCacheDataset(args.cache)
    if int(dataset.manifest["execution_horizon"]) != args.execution_horizon:
        raise ValueError("cache execution horizon does not match the evaluation protocol")
    _, holdout = deterministic_episode_split_indices(
        dataset,
        heldout_fraction=args.heldout_fraction,
        seed=args.split_seed,
    )
    if args.max_records > 0:
        holdout = holdout[: args.max_records]
    loader = DataLoader(
        Subset(dataset, holdout),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_query_records,
        pin_memory=args.device.startswith("cuda"),
    )
    device = torch.device(args.device)
    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device)
    model.action_space.load_norm_stats(args.norm_stats)
    freeze_module(model)
    action_adapter = SimVLAActionAdapter(model)
    adapters: dict[str, Any] = {}
    checkpoints = _parse_adapter_specs(args.adapter)
    for variant, path in checkpoints.items():
        adapter, payload = load_adapter_checkpoint(path, device=device)
        if adapter.variant != variant:
            raise ValueError(f"checkpoint {path} is {adapter.variant}, not {variant}")
        adapter.eval()
        adapters[variant] = (adapter, payload)
    row_values: dict[str, dict[str, list[float]]] = {
        "hold_condition": {},
        **{variant: {} for variant in adapters},
    }
    row_latency: dict[str, list[float]] = {row: [] for row in row_values}
    per_record: list[dict[str, Any]] = []
    teacher_reload_diffs: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress_path = output / "eval_progress.jsonl"
    started = time.time()
    processed_records = 0
    progress_bar = tqdm(
        loader,
        total=len(loader),
        desc=f"LatentLoop offline R{args.execution_horizon}",
        dynamic_ncols=True,
        mininterval=args.tqdm_mininterval,
        disable=args.disable_tqdm,
    )
    for batch_index, raw_batch in enumerate(progress_bar, start=1):
        batch = _to_device(raw_batch, device)
        with torch.no_grad():
            teacher_reload = action_adapter.decode_action_from_condition(
                batch["next_full_condition"],
                batch["next_proprio"],
                steps=args.flow_steps,
                initial_noise=batch["next_initial_noise"],
            )
            teacher_reload_diffs.extend(
                (teacher_reload - batch["next_teacher_action_chunk"])
                .abs()
                .flatten(start_dim=1)
                .max(dim=1)
                .values.cpu()
                .tolist()
            )
        row_predictions: dict[str, tuple[Tensor | None, Tensor]] = {}
        _synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            hold_action = action_adapter.decode_action_from_condition(
                batch["full_condition"],
                batch["next_proprio"],
                steps=args.flow_steps,
                initial_noise=batch["next_initial_noise"],
            )
        _synchronize(device)
        row_latency["hold_condition"].append(
            1000.0 * (time.perf_counter() - started) / batch["proprio"].shape[0]
        )
        row_predictions["hold_condition"] = (batch["full_condition"], hold_action)
        for variant, (adapter, _) in adapters.items():
            _synchronize(device)
            started = time.perf_counter()
            with torch.no_grad():
                observation = adapter.encode_observation(
                    batch["raw_rgb"],
                    batch["next_raw_rgb"],
                    batch["proprio"],
                    batch["next_proprio"],
                )
                action_feature = adapter.encode_executed_actions(
                    batch["executed_subchunk"],
                    batch["execution_horizon"],
                    batch["elapsed_time"],
                    reference_feature=observation,
                )
                age = torch.ones_like(batch["execution_horizon"])
                if variant == "action_chunk_correction":
                    condition = None
                    action = adapter.correct_action_chunk(
                        batch["teacher_action_chunk"],
                        observation,
                        action_feature,
                        execution_horizon=batch["execution_horizon"],
                        elapsed_time=batch["elapsed_time"],
                        query_age=age,
                    )
                elif variant == "nonrecurrent_condition":
                    condition = adapter.predict_nonrecurrent_condition(
                        batch["full_condition"],
                        observation,
                        action_feature,
                        execution_horizon=batch["execution_horizon"],
                        elapsed_time=batch["elapsed_time"],
                        query_age=age,
                    )
                    action = action_adapter.decode_action_from_condition(
                        condition,
                        batch["next_proprio"],
                        steps=args.flow_steps,
                        initial_noise=batch["next_initial_noise"],
                    )
                else:
                    condition = adapter.update_recurrent_condition(
                        batch["full_condition"],
                        observation,
                        action_feature,
                        execution_horizon=batch["execution_horizon"],
                        elapsed_time=batch["elapsed_time"],
                        query_age=age,
                    )
                    action = action_adapter.decode_action_from_condition(
                        condition,
                        batch["next_proprio"],
                        steps=args.flow_steps,
                        initial_noise=batch["next_initial_noise"],
                    )
            _synchronize(device)
            row_latency[variant].append(
                1000.0 * (time.perf_counter() - started) / batch["proprio"].shape[0]
            )
            row_predictions[variant] = (condition, action)
        for row_name, (condition, action) in row_predictions.items():
            action_errors = _action_errors(
                action,
                batch["next_teacher_action_chunk"],
                batch["execution_horizon"],
            )
            for metric, values in action_errors.items():
                row_values[row_name].setdefault(metric, []).extend(values.cpu().tolist())
            if condition is not None:
                condition_mse = F.mse_loss(
                    condition,
                    batch["next_full_condition"],
                    reduction="none",
                ).flatten(start_dim=1).mean(dim=1)
                condition_normalized = torch.stack(
                    [
                        normalized_condition_mse(condition[index:index + 1], batch["next_full_condition"][index:index + 1])
                        for index in range(condition.shape[0])
                    ]
                )
                cosine = F.cosine_similarity(
                    condition.flatten(start_dim=1),
                    batch["next_full_condition"].flatten(start_dim=1),
                    dim=1,
                )
                row_values[row_name].setdefault("condition_mse", []).extend(condition_mse.cpu().tolist())
                row_values[row_name].setdefault("condition_normalized_mse", []).extend(condition_normalized.cpu().tolist())
                row_values[row_name].setdefault("condition_cosine", []).extend(cosine.cpu().tolist())
            for index in range(action.shape[0]):
                per_record.append(
                    {
                        "row": row_name,
                        "task_id": int(batch["task_id"][index].item()),
                        "episode_id": batch["episode_id"][index],
                        "query_index": int(batch["query_index"][index].item()),
                        **{
                            metric: float(values[index].item())
                            for metric, values in action_errors.items()
                        },
                    }
                )
        processed_records += int(batch["proprio"].shape[0])
        elapsed = time.time() - started
        event = {
            "batch": batch_index,
            "batches": len(loader),
            "processed_records": processed_records,
            "heldout_records": len(holdout),
            "elapsed_seconds": elapsed,
            "records_per_second": processed_records / max(elapsed, 1e-9),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
            ),
        }
        if (
            batch_index == 1
            or batch_index == len(loader)
            or batch_index % args.progress_interval == 0
        ):
            _append_jsonl(progress_path, event)
        progress_bar.set_postfix(
            records=f"{processed_records}/{len(holdout)}",
            rps=f"{event['records_per_second']:.2f}",
        )
    progress_bar.close()
    elapsed_seconds = time.time() - started
    metrics: dict[str, Any] = {}
    for row_name, values in row_values.items():
        parameter_count = 0
        if row_name in adapters:
            parameter_count = sum(
                parameter.numel()
                for parameter in adapters[row_name][0].parameters()
                if parameter.requires_grad
            )
        metrics[row_name] = {
            "trainable_parameters": parameter_count,
            "latency_ms": distribution_summary(row_latency[row_name]),
            **{metric: distribution_summary(items) for metric, items in values.items()},
        }
    gate: dict[str, Any] = {
        "execution_horizon": args.execution_horizon,
        "chunk_aware_present": "chunk_aware_latentloop" in row_values,
        "old_observation_only_present": "old_observation_only" in row_values,
    }
    chunk = row_values.get("chunk_aware_latentloop", {}).get("prefix_l1", [])
    hold = row_values["hold_condition"].get("prefix_l1", [])
    old = row_values.get("old_observation_only", {}).get("prefix_l1", [])
    if chunk and hold and len(chunk) == len(hold):
        hold_minus_chunk = [h - c for h, c in zip(hold, chunk)]
        hold_ci = _paired_bootstrap_ci(hold_minus_chunk, args.split_seed)
        gate["hold_minus_chunk_prefix_l1_mean"] = sum(hold_minus_chunk) / len(hold_minus_chunk)
        gate["hold_minus_chunk_prefix_l1_ci95"] = hold_ci
        gate["offline_chunk_aware_beats_hold_prefix_error"] = bool(hold_ci[0] is not None and hold_ci[0] > 0)
    if chunk and old and len(chunk) == len(old):
        old_minus_chunk = [o - c for o, c in zip(old, chunk)]
        old_ci = _paired_bootstrap_ci(old_minus_chunk, args.split_seed + 1)
        gate["old_minus_chunk_prefix_l1_mean"] = sum(old_minus_chunk) / len(old_minus_chunk)
        gate["old_minus_chunk_prefix_l1_ci95"] = old_ci
        gate["offline_chunk_aware_beats_old_observation_only_prefix_error"] = bool(
            old_ci[0] is not None and old_ci[0] > 0
        )
    gate["OFFLINE_PREFIX_GATE_PASS"] = bool(
        gate.get("offline_chunk_aware_beats_hold_prefix_error", False)
        and gate.get("offline_chunk_aware_beats_old_observation_only_prefix_error", False)
    )
    with (output / "offline_episode_query_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(per_record[0]) if per_record else ["row"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_record)
    result = {
        "cache": str(Path(args.cache).resolve()),
        "heldout_records": len(holdout),
        "execution_horizon": args.execution_horizon,
        "teacher_same_noise_reload_max_abs_diff": max(teacher_reload_diffs, default=None),
        "rows": metrics,
        "gate": gate,
        "efficiency": {
            "elapsed_seconds": elapsed_seconds,
            "records_per_second": len(holdout) / max(elapsed_seconds, 1e-9),
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
            "peak_cuda_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
            ),
        },
    }
    _write_json(output / "offline_metrics.json", result)
    _write_json(output / "offline_gate.json", gate)
    return result


def main() -> int:
    """Parse held-out cache evaluation options."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument("--execution-horizon", type=int, choices=(1, 2, 5), required=True)
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--tqdm-mininterval", type=float, default=1.0)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
