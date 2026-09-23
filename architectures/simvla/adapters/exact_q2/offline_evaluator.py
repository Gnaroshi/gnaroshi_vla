"""Paired validation and scientific gate inputs for exact-q2 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
for path in (ROOT, UPSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter  # noqa: E402
from architectures.simvla.adapters.exact_q2.simvla_exact_q2_adapter import (  # noqa: E402
    freeze_module,
    load_exact_q2_checkpoint,
)
from architectures.simvla.adapters.hierarchical_correction.source_locked_loading import (  # noqa: E402
    load_source_locked_simvla,
)
from architectures.simvla.adapters.latentloop.checkpoint import load_adapter_checkpoint  # noqa: E402
from architectures.simvla.adapters.latentloop.source_lock import (  # noqa: E402
    collect_source_lock,
    require_empty_output,
    resolve_huggingface_checkpoint,
    sha256_file,
)
from methods.latentloop.training.losses import normalized_condition_mse  # noqa: E402
from methods.simvla_exact_q2.dataset import (  # noqa: E402
    ExactQ2TupleDataset,
    collate_exact_q2,
)
from methods.simvla_exact_q2.losses import per_example_prefix_l1  # noqa: E402
from methods.simvla_exact_q2.validation import (  # noqa: E402
    distribution,
    paired_bootstrap_ci95,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _source_signature(lock: dict[str, Any]) -> dict[str, Any]:
    checkpoint = lock["checkpoint"]
    return {
        "simvla_upstream_commit": lock["simvla_upstream_commit"],
        "checkpoint_identifier": checkpoint["identifier"],
        "checkpoint_revision": checkpoint["revision"],
        "checkpoint_blob_sha256": checkpoint["hf_blob_key_sha256"],
        "norm_stats_sha256": lock["norm_stats_sha256"],
    }


def _require_source(expected: dict[str, Any], lock: dict[str, Any], label: str) -> None:
    actual = _source_signature(lock)
    fields = (
        "simvla_upstream_commit",
        "checkpoint_identifier",
        "checkpoint_revision",
        "checkpoint_blob_sha256",
        "norm_stats_sha256",
    )
    mismatch = {name: (expected.get(name), actual.get(name)) for name in fields if expected.get(name) != actual.get(name)}
    if mismatch:
        raise RuntimeError(f"{label} source lock differs from exact-q2 cache: {mismatch}")


def _transition_features(
    adapter: Any,
    previous_rgb: Tensor,
    current_rgb: Tensor,
    previous_proprio: Tensor,
    current_proprio: Tensor,
    executed: Tensor,
    elapsed: Tensor,
) -> tuple[Tensor, Tensor]:
    observation = adapter.encode_observation(
        previous_rgb, current_rgb, previous_proprio, current_proprio
    )
    action = adapter.encode_executed_actions(
        executed,
        5,
        elapsed,
        reference_feature=observation,
    )
    return observation, action


def _legacy_recurrent_q1_q2(adapter: Any, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
    obs1, action1 = _transition_features(
        adapter,
        batch["q0_raw_rgb"],
        batch["q1_raw_rgb"],
        batch["q0_proprio"],
        batch["q1_proprio"],
        batch["x0_executed"],
        batch["elapsed_q0_to_q1"],
    )
    c1 = adapter.update_recurrent_condition(
        batch["c0_full"],
        obs1,
        action1,
        execution_horizon=5,
        elapsed_time=batch["elapsed_q0_to_q1"],
        query_age=1,
    )
    obs2, action2 = _transition_features(
        adapter,
        batch["q1_raw_rgb"],
        batch["q2_raw_rgb"],
        batch["q1_proprio"],
        batch["q2_proprio"],
        batch["x1_executed"],
        batch["elapsed_q1_to_q2"],
    )
    c2 = adapter.update_recurrent_condition(
        c1,
        obs2,
        action2,
        execution_horizon=5,
        elapsed_time=batch["elapsed_q1_to_q2"],
        query_age=2,
    )
    return c1, c2


def _legacy_nonrecurrent_q2(adapter: Any, batch: dict[str, Any]) -> Tensor:
    observation = adapter.encode_observation(
        batch["q0_raw_rgb"],
        batch["q2_raw_rgb"],
        batch["q0_proprio"],
        batch["q2_proprio"],
    )
    action_features = []
    for executed, elapsed in (
        (batch["x0_executed"], batch["elapsed_q0_to_q1"]),
        (batch["x1_executed"], batch["elapsed_q1_to_q2"]),
    ):
        action_features.append(
            adapter.encode_executed_actions(
                executed, 5, elapsed, reference_feature=observation
            )
        )
    return adapter.predict_nonrecurrent_condition(
        batch["c0_full"],
        observation,
        torch.stack(action_features).mean(dim=0),
        execution_horizon=5,
        elapsed_time=batch["elapsed_q0_to_q1"] + batch["elapsed_q1_to_q2"],
        query_age=2,
    )


def _condition_per_example(prediction: Tensor, target: Tensor) -> Tensor:
    pred = F.layer_norm(prediction, (prediction.shape[-1],))
    full = F.layer_norm(target.detach(), (target.shape[-1],))
    return (pred - full).square().mean(dim=(1, 2))


def _action_metrics(
    prediction: Tensor,
    target: Tensor,
    *,
    execution_horizon: int,
) -> dict[str, Tensor]:
    prefix = prediction[:, :execution_horizon] - target[:, :execution_horizon].detach()
    return {
        "q2_prefix_l1": prefix.abs().mean(dim=(1, 2)),
        "q2_chunk_l1": (prediction - target.detach()).abs().mean(dim=(1, 2)),
        "q2_gripper_command_l1": prefix[..., 6].abs().mean(dim=1),
    }


def _append(values: dict[str, dict[str, list[float]]], row: str, metrics: dict[str, Tensor]) -> None:
    for name, tensor in metrics.items():
        values[row][name].extend(float(value) for value in tensor.detach().cpu().flatten())


def _load_legacy(path: str, expected_variant: str, device: torch.device) -> tuple[Any, dict]:
    adapter, payload = load_adapter_checkpoint(path, device=device)
    if adapter.variant != expected_variant:
        raise ValueError(f"{path} contains {adapter.variant}, expected {expected_variant}")
    freeze_module(adapter)
    return adapter, payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    from models.modeling_smolvlm_vla import SmolVLMVLA

    output = require_empty_output(args.output)
    device = torch.device(args.device)
    dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    dataset = ExactQ2TupleDataset(dataset_manifest, split, partition="validation")
    if args.max_examples > 0:
        dataset.rows = dataset.rows[: args.max_examples]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_exact_q2,
        pin_memory=device.type == "cuda",
    )
    candidate, payload = load_exact_q2_checkpoint(args.candidate_checkpoint, device=device)
    if candidate.candidate != args.candidate:
        raise ValueError("checkpoint candidate differs from --candidate")
    freeze_module(candidate)
    step = int(payload["step"])
    cache_source = dataset_manifest["source_signature"]
    _require_source(cache_source, payload["metadata"]["source_lock"], "candidate checkpoint")

    runtime_lock = collect_source_lock(checkpoint=args.checkpoint, norm_stats_path=args.norm_stats)
    runtime_lock["processor_checkpoint"] = resolve_huggingface_checkpoint(
        "HuggingFaceTB/SmolVLM-500M-Instruct"
    )
    _require_source(cache_source, runtime_lock, "evaluation runtime")
    _write_json(output / "source_lock.json", runtime_lock)
    model = load_source_locked_simvla(SmolVLMVLA, runtime_lock, device=device)
    model.action_space.load_norm_stats(args.norm_stats)
    freeze_module(model)
    action_adapter = SimVLAActionAdapter(model)

    old, old_payload = _load_legacy(args.old_observation_checkpoint, "old_observation_only", device)
    _require_source(cache_source, old_payload["metadata"]["source_lock"], "old observation reference")
    historical: dict[str, tuple[Any, dict]] = {}
    if args.historical_recurrent_checkpoint:
        historical["historical_age1_recurrent"] = _load_legacy(
            args.historical_recurrent_checkpoint, "chunk_aware_latentloop", device
        )
    if args.historical_nonrecurrent_checkpoint:
        historical["historical_age1_nonrecurrent"] = _load_legacy(
            args.historical_nonrecurrent_checkpoint, "nonrecurrent_condition", device
        )
    for name, (_, historical_payload) in historical.items():
        _require_source(cache_source, historical_payload["metadata"]["source_lock"], name)

    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    gripper_values: dict[str, list[float]] = defaultdict(list)
    example_rows: list[dict[str, Any]] = []
    teacher_reload_max = 0.0
    started = time.time()
    progress = tqdm(loader, desc=f"exact-q2 validation/{args.candidate}", dynamic_ncols=True)
    with torch.no_grad():
        for batch in progress:
            batch = _to_device(batch, device)
            prediction = candidate(batch)
            a1_pred = action_adapter.decode_action_from_condition(
                prediction.c1,
                batch["q1_proprio"],
                steps=args.flow_steps,
                initial_noise=batch["epsilon1"],
            )
            a2_pred = action_adapter.decode_action_from_condition(
                prediction.c2,
                batch["q2_proprio"],
                steps=args.flow_steps,
                initial_noise=batch["epsilon2"],
            )
            q1_metrics = {
                "q1_prefix_l1": per_example_prefix_l1(a1_pred, batch["a1_full"], 5),
                "q1_condition_normalized_mse": _condition_per_example(
                    prediction.c1, batch["c1_full"]
                ),
            }
            candidate_metrics = {
                **_action_metrics(a2_pred, batch["a2_full"], execution_horizon=5),
                "q2_condition_normalized_mse": _condition_per_example(
                    prediction.c2, batch["c2_full"]
                ),
                **q1_metrics,
            }
            _append(values, args.candidate, candidate_metrics)
            gripper_values[args.candidate].extend(
                float(value) for value in a2_pred[:, :5, 6].detach().cpu().flatten()
            )

            hold_action = action_adapter.decode_action_from_condition(
                batch["c0_full"],
                batch["q2_proprio"],
                steps=args.flow_steps,
                initial_noise=batch["epsilon2"],
            )
            _append(
                values,
                "hold_stale_condition",
                {
                    **_action_metrics(hold_action, batch["a2_full"], execution_horizon=5),
                    "q2_condition_normalized_mse": _condition_per_example(
                        batch["c0_full"], batch["c2_full"]
                    ),
                },
            )

            _, old_c2 = _legacy_recurrent_q1_q2(old, batch)
            old_action = action_adapter.decode_action_from_condition(
                old_c2,
                batch["q2_proprio"],
                steps=args.flow_steps,
                initial_noise=batch["epsilon2"],
            )
            _append(
                values,
                "old_observation_only",
                {
                    **_action_metrics(old_action, batch["a2_full"], execution_horizon=5),
                    "q2_condition_normalized_mse": _condition_per_example(old_c2, batch["c2_full"]),
                },
            )

            for name, (legacy, _) in historical.items():
                if name == "historical_age1_recurrent":
                    _, condition = _legacy_recurrent_q1_q2(legacy, batch)
                else:
                    condition = _legacy_nonrecurrent_q2(legacy, batch)
                action = action_adapter.decode_action_from_condition(
                    condition,
                    batch["q2_proprio"],
                    steps=args.flow_steps,
                    initial_noise=batch["epsilon2"],
                )
                _append(
                    values,
                    name,
                    {
                        **_action_metrics(action, batch["a2_full"], execution_horizon=5),
                        "q2_condition_normalized_mse": _condition_per_example(
                            condition, batch["c2_full"]
                        ),
                    },
                )

            teacher_reload = action_adapter.decode_action_from_condition(
                batch["c2_full"],
                batch["q2_proprio"],
                steps=args.flow_steps,
                initial_noise=batch["epsilon2"],
            )
            teacher_reload_max = max(
                teacher_reload_max,
                float((teacher_reload - batch["a2_full"]).abs().max().item()),
            )
            batch_size = a2_pred.shape[0]
            for index in range(batch_size):
                example_rows.append(
                    {
                        "tuple_id": batch["tuple_id"][index],
                        "task_id": int(batch["task_id"][index].item()),
                        "episode_id": batch["episode_id"][index],
                        "q0_query_index": int(batch["q0_query_index"][index].item()),
                        "q2_query_index": int(batch["q2_query_index"][index].item()),
                        "epsilon2_sha256": batch["epsilon2_sha256"][index],
                        "candidate_q1_prefix_l1": float(q1_metrics["q1_prefix_l1"][index].item()),
                        "candidate_q2_prefix_l1": float(candidate_metrics["q2_prefix_l1"][index].item()),
                        "hold_q2_prefix_l1": float(values["hold_stale_condition"]["q2_prefix_l1"][-batch_size + index]),
                        "old_observation_q2_prefix_l1": float(values["old_observation_only"]["q2_prefix_l1"][-batch_size + index]),
                    }
                )

    same_noise = teacher_reload_max <= args.teacher_reload_tolerance
    candidate_prefix = values[args.candidate]["q2_prefix_l1"]
    hold_prefix = values["hold_stale_condition"]["q2_prefix_l1"]
    old_prefix = values["old_observation_only"]["q2_prefix_l1"]
    predicted_gripper = gripper_values[args.candidate]
    signs = {value >= 0.0 for value in predicted_gripper}
    finite = all(
        torch.isfinite(torch.tensor(samples)).all().item()
        for metrics in values.values()
        for samples in metrics.values()
    )
    summarized = {
        row: {metric: distribution(samples) for metric, samples in metrics.items()}
        for row, metrics in values.items()
    }
    candidate_summary = {
        **summarized[args.candidate],
        "finite": bool(finite),
        "gripper_noncollapsed": len(signs) == 2 and len(set(predicted_gripper)) > 1,
    }
    selected_by_validation = False
    selection_payload = None
    if args.selection_json:
        selection_payload = json.loads(Path(args.selection_json).read_text(encoding="utf-8"))
        selected_by_validation = (
            selection_payload.get("selection_data") == "validation_only"
            and Path(selection_payload["selected_checkpoint"]).resolve()
            == Path(args.candidate_checkpoint).resolve()
        )
    result = {
        "schema_version": "simvla_exact_q2_validation_v1",
        "partition": "validation",
        "candidate": args.candidate,
        "checkpoint": str(Path(args.candidate_checkpoint).resolve()),
        "step": step,
        "examples": len(candidate_prefix),
        "metrics": candidate_summary,
        "references": {
            "hold_stale_condition": summarized["hold_stale_condition"],
            "old_observation_only": summarized["old_observation_only"],
            **{name: summarized[name] for name in historical},
        },
        "paired_ci95": {
            "candidate_minus_hold": paired_bootstrap_ci95(
                (candidate_value - reference for candidate_value, reference in zip(candidate_prefix, hold_prefix)),
                seed=args.bootstrap_seed,
                samples=args.bootstrap_samples,
            ),
            "candidate_minus_old_observation": paired_bootstrap_ci95(
                (candidate_value - reference for candidate_value, reference in zip(candidate_prefix, old_prefix)),
                seed=args.bootstrap_seed + 1,
                samples=args.bootstrap_samples,
            ),
        },
        "prerequisites": {
            "cache_integrity": True,
            "same_noise": same_noise,
            "selected_by_validation_only": selected_by_validation,
        },
        "teacher_same_noise_reload_max_abs_diff": teacher_reload_max,
        "selection": selection_payload,
        "gripper_representation": {
            "type": "continuous_postprocessed_environment_command",
            "binary_threshold_in_source": False,
            "metric": "continuous command L1 plus sign/switch semantics",
        },
        "elapsed_seconds": time.time() - started,
    }
    validation_path = output / f"validation_step_{step:06d}.json"
    _write_json(validation_path, result)
    reference_manifest = {
        "schema_version": "simvla_exact_q2_reference_manifest_v1",
        "cache_manifest_sha256": dataset_manifest["cache_manifest_sha256"],
        "same_validation_tuple_ids": split["validation_tuple_ids"],
        "same_noise": True,
        "rows": {
            "full_teacher_reference": "C2_full,q2_proprio,epsilon2",
            "hold_stale_condition": "C0_full,q2_proprio,epsilon2",
            "old_observation_only": str(Path(args.old_observation_checkpoint).resolve()),
            **{name: str(Path(path).resolve()) for name, path in (
                ("historical_age1_recurrent", args.historical_recurrent_checkpoint),
                ("historical_age1_nonrecurrent", args.historical_nonrecurrent_checkpoint),
            ) if path},
        },
    }
    _write_json(output / "r5_exact_q2_reference_manifest.json", reference_manifest)
    with (output / "r5_exact_q2_offline_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row", "q2_prefix_mean", "q2_prefix_p95", "q2_prefix_p99", "q2_chunk_mean", "q2_condition_mean"],
        )
        writer.writeheader()
        for name, metrics in summarized.items():
            writer.writerow(
                {
                    "row": name,
                    "q2_prefix_mean": metrics["q2_prefix_l1"]["mean"],
                    "q2_prefix_p95": metrics["q2_prefix_l1"]["p95"],
                    "q2_prefix_p99": metrics["q2_prefix_l1"]["p99"],
                    "q2_chunk_mean": metrics["q2_chunk_l1"]["mean"],
                    "q2_condition_mean": metrics["q2_condition_normalized_mse"]["mean"],
                }
            )
    with (output / "paired_validation_examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(example_rows[0]))
        writer.writeheader()
        writer.writerows(example_rows)
    print(json.dumps({"validation": str(validation_path), "examples": len(candidate_prefix)}, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", required=True, choices=("recurrent_exact_q2", "direct_exact_q2"))
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--selection-json", default="")
    parser.add_argument("--old-observation-checkpoint", required=True)
    parser.add_argument("--historical-recurrent-checkpoint", default="")
    parser.add_argument("--historical-nonrecurrent-checkpoint", default="")
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260815)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--teacher-reload-tolerance", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = run(args)
    return 0 if result["prerequisites"]["same_noise"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
