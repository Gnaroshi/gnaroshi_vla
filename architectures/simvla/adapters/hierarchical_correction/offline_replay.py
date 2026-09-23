"""Teacher-forced query-trace replay and endpoint parity for the fixed hybrid."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
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
from architectures.simvla.adapters.latentloop.query_cache_state import (  # noqa: E402
    SimVLAQueryObservation,
    tensor_hash,
)
from architectures.simvla.adapters.latentloop.source_lock import (  # noqa: E402
    collect_source_lock,
    require_empty_output,
    resolve_huggingface_checkpoint,
    sha256_file,
)
from architectures.simvla.adapters.hierarchical_correction.source_locked_loading import (  # noqa: E402
    load_source_locked_processor,
    load_source_locked_simvla,
)
from architectures.simvla.adapters.hierarchical_correction.simvla_hybrid_policy import (  # noqa: E402
    RealSimVLAHierarchicalCorrectionPolicy,
)
from architectures.simvla.adapters.latentloop.simvla_policy import (  # noqa: E402
    RealSimVLALatentLoopPolicy,
)
from methods.hierarchical_correction.schedules import (  # noqa: E402
    ExecutionLevel,
    HierarchicalSchedule,
)
from methods.hierarchical_correction.provenance import (  # noqa: E402
    experiment_source_signature,
    hierarchical_source_manifest,
)
from methods.latentloop.eval import distribution_summary  # noqa: E402
from methods.latentloop.training.query_cache_dataset import (  # noqa: E402
    iter_query_records,
    load_manifest,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _to_batch(value: Tensor, device: torch.device) -> Tensor:
    return value.unsqueeze(0).to(device)


def _features(
    adapter: Any,
    record: dict[str, Any],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    previous_rgb = _to_batch(record["raw_rgb"], device)
    current_rgb = _to_batch(record["next_raw_rgb"], device)
    previous_proprio = _to_batch(record["proprio"], device)
    current_proprio = _to_batch(record["next_proprio"], device)
    observation = adapter.encode_observation(
        previous_rgb,
        current_rgb,
        previous_proprio,
        current_proprio,
    )
    action = adapter.encode_executed_actions(
        _to_batch(record["executed_subchunk"], device),
        int(record["execution_horizon"]),
        float(record["elapsed_time"]),
        reference_feature=observation,
    )
    return observation, action


def _update_condition(
    adapter: Any,
    previous: Tensor,
    features: tuple[Tensor, Tensor],
    *,
    query_age: int,
    record: dict[str, Any],
) -> Tensor:
    return adapter.update_recurrent_condition(
        previous,
        features[0],
        features[1],
        execution_horizon=int(record["execution_horizon"]),
        elapsed_time=float(record["elapsed_time"]),
        query_age=query_age,
    )


def _correct_action(
    adapter: Any,
    previous: Tensor,
    features: tuple[Tensor, Tensor],
    *,
    query_age: int,
    record: dict[str, Any],
) -> tuple[Tensor, dict[str, float]]:
    module = adapter.action_correction
    if module is None:
        raise RuntimeError("action correction checkpoint lacks its correction module")
    output = module(
        previous,
        features[0],
        features[1],
        execution_horizon=int(record["execution_horizon"]),
        elapsed_time=float(record["elapsed_time"]),
        query_age=query_age,
    )
    residual = output.action_chunk - output.shifted.actions
    return output.action_chunk, {
        "all_l1": float(residual.abs().mean().item()),
        "arm_l1": float(output.arm_residual.abs().mean().item()),
        "gripper_l1": float(output.gripper_residual.abs().mean().item()),
    }


def _decode(
    action_adapter: SimVLAActionAdapter,
    condition: Tensor,
    record: dict[str, Any],
    device: torch.device,
    flow_steps: int,
) -> Tensor:
    return action_adapter.decode_action_from_condition(
        condition,
        _to_batch(record["next_proprio"], device),
        steps=flow_steps,
        initial_noise=_to_batch(record["next_initial_noise"], device),
    )


def _error(prediction: Tensor, teacher: Tensor) -> dict[str, float]:
    difference = prediction - teacher
    return {
        "chunk_l1": float(difference.abs().mean().item()),
        "first_action_l1": float(difference[:, 0].abs().mean().item()),
        "first_arm_l1": float(difference[:, 0, :6].abs().mean().item()),
        "first_gripper_abs": float(difference[:, 0, 6].abs().mean().item()),
        "first_gripper_match": float(
            ((prediction[:, 0, 6] >= 0) == (teacher[:, 0, 6] >= 0)).float().mean().item()
        ),
    }


def _initial_state(record: dict[str, Any], device: torch.device) -> dict[str, Any]:
    condition = _to_batch(record["full_condition"], device)
    action = _to_batch(record["teacher_action_chunk"], device)
    return {
        "expected_query": int(record["query_index"]),
        "hybrid_condition": condition.clone(),
        "hybrid_action": action.clone(),
        "pure_condition": condition.clone(),
        "full_anchor_condition": condition.clone(),
        "last_hybrid_regeneration": int(record["query_index"]),
        "last_hybrid_regeneration_level": int(ExecutionLevel.FULL_REFRESH),
        "last_regenerated_action_hash": tensor_hash(action),
    }


def _endpoint_policies(
    *,
    model: Any,
    processor: Any,
    condition_adapter: Any,
    correction_adapter: Any,
    args: argparse.Namespace,
    task_id: int,
    episode_id: str,
) -> dict[str, RealSimVLALatentLoopPolicy]:
    common = {
        "model": model,
        "processor": processor,
        "execution_horizon": 1,
        "checkpoint_id": args.checkpoint,
        "flow_steps": args.flow_steps,
        "image_size": args.image_size,
        "client_resize_size": args.client_resize_size,
        "device": torch.device(args.device),
        "suite": args.suite,
        "task_id": task_id,
        "episode_id": episode_id,
        "action_noise_seed_base": args.action_noise_seed_base,
        "log_action_chunks": False,
        "teacher_tracking": False,
    }
    return {
        "full_reference": RealSimVLALatentLoopPolicy(
            adapter=None,
            mode="full",
            full_query_interval=1,
            row_name="endpoint_full_reference",
            **common,
        ),
        "full_hybrid": RealSimVLAHierarchicalCorrectionPolicy(
            condition_adapter=condition_adapter,
            action_correction_adapter=correction_adapter,
            full_refresh_interval=1,
            action_regeneration_interval=1,
            row_name="endpoint_full_hybrid",
            **common,
        ),
        "condition_reference": RealSimVLALatentLoopPolicy(
            adapter=condition_adapter,
            mode="chunk_aware_latentloop",
            full_query_interval=4,
            row_name="endpoint_condition_reference",
            **common,
        ),
        "condition_hybrid": RealSimVLAHierarchicalCorrectionPolicy(
            condition_adapter=condition_adapter,
            action_correction_adapter=correction_adapter,
            full_refresh_interval=4,
            action_regeneration_interval=1,
            row_name="endpoint_condition_hybrid",
            **common,
        ),
        "action_reference": RealSimVLALatentLoopPolicy(
            adapter=correction_adapter,
            mode="action_chunk_correction",
            full_query_interval=4,
            row_name="endpoint_action_reference",
            **common,
        ),
        "action_hybrid": RealSimVLAHierarchicalCorrectionPolicy(
            condition_adapter=condition_adapter,
            action_correction_adapter=correction_adapter,
            full_refresh_interval=4,
            action_regeneration_interval=4,
            row_name="endpoint_action_hybrid",
            **common,
        ),
    }


def _anchor_endpoint_policy(
    policy: RealSimVLALatentLoopPolicy,
    *,
    condition: Tensor,
    action: Tensor,
    observation_rgb: Tensor,
    proprio: Tensor,
    query_index: int,
    flow_noise_hash: str,
) -> None:
    observation = SimVLAQueryObservation(observation_rgb, proprio)
    if isinstance(policy, RealSimVLAHierarchicalCorrectionPolicy):
        policy.hybrid_cache.full_refresh(
            condition,
            action,
            observation,
            policy_query_index=query_index,
        )
        policy.hybrid_state.apply_query(
            policy_query_index=query_index,
            level=ExecutionLevel.FULL_REFRESH,
            condition_cache_hash=tensor_hash(condition),
            action_chunk_cache_hash=tensor_hash(action),
            flow_noise_hash=flow_noise_hash,
        )
    else:
        policy.query_cache.full_refresh(condition, observation)
    policy.cached_condition = condition.detach()
    policy.cached_raw_rgb = observation_rgb.detach()
    policy.cached_proprio = proprio.detach()
    policy.cached_action_chunk = action.detach()
    policy.actions_sent_since_query = []
    policy.action_queue.clear()
    policy.query_index = query_index + 1


def _run_endpoint_lightweight_query(
    policy: RealSimVLALatentLoopPolicy,
    *,
    record: dict[str, Any],
    device: torch.device,
) -> None:
    policy.action_queue.clear()
    executed = record["executed_subchunk"].to(device)
    policy.actions_sent_since_query = [action.detach().clone() for action in executed]
    policy._refill_action_queue(
        {
            "raw_rgb": _to_batch(record["next_raw_rgb"], device),
            "proprio": _to_batch(record["next_proprio"], device),
        }
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Replay real query records without stepping LIBERO."""

    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    output = require_empty_output(args.output)
    cache = Path(args.cache).expanduser().resolve()
    manifest = load_manifest(cache)
    if int(manifest["execution_horizon"]) != 1:
        raise ValueError("hierarchical replay is locked to the R=1 cache")
    condition_path = Path(args.condition_checkpoint).expanduser().resolve()
    correction_path = Path(args.action_correction_checkpoint).expanduser().resolve()
    for path in (condition_path, correction_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_lock = collect_source_lock(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    source_lock["cache"] = {
        "path": str(cache),
        "manifest_sha256": sha256_file(cache / "manifest.json"),
        "schema_version": manifest["schema_version"],
        "execution_horizon": manifest["execution_horizon"],
    }
    source_lock["hierarchical_checkpoints"] = {
        "condition": {"path": str(condition_path), "sha256": sha256_file(condition_path)},
        "action_correction": {
            "path": str(correction_path),
            "sha256": sha256_file(correction_path),
        },
    }
    source_lock["processor_checkpoint"] = resolve_huggingface_checkpoint(
        args.smolvlm_model_path
    )
    source_lock["hierarchical_implementation"] = hierarchical_source_manifest(ROOT)
    source_signature = experiment_source_signature(source_lock)
    _write_json(output / "source_lock.json", source_lock)
    device = torch.device(args.device)
    model = load_source_locked_simvla(SmolVLMVLA, source_lock, device=device)
    model.action_space.load_norm_stats(args.norm_stats)
    freeze_module(model)
    processor = (
        load_source_locked_processor(SmolVLMVLAProcessor, source_lock)
        if args.mode == "parity"
        else None
    )
    action_adapter = SimVLAActionAdapter(model)
    condition_adapter, _ = load_adapter_checkpoint(condition_path, device=device)
    correction_adapter, _ = load_adapter_checkpoint(correction_path, device=device)
    freeze_module(condition_adapter)
    freeze_module(correction_adapter)
    if condition_adapter.variant != "chunk_aware_latentloop":
        raise ValueError("condition checkpoint has the wrong variant")
    if correction_adapter.variant != "action_chunk_correction":
        raise ValueError("action-correction checkpoint has the wrong variant")

    schedule = HierarchicalSchedule(4, 2, 1)
    k1_summary = _load_k1(args.k1_summary)
    k1_pass = bool(k1_summary.get("k1_parity", {}).get("K1_PARITY_PASS", False))
    prior_endpoint_parity = (
        _load_k1(args.endpoint_parity_summary) if args.mode == "gate" else None
    )
    if prior_endpoint_parity is not None and not prior_endpoint_parity.get(
        "ENDPOINT_PARITY_PASS", False
    ):
        raise RuntimeError("offline gate is blocked by endpoint parity")
    if prior_endpoint_parity is not None and prior_endpoint_parity.get(
        "source_signature"
    ) != source_signature:
        raise RuntimeError("offline gate source differs from endpoint parity source")
    trace_path = output / "offline_trace.jsonl"
    state: dict[str, Any] | None = None
    endpoint_policies: dict[str, RealSimVLALatentLoopPolicy] | None = None
    current_episode: tuple[int, str] | None = None
    episodes_seen: set[tuple[int, str]] = set()
    selected_episodes_by_task: dict[int, int] = {}
    selected_episode = False
    records_in_episode = 0
    records_processed = 0
    parity_records = 0
    k1_current_max = 0.0
    condition_endpoint_max = 0.0
    action_endpoint_max = 0.0
    diagnostics_action_max = 0.0
    diagnostics_rng_unchanged = True
    diagnostics_noise_hash_match = True
    teacher_reload_max = 0.0
    k1_current_comparisons = 0
    condition_endpoint_comparisons = 0
    action_endpoint_comparisons = 0
    finite_failures = 0
    reset_failures = 0
    reset_checks = 0
    level1_metrics: dict[str, list[float]] = {
        "condition_mse": [],
        "condition_cosine": [],
        "regenerated_chunk_l1": [],
        "regenerated_first_action_l1": [],
        "regenerated_first_gripper_match": [],
        "pure_condition_first_action_l1": [],
        "hold_first_action_l1": [],
        "pre_regeneration_first_action_l1": [],
        "post_minus_pre_first_action_l1": [],
    }
    correction_residuals: dict[int, list[float]] = {}
    condition_drifts: dict[int, list[float]] = {}
    hybrid_gripper_signs: list[bool] = []
    progress = tqdm(
        total=min(int(manifest["total_records"]), args.max_records) if args.max_records > 0 else int(manifest["total_records"]),
        desc=f"Hierarchical offline {args.mode}",
        mininterval=args.tqdm_mininterval,
    )
    started = time.perf_counter()
    with torch.no_grad():
        for record in iter_query_records(cache):
            key = (int(record["task_id"]), str(record["episode_id"]))
            if key != current_episode:
                if args.max_episodes > 0 and len(episodes_seen) >= args.max_episodes:
                    break
                current_episode = key
                selected_episode = (
                    args.episodes_per_task <= 0
                    or selected_episodes_by_task.get(key[0], 0) < args.episodes_per_task
                )
                if selected_episode:
                    selected_episodes_by_task[key[0]] = selected_episodes_by_task.get(key[0], 0) + 1
                    episodes_seen.add(key)
                    records_in_episode = 0
                    state = _initial_state(record, device)
                    if args.mode == "parity":
                        assert processor is not None
                        endpoint_policies = _endpoint_policies(
                            model=model,
                            processor=processor,
                            condition_adapter=condition_adapter,
                            correction_adapter=correction_adapter,
                            args=args,
                            task_id=key[0],
                            episode_id=key[1],
                        )
                        for policy in endpoint_policies.values():
                            _anchor_endpoint_policy(
                                policy,
                                condition=_to_batch(record["full_condition"], device),
                                action=_to_batch(record["teacher_action_chunk"], device),
                                observation_rgb=_to_batch(record["raw_rgb"], device),
                                proprio=_to_batch(record["proprio"], device),
                                query_index=int(record["query_index"]),
                                flow_noise_hash=str(record["action_noise_hash"]),
                            )
            if not selected_episode:
                continue
            assert state is not None
            query_index = int(record["query_index"])
            if query_index != state["expected_query"]:
                raise RuntimeError(f"non-contiguous episode {key}: expected {state['expected_query']}, got {query_index}")
            target_query = int(record["next_query_index"])
            target_level = schedule.level(target_query)
            target_age = schedule.query_age(target_query)
            teacher_condition = _to_batch(record["next_full_condition"], device)
            teacher_action = _to_batch(record["next_teacher_action_chunk"], device)
            if target_level is ExecutionLevel.FULL_REFRESH:
                state.update(
                    {
                        "hybrid_condition": teacher_condition.clone(),
                        "hybrid_action": teacher_action.clone(),
                        "pure_condition": teacher_condition.clone(),
                        "full_anchor_condition": teacher_condition.clone(),
                        "last_hybrid_regeneration": target_query,
                        "last_hybrid_regeneration_level": int(
                            ExecutionLevel.FULL_REFRESH
                        ),
                        "last_regenerated_action_hash": tensor_hash(teacher_action),
                    }
                )
                hybrid_action = teacher_action
                residual = None
                correction_input_hash = None
                counterfactual_correction_input_hash = None
                condition_drift = None
                if endpoint_policies is not None:
                    for policy in endpoint_policies.values():
                        _anchor_endpoint_policy(
                            policy,
                            condition=teacher_condition,
                            action=teacher_action,
                            observation_rgb=_to_batch(record["next_raw_rgb"], device),
                            proprio=_to_batch(record["next_proprio"], device),
                            query_index=target_query,
                            flow_noise_hash=str(record["next_action_noise_hash"]),
                        )
            else:
                previous_hybrid_condition = state["hybrid_condition"]
                condition_features = _features(condition_adapter, record, device)
                correction_features = _features(correction_adapter, record, device)
                updated_hybrid = _update_condition(
                    condition_adapter,
                    state["hybrid_condition"],
                    condition_features,
                    query_age=target_age,
                    record=record,
                )
                condition_difference_from_cache = updated_hybrid - previous_hybrid_condition
                condition_drift = {
                    "l1": float(condition_difference_from_cache.abs().mean().item()),
                    "l2": float(
                        torch.linalg.vector_norm(
                            condition_difference_from_cache.flatten(start_dim=1), dim=1
                        )
                        .mean()
                        .item()
                    ),
                    "mse": float(condition_difference_from_cache.square().mean().item()),
                }
                condition_drifts.setdefault(target_age, []).append(condition_drift["l1"])
                pure_condition = _update_condition(
                    condition_adapter,
                    state["pure_condition"],
                    condition_features,
                    query_age=target_age,
                    record=record,
                )
                pure_condition_action: Tensor | None = None
                if target_level is ExecutionLevel.CONDITION_REGENERATION or args.mode == "parity":
                    pure_condition_action = _decode(
                        action_adapter, pure_condition, record, device, args.flow_steps
                    )
                cached_action_hash = tensor_hash(state["hybrid_action"])
                if target_level is ExecutionLevel.CONDITION_REGENERATION:
                    correction_input_hash = None
                    counterfactual_correction_input_hash = cached_action_hash
                    pre_regeneration, _ = _correct_action(
                        correction_adapter,
                        state["hybrid_action"],
                        correction_features,
                        query_age=target_age,
                        record=record,
                    )
                    hybrid_action = _decode(
                        action_adapter, updated_hybrid, record, device, args.flow_steps
                    )
                    residual = None
                    state["last_hybrid_regeneration"] = target_query
                    state["last_hybrid_regeneration_level"] = int(
                        ExecutionLevel.CONDITION_REGENERATION
                    )
                    state["last_regenerated_action_hash"] = tensor_hash(hybrid_action)
                    condition_difference = updated_hybrid - teacher_condition
                    regenerated = _error(hybrid_action, teacher_action)
                    assert pure_condition_action is not None
                    pure_errors = _error(pure_condition_action, teacher_action)
                    hold_action = _decode(
                        action_adapter,
                        state["full_anchor_condition"],
                        record,
                        device,
                        args.flow_steps,
                    )
                    hold_errors = _error(hold_action, teacher_action)
                    pre_errors = _error(pre_regeneration, teacher_action)
                    level1_metrics["condition_mse"].append(float(condition_difference.square().mean().item()))
                    level1_metrics["condition_cosine"].append(
                        float(F.cosine_similarity(updated_hybrid.flatten(1), teacher_condition.flatten(1)).mean().item())
                    )
                    level1_metrics["regenerated_chunk_l1"].append(regenerated["chunk_l1"])
                    level1_metrics["regenerated_first_action_l1"].append(regenerated["first_action_l1"])
                    level1_metrics["regenerated_first_gripper_match"].append(regenerated["first_gripper_match"])
                    level1_metrics["pure_condition_first_action_l1"].append(pure_errors["first_action_l1"])
                    level1_metrics["hold_first_action_l1"].append(hold_errors["first_action_l1"])
                    level1_metrics["pre_regeneration_first_action_l1"].append(pre_errors["first_action_l1"])
                    level1_metrics["post_minus_pre_first_action_l1"].append(
                        regenerated["first_action_l1"] - pre_errors["first_action_l1"]
                    )
                else:
                    correction_input_hash = cached_action_hash
                    counterfactual_correction_input_hash = None
                    if (
                        target_query == int(state["last_hybrid_regeneration"]) + 1
                        and int(state["last_hybrid_regeneration_level"])
                        == int(ExecutionLevel.CONDITION_REGENERATION)
                    ):
                        reset_checks += 1
                        if correction_input_hash != state["last_regenerated_action_hash"]:
                            reset_failures += 1
                    hybrid_action, residual = _correct_action(
                        correction_adapter,
                        state["hybrid_action"],
                        correction_features,
                        query_age=target_age,
                        record=record,
                    )
                    correction_residuals.setdefault(target_age, []).append(residual["all_l1"])

                if args.mode == "parity":
                    assert endpoint_policies is not None
                    for policy in endpoint_policies.values():
                        _run_endpoint_lightweight_query(
                            policy,
                            record=record,
                            device=device,
                        )
                    full_reference = endpoint_policies["full_reference"]
                    full_hybrid = endpoint_policies["full_hybrid"]
                    condition_reference = endpoint_policies["condition_reference"]
                    condition_hybrid = endpoint_policies["condition_hybrid"]
                    action_reference = endpoint_policies["action_reference"]
                    action_hybrid = endpoint_policies["action_hybrid"]
                    assert full_reference.cached_condition is not None
                    assert full_hybrid.cached_condition is not None
                    assert full_reference.cached_action_chunk is not None
                    assert full_hybrid.cached_action_chunk is not None
                    assert condition_reference.cached_condition is not None
                    assert condition_hybrid.cached_condition is not None
                    assert condition_reference.cached_action_chunk is not None
                    assert condition_hybrid.cached_action_chunk is not None
                    assert action_reference.cached_action_chunk is not None
                    assert action_hybrid.cached_action_chunk is not None
                    k1_current_max = max(
                        k1_current_max,
                        float(
                            (
                                full_reference.cached_condition
                                - full_hybrid.cached_condition
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                        float(
                            (
                                full_reference.cached_action_chunk
                                - full_hybrid.cached_action_chunk
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                    )
                    condition_endpoint_max = max(
                        condition_endpoint_max,
                        float(
                            (
                                condition_reference.cached_condition
                                - condition_hybrid.cached_condition
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                        float(
                            (
                                condition_reference.cached_action_chunk
                                - condition_hybrid.cached_action_chunk
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                    )
                    action_endpoint_max = max(
                        action_endpoint_max,
                        float(
                            (
                                action_reference.cached_action_chunk
                                - action_hybrid.cached_action_chunk
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                    )
                    k1_current_comparisons += 1
                    condition_endpoint_comparisons += 1
                    action_endpoint_comparisons += 1

                    if parity_records < args.parity_records:
                        before_rng = torch.get_rng_state().clone()
                        before_cuda_rng = (
                            torch.cuda.get_rng_state(device).clone()
                            if device.type == "cuda"
                            else None
                        )
                        action_before = condition_hybrid.cached_action_chunk.clone()
                        diagnostic_batch = condition_hybrid.preprocess(
                            record["next_raw_rgb"][0].numpy(),
                            record["next_raw_rgb"][1].numpy(),
                            record["next_proprio"].numpy(),
                            str(record["language_instruction"]),
                        )
                        tracking = condition_hybrid._teacher_tracking_comparison(
                            diagnostic_batch,
                            condition=condition_hybrid.cached_condition,
                            action_chunk=condition_hybrid.cached_action_chunk,
                            policy_query_index=target_query,
                            query_age=target_age,
                            source="condition_regeneration_endpoint",
                        )
                        after_rng = torch.get_rng_state().clone()
                        after_cuda_rng = (
                            torch.cuda.get_rng_state(device).clone()
                            if device.type == "cuda"
                            else None
                        )
                        action_after = condition_hybrid.cached_action_chunk
                        diagnostics_action_max = max(
                            diagnostics_action_max,
                            float((action_before - action_after).abs().max().item()),
                        )
                        diagnostics_rng_unchanged = (
                            diagnostics_rng_unchanged
                            and torch.equal(before_rng, after_rng)
                        )
                        if before_cuda_rng is not None and after_cuda_rng is not None:
                            diagnostics_rng_unchanged = (
                                diagnostics_rng_unchanged
                                and torch.equal(before_cuda_rng, after_cuda_rng)
                            )
                        diagnostics_noise_hash_match = (
                            diagnostics_noise_hash_match
                            and tracking["action_noise_hash"]
                            == tensor_hash(
                                _to_batch(record["next_initial_noise"], device)
                            )
                        )
                        reload_action = _decode(
                            action_adapter,
                            teacher_condition,
                            record,
                            device,
                            args.flow_steps,
                        )
                        teacher_reload_max = max(
                            teacher_reload_max,
                            float((reload_action - teacher_action).abs().max().item()),
                        )
                        parity_records += 1
                state.update(
                    {
                        "hybrid_condition": updated_hybrid,
                        "hybrid_action": hybrid_action,
                        "pure_condition": pure_condition,
                    }
                )

            if not bool(torch.isfinite(state["hybrid_condition"]).all()) or not bool(
                torch.isfinite(hybrid_action).all()
            ):
                finite_failures += 1
            hybrid_gripper_signs.append(bool((hybrid_action[0, 0, 6] >= 0).item()))
            trace = {
                "task_id": key[0],
                "episode_id": key[1],
                "source_query_index": query_index,
                "policy_query_index": target_query,
                "execution_level": int(target_level),
                "query_age": target_age,
                "condition_hash": tensor_hash(state["hybrid_condition"]),
                "action_chunk_hash": tensor_hash(hybrid_action),
                "action_correction_input_chunk_hash": correction_input_hash,
                "counterfactual_correction_input_chunk_hash": counterfactual_correction_input_hash,
                "last_action_regeneration_query": state["last_hybrid_regeneration"],
                "action_correction_residual": residual,
                "condition_cache_drift": condition_drift,
                "teacher_forced_executed_subchunk": True,
                "flow_noise_hash": str(record["next_action_noise_hash"])
                if target_level is not ExecutionLevel.ACTION_CORRECTION
                else None,
            }
            _append_jsonl(trace_path, trace)
            state["expected_query"] = target_query
            records_processed += 1
            records_in_episode += 1
            progress.update(1)
            if args.max_records > 0 and records_processed >= args.max_records:
                break
            if (
                args.max_records_per_episode > 0
                and records_in_episode >= args.max_records_per_episode
            ):
                selected_episode = False
    progress.close()

    exact_schedule = [int(level) for level in schedule.levels(5)] == [2, 0, 1, 0, 2]
    if args.mode == "parity":
        current_k1_pass = k1_current_comparisons > 0 and k1_current_max == 0.0
        endpoint_parity = {
            "K1_PREEXISTING_PARITY_PASS": k1_pass,
            "K1_CURRENT_FULL_PATH_MAX_ABS_DIFF": k1_current_max,
            "K1_CURRENT_FULL_PATH_COMPARISONS": k1_current_comparisons,
            "K1_CURRENT_FULL_PATH_PARITY_PASS": current_k1_pass,
            "K1_FULL_PATH_PARITY_PASS": k1_pass and current_k1_pass,
            "KG1_CONDITION_ENDPOINT_MAX_ABS_DIFF": condition_endpoint_max,
            "KG1_CONDITION_ENDPOINT_COMPARISONS": condition_endpoint_comparisons,
            "KG1_CONDITION_ENDPOINT_PASS": condition_endpoint_comparisons > 0
            and condition_endpoint_max == 0.0,
            "KG_EQUALS_KF_ACTION_ENDPOINT_MAX_ABS_DIFF": action_endpoint_max,
            "KG_EQUALS_KF_ACTION_ENDPOINT_COMPARISONS": action_endpoint_comparisons,
            "KG_EQUALS_KF_ACTION_ENDPOINT_PASS": action_endpoint_comparisons > 0
            and action_endpoint_max == 0.0,
            "KF4_KG2_LEVEL_SEQUENCE": [int(level) for level in schedule.levels(5)],
            "KF4_KG2_SCHEDULE_PASS": exact_schedule,
            "DIAGNOSTICS_ACTION_MAX_ABS_DIFF": diagnostics_action_max,
            "DIAGNOSTICS_GLOBAL_RNG_UNCHANGED": diagnostics_rng_unchanged,
            "DIAGNOSTICS_FLOW_NOISE_HASH_MATCH": diagnostics_noise_hash_match,
            "DIAGNOSTICS_ON_OFF_PARITY_PASS": diagnostics_action_max == 0.0
            and diagnostics_rng_unchanged
            and diagnostics_noise_hash_match
            and parity_records > 0,
            "TEACHER_ACTION_RELOAD_MAX_ABS_DIFF": teacher_reload_max,
            "TEACHER_ACTION_RELOAD_PASS": teacher_reload_max == 0.0,
            "parity_records": parity_records,
            "source_signature": source_signature,
        }
        endpoint_parity["ENDPOINT_PARITY_PASS"] = all(
            endpoint_parity[name]
            for name in (
                "K1_FULL_PATH_PARITY_PASS",
                "KG1_CONDITION_ENDPOINT_PASS",
                "KG_EQUALS_KF_ACTION_ENDPOINT_PASS",
                "KF4_KG2_SCHEDULE_PASS",
                "DIAGNOSTICS_ON_OFF_PARITY_PASS",
                "TEACHER_ACTION_RELOAD_PASS",
            )
        )
    else:
        assert prior_endpoint_parity is not None
        endpoint_parity = prior_endpoint_parity
    _write_json(output / "endpoint_parity_summary.json", endpoint_parity)
    level1_summary = {
        name: distribution_summary(values) for name, values in level1_metrics.items()
    }
    no_worse = all(
        hybrid <= pure + args.numeric_tolerance
        for hybrid, pure in zip(
            level1_metrics["regenerated_first_action_l1"],
            level1_metrics["pure_condition_first_action_l1"],
        )
    ) and bool(level1_metrics["regenerated_first_action_l1"])
    gripper_noncollapsed = bool(hybrid_gripper_signs) and any(hybrid_gripper_signs) and not all(
        hybrid_gripper_signs
    )
    offline = {
        "replay_scope": "teacher-forced query-boundary replay; no environment execution",
        "records_processed": records_processed,
        "episodes_processed": len(episodes_seen),
        "elapsed_seconds": time.perf_counter() - started,
        "level1_age2": level1_summary,
        "correction_residual_by_age": {
            str(age): distribution_summary(values) for age, values in sorted(correction_residuals.items())
        },
        "condition_cache_drift_l1_by_age": {
            str(age): distribution_summary(values) for age, values in sorted(condition_drifts.items())
        },
        "finite_failures": finite_failures,
        "hybrid_gripper_positive_fraction": sum(hybrid_gripper_signs) / max(len(hybrid_gripper_signs), 1),
        "hybrid_gripper_noncollapsed": gripper_noncollapsed,
        "regenerated_first_action_no_worse_than_pure_condition": no_worse,
        "regeneration_recovers_condition_path": (
            bool(level1_metrics["post_minus_pre_first_action_l1"])
            and sum(level1_metrics["post_minus_pre_first_action_l1"])
            / len(level1_metrics["post_minus_pre_first_action_l1"])
            <= 0.0
        ),
        "action_correction_reset_failures": reset_failures,
        "action_correction_reset_checks": reset_checks,
        "action_correction_resets_after_regeneration": reset_checks > 0
        and reset_failures == 0,
        "endpoint_parity": endpoint_parity,
        "source_signature": source_signature,
    }
    offline["ONLINE_EVALUATION_GATE_PASS"] = bool(
        endpoint_parity["ENDPOINT_PARITY_PASS"]
        and finite_failures == 0
        and gripper_noncollapsed
        and no_worse
        and offline["action_correction_resets_after_regeneration"]
    )
    _write_json(output / "offline_replay_summary.json", offline)
    return endpoint_parity if args.mode == "parity" else offline


def _load_k1(path: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"K1 parity summary not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parity", "gate"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--k1-summary", required=True)
    parser.add_argument("--endpoint-parity-summary", default="")
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument(
        "--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct"
    )
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--client-resize-size", type=int, default=224)
    parser.add_argument("--action-noise-seed-base", type=int, default=20260804)
    parser.add_argument("--condition-checkpoint", required=True)
    parser.add_argument("--action-correction-checkpoint", required=True)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-records-per-episode", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--episodes-per-task", type=int, default=0)
    parser.add_argument("--parity-records", type=int, default=16)
    parser.add_argument("--numeric-tolerance", type=float, default=1e-7)
    parser.add_argument("--tqdm-mininterval", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.mode == "parity" and args.max_records <= 0:
        raise ValueError("parity mode requires a positive --max-records")
    if args.mode == "gate" and not args.endpoint_parity_summary:
        raise ValueError("gate mode requires --endpoint-parity-summary")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
