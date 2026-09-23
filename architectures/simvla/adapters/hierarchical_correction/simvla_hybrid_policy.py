"""Official-queue SimVLA policy with three fixed hierarchical execution levels."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor

from architectures.simvla.adapters.latentloop.condition_adapter import (
    SimVLAChunkAwareAdapter,
)
from architectures.simvla.adapters.latentloop.query_cache_state import (
    SimVLAQueryObservation,
    tensor_hash,
)
from architectures.simvla.adapters.latentloop.simvla_policy import (
    RealSimVLALatentLoopPolicy,
)
from methods.hierarchical_correction.policy_state import HierarchicalPolicyState
from methods.hierarchical_correction.schedules import ExecutionLevel, HierarchicalSchedule
from methods.latentloop.modules.action_chunk_correction import shift_action_chunk

from .cache_state import SimVLAHybridCache


def hybrid_parameter_audit(
    condition_adapter: SimVLAChunkAwareAdapter,
    action_correction_adapter: SimVLAChunkAwareAdapter,
) -> dict[str, Any]:
    """Report the no-sharing parameter cost of composing existing checkpoints."""

    condition_count = sum(parameter.numel() for parameter in condition_adapter.parameters())
    correction_count = sum(
        parameter.numel() for parameter in action_correction_adapter.parameters()
    )
    return {
        "condition_adapter_parameters": condition_count,
        "action_correction_adapter_parameters": correction_count,
        "combined_adapter_parameters": condition_count + correction_count,
        "shared_parameters": 0,
        "separate_observation_encoders_preserved": True,
        "separate_executed_action_encoders_preserved": True,
        "official_simvla_base_parameters_excluded": True,
    }


class RealSimVLAHierarchicalCorrectionPolicy(RealSimVLALatentLoopPolicy):
    """Compose existing recurrent-condition and action-correction checkpoints."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        condition_adapter: SimVLAChunkAwareAdapter,
        action_correction_adapter: SimVLAChunkAwareAdapter,
        full_refresh_interval: int,
        action_regeneration_interval: int,
        execution_horizon: int,
        checkpoint_id: str,
        flow_steps: int,
        image_size: int,
        client_resize_size: int,
        device: torch.device,
        suite: str,
        row_name: str,
        task_id: int,
        episode_id: str,
        action_noise_seed_base: int,
        log_action_chunks: bool = False,
        teacher_tracking: bool = False,
    ) -> None:
        if condition_adapter.variant not in {
            "chunk_aware_latentloop",
            "nonrecurrent_condition",
        }:
            raise ValueError(
                "condition_adapter must be chunk_aware_latentloop or nonrecurrent_condition"
            )
        if action_correction_adapter.variant != "action_chunk_correction":
            raise ValueError("action_correction_adapter must be action_chunk_correction")
        self.hierarchical_schedule = HierarchicalSchedule(
            full_refresh_interval=full_refresh_interval,
            action_regeneration_interval=action_regeneration_interval,
            execution_horizon=execution_horizon,
        )
        self.condition_update_adapter = condition_adapter
        self.regeneration_mode = condition_adapter.variant
        self.action_correction_adapter = action_correction_adapter
        super().__init__(
            model=model,
            processor=processor,
            adapter=condition_adapter,
            mode="chunk_aware_latentloop",
            full_query_interval=full_refresh_interval,
            execution_horizon=execution_horizon,
            checkpoint_id=checkpoint_id,
            flow_steps=flow_steps,
            image_size=image_size,
            client_resize_size=client_resize_size,
            device=device,
            suite=suite,
            row_name=row_name,
            task_id=task_id,
            episode_id=episode_id,
            action_noise_seed_base=action_noise_seed_base,
            log_action_chunks=log_action_chunks,
            teacher_tracking=teacher_tracking,
        )

    def reset(self) -> None:
        super().reset()
        self.hybrid_cache = SimVLAHybridCache()
        self.hybrid_state = HierarchicalPolicyState(self.hierarchical_schedule)

    def _record_previous_execution(self) -> None:
        if not self.actions_sent_since_query:
            return
        if len(self.actions_sent_since_query) != self.execution_horizon:
            raise AssertionError(
                "policy query refilled before the complete executed subchunk was sent"
            )
        executed = torch.stack(self.actions_sent_since_query, dim=0).unsqueeze(0)
        self.hybrid_cache.record_executed_subchunk(executed)
        self.actions_sent_since_query = []

    def _features_for_adapter(
        self,
        adapter: SimVLAChunkAwareAdapter,
        *,
        prefix: str,
        previous_observation: SimVLAQueryObservation,
        current_observation: SimVLAQueryObservation,
        executed_actions: Tensor,
        elapsed_time: float,
    ) -> tuple[Tensor, Tensor]:
        self._synchronize_device()
        started = time.perf_counter()
        observation_feature = adapter.encode_observation(
            previous_observation.raw_rgb,
            current_observation.raw_rgb,
            previous_observation.proprio,
            current_observation.proprio,
        )
        self._synchronize_device()
        self.metrics.latencies.setdefault(f"{prefix}_observation_encoder_ms", []).append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_observation_encoder_calls"] += 1
        self.metrics.counters[f"num_{prefix}_observation_encoder_calls"] += 1

        self._synchronize_device()
        started = time.perf_counter()
        action_feature = adapter.encode_executed_actions(
            executed_actions,
            self.execution_horizon,
            elapsed_time,
            reference_feature=observation_feature,
        )
        self._synchronize_device()
        self.metrics.latencies.setdefault(f"{prefix}_executed_action_encoder_ms", []).append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_executed_action_encoder_calls"] += 1
        self.metrics.counters[f"num_{prefix}_executed_action_encoder_calls"] += 1
        return observation_feature, action_feature

    def _updated_condition(
        self,
        inputs: dict[str, Any],
        current_observation: SimVLAQueryObservation,
        *,
        query_age: int,
        elapsed_time: float,
    ) -> Tensor:
        observation_feature, action_feature = self._features_for_adapter(
            self.condition_update_adapter,
            prefix="condition",
            previous_observation=inputs["previous_query_observation"],
            current_observation=current_observation,
            executed_actions=inputs["executed_subchunk"],
            elapsed_time=elapsed_time,
        )
        self._synchronize_device()
        started = time.perf_counter()
        condition = self.condition_update_adapter.update_recurrent_condition(
            inputs["previous_condition"],
            observation_feature,
            action_feature,
            execution_horizon=self.execution_horizon,
            elapsed_time=elapsed_time,
            query_age=query_age,
        )
        self._synchronize_device()
        self.metrics.latencies.setdefault("condition_updater_ms", []).append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_condition_updater_calls"] += 1
        return condition

    def _nonrecurrent_condition(
        self,
        inputs: dict[str, Any],
        current_observation: SimVLAQueryObservation,
        *,
        query_age: int,
        elapsed_time: float,
    ) -> Tensor:
        """Regenerate from the full anchor and all actually executed subchunks."""

        adapter = self.condition_update_adapter
        self._synchronize_device()
        started = time.perf_counter()
        observation_feature = adapter.encode_observation(
            inputs["anchor_observation"].raw_rgb,
            current_observation.raw_rgb,
            inputs["anchor_observation"].proprio,
            current_observation.proprio,
        )
        self._synchronize_device()
        self.metrics.latencies.setdefault("condition_observation_encoder_ms", []).append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_observation_encoder_calls"] += 1
        self.metrics.counters["num_condition_observation_encoder_calls"] += 1

        action_features: list[Tensor] = []
        for subchunk in inputs["executed_subchunks_since_anchor"]:
            self._synchronize_device()
            started = time.perf_counter()
            action_features.append(
                adapter.encode_executed_actions(
                    subchunk,
                    self.execution_horizon,
                    elapsed_time,
                    reference_feature=observation_feature,
                )
            )
            self._synchronize_device()
            self.metrics.latencies.setdefault(
                "condition_executed_action_encoder_ms", []
            ).append((time.perf_counter() - started) * 1000.0)
            self.metrics.counters["num_executed_action_encoder_calls"] += 1
            self.metrics.counters["num_condition_executed_action_encoder_calls"] += 1
        if not action_features:
            raise RuntimeError("nonrecurrent regeneration requires executed action history")
        action_history = torch.stack(action_features, dim=0).mean(dim=0)
        self._synchronize_device()
        started = time.perf_counter()
        condition = adapter.predict_nonrecurrent_condition(
            inputs["anchor_condition"],
            observation_feature,
            action_history,
            execution_horizon=self.execution_horizon,
            elapsed_time=elapsed_time,
            query_age=query_age,
        )
        self._synchronize_device()
        self.metrics.latencies.setdefault("nonrecurrent_predictor_ms", []).append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_nonrecurrent_condition_calls"] += 1
        return condition

    def _correct_action_chunk(
        self,
        inputs: dict[str, Any],
        current_observation: SimVLAQueryObservation,
        *,
        query_age: int,
        elapsed_time: float,
    ) -> tuple[Tensor, dict[str, float]]:
        observation_feature, action_feature = self._features_for_adapter(
            self.action_correction_adapter,
            prefix="correction",
            previous_observation=inputs["previous_query_observation"],
            current_observation=current_observation,
            executed_actions=inputs["executed_subchunk"],
            elapsed_time=elapsed_time,
        )
        correction = self.action_correction_adapter.action_correction
        if correction is None:
            raise RuntimeError("action-correction checkpoint lacks its correction module")
        self._synchronize_device()
        started = time.perf_counter()
        output = correction(
            inputs["previous_action_chunk"],
            observation_feature,
            action_feature,
            execution_horizon=self.execution_horizon,
            elapsed_time=elapsed_time,
            query_age=query_age,
        )
        self._synchronize_device()
        self.metrics.latencies.setdefault("action_correction_ms", []).append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_action_correction_calls"] += 1
        residual = output.action_chunk - output.shifted.actions
        residual_metrics = {
            "all_l1": float(residual.abs().mean().item()),
            "arm_l1": float(output.arm_residual.abs().mean().item()),
            "gripper_l1": float(output.gripper_residual.abs().mean().item()),
            "shifted_valid_fraction": float(output.shifted.validity_mask.float().mean().item()),
        }
        return output.action_chunk, residual_metrics

    def _refill_action_queue(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        self._synchronize_device()
        query_started = time.perf_counter()
        self._record_previous_execution()
        self.metrics.counters["num_policy_queries"] += 1
        query_index = int(self.query_index)
        level = self.hierarchical_schedule.level(query_index)
        query_age = self.hierarchical_schedule.query_age(query_index)
        self.metrics.counters[f"num_level{int(level)}_queries"] += 1
        current_observation = SimVLAQueryObservation(
            raw_rgb=batch["raw_rgb"],
            proprio=batch["proprio"],
        )
        noise_seed: int | None = None
        correction_input_hash: str | None = None
        residual_metrics: dict[str, float] | None = None
        condition_drift_metrics: dict[str, float] | None = None
        cache_event: dict[str, Any]

        if level is ExecutionLevel.FULL_REFRESH:
            condition, action_chunk, noise_seed = self._full_refresh(
                batch,
                policy_query_index=query_index,
            )
            self.hybrid_cache.full_refresh(
                condition,
                action_chunk,
                current_observation,
                policy_query_index=query_index,
            )
            cache_event = self.hybrid_cache.trace[-1]
        else:
            inputs = self.hybrid_cache.lightweight_inputs(current_observation)
            elapsed_time = self.execution_horizon / 20.0
            if self.regeneration_mode == "nonrecurrent_condition":
                condition = (
                    self._nonrecurrent_condition(
                        inputs,
                        current_observation,
                        query_age=query_age,
                        elapsed_time=elapsed_time,
                    )
                    if level is ExecutionLevel.CONDITION_REGENERATION
                    else inputs["previous_condition"]
                )
            else:
                condition = self._updated_condition(
                    inputs,
                    current_observation,
                    query_age=query_age,
                    elapsed_time=elapsed_time,
                )
            condition_drift = condition - inputs["previous_condition"]
            condition_drift_metrics = {
                "l1": float(condition_drift.abs().mean().item()),
                "l2": float(
                    torch.linalg.vector_norm(condition_drift.flatten(start_dim=1), dim=1)
                    .mean()
                    .item()
                ),
                "mse": float(condition_drift.square().mean().item()),
            }
            if level is ExecutionLevel.CONDITION_REGENERATION:
                action_chunk, noise_seed = self._decode(
                    condition,
                    batch["proprio"],
                    policy_query_index=query_index,
                )
            else:
                correction_input_hash = str(inputs["previous_action_chunk_hash"])
                action_chunk, residual_metrics = self._correct_action_chunk(
                    inputs,
                    current_observation,
                    query_age=query_age,
                    elapsed_time=self.execution_horizon / 20.0,
                )
            cache_event = self.hybrid_cache.commit_lightweight(
                level=level,
                condition=condition,
                action_chunk=action_chunk,
                observation=current_observation,
                policy_query_index=query_index,
            )

        condition_hash = tensor_hash(condition)
        action_hash = tensor_hash(action_chunk)
        noise_hash = self.action_noise_hashes.get(query_index)
        record = self.hybrid_state.apply_query(
            policy_query_index=query_index,
            level=level,
            condition_cache_hash=condition_hash,
            action_chunk_cache_hash=action_hash,
            flow_noise_hash=noise_hash,
            action_correction_input_chunk_hash=correction_input_hash,
            action_correction_residual=residual_metrics,
            condition_cache_drift=condition_drift_metrics,
        )
        record.update(
            {
                "row_name": self.row_name,
                "full_refresh_interval": self.hierarchical_schedule.full_refresh_interval,
                "action_regeneration_interval": self.hierarchical_schedule.action_regeneration_interval,
                "execution_horizon": self.execution_horizon,
                "action_noise_seed": noise_seed,
                "action_chunk_shape": list(action_chunk.shape),
                "condition_cache_advanced": bool(
                    level is ExecutionLevel.FULL_REFRESH
                    or cache_event.get("condition_cache_advanced", False)
                ),
                "action_cache_replaced": bool(
                    level is ExecutionLevel.FULL_REFRESH
                    or cache_event.get("action_cache_replaced", False)
                ),
                "correction_consumes_latest_action_cache": (
                    None
                    if level is not ExecutionLevel.ACTION_CORRECTION
                    else correction_input_hash == record["previous_action_chunk_cache_hash"]
                ),
            }
        )
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action_chunk.detach()
        self.action_queue.clear()
        for action in action_chunk[0, : self.execution_horizon]:
            self.action_queue.append((action.detach(), level.source))
        self.latentloop_query_trace.append(record)
        if self.log_action_chunks:
            self.latentloop_action_chunks.append(
                {**record, "action_chunk": action_chunk.detach().cpu()}
            )
        self.metrics.latencies.setdefault("policy_query_total_ms", []).append(
            (time.perf_counter() - query_started) * 1000.0
        )
        if self.teacher_tracking and level is not ExecutionLevel.FULL_REFRESH:
            self._pending_teacher_tracking = (
                batch,
                condition,
                action_chunk,
                query_index,
                query_age,
                level.source,
                record,
            )
        self.query_index += 1
        return {
            "refreshed": level is ExecutionLevel.FULL_REFRESH,
            "age": query_age,
            "queue_mode": level.source,
            "execution_level": int(level),
        }


class RealSimVLAStaleActionChunkPolicy(RealSimVLALatentLoopPolicy):
    """Parameter-free control that shifts a cached chunk and zero-fills its tail."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(adapter=None, mode="hold_action_chunk", **kwargs)

    def _lightweight_action(
        self,
        batch: dict[str, Tensor],
        *,
        policy_query_index: int,
        query_age: int,
    ) -> tuple[Tensor, Tensor, int | None]:
        if self.cached_action_chunk is None:
            raise RuntimeError("stale action-chunk control requires a cached action chunk")
        current_observation = SimVLAQueryObservation(
            raw_rgb=batch["raw_rgb"],
            proprio=batch["proprio"],
        )
        inputs = self.query_cache.lightweight_transition_inputs(current_observation)
        shifted = shift_action_chunk(self.cached_action_chunk, self.execution_horizon)
        condition = inputs["previous_condition"]
        action_chunk = shifted.actions
        self.query_cache.commit_lightweight_update(condition, current_observation)
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action_chunk.detach()
        self.metrics.counters["num_stale_action_chunk_shifts"] += 1
        self.metrics.counters["num_stale_tail_tokens"] += int(
            (~shifted.validity_mask).sum().item()
        )
        return condition, action_chunk, None
