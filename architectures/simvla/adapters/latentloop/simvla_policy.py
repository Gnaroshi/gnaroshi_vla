"""State and cache semantics for an official-queue SimVLA LatentLoop policy."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    EvalStepOutput,
    RealSimVLADCLDPolicy,
)

from .action_adapter import ActionNoiseKey, explicit_action_noise
from .condition_adapter import SimVLAChunkAwareAdapter
from .query_cache_state import (
    RecursiveQueryCache,
    SimVLAQueryObservation,
    tensor_hash as _tensor_hash,
)


class RealSimVLALatentLoopPolicy(RealSimVLADCLDPolicy):
    """Official-queue policy that changes only the condition/action correction branch."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        adapter: SimVLAChunkAwareAdapter | None,
        mode: str,
        full_query_interval: int,
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
        self.latentloop_adapter = adapter
        self.latentloop_mode = str(mode)
        self.checkpoint_id = str(checkpoint_id)
        self.episode_id = str(episode_id)
        self.execution_horizon = int(execution_horizon)
        self.full_query_interval = int(full_query_interval)
        self.teacher_tracking = bool(teacher_tracking)
        if self.execution_horizon not in {1, 2, 5}:
            raise ValueError("execution_horizon must be 1, 2, or 5")
        if self.full_query_interval < 1:
            raise ValueError("full_query_interval must be positive")
        super().__init__(
            model=model,
            processor=processor,
            dcld_core=None,
            mode="full",
            refresh_every=self.full_query_interval,
            flow_steps=flow_steps,
            image_size=image_size,
            replan_steps=self.execution_horizon,
            client_resize_size=client_resize_size,
            device=device,
            suite=suite,
            row_name=row_name,
            task_id=task_id,
            trial_id=0,
            paired_action_noise=True,
            action_noise_seed_base=action_noise_seed_base,
            log_action_chunks=log_action_chunks,
        )

    def reset(self) -> None:
        """Reset inherited action queue plus all LatentLoop episode caches."""

        super().reset()
        self.query_cache = RecursiveQueryCache()
        self.actions_sent_since_query: list[Tensor] = []
        self.latentloop_query_trace: list[dict[str, Any]] = []
        self.latentloop_action_chunks: list[dict[str, Any]] = []
        self.latentloop_tracking_trace: list[dict[str, Any]] = []
        self.action_noise_hashes: dict[int, str] = {}
        self._pending_teacher_tracking: tuple[dict[str, Tensor], Tensor, Tensor, int, int, str, dict[str, Any]] | None = None

    def _action_noise_key(self, policy_query_index: int) -> ActionNoiseKey:
        """Return the row/K/R-independent key for one policy query."""

        return ActionNoiseKey(
            checkpoint=self.checkpoint_id,
            task_id=self.task_id,
            episode_id=self.episode_id,
            policy_query_index=policy_query_index,
            seed_base=self.action_noise_seed_base,
        )

    def _synchronize_device(self) -> None:
        """Synchronize CUDA only at explicit latency boundaries."""

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _decode(
        self,
        condition: Tensor,
        proprio: Tensor,
        *,
        policy_query_index: int,
    ) -> tuple[Tensor, int | None]:
        self._synchronize_device()
        started = time.perf_counter()
        noise_key = self._action_noise_key(policy_query_index)
        initial_noise = explicit_action_noise(
            noise_key,
            batch_size=condition.shape[0],
            action_horizon=self.action_adapter.num_actions,
            action_dim=self.action_adapter.dim_action,
            device=proprio.device,
            dtype=proprio.dtype,
        )
        decoded = self.action_adapter.decode_action_from_condition(
            condition,
            proprio,
            steps=self.flow_steps,
            initial_noise=initial_noise,
            deterministic=False,
            return_debug=True,
        )
        self._synchronize_device()
        self.metrics.latencies["action_transformer_ms"].append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_action_transformer_calls"] += int(
            decoded.debug.get("iterations", 0)
        )
        self.metrics.counters["num_action_transformer_decodes"] += 1
        self.action_noise_hashes[policy_query_index] = _tensor_hash(decoded.initial_noise)
        return decoded.action, noise_key.seed()

    def _full_refresh(
        self,
        batch: dict[str, Tensor],
        *,
        policy_query_index: int,
    ) -> tuple[Tensor, Tensor, int | None]:
        """Run the unchanged full condition path with synchronized latency."""

        self._synchronize_device()
        started = time.perf_counter()
        condition = self.condition_adapter.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
        self._synchronize_device()
        self.metrics.latencies["VLM_encoder_ms"].append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_full_vlm_calls"] += 1
        action_chunk, seed = self._decode(
            condition,
            batch["proprio"],
            policy_query_index=policy_query_index,
        )
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action_chunk.detach()
        return condition, action_chunk, seed

    def _record_previous_execution(self) -> None:
        if not self.actions_sent_since_query:
            return
        if len(self.actions_sent_since_query) != self.execution_horizon:
            raise AssertionError(
                "policy query refilled before the complete executed subchunk was sent"
            )
        executed = torch.stack(self.actions_sent_since_query, dim=0).unsqueeze(0)
        self.query_cache.record_executed_subchunk(executed)
        self.actions_sent_since_query = []

    def _adapter_features(
        self,
        *,
        previous_observation: SimVLAQueryObservation,
        current_observation: SimVLAQueryObservation,
        executed_actions: Tensor,
        elapsed_time: float,
    ) -> tuple[Tensor, Tensor]:
        if self.latentloop_adapter is None:
            raise RuntimeError(f"mode {self.latentloop_mode} requires an adapter checkpoint")
        batch_size = current_observation.proprio.shape[0]
        if self.latentloop_mode == "no_observation":
            observation_feature = current_observation.proprio.new_zeros(
                (batch_size, self.latentloop_adapter.config.observation_dim)
            )
        else:
            self._synchronize_device()
            t0 = time.perf_counter()
            observation_feature = self.latentloop_adapter.encode_observation(
                previous_observation.raw_rgb,
                current_observation.raw_rgb,
                previous_observation.proprio,
                current_observation.proprio,
            )
            self._synchronize_device()
            self.metrics.latencies["FastEncoder_ms"].append((time.perf_counter() - t0) * 1000.0)
            self.metrics.counters["num_observation_encoder_calls"] += 1
        if self.latentloop_adapter.action_encoder is None:
            action_feature = observation_feature.new_zeros(
                (batch_size, self.latentloop_adapter.config.action_feature_dim)
            )
        else:
            self._synchronize_device()
            t0 = time.perf_counter()
            action_feature = self.latentloop_adapter.encode_executed_actions(
                executed_actions,
                self.execution_horizon,
                elapsed_time,
                reference_feature=observation_feature,
            )
            self._synchronize_device()
            self.metrics.latencies.setdefault("executed_action_encoder_ms", []).append(
                (time.perf_counter() - t0) * 1000.0
            )
            self.metrics.counters["num_executed_action_encoder_calls"] += 1
        return observation_feature, action_feature

    def _teacher_tracking_comparison(
        self,
        batch: dict[str, Tensor],
        *,
        condition: Tensor,
        action_chunk: Tensor,
        policy_query_index: int,
        query_age: int,
        source: str,
    ) -> dict[str, Any]:
        """Compare against a logging-only full teacher without operational counters."""

        self._synchronize_device()
        started = time.perf_counter()
        with torch.no_grad():
            teacher_condition = self.condition_adapter.encode_condition(
                input_ids=batch["input_ids"],
                image_input=batch["image_input"],
                image_mask=batch["image_mask"],
            )
            noise_key = self._action_noise_key(policy_query_index)
            initial_noise = explicit_action_noise(
                noise_key,
                batch_size=teacher_condition.shape[0],
                action_horizon=self.action_adapter.num_actions,
                action_dim=self.action_adapter.dim_action,
                device=batch["proprio"].device,
                dtype=batch["proprio"].dtype,
            )
            teacher_action = self.action_adapter.decode_action_from_condition(
                teacher_condition,
                batch["proprio"],
                steps=self.flow_steps,
                initial_noise=initial_noise,
            )
        self._synchronize_device()
        action_difference = action_chunk - teacher_action
        prefix = action_difference[:, : self.execution_horizon]
        tracking: dict[str, Any] = {
            "policy_query_index": int(policy_query_index),
            "query_age": int(query_age),
            "source": source,
            "action_noise_hash": _tensor_hash(initial_noise),
            "same_noise_chunk_l1": float(action_difference.abs().mean().item()),
            "same_noise_chunk_l2": float(
                torch.linalg.vector_norm(action_difference.flatten(start_dim=1), dim=1)
                .mean()
                .item()
            ),
            "executed_prefix_l1": float(prefix.abs().mean().item()),
            "executed_prefix_l2": float(
                torch.linalg.vector_norm(prefix.flatten(start_dim=1), dim=1).mean().item()
            ),
            "teacher_tracking_ms": (time.perf_counter() - started) * 1000.0,
            "excluded_from_operational_latency_and_counters": True,
        }
        if source != "action_chunk_correction":
            condition_difference = condition - teacher_condition
            tracking.update(
                {
                    "condition_mse": float(condition_difference.square().mean().item()),
                    "condition_normalized_mse": float(
                        F.mse_loss(
                            F.layer_norm(condition, (condition.shape[-1],)),
                            F.layer_norm(
                                teacher_condition,
                                (teacher_condition.shape[-1],),
                            ),
                        ).item()
                    ),
                    "condition_cosine": float(
                        F.cosine_similarity(
                            condition.flatten(start_dim=1),
                            teacher_condition.flatten(start_dim=1),
                            dim=1,
                        )
                        .mean()
                        .item()
                    ),
                }
            )
        self.metrics.counters["num_teacher_tracking_full_calls"] += 1
        self.metrics.counters["num_teacher_tracking_action_decodes"] += 1
        self.latentloop_tracking_trace.append(tracking)
        return tracking

    def _lightweight_action(
        self,
        batch: dict[str, Tensor],
        *,
        policy_query_index: int,
        query_age: int,
    ) -> tuple[Tensor, Tensor, int | None]:
        current_observation = SimVLAQueryObservation(
            raw_rgb=batch["raw_rgb"],
            proprio=batch["proprio"],
        )
        inputs = self.query_cache.lightweight_transition_inputs(current_observation)
        elapsed_time = self.execution_horizon / 20.0
        if self.latentloop_mode == "hold_condition":
            condition = inputs["previous_condition"]
            action_chunk, seed = self._decode(
                condition,
                batch["proprio"],
                policy_query_index=policy_query_index,
            )
            self.metrics.counters["num_hold_condition_queries"] += 1
            self.query_cache.commit_lightweight_update(condition, current_observation)
            self.cached_condition = condition.detach()
            self.cached_raw_rgb = batch["raw_rgb"].detach()
            self.cached_proprio = batch["proprio"].detach()
            self.cached_action_chunk = action_chunk.detach()
            return condition, action_chunk, seed
        if self.latentloop_mode == "nonrecurrent_condition":
            previous_observation = inputs["anchor_observation"]
            action_features: list[Tensor] = []
            observation_feature, latest_action_feature = self._adapter_features(
                previous_observation=previous_observation,
                current_observation=current_observation,
                executed_actions=inputs["executed_subchunk"],
                elapsed_time=elapsed_time,
            )
            action_features.append(latest_action_feature)
            if self.latentloop_adapter is None:
                raise AssertionError("adapter unexpectedly missing")
            for subchunk in inputs["executed_subchunks_since_anchor"][:-1]:
                if self.latentloop_adapter.action_encoder is not None:
                    self._synchronize_device()
                    history_started = time.perf_counter()
                    action_features.append(
                        self.latentloop_adapter.encode_executed_actions(
                            subchunk,
                            self.execution_horizon,
                            elapsed_time,
                            reference_feature=observation_feature,
                        )
                    )
                    self._synchronize_device()
                    self.metrics.latencies.setdefault(
                        "executed_action_encoder_ms", []
                    ).append((time.perf_counter() - history_started) * 1000.0)
                    self.metrics.counters["num_executed_action_encoder_calls"] += 1
            action_feature = torch.stack(action_features, dim=0).mean(dim=0)
        else:
            observation_feature, action_feature = self._adapter_features(
                previous_observation=inputs["previous_query_observation"],
                current_observation=current_observation,
                executed_actions=inputs["executed_subchunk"],
                elapsed_time=elapsed_time,
            )
        if self.latentloop_mode == "action_chunk_correction":
            if self.latentloop_adapter is None or self.cached_action_chunk is None:
                raise RuntimeError("action correction requires adapter and previous action chunk")
            self._synchronize_device()
            t0 = time.perf_counter()
            action_chunk = self.latentloop_adapter.correct_action_chunk(
                self.cached_action_chunk,
                observation_feature,
                action_feature,
                execution_horizon=self.execution_horizon,
                elapsed_time=elapsed_time,
                query_age=query_age,
            )
            self._synchronize_device()
            condition = inputs["previous_condition"]
            seed = None
            self.metrics.latencies.setdefault("action_correction_ms", []).append(
                (time.perf_counter() - t0) * 1000.0
            )
            self.metrics.counters["num_action_correction_calls"] += 1
        elif self.latentloop_mode == "nonrecurrent_condition":
            if self.latentloop_adapter is None:
                raise RuntimeError("nonrecurrent mode requires adapter")
            self._synchronize_device()
            t0 = time.perf_counter()
            condition = self.latentloop_adapter.predict_nonrecurrent_condition(
                inputs["anchor_condition"],
                observation_feature,
                action_feature,
                execution_horizon=self.execution_horizon,
                elapsed_time=elapsed_time,
                query_age=query_age,
            )
            self._synchronize_device()
            self.metrics.latencies.setdefault("nonrecurrent_predictor_ms", []).append(
                (time.perf_counter() - t0) * 1000.0
            )
            action_chunk, seed = self._decode(
                condition,
                batch["proprio"],
                policy_query_index=policy_query_index,
            )
            self.metrics.counters["num_nonrecurrent_condition_calls"] += 1
        else:
            if self.latentloop_adapter is None:
                raise RuntimeError("recurrent mode requires adapter")
            self._synchronize_device()
            t0 = time.perf_counter()
            condition = self.latentloop_adapter.update_recurrent_condition(
                inputs["previous_condition"],
                observation_feature,
                action_feature,
                execution_horizon=self.execution_horizon,
                elapsed_time=elapsed_time,
                query_age=query_age,
            )
            self._synchronize_device()
            self.metrics.latencies.setdefault("condition_updater_ms", []).append(
                (time.perf_counter() - t0) * 1000.0
            )
            action_chunk, seed = self._decode(
                condition,
                batch["proprio"],
                policy_query_index=policy_query_index,
            )
            self.metrics.counters["num_condition_updater_calls"] += 1
        self.query_cache.commit_lightweight_update(condition, current_observation)
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action_chunk.detach()
        return condition, action_chunk, seed

    def _refill_action_queue(self, batch: dict[str, Tensor]) -> dict[str, Any]:
        self._synchronize_device()
        query_started = time.perf_counter()
        self._record_previous_execution()
        self.metrics.counters["num_policy_queries"] += 1
        query_index = int(self.query_index)
        full_refresh = (
            self.latentloop_mode == "full"
            or self.cached_condition is None
            or query_index % self.full_query_interval == 0
        )
        query_age = query_index % self.full_query_interval
        if full_refresh:
            condition, action_chunk, noise_seed = self._full_refresh(
                batch,
                policy_query_index=query_index,
            )
            self.query_cache.full_refresh(
                condition,
                SimVLAQueryObservation(batch["raw_rgb"], batch["proprio"]),
            )
            source = "full_refresh"
        else:
            condition, action_chunk, noise_seed = self._lightweight_action(
                batch,
                policy_query_index=query_index,
                query_age=query_age,
            )
            source = self.latentloop_mode
        self.action_queue.clear()
        for action in action_chunk[0, : self.execution_horizon]:
            self.action_queue.append((action.detach(), source))
        record = {
            "policy_query_index": query_index,
            "query_age": query_age,
            "full_refresh": bool(full_refresh),
            "source": source,
            "action_noise_seed": noise_seed,
            "action_noise_hash": self.action_noise_hashes.get(query_index),
            "action_chunk_hash": _tensor_hash(action_chunk),
            "condition_hash": _tensor_hash(condition),
            "action_chunk_shape": list(action_chunk.shape),
        }
        self.latentloop_query_trace.append(record)
        if self.log_action_chunks:
            self.latentloop_action_chunks.append(
                {**record, "action_chunk": action_chunk.detach().cpu()}
            )
        self.metrics.latencies.setdefault("policy_query_total_ms", []).append(
            (time.perf_counter() - query_started) * 1000.0
        )
        if self.teacher_tracking and not full_refresh:
            self._pending_teacher_tracking = (
                batch,
                condition,
                action_chunk,
                query_index,
                query_age,
                source,
                record,
            )
        self.query_index += 1
        return {"refreshed": full_refresh, "age": query_age, "queue_mode": source}

    def act(self, image0: np.ndarray, image1: np.ndarray, proprio: np.ndarray, prompt: str) -> EvalStepOutput:
        """Return one queued action and stage it as actually sent to the environment."""

        self._synchronize_device()
        total_t0 = time.perf_counter()
        self.metrics.counters["num_env_steps"] += 1
        refill_info: dict[str, Any] = {
            "refreshed": False,
            "age": self.query_index % self.full_query_interval,
            "queue_mode": "queued",
        }
        if not self.action_queue:
            refill_info = self._refill_action_queue(
                self.preprocess(image0, image1, proprio, prompt)
            )
        action, source = self.action_queue.popleft()
        self.actions_sent_since_query.append(action.detach().clone())
        self.metrics.counters["num_action_queue_steps"] += 1
        self.metrics.counters["num_environment_actions"] += 1
        self.metrics.latencies["policy_total_ms"].append((time.perf_counter() - total_t0) * 1000.0)
        if self._pending_teacher_tracking is not None:
            (
                tracking_batch,
                tracking_condition,
                tracking_action,
                tracking_query_index,
                tracking_query_age,
                tracking_source,
                tracking_record,
            ) = self._pending_teacher_tracking
            tracking_record["teacher_tracking"] = self._teacher_tracking_comparison(
                tracking_batch,
                condition=tracking_condition,
                action_chunk=tracking_action,
                policy_query_index=tracking_query_index,
                query_age=tracking_query_age,
                source=tracking_source,
            )
            self._pending_teacher_tracking = None
        action_np = action.detach().cpu().numpy().astype(np.float32)
        self.metrics.observe_action(action_np)
        self.step_index += 1
        return EvalStepOutput(
            action=action_np,
            info={
                "mode": self.latentloop_mode,
                "source": source,
                "refreshed": bool(refill_info["refreshed"]),
                "age": int(refill_info["age"]),
                "execution_horizon": self.execution_horizon,
                "full_query_interval": self.full_query_interval,
                "counters": dict(self.metrics.counters),
            },
        )
