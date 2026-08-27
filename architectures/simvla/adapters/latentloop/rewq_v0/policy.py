"""SimVLA H=10/R=5 runtime policy for the provisional rewq v0 router."""

from __future__ import annotations

import time
from typing import Any

import torch

from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair
from methods.latentloop.modules.rewq_v0 import (
    COMPUTE_MODES,
    MODE_BY_ID,
    ComputeCostTable,
    ModeAssessment,
    RecoverabilityHead,
    RecoverabilityRouter,
    RecoverySafetyEnvelope,
    SplitConformalCalibration,
)
from ..efficient_multirate.contracts import GENERATION_SCHEDULES
from ..efficient_multirate.coupled_condition_generation import (
    ConditionUpdateWithCode,
    condition_update_with_code,
)
from ..efficient_multirate.fixed_2x2_eval import SynchronizedCombinedK_CN_GPolicy
from .features import (
    SimVLARecoverabilityFeatureConfig,
    build_simvla_recoverability_features,
)


class RewqV0SimVLAPolicy(SynchronizedCombinedK_CN_GPolicy):
    """Jointly choose condition and learned generation compute at each query.

    The original SimVLA query cadence (five executed actions per ten-action
    chunk) is invariant.  A rejected U_C candidate is never action-decoded.
    """

    def __init__(
        self,
        *,
        recoverability_head: RecoverabilityHead,
        envelope: RecoverySafetyEnvelope,
        conformal: SplitConformalCalibration,
        costs: ComputeCostTable,
        feature_config: SimVLARecoverabilityFeatureConfig | None = None,
        **kwargs: Any,
    ) -> None:
        config = feature_config or SimVLARecoverabilityFeatureConfig()
        if recoverability_head.input_dim != config.input_dim:
            raise ValueError("recoverability head and feature dimensions differ")
        device = torch.device(kwargs["device"])
        self.recoverability_head = recoverability_head.to(device).eval()
        for parameter in self.recoverability_head.parameters():
            parameter.requires_grad_(False)
        self.recoverability_router = RecoverabilityRouter(
            envelope=envelope,
            conformal=conformal,
            costs=costs,
            max_approximate_age=config.max_age,
        )
        self.recoverability_feature_config = config
        self.approximate_age = 0
        self.exact_anchor_condition: torch.Tensor | None = None
        self.mode_trace: list[dict[str, Any]] = []
        self.recoverability_tensor_trace: list[dict[str, Any]] = []
        super().__init__(
            generation_updater=kwargs.pop("generation_updater"),
            n_g=3,
            k_c=4,
            row_name="rewq_v0_recoverability_router",
            **kwargs,
        )
        for module in (self.model, self.native_v0, self.generation_loop.updater):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.metrics.latencies.setdefault("recoverability_router_ms", [])
        self.metrics.counters.setdefault("num_recoverability_head_calls", 0)
        self.metrics.counters.setdefault("num_rejected_condition_candidates", 0)
        for mode in COMPUTE_MODES:
            self.metrics.counters.setdefault(f"num_mode_{mode.name}", 0)

    def reset(self) -> None:
        super().reset()
        self.approximate_age = 0
        self.exact_anchor_condition = None
        self.mode_trace = []
        self.recoverability_tensor_trace = []

    def _set_generation_compute(self, n_g: int) -> None:
        if int(n_g) not in {2, 3}:
            raise ValueError("rewq v0 generation compute must be N_G=2 or N_G=3")
        self.n_g = int(n_g)
        self.full_step_indices = GENERATION_SCHEDULES[self.n_g]

    def _full_refresh_mode(
        self,
        batch: dict[str, torch.Tensor],
        *,
        policy_query_index: int,
        n_g: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        self._set_generation_compute(n_g)
        condition, action, seed = super()._full_refresh(
            batch,
            policy_query_index=policy_query_index,
        )
        self.exact_anchor_condition = condition.detach()
        self.approximate_age = 0
        return condition, action, seed

    def _candidate_update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        candidate_age: int,
    ) -> tuple[ConditionUpdateWithCode, ModeAssessment]:
        if self.cached_condition is None or self.cached_raw_rgb is None:
            raise RuntimeError("rewq v0 requires a preceding condition and image")
        if self.cached_proprio is None or self.cached_action_chunk is None:
            raise RuntimeError("rewq v0 requires preceding proprio and action chunk")
        if self.condition_layout is None or self.exact_anchor_condition is None:
            raise RuntimeError("rewq v0 requires an exact condition anchor")
        pair = NativeV0ObservationPair(
            previous_images=self.cached_raw_rgb,
            current_images=batch["raw_rgb"],
            previous_proprio=self.cached_proprio,
            current_proprio=batch["proprio"],
        )
        self._sync()
        started = time.perf_counter()
        with torch.no_grad():
            exposed = condition_update_with_code(
                self.native_v0,
                self.cached_condition,
                pair,
                valid_mask=self.condition_layout.valid_mask,
                group_ids=self.condition_layout.group_ids,
                # The frozen V0 checkpoint contains age embeddings 1--3.  For
                # longer recursive use, retain the learned oldest embedding
                # while the router receives the true age 4--7 and can reject it.
                age=min(int(candidate_age), 3),
            )
        self._sync()
        self.metrics.latencies.setdefault("condition_updater_ms", []).append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_condition_updater_calls"] += 1
        self.metrics.counters["num_observation_encoder_calls"] += 1

        self._sync()
        route_started = time.perf_counter()
        with torch.no_grad():
            features = build_simvla_recoverability_features(
                delta_feature=exposed.condition_change_code,
                update=exposed.update,
                anchor_condition=self.exact_anchor_condition,
                valid_mask=self.condition_layout.valid_mask,
                group_ids=self.condition_layout.group_ids,
                previous_action_chunk=self.cached_action_chunk,
                previous_proprio=self.cached_proprio,
                current_proprio=batch["proprio"],
                candidate_age=torch.full(
                    (batch["proprio"].shape[0],),
                    int(candidate_age),
                    device=batch["proprio"].device,
                    dtype=torch.long,
                ),
                config=self.recoverability_feature_config,
            )
            prediction = self.recoverability_head(features)
            assessment = self.recoverability_router.assess(
                prediction,
                candidate_age=torch.full(
                    (features.shape[0],),
                    int(candidate_age),
                    device=features.device,
                    dtype=torch.long,
                ),
                anchor_available=torch.ones(
                    (features.shape[0],), device=features.device, dtype=torch.bool
                ),
            )
        self._sync()
        self.metrics.latencies["recoverability_router_ms"].append(
            (time.perf_counter() - route_started) * 1000.0
        )
        self.metrics.counters["num_recoverability_head_calls"] += 1
        if assessment.selected_mode_id.numel() != 1:
            raise ValueError("online rewq v0 policy expects batch size one")
        return exposed, assessment

    def _commit_approximate(
        self,
        exposed: ConditionUpdateWithCode,
        batch: dict[str, torch.Tensor],
        *,
        policy_query_index: int,
        n_g: int,
        candidate_age: int,
    ) -> tuple[torch.Tensor, int | None]:
        self._set_generation_compute(n_g)
        action, seed = self._decode(
            exposed.update.condition,
            batch["proprio"],
            policy_query_index=policy_query_index,
        )
        self.cached_condition = exposed.update.condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action.detach()
        self.approximate_age = int(candidate_age)
        return action, seed

    def _refill_action_queue(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self.metrics.counters["num_policy_queries"] += 1
        query = int(self.query_index)
        candidate_age = self.approximate_age + 1
        candidate: ConditionUpdateWithCode | None = None

        if query == 0 or candidate_age > self.recoverability_feature_config.max_age:
            mode_id = 0
        else:
            candidate, assessment = self._candidate_update(
                batch,
                candidate_age=candidate_age,
            )
            self.recoverability_tensor_trace.append(
                {
                    "policy_query_index": query,
                    "candidate_age": candidate_age,
                    "admissible": assessment.admissible.detach(),
                    "continuous_ucb": assessment.continuous_ucb.detach(),
                    "event_probability_ucb": assessment.event_probability_ucb.detach(),
                    "mode_cost_ms": assessment.mode_cost_ms.detach(),
                }
            )
            # One scalar synchronization is required to dispatch the selected
            # external compute graph.  Score construction itself remains GPU-only.
            mode_id = int(assessment.selected_mode_id.item())
        mode = MODE_BY_ID[mode_id]
        self.metrics.counters[f"num_mode_{mode.name}"] += 1

        if mode.exact_condition:
            if candidate is not None:
                self.metrics.counters["num_rejected_condition_candidates"] += 1
            _, action_chunk, seed = self._full_refresh_mode(
                batch,
                policy_query_index=query,
                n_g=mode.generation_n_g,
            )
        else:
            if candidate is None:
                raise RuntimeError("approximate mode selected without a U_C candidate")
            action_chunk, seed = self._commit_approximate(
                candidate,
                batch,
                policy_query_index=query,
                n_g=mode.generation_n_g,
                candidate_age=candidate_age,
            )

        source = mode.name
        self.action_queue.clear()
        for action in action_chunk[0, :5]:
            self.action_queue.append((action.detach(), source))
        record = {
            "policy_query_index": query,
            "candidate_age": 0 if query == 0 else candidate_age,
            "selected_mode_id": mode.mode_id,
            "selected_mode": mode.name,
            "full_vlm_called": mode.exact_condition,
            "condition_updater_called": candidate is not None,
            "generation_n_g": mode.generation_n_g,
            "action_noise_seed": seed,
            "action_horizon": 10,
            "execution_horizon": 5,
        }
        self.mode_trace.append(record)
        self.query_trace.append(record.copy())
        if self.log_action_chunks:
            self.action_chunk_records.append(
                {
                    "suite": self.suite,
                    "task_id": self.task_id,
                    "trial_id": self.trial_id,
                    "episode_step_index": int(self.step_index),
                    "policy_query_index": query,
                    "row_name": self.row_name,
                    "mode": self.mode,
                    "queue_mode": source,
                    "full_vlm_called": mode.exact_condition,
                    "condition_updater_called": candidate is not None,
                    "generation_n_g": mode.generation_n_g,
                    "action_noise_seed": seed,
                    "action_chunk_shape": list(action_chunk.shape),
                    "action_chunk": action_chunk.detach().cpu().float(),
                }
            )
        self.query_index += 1

        queries = int(self.metrics.counters["num_policy_queries"])
        decodes = int(self.metrics.counters["num_action_transformer_decodes"])
        if decodes != queries:
            raise RuntimeError("rewq v0 changed one-decode-per-query accounting")
        return {
            "refreshed": mode.exact_condition,
            "age": self.approximate_age,
            "queue_mode": source,
            "action_noise_seed": seed,
            "generation_n_g": mode.generation_n_g,
        }

    def recoverability_trace_cpu(self) -> list[dict[str, Any]]:
        """Materialize diagnostic tensors only after an episode has finished."""

        result: list[dict[str, Any]] = []
        for record in self.recoverability_tensor_trace:
            result.append(
                {
                    "policy_query_index": int(record["policy_query_index"]),
                    "candidate_age": int(record["candidate_age"]),
                    "admissible": record["admissible"].cpu().tolist(),
                    "continuous_ucb": record["continuous_ucb"].cpu().tolist(),
                    "event_probability_ucb": record[
                        "event_probability_ucb"
                    ].cpu().tolist(),
                    "mode_cost_ms": record["mode_cost_ms"].cpu().tolist(),
                }
            )
        return result

    def scientific_contract(self) -> dict[str, Any]:
        return {
            **self.recoverability_router.contract(),
            "base_simvla_frozen": not any(
                parameter.requires_grad for parameter in self.model.parameters()
            ),
            "condition_updater_frozen": not any(
                parameter.requires_grad for parameter in self.native_v0.parameters()
            ),
            "generation_updater_frozen": not any(
                parameter.requires_grad
                for parameter in self.generation_loop.updater.parameters()
            ),
            "recoverability_head_parameters": sum(
                parameter.numel() for parameter in self.recoverability_head.parameters()
            ),
            "one_action_decode_per_query": True,
            "candidate_decoded_before_routing": False,
            "condition_updater_age_embedding_max": 3,
            "condition_updater_long_age_rule": "min(true_age,3)",
            "router_receives_unsaturated_true_age": True,
            "dynamic_action_execution": False,
        }
