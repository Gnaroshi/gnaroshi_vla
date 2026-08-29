"""SimVLA policy hook for action-equivalent selective condition refresh."""

from __future__ import annotations

import time
from typing import Any

import torch

from architectures.simvla.adapters.latentloop.action_equivalent_refresh.features import (
    SimVLAActionFidelityFeatureConfig,
    build_simvla_action_fidelity_features,
)
from ..efficient_multirate.coupled_condition_generation import (
    ConditionUpdateWithCode,
    condition_update_with_code,
)
from ..efficient_multirate.fixed_2x2_eval import (
    SynchronizedCombinedK_CN_GPolicy,
)
from methods.latentloop.modules.action_equivalent_refresh import (
    ActionEquivalentRefreshRouter,
    ActionFidelityHead,
    ActionFidelityPrediction,
    ExactCallBudgetCalibration,
    RefreshDecision,
)
from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair


class ActionEquivalentRefreshSimVLAPolicy(SynchronizedCombinedK_CN_GPolicy):
    """Select exact ``F`` versus approximate ``U_C`` at fixed query times.

    SimVLA's H=10/R=5 action queue and the validated learned N_G=3 generation
    schedule remain unchanged.  The router runs only after a cheap candidate
    condition update and before action generation, so rejected candidates do
    not incur a second action decode.
    """

    def __init__(
        self,
        *,
        risk_head: ActionFidelityHead,
        calibration: ExactCallBudgetCalibration,
        feature_config: SimVLAActionFidelityFeatureConfig | None = None,
        **kwargs: Any,
    ) -> None:
        config = feature_config or SimVLAActionFidelityFeatureConfig()
        if config.max_age != 3 or calibration.max_approximate_age != 3:
            raise ValueError("primary selective-refresh policy is fixed to maximum age 3")
        if risk_head.input_dim != config.input_dim:
            raise ValueError("risk-head input dimension differs from SimVLA features")
        device = torch.device(kwargs["device"])
        self.risk_head = risk_head.to(device).eval()
        for parameter in self.risk_head.parameters():
            parameter.requires_grad_(False)
        self.refresh_router = ActionEquivalentRefreshRouter(calibration)
        self.feature_config = config
        self.refresh_decisions: list[dict[str, Any]] = []
        super().__init__(
            generation_updater=kwargs.pop("generation_updater"),
            n_g=3,
            k_c=4,
            row_name="action_equivalent_refresh_ng3",
            **kwargs,
        )
        for module in (self.model, self.native_v0, self.generation_loop.updater):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.refresh_every = 4
        self.metrics.latencies.setdefault("risk_head_ms", [])

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "refresh_router"):
            self.refresh_router.reset()
        self.refresh_decisions = []

    def _candidate_update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        candidate_age: int,
    ) -> tuple[ConditionUpdateWithCode, torch.Tensor, ActionFidelityPrediction]:
        if self.cached_condition is None or self.cached_raw_rgb is None:
            raise RuntimeError("selective refresh requires a preceding condition and image")
        if self.cached_proprio is None or self.cached_action_chunk is None:
            raise RuntimeError("selective refresh requires preceding proprio and action chunk")
        if self.condition_layout is None:
            raise RuntimeError("selective refresh requires the exact token layout")
        pair = NativeV0ObservationPair(
            previous_images=self.cached_raw_rgb,
            current_images=batch["raw_rgb"],
            previous_proprio=self.cached_proprio,
            current_proprio=batch["proprio"],
        )
        self._sync()
        update_started = time.perf_counter()
        with torch.no_grad():
            exposed = condition_update_with_code(
                self.native_v0,
                self.cached_condition,
                pair,
                valid_mask=self.condition_layout.valid_mask,
                group_ids=self.condition_layout.group_ids,
                age=candidate_age,
            )
        self._sync()
        self.metrics.latencies.setdefault("condition_updater_ms", []).append(
            (time.perf_counter() - update_started) * 1000.0
        )
        self.metrics.counters["num_condition_updater_calls"] += 1
        self.metrics.counters["num_observation_encoder_calls"] += 1
        self.metrics.counters["num_candidate_condition_updates"] += 1

        features = build_simvla_action_fidelity_features(
            delta_feature=exposed.condition_change_code,
            update=exposed.update,
            valid_mask=self.condition_layout.valid_mask,
            group_ids=self.condition_layout.group_ids,
            previous_action_chunk=self.cached_action_chunk,
            previous_proprio=self.cached_proprio,
            current_proprio=batch["proprio"],
            candidate_age=candidate_age,
            config=self.feature_config,
        )
        self._sync()
        risk_started = time.perf_counter()
        with torch.no_grad():
            prediction = self.risk_head(features)
        self._sync()
        self.metrics.latencies["risk_head_ms"].append(
            (time.perf_counter() - risk_started) * 1000.0
        )
        self.metrics.counters["num_risk_head_calls"] += 1
        return exposed, features, prediction

    def _commit_approximate(
        self,
        exposed: ConditionUpdateWithCode,
        batch: dict[str, torch.Tensor],
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, int | None]:
        condition = exposed.update.condition
        action, seed = self._decode(
            condition,
            batch["proprio"],
            policy_query_index=policy_query_index,
        )
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action.detach()
        self.metrics.counters["num_accepted_condition_updates"] += 1
        return action, seed

    def _record_decision(
        self,
        *,
        query: int,
        decision: RefreshDecision,
        prediction: ActionFidelityPrediction | None,
        source: str,
        seed: int | None,
    ) -> None:
        record = {
            "policy_query_index": int(query),
            "candidate_age": int(decision.candidate_age),
            "use_exact": bool(decision.use_exact),
            "reason": decision.reason,
            "risk_score": decision.risk_score,
            "arm_q90": (
                None if prediction is None else float(prediction.arm_q90.item())
            ),
            "direction_q90": (
                None if prediction is None else float(prediction.direction_q90.item())
            ),
            "gripper_mismatch_probability": (
                None
                if prediction is None
                else float(prediction.gripper_mismatch_probability.item())
            ),
            "source": source,
            "action_noise_seed": seed,
            "action_horizon": 10,
            "execution_horizon": 5,
            "generation_n_g": 3,
        }
        self.refresh_decisions.append(record)
        self.query_trace.append(record.copy())

    def _refill_action_queue(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        self.metrics.counters["num_policy_queries"] += 1
        query = int(self.query_index)
        prediction: ActionFidelityPrediction | None = None
        if not self.refresh_router.candidate_required():
            decision = self.refresh_router.decide()
            _, action_chunk, seed = self._full_refresh(
                batch, policy_query_index=query
            )
            source = f"selective_exact_{decision.reason}"
            if decision.reason == "max_age":
                self.metrics.counters["num_forced_age_refreshes"] += 1
        else:
            candidate_age = self.refresh_router.approximate_age + 1
            exposed, _, prediction = self._candidate_update(
                batch, candidate_age=candidate_age
            )
            decision = self.refresh_router.decide(prediction)
            if decision.use_exact:
                _, action_chunk, seed = self._full_refresh(
                    batch, policy_query_index=query
                )
                source = "selective_exact_risk"
                self.metrics.counters["num_risk_triggered_refreshes"] += 1
                self.metrics.counters["num_rejected_condition_updates"] += 1
            else:
                action_chunk, seed = self._commit_approximate(
                    exposed,
                    batch,
                    policy_query_index=query,
                )
                source = "selective_approximate"

        self.action_queue.clear()
        for action in action_chunk[0, :5]:
            self.action_queue.append((action.detach(), source))
        self._record_decision(
            query=query,
            decision=decision,
            prediction=prediction,
            source=source,
            seed=seed,
        )
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
                    "refreshed": bool(decision.use_exact),
                    "full_vlm_called": bool(decision.use_exact),
                    "condition_updater_called": prediction is not None,
                    "risk_score": decision.risk_score,
                    "action_noise_seed": seed,
                    "action_chunk_shape": list(action_chunk.shape),
                    "action_chunk": action_chunk.detach().cpu().float(),
                }
            )
        self.query_index += 1
        queries = int(self.metrics.counters["num_policy_queries"])
        full_calls = int(self.metrics.counters["num_full_vlm_calls"])
        accepted = int(self.metrics.counters["num_accepted_condition_updates"])
        if full_calls + accepted != queries:
            raise RuntimeError("selective refresh changed the policy-query accounting")
        return {
            "refreshed": bool(decision.use_exact),
            "age": int(decision.candidate_age),
            "queue_mode": source,
            "action_noise_seed": seed,
            "risk_score": decision.risk_score,
            "refresh_reason": decision.reason,
        }

    def scientific_contract(self) -> dict[str, Any]:
        return {
            **self.refresh_router.contract(),
            "simvla_action_horizon": 10,
            "simvla_execution_horizon": 5,
            "generation_n_g": 3,
            "generation_full_step_indices": list(self.full_step_indices),
            "condition_candidate_max_age": 3,
            "exact_candidate_same_action_noise": True,
            "risk_head_total_parameters": sum(
                parameter.numel() for parameter in self.risk_head.parameters()
            ),
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
            "dynamic_action_execution": False,
            "dynamic_generation_n_g": False,
        }
