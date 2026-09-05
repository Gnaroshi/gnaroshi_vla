"""Deployment policy preserving SimVLA's fresh-H10/execute-R5 protocol."""

from __future__ import annotations

import time
from typing import Any

import torch

from .bootstrap import configure_model_imports


configure_model_imports()

from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (  # noqa: E402
    GENERATION_SCHEDULES,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_policy import (  # noqa: E402
    RealSimVLAGenerationPolicy,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (  # noqa: E402
    condition_update_with_code,
)
from architectures.simvla.adapters.latentloop.native_v0_policy import (  # noqa: E402
    RealSimVLANativeV0Policy,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (  # noqa: E402
    RealSimVLADCLDPolicy,
)
from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair  # noqa: E402
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop  # noqa: E402
from architectures.simvla.adapters.real_world_training.dataset import (  # noqa: E402
    align_current_rotvec_proprio,
)


def _real_condition_observation_pair(
    policy: Any, batch: dict[str, torch.Tensor]
) -> NativeV0ObservationPair:
    if policy.cached_raw_rgb is None or policy.cached_proprio is None:
        raise RuntimeError("real Condition update requires the preceding query observation")
    return NativeV0ObservationPair(
        previous_images=policy.cached_raw_rgb,
        current_images=batch["raw_rgb"],
        previous_proprio=policy.cached_proprio,
        current_proprio=align_current_rotvec_proprio(
            policy.cached_proprio, batch["proprio"]
        ),
    )


class FullSimVLARealPolicy(RealSimVLADCLDPolicy):
    """Frozen baseline: full condition and ten flow evaluations per fresh chunk."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            dcld_core=None,
            mode="full",
            refresh_every=1,
            flow_steps=10,
            image_size=384,
            replan_steps=5,
            client_resize_size=224,
            row_name="real_baseline_kc1_ng10",
            paired_action_noise=True,
            **kwargs,
        )


class ConditionLoopSimVLARealPolicy(RealSimVLANativeV0Policy):
    """Condition-only comparator: K_C=2 with the full ten-step generator."""

    def __init__(self, **kwargs: Any) -> None:
        self.k_c = 2
        self.n_g = 10
        super().__init__(
            flow_steps=10,
            image_size=384,
            client_resize_size=224,
            **kwargs,
        )
        self.mode = "real_condition_loop_kc2_ng10"
        self.row_name = self.mode
        self.refresh_every = self.k_c

    def _condition_observation_pair(
        self, batch: dict[str, torch.Tensor]
    ) -> NativeV0ObservationPair:
        return _real_condition_observation_pair(self, batch)

    def _refill_action_queue(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self.metrics.counters["num_policy_queries"] += 1
        query = int(self.query_index)
        age = query % self.k_c
        if age == 0:
            _, action_chunk, seed = self._full_refresh(
                batch, policy_query_index=query
            )
            source = "full_refresh"
            refreshed = True
        else:
            _, action_chunk, seed = self._v0_update(
                batch, age=age, policy_query_index=query
            )
            source = "condition_update"
            refreshed = False

        self.action_queue.clear()
        for action in action_chunk[0, :5]:
            self.action_queue.append((action.detach(), source))
        self.query_trace.append(
            {
                "policy_query_index": query,
                "condition_age": age,
                "source": source,
                "full_vlm_called": refreshed,
                "condition_updater_called": not refreshed,
                "action_noise_seed": seed,
                "action_horizon": 10,
                "execution_horizon": 5,
                "condition_refresh_interval": self.k_c,
                "generation_full_evaluations": self.n_g,
            }
        )
        self.query_index += 1

        queries = int(self.metrics.counters["num_policy_queries"])
        full_calls = int(self.metrics.counters["num_full_vlm_calls"])
        update_calls = int(self.metrics.counters["num_condition_updater_calls"])
        expected_full = (queries + self.k_c - 1) // self.k_c
        if full_calls != expected_full or update_calls != queries - expected_full:
            raise RuntimeError("K_C=2 condition-only schedule counter drift")
        expected_generation_calls = queries * self.n_g
        observed_generation_calls = int(
            self.metrics.counters["num_action_transformer_calls"]
        )
        if observed_generation_calls != expected_generation_calls:
            raise RuntimeError("N_G=10 condition-only generation counter drift")
        return {
            "refreshed": refreshed,
            "age": age,
            "queue_mode": source,
            "action_noise_seed": seed,
        }


class LatentLoopSimVLARealPolicy(RealSimVLANativeV0Policy):
    """Selected deployment point: K_C=2, N_G=3, and real condition-code coupling."""

    def __init__(self, *, generation_updater: Any, **kwargs: Any) -> None:
        self.k_c = 2
        self.n_g = 3
        self.full_step_indices = GENERATION_SCHEDULES[self.n_g]
        super().__init__(
            flow_steps=10,
            image_size=384,
            client_resize_size=224,
            **kwargs,
        )
        self.mode = "real_latentloop_kc2_ng3"
        self.row_name = self.mode
        self.refresh_every = self.k_c
        self.generation_loop = SimVLAGenerationLoop(
            generation_updater.eval(), self.model.transformer.action_decoder
        ).to(self.device)
        self.generation_loop.eval()
        self.metrics.latencies.setdefault("generation_loop_ms", [])
        self._active_condition_change_code: torch.Tensor | None = None
        self.condition_change_code_norms: list[float] = []

    def reset(self) -> None:
        super().reset()
        self.metrics.latencies.setdefault("generation_loop_ms", [])
        self._active_condition_change_code = None
        self.condition_change_code_norms = []

    def _full_refresh(
        self,
        batch: dict[str, torch.Tensor],
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        self._active_condition_change_code = None
        return super()._full_refresh(batch, policy_query_index=policy_query_index)

    def _v0_update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        age: int,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        if age != 1:
            raise ValueError("real K_C=2 coupling has exactly one update age")
        if (
            self.cached_condition is None
            or self.cached_raw_rgb is None
            or self.cached_proprio is None
        ):
            raise RuntimeError("coupled update requires the preceding query state")
        if self.condition_layout is None:
            raise RuntimeError("coupled update requires the full-refresh token layout")
        pair = _real_condition_observation_pair(self, batch)
        self._sync()
        started = time.perf_counter()
        with torch.no_grad():
            exposed = condition_update_with_code(
                self.native_v0,
                self.cached_condition,
                pair,
                valid_mask=self.condition_layout.valid_mask,
                group_ids=self.condition_layout.group_ids,
                age=1,
            )
        self._sync()
        self.metrics.latencies.setdefault("condition_updater_ms", []).append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_condition_updater_calls"] += 1
        self.metrics.counters["num_observation_encoder_calls"] += 1
        self.metrics.counters["num_condition_change_code_queries"] += 1
        code = exposed.condition_change_code.detach()
        code_norm = code.float().norm(dim=-1)
        if not bool(torch.isfinite(code_norm).all()):
            raise RuntimeError("online real Condition Updater produced a non-finite change code")
        self._active_condition_change_code = code
        self.condition_change_code_norms.extend(code_norm.cpu().tolist())
        action, seed = self._decode(
            exposed.update.condition,
            batch["proprio"],
            policy_query_index=policy_query_index,
        )
        self.cached_condition = exposed.update.condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action.detach()
        return exposed.update.condition, action, seed

    def _decode(
        self,
        condition: torch.Tensor,
        proprio: torch.Tensor,
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, int | None]:
        code = self._active_condition_change_code
        if code is None:
            code = condition.new_zeros(
                (condition.shape[0], self.generation_loop.updater.condition_code_dim)
            )
        return RealSimVLAGenerationPolicy._decode_with_condition_code(
            self,
            condition,
            proprio,
            policy_query_index=policy_query_index,
            condition_change_code=code,
        )

    def _refill_action_queue(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self.metrics.counters["num_policy_queries"] += 1
        query = int(self.query_index)
        age = query % self.k_c
        if age == 0:
            _, action_chunk, seed = self._full_refresh(
                batch, policy_query_index=query
            )
            source = "full_refresh"
            refreshed = True
        else:
            _, action_chunk, seed = self._v0_update(
                batch, age=age, policy_query_index=query
            )
            source = "condition_update"
            refreshed = False

        self.action_queue.clear()
        for action in action_chunk[0, :5]:
            self.action_queue.append((action.detach(), source))
        self.query_trace.append(
            {
                "policy_query_index": query,
                "condition_age": age,
                "source": source,
                "full_vlm_called": refreshed,
                "condition_updater_called": not refreshed,
                "action_noise_seed": seed,
                "action_horizon": 10,
                "execution_horizon": 5,
                "condition_refresh_interval": self.k_c,
                "generation_full_evaluations": self.n_g,
                "condition_change_code_used": bool(
                    self._active_condition_change_code is not None
                ),
            }
        )
        self.query_index += 1

        queries = int(self.metrics.counters["num_policy_queries"])
        full_calls = int(self.metrics.counters["num_full_vlm_calls"])
        update_calls = int(self.metrics.counters["num_condition_updater_calls"])
        expected_full = (queries + self.k_c - 1) // self.k_c
        if full_calls != expected_full or update_calls != queries - expected_full:
            raise RuntimeError("K_C=2 condition schedule counter drift")
        expected_generation_calls = queries * self.n_g
        observed_generation_calls = int(
            self.metrics.counters["num_action_transformer_calls"]
        )
        if observed_generation_calls != expected_generation_calls:
            raise RuntimeError("N_G=3 generation schedule counter drift")
        return {
            "refreshed": refreshed,
            "age": age,
            "queue_mode": source,
            "action_noise_seed": seed,
        }
