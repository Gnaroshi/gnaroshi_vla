"""Deployment policy preserving SimVLA's fresh-H10/execute-R5 protocol."""

from __future__ import annotations

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
from architectures.simvla.adapters.latentloop.native_v0_policy import (  # noqa: E402
    RealSimVLANativeV0Policy,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (  # noqa: E402
    RealSimVLADCLDPolicy,
)
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop  # noqa: E402


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
    """Selected deployment point: K_C=2 and N_G=3 with uncoupled updates."""

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

    def reset(self) -> None:
        super().reset()
        self.metrics.latencies.setdefault("generation_loop_ms", [])

    def _decode(
        self,
        condition: torch.Tensor,
        proprio: torch.Tensor,
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, int | None]:
        # This delegates to the validated uncoupled path, whose condition-change
        # code is exactly zero at every generation update.
        return RealSimVLAGenerationPolicy._decode(
            self,
            condition,
            proprio,
            policy_query_index=policy_query_index,
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
