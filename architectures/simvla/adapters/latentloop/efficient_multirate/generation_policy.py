"""Inference policy that replaces selected SimVLA action-transformer flow calls."""

from __future__ import annotations

import time
from typing import Any

import torch

from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    GENERATION_SCHEDULES,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_hidden import (
    full_generation_step_with_hidden,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import RealSimVLADCLDPolicy
from methods.latentloop.modules.simvla_generation_loop import (
    SimVLAGenerationHiddenUpdater,
    SimVLAGenerationLoop,
)


class RealSimVLAGenerationPolicy(RealSimVLADCLDPolicy):
    """Keep full VLM refreshes and reduce only action-transformer NFE."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        updater: SimVLAGenerationHiddenUpdater,
        n_g: int,
        device: torch.device,
        suite: str,
        task_id: int,
        trial_id: int,
        action_noise_seed_base: int,
        log_action_chunks: bool = False,
    ) -> None:
        if int(n_g) not in GENERATION_SCHEDULES or int(n_g) == 10:
            raise ValueError("Generation policy requires enabled reduced N_G in {2,3,5}")
        self.n_g = int(n_g)
        self.full_step_indices = GENERATION_SCHEDULES[self.n_g]
        self.generation_loop = SimVLAGenerationLoop(
            updater.eval(), model.transformer.action_decoder
        ).to(device)
        self.generation_loop.eval()
        super().__init__(
            model=model,
            processor=processor,
            dcld_core=None,
            mode="full",
            refresh_every=1,
            flow_steps=10,
            image_size=384,
            replan_steps=5,
            client_resize_size=224,
            device=device,
            suite=suite,
            row_name=f"generation_ng{self.n_g}",
            task_id=task_id,
            trial_id=trial_id,
            paired_action_noise=True,
            action_noise_seed_base=action_noise_seed_base,
            log_action_chunks=log_action_chunks,
        )

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _decode(
        self,
        condition: torch.Tensor,
        proprio: torch.Tensor,
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, int | None]:
        initial_noise, seed = self._paired_initial_noise(
            condition, proprio, policy_query_index
        )
        if initial_noise is None:
            raise RuntimeError("Generation evaluation requires explicit paired noise")
        normalized_proprio = self.action_adapter.normalize_proprio(proprio)

        def full_step(
            noisy_action: torch.Tensor, tau: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            output = full_generation_step_with_hidden(
                self.model.transformer,
                condition=condition,
                noisy_action=noisy_action,
                proprio=normalized_proprio,
                tau=tau,
                dt=-0.1,
            )
            return output.action_hidden, output.velocity

        self._sync()
        started = time.perf_counter()
        with torch.no_grad():
            trace = self.generation_loop(
                initial_noise,
                full_step=full_step,
                full_step_indices=self.full_step_indices,
                proprio=normalized_proprio,
                condition=condition,
                condition_valid_mask=None,
                condition_change_code=condition.new_zeros(
                    (condition.shape[0], self.generation_loop.updater.condition_code_dim)
                ),
            )
            action = self.action_adapter.action_space.postprocess(
                trace.final_noisy_action
            )
        self._sync()
        elapsed = (time.perf_counter() - started) * 1000.0
        self.metrics.latencies["action_transformer_ms"].append(elapsed)
        self.metrics.latencies["generation_loop_ms"].append(elapsed)
        self.metrics.counters["num_action_transformer_calls"] += self.n_g
        self.metrics.counters["num_action_transformer_decodes"] += 1
        self.metrics.counters["num_generation_decoder_only_steps"] += 10 - self.n_g
        return action, seed
