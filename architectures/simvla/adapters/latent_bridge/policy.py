"""Stateful SimVLA policy using the official Latent Bridge recurrence."""

from __future__ import annotations

import time
from typing import Any

import torch

from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    RealSimVLADCLDPolicy,
)

from .condition_hook import SimVLAConditionWithStableHook
from .model import SimVLALatentBridge


class RealSimVLALatentBridgePolicy(RealSimVLADCLDPolicy):
    """Refresh the full SimVLA condition periodically and bridge between anchors.

    The action transformer still produces a fresh H=10 chunk at every policy
    query, and the environment still executes R=5 actions from that chunk.
    """

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        bridge: SimVLALatentBridge,
        refresh_every: int,
        flow_steps: int,
        image_size: int,
        replan_steps: int,
        client_resize_size: int,
        device: torch.device,
        suite: str = "",
        row_name: str = "latent_bridge",
        task_id: int = -1,
        trial_id: int = -1,
        paired_action_noise: bool = True,
        action_noise_seed_base: int = 20260901,
        log_action_chunks: bool = False,
        collect_dagger_teacher: bool = False,
    ) -> None:
        self.bridge = bridge.eval()
        self.collect_dagger_teacher = bool(collect_dagger_teacher)
        self.cached_stable: torch.Tensor | None = None
        self.dagger_transitions: list[dict[str, Any]] = []
        super().__init__(
            model=model,
            processor=processor,
            dcld_core=None,
            mode="latent_bridge",
            refresh_every=refresh_every,
            flow_steps=flow_steps,
            image_size=image_size,
            replan_steps=replan_steps,
            client_resize_size=client_resize_size,
            device=device,
            suite=suite,
            row_name=row_name,
            task_id=task_id,
            trial_id=trial_id,
            paired_action_noise=paired_action_noise,
            action_noise_seed_base=action_noise_seed_base,
            log_action_chunks=log_action_chunks,
        )
        if replan_steps != 5:
            raise ValueError("SimVLA Latent Bridge comparison requires original R=5")
        self.condition_hook = SimVLAConditionWithStableHook(
            model, stable_layer_index=bridge.config.stable_layer_index
        )

    def reset(self) -> None:
        super().reset()
        self.cached_stable = None
        self.dagger_transitions = []

    def close(self) -> None:
        self.condition_hook.close()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _full_refresh(
        self,
        batch: dict[str, torch.Tensor],
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        self._sync()
        started = time.perf_counter()
        extracted = self.condition_hook.encode(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
        self._sync()
        self.metrics.latencies["VLM_encoder_ms"].append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_full_vlm_calls"] += 1
        condition = extracted.condition
        action_chunk, seed = self._decode(
            condition, batch["proprio"], policy_query_index=policy_query_index
        )
        self.cached_condition = condition.detach()
        self.cached_stable = extracted.stable.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action_chunk.detach()
        return condition, action_chunk, seed

    def _decode(
        self,
        condition: torch.Tensor,
        proprio: torch.Tensor,
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, int | None]:
        self._sync()
        started = time.perf_counter()
        action, seed = super()._decode(
            condition, proprio, policy_query_index=policy_query_index
        )
        self._sync()
        self.metrics.latencies["action_transformer_ms"][-1] = (
            time.perf_counter() - started
        ) * 1000.0
        self.metrics.counters["num_action_transformer_decodes"] += 1
        return action, seed

    def _dcld_condition(
        self,
        batch: dict[str, torch.Tensor],
        dcld_mode: str,
        age: int,
    ) -> torch.Tensor:
        if dcld_mode != "latent_bridge":
            raise RuntimeError(f"unexpected Latent Bridge mode: {dcld_mode}")
        if self.cached_condition is None or self.cached_stable is None:
            raise RuntimeError("Latent Bridge update requires a full condition anchor")
        if self.cached_action_chunk is None:
            raise RuntimeError("Latent Bridge update requires the previous predicted action")
        previous_condition = self.cached_condition
        previous_action = self.cached_action_chunk[:, 0, :]
        teacher = None
        if self.collect_dagger_teacher:
            with torch.inference_mode():
                teacher = self.condition_hook.encode(
                    input_ids=batch["input_ids"],
                    image_input=batch["image_input"],
                    image_mask=batch["image_mask"],
                ).condition.detach()
            self.metrics.counters["num_dagger_teacher_vlm_calls"] += 1
        bridge_dtype = next(self.bridge.parameters()).dtype
        self._sync()
        started = time.perf_counter()
        with torch.inference_mode():
            bridge_condition = previous_condition
            bridge_stable = self.cached_stable
            if self.bridge.config.token_mode == "image_only":
                token_count = self.bridge.config.image_token_count
                bridge_condition = bridge_condition[:, :token_count]
                bridge_stable = bridge_stable[:, :token_count]
            predicted = self.bridge.predict_next(
                bridge_condition.to(dtype=bridge_dtype),
                bridge_stable.to(dtype=bridge_dtype),
                batch["proprio"].to(dtype=bridge_dtype),
                previous_action.to(dtype=bridge_dtype),
            ).to(dtype=previous_condition.dtype)
            if self.bridge.config.token_mode == "image_only":
                next_condition = previous_condition.clone()
                next_condition[:, : self.bridge.config.image_token_count] = predicted
            else:
                next_condition = predicted
        self._sync()
        self.metrics.latencies.setdefault("condition_updater_ms", []).append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_condition_updater_calls"] += 1
        self.metrics.counters["num_latent_bridge_calls"] += 1
        if teacher is not None:
            self.dagger_transitions.append(
                {
                    "condition_input": previous_condition[0].detach().to("cpu", torch.bfloat16),
                    "condition_target": teacher[0].detach().to("cpu", torch.bfloat16),
                    "stable_anchor": self.cached_stable[0].detach().to("cpu", torch.bfloat16),
                    "state": batch["proprio"][0].detach().to("cpu", torch.float32),
                    "previous_action": previous_action[0].detach().to("cpu", torch.float32),
                    "age": int(age),
                    "policy_query_index": int(self.query_index),
                }
            )
        self.cached_condition = next_condition.detach()
        return next_condition
