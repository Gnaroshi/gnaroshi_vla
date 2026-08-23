"""Official-queue native H=10/R=5 policy for corrected SimVLA V0 K=4."""

from __future__ import annotations

import time
from typing import Any

import torch

from architectures.simvla.adapters.latentloop.native_v0_condition_hook import (
    ConditionTokenLayout,
    build_condition_token_layout,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import RealSimVLADCLDPolicy
from methods.latentloop.modules.native_simvla_v0 import (
    NativeSimVLAV0,
    NativeV0ObservationPair,
)


class RealSimVLANativeV0Policy(RealSimVLADCLDPolicy):
    """Replace only fused-condition refreshes q1/q2/q3 with recursive V0."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        adapter: NativeSimVLAV0,
        checkpoint_id: str,
        device: torch.device,
        suite: str,
        task_id: int,
        trial_id: int,
        action_noise_seed_base: int,
        client_resize_size: int = 224,
        image_size: int = 384,
        flow_steps: int = 10,
        log_action_chunks: bool = False,
    ) -> None:
        if adapter.rank_dim != 64:
            raise ValueError("scientific native V0 policy requires primary rank 64")
        self.native_v0 = adapter
        self.checkpoint_id = str(checkpoint_id)
        self.condition_layout: ConditionTokenLayout | None = None
        self.query_trace: list[dict[str, Any]] = []
        super().__init__(
            model=model,
            processor=processor,
            dcld_core=None,
            mode="native_v0_k4",
            refresh_every=4,
            flow_steps=flow_steps,
            image_size=image_size,
            replan_steps=5,
            client_resize_size=client_resize_size,
            device=device,
            suite=suite,
            row_name="native_v0_k4",
            task_id=task_id,
            trial_id=trial_id,
            paired_action_noise=True,
            action_noise_seed_base=action_noise_seed_base,
            log_action_chunks=log_action_chunks,
        )

    def reset(self) -> None:
        super().reset()
        self.condition_layout = None
        self.query_trace = []

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
        condition = self.condition_adapter.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
        self._sync()
        self.metrics.latencies["VLM_encoder_ms"].append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_full_vlm_calls"] += 1
        action, seed = self._decode(
            condition,
            batch["proprio"],
            policy_query_index=policy_query_index,
        )
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action.detach()
        tokenizer = self.processor.tokenizer
        self.condition_layout = build_condition_token_layout(
            condition=condition,
            image_mask=batch["image_mask"],
            input_ids=batch["input_ids"],
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
            special_token_ids=getattr(tokenizer, "all_special_ids", ()),
        )
        return condition, action, seed

    def _decode(
        self,
        condition: torch.Tensor,
        proprio: torch.Tensor,
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, int | None]:
        self._sync()
        started = time.perf_counter()
        initial_noise, seed = self._paired_initial_noise(
            condition,
            proprio,
            policy_query_index,
        )
        decoded = self.action_adapter.decode_action_from_condition(
            condition,
            proprio,
            steps=self.flow_steps,
            initial_noise=initial_noise,
            return_debug=True,
        )
        self._sync()
        self.metrics.latencies["action_transformer_ms"].append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_action_transformer_calls"] += int(
            decoded.debug.get("iterations", 0)
        )
        self.metrics.counters["num_action_transformer_decodes"] += 1
        return decoded.action, seed

    def _v0_update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        age: int,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        if self.cached_condition is None or self.cached_raw_rgb is None or self.cached_proprio is None:
            raise RuntimeError("V0 update requires the preceding condition and query observation")
        if self.condition_layout is None:
            raise RuntimeError("V0 token layout was not established at the full refresh")
        if age not in {1, 2, 3}:
            raise ValueError("native K4 V0 update age must be 1, 2, or 3")
        pair = NativeV0ObservationPair(
            previous_images=self.cached_raw_rgb,
            current_images=batch["raw_rgb"],
            previous_proprio=self.cached_proprio,
            current_proprio=batch["proprio"],
        )
        self._sync()
        started = time.perf_counter()
        with torch.no_grad():
            update = self.native_v0.update_once(
                self.cached_condition,
                pair,
                valid_mask=self.condition_layout.valid_mask,
                group_ids=self.condition_layout.group_ids,
                age=age,
            )
        self._sync()
        self.metrics.latencies.setdefault("condition_updater_ms", []).append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_condition_updater_calls"] += 1
        self.metrics.counters["num_observation_encoder_calls"] += 1
        action, seed = self._decode(
            update.condition,
            batch["proprio"],
            policy_query_index=policy_query_index,
        )
        self.cached_condition = update.condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action.detach()
        return update.condition, action, seed

    def _refill_action_queue(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self.metrics.counters["num_policy_queries"] += 1
        query = int(self.query_index)
        age = query % 4
        if age == 0:
            condition, action_chunk, seed = self._full_refresh(
                batch,
                policy_query_index=query,
            )
            source = "full_refresh"
            refreshed = True
        else:
            condition, action_chunk, seed = self._v0_update(
                batch,
                age=age,
                policy_query_index=query,
            )
            source = "native_v0"
            refreshed = False
        self.action_queue.clear()
        for action in action_chunk[0, :5]:
            self.action_queue.append((action.detach(), source))
        self.query_trace.append(
            {
                "policy_query_index": query,
                "age": age,
                "source": source,
                "full_vlm_called": refreshed,
                "condition_updater_called": not refreshed,
                "action_noise_seed": seed,
                "action_horizon": 10,
                "execution_horizon": 5,
            }
        )
        self.query_index += 1
        return {
            "refreshed": refreshed,
            "age": age,
            "queue_mode": source,
            "action_noise_seed": seed,
        }
