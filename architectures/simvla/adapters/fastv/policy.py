"""Original-protocol SimVLA policies for paired baseline/FastV evaluation."""

from __future__ import annotations

import time
from typing import Any

import torch

from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    RealSimVLADCLDPolicy,
)

from .encoder import FastVConditionEncoder, FastVForwardConfig


class SynchronizedBaselinePolicy(RealSimVLADCLDPolicy):
    """Unmodified full SimVLA policy with synchronized latency measurement."""

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
            condition, batch["proprio"], policy_query_index=policy_query_index
        )
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action.detach()
        return condition, action, seed


class FastVPolicy(SynchronizedBaselinePolicy):
    """SimVLA policy whose VLM text stack uses physical FastV pruning."""

    def __init__(self, *, fastv_config: FastVForwardConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fastv_encoder = FastVConditionEncoder(self.model, fastv_config)
        self.fastv_debug_records: list[dict[str, Any]] = []

    def _full_refresh(
        self,
        batch: dict[str, torch.Tensor],
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        self._sync()
        started = time.perf_counter()
        condition = self.fastv_encoder.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
        self._sync()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.latencies["VLM_encoder_ms"].append(elapsed_ms)
        self.metrics.latencies.setdefault("FastV_VLM_ms", []).append(elapsed_ms)
        self.metrics.counters["num_full_vlm_calls"] += 1
        self.metrics.counters["num_fastv_calls"] += 1
        debug = self.fastv_encoder.last_debug
        if debug is None:
            raise RuntimeError("FastV condition encoder did not emit debug metadata")
        self.fastv_debug_records.append({"policy_query_index": policy_query_index, **debug})
        self.metrics.counters["num_visual_tokens_before"] += int(
            debug["visual_tokens_before"]
        )
        self.metrics.counters["num_visual_tokens_kept"] += int(
            debug["visual_tokens_kept"]
        )
        self.metrics.counters["num_visual_tokens_pruned"] += int(
            debug["visual_tokens_pruned"]
        )
        action, seed = self._decode(
            condition, batch["proprio"], policy_query_index=policy_query_index
        )
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action.detach()
        return condition, action, seed
