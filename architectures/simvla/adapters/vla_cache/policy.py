"""SimVLA policy using actual VLA-Cache decoder reuse."""

from __future__ import annotations

import time
from typing import Any

from architectures.simvla.wrappers.dcld_eval.rollout_runner import RealSimVLADCLDPolicy

from .official_contract import VLACacheConfig
from .smolvlm_runtime import SimVLAVLACacheBackbone


class VLACacheSimVLAPolicy(RealSimVLADCLDPolicy):
    """Training-free VLA-Cache baseline with unchanged H=10/R=5 control."""

    def __init__(self, *, enable_reuse: bool = True, **kwargs: Any) -> None:
        self.enable_reuse = bool(enable_reuse)
        mode = "vla_cache" if self.enable_reuse else "vla_cache_full"
        fixed = {
            "flow_steps": 10,
            "image_size": 384,
            "replan_steps": 5,
            "client_resize_size": 224,
        }
        for name, expected in fixed.items():
            observed = kwargs.pop(name, expected)
            if int(observed) != expected:
                raise ValueError(f"VLA-Cache requires {name}={expected}, got {observed}")
        row_name = kwargs.pop("row_name", f"{mode}_kc1_ng10")
        paired_action_noise = bool(kwargs.pop("paired_action_noise", True))
        super().__init__(
            dcld_core=None,
            mode=mode,
            refresh_every=1,
            **fixed,
            row_name=row_name,
            paired_action_noise=paired_action_noise,
            **kwargs,
        )
        self.vla_cache = SimVLAVLACacheBackbone(
            self.model,
            VLACacheConfig(),
            enable_reuse=self.enable_reuse,
        )
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.query_trace: list[dict[str, Any]] = []
        if hasattr(self, "vla_cache"):
            self.vla_cache.reset()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)

    def _decode(self, condition, proprio, *, policy_query_index):
        self._sync()
        started = time.perf_counter()
        action, seed = super()._decode(
            condition,
            proprio,
            policy_query_index=policy_query_index,
        )
        self._sync()
        self.metrics.latencies["action_transformer_ms"][-1] = (
            time.perf_counter() - started
        ) * 1000.0
        return action, seed

    def _refill_action_queue(self, batch):
        self.metrics.counters["num_policy_queries"] += 1
        query = int(self.query_index)
        self._sync()
        started = time.perf_counter()
        condition = self.vla_cache.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
        self._sync()
        self.metrics.latencies["VLM_encoder_ms"].append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_vlm_queries"] += 1
        decoder = self.vla_cache.last_report["decoder"]
        if decoder["first_query"]:
            self.metrics.counters["num_vla_cache_anchor_queries"] += 1
        else:
            self.metrics.counters["num_vla_cache_nonanchor_queries"] += 1
        if decoder["actual_kv_reuse"]:
            self.metrics.counters["num_actual_kv_reuse_queries"] += 1
        self.metrics.counters["computed_text_token_layers"] += int(
            decoder["computed_token_layers"]
        )
        self.metrics.counters["skipped_text_token_layers"] += int(
            decoder["skipped_token_layers"]
        )

        action_chunk, seed = self._decode(
            condition,
            batch["proprio"],
            policy_query_index=query,
        )
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action_chunk.detach()
        self.action_queue.clear()
        for action in action_chunk[0, :5]:
            self.action_queue.append((action.detach(), self.mode))
        self.query_trace.append(
            {
                "policy_query_index": query,
                "source": self.mode,
                "action_noise_seed": seed,
                "action_horizon": 10,
                "execution_horizon": 5,
                "generation_full_evaluations": 10,
                "vla_cache": self.vla_cache.last_report,
            }
        )
        self.query_index += 1
        return {
            "refreshed": True,
            "age": 0,
            "queue_mode": self.mode,
            "action_noise_seed": seed,
        }


# Backward-compatible name used by the real-world deployment adapter.
VLACacheSimVLARealPolicy = VLACacheSimVLAPolicy
