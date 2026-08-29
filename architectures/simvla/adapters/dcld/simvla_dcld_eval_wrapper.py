"""Stateful SimVLA+DCLD evaluation wrapper skeleton."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from methods.dcld.eval import LatencyAccumulator, PeriodicSkipScheduler
from methods.dcld.modules import DCLDCore

from .simvla_action_adapter import SimVLAActionAdapter
from .simvla_condition_adapter import SimVLAConditionAdapter
from .simvla_delta_obs_adapter import SimVLADeltaObsAdapter


@dataclass
class SimVLADCLDStepOutput:
    action: torch.Tensor
    condition: torch.Tensor
    refreshed: bool
    age: int
    info: dict[str, Any]


class SimVLADCLDEvalWrapper:
    """Wrap a loaded SimVLA model with DCLD condition updates."""

    def __init__(
        self,
        model: Any,
        dcld_core: DCLDCore,
        *,
        refresh_every: int = 4,
        mode: str = "real_delta",
        action_steps: int = 10,
    ) -> None:
        self.model = model
        self.dcld_core = dcld_core
        self.mode = mode
        self.action_steps = int(action_steps)
        self.scheduler = PeriodicSkipScheduler(refresh_every=refresh_every)
        self.action_adapter = SimVLAActionAdapter(model)
        self.condition_adapter = SimVLAConditionAdapter(model, self.action_adapter)
        self.delta_adapter = SimVLADeltaObsAdapter()
        self.latency = LatencyAccumulator()
        self.counters: Counter[str] = Counter()
        self.cached_condition: torch.Tensor | None = None
        self.cached_batch: dict[str, torch.Tensor] | None = None
        self.cached_action: torch.Tensor | None = None

    def reset_cache(self) -> None:
        self.cached_condition = None
        self.cached_batch = None
        self.cached_action = None

    def reset_counters(self) -> None:
        self.counters.clear()

    def counter_summary(self) -> dict[str, int]:
        return dict(self.counters)

    def _decode_action(self, condition: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        action_out = self.action_adapter.decode_action_from_condition(
            condition,
            proprio,
            steps=self.action_steps,
            return_debug=True,
        )
        self.counters["num_action_decoder_calls"] += 1
        self.counters["num_action_transformer_calls"] += int(action_out.debug.get("iterations", 0))
        return action_out.action

    def step(self, batch: dict[str, torch.Tensor], step_index: int) -> SimVLADCLDStepOutput:
        self.counters["num_policy_steps"] += 1
        eval_mode = self.mode
        dcld_mode = "real_delta" if eval_mode == "stepwise_dcld" else eval_mode
        dcld_modes = {"real_delta", "no_delta", "shuffled_delta", "proprio_only", "image_only"}
        decision = self.scheduler.decision(step_index)
        need_refresh = decision.refresh or self.cached_condition is None or self.cached_batch is None

        if eval_mode == "full":
            need_refresh = True

        if need_refresh:
            with self.latency.time("full_condition"):
                condition = self.condition_adapter.encode_condition(
                    input_ids=batch["input_ids"],
                    image_input=batch["image_input"],
                    image_mask=batch["image_mask"],
                )
            self.cached_condition = condition.detach()
            self.cached_batch = self.delta_adapter.clone_cache_batch(batch)
            refreshed = True
            age = 0
            dcld_debug = {}
            self.counters["num_full_vlm_calls"] += 1
            with self.latency.time("action_decode"):
                action = self._decode_action(condition, batch["proprio"])
            self.cached_action = action.detach()
        elif eval_mode == "hold_action":
            if self.cached_action is None:
                raise RuntimeError("hold_action requires a cached action")
            condition = self.cached_condition
            action = self.cached_action
            refreshed = False
            age = decision.age
            dcld_debug = {}
            self.counters["num_hold_action_steps"] += 1
        elif eval_mode == "native_action_chunk":
            if self.cached_action is None:
                raise RuntimeError("native_action_chunk requires a cached action chunk")
            condition = self.cached_condition
            action = self.cached_action
            refreshed = False
            age = decision.age
            dcld_debug = {}
            self.counters["num_native_chunk_steps"] += 1
        elif eval_mode == "hold_condition":
            condition = self.cached_condition
            refreshed = False
            age = decision.age
            dcld_debug = {}
            self.counters["num_hold_condition_steps"] += 1
            with self.latency.time("action_decode"):
                action = self._decode_action(condition, batch["proprio"])
        else:
            if dcld_mode not in dcld_modes:
                raise ValueError(f"Unknown SimVLA DCLD eval mode: {eval_mode}")
            delta_obs = self.delta_adapter.make_delta_observation(
                self.cached_batch,
                batch,
                age=decision.age,
            )
            with self.latency.time("dcld_update"):
                update = self.dcld_core.update_latent(
                    self.cached_condition,
                    delta_obs,
                    dt=1.0,
                    age=float(decision.age),
                    mode=dcld_mode,
                )
            condition = update.latent
            self.cached_condition = condition.detach()
            refreshed = False
            age = decision.age
            dcld_debug = update.debug
            self.counters["num_dcld_updates"] += 1
            if dcld_mode == "no_delta":
                self.counters["num_no_delta_steps"] += 1
            else:
                self.counters["num_fast_encoder_calls"] += 1
            with self.latency.time("action_decode"):
                action = self._decode_action(condition, batch["proprio"])
            self.cached_action = action.detach()

        return SimVLADCLDStepOutput(
            action=action,
            condition=condition,
            refreshed=refreshed,
            age=age,
            info={
                "mode": self.mode,
                "decision": decision,
                "dcld_debug": dcld_debug,
                "counters": self.counter_summary(),
            },
        )
