"""Mechanism controls for the synchronized SimVLA K_C=2, N_G=3 policy."""

from __future__ import annotations

import time
from typing import Any

import torch

from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    GENERATION_SCHEDULES,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_policy import (
    RealSimVLAGenerationPolicy,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.kc_frontier_contracts import (
    expected_call_counts,
    mechanical_control_row_name,
)
from architectures.simvla.adapters.latentloop.native_v0_policy import (
    RealSimVLANativeV0Policy,
)
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop


class SynchronizedMechanicalControlPolicy(RealSimVLANativeV0Policy):
    """Change only the skipped-query mechanism of K_C=2, N_G=3.

    Every row uses a full VLM condition and the learned N_G=3 decoder at the
    anchor query. The second query either holds the anchor condition, replays
    the unused half of its native H=10 chunk, repeats its last action, or runs
    the trained condition updater with an explicitly zero observation feature.
    """

    def __init__(
        self,
        *,
        control_mode: str,
        generation_updater: Any,
        **kwargs: Any,
    ) -> None:
        row_name = mechanical_control_row_name(control_mode)
        super().__init__(**kwargs)
        self.control_mode = str(control_mode)
        self.mode = row_name
        self.row_name = row_name
        self.k_c = 2
        self.refresh_every = 2
        self.n_g = 3
        self.full_step_indices = GENERATION_SCHEDULES[self.n_g]
        self.generation_loop = SimVLAGenerationLoop(
            generation_updater.eval(), self.model.transformer.action_decoder
        ).to(self.device)
        self.generation_loop.eval()
        self.metrics.latencies.setdefault("generation_loop_ms", [])

    def _decode(
        self,
        condition: torch.Tensor,
        proprio: torch.Tensor,
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, int | None]:
        return RealSimVLAGenerationPolicy._decode(
            self,
            condition,
            proprio,
            policy_query_index=policy_query_index,
        )

    def _zero_observation_update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        if self.cached_condition is None:
            raise RuntimeError("no-observation control requires an anchor condition")
        if self.condition_layout is None:
            raise RuntimeError("no-observation control requires the anchor token layout")
        zero_feature = self.cached_condition.new_zeros(
            (self.cached_condition.shape[0], self.native_v0.delta_dim)
        )
        self._sync()
        started = time.perf_counter()
        with torch.no_grad():
            update = self.native_v0.condition_updater(
                self.cached_condition,
                zero_feature,
                valid_mask=self.condition_layout.valid_mask,
                group_ids=self.condition_layout.group_ids,
                age=1,
            )
        self._sync()
        self.metrics.latencies.setdefault("condition_updater_ms", []).append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_condition_updater_calls"] += 1
        self.metrics.counters["num_no_observation_updates"] += 1
        action_chunk, seed = self._decode(
            update.condition,
            batch["proprio"],
            policy_query_index=policy_query_index,
        )
        self.cached_condition = update.condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action_chunk.detach()
        return update.condition, action_chunk, seed

    def _record_decoded_chunk(
        self,
        *,
        query: int,
        source: str,
        refreshed: bool,
        condition_updater_called: bool,
        seed: int | None,
        action_chunk: torch.Tensor,
    ) -> None:
        if not self.log_action_chunks:
            return
        if seed is None:
            raise RuntimeError("every decoded mechanical-control chunk needs paired noise")
        self.action_chunk_records.append(
            {
                "suite": self.suite,
                "task_id": self.task_id,
                "trial_id": self.trial_id,
                "episode_step_index": int(self.step_index),
                "policy_query_index": query,
                "row_name": self.row_name,
                "mode": self.mode,
                "k": self.k_c,
                "queue_mode": source,
                "refreshed": bool(refreshed),
                "full_vlm_called": bool(refreshed),
                "condition_updater_called": bool(condition_updater_called),
                "paired_action_noise": True,
                "action_noise_seed": int(seed),
                "action_chunk_shape": list(action_chunk.shape),
                "action_chunk": action_chunk.detach().cpu().float(),
            }
        )

    def _refill_action_queue(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        self.metrics.counters["num_policy_queries"] += 1
        query = int(self.query_index)
        age = query % self.k_c
        refreshed = age == 0
        condition_updater_called = False
        decoded = False
        seed: int | None = None

        if refreshed:
            _, action_chunk, seed = self._full_refresh(
                batch, policy_query_index=query
            )
            source = "full_refresh"
            decoded = True
        elif self.control_mode == "hold_condition":
            if self.cached_condition is None:
                raise RuntimeError("hold-condition control has no anchor condition")
            action_chunk, seed = self._decode(
                self.cached_condition,
                batch["proprio"],
                policy_query_index=query,
            )
            self.cached_action_chunk = action_chunk.detach()
            source = "hold_condition"
            decoded = True
        elif self.control_mode == "native_chunk_replay":
            if self.cached_action_chunk is None or self.cached_action_chunk.shape[1] != 10:
                raise RuntimeError("native chunk replay requires an exact H=10 anchor chunk")
            action_chunk = self.cached_action_chunk[:, 5:10]
            if action_chunk.shape[1] != 5:
                raise RuntimeError("native chunk replay must execute anchor actions 5:10")
            source = "native_action_chunk"
            self.metrics.counters["num_native_chunk_replay_queries"] += 1
        elif self.control_mode == "hold_action":
            if self.cached_executed_action is None:
                raise RuntimeError("hold-action control has no executed anchor action")
            action_chunk = self.cached_executed_action.reshape(1, 1, -1).repeat(
                1, 5, 1
            )
            source = "hold_action"
            self.metrics.counters["num_hold_action_queries"] += 1
        elif self.control_mode == "no_observation":
            _, action_chunk, seed = self._zero_observation_update(
                batch, policy_query_index=query
            )
            source = "no_observation"
            condition_updater_called = True
            decoded = True
        else:
            raise RuntimeError(f"unknown mechanical control: {self.control_mode}")

        self.action_queue.clear()
        for action in action_chunk[0, :5]:
            self.action_queue.append((action.detach(), source))
        if len(self.action_queue) != 5:
            raise RuntimeError("mechanical control must enqueue exactly R=5 actions")

        self.query_trace.append(
            {
                "policy_query_index": query,
                "age": age,
                "source": source,
                "full_vlm_called": refreshed,
                "condition_updater_called": condition_updater_called,
                "action_decoder_called": decoded,
                "action_noise_seed": seed,
                "action_horizon": 10,
                "execution_horizon": 5,
                "k_c": self.k_c,
                "n_g": self.n_g,
            }
        )
        if decoded:
            self._record_decoded_chunk(
                query=query,
                source=source,
                refreshed=refreshed,
                condition_updater_called=condition_updater_called,
                seed=seed,
                action_chunk=action_chunk,
            )

        self.query_index += 1
        expected = expected_call_counts(self.row_name, self.query_index)
        counters = self.metrics.counters
        observed = {
            "full_vlm_calls": int(counters["num_full_vlm_calls"]),
            "condition_updater_calls": int(counters["num_condition_updater_calls"]),
            "full_action_transformer_calls": int(
                counters["num_action_transformer_calls"]
            ),
            "generation_loop_updates": int(
                counters["num_generation_decoder_only_steps"]
            ),
            "integration_updates": int(counters["num_action_transformer_calls"])
            + int(counters["num_generation_decoder_only_steps"]),
        }
        if observed != expected:
            raise RuntimeError(
                f"mechanical-control counter drift: observed={observed} expected={expected}"
            )
        return {
            "refreshed": refreshed,
            "age": age,
            "queue_mode": source,
            "action_noise_seed": seed,
        }
