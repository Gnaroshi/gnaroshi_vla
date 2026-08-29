"""Fixed-interval V0 runtime with an exact K_q=1 bypass."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
from torch import Tensor

from methods.variable_time_latentloop.decisions import RefreshDecision, RefreshStats

from .prefix_kv_hook import PrefixKVHook, PrefixKVState
from .transition_core import OpenPIKVLatentLoop


@dataclass
class LatentLoopRuntimeState:
    query_index: int = 0
    full_refresh_query: int = -1
    prefix_state: PrefixKVState | None = None
    anchor_state: PrefixKVState | None = None
    previous_prefix_embeddings: Tensor | None = None
    updater_calls: int = 0

    @property
    def age(self) -> int:
        if self.full_refresh_query < 0:
            return 0
        return self.query_index - self.full_refresh_query


@dataclass(frozen=True)
class PolicyQueryOutput:
    normalized_actions: Tensor
    prefix_state: PrefixKVState | None
    decision: RefreshDecision
    metrics: dict[str, float | int | str]


class OpenPILatentLoopPolicy:
    """Use the frozen action expert with full or recurrently predicted prefix KV.

    At ``K_q=1`` this class calls the original full sampler immediately. It does
    not extract KV and does not call the updater, which makes the bypass auditable.
    """

    def __init__(
        self,
        model: Any,
        adapter: OpenPIKVLatentLoop | None,
        *,
        k_q: int = 4,
        num_flow_steps: int = 10,
        execution_horizon: int = 5,
    ) -> None:
        if k_q < 1:
            raise ValueError("k_q must be positive")
        if model.config.action_horizon != 10:
            raise ValueError("pinned pi0.5 action horizon must be 10")
        if execution_horizon != 5:
            raise ValueError("pinned LIBERO execution horizon must be 5")
        if k_q > 1 and adapter is None:
            raise ValueError("an adapter is required when k_q > 1")
        self.model = model
        self.adapter = adapter
        self.k_q = int(k_q)
        self.num_flow_steps = int(num_flow_steps)
        self.execution_horizon = int(execution_horizon)
        self.hook = PrefixKVHook(model)
        self.runtime = LatentLoopRuntimeState()
        self.stats = RefreshStats()

    def reset(self) -> None:
        self.runtime = LatentLoopRuntimeState()
        self.stats = RefreshStats()

    def _full_sample(self, observation: Any, noise: Tensor) -> Tensor:
        sampler = getattr(self.model.sample_actions, "_torchdynamo_orig_callable", self.model.sample_actions)
        return sampler(noise.device, observation, noise=noise, num_steps=self.num_flow_steps)

    @torch.no_grad()
    def query(
        self,
        observation: Any,
        noise: Tensor,
        *,
        executed_actions: Tensor | None = None,
    ) -> PolicyQueryOutput:
        if self.k_q == 1:
            started = time.perf_counter()
            actions = self._full_sample(observation, noise)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.stats.record(RefreshDecision.FULL_PREFIX, self.execution_horizon)
            self.runtime.query_index += 1
            return PolicyQueryOutput(
                normalized_actions=actions,
                prefix_state=None,
                decision=RefreshDecision.FULL_PREFIX,
                metrics={
                    "path": "k1_exact_bypass",
                    "query_index": self.runtime.query_index - 1,
                    "updater_calls": 0,
                    "full_infer_ms": elapsed_ms,
                },
            )

        if executed_actions is None and self.runtime.query_index > 0:
            raise ValueError("bridge queries require the actions actually executed since the previous query")
        full_query = self.runtime.query_index % self.k_q == 0 or self.runtime.prefix_state is None
        if full_query:
            extraction = self.hook.extract(observation)
            actions, action_metrics = self.hook.sample_actions_from_state(
                extraction.state,
                extraction.robot_state,
                noise,
                num_steps=self.num_flow_steps,
            )
            self.runtime.prefix_state = extraction.state
            self.runtime.anchor_state = extraction.state
            self.runtime.previous_prefix_embeddings = extraction.state.embeddings
            self.runtime.full_refresh_query = self.runtime.query_index
            decision = RefreshDecision.FULL_PREFIX
            metrics: dict[str, float | int | str] = {
                "path": "full_prefix",
                "prefix_embedding_ms": extraction.prefix_embedding_ms,
                "full_prefix_ms": extraction.full_prefix_ms,
                **action_metrics,
            }
        else:
            assert self.adapter is not None
            assert self.runtime.prefix_state is not None
            assert self.runtime.previous_prefix_embeddings is not None
            current_prefix, robot_state, embedding_ms = self.hook.embed(observation)
            if executed_actions is None:
                raise AssertionError("executed actions were validated above")
            if executed_actions.ndim != 3 or executed_actions.shape[1] != self.execution_horizon:
                raise ValueError("V0 requires the actual ordered [B,R,7] executed subchunk")
            started = time.perf_counter()
            update = self.adapter(
                self.runtime.prefix_state,
                current_prefix,
                self.runtime.previous_prefix_embeddings,
                executed_actions,
                robot_state,
                delta_q=1,
                delta_a=self.execution_horizon,
                full_refresh_age=self.runtime.age,
                executed_action_lengths=torch.full(
                    (executed_actions.shape[0],),
                    self.execution_horizon,
                    device=executed_actions.device,
                    dtype=torch.long,
                ),
            )
            if noise.device.type == "cuda":
                torch.cuda.synchronize(noise.device)
            updater_ms = (time.perf_counter() - started) * 1000.0
            actions, action_metrics = self.hook.sample_actions_from_state(
                update.state,
                robot_state,
                noise,
                num_steps=self.num_flow_steps,
            )
            self.runtime.prefix_state = update.state
            self.runtime.previous_prefix_embeddings = current_prefix.embeddings
            self.runtime.updater_calls += 1
            decision = RefreshDecision.SEQUENTIAL
            metrics = {
                "path": "v0_recurrent",
                "prefix_embedding_ms": embedding_ms,
                "updater_ms": updater_ms,
                "gate_mean": float(update.gate.mean().item()),
                **action_metrics,
            }

        query_index = self.runtime.query_index
        age_for_query = self.runtime.age
        self.stats.record(decision, self.execution_horizon)
        self.runtime.query_index += 1
        metrics.update(
            {
                "query_index": query_index,
                "age": age_for_query,
                "updater_calls": self.runtime.updater_calls,
                "k_q": self.k_q,
                "k_a": self.k_q * self.execution_horizon,
            }
        )
        return PolicyQueryOutput(
            normalized_actions=actions,
            prefix_state=self.runtime.prefix_state,
            decision=decision,
            metrics=metrics,
        )
