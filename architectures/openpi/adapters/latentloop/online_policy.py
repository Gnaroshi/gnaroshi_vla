"""OpenPI-compatible external inference policy for LatentLoop rows."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Literal

import jax
import numpy as np
import torch
from torch import Tensor

from methods.variable_time_latentloop.decisions import RefreshDecision, RefreshStats
from methods.variable_time_latentloop.operation_counters_v2 import (
    OperationCountersV2,
    full_hook_query,
    latent_query,
    native_full_query,
)

from .dynamic_policy import BudgetedDynamicPolicy
from .policy_io import explicit_policy_noise, policy_noise_seed, postprocess_policy_actions, prepare_policy_observation
from .prefix_kv_hook import PrefixKVHook, PrefixKVState, tensor_sha256
from .recurrent_policy import OpenPILatentLoopPolicy
from .transition_core import OpenPIKVLatentLoop
from .variable_time_policy import VariableTimeStateManager


ServingMode = Literal["original", "k1", "hold", "latent_bridge", "v0", "v1", "v2"]


def _tree_sha256(tree: Any) -> str:
    digest = hashlib.sha256()
    leaves, _ = jax.tree.flatten(tree)
    for value in leaves:
        if isinstance(value, Tensor):
            digest.update(tensor_sha256(value).encode("ascii"))
        else:
            array = np.ascontiguousarray(np.asarray(value))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


class LatentLoopServingPolicy:
    """Stateful policy exposed through OpenPI's unchanged websocket server.

    The LIBERO client supplies an episode key and query index in the
    ``latentloop`` request field. All action noise is then derived from that key,
    independent of request order or trajectory length.
    """

    def __init__(
        self,
        base_policy: Any,
        adapter: OpenPIKVLatentLoop | None,
        *,
        mode: ServingMode,
        k_q: int = 4,
        flow_steps: int = 10,
        execution_horizon: int = 5,
        noise_seed_base: int = 7,
        dynamic_policy: BudgetedDynamicPolicy | None = None,
        k1_audit: bool = False,
    ) -> None:
        if mode in {"latent_bridge", "v0", "v1", "v2"} and adapter is None:
            raise ValueError(f"{mode} requires a trained adapter")
        if mode == "v2" and dynamic_policy is None:
            raise ValueError("v2 requires frozen dynamic calibration")
        self.base_policy = base_policy
        self.model = base_policy._model  # noqa: SLF001
        self.adapter = adapter
        self.mode = mode
        self.k_q = int(k_q)
        self.flow_steps = int(flow_steps)
        self.execution_horizon = int(execution_horizon)
        self.noise_seed_base = int(noise_seed_base)
        self.dynamic_policy = dynamic_policy
        self.k1_audit = bool(k1_audit)
        self.hook = PrefixKVHook(self.model)
        self.recurrent = OpenPILatentLoopPolicy(
            self.model,
            adapter if mode in {"latent_bridge", "v0"} else None,
            k_q=1 if mode == "k1" else k_q,
            num_flow_steps=flow_steps,
            execution_horizon=execution_horizon,
        ) if mode in {"k1", "latent_bridge", "v0"} else None
        self.variable = VariableTimeStateManager(adapter, execution_horizon) if mode in {"v1", "v2"} else None
        self.stats = RefreshStats()
        self.operation_counters = OperationCountersV2()
        self.query_index = 0
        self.full_state: PrefixKVState | None = None
        self.previous_executed_actions: Tensor | None = None
        self.episode_key: tuple[str, int, int] | None = None
        self._metadata = {
            **getattr(base_policy, "metadata", {}),
            "latentloop_mode": mode,
            "k_q": k_q,
            "k_a": k_q * execution_horizon,
            "execution_horizon": execution_horizon,
            "action_horizon": 10,
            "explicit_noise": True,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def reset(self, episode_key: tuple[str, int, int]) -> None:
        self.query_index = 0
        self.full_state = None
        self.previous_executed_actions = None
        self.episode_key = episode_key
        self.stats = RefreshStats()
        self.operation_counters = OperationCountersV2()
        if self.recurrent is not None:
            self.recurrent.reset()
        if self.dynamic_policy is not None:
            self.dynamic_policy.reset()
        self.variable = (
            VariableTimeStateManager(self.adapter, self.execution_horizon)
            if self.mode in {"v1", "v2"} and self.adapter is not None
            else None
        )

    def _original(self, observation: Any, noise: Tensor) -> Tensor:
        sampler = getattr(self.model.sample_actions, "_torchdynamo_orig_callable", self.model.sample_actions)
        return sampler(noise.device, observation, noise=noise, num_steps=self.flow_steps)

    def _full_from_hook(self, observation: Any, noise: Tensor) -> tuple[Tensor, PrefixKVState, dict[str, float]]:
        extraction = self.hook.extract(observation)
        actions, action_metrics = self.hook.sample_actions_from_state(
            extraction.state,
            extraction.robot_state,
            noise,
            num_steps=self.flow_steps,
        )
        return actions, extraction.state, {
            "prefix_embedding_ms": extraction.prefix_embedding_ms,
            "full_prefix_ms": extraction.full_prefix_ms,
            **action_metrics,
        }

    @torch.no_grad()
    def infer(self, raw_observation: dict[str, Any]) -> dict[str, Any]:
        raw = dict(raw_observation)
        request = dict(raw.pop("latentloop", {}))
        requested_path = str(request.get("policy_path", self.mode))
        if requested_path not in {self.mode, "original"}:
            raise ValueError(f"server mode {self.mode} cannot execute requested path {requested_path}")
        suite = str(request.get("suite", "unknown"))
        task_id = int(request.get("task_id", -1))
        episode_id = int(request.get("episode_id", -1))
        requested_query = int(request.get("query_index", 0))
        episode_key = (suite, task_id, episode_id)
        if bool(request.get("reset", False)) or episode_key != self.episode_key:
            self.reset(episode_key)
        if requested_query != self.query_index:
            raise RuntimeError(
                f"noncontiguous policy query: expected {self.query_index}, got {requested_query}"
            )

        observation, transformed = prepare_policy_observation(self.base_policy, raw)
        device = torch.device(self.base_policy._pytorch_device)  # noqa: SLF001
        seed = policy_noise_seed(self.noise_seed_base, suite, task_id, episode_id, self.query_index)
        noise = explicit_policy_noise(
            (1, self.model.config.action_horizon, self.model.config.action_dim),
            seed=seed,
            device=device,
        )
        started = time.perf_counter()
        metrics: dict[str, Any] = {}
        query_operations = OperationCountersV2()

        if requested_path == "original":
            normalized = self._original(observation, noise)
            decision = RefreshDecision.FULL_PREFIX
            self.stats.record(decision, self.execution_horizon)
            metrics["path"] = "original_full"
            metrics["age"] = 0
            query_operations = native_full_query(self.flow_steps)
        elif self.mode in {"k1", "latent_bridge", "v0"}:
            assert self.recurrent is not None
            result = self.recurrent.query(
                observation,
                noise,
                executed_actions=self.previous_executed_actions,
            )
            normalized = result.normalized_actions
            decision = result.decision
            metrics.update(result.metrics)
            self.stats = self.recurrent.stats
            query_operations = (
                native_full_query(self.flow_steps)
                if self.mode == "k1"
                else (
                    full_hook_query(self.flow_steps)
                    if decision is RefreshDecision.FULL_PREFIX
                    else latent_query(self.flow_steps, direct=False)
                )
            )
            if self.k1_audit:
                reference = self._original(observation, noise)
                query_operations.add(native_full_query(self.flow_steps))
                metrics.update(
                    {
                        "k1_reference_exact": bool(torch.equal(reference, normalized)),
                        "k1_reference_max_abs": float((reference - normalized).abs().max().item()),
                    }
                )
        elif self.mode == "hold":
            full_query = self.query_index % self.k_q == 0 or self.full_state is None
            if full_query:
                normalized, self.full_state, timing = self._full_from_hook(observation, noise)
                metrics.update(timing)
                decision = RefreshDecision.FULL_PREFIX
                query_operations = full_hook_query(self.flow_steps)
            else:
                assert self.full_state is not None
                _, _, _, _, robot_state = self.model._preprocess_observation(observation, train=False)  # noqa: SLF001
                normalized, timing = self.hook.sample_actions_from_state(
                    self.full_state, robot_state, noise, num_steps=self.flow_steps
                )
                metrics.update(timing)
                decision = RefreshDecision.SEQUENTIAL
                query_operations = OperationCountersV2(
                    action_expert_calls=1,
                    flow_iterations=self.flow_steps,
                    cache_rebuild_calls=1,
                )
            self.stats.record(decision, self.execution_horizon)
            metrics["path"] = "hold_stale_kv"
            metrics["age"] = self.query_index % self.k_q
        else:
            assert self.variable is not None
            forced = (
                self.dynamic_policy.forced_decision(self.query_index)
                if self.mode == "v2" and self.dynamic_policy is not None
                else None
            )
            full_age = (
                self.query_index - self.variable.anchor.query_index
                if self.variable.anchor is not None
                else 0
            )
            # The direct model is source-locked to delta_q in {1,2,3}. A
            # deterministic M_full safety refresh at delta_q=4 is therefore a
            # pre-transition safety gate; all in-domain V2 Level-2 decisions
            # still pay for both sequential and direct candidates.
            max_age_safety_full = (
                self.mode == "v2"
                and forced is RefreshDecision.FULL_PREFIX
                and full_age not in {1, 2, 3}
            )
            full_query = (
                self.query_index == 0
                or (self.mode == "v1" and self.query_index % self.k_q == 0)
                or max_age_safety_full
            )
            if full_query:
                normalized, full_state, timing = self._full_from_hook(observation, noise)
                self.full_state = full_state
                self.variable.reset(full_state, self.query_index)
                metrics.update(timing)
                decision = RefreshDecision.FULL_PREFIX
                metrics["age"] = 0
                query_operations = full_hook_query(self.flow_steps)
                if self.mode == "v2":
                    assert self.dynamic_policy is not None
                    self.dynamic_policy.record_full(self.query_index)
                    metrics["max_age_safety_pre_gate"] = bool(max_age_safety_full)
            else:
                if self.previous_executed_actions is None:
                    raise RuntimeError("bridge query is missing actually executed actions")
                current_prefix, robot_state, embedding_ms = self.hook.embed(observation)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                updater_started = time.perf_counter()
                updated = self.variable.step(
                    current_prefix,
                    robot_state,
                    self.previous_executed_actions,
                    self.query_index,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                metrics.update(
                    {
                        "prefix_embedding_ms": embedding_ms,
                        "updater_ms": (time.perf_counter() - updater_started) * 1000.0,
                        "defect_score": float(updated.defect.item()),
                        "age": self.query_index - self.variable.anchor.query_index,
                        "causal_input_provenance": updated.input_provenance,
                    }
                )
                query_operations = latent_query(self.flow_steps, direct=True)
                if self.mode == "v1":
                    selected_state = updated.sequential.state
                    decision = RefreshDecision.SEQUENTIAL
                else:
                    assert self.dynamic_policy is not None
                    dynamic = self.dynamic_policy.decide(
                        float(updated.defect.item()),
                        self.query_index,
                        forced=forced,
                    )
                    decision = dynamic.decision
                    metrics["predicted_action_error"] = dynamic.predicted_error
                    if decision is RefreshDecision.SEQUENTIAL:
                        selected_state = updated.sequential.state
                    elif decision is RefreshDecision.DIRECT_REANCHOR:
                        selected_state = updated.direct.state
                        self.variable.direct_reanchor(updated, query_index=self.query_index)
                        query_operations.direct_reanchor_events = 1
                    else:
                        extraction = self.hook.extract_from_embedding(
                            current_prefix, robot_state, embedding_ms=embedding_ms
                        )
                        selected_state = extraction.state
                        self.full_state = selected_state
                        self.variable.reset(selected_state, self.query_index)
                        metrics["full_prefix_ms"] = extraction.full_prefix_ms
                        query_operations.prefix_transformer_calls += 1
                        query_operations.full_prefix_refreshes += 1
                normalized, action_timing = self.hook.sample_actions_from_state(
                    selected_state, robot_state, noise, num_steps=self.flow_steps
                )
                metrics.update(action_timing)
            if self.mode == "v1":
                self.stats.record(decision, self.execution_horizon)
            else:
                assert self.dynamic_policy is not None
                self.stats = self.dynamic_policy.stats

        processed = postprocess_policy_actions(self.base_policy, transformed["state"], normalized)
        actions = np.asarray(processed["actions"])
        self.previous_executed_actions = torch.as_tensor(
            actions[: self.execution_horizon], device=device, dtype=torch.float32
        ).unsqueeze(0)
        self.operation_counters.add(query_operations)
        metrics.update(
            {
                "suite": suite,
                "task_id": task_id,
                "episode_id": episode_id,
                "query_index": self.query_index,
                "noise_seed": seed,
                "observation_hash": _tree_sha256(transformed),
                "normalized_action_hash": tensor_sha256(normalized),
                "postprocessed_action_hash": hashlib.sha256(actions.tobytes()).hexdigest(),
                "decision": decision.name.lower(),
                **query_operations.to_dict(),
                "cumulative_operation_counters": self.operation_counters.to_dict(),
                "infer_ms": (time.perf_counter() - started) * 1000.0,
                "peak_vram_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
                **self.stats.to_dict(),
            }
        )
        self.query_index += 1
        return {"actions": actions, "latentloop_metrics": metrics}
