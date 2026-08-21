"""V1 direct/composed variable-time policy state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor

from methods.variable_time_latentloop.defect import normalized_latent_defect

from .prefix_kv_hook import PrefixEmbeddingState, PrefixKVState
from .transition_core import OpenPIKVLatentLoop, OpenPITransitionOutput


@dataclass
class VariableTimeAnchor:
    state: PrefixKVState
    query_index: int
    prefix_embeddings: Tensor
    action_history: list[Tensor]
    prefix_history: list[PrefixEmbeddingState]
    robot_history: list[Tensor]


@dataclass(frozen=True)
class VariableTimeOutput:
    sequential: OpenPITransitionOutput
    direct: OpenPITransitionOutput
    defect: Tensor
    input_provenance: dict[str, Any]


class VariableTimeStateManager:
    """Maintain one shared transition under sequential and anchor-direct paths."""

    def __init__(self, adapter: OpenPIKVLatentLoop, execution_horizon: int = 5) -> None:
        self.adapter = adapter
        self.execution_horizon = int(execution_horizon)
        self.anchor: VariableTimeAnchor | None = None
        self.sequential_state: PrefixKVState | None = None
        self.previous_prefix: Tensor | None = None
        self.sequential_anchor_query: int | None = None
        self.sequential_action_history: list[Tensor] = []

    def reset(self, full_state: PrefixKVState, query_index: int) -> None:
        self.anchor = VariableTimeAnchor(
            state=full_state,
            query_index=query_index,
            prefix_embeddings=full_state.embeddings,
            action_history=[],
            prefix_history=[],
            robot_history=[],
        )
        self.sequential_state = full_state
        self.previous_prefix = full_state.embeddings
        self.sequential_anchor_query = int(query_index)
        self.sequential_action_history = []

    def step(
        self,
        current_prefix: PrefixEmbeddingState,
        robot_state: Tensor,
        executed_actions: Tensor,
        query_index: int,
    ) -> VariableTimeOutput:
        if self.anchor is None or self.sequential_state is None or self.previous_prefix is None:
            raise RuntimeError("a full-prefix anchor must be installed before a variable-time step")
        if self.sequential_anchor_query is None:
            raise RuntimeError("sequential state has no causal anchor query")
        if executed_actions.ndim != 3 or executed_actions.shape[1] != self.execution_horizon:
            raise ValueError("V1/V2 must receive the actual ordered [B,R,7] executed subchunk")
        self.anchor.action_history.append(executed_actions)
        self.anchor.prefix_history.append(current_prefix)
        self.anchor.robot_history.append(robot_state)
        age = query_index - self.anchor.query_index
        if age not in {1, 2, 3}:
            raise ValueError("the K_q=4 variable-time family supports delta_q in {1,2,3}")
        sequential_age = query_index - self.sequential_anchor_query
        self.sequential_action_history.append(executed_actions)
        sequential = self.adapter(
            self.sequential_state,
            current_prefix,
            self.previous_prefix,
            executed_actions,
            robot_state,
            delta_q=1,
            delta_a=self.execution_horizon,
            full_refresh_age=age,
            executed_action_lengths=torch.full(
                (executed_actions.shape[0],),
                self.execution_horizon,
                device=executed_actions.device,
                dtype=torch.long,
            ),
        )
        all_actions = torch.cat(self.anchor.action_history, dim=1)
        direct = self.adapter(
            self.anchor.state,
            current_prefix,
            self.anchor.prefix_embeddings,
            all_actions,
            robot_state,
            delta_q=age,
            delta_a=age * self.execution_horizon,
            full_refresh_age=age,
            executed_action_lengths=torch.full(
                (all_actions.shape[0],),
                all_actions.shape[1],
                device=all_actions.device,
                dtype=torch.long,
            ),
            intermediate_prefix_embeddings=torch.stack(
                [item.embeddings for item in self.anchor.prefix_history], dim=1
            ),
            robot_state_history=torch.stack(self.anchor.robot_history, dim=1),
        )
        defect = normalized_latent_defect(sequential.encoded_state, direct.encoded_state)
        self.sequential_state = sequential.state
        self.previous_prefix = current_prefix.embeddings
        return VariableTimeOutput(
            sequential=sequential,
            direct=direct,
            defect=defect,
            input_provenance={
                "query_index": int(query_index),
                "full_anchor_query": int(self.anchor.query_index),
                "sequential_anchor_query": int(self.sequential_anchor_query),
                "direct_delta_q": int(age),
                "direct_delta_a": int(all_actions.shape[1]),
                "sequential_delta_q": 1,
                "sequential_delta_a": int(executed_actions.shape[1]),
                "sequential_age_since_reanchor": int(sequential_age),
                "executed_action_subchunks": len(self.anchor.action_history),
                "tensor_sources": {
                    "sequential_previous_state": "current recurrent predicted state",
                    "direct_previous_state": "latest actual full-prefix anchor",
                    "current_prefix_embeddings": "current causal observation",
                    "prefix_history": "ordered causal observations after latest full anchor",
                    "sequential_actions": "actual immediately preceding executed R-action subchunk",
                    "direct_actions": "ordered actual executed subchunks since latest full anchor",
                    "robot_state_history": "ordered causal robot states through current query",
                },
                "direct_inputs": [
                    "latest_actual_full_prefix_anchor",
                    "causal_prefix_embeddings_through_current_query",
                    "actual_executed_action_subchunks_through_current_query",
                    "causal_robot_states_through_current_query",
                ],
                "forbidden_inputs_present": {
                    "current_full_kv": False,
                    "future_observation": False,
                    "future_action": False,
                    "final_success_label": False,
                },
            },
        )

    def direct_reanchor(self, output: VariableTimeOutput, *, query_index: int) -> None:
        """Restart only the recurrent chain; retain the latest actual full anchor."""

        self.sequential_state = output.direct.state
        self.previous_prefix = output.direct.state.embeddings
        self.sequential_anchor_query = int(query_index)
        self.sequential_action_history = []
