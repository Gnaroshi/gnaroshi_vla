"""Unambiguous operation counters for pi0.5 efficiency claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class OperationCountersV2:
    vision_encoder_calls: int = 0
    prefix_embedding_calls: int = 0
    prefix_transformer_calls: int = 0
    latentloop_sequential_calls: int = 0
    latentloop_direct_calls: int = 0
    direct_reanchor_events: int = 0
    full_prefix_refreshes: int = 0
    action_expert_calls: int = 0
    flow_iterations: int = 0
    cache_rebuild_calls: int = 0

    def add(self, other: "OperationCountersV2") -> None:
        for name, value in asdict(other).items():
            setattr(self, name, int(getattr(self, name)) + int(value))

    def to_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


def full_hook_query(flow_steps: int) -> OperationCountersV2:
    return OperationCountersV2(
        vision_encoder_calls=1,
        prefix_embedding_calls=1,
        prefix_transformer_calls=1,
        full_prefix_refreshes=1,
        action_expert_calls=1,
        flow_iterations=int(flow_steps),
        cache_rebuild_calls=1,
    )


def native_full_query(flow_steps: int) -> OperationCountersV2:
    return OperationCountersV2(
        vision_encoder_calls=1,
        prefix_embedding_calls=1,
        prefix_transformer_calls=1,
        full_prefix_refreshes=1,
        action_expert_calls=1,
        flow_iterations=int(flow_steps),
    )


def latent_query(flow_steps: int, *, direct: bool, reanchor: bool = False) -> OperationCountersV2:
    return OperationCountersV2(
        vision_encoder_calls=1,
        prefix_embedding_calls=1,
        latentloop_sequential_calls=1,
        latentloop_direct_calls=int(direct),
        direct_reanchor_events=int(reanchor),
        action_expert_calls=1,
        flow_iterations=int(flow_steps),
        cache_rebuild_calls=1,
    )
