"""Evaluation contracts for the SimVLA adaptation of VLA-Cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .official_contract import VLACacheConfig


@dataclass(frozen=True)
class EvaluationRow:
    name: str
    enable_reuse: bool
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ROWS = {
    "vla_cache_full": EvaluationRow(
        name="vla_cache_full",
        enable_reuse=False,
        description=(
            "Matched eager-attention control. It uses the VLA-Cache decoder "
            "backend but recomputes every token and performs no KV reuse."
        ),
    ),
    "vla_cache": EvaluationRow(
        name="vla_cache",
        enable_reuse=True,
        description=(
            "Training-free VLA-Cache adaptation with actual decoder-token "
            "skipping and retained K/V at selected visual-token positions."
        ),
    ),
}


def evaluation_row(name: str) -> EvaluationRow:
    try:
        return ROWS[name]
    except KeyError as exc:
        raise ValueError(f"unsupported VLA-Cache row: {name}") from exc


def scientific_contract() -> dict[str, object]:
    return {
        "method": "VLA-Cache",
        "training_required": False,
        "condition_refresh_interval": 1,
        "action_horizon": 10,
        "execution_horizon": 5,
        "flow_steps": 10,
        "control_protocol_changed": False,
        "action_generator_changed": False,
        "vision_encoder_recomputed_each_query": True,
        "text_decoder_optimization": (
            "stable-minus-task-relevant visual-token selection, layerwise "
            "token skipping, and prior K/V retention"
        ),
        "adaptation": VLACacheConfig().to_dict(),
        "comparison_note": (
            "This is a method-faithful SimVLA adaptation because official "
            "VLA-Cache does not publish a SimVLA integration."
        ),
    }
