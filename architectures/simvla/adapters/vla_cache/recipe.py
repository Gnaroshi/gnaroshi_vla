"""Evaluation contracts for the SimVLA adaptation of VLA-Cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .official_contract import VLACacheConfig

IMPLEMENTATION_VERSION = "oft_runtime_v3"
OFFICIAL_NORM_SHA256 = "5e4dcf9026271137e102f6f784d345f0f03c1fd9963b679631b110a16788149e"


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
            "Native SimVLA forward without adapter token processing or KV reuse."
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
        "implementation_version": IMPLEMENTATION_VERSION,
        "task_relevance_queries": "valid tokenizer text positions; native model mask unchanged",
        "entropy_maps": "all actual decoder layers; no metadata sentinel",
        "sparse_runtime": "selection before vision; precomputed gather indices; unchanged selected tokens",
        "dense_condition_reconstruction": "previous final hidden at removed positions",
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
            "SimVLA adaptation of the pinned official OFT code. Connector budgets "
            "and dense hidden reconstruction are explicit architecture adaptations, "
            "not an official SimVLA reproduction."
        ),
    }
