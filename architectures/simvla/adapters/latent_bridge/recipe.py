"""Evidence-backed scientific contracts for the SimVLA Latent Bridge port."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PAPER_URL = "https://arxiv.org/html/2605.02739"
OFFICIAL_REPOSITORY_URL = "https://github.com/1999Lyd/Latent-Bridge"
STABLE_LAYER_MIN_COSINE = 0.999
SIMVLA_TOKEN_EVIDENCE = {
    "artifact": (
        "codex_outputs/simvla_multiepisode_latent_continuity_20260901_134735/"
        "data/merged/aggregate_similarity_by_offset.csv"
    ),
    "episodes": 20,
    "policy_queries": 1062,
    "adjacent_cosine": {
        "all_fused_tokens": 0.9435860360767013,
        "visual_positions_72": 0.9491602565070718,
        "language_positions_50": 0.9355591580185918,
    },
    "decision": (
        "Use all 122 fused condition positions for the primary SimVLA port. "
        "The official GR00T image-only shortcut relies on nearly static text "
        "tokens (>0.9999 cosine), which does not hold at SimVLA's fused boundary."
    ),
}


@dataclass(frozen=True)
class TrainingRecipe:
    stage: str
    epochs: int
    learning_rate: float
    effective_batch_size: int = 64
    weight_decay: float = 1e-4
    cosine_weight: float = 1.0
    max_grad_norm: float = 1.0
    precision: str = "bf16"
    token_mode: str = "all"
    flow_matching: bool = False
    hidden_dim: int = 768
    num_heads: int = 12
    num_blocks: int = 12
    stable_layer_index: int = 10

    def serializable(self) -> dict[str, Any]:
        return asdict(self)


PAPER_FEATURE_RECIPES = {
    "r0": TrainingRecipe(stage="r0", epochs=200, learning_rate=3e-4),
    "r1": TrainingRecipe(stage="r1", epochs=100, learning_rate=3e-5),
}


@dataclass(frozen=True)
class EvaluationRow:
    name: str
    refresh_every: int
    uses_bridge: bool
    full_vlm_call_saving: float
    scientific_role: str

    def serializable(self) -> dict[str, Any]:
        return asdict(self)


EVALUATION_ROWS = {
    "baseline_k1": EvaluationRow(
        name="baseline_k1",
        refresh_every=1,
        uses_bridge=False,
        full_vlm_call_saving=0.0,
        scientific_role="Frozen full SimVLA at every policy query.",
    ),
    "latent_bridge_f3": EvaluationRow(
        name="latent_bridge_f3",
        refresh_every=3,
        uses_bridge=True,
        full_vlm_call_saving=2.0 / 3.0,
        scientific_role=(
            "Feature-bridge recurrence used for the released GR00T DAgger data "
            "collection and the paper's feature-space bridge comparison."
        ),
    ),
    "latent_bridge_f4": EvaluationRow(
        name="latent_bridge_f4",
        refresh_every=4,
        uses_bridge=True,
        full_vlm_call_saving=0.75,
        scientific_role=(
            "Matched backbone-call budget for SimVLA K_C=4. The official f=4 "
            "main operating point is pi0.5 KV Bridge, not released SimVLA code."
        ),
    ),
}


def training_recipe(stage: str) -> TrainingRecipe:
    try:
        return PAPER_FEATURE_RECIPES[stage]
    except KeyError as exc:
        raise ValueError(f"unknown Latent Bridge training stage: {stage}") from exc


def evaluation_row(name: str) -> EvaluationRow:
    try:
        return EVALUATION_ROWS[name]
    except KeyError as exc:
        raise ValueError(f"unknown Latent Bridge evaluation row: {name}") from exc


def scientific_contract() -> dict[str, Any]:
    return {
        "paper": PAPER_URL,
        "official_repository": OFFICIAL_REPOSITORY_URL,
        "training_recipes": {
            name: recipe.serializable()
            for name, recipe in PAPER_FEATURE_RECIPES.items()
        },
        "evaluation_rows": {
            name: row.serializable() for name, row in EVALUATION_ROWS.items()
        },
        "simvla_token_evidence": SIMVLA_TOKEN_EVIDENCE,
        "data_contract": {
            "r0_sync_episodes_per_task": 30,
            "r1_dagger_episodes_per_task": 30,
            "r1_dagger_refresh_every": 3,
            "final_seeds": [0, 1, 2],
            "final_episodes_per_task_per_seed": 20,
            "stable_layer_min_adjacent_cosine": STABLE_LAYER_MIN_COSINE,
        },
        "scope": (
            "Official Latent Bridge algorithm adapted to SimVLA's feature "
            "interface; not an official Latent Bridge SimVLA implementation."
        ),
    }
