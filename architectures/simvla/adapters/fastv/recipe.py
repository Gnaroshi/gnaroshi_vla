"""Scientific contract for the SimVLA FastV comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


FASTV_PAPER_URL = "https://arxiv.org/html/2403.06764"
FASTV_REPOSITORY_URL = "https://github.com/pkunlp-icler/FastV"
LATENT_BRIDGE_PAPER_URL = "https://arxiv.org/html/2605.02739"


@dataclass(frozen=True)
class EvaluationRow:
    name: str
    uses_fastv: bool
    prune_layer: int | None
    prune_ratio: float
    score_mode: str | None
    restore_mode: str | None
    scientific_role: str

    def serializable(self) -> dict[str, Any]:
        return asdict(self)


EVALUATION_ROWS = {
    "baseline_k1": EvaluationRow(
        name="baseline_k1",
        uses_fastv=False,
        prune_layer=None,
        prune_ratio=0.0,
        score_mode=None,
        restore_mode=None,
        scientific_role=(
            "Frozen official SimVLA condition path at every policy query. "
            "No FastV object or patched text forward is constructed."
        ),
    ),
    "fastv_k2_r50": EvaluationRow(
        name="fastv_k2_r50",
        uses_fastv=True,
        prune_layer=2,
        prune_ratio=0.5,
        score_mode="text_mean_first_k",
        restore_mode="zero_scatter",
        scientific_role=(
            "Primary dual-system comparison: FastV paper K=2/R=0.5 with the "
            "text-to-image scoring and full-length zero restoration documented "
            "by the Latent Bridge VLA adaptation."
        ),
    ),
    "fastv_k2_r50_hf_last_token": EvaluationRow(
        name="fastv_k2_r50_hf_last_token",
        uses_fastv=True,
        prune_layer=2,
        prune_ratio=0.5,
        score_mode="hf_last_token_at_k",
        restore_mode="zero_scatter",
        scientific_role=(
            "Diagnostic reproduction of the current official Hugging Face "
            "FastV code, which ranks image tokens with the last query token at K."
        ),
    ),
}


def evaluation_row(name: str) -> EvaluationRow:
    try:
        return EVALUATION_ROWS[name]
    except KeyError as exc:
        raise ValueError(f"unknown FastV evaluation row: {name}") from exc


def scientific_contract() -> dict[str, Any]:
    return {
        "fastv_paper": FASTV_PAPER_URL,
        "fastv_official_repository": FASTV_REPOSITORY_URL,
        "dual_system_adaptation_reference": LATENT_BRIDGE_PAPER_URL,
        "evaluation_rows": {
            name: row.serializable() for name, row in EVALUATION_ROWS.items()
        },
        "source_discrepancy": {
            "paper": (
                "Section 4.1 ranks visual tokens by the average attention they "
                "receive from other tokens at the filtering layer."
            ),
            "official_hf_code": (
                "The pinned HF implementation averages heads but uses only the "
                "last query token's attention at layer K."
            ),
            "primary_choice": (
                "Use mean attention from all non-visual query positions across "
                "the first K layers, matching the documented dual-system VLA "
                "adaptation used by Latent Bridge. Keep the official-HF rule as "
                "an explicit diagnostic row."
            ),
        },
        "simvla_interface": {
            "expected_visual_positions": 72,
            "expected_nonvisual_positions": 50,
            "expected_full_sequence": 122,
            "k2_r50_compact_sequence": 86,
            "action_horizon": 10,
            "execution_horizon": 5,
            "flow_steps": 10,
            "control_contract": (
                "Policy-query timing, observations, action horizon, execution "
                "horizon, and flow integration are unchanged."
            ),
        },
        "scope": (
            "Official FastV algorithm adapted to SimVLA's fused-condition "
            "interface; not an official FastV SimVLA implementation."
        ),
    }
