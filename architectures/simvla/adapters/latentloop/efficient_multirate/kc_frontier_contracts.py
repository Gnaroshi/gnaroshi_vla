"""Row contracts for the SimVLA condition-refresh efficiency frontier."""

from __future__ import annotations

from dataclasses import dataclass

from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_contracts import (
    COMBINED_ROW,
    CONDITION_ROW,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FULL_ROW,
    GENERATION_ROW,
    NAIVE_ROW,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    COUPLED_CONFIGS,
    COUPLED_KC3_ROW,
    COUPLED_ROW,
    coupled_row_name,
)


FRONTIER_K_C_VALUES = (2, 3, 4)
FRONTIER_N_G_VALUES = (10, 3)
PAPER_K_C_VALUES = (1, 2, 3)
PAPER_REDUCED_COMPUTE_VALUES = (2, 3, 5)
LEARNED_CONFIGS = tuple(dict.fromkeys(tuple(
    (k_c, n_g) for k_c in FRONTIER_K_C_VALUES for n_g in FRONTIER_N_G_VALUES
) + tuple(
    (k_c, n_g) for k_c in (2, 3) for n_g in PAPER_REDUCED_COMPUTE_VALUES
)))
NAIVE_CONFIGS = tuple(
    (k_c, nfe) for k_c in (2, 3) for nfe in PAPER_REDUCED_COMPUTE_VALUES
)
GENERATION_CONFIGS = PAPER_REDUCED_COMPUTE_VALUES
NAIVE_GENERATION_CONFIGS = PAPER_REDUCED_COMPUTE_VALUES


@dataclass(frozen=True)
class EfficiencyRowSpec:
    row: str
    k_c: int
    n_g: int
    uses_condition: bool
    uses_generation: bool
    coupled: bool = False
    naive_nfe: bool = False


def condition_row_name(k_c: int, n_g: int) -> str:
    pair = (int(k_c), int(n_g))
    if pair not in LEARNED_CONFIGS:
        raise ValueError(f"unsupported learned (K_C, N_G): {pair}")
    return f"condition_kc{int(k_c)}_ng{int(n_g)}"


def naive_condition_row_name(k_c: int, nfe: int) -> str:
    pair = (int(k_c), int(nfe))
    if pair not in NAIVE_CONFIGS:
        raise ValueError(f"unsupported naive (K_C, NFE): {pair}")
    return f"condition_kc{int(k_c)}_naive_nfe{int(nfe)}"


def generation_row_name(n_g: int) -> str:
    value = int(n_g)
    if value not in GENERATION_CONFIGS:
        raise ValueError(f"unsupported Generation schedule: {value}")
    return f"generation_ng{value}"


def naive_generation_row_name(nfe: int) -> str:
    value = int(nfe)
    if value not in NAIVE_GENERATION_CONFIGS:
        raise ValueError(f"unsupported naive NFE: {value}")
    return f"naive_nfe{value}"


ROW_SPECS = {
    FULL_ROW: EfficiencyRowSpec(FULL_ROW, 1, 10, False, False),
    **{
        generation_row_name(n_g): EfficiencyRowSpec(
            generation_row_name(n_g), 1, n_g, False, True
        )
        for n_g in GENERATION_CONFIGS
    },
    **{
        naive_generation_row_name(nfe): EfficiencyRowSpec(
            naive_generation_row_name(nfe),
            1,
            nfe,
            False,
            False,
            False,
            True,
        )
        for nfe in NAIVE_GENERATION_CONFIGS
    },
    **{
        condition_row_name(k_c, n_g): EfficiencyRowSpec(
            condition_row_name(k_c, n_g),
            k_c,
            n_g,
            True,
            n_g in PAPER_REDUCED_COMPUTE_VALUES,
        )
        for k_c, n_g in LEARNED_CONFIGS
    },
    **{
        naive_condition_row_name(k_c, nfe): EfficiencyRowSpec(
            naive_condition_row_name(k_c, nfe),
            k_c,
            nfe,
            True,
            False,
            False,
            True,
        )
        for k_c, nfe in NAIVE_CONFIGS
    },
    **{
        coupled_row_name(k_c, n_g): EfficiencyRowSpec(
            coupled_row_name(k_c, n_g), k_c, n_g, True, True, True
        )
        for k_c, n_g in COUPLED_CONFIGS
    },
}

if condition_row_name(2, 10) != CONDITION_ROW:
    raise RuntimeError("fixed condition row name changed")
if condition_row_name(2, 3) != COMBINED_ROW:
    raise RuntimeError("fixed combined row name changed")
if generation_row_name(3) != GENERATION_ROW:
    raise RuntimeError("Generation N_G=3 row name changed")
if naive_generation_row_name(3) != NAIVE_ROW:
    raise RuntimeError("naive NFE=3 row name changed")
if coupled_row_name(2, 3) != COUPLED_ROW or coupled_row_name(3, 3) != COUPLED_KC3_ROW:
    raise RuntimeError("coupled N_G=3 row names changed")

PAPER_ANCHOR_ROWS = (
    FULL_ROW,
    condition_row_name(2, 10),
    condition_row_name(3, 10),
)
PAPER_NAIVE_ROWS = tuple(
    naive_generation_row_name(nfe) if k_c == 1 else naive_condition_row_name(k_c, nfe)
    for k_c in PAPER_K_C_VALUES
    for nfe in PAPER_REDUCED_COMPUTE_VALUES
)
PAPER_LEARNED_ROWS = tuple(
    generation_row_name(n_g) if k_c == 1 else condition_row_name(k_c, n_g)
    for k_c in PAPER_K_C_VALUES
    for n_g in PAPER_REDUCED_COMPUTE_VALUES
)
PAPER_COUPLED_ROWS = tuple(
    coupled_row_name(k_c, n_g)
    for k_c in (2, 3)
    for n_g in PAPER_REDUCED_COMPUTE_VALUES
)
PAPER_GRID_ROWS = (
    PAPER_ANCHOR_ROWS + PAPER_NAIVE_ROWS + PAPER_LEARNED_ROWS + PAPER_COUPLED_ROWS
)
if len(PAPER_GRID_ROWS) != 27 or len(set(PAPER_GRID_ROWS)) != 27:
    raise RuntimeError("paper grid must contain exactly 27 unique rows")

LEGACY_FIXED_ROWS = {
    FULL_ROW,
    GENERATION_ROW,
    condition_row_name(2, 10),
    condition_row_name(2, 3),
    coupled_row_name(2, 3),
}


def is_frontier_row(row: str) -> bool:
    row_spec(row)
    return row not in LEGACY_FIXED_ROWS

EVAL_ROWS = tuple(dict.fromkeys(PAPER_GRID_ROWS + tuple(ROW_SPECS)))
CONDITION_ROWS = tuple(
    name for name, spec in ROW_SPECS.items() if spec.uses_condition and not spec.coupled
)
GENERATION_ROWS = tuple(
    name for name, spec in ROW_SPECS.items() if spec.uses_generation and not spec.coupled
)
FRONTIER_ROWS = tuple(
    condition_row_name(k_c, n_g)
    for k_c in (3, 4)
    for n_g in FRONTIER_N_G_VALUES
)
JOINT_NFE_ROWS = (
    naive_condition_row_name(2, 3),
    condition_row_name(2, 2),
    naive_condition_row_name(2, 2),
    naive_condition_row_name(3, 3),
)


def row_spec(row: str) -> EfficiencyRowSpec:
    try:
        return ROW_SPECS[str(row)]
    except KeyError as exc:
        raise ValueError(f"unknown efficiency row: {row}") from exc


def expected_call_counts(row: str, policy_queries: int) -> dict[str, int]:
    spec = row_spec(row)
    queries = int(policy_queries)
    if queries < 0:
        raise ValueError("policy query count must be non-negative")
    full_vlm_calls = (
        (queries + spec.k_c - 1) // spec.k_c if spec.uses_condition else queries
    )
    condition_calls = queries - full_vlm_calls if spec.uses_condition else 0
    full_action_calls = queries * spec.n_g
    generation_updates = queries * (10 - spec.n_g) if spec.uses_generation else 0
    return {
        "full_vlm_calls": full_vlm_calls,
        "condition_updater_calls": condition_calls,
        "full_action_transformer_calls": full_action_calls,
        "generation_loop_updates": generation_updates,
        "integration_updates": full_action_calls + generation_updates,
    }
