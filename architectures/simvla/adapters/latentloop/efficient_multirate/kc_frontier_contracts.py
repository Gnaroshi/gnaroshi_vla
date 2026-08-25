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
)
from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    COUPLED_ROW,
)


FRONTIER_K_C_VALUES = (2, 3, 4)
FRONTIER_N_G_VALUES = (10, 3)


@dataclass(frozen=True)
class EfficiencyRowSpec:
    row: str
    k_c: int
    n_g: int
    uses_condition: bool
    uses_generation: bool
    coupled: bool = False


def condition_row_name(k_c: int, n_g: int) -> str:
    if int(k_c) not in FRONTIER_K_C_VALUES:
        raise ValueError(f"K_C must be one of {FRONTIER_K_C_VALUES}")
    if int(n_g) not in FRONTIER_N_G_VALUES:
        raise ValueError(f"N_G must be one of {FRONTIER_N_G_VALUES}")
    return f"condition_kc{int(k_c)}_ng{int(n_g)}"


ROW_SPECS = {
    FULL_ROW: EfficiencyRowSpec(FULL_ROW, 1, 10, False, False),
    GENERATION_ROW: EfficiencyRowSpec(GENERATION_ROW, 1, 3, False, True),
    **{
        condition_row_name(k_c, n_g): EfficiencyRowSpec(
            condition_row_name(k_c, n_g), k_c, n_g, True, n_g == 3
        )
        for k_c in FRONTIER_K_C_VALUES
        for n_g in FRONTIER_N_G_VALUES
    },
    COUPLED_ROW: EfficiencyRowSpec(COUPLED_ROW, 2, 3, True, True, True),
}

if condition_row_name(2, 10) != CONDITION_ROW:
    raise RuntimeError("fixed condition row name changed")
if condition_row_name(2, 3) != COMBINED_ROW:
    raise RuntimeError("fixed combined row name changed")

EVAL_ROWS = tuple(ROW_SPECS)
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
