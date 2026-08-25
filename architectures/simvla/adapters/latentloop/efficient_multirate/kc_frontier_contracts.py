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
    COUPLED_KC3_ROW,
    COUPLED_ROW,
)


FRONTIER_K_C_VALUES = (2, 3, 4)
FRONTIER_N_G_VALUES = (10, 3)
LEARNED_CONFIGS = tuple(
    (k_c, n_g) for k_c in FRONTIER_K_C_VALUES for n_g in FRONTIER_N_G_VALUES
) + ((2, 2),)
NAIVE_CONFIGS = ((2, 3), (2, 2), (3, 3))


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


ROW_SPECS = {
    FULL_ROW: EfficiencyRowSpec(FULL_ROW, 1, 10, False, False),
    GENERATION_ROW: EfficiencyRowSpec(GENERATION_ROW, 1, 3, False, True),
    **{
        condition_row_name(k_c, n_g): EfficiencyRowSpec(
            condition_row_name(k_c, n_g), k_c, n_g, True, n_g in {2, 3}
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
    COUPLED_ROW: EfficiencyRowSpec(COUPLED_ROW, 2, 3, True, True, True),
    COUPLED_KC3_ROW: EfficiencyRowSpec(
        COUPLED_KC3_ROW, 3, 3, True, True, True
    ),
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
