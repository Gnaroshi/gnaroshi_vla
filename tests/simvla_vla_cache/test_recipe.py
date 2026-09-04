import pytest

from architectures.simvla.adapters.vla_cache.recipe import (
    ROWS,
    evaluation_row,
    scientific_contract,
)


def test_rows_separate_reuse_from_matched_backend_control():
    assert set(ROWS) == {"vla_cache", "vla_cache_full"}
    assert evaluation_row("vla_cache").enable_reuse is True
    assert evaluation_row("vla_cache_full").enable_reuse is False
    with pytest.raises(ValueError):
        evaluation_row("baseline")


def test_scientific_contract_preserves_simvla_control_and_flow_schedule():
    contract = scientific_contract()
    assert contract["training_required"] is False
    assert contract["condition_refresh_interval"] == 1
    assert contract["action_horizon"] == 10
    assert contract["execution_horizon"] == 5
    assert contract["flow_steps"] == 10
    assert contract["control_protocol_changed"] is False
    assert contract["action_generator_changed"] is False
