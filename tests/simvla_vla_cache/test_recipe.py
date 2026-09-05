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


def test_norm_identity_does_not_depend_on_old_server_path(tmp_path):
    from architectures.simvla.adapters.vla_cache.eval import validate_norm_stats
    wrong = tmp_path / "norm.json"
    wrong.write_text("{}")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        validate_norm_stats(wrong)


def test_implementation_identity_is_stable_and_covers_runtime():
    from architectures.simvla.adapters.vla_cache.eval import implementation_identity
    identity = implementation_identity()
    assert identity == implementation_identity()
    assert {"smolvlm_runtime.py", "official_contract.py", "policy.py", "eval.py"} <= identity.keys()


@pytest.mark.parametrize("suite,steps", [("libero_10", 900), ("libero_spatial", 800), ("libero_object", 800), ("libero_goal", 800)])
def test_paper_manifest_suite_limits(tmp_path, suite, steps):
    import json
    from architectures.simvla.adapters.vla_cache.eval import _load_manifest
    data = dict(checkpoint="YuankaiLuo/SimVLA-LIBERO",
                checkpoint_revision="93dc4d90b0596c652ad2840ad743c62b9c4473fb",
                norm_stats="unused", suite=suite, action_horizon=10, execution_horizon=5,
                flow_steps=10, num_wait_steps=10, max_policy_actions=steps,
                environment_resolution=256, client_resize_size=224, model_image_size=384,
                determinism_seed=1, action_noise_seed_base=2,
                episodes=[dict(task_id=0, trial_id=0)], task_iteration_order={"rank0": [0]})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    assert _load_manifest(path, row="vla_cache", max_episodes=None)["suite"] == suite
    data["max_policy_actions"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="paper evaluation contract"):
        _load_manifest(path, row="vla_cache", max_episodes=None)
