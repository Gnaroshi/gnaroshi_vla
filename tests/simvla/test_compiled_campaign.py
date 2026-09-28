import json
from types import SimpleNamespace

import pytest

from tools.simvla.compiled_campaign import SEEDS, digest, jobs, summarize_cell, validate_manifest
from tools.simvla.compiled_policy import check_policy, check_reset, expected_counts


def manifest():
    d = dict(suite="libero_10", determinism_seed=SEEDS["seed01"][0], action_noise_seed_base=SEEDS["seed01"][1],
        environment_seed=7, action_horizon=10, execution_horizon=5, flow_steps=10,
        num_wait_steps=10, client_resize_size=224, model_image_size=384, environment_resolution=256, max_policy_actions=900,
        episodes=[dict(task_id=t, trial_id=r, init_state_index=r, suite="libero_10") for t in range(10) for r in range(50)])
    d["manifest_sha256"] = digest(d)
    return d


def test_manifest_is_exact_500_paired_episodes():
    d = manifest()
    validate_manifest(d, "libero_10", "seed01")
    d["episodes"].pop()
    d["manifest_sha256"] = digest({k: v for k, v in d.items() if k != "manifest_sha256"})
    with pytest.raises(RuntimeError, match="50 trials"): validate_manifest(d, "libero_10", "seed01")


@pytest.mark.parametrize("field", ["action_horizon", "execution_horizon", "flow_steps", "environment_seed", "max_policy_actions"])
def test_manifest_rejects_semantics_changes(field):
    d = manifest()
    d[field] += 1
    d["manifest_sha256"] = digest({k: v for k, v in d.items() if k != "manifest_sha256"})
    with pytest.raises(RuntimeError): validate_manifest(d, "libero_10", "seed01")


@pytest.mark.parametrize("row", ["baseline", "naive_nfe3", "condition_naive3", "ours_kc2_ng3", "generation_ng3", "condition_nfe10", "latent_bridge_f2", "latent_bridge_f3", "latent_bridge_f4"])
def test_call_counters_and_queue(row):
    for actions in [1, 5, 6, 39, 40, 41, 900]:
        queries = (actions + 4) // 5
        counters = {**expected_counts(row, queries), "num_policy_queries": queries}
        p = SimpleNamespace(metrics=SimpleNamespace(counters=counters), step_index=actions)
        check_policy(p, row)
        counters["num_action_transformer_calls"] += 1
        with pytest.raises(RuntimeError): check_policy(p, row)


def test_reset_rejects_stale_condition():
    p = SimpleNamespace(reset=lambda: None, query_index=0, step_index=0, action_queue=[], cached_condition=1)
    with pytest.raises(RuntimeError, match="Stale"): check_reset(p)


def test_low_sr_is_not_failure_and_compile_time_does_not_enter_latency(tmp_path):
    (tmp_path / "episodes").mkdir()
    for trial, valid in [(0, True), (1, False)]:
        (tmp_path / "episodes" / f"{trial}.json").write_text(json.dumps({"identity": "same", "task_id": 0,
            "trial_id": trial, "success": 0, "episode_length": 5, "policy_ms_total": 50 if valid else 5000, "timing_valid": valid}))
    r = summarize_cell(tmp_path, "same", [(0, 0), (0, 1)])
    assert r["episodes"] == 2 and r["success_rate"] == 0
    assert r["pooled_policy_ms_per_action"] == 10
    assert summarize_cell(tmp_path, "same", [(0, 0), (0, 1), (0, 2)]) is None
    with pytest.raises(RuntimeError, match="Mixed"): summarize_cell(tmp_path, "wrong", [(0, 0), (0, 1)])


def test_registry_unique_and_long_first():
    c = dict(long_rows=["baseline", "ours", "bridge"], other_rows=["baseline", "ours"], other_suites=["libero_goal"], seeds=list(SEEDS))
    work = jobs(c)
    assert len(work) == len(set(work)) == 15
    assert all(suite == "libero_10" for suite, _, _ in work[:9])
