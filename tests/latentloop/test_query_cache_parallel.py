"""CPU-only checks for deterministic query-cache worker partitioning."""

from types import SimpleNamespace

from architectures.simvla.adapters.latentloop.query_cache_generator import (
    episode_env_seed,
    partition_episode_specs,
)
from architectures.simvla.adapters.latentloop.query_cache_pipeline import (
    _phase_config,
)


def test_episode_partitions_are_balanced_unique_and_complete() -> None:
    task_ids = list(range(9, -1, -1))
    partitions = [
        partition_episode_specs(task_ids, 2, worker_index, 4)
        for worker_index in range(4)
    ]
    flattened = [spec for partition in partitions for spec in partition]
    assert [len(partition) for partition in partitions] == [5, 5, 5, 5]
    assert len({ordinal for ordinal, _, _ in flattened}) == 20
    assert {(task_id, trial_id) for _, task_id, trial_id in flattened} == {
        (task_id, trial_id) for task_id in task_ids for trial_id in range(2)
    }


def test_episode_environment_seed_is_stable_and_episode_specific() -> None:
    first = episode_env_seed(7, 9, 0)
    assert first == episode_env_seed(7, 9, 0)
    assert first != episode_env_seed(7, 9, 1)
    assert first != episode_env_seed(7, 8, 0)
    assert 0 <= first < 2**31 - 1


def test_phase_config_locks_osmesa_master_seed_and_environment_lifecycle() -> None:
    args = SimpleNamespace(
        checkpoint="YuankaiLuo/SimVLA-LIBERO",
        norm_stats="/tmp/libero_norm.json",
        suite="libero_10",
        max_tasks=10,
        num_wait_steps=10,
        flow_steps=10,
        client_resize_size=224,
        image_size=384,
        resolution=256,
        control_hz=20.0,
        seed=7,
        action_noise_seed_base=20260804,
        experiment_seed=20260815,
        render_backend="osmesa",
        task_order="official_reverse",
        records_per_shard=128,
    )
    config = _phase_config(
        args,
        name="query_osmesa_r5_full_10x20",
        execution_horizon=5,
        num_trials=20,
        max_policy_queries=180,
        gpus=("4", "5", "6", "7"),
    )
    assert config["experiment_seed"] == 20260815
    assert config["effective_seed_plan"]["process_seed"] == 20260815
    assert config["render_backend"] == "osmesa"
    assert config["environment_lifecycle"] == "fresh_environment_per_episode"

    args.render_backend = "egl"
    egl = _phase_config(
        args,
        name="query_osmesa_r5_full_10x20",
        execution_horizon=5,
        num_trials=20,
        max_policy_queries=180,
        gpus=("4", "5", "6", "7"),
    )
    assert egl["render_backend"] == "egl"
    assert egl != config
