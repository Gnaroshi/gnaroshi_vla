"""CPU-only checks for deterministic query-cache worker partitioning."""

from architectures.simvla.adapters.latentloop.query_cache_generator import (
    episode_env_seed,
    partition_episode_specs,
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
