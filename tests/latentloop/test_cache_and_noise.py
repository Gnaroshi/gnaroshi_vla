"""CPU-only tests for query cache, recursive state, and paired action noise."""

from __future__ import annotations

from pathlib import Path

import torch

from architectures.simvla.adapters.latentloop.action_adapter import (
    ActionNoiseKey,
    explicit_action_noise,
)
from architectures.simvla.adapters.latentloop.query_cache_state import (
    RecursiveQueryCache,
    SimVLAQueryObservation,
)
from methods.latentloop.training import (
    QueryCacheShardWriter,
    deterministic_episode_split_indices,
    merge_query_cache_parts,
    validate_on_policy_cache,
    validate_query_cache,
)
from methods.latentloop.training.query_cache_dataset import tensor_sha256


def _record(r: int, query_index: int = 0) -> dict[str, object]:
    noise = torch.arange(70, dtype=torch.float32).reshape(10, 7)
    next_noise = noise + 1
    actions = torch.linspace(-1, 1, r * 7).reshape(r, 7)
    return {
        "task_id": 2,
        "episode_id": "task02_trial000",
        "query_index": query_index,
        "next_query_index": query_index + 1,
        "absolute_env_timestep": 10 + query_index * r,
        "next_absolute_env_timestep": 10 + (query_index + 1) * r,
        "language_instruction": "test instruction",
        "task_identifier": "test/task.bddl",
        "execution_horizon": r,
        "elapsed_time": r / 20.0,
        "action_noise_hash": tensor_sha256(noise),
        "next_action_noise_hash": tensor_sha256(next_noise),
        "provenance": {"test": True},
        "raw_rgb": torch.zeros(2, 8, 8, 3),
        "proprio": torch.zeros(8),
        "full_condition": torch.zeros(122, 960),
        "teacher_action_chunk": torch.zeros(10, 7),
        "initial_noise": noise,
        "executed_subchunk": actions,
        "executed_env_actions": actions.clone(),
        "next_raw_rgb": torch.ones(2, 8, 8, 3),
        "next_proprio": torch.ones(8),
        "next_full_condition": torch.ones(122, 960),
        "next_teacher_action_chunk": torch.ones(10, 7),
        "next_initial_noise": next_noise,
    }


def test_r1_and_r5_cache_serialization(tmp_path: Path) -> None:
    for r in (1, 5):
        root = tmp_path / f"r{r}"
        writer = QueryCacheShardWriter(root, execution_horizon=r, metadata={"test": True})
        writer.add(_record(r))
        writer.close()
        result = validate_query_cache(root)
        assert result["passed"], result["errors"]
        assert result["execution_horizon"] == r
        assert result["records"] == 1


def test_copy_free_nested_cache_merge(tmp_path: Path) -> None:
    root = tmp_path / "merged"
    parts: list[Path] = []
    for worker_index in range(2):
        part = root / "parts" / f"worker{worker_index:02d}"
        episode = part / "episodes" / f"task02_trial{worker_index:03d}"
        record = _record(1)
        record["episode_id"] = f"task02_trial{worker_index:03d}"
        writer = QueryCacheShardWriter(
            episode,
            execution_horizon=1,
            metadata={"worker_index": worker_index},
        )
        writer.add(record)
        writer.close()
        merge_query_cache_parts(part, [episode], metadata={"worker_index": worker_index})
        parts.append(part)

    manifest = merge_query_cache_parts(root, parts, metadata={"test_merge": True})
    assert manifest["total_records"] == 2
    assert all(
        shard["file"].startswith("parts/worker")
        and "/episodes/task02_trial" in shard["file"]
        for shard in manifest["shards"]
    )
    assert not any(root.glob("query_shard_*.pt"))
    result = validate_query_cache(root)
    assert result["passed"], result["errors"]
    assert result["episodes"] == 2


def test_on_policy_extra_fields_survive_serialization(tmp_path: Path) -> None:
    root = tmp_path / "on_policy"
    record = _record(5)
    record["predicted_condition"] = torch.full((122, 960), 2.0)
    record["rollout_depth"] = 1
    record["rollout_episode_id"] = "task02_trial000"
    record["adapter_checkpoint"] = "/tmp/adapter.pt"
    writer = QueryCacheShardWriter(root, execution_horizon=5, metadata={"cache_kind": "on_policy"})
    writer.add(record)
    writer.close()
    result = validate_on_policy_cache(root, maximum_rollout_depth=1)
    assert result["passed"], result["errors"]
    assert result["rollout_depth_counts"] == {1: 1}


def test_paired_noise_is_exact_and_does_not_consume_global_rng() -> None:
    key = ActionNoiseKey("checkpoint", 3, "episode", 7, 1234)
    before = torch.get_rng_state().clone()
    first = explicit_action_noise(
        key,
        batch_size=1,
        action_horizon=10,
        action_dim=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    after = torch.get_rng_state().clone()
    second = explicit_action_noise(
        key,
        batch_size=1,
        action_horizon=10,
        action_dim=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(before, after)
    assert torch.equal(first, second)


def test_recursive_cache_advances_adjacent_observation_each_query() -> None:
    cache = RecursiveQueryCache()
    condition = torch.zeros(1, 2, 960)
    observation0 = SimVLAQueryObservation(torch.zeros(1, 2, 4, 4, 3), torch.zeros(1, 8))
    observation1 = SimVLAQueryObservation(torch.ones(1, 2, 4, 4, 3), torch.ones(1, 8))
    observation2 = SimVLAQueryObservation(torch.full((1, 2, 4, 4, 3), 2.0), torch.full((1, 8), 2.0))
    cache.full_refresh(condition, observation0)
    cache.record_executed_subchunk(torch.zeros(1, 5, 7))
    first = cache.lightweight_transition_inputs(observation1)
    assert torch.equal(first["previous_query_observation"].raw_rgb, observation0.raw_rgb)
    cache.commit_lightweight_update(condition + 1, observation1)
    cache.record_executed_subchunk(torch.ones(1, 5, 7))
    second = cache.lightweight_transition_inputs(observation2)
    assert torch.equal(second["previous_query_observation"].raw_rgb, observation1.raw_rgb)
    assert torch.equal(second["executed_subchunk"], torch.ones(1, 5, 7))
    assert cache.query_age == 1
    assert cache.trace[-2]["event"] == "lightweight_commit"
    assert cache.trace[-2]["observation_cache_advanced"]


def test_episode_split_has_no_cross_partition_episode_leakage() -> None:
    dataset = [
        {"task_id": task, "episode_id": f"ep{episode:02d}"}
        for task in range(2)
        for episode in range(20)
        for _ in range(2)
    ]
    train, heldout = deterministic_episode_split_indices(
        dataset,
        heldout_fraction=0.2,
        seed=20260804,
    )
    train_keys = {(dataset[index]["task_id"], dataset[index]["episode_id"]) for index in train}
    heldout_keys = {
        (dataset[index]["task_id"], dataset[index]["episode_id"]) for index in heldout
    }
    assert train_keys.isdisjoint(heldout_keys)
