#!/usr/bin/env python3
"""Generate only source-locked, manifest-separated schema-v2 teacher cache rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from _common import DEFAULT_CHECKPOINT, load_local_policy, require_run
from generate_pi05_latentloop_cache import compact_next_observation, raw_policy_observation
from pi05_stage_gate_v2 import verify_stage
from architectures.openpi.adapters.latentloop.cache_contract_v2 import (
    array_hash,
    load_final_evaluation_manifest,
    load_split_contract,
    resolve_task_identity,
    tensor_contract_from_record,
    tree_hash,
    validate_record_v2,
)
from architectures.openpi.adapters.latentloop.cache_io import EpisodeCacheWriter
from architectures.openpi.adapters.latentloop.full_cache_contract_v2 import (
    expected_query_spec,
    full_cache_episode_identity,
    load_full_cache_inventory,
    sha256_file,
)
from architectures.openpi.adapters.latentloop.policy_io import (
    explicit_policy_noise,
    policy_noise_seed,
    postprocess_policy_actions,
    prepare_policy_observation,
)
from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixKVHook
from architectures.openpi.adapters.latentloop.serialization import prefix_state_to_dict


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(lock: dict[str, Any]) -> dict[str, str]:
    return {
        "source": lock["ours_and_upstream_source"]["combined_sha256"],
        "checkpoint": lock["checkpoint"]["model_sha256"],
        "norm_stats": lock["normalization"]["sha256"],
        "config": lock["checkpoint"]["config_sha256"],
        "preprocessing": lock["preprocessing"]["combined_sha256"],
        "postprocessing": lock["postprocessing"]["combined_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--k1-gate", required=True)
    parser.add_argument("--freeze-gate", required=True)
    parser.add_argument("--final-evaluation-manifest", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--full-cache-inventory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--execution-horizon", type=int, default=5)
    parser.add_argument("--noise-seed-base", type=int, default=20260820)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-queries-per-episode", type=int, default=2)
    parser.add_argument("--suite", default="all")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    full_cache = args.full_cache_inventory is not None
    require_run(
        args.run,
        "OPENPI_LATENTLOOP_FULL_CACHE_RUN" if full_cache else "OPENPI_LATENTLOOP_CACHE_RUN",
    )
    if args.execution_horizon != 5:
        raise ValueError("pinned execution horizon is R=5")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0,num-shards)")
    if full_cache:
        if args.max_episodes is not None or args.max_queries_per_episode != 2:
            raise ValueError("full-cache generation forbids bounded smoke truncation arguments")
        if args.suite != "all":
            raise ValueError("the frozen full cache must cover all four suites")
    else:
        if args.max_episodes is None or not 5 <= args.max_episodes <= 32:
            raise ValueError("repair-stage cache generation is bounded to a 5..32 episode schema smoke")
        if not 1 <= args.max_queries_per_episode <= 4:
            raise ValueError("repair-stage cache generation is bounded to 1..4 queries per episode")

    output = Path(args.output).resolve()
    gate_artifacts = [args.k1_gate, args.freeze_gate]
    if full_cache:
        gate_artifacts.append(args.full_cache_inventory)
    gate = verify_stage(
        "stage2_full_cache" if full_cache else "stage2_cache_smoke",
        args.source_lock,
        gate_artifacts,
        output_candidate=output,
    )
    source_lock = json.loads(Path(args.source_lock).read_text(encoding="utf-8"))
    if Path(source_lock["checkpoint"]["directory"]).resolve() != Path(args.checkpoint).resolve():
        raise ValueError("checkpoint mismatch: cache checkpoint differs from source lock v2")
    final_manifest = load_final_evaluation_manifest(args.final_evaluation_manifest)
    split_mapping, split_contract = load_split_contract(args.split_contract, final_manifest)
    inventory = None
    inventory_by_identity = {}
    if full_cache:
        inventory = load_full_cache_inventory(
            args.full_cache_inventory,
            source_lock_id=gate["source_lock_id"],
            split_contract=split_contract,
            final_manifest=final_manifest,
            split_contract_path=args.split_contract,
            final_manifest_path=args.final_evaluation_manifest,
        )
        if int(inventory["num_shards"]) != args.num_shards:
            raise ValueError("--num-shards differs from the frozen full-cache inventory")
        if int(inventory["protocol"]["noise_seed_base"]) != args.noise_seed_base:
            raise ValueError("--noise-seed-base differs from the frozen full-cache inventory")
        inventory_by_identity = {
            full_cache_episode_identity(row): row for row in inventory["episodes"]
        }
    allowed_suites = {row["suite"] for row in final_manifest["tasks"]}
    if args.suite != "all" and args.suite not in allowed_suites:
        raise ValueError(f"suite is absent from frozen task catalog: {args.suite}")

    from openpi.training import config as config_api
    from openpi.training import data_loader

    train_config = config_api.get_config("pi05_libero_lora_pytorch")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = data_loader.create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    base_dataset = getattr(dataset, "_dataset", dataset)
    starts = base_dataset.episode_data_index["from"].tolist()
    stops = base_dataset.episode_data_index["to"].tolist()
    policy = load_local_policy(args.checkpoint, args.device, flow_steps=args.flow_steps)
    model = policy._model  # noqa: SLF001
    if int(model.config.action_horizon) != 10:
        raise ValueError("source-audited pi0.5 action horizon must be H=10")
    hook = PrefixKVHook(model)
    source_hashes = _source_hashes(source_lock)
    writer: EpisodeCacheWriter | None = None
    task_catalog = {
        (str(row["suite"]), int(row["benchmark_task_index"])): row for row in final_manifest["tasks"]
    }
    if full_cache:
        selected_specs = [
            row for row in inventory["episodes"] if int(row["shard_index"]) == args.shard_index
        ]
        selected = [int(row["episode_id"]) for row in selected_specs]
        if not selected:
            raise RuntimeError(f"full-cache shard {args.shard_index} has no frozen episodes")
    else:
        candidates = [index for index in range(len(starts)) if index % args.num_shards == args.shard_index]
        by_role: dict[str, int] = {}
        eligible: list[int] = []
        for episode_id in candidates:
            identity_sample = dataset[int(starts[episode_id])]
            task = resolve_task_identity(
                int(identity_sample["task_index"]), str(identity_sample["prompt"]), final_manifest
            )
            if args.suite != "all" and task["suite"] != args.suite:
                continue
            key = (
                str(task["suite"]),
                int(task["benchmark_task_index"]),
                "teacher_demonstration",
                str(episode_id),
            )
            assignment = split_mapping.get(key)
            if assignment is None:
                raise RuntimeError(f"missing split-contract assignment for demonstration episode {key}")
            eligible.append(episode_id)
            by_role.setdefault(str(assignment["role"]), episode_id)
        required_roles = tuple(split_contract["required_cache_roles"])
        missing_roles = sorted(set(required_roles) - by_role.keys())
        if missing_roles:
            raise RuntimeError(f"bounded cache smoke cannot cover frozen roles {missing_roles}")
        selected = list(dict.fromkeys([by_role[role] for role in required_roles] + eligible))[
            : args.max_episodes
        ]
    total_records = 0
    started_at = time.time()

    for selection_index, episode_id in enumerate(selected):
        start, stop = int(starts[episode_id]), int(stops[episode_id])
        identity_sample = dataset[start]
        task = resolve_task_identity(
            int(identity_sample["task_index"]), str(identity_sample["prompt"]), final_manifest
        )
        suite = str(task["suite"])
        benchmark_task_index = int(task["benchmark_task_index"])
        if args.suite != "all" and suite != args.suite:
            continue
        episode_key = (suite, benchmark_task_index, "teacher_demonstration", str(episode_id))
        assignment = split_mapping.get(episode_key)
        if assignment is None:
            raise RuntimeError(f"missing split-contract assignment for demonstration episode {episode_key}")
        if assignment.get("environment_seed") is None or not assignment.get("initial_state_identifier"):
            raise RuntimeError(f"split assignment lacks environment provenance: {episode_key}")
        inventory_episode = inventory_by_identity.get((suite, benchmark_task_index, str(episode_id)))
        if full_cache:
            if inventory_episode is None:
                raise RuntimeError(f"episode is absent from full-cache inventory: {episode_key}")
            if (start, stop) != (
                int(inventory_episode["dataset_frame_start"]),
                int(inventory_episode["dataset_frame_stop"]),
            ):
                raise RuntimeError(f"dataset frame bounds differ from full-cache inventory: {episode_key}")
            frame_indices = [
                expected_query_spec(
                    inventory_episode,
                    query,
                    execution_horizon=args.execution_horizon,
                    noise_seed_base=args.noise_seed_base,
                )["dataset_frame_index"]
                for query in range(int(inventory_episode["query_count"]))
            ]
        else:
            frame_indices = list(
                range(start, stop - args.execution_horizon, args.execution_horizon)
            )
        records: list[dict[str, Any]] = []
        for frame in frame_indices:
            current = dataset[frame]
            next_sample = dataset[frame + args.execution_horizon]
            if bool(torch.as_tensor(current["actions_is_pad"][: args.execution_horizon]).any()):
                if full_cache:
                    raise RuntimeError(f"frozen full-cache query crosses padded actions: {episode_key}/{frame}")
                break
            query_index = len(records)
            if int(current["frame_index"]) != query_index * args.execution_horizon:
                raise RuntimeError(f"dataset frame index violates exact native R progression: {episode_key}")
            raw_observation = raw_policy_observation(current)
            observation, transformed = prepare_policy_observation(policy, raw_observation)
            seed = policy_noise_seed(
                args.noise_seed_base, suite, benchmark_task_index, episode_id, query_index
            )
            if inventory_episode is not None:
                query_spec = expected_query_spec(
                    inventory_episode,
                    query_index,
                    execution_horizon=args.execution_horizon,
                    noise_seed_base=args.noise_seed_base,
                )
                if seed != query_spec["action_noise_seed"]:
                    raise RuntimeError("generator noise seed differs from frozen full-cache inventory")
            with torch.no_grad():
                extraction = hook.extract(observation)
                noise = explicit_policy_noise(
                    (1, model.config.action_horizon, model.config.action_dim),
                    seed=seed,
                    device=args.device,
                )
                teacher, action_timing = hook.sample_actions_from_state(
                    extraction.state,
                    extraction.robot_state,
                    noise,
                    num_steps=args.flow_steps,
                )
                postprocessed = postprocess_policy_actions(
                    policy, transformed["state"], teacher
                )["actions"]
            raw_images = {
                "base_0_rgb": raw_observation["observation/image"],
                "left_wrist_0_rgb": raw_observation["observation/wrist_image"],
            }
            preprocessed_images = {
                key: value[0].detach().cpu().to(torch.float16) for key, value in observation.images.items()
            }
            state_payload = prefix_state_to_dict(extraction.state)
            executed = torch.as_tensor(current["actions"][: args.execution_horizon], dtype=torch.float32)[..., :7].cpu()
            record = {
                "suite": suite,
                "task_id": benchmark_task_index,
                "benchmark_task_index": benchmark_task_index,
                "dataset_task_index": int(current["task_index"]),
                "canonical_task_name": str(task["canonical_task_name"]),
                "canonical_instruction": str(task["canonical_instruction"]),
                "language_instruction": str(task["canonical_instruction"]),
                "episode_namespace": "teacher_demonstration",
                "episode_id": episode_id,
                "environment_seed": assignment["environment_seed"],
                "initial_state_identifier": str(assignment["initial_state_identifier"]),
                "query_index": query_index,
                "policy_query_index": query_index,
                "environment_step": int(current["frame_index"]),
                "absolute_environment_step": int(current["frame_index"]),
                "raw_images": raw_images,
                "raw_image_identity": tree_hash(raw_images),
                "preprocessed_images": preprocessed_images,
                "preprocessed_image_hash": tree_hash(preprocessed_images),
                "robot_state_raw": torch.as_tensor(current["state"]).cpu(),
                "robot_state_normalized": extraction.robot_state[0].detach().cpu(),
                **state_payload,
                "action_noise": noise[0].detach().cpu(),
                "action_noise_seed": seed,
                "action_noise_hash": array_hash(noise[0]),
                "teacher_action_chunk_normalized": teacher[0].detach().cpu(),
                "teacher_action_chunk_postprocessed": torch.as_tensor(postprocessed).cpu(),
                "executed_actions": executed,
                "executed_actions_postprocessed": executed,
                "executed_action_length": len(executed),
                "gripper_conversion": "LiberoOutputs continuous source-correct 7D action; no binary target rewrite",
                "next_query_observation": compact_next_observation(next_sample),
                "source_lock_id": gate["source_lock_id"],
                "source_hashes": source_hashes,
                "timing_ms": {
                    "prefix_embedding": extraction.prefix_embedding_ms,
                    "full_prefix": extraction.full_prefix_ms,
                    **action_timing,
                },
            }
            tensor_contract = tensor_contract_from_record(record)
            if writer is None:
                validate_record_v2(
                    record,
                    expected_contract=tensor_contract,
                    expected_source_lock_id=gate["source_lock_id"],
                    expected_source_hashes=source_hashes,
                    task_catalog=task_catalog,
                    execution_horizon=args.execution_horizon,
                )
                metadata = {
                    "schema_version": 2,
                    "cache_scope": "full_shard_v2" if full_cache else "bounded_smoke_v2",
                    "source_lock": str(Path(args.source_lock).resolve()),
                    "source_lock_id": gate["source_lock_id"],
                    "source_hashes": source_hashes,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "final_evaluation_manifest": str(Path(args.final_evaluation_manifest).resolve()),
                    "final_evaluation_manifest_sha256": _sha256(args.final_evaluation_manifest),
                    "split_contract": str(Path(args.split_contract).resolve()),
                    "split_contract_sha256": _sha256(args.split_contract),
                    "tensor_contract": tensor_contract,
                    "action_horizon_h": tensor_contract["action_horizon_h"],
                    "execution_horizon_r": args.execution_horizon,
                    "noise_seed_base": args.noise_seed_base,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                }
                if full_cache:
                    metadata.update(
                        {
                            "full_cache_inventory": str(
                                Path(args.full_cache_inventory).resolve()
                            ),
                            "full_cache_inventory_id": inventory["inventory_id"],
                            "full_cache_inventory_sha256": sha256_file(
                                args.full_cache_inventory
                            ),
                            "expected_shard_episodes": int(
                                inventory["statistics"]["episodes_by_shard"][str(args.shard_index)]
                            ),
                            "expected_shard_queries": int(
                                inventory["statistics"]["queries_by_shard"][str(args.shard_index)]
                            ),
                        }
                    )
                else:
                    metadata.update(
                        {
                            "repair_stage_max_episodes": args.max_episodes,
                            "repair_stage_max_queries_per_episode": args.max_queries_per_episode,
                        }
                    )
                writer = EpisodeCacheWriter(output, metadata)
            validate_record_v2(
                record,
                expected_contract=writer.metadata["tensor_contract"],
                expected_source_lock_id=gate["source_lock_id"],
                expected_source_hashes=source_hashes,
                task_catalog=task_catalog,
                execution_horizon=args.execution_horizon,
            )
            records.append(record)
            total_records += 1
            if not full_cache and len(records) >= args.max_queries_per_episode:
                break
        if inventory_episode is not None and len(records) != int(inventory_episode["query_count"]):
            raise RuntimeError(f"generated query count differs from full-cache inventory: {episode_key}")
        if records:
            assert writer is not None
            writer.write_episode(
                records,
                suite=suite,
                task_id=benchmark_task_index,
                episode_id=episode_id,
                split=str(assignment["role"]),
            )
        print(
            f"episode {selection_index + 1}/{len(selected)} id={episode_id} "
            f"suite={suite} task={benchmark_task_index} records={len(records)}",
            flush=True,
        )
    if writer is None:
        raise RuntimeError("no source-locked cache record was produced; output root was not created")
    statistics = {
        "episodes": len(writer.entries),
        "records": total_records,
        "elapsed_seconds": time.time() - started_at,
        "schema_v2": True,
    }
    if full_cache:
        expected_episodes = int(inventory["statistics"]["episodes_by_shard"][str(args.shard_index)])
        expected_queries = int(inventory["statistics"]["queries_by_shard"][str(args.shard_index)])
        if (statistics["episodes"], statistics["records"]) != (
            expected_episodes,
            expected_queries,
        ):
            raise RuntimeError("full-cache shard totals differ from the frozen inventory")
    manifest = writer.finalize(statistics)
    print(manifest)


if __name__ == "__main__":
    main()
