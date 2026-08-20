#!/usr/bin/env python3
"""Generate episode-sharded pi0.5 teacher states at native R=5 boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from _common import DEFAULT_CHECKPOINT, load_local_policy, refuse_nonempty_output, require_gate, require_run
from architectures.openpi.adapters.latentloop.cache_io import EpisodeCacheWriter, episode_split
from architectures.openpi.adapters.latentloop.policy_io import (
    explicit_policy_noise,
    policy_noise_seed,
    postprocess_policy_actions,
    prepare_policy_observation,
)
from architectures.openpi.adapters.latentloop.prefix_kv_hook import PrefixKVHook
from architectures.openpi.adapters.latentloop.serialization import prefix_state_to_dict


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def suite_for_task(task_index: int) -> tuple[str, int]:
    if not 0 <= task_index < 40:
        raise ValueError(f"unexpected LIBERO task index: {task_index}")
    return SUITES[task_index // 10], task_index % 10


def image_uint8(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def raw_policy_observation(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation/image": image_uint8(sample["image"]),
        "observation/wrist_image": image_uint8(sample["wrist_image"]),
        "observation/state": np.asarray(sample["state"]),
        "prompt": str(sample["prompt"]),
    }


def compact_next_observation(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "image": image_uint8(sample["image"]),
        "wrist_image": image_uint8(sample["wrist_image"]),
        "state": torch.as_tensor(sample["state"]).cpu(),
        "frame_index": int(sample["frame_index"]),
    }


def main() -> None:
    raise RuntimeError(
        "DISABLED_SUPERSEDED_CACHE_V1: use generate_pi05_latentloop_cache_v2.py with "
        "source-lock-v2, real K1/freeze gates, a frozen final-evaluation manifest, and an explicit split contract"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--k1-gate", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--execution-horizon", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260820)
    parser.add_argument("--noise-seed-base", type=int, default=20260820)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--suite", choices=("all", *SUITES), default="all")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require_run(args.run, "OPENPI_LATENTLOOP_CACHE_RUN")
    if args.execution_horizon != 5:
        raise ValueError("the pinned LIBERO execution horizon is R=5")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0,num-shards)")
    source_lock = require_gate(args.source_lock, "source_lock_pass")
    if Path(source_lock["checkpoint"]["directory"]).resolve() != Path(args.checkpoint).resolve():
        raise ValueError("cache checkpoint does not match the frozen source lock")
    require_gate(args.k1_gate, "hard_gate_pass")
    output = refuse_nonempty_output(args.output)

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
    if model.config.action_horizon != 10:
        raise ValueError("source-audited pi0.5 action horizon must be H=10")
    hook = PrefixKVHook(model)
    source_hashes = {
        "source": source_lock["openpi_source"]["combined_sha256"],
        "checkpoint": source_lock["checkpoint"]["model_sha256"],
        "norm_stats": source_lock["normalization"]["sha256"],
        "config": source_lock["checkpoint"]["config_sha256"],
    }
    metadata = {
        "schema_version": 1,
        "source_lock": str(Path(args.source_lock).resolve()),
        "source_hashes": source_hashes,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": "physical-intelligence/libero demonstrations",
        "dataset_episodes": len(starts),
        "dataset_frames": len(dataset),
        "action_horizon_h": 10,
        "execution_horizon_r": 5,
        "delta_q_supported": [1, 2, 3],
        "delta_a_supported": [5, 10, 15],
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "calibration_fraction": args.calibration_fraction,
        "noise_seed_base": args.noise_seed_base,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "action_grounding": (
            "logged postprocessed LIBERO demonstration actions that actually connect consecutive cached observations"
        ),
        "teacher_actions": "frozen local pi0.5 same-noise full action chunks",
        "evaluation_leakage": "no final LIBERO evaluation initial-state episode is used",
    }
    writer = EpisodeCacheWriter(output, metadata)
    selected = [episode for episode in range(len(starts)) if episode % args.num_shards == args.shard_index]
    if args.max_episodes is not None:
        selected = selected[: args.max_episodes]
    total_records = 0
    total_bytes_uncompressed = 0
    started_at = time.time()

    for selection_index, episode_id in enumerate(selected):
        start, stop = int(starts[episode_id]), int(stops[episode_id])
        identity_sample = dataset[start]
        suite, task_id = suite_for_task(int(identity_sample["task_index"]))
        if args.suite != "all" and suite != args.suite:
            continue
        split = episode_split(
            suite,
            task_id,
            episode_id,
            seed=args.split_seed,
            validation_fraction=args.validation_fraction,
            calibration_fraction=args.calibration_fraction,
        )
        records: list[dict[str, Any]] = []
        for query_index, frame in enumerate(range(start, stop - args.execution_horizon, args.execution_horizon)):
            current = dataset[frame]
            next_sample = dataset[frame + args.execution_horizon]
            if bool(torch.as_tensor(current["actions_is_pad"][: args.execution_horizon]).any()):
                continue
            raw_observation = raw_policy_observation(current)
            observation, transformed = prepare_policy_observation(policy, raw_observation)
            extraction = hook.extract(observation)
            seed = policy_noise_seed(args.noise_seed_base, suite, task_id, episode_id, query_index)
            noise = explicit_policy_noise(
                (1, model.config.action_horizon, model.config.action_dim), seed=seed, device=args.device
            )
            teacher, action_timing = hook.sample_actions_from_state(
                extraction.state, extraction.robot_state, noise, num_steps=args.flow_steps
            )
            postprocessed = postprocess_policy_actions(policy, transformed["state"], teacher)["actions"]
            preprocessed_images = {
                key: value[0].detach().cpu().to(torch.float16)
                for key, value in observation.images.items()
            }
            state_payload = prefix_state_to_dict(extraction.state)
            record = {
                "suite": suite,
                "task_id": task_id,
                "episode_id": episode_id,
                "query_index": query_index,
                "environment_step": int(current["frame_index"]),
                "language_instruction": str(current["prompt"]),
                "raw_images": {
                    "base_0_rgb": raw_observation["observation/image"],
                    "left_wrist_0_rgb": raw_observation["observation/wrist_image"],
                },
                "preprocessed_images": preprocessed_images,
                "robot_state_raw": torch.as_tensor(current["state"]).cpu(),
                "robot_state_normalized": extraction.robot_state[0].detach().cpu(),
                **state_payload,
                "action_noise": noise[0].detach().cpu(),
                "action_noise_seed": seed,
                "teacher_action_chunk_normalized": teacher[0].detach().cpu(),
                "teacher_action_chunk_postprocessed": torch.as_tensor(postprocessed).cpu(),
                "executed_actions": torch.as_tensor(
                    current["actions"][: args.execution_horizon], dtype=torch.float32
                ).cpu(),
                "next_query_observation": compact_next_observation(next_sample),
                "source_hashes": source_hashes,
                "timing_ms": {
                    "prefix_embedding": extraction.prefix_embedding_ms,
                    "full_prefix": extraction.full_prefix_ms,
                    **action_timing,
                },
            }
            records.append(record)
            total_records += 1
            total_bytes_uncompressed += sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    state_payload["prefix_embeddings"],
                    state_payload["pre_rope_keys"],
                    state_payload["values"],
                )
            )
        if records:
            writer.write_episode(
                records,
                suite=suite,
                task_id=task_id,
                episode_id=episode_id,
                split=split,
            )
        print(
            f"episode {selection_index + 1}/{len(selected)} id={episode_id} "
            f"suite={suite} records={len(records)} elapsed={time.time() - started_at:.1f}s",
            flush=True,
        )

    manifest_path = writer.finalize(
        {
            "records": total_records,
            "kv_embedding_uncompressed_bytes": total_bytes_uncompressed,
            "elapsed_seconds": time.time() - started_at,
        }
    )
    report = output / "pi05_latentloop_cache_report.md"
    report.write_text(
        "\n".join(
            (
                "# pi0.5 LatentLoop cache report",
                "",
                f"- Episodes: `{len(writer.entries)}`",
                f"- Query records: `{total_records}`",
                f"- KV + embedding tensor bytes before container overhead: `{total_bytes_uncompressed}`",
                "- H/R: `10/5`",
                "- Split unit: episode",
                "- Action grounding: logged demonstration actions that connect cached observations",
                "- Teacher target: frozen local pi0.5 full KV and same-noise action chunk",
                "- Final 2,000 evaluation episodes: not used",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
