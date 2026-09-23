"""Collect official-style R0 feature transitions from full SimVLA rollouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    build_env_obs,
    get_libero_env,
)

from .condition_hook import SimVLAConditionWithStableHook
from .dataset import SYNC_SCHEMA, sha256_file
from .eval import SynchronizedFullPolicy, _configure_paths, _load_simvla, _write_json
from .provenance import (
    latent_bridge_source_manifest,
    simvla_latent_bridge_integration_manifest,
)
from .recipe import STABLE_LAYER_MIN_COSINE


def _git(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


class RecordingFullPolicy(SynchronizedFullPolicy):
    """Full SimVLA policy that records one feature tuple per policy query."""

    def __init__(self, *, stable_layer_index: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.records: list[dict[str, torch.Tensor]] = []
        self.condition_hook = SimVLAConditionWithStableHook(
            self.model, stable_layer_index=stable_layer_index
        )

    def close(self) -> None:
        self.condition_hook.close()

    def _full_refresh(
        self,
        batch: dict[str, torch.Tensor],
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        self._sync()
        started = time.perf_counter()
        extracted = self.condition_hook.encode(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
        self._sync()
        self.metrics.latencies["VLM_encoder_ms"].append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_full_vlm_calls"] += 1
        condition = extracted.condition
        action, seed = self._decode(
            condition, batch["proprio"], policy_query_index=policy_query_index
        )
        self.records.append(
            {
                "condition": condition[0].detach().to("cpu", torch.bfloat16),
                "stable": extracted.stable[0].detach().to("cpu", torch.bfloat16),
                "state": batch["proprio"][0].detach().to("cpu", torch.float32),
                "action": action[0, 0].detach().to("cpu", torch.float32),
            }
        )
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action.detach()
        return condition, action, seed


def _transitions(records: list[dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    return [
        {
            "condition_input": records[index]["condition"],
            "condition_target": records[index + 1]["condition"],
            "stable_anchor": records[index]["stable"],
            "state": records[index]["state"],
            "previous_action": records[index]["action"],
            "age": 1,
            "policy_query_index": index,
        }
        for index in range(len(records) - 1)
    ]


def _adjacent_feature_statistics(
    records: list[dict[str, torch.Tensor]], key: str
) -> list[float]:
    values: list[float] = []
    for index in range(len(records) - 1):
        left = records[index][key].float().reshape(1, -1)
        right = records[index + 1][key].float().reshape(1, -1)
        values.append(float(F.cosine_similarity(left, right, dim=-1).item()))
    return values


def collect(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_trials < 2 or args.trial_offset < 0:
        raise ValueError("sync collection requires at least two trials and non-negative offset")
    upstream, libero = _configure_paths()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing sync output: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    device = torch.device(args.device)
    model, processor = _load_simvla(args, device)
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_count = min(int(suite.get_num_tasks()), args.max_tasks or int(suite.get_num_tasks()))
    task_ids = list(reversed(range(task_count)))
    shards: list[dict[str, Any]] = []
    stable_adjacent_cosines: list[float] = []
    final_adjacent_cosines: list[float] = []
    successes = 0
    progress = tqdm(
        total=task_count * args.num_trials, desc="full SimVLA sync collection", dynamic_ncols=True
    )
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, prompt = get_libero_env(task, 256, args.environment_seed)
        try:
            for local_trial_id in range(args.num_trials):
                trial_id = args.trial_offset + local_trial_id
                env.reset()
                obs = env.set_init_state(init_states[trial_id % len(init_states)])
                for _ in range(10):
                    obs, _, done, _ = env.step([0.0] * 6 + [-1.0])
                policy = RecordingFullPolicy(
                    model=model,
                    processor=processor,
                    dcld_core=None,
                    mode="full",
                    refresh_every=1,
                    flow_steps=10,
                    image_size=384,
                    replan_steps=5,
                    client_resize_size=224,
                    device=device,
                    suite=args.suite,
                    row_name="sync_full_simvla",
                    task_id=task_id,
                    trial_id=trial_id,
                    paired_action_noise=True,
                    action_noise_seed_base=args.action_noise_seed_base,
                    log_action_chunks=False,
                    stable_layer_index=args.stable_layer_index,
                )
                success = False
                try:
                    for _ in range(args.max_policy_steps):
                        image0, image1, proprio = build_env_obs(obs)
                        action = policy.act(image0, image1, proprio, prompt)
                        obs, _, done, _ = env.step(action.action.tolist())
                        if done:
                            success = True
                            break
                finally:
                    policy.close()
                transitions = _transitions(policy.records)
                stable_adjacent_cosines.extend(
                    _adjacent_feature_statistics(policy.records, "stable")
                )
                final_adjacent_cosines.extend(
                    _adjacent_feature_statistics(policy.records, "condition")
                )
                shard = temporary / f"task{task_id:02d}_trial{trial_id:03d}.pt"
                torch.save(
                    {
                        "schema_version": SYNC_SCHEMA,
                        "task_id": task_id,
                        "trial_id": trial_id,
                        "success": success,
                        "policy_queries": len(policy.records),
                        "transitions": transitions,
                    },
                    shard,
                )
                shards.append(
                    {
                        "file": shard.name,
                        "sha256": sha256_file(shard),
                        "size_bytes": shard.stat().st_size,
                        "task_id": task_id,
                        "trial_id": trial_id,
                        "success": success,
                        "policy_queries": len(policy.records),
                        "transitions": len(transitions),
                    }
                )
                successes += int(success)
                progress.update(1)
                progress.set_postfix(sr=f"{100 * successes / len(shards):.1f}%")
        finally:
            env.close()
    progress.close()
    if not stable_adjacent_cosines or not final_adjacent_cosines:
        raise RuntimeError("sync collection produced no adjacent feature pairs")
    stable_mean = float(sum(stable_adjacent_cosines) / len(stable_adjacent_cosines))
    final_mean = float(sum(final_adjacent_cosines) / len(final_adjacent_cosines))
    stable_layer_contract = {
        "metric": "mean flattened adjacent-query cosine",
        "threshold_source": "Latent Bridge paper Section 3.2, stable layer cosine > 0.999",
        "threshold": STABLE_LAYER_MIN_COSINE,
        "stable_layer_index": args.stable_layer_index,
        "stable_adjacent_cosine": stable_mean,
        "final_condition_adjacent_cosine": final_mean,
        "pairs": len(stable_adjacent_cosines),
        "passed": stable_mean > STABLE_LAYER_MIN_COSINE,
    }
    manifest = {
        "schema_version": SYNC_SCHEMA,
        "data_role": "on_policy_frozen_full_simvla_rollouts",
        "checkpoint": args.checkpoint,
        "norm_stats": str(Path(args.norm_stats).expanduser().resolve()),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "simvla_upstream_root": str(upstream),
        "simvla_upstream_commit": _git(upstream),
        "libero_root": str(libero),
        "libero_commit": _git(libero),
        "latent_bridge_upstream": latent_bridge_source_manifest(),
        "simvla_latent_bridge_integration": simvla_latent_bridge_integration_manifest(),
        "suite": args.suite,
        "task_ids": task_ids,
        "trial_offset": args.trial_offset,
        "trials_per_task": args.num_trials,
        "environment_seed": args.environment_seed,
        "action_noise_seed_base": args.action_noise_seed_base,
        "action_horizon": 10,
        "execution_horizon": 5,
        "flow_steps": 10,
        "stable_layer_index": args.stable_layer_index,
        "stable_layer_contract": stable_layer_contract,
        "shards": shards,
        "episodes": len(shards),
        "successes": successes,
        "total_transitions": sum(int(item["transitions"]) for item in shards),
        "final_evaluation_episodes_used": False,
        "renderer": {
            key: os.environ.get(key)
            for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "EGL_DEVICE_ID")
        },
    }
    _write_json(temporary / "manifest.json", manifest)
    os.replace(temporary, output)
    return {
        "verdict": "SIMVLA_LATENT_BRIDGE_SYNC_COLLECTION_COMPLETE",
        "output": str(output),
        "episodes": len(shards),
        "successes": successes,
        "transitions": manifest["total_transitions"],
        "stable_layer_contract": stable_layer_contract,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", required=True)
    value.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    value.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    value.add_argument("--norm-stats", required=True)
    value.add_argument(
        "--suite",
        choices=("libero_10", "libero_spatial", "libero_object", "libero_goal"),
        default="libero_10",
    )
    value.add_argument("--num-trials", type=int, default=30)
    value.add_argument("--trial-offset", type=int, default=0)
    value.add_argument("--max-tasks", type=int)
    value.add_argument("--max-policy-steps", type=int, default=900)
    value.add_argument("--action-noise-seed-base", type=int, default=20260901)
    value.add_argument("--environment-seed", type=int, default=7)
    value.add_argument("--stable-layer-index", type=int, default=10)
    value.add_argument("--device", default="cuda")
    return value


def main() -> None:
    result = collect(parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
