"""Gated Stage-T2 on-policy query-boundary cache generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
LIBERO_ROOT = UPSTREAM / "evaluation" / "libero" / "LIBERO"
for path in (ROOT, UPSTREAM, LIBERO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter, SimVLAConditionAdapter  # noqa: E402
from architectures.simvla.adapters.latentloop.checkpoint import (  # noqa: E402
    freeze_module,
    load_adapter_checkpoint,
)
from architectures.simvla.adapters.latentloop.query_cache_generator import (  # noqa: E402
    query_snapshot,
    transition_record,
)
from architectures.simvla.adapters.latentloop.source_lock import (  # noqa: E402
    collect_source_lock,
    require_empty_output,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (  # noqa: E402
    RealSimVLADCLDPolicy,
    build_env_obs,
    get_libero_env,
)
from methods.latentloop.training import (  # noqa: E402
    QueryCacheShardWriter,
    validate_on_policy_cache,
)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_gate(path: str, execution_horizon: int, maximum_rollout_depth: int) -> dict[str, Any]:
    gate = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ["T1_K2_OFFLINE_PASS", "T1_K2_ONLINE_PASS"]
    if maximum_rollout_depth >= 3:
        required.append("R5_K3_PASS")
    failed = [name for name in required if not bool(gate.get(name, False))]
    if failed:
        raise RuntimeError(f"on-policy generation is blocked by gate fields: {failed}")
    if int(gate.get("execution_horizon", execution_horizon)) != execution_horizon:
        raise RuntimeError("gate execution horizon does not match this cache")
    return gate


def _adapter_next_condition(
    *,
    adapter: Any,
    previous_condition: torch.Tensor,
    current: dict[str, Any],
    next_query: dict[str, Any],
    executed_actions: torch.Tensor,
    execution_horizon: int,
    elapsed_time: float,
    next_query_age: int,
) -> torch.Tensor:
    """Apply one recursive adapter update using the actions actually executed."""

    device = previous_condition.device
    observation = adapter.encode_observation(
        current["raw_rgb"].unsqueeze(0).to(device),
        next_query["raw_rgb"].unsqueeze(0).to(device),
        current["proprio"].unsqueeze(0).to(device),
        next_query["proprio"].unsqueeze(0).to(device),
    )
    action_feature = adapter.encode_executed_actions(
        executed_actions.unsqueeze(0).to(device),
        execution_horizon,
        elapsed_time,
        reference_feature=observation,
    )
    return adapter.update_recurrent_condition(
        previous_condition,
        observation,
        action_feature,
        execution_horizon=execution_horizon,
        elapsed_time=elapsed_time,
        query_age=next_query_age,
    )


def _execute_subchunk(
    env: Any,
    action_chunk: torch.Tensor,
    execution_horizon: int,
) -> tuple[Any, bool, list[torch.Tensor]]:
    """Execute at most R final postprocessed actions and return exact sent tensors."""

    obs = None
    done = False
    sent: list[torch.Tensor] = []
    for action in action_chunk[0, :execution_horizon]:
        obs, _, done, _ = env.step(action.detach().cpu().tolist())
        sent.append(action.detach().cpu())
        if done:
            break
    return obs, bool(done), sent


def generate_on_policy_cache(args: argparse.Namespace) -> dict[str, Any]:
    """Roll out the adapter while a frozen full SimVLA branch logs T2 targets."""

    from libero.libero import benchmark
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    output = require_empty_output(args.output)
    gate = _load_gate(
        args.gate_decision_json,
        args.execution_horizon,
        args.maximum_rollout_depth,
    )
    source_lock = collect_source_lock(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    metadata = {
        "cache_kind": "on_policy_recursive_distillation",
        "source_lock": source_lock,
        "adapter_checkpoint": str(Path(args.adapter_checkpoint).resolve()),
        "gate": gate,
        "execution_horizon": args.execution_horizon,
        "maximum_rollout_depth": args.maximum_rollout_depth,
        "flow_steps": args.flow_steps,
        "client_resize_size": args.client_resize_size,
        "image_size": args.image_size,
        "task_order": args.task_order,
    }
    writer = QueryCacheShardWriter(
        output,
        execution_horizon=args.execution_horizon,
        metadata=metadata,
        records_per_shard=args.records_per_shard,
    )
    (output / "source_lock.json").write_text(
        json.dumps(source_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "gate_snapshot.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    device = torch.device(args.device)
    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device)
    model.action_space.load_norm_stats(args.norm_stats)
    freeze_module(model)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)
    condition_adapter = SimVLAConditionAdapter(model)
    action_adapter = SimVLAActionAdapter(model)
    adapter, adapter_payload = load_adapter_checkpoint(args.adapter_checkpoint, device=device)
    if adapter.variant != "chunk_aware_latentloop":
        raise ValueError("T2 on-policy generation requires a chunk-aware checkpoint")
    adapter.eval()
    freeze_module(adapter)
    os.environ.setdefault("LIBERO_ROOT", str(LIBERO_ROOT))
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = (
        list(range(suite.n_tasks - 1, -1, -1))
        if args.task_order == "official_reverse"
        else list(range(suite.n_tasks))
    )[: args.max_tasks]
    progress_path = output / "generation_progress.jsonl"
    records = 0
    episodes = 0
    successes = 0
    incomplete = 0
    depth_counts: dict[int, int] = {}
    started = time.time()
    elapsed_time = args.execution_horizon / float(args.control_hz)
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, prompt = get_libero_env(task, args.resolution, args.seed)
        try:
            for trial_id in range(args.num_trials):
                episodes += 1
                episode_id = f"task{task_id:02d}_trial{trial_id:03d}"
                env.reset()
                obs = env.set_init_state(init_states[trial_id % len(init_states)])
                env_timestep = 0
                done = False
                for _ in range(args.num_wait_steps):
                    obs, _, done, _ = env.step([0.0] * 6 + [-1.0])
                    env_timestep += 1
                preprocess_policy = RealSimVLADCLDPolicy(
                    model=model,
                    processor=processor,
                    dcld_core=None,
                    mode="full",
                    refresh_every=1,
                    flow_steps=args.flow_steps,
                    image_size=args.image_size,
                    replan_steps=args.execution_horizon,
                    client_resize_size=args.client_resize_size,
                    device=device,
                )
                image0, image1, proprio = build_env_obs(obs)
                current = query_snapshot(
                    policy=preprocess_policy,
                    condition_adapter=condition_adapter,
                    action_adapter=action_adapter,
                    image0=image0,
                    image1=image1,
                    proprio=proprio,
                    prompt=prompt,
                    checkpoint=args.checkpoint,
                    task_id=task_id,
                    episode_id=episode_id,
                    query_index=0,
                    env_timestep=env_timestep,
                    seed_base=args.action_noise_seed_base,
                    flow_steps=args.flow_steps,
                )
                query_index = 0
                segment_index = 0
                while not done and query_index < args.max_policy_queries:
                    # Warm up one recursive prediction from an exact full anchor.
                    obs, done, warmup_actions = _execute_subchunk(
                        env,
                        current["action_chunk_device"],
                        args.execution_horizon,
                    )
                    env_timestep += len(warmup_actions)
                    if len(warmup_actions) != args.execution_horizon or done:
                        incomplete += 1
                        break
                    next_query_index = query_index + 1
                    image0, image1, proprio = build_env_obs(obs)
                    next_query = query_snapshot(
                        policy=preprocess_policy,
                        condition_adapter=condition_adapter,
                        action_adapter=action_adapter,
                        image0=image0,
                        image1=image1,
                        proprio=proprio,
                        prompt=prompt,
                        checkpoint=args.checkpoint,
                        task_id=task_id,
                        episode_id=episode_id,
                        query_index=next_query_index,
                        env_timestep=env_timestep,
                        seed_base=args.action_noise_seed_base,
                        flow_steps=args.flow_steps,
                    )
                    with torch.no_grad():
                        predicted_condition = _adapter_next_condition(
                            adapter=adapter,
                            previous_condition=current["condition_device"].detach(),
                            current=current,
                            next_query=next_query,
                            executed_actions=torch.stack(warmup_actions),
                            execution_horizon=args.execution_horizon,
                            elapsed_time=elapsed_time,
                            next_query_age=1,
                        ).detach()
                    current = next_query
                    query_index = next_query_index
                    cache_episode_id = f"{episode_id}_segment{segment_index:04d}"
                    for rollout_depth in range(1, args.maximum_rollout_depth + 1):
                        if done or query_index >= args.max_policy_queries:
                            break
                        current_noise = current["initial_noise"].unsqueeze(0).to(device)
                        with torch.no_grad():
                            predicted_action = action_adapter.decode_action_from_condition(
                                predicted_condition,
                                current["proprio_device"],
                                steps=args.flow_steps,
                                initial_noise=current_noise,
                            )
                        obs, done, actions_sent = _execute_subchunk(
                            env,
                            predicted_action,
                            args.execution_horizon,
                        )
                        env_timestep += len(actions_sent)
                        if len(actions_sent) != args.execution_horizon or done:
                            incomplete += 1
                            break
                        next_query_index = query_index + 1
                        image0, image1, proprio = build_env_obs(obs)
                        next_query = query_snapshot(
                            policy=preprocess_policy,
                            condition_adapter=condition_adapter,
                            action_adapter=action_adapter,
                            image0=image0,
                            image1=image1,
                            proprio=proprio,
                            prompt=prompt,
                            checkpoint=args.checkpoint,
                            task_id=task_id,
                            episode_id=episode_id,
                            query_index=next_query_index,
                            env_timestep=env_timestep,
                            seed_base=args.action_noise_seed_base,
                            flow_steps=args.flow_steps,
                        )
                        executed = torch.stack(actions_sent)
                        record = transition_record(
                            current=current,
                            next_query=next_query,
                            executed_actions=executed,
                            task_id=task_id,
                            episode_id=cache_episode_id,
                            task_identifier=f"{task.problem_folder}/{task.bddl_file}",
                            language_instruction=prompt,
                            execution_horizon=args.execution_horizon,
                            elapsed_time=elapsed_time,
                            provenance={
                                "checkpoint": args.checkpoint,
                                "norm_stats": args.norm_stats,
                                "adapter_checkpoint": str(Path(args.adapter_checkpoint).resolve()),
                                "adapter_step": int(adapter_payload.get("step", -1)),
                                "task_bddl": f"{task.problem_folder}/{task.bddl_file}",
                                "init_state_index": trial_id % len(init_states),
                                "rollout_episode_id": episode_id,
                                "current_action_noise_seed": current["action_noise_seed"],
                                "next_action_noise_seed": next_query["action_noise_seed"],
                                "rollout_semantics": "predicted_condition_action_executed_teacher_logging_only",
                            },
                        )
                        record["predicted_condition"] = predicted_condition[0].detach().cpu()
                        record["rollout_depth"] = int(rollout_depth)
                        record["rollout_episode_id"] = episode_id
                        record["adapter_checkpoint"] = str(Path(args.adapter_checkpoint).resolve())
                        writer.add(record)
                        records += 1
                        depth_counts[rollout_depth] = depth_counts.get(rollout_depth, 0) + 1
                        with torch.no_grad():
                            predicted_condition = _adapter_next_condition(
                                adapter=adapter,
                                previous_condition=predicted_condition,
                                current=current,
                                next_query=next_query,
                                executed_actions=executed,
                                execution_horizon=args.execution_horizon,
                                elapsed_time=elapsed_time,
                                next_query_age=rollout_depth + 1,
                            ).detach()
                        current = next_query
                        query_index = next_query_index
                    segment_index += 1
                successes += int(done)
                _append_jsonl(
                    progress_path,
                    {
                        "task_id": task_id,
                        "trial_id": trial_id,
                        "records_total": records,
                        "rollout_depth_counts": depth_counts,
                        "success": bool(done),
                        "elapsed_seconds": time.time() - started,
                    },
                )
        finally:
            env.close()
    writer.close()
    validation = validate_on_policy_cache(
        output,
        maximum_rollout_depth=args.maximum_rollout_depth,
    )
    summary = {
        "cache_dir": str(output),
        "records": records,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / max(episodes, 1),
        "rollout_depth_counts": depth_counts,
        "incomplete_terminal_transitions_discarded": incomplete,
        "validation": validation,
    }
    (output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    """Parse gated on-policy cache generation arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--gate-decision-json", required=True)
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--execution-horizon", type=int, choices=(1, 2, 5), required=True)
    parser.add_argument("--maximum-rollout-depth", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-policy-queries", type=int, default=3)
    parser.add_argument("--num-wait-steps", type=int, default=10)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--client-resize-size", type=int, default=224)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-noise-seed-base", type=int, default=20260804)
    parser.add_argument("--task-order", choices=("official_reverse", "ascending"), default="official_reverse")
    parser.add_argument("--records-per-shard", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    result = generate_on_policy_cache(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["validation"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
