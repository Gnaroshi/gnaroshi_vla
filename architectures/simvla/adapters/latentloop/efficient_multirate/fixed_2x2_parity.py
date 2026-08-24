"""Bounded real-LIBERO parity and counter gate for the fixed 2x2 rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_contracts import (
    COMBINED_ROW,
    CONDITION_ROW,
    FROZEN_CONDITION_CHECKPOINT_SHA256,
    FROZEN_CONDITION_SOURCE_SHA256,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval import (
    _ensure_generation_latency_schema,
    _make_policy,
    _validate_fixed_2x2_counters,
    _verify_provenance,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    load_generation_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FROZEN_CHECKPOINT_REVISION,
    FROZEN_GENERATION_CHECKPOINT_SHA256,
    FROZEN_GENERATION_SOURCE_SHA256,
    FROZEN_NORM_STATS_SHA256,
    atomic_write_json,
    load_json,
    require_egl_preflight,
    validate_manifest_identity,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_policy import (
    RealSimVLAGenerationPolicy,
)
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    load_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.native_v0_long_eval import (
    _SynchronizedFullPolicy,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    build_env_obs,
    get_libero_env,
)


def _queue_tensor(policy: Any) -> torch.Tensor:
    return torch.stack([action for action, _ in policy.action_queue], dim=0)


def _counter_gate(policy: Any, row: str) -> dict[str, Any]:
    counters = policy.metrics.counters
    return _validate_fixed_2x2_counters(
        row,
        policy_queries=int(counters.get("num_policy_queries", 0)),
        full_vlm_calls=int(counters.get("num_full_vlm_calls", 0)),
        condition_updater_calls=int(counters.get("num_condition_updater_calls", 0)),
        full_action_transformer_calls=int(
            counters.get("num_action_transformer_calls", 0)
        ),
        generation_loop_updates=int(
            counters.get("num_generation_decoder_only_steps", 0)
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing parity output: {output}")
    physical_gpu_id = int(args.physical_gpu_id)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu_id):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly the requested GPU")
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(physical_gpu_id):
        raise RuntimeError("MUJOCO_EGL_DEVICE_ID must equal the physical GPU ID")
    provenance = _verify_provenance(args)
    preflight = require_egl_preflight(args.egl_preflight, physical_gpu_id)
    manifest = load_json(args.manifest)
    manifest_report = validate_manifest_identity(
        manifest, expected_manifest_sha256=args.expected_manifest_sha256
    )
    if manifest_report["verdict"] != "EPISODE_MANIFEST_PASS":
        raise RuntimeError(json.dumps(manifest_report, indent=2, sort_keys=True))
    configure_strict_torch_determinism(int(manifest["determinism_seed"]))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("parity requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    bundle = Path(args.bundle_root).expanduser().resolve()
    norm_stats = bundle / "norm" / "libero_norm_official_32700d0.json"
    model, processor, _ = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    freeze_module(model)
    condition_updater, condition_payload = load_native_v0_checkpoint(
        args.condition_checkpoint, device=device, require_final_150k=True
    )
    condition_source = condition_payload.get("source_lock", {})
    if int(condition_payload.get("global_optimizer_step", -1)) != 150_000:
        raise RuntimeError("Condition checkpoint step mismatch")
    if condition_source.get("combined_sha256") != FROZEN_CONDITION_SOURCE_SHA256:
        raise RuntimeError("Condition checkpoint source mismatch")
    if condition_source.get("norm_stats_sha256") != FROZEN_NORM_STATS_SHA256:
        raise RuntimeError("Condition checkpoint norm mismatch")
    if condition_source.get("checkpoint", {}).get("revision") != FROZEN_CHECKPOINT_REVISION:
        raise RuntimeError("Condition checkpoint SimVLA revision mismatch")
    freeze_module(condition_updater)
    generation_updater, generation_payload = load_generation_checkpoint(
        bundle / "checkpoint" / "generation_step_030000.pt", device=device
    )
    if int(generation_payload.get("optimizer_step", -1)) != 30_000:
        raise RuntimeError("Generation checkpoint step mismatch")
    if generation_payload.get("source_lock", {}).get("combined_sha256") != FROZEN_GENERATION_SOURCE_SHA256:
        raise RuntimeError("Generation checkpoint source mismatch")
    freeze_module(generation_updater)

    from libero.libero import benchmark

    first = sorted(
        manifest["episodes"], key=lambda item: (int(item["task_id"]), int(item["trial_id"]))
    )[0]
    task_id = int(first["task_id"])
    trial_id = int(first["trial_id"])
    suite = benchmark.get_benchmark_dict()[str(manifest["suite"])]()
    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)
    env, prompt = get_libero_env(
        task, int(manifest["environment_resolution"]), int(manifest["environment_seed"])
    )
    try:
        env.reset()
        observation0 = env.set_init_state(
            initial_states[int(first["init_state_index"]) % len(initial_states)]
        )
        for _ in range(int(manifest["num_wait_steps"])):
            observation0, _, _, _ = env.step([0.0] * 6 + [-1.0])
        observation1, _, _, _ = env.step([0.0] * 6 + [-1.0])
    finally:
        env.close()
    image00, image01, proprio0 = build_env_obs(observation0)
    image10, image11, proprio1 = build_env_obs(observation1)
    seed_base = int(manifest["action_noise_seed_base"])

    full = _SynchronizedFullPolicy(
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
        suite=str(manifest["suite"]),
        row_name="full_nfe10",
        task_id=task_id,
        trial_id=trial_id,
        paired_action_noise=True,
        action_noise_seed_base=seed_base,
        log_action_chunks=True,
    )
    _ensure_generation_latency_schema()
    generation = RealSimVLAGenerationPolicy(
        model=model,
        processor=processor,
        updater=generation_updater,
        n_g=3,
        device=device,
        suite=str(manifest["suite"]),
        task_id=task_id,
        trial_id=trial_id,
        action_noise_seed_base=seed_base,
        log_action_chunks=True,
    )
    condition = _make_policy(
        row=CONDITION_ROW,
        model=model,
        processor=processor,
        condition_updater=condition_updater,
        generation_updater=None,
        device=device,
        suite=str(manifest["suite"]),
        task_id=task_id,
        trial_id=trial_id,
        action_noise_seed_base=seed_base,
    )
    combined = _make_policy(
        row=COMBINED_ROW,
        model=model,
        processor=processor,
        condition_updater=condition_updater,
        generation_updater=generation_updater,
        device=device,
        suite=str(manifest["suite"]),
        task_id=task_id,
        trial_id=trial_id,
        action_noise_seed_base=seed_base,
    )

    for policy in (full, generation, condition, combined):
        batch0 = policy.preprocess(image00, image01, proprio0, prompt)
        policy._refill_action_queue(batch0)
    q0_condition_equal = torch.equal(_queue_tensor(full), _queue_tensor(condition))
    q0_generation_equal = torch.equal(
        _queue_tensor(generation), _queue_tensor(combined)
    )
    for policy in (condition, combined):
        policy.action_queue.clear()
        batch1 = policy.preprocess(image10, image11, proprio1, prompt)
        policy._refill_action_queue(batch1)
    updated_condition_equal = torch.equal(
        condition.cached_condition, combined.cached_condition
    )
    condition_gate = _counter_gate(condition, CONDITION_ROW)
    combined_gate = _counter_gate(combined, COMBINED_ROW)
    checks = {
        "condition_q0_matches_full_nfe10": q0_condition_equal,
        "combined_q0_matches_generation_ng3": q0_generation_equal,
        "condition_state_equal_before_decoders_at_q1": updated_condition_equal,
        "condition_counter_gate": condition_gate["verdict"]
        == "FIXED_2X2_COUNTER_PASS",
        "combined_counter_gate": combined_gate["verdict"]
        == "FIXED_2X2_COUNTER_PASS",
    }
    result = {
        "verdict": "FIXED_2X2_PARITY_PASS" if all(checks.values()) else "FIXED_2X2_PARITY_FAIL",
        "manifest_sha256": manifest["manifest_sha256"],
        "condition_source_combined_sha256": FROZEN_CONDITION_SOURCE_SHA256,
        "generation_source_combined_sha256": FROZEN_GENERATION_SOURCE_SHA256,
        "condition_checkpoint_sha256": FROZEN_CONDITION_CHECKPOINT_SHA256,
        "generation_checkpoint_sha256": FROZEN_GENERATION_CHECKPOINT_SHA256,
        "task_id": task_id,
        "trial_id": trial_id,
        "checks": checks,
        "condition_counter_gate": condition_gate,
        "combined_counter_gate": combined_gate,
        "egl_preflight": preflight,
        "provenance": provenance,
    }
    atomic_write_json(output, result)
    if result["verdict"] != "FIXED_2X2_PARITY_PASS":
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--condition-checkpoint", required=True)
    parser.add_argument("--fixed-2x2-source-lock", required=True)
    parser.add_argument("--egl-preflight", required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument(
        "--classification", choices=("RB2_CONFIRMATORY_EGL",), required=True
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    return parser


def main() -> int:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
