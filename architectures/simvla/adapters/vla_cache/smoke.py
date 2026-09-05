"""Bounded real-observation parity and encoder timing, not a success-rate test."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
from architectures.simvla.wrappers.dcld_eval.rollout_runner import RealSimVLADCLDPolicy, build_env_obs, get_libero_env

from .eval import _configure_paths, _load_manifest, _load_model, implementation_identity, validate_norm_stats
from .policy import VLACacheSimVLAPolicy
from .recipe import scientific_contract


def difference(left, right):
    delta = (left.float() - right.float()).abs()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "bitwise_equal": torch.equal(left, right)}


def encode(policy, batch):
    adapter = getattr(policy, "vla_cache", policy.condition_adapter)
    inputs = {key: batch[key] for key in ("input_ids", "image_input", "image_mask")}
    if hasattr(policy, "vla_cache"):
        inputs["text_attention_mask"] = batch.get("text_attention_mask")
    return adapter.encode_condition(**inputs)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def run(args):
    _configure_paths()
    manifest = _load_manifest(Path(args.episode_manifest), row="vla_cache", max_episodes=1)
    validate_norm_stats(Path(args.norm_stats))
    configure_strict_torch_determinism(int(manifest["determinism_seed"]))
    device = torch.device(args.device)
    # Independently loaded reference: adapter construction must not alter it.
    native_model, processor = _load_model(args, manifest, device)
    cache_model, _ = _load_model(args, manifest, device)
    episode = manifest["selected_episodes"][0]
    common = dict(processor=processor, device=device, suite="libero_10",
                  task_id=int(episode["task_id"]), trial_id=int(episode["trial_id"]),
                  action_noise_seed_base=int(manifest["action_noise_seed_base"]), log_action_chunks=False)
    native = RealSimVLADCLDPolicy(model=native_model, dcld_core=None, mode="full", refresh_every=1,
                                 flow_steps=10, image_size=384, replan_steps=5, client_resize_size=224,
                                 row_name="full", paired_action_noise=True, **common)
    backend_before = native_model.vlm.model.text_model.config._attn_implementation
    off = VLACacheSimVLAPolicy(model=cache_model, enable_reuse=False, **common)
    fast = VLACacheSimVLAPolicy(model=cache_model, diagnostics=True, **common)
    slow = VLACacheSimVLAPolicy(model=cache_model, optimized=False, diagnostics=True, **common)
    checks = {"independent_reference_model": native_model is not cache_model,
              "native_backend_unchanged": backend_before == native_model.vlm.model.text_model.config._attn_implementation,
              "cache_off_uses_native_forward": off.vla_cache.text_decoder is None}

    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    env, prompt = get_libero_env(suite.get_task(int(episode["task_id"])),
                                 int(manifest["environment_resolution"]), int(episode["environment_seed"]))
    batches, raw_inputs, comparisons = [], [], []
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(int(episode["task_id"]))[int(episode["init_state_index"])])
        for _ in range(int(manifest["num_wait_steps"])):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        for query in range(args.queries):
            images_state = build_env_obs(obs)
            raw_inputs.append(images_state)
            batch = fast.preprocess(*images_state, prompt)
            batches.append(batch)
            outputs = {name: encode(policy, batch) for name, policy in
                       (("native", native), ("off", off), ("fast", fast), ("slow", slow))}
            policy_by_name = {"native": native, "off": off, "fast": fast, "slow": slow}
            actions = {name: policy_by_name[name]._decode(value, batch["proprio"], policy_query_index=query)[0]
                       for name, value in outputs.items()}
            report = copy.deepcopy(fast.vla_cache.last_report)
            comparisons.append({"query": query, "input_image_sha256": tensor_hash(batch["image_input"]),
                                "valid_text_tokens": int(batch["text_attention_mask"].sum()),
                                "text_slots": batch["input_ids"].numel(),
                                "off_condition": difference(outputs["native"], outputs["off"]),
                                "off_action": difference(actions["native"], actions["off"]),
                                "optimized_condition": difference(outputs["slow"], outputs["fast"]),
                                "optimized_action": difference(actions["slow"], actions["fast"]),
                                "cache_vs_native_condition": difference(outputs["native"], outputs["fast"]),
                                "cache_vs_native_action": difference(actions["native"], actions["fast"]),
                                "identical_sparse_selection": report == slow.vla_cache.last_report,
                                "cache": report})
            for _ in range(5):
                action = native.act(*build_env_obs(obs), prompt)
                obs, _, done, _ = env.step(action.action.tolist())
                if done:
                    break
            if done:
                break
    finally:
        env.close()

    checks.update({
        "native_off_conditions_bitwise_equal": all(x["off_condition"]["bitwise_equal"] for x in comparisons),
        "native_off_actions_bitwise_equal": all(x["off_action"]["bitwise_equal"] for x in comparisons),
        "optimized_sparse_selection_equal": all(x["identical_sparse_selection"] for x in comparisons),
        "optimized_conditions_equal": all(x["optimized_condition"]["max_abs"] <= args.optimization_atol for x in comparisons),
        "optimized_actions_equal": all(x["optimized_action"]["max_abs"] <= args.optimization_atol for x in comparisons),
        "first_anchor_close_to_native": comparisons[0]["cache_vs_native_condition"]["max_abs"] <= args.anchor_atol,
        "real_queries_skip_token_layers": any(x["cache"]["decoder"]["skipped_token_layers"] > 0 for x in comparisons[1:]),
    })
    fast.reset()
    for _ in range(5):
        fast.act(*raw_inputs[0], prompt)
    checks["native_h10_r5_queue"] = (fast.cached_action_chunk.shape[1] == 10 and
                                     len(fast.action_queue) == 0 and fast.query_index == 1)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    policies = {"native": native, "cache_off_native": off, "cache_on_optimized": fast, "cache_on_reference": slow}
    timings = {name: [] for name in policies}
    for policy in (fast, slow):
        policy.vla_cache.diagnostics = False
        policy.vla_cache.text_decoder.diagnostics = False
    for round_index in range(args.benchmark_rounds + 1):
        names = list(policies)
        names = names[round_index % len(names):] + names[:round_index % len(names)]
        for name in names:
            policy = policies[name]
            policy.reset()
            for query, batch in enumerate(batches):
                sync()
                start = time.perf_counter()
                encode(policy, batch)
                sync()
                elapsed = (time.perf_counter() - start) * 1000
                if round_index:
                    timings[name].append({"round": round_index, "query": query, "ms": elapsed})
    result = {
        "verdict": "SIMVLA_VLA_CACHE_REAL_CHECKPOINT_SMOKE_PASS" if all(checks.values()) else "SIMVLA_VLA_CACHE_REAL_CHECKPOINT_SMOKE_FAIL",
        "checks": checks, "implementation_identity": implementation_identity(),
        "manifest_file_sha256": manifest["manifest_file_sha256"], "episode": episode,
        "hostname": platform.node(), "torch": torch.__version__, "dtype": str(next(native_model.parameters()).dtype),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "native_attention_backend": backend_before, "queries": len(batches), "comparisons": comparisons,
        "encoder_benchmark": {name: {"samples": samples, "mean_ms": float(np.mean([s["ms"] for s in samples])),
            "nonanchor_mean_ms": float(np.mean([s["ms"] for s in samples if s["query"] > 0]))}
            for name, samples in timings.items()},
        "scope": "One real LIBERO episode prefix; encoder timings only, not SR or end-to-end speedup. Cache-vs-native differences after the first anchor are expected approximations.",
        "tolerances": {"optimization_atol": args.optimization_atol, "anchor_atol": args.anchor_atol},
        "scientific_contract": scientific_contract(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not all(checks.values()):
        raise RuntimeError(f"{result['verdict']}: {checks}")
    return result


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--output", required=True)
    value.add_argument("--episode-manifest", required=True)
    value.add_argument("--smolvlm-model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    value.add_argument("--norm-stats", required=True)
    value.add_argument("--queries", type=int, default=4, choices=range(2, 9))
    value.add_argument("--benchmark-rounds", type=int, default=3, choices=range(1, 11))
    value.add_argument("--optimization-atol", type=float, default=1e-6)
    value.add_argument("--anchor-atol", type=float, default=5e-4)
    value.add_argument("--device", default="cuda")
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({k: result[k] for k in ("verdict", "checks", "queries", "dtype")}, indent=2))
