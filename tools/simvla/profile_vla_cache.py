"""Bounded, common-input attribution of VLA-Cache runtime costs on rb2."""

import argparse
from contextlib import ExitStack
from dataclasses import replace
from functools import wraps
import json
import os
import platform
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from architectures.simvla.adapters.vla_cache import smolvlm_runtime as runtime
from architectures.simvla.adapters.vla_cache.eval import (
    _configure_paths, _load_manifest, _load_model, implementation_identity, validate_norm_stats,
)
from architectures.simvla.adapters.vla_cache.policy import VLACacheSimVLAPolicy
from architectures.simvla.adapters.vla_cache.smoke import encode
from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
from architectures.simvla.wrappers.dcld_eval.rollout_runner import build_env_obs, get_libero_env


def profiled(name, function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(name):
            return function(*args, **kwargs)
    return wrapped


def main(args):
    _configure_paths()
    manifest = _load_manifest(Path(args.episode_manifest), row="vla_cache", max_episodes=1)
    validate_norm_stats(Path(args.norm_stats))
    configure_strict_torch_determinism(manifest["determinism_seed"])
    model, processor = _load_model(args, manifest, torch.device("cuda"))
    episode = manifest["selected_episodes"][0]
    common = dict(model=model, processor=processor, device=torch.device("cuda"), suite="libero_10",
                  task_id=episode["task_id"], trial_id=episode["trial_id"],
                  action_noise_seed_base=manifest["action_noise_seed_base"], log_action_chunks=False)
    native = VLACacheSimVLAPolicy(enable_reuse=False, **common)
    full = VLACacheSimVLAPolicy(**common)
    full.vla_cache.config = replace(full.vla_cache.config, similarity_threshold=2.)
    sparse = VLACacheSimVLAPolicy(**common)
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    env, prompt = get_libero_env(suite.get_task(episode["task_id"]), 256, episode["environment_seed"])
    batches, observations = [], []
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(episode["task_id"])[episode["init_state_index"]])
        for _ in range(10):
            obs, _, _, _ = env.step([0.] * 6 + [-1.])
        for _ in range(4):
            observations.append(build_env_obs(obs))
            batches.append(sparse.preprocess(*observations[-1], prompt))
            for _ in range(5):
                obs, _, done, _ = env.step(native.act(*build_env_obs(obs), prompt).action.tolist())
            if done:
                break
    finally:
        env.close()
    model.vlm.model.text_model.config._attn_implementation = "sdpa"
    policies = {"native_sdpa": native, "native_eager": native, "adapter_full": full, "adapter_sparse": sparse}
    wall = {}
    with torch.no_grad():
        for mode in ("no_grad", "inference_mode"):
            samples = {key: [] for key in policies}
            context = torch.inference_mode if mode == "inference_mode" else torch.no_grad
            for round_index in range(args.rounds + 2):
                keys = list(policies)
                keys = keys[round_index % 4:] + keys[:round_index % 4]
                for key in keys:
                    policy = policies[key]
                    policy.reset()
                    model.vlm.model.text_model.config._attn_implementation = "eager" if key == "native_eager" else "sdpa"
                    with context():
                        for query, batch in enumerate(batches):
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            encode(policy, batch)
                            torch.cuda.synchronize()
                            if round_index >= 2:
                                samples[key].append({"query": query, "ms": (time.perf_counter() - start) * 1000})
            wall[mode] = {key: {"mean_ms": float(np.mean([x["ms"] for x in values])),
                                      "median_ms": float(np.median([x["ms"] for x in values])), "samples": values}
                          for key, values in samples.items()}

        # Include preprocessing, 10 flow steps, queueing and action postprocessing.
        # Every arm receives the same recorded observations, not its own rollout.
        policy_samples = {key: [] for key in ("native_sdpa", "adapter_full", "adapter_sparse")}
        model.vlm.model.text_model.config._attn_implementation = "sdpa"
        for round_index in range(args.rounds + 2):
            keys = list(policy_samples)
            keys = keys[round_index % 3:] + keys[:round_index % 3]
            for key in keys:
                policy = policies[key]
                policy.reset()
                for query, observation in enumerate(observations):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(5):
                        policy.act(*observation, prompt)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    if round_index >= 2:
                        policy_samples[key].append({"query": query, "ms_per_query_and_5_actions": elapsed,
                                                    "encoder_ms": policy.metrics.latencies["VLM_encoder_ms"][-1],
                                                    "action_transformer_ms": policy.metrics.latencies["action_transformer_ms"][-1]})
        policy_wall = {key: {"mean_ms_per_query_and_5_actions": float(np.mean([x["ms_per_query_and_5_actions"] for x in values])),
                             "mean_ms_per_action": float(np.mean([x["ms_per_query_and_5_actions"] for x in values])) / 5,
                             "samples": values} for key, values in policy_samples.items()}

        profiles = {}
        with ExitStack() as stack:
            def wrap(owner, attr, name):
                stack.enter_context(patch.object(owner, attr, profiled(name, getattr(owner, attr))))
            wrap(runtime.SimVLAVLACacheBackbone, "_inputs_embeds", "cache/vision_connector")
            wrap(runtime.IndexedReuseDecoder, "forward", "cache/decoder")
            wrap(runtime.IndexedReuseDecoder, "previous_visual_importance", "cache/task_importance")
            wrap(runtime, "reusable_visual_positions", "cache/selection")
            wrap(runtime, "layer_reuse_schedule", "cache/entropy_schedule")
            for key in ("native_sdpa", "adapter_full", "adapter_sparse"):
                model.vlm.model.text_model.config._attn_implementation = "sdpa"
                policy = policies[key]
                policy.reset()
                encode(policy, batches[0])
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    with torch.profiler.record_function("encoder"):
                        encode(policy, batches[1])
                    torch.cuda.synchronize()
                events = prof.key_averages()
                profiles[key] = [{"name": e.key, "count": e.count, "cpu_us": e.cpu_time_total,
                                  "self_cpu_us": e.self_cpu_time_total,
                                  "device_us": e.device_time_total, "self_device_us": e.self_device_time_total}
                                 for e in sorted(events, key=lambda x: x.self_cpu_time_total, reverse=True)]

    result = {"wall": wall, "policy_wall": policy_wall, "profiles": profiles, "queries": len(batches),
              "hostname": platform.node(), "implementation_identity": implementation_identity(),
              "manifest_file_sha256": manifest["manifest_file_sha256"], "episode": episode,
              "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "MUJOCO_GL", "PYOPENGL_PLATFORM", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")},
              "gpu": torch.cuda.get_device_name(), "threads": torch.get_num_threads(),
              "dtype": str(next(model.parameters()).dtype), "torch": torch.__version__,
              "scope": "Same frozen weights and 4 real observations. Wall timings have no profiler. Policy timings include H10/R5 action execution interface, not environment stepping/rendering. Not SR or a paper latency replacement. Full-adapter threshold=2 disables reuse for attribution only, never a scientific evaluation row.",
              "sparse_report": sparse.vla_cache.last_report}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({mode: {key: x["mean_ms"] for key, x in values.items()} for mode, values in wall.items()}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episode-manifest", required=True)
    p.add_argument("--norm-stats", required=True)
    p.add_argument("--smolvlm-model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    p.add_argument("--output", required=True)
    p.add_argument("--rounds", type=int, default=8)
    main(p.parse_args())
