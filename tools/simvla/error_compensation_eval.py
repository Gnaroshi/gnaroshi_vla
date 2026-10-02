"""Paired 500-episode efficacy evaluation, with independently resumable episodes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from types import MethodType

from tools.simvla.error_compensation_common import (
    CONFIG, ARMS, ROWS, configure, identity, read_json, snapshots, write_json,
)


def make_policy(c, row, *, smoke=False, k_c=2):
    import torch
    from architectures.simvla.adapters.latentloop.native_v0_runtime import load_frozen_simvla, freeze_module
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.native_v0_long_eval import _SynchronizedFullPolicy
    from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval import (
        SynchronizedConditionK_CPolicy, SynchronizedConditionNaiveNFEPolicy,
    )
    from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import (
        install_exact_uint8_delta_path, use_normalized_float_delta_inputs,
    )
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import load_generation_checkpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.error_compensation_train import load_candidate
    from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop
    from tools.simvla.compile_runtime import ActionStep

    paths = snapshots(c)
    model, processor, _ = load_frozen_simvla(checkpoint=paths["checkpoint_snapshot"],
        smolvlm_model=paths["backbone_snapshot"], norm_stats=c["norm_stats"], device=torch.device("cuda"))
    common = dict(model=model, processor=processor, device=torch.device("cuda"), suite="libero_10",
        task_id=0, trial_id=0, action_noise_seed_base=c["action_noise_seed_base"], log_action_chunks=False)
    if row == "baseline":
        policy = _SynchronizedFullPolicy(dcld_core=None, mode="full", refresh_every=1,
            flow_steps=10, image_size=384, replan_steps=5, client_resize_size=224,
            paired_action_noise=True, **common)
    else:
        adapter, _ = load_native_v0_checkpoint(c["condition_checkpoint"], device="cuda", require_final_150k=True)
        freeze_module(adapter)
        install_exact_uint8_delta_path(adapter)
        use_normalized_float_delta_inputs(adapter)
        policy = SynchronizedConditionK_CPolicy(adapter=adapter, checkpoint_id=c["condition_checkpoint"],
            k_c=k_c, row_name=row, **common)
        if row == "condition_full10":
            pass
        elif row.startswith("condition_naive"):
            policy.nfe = int(row.removeprefix("condition_naive"))
            policy._decode = MethodType(SynchronizedConditionNaiveNFEPolicy._decode, policy)
        else:
            parent, _ = load_generation_checkpoint(c["generation_checkpoint"], device="cuda")
            freeze_module(parent)
            step = ActionStep(model.transformer).eval()
            parent_loop = SimVLAGenerationLoop(parent, step.decoder).eval()
            candidate_loop = None
            if row in ARMS:
                if smoke:
                    candidate, _ = load_generation_checkpoint(Path(c["output"]) / "smoke" / row / "latest.pt", device="cuda")
                    freeze_module(candidate)
                else:
                    candidate = load_candidate(c, row, "cuda")
                candidate_loop = SimVLAGenerationLoop(candidate, step.decoder).eval()
                policy._condition_code = None
                def capture(_module, _inputs, output):
                    policy._condition_code = output
                policy._code_handle = adapter.delta_encoder.register_forward_hook(capture)
            policy._experiment_loops = [parent_loop] + ([candidate_loop] if candidate_loop is not None else [])
            def decode(self, condition, proprio, *, policy_query_index):
                noise, noise_seed = self._paired_initial_noise(condition, proprio, policy_query_index)
                normalized = self.action_adapter.normalize_proprio(proprio)
                updated = policy_query_index % k_c != 0 and candidate_loop is not None
                loop = candidate_loop if updated else parent_loop
                code = condition.new_zeros(condition.shape[0], parent.condition_code_dim)
                if updated and row != "true_condition_no_code":
                    if self._condition_code is None:
                        raise RuntimeError("Missing live Condition code")
                    code = self._condition_code
                mask = self.condition_layout.valid_mask if updated else None
                trace = loop(noise, full_step=lambda x, t: step(condition, x, normalized, t),
                    full_step_indices=(0, 4, 8), proprio=normalized, condition=condition,
                    condition_valid_mask=mask, condition_change_code=code)
                self.metrics.counters["num_action_transformer_calls"] += 3
                self.metrics.counters["num_action_transformer_decodes"] += 1
                self.metrics.counters["num_generation_decoder_only_steps"] += 7
                return self.action_adapter.action_space.postprocess(trace.final_noisy_action), noise_seed
            policy._decode = MethodType(decode, policy)
    policy._sync = lambda: None
    policy.row_name = row
    return policy


def check_counts(policy, row, actual, k_c=2):
    q = int(policy.metrics.counters["num_policy_queries"])
    nfe = int(row.removeprefix("condition_naive")) if row.startswith("condition_naive") else (10 if row in ("baseline", "condition_full10") else 3)
    full = q if row == "baseline" else (q + k_c - 1) // k_c
    cheap = 7 * q if row in ("parent", *ARMS) else 0
    expected = {"transformer": nfe * q, "generation": cheap, "condition": q - full}
    observed = {key: actual.get(key, 0) for key in expected}
    if observed != expected or policy.metrics.counters["num_full_vlm_calls"] != full:
        raise RuntimeError(f"Invocation counter mismatch: {observed} != {expected}")
    if q != (policy.step_index + 4) // 5:
        raise RuntimeError("Action execution horizon is not R=5")
    return {"queries": q, "full_vlm": full, **observed}


def run(c, row, *, smoke=False, k_c=4, policy_factory=None, counter_row=None, counter_checker=None):
    configure(c)
    import numpy as np
    import torch
    from libero.libero import benchmark
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import build_env_obs, get_libero_env, video_frame_from_obs, save_episode_video
    torch.set_num_threads(1)
    configure_strict_torch_determinism(c["evaluation_seed"])
    run_id = identity(c)
    directory = Path(c["output"]) / ("eval_smoke" if smoke else "online") / f"kc{k_c}_{row}"
    manifest = read_json(Path(c["output"]) / "episode_manifest.json")
    specs = manifest["episodes"][:1] if smoke else manifest["episodes"]
    done = []
    directory.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        policy = (policy_factory or make_policy)(c, row, smoke=smoke, k_c=k_c)
        calls = Counter()
        handles = []
        def add(module, name):
            handles.append(module.register_forward_pre_hook(lambda _m, _i: calls.update([name])))
        add(policy.model.transformer.blocks[0], "transformer")
        if hasattr(policy, "native_v0") and policy.native_v0.condition_updater is not None:
            add(policy.native_v0.condition_updater, "condition")
        # Count the actual updater calls for both parent and candidate, including closure-owned modules.
        if hasattr(policy, "_experiment_loops"):
            for loop in policy._experiment_loops:
                add(loop.updater, "generation")
        suite = benchmark.get_benchmark_dict()["libero_10"]()
        env = None
        last_task = None
        try:
            for spec in specs:
                task, trial = spec["task_id"], spec["trial_id"]
                path = directory / "episodes" / f"task{task}_trial{trial}.json"
                if path.exists():
                    saved = read_json(path)
                    if saved["identity"] != run_id or saved["row"] != row or saved["k_c"] != k_c:
                        raise RuntimeError("Mixed episode provenance")
                    done.append(saved)
                    continue
                configure_strict_torch_determinism(c["evaluation_seed"])
                if task != last_task:
                    if env is not None: env.close()
                    env, prompt = get_libero_env(suite.get_task(task), 256, 7)
                    states = suite.get_task_init_states(task)
                    last_task = task
                # Every episode has its own seed/reset boundary, so recovery does
                # not merge partial trajectories or depend on previous failures.
                env.seed(7)
                env.reset()
                obs = env.set_init_state(states[trial])
                for _ in range(10): obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
                policy.reset()
                policy.task_id, policy.trial_id = task, trial
                calls.clear()
                if hasattr(policy, "_condition_code"): policy._condition_code = None
                frames, timing = [], []
                success = False
                started = last_progress = time.monotonic()
                for index in range(int(c.get('smoke_policy_actions',26)) if smoke else 900):
                    inputs = build_env_obs(obs)
                    if trial == 0 and task in (9, 4) and index % 2 == 0:
                        frames.append(video_frame_from_obs(obs))
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    step = policy.act(*inputs, prompt)
                    torch.cuda.synchronize()
                    timing.append((time.perf_counter() - tick) * 1000)
                    if not np.isfinite(step.action).all(): raise RuntimeError("Nonfinite action")
                    obs, _, success, _ = env.step(step.action.tolist())
                    if time.monotonic() - last_progress > 30:
                        write_json(directory / "progress.json", {"task": task, "trial": trial,
                            "actions": index + 1, "completed": len(done), "total": len(specs),
                            "successes": sum(r["success"] for r in done)})
                        last_progress = time.monotonic()
                    if success: break
                counts = (counter_checker or check_counts)(policy, counter_row or row, calls, k_c)
                ages = sorted({int(t["age"]) for t in policy.query_trace})
                if smoke and ages != list(range(k_c)):
                    raise RuntimeError(f"Smoke did not cover all Condition ages: {ages}")
                result = {"identity": run_id, "row": row, "k_c": k_c, "task_id": task, "trial_id": trial,
                    "condition_ages_seen": ages,
                    "success": bool(success), "episode_length": index + 1, "counters": counts,
                    "policy_ms_total": sum(timing), "wall_seconds": time.monotonic() - started,
                    "timing_scope": "eager policy.act, outer CUDA sync, invocation hooks; sd1 screening only"}
                if hasattr(policy, 'extra_episode_metrics'):
                    result.update(policy.extra_episode_metrics())
                write_json(path, result)
                done.append(result)
                if frames:
                    try:
                        save_episode_video(frames, directory / f"task{task}_trial{trial}.mp4", fps=10)
                    except Exception as exc:
                        write_json(directory / f"task{task}_video_error.json", {"error": str(exc)})
                print(f"kc{k_c}/{row}: {len(done)}/{len(specs)} success={sum(r['success'] for r in done)}/{len(done)} task={task} trial={trial}", flush=True)
        finally:
            for handle in handles: handle.remove()
            if env is not None: env.close()
    summary = {"verdict": "SMOKE_PASS" if smoke else "EVALUATION_COMPLETE",
        "identity": run_id, "row": row, "k_c": k_c, "candidate_training_k_c": c.get('training_k_c',4) if row in ARMS or policy_factory else None,
        "episodes": len(done), "successes": sum(r["success"] for r in done),
        "success_rate": sum(r["success"] for r in done) / len(done),
        "policy_ms_per_action": sum(r["policy_ms_total"] for r in done) / sum(r["episode_length"] for r in done),
        "paper_latency": False, "gpu": torch.cuda.get_device_name(0), "compile": False,
        "per_task_successes": {str(t): sum(r["success"] for r in done if r["task_id"] == t) for t in range(10)}}
    if any('component_timing' in r for r in done):
        keys = set().union(*(r.get('component_timing', {}) for r in done))
        total_actions = sum(r['episode_length'] for r in done)
        summary['component_timing'] = {}
        for name in sorted(keys):
            entries = [r.get('component_timing', {}).get(name, {}) for r in done]
            total = sum(e.get('cuda_ms_total', 0.0) for e in entries)
            calls = sum(e.get('calls', 0) for e in entries)
            summary['component_timing'][name] = dict(calls=calls, cuda_ms_total=total,
                cuda_ms_per_call=total/calls if calls else None, cuda_ms_per_action=total/total_actions)
        summary['component_timing_scope'] = 'CUDA-event intervals; nested observation/residual inside condition_query. Do not sum nested timers. Sensor/render outside policy timer.'
    write_json(directory / "summary.json", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--row", choices=ROWS, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--k-c", type=int, choices=(3, 4), default=4)
    a = p.parse_args()
    run(read_json(a.config), a.row, smoke=a.smoke, k_c=a.k_c)
