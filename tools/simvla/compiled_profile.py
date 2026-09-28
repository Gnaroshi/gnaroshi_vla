"""Separate matched-observation latency and component diagnostics."""

import logging
import time

from tools.simvla.compile_benchmark import Replay, sha, verify_recorded_inputs, write_json
from tools.simvla.compiled_policy import attach_policy, base_row, check_policy, check_reset


def profile(c, output, suite, seed, row):
    import numpy as np
    import torch
    from tools.simvla.compile_runtime import Compiler
    from tools.simvla.compile_audit import WarningCapture
    from tools.simvla.compiled_campaign import move, read_json, check_compiler, digest, sources
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    contract = read_json(output / "campaign_contract.json")
    if sources(c) != contract["source_files"]: raise RuntimeError("Source changed")
    m = read_json(output / "manifests" / suite / seed / "episode_manifest.json")
    observations_path = output / "rows" / suite / seed / "baseline/observations.pt"
    observations = torch.load(observations_path, map_location="cpu", weights_only=False)
    if len(observations) != 8: raise RuntimeError("Eight baseline query observations required")
    directory = output / "latency" / suite / seed / row
    identity = digest({"campaign": digest(contract), "observations": sha(observations_path), "row": row})
    destination = directory / "profile.json"
    if destination.exists():
        previous = read_json(destination)
        if previous["identity"] != identity: raise RuntimeError("Profile identity changed")
        if not previous["timing_valid"]: raise RuntimeError("Previous latency profile needs review")
        return
    torch.set_num_threads(1)
    configure_strict_torch_determinism(m["determinism_seed"])
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total - 2 * 1024**3) / total)
    from pathlib import Path
    verify_recorded_inputs(Path(c["recorded_input"]))
    sample = move(torch.load(Path(c["recorded_input"]) / "recorded_inputs.pt", map_location="cpu", weights_only=False)[:1])
    compiler = Compiler(True)
    capture = WarningCapture()
    for name in ("torch._dynamo", "torch._inductor"): logging.getLogger(name).addHandler(capture)
    with torch.inference_mode():
        replay = Replay(c, base_row(row), compiler, sample)
        policy = attach_policy(replay, c, row, m)
        policy.task_id, policy.trial_id = 9, 0
        def sequence():
            check_reset(policy)
            elapsed, actions = [], []
            for image0, image1, proprio, prompt in observations:
                for _ in range(5):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    step = policy.act(image0, image1, proprio, prompt)
                    torch.cuda.synchronize()
                    elapsed.append((time.perf_counter()-start)*1000)
                    actions.append(step.action)
            check_policy(policy, row)
            return elapsed, np.stack(actions)
        for _ in range(2): sequence()
        check_compiler(compiler, row)
        before = compiler.graph_count()
        measurements = []
        for _ in range(10):
            times, reference = sequence()
            measurements.extend(times)
        uninstrumented_stable = compiler.graph_count() == before
        component = {}
        restored = []
        def instrument(obj, attribute, name):
            original = getattr(obj, attribute)
            restored.append((obj, attribute, original))
            component[name] = []
            def timed(*args, **kwargs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                value = original(*args, **kwargs)
                torch.cuda.synchronize()
                component[name].append((time.perf_counter()-start)*1000)
                return value
            setattr(obj, attribute, timed)
        instrument(policy, "preprocess", "cpu_preprocess_and_transfer")
        instrument(policy, "_decode", "action_generation_inclusive")
        instrument(replay.model, "forward_vlm_efficient", "vlm")
        instrument(replay.step, "forward", "full_action_transformer_and_decoder")
        if replay.native is not None:
            instrument(replay.native.delta_encoder, "forward", "observation_encoder")
            instrument(replay.native.condition_updater, "forward", "condition_updater")
        if replay.loop is not None:
            instrument(replay.loop.updater, "forward", "generation_updater")
        if replay.bridge is not None:
            instrument(replay.bridge, "predict_next", "latent_bridge")
        try:
            for _ in range(2): sequence()
            for values in component.values(): values.clear()
            diagnostic_before = compiler.graph_count()
            for _ in range(5): _, diagnostic = sequence()
        finally:
            for obj, attribute, original in reversed(restored): setattr(obj, attribute, original)
            if hasattr(policy, "close"): policy.close()
        fallback = any(any(t in msg for t in ("hit config.recompile_limit", "hit config.cache_size_limit", "WON'T CONVERT")) for msg in capture.messages)
        timing_valid = uninstrumented_stable and not fallback
        write_json(destination, {"identity": identity, "observations_sha256": sha(observations_path),
            "timing_valid": timing_valid, "warnings": capture.messages,
            "scope": "matched baseline observations, full CPU policy.act incl. queue; not a new SR experiment",
            "compile_stable_during_primary_measurement": uninstrumented_stable,
            "policy_ms_per_action": float(np.mean(measurements)), "policy_ms_per_action_p95": float(np.percentile(measurements, 95)),
            "policy_ms_per_query_amortized": float(np.sum(measurements)/80),
            "samples": measurements, "component_diagnostic": {name: {"calls": len(v), "mean_ms_per_call": float(np.mean(v)) if v else None,
                "amortized_ms_per_query": sum(v)/40} for name, v in component.items()},
            "component_diagnostic_compiler_stable": diagnostic_before == compiler.graph_count(),
            "instrumented_action_max_abs_diff": float(np.max(np.abs(reference-diagnostic))),
            "component_note": "inclusive/nested timers with extra synchronization; do not sum or substitute for primary latency",
            "compiler": compiler.records})
        if not timing_valid: raise RuntimeError("Latency saved but requires review: recompilation or fallback")
    print(f"PROFILE_COMPLETE row={row} output={destination}", flush=True)
