"""Bounded real-checkpoint audit; no environment rollout or model training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.simvla.compile_benchmark import (
    DEFAULT_CONFIG, ROOT, Replay, configure, model_asset_identity, read_json, sha, source_identity,
    verify_recorded_inputs, write_json,
)
from tools.simvla.compile_checks import assess_result, read_log


def difference(actual, expected):
    import torch
    delta = (actual.float() - expected.float()).abs()
    return {"bitwise_equal": torch.equal(actual, expected),
        "finite": bool(torch.isfinite(actual).all()), "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "first5_gripper_changes": int(((actual[:, :5, 6] > 0) != (expected[:, :5, 6] > 0)).sum())}


def make_reference(replay, c):
    import torch
    from architectures.simvla.adapters.latentloop.native_v0_long_eval import _SynchronizedFullPolicy
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_eval import SynchronizedNaiveNFE3Policy
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_eval import _ensure_generation_latency_schema
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_policy import RealSimVLAGenerationPolicy
    from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval import (
        SynchronizedConditionK_CPolicy, SynchronizedConditionNaiveNFEPolicy,
        SynchronizedCombinedK_CN_GPolicy,
    )
    _ensure_generation_latency_schema()
    common = dict(model=replay.model, processor=replay.processor, device=torch.device("cuda"),
        suite="libero_10", task_id=0, trial_id=0, action_noise_seed_base=c["seed"], log_action_chunks=True)
    row = replay.row
    if row in {"condition_naive3", "condition_nfe10", "ours_kc2_ng3"}:
        common.update(adapter=replay.native, checkpoint_id=c["condition_checkpoint"], k_c=2)
        if row == "condition_naive3": return SynchronizedConditionNaiveNFEPolicy(nfe=3, **common)
        if row == "ours_kc2_ng3": return SynchronizedCombinedK_CN_GPolicy(generation_updater=replay.loop.updater, n_g=3, **common)
        return SynchronizedConditionK_CPolicy(**common)
    if row == "generation_ng3":
        return RealSimVLAGenerationPolicy(updater=replay.loop.updater, n_g=3, **common)
    common.update(flow_steps=10, image_size=384, replan_steps=5, client_resize_size=224,
        paired_action_noise=True)
    if row == "latent_bridge_f2":
        from architectures.simvla.adapters.latent_bridge.policy import RealSimVLALatentBridgePolicy
        return RealSimVLALatentBridgePolicy(bridge=replay.bridge, refresh_every=2, **common)
    cls = SynchronizedNaiveNFE3Policy if row == "naive_nfe3" else _SynchronizedFullPolicy
    return cls(dcld_core=None, mode="full", refresh_every=1, **common)


def reference_actions(policy, sample):
    import torch
    policy.reset()
    policy._paired_initial_noise = lambda condition, proprio, query: (sample["queries"][query]["noise"], 0)
    for batch in sample["queries"]:
        policy._refill_action_queue(batch)
        if len(policy.action_queue) != 5:
            raise AssertionError("Original execution queue must contain R=5 actions")
        policy.action_queue.clear()
    chunks = [v["action_chunk"] for v in policy.action_chunk_records]
    if len(chunks) != 4 or any(tuple(x.shape) != (1, 10, 7) for x in chunks):
        raise AssertionError("Original policy must generate four fresh H=10 chunks")
    return torch.cat(chunks, 0).cuda()


def count_calls(replay, sample):
    counters = Counter()
    modules = {"vlm": replay.model.vlm.model.text_model,
        "action_transformer": replay.model.transformer.blocks[0],
        "decoder": replay.model.transformer.action_decoder}
    if replay.native is not None:
        modules.update(observation_encoder=replay.native.delta_encoder,
                       condition_updater=replay.native.condition_updater)
    if replay.loop is not None: modules["generation_updater"] = replay.loop.updater
    if replay.bridge is not None: modules["bridge"] = replay.bridge
    handles = []
    for name, module in modules.items():
        def count(_module, _inputs, key=name): counters.update([key])
        handles.append(module.register_forward_pre_hook(count))
    try:
        replay(sample)
    finally:
        for handle in handles: handle.remove()
    expected = {"vlm": 2 if replay.native is not None or replay.bridge is not None else 4,
        "action_transformer": 12 if replay.row in {"naive_nfe3", "condition_naive3", "ours_kc2_ng3", "generation_ng3"} else 40,
        "decoder": 12 if replay.row in {"naive_nfe3", "condition_naive3"} else 40}
    if replay.native is not None: expected.update(observation_encoder=2, condition_updater=2)
    if replay.loop is not None: expected["generation_updater"] = 28
    if replay.bridge is not None: expected["bridge"] = 2
    return {"observed": dict(counters), "expected": expected, "passed": dict(counters) == expected}


class WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def audit_row(c, old, output, row):
    import torch
    from tools.simvla.compile_runtime import Compiler
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism

    directory = output / row
    if (directory / "audit.json").exists():
        raise RuntimeError("Audit result already exists; choose a new output")
    write_json(directory / "audit_context.json", {"config": c, "source_files": source_identity(c),
        "command": sys.argv, "input_contract": verify_recorded_inputs(old)})
    torch.set_num_threads(1)
    determinism = configure_strict_torch_determinism(c["seed"])
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total - 2 * 1024**3) / total)
    contract = verify_recorded_inputs(old)
    def move(value):
        if torch.is_tensor(value): return value.cuda()
        if isinstance(value, dict): return {k: move(v) for k, v in value.items()}
        if isinstance(value, list): return [move(v) for v in value]
        return value
    samples = move(torch.load(old / "recorded_inputs.pt", map_location="cpu", weights_only=False))
    directory.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        replay = Replay(c, row, Compiler(False), samples)
        modules = [replay.model] + [x for x in (replay.native, replay.bridge, replay.loop) if x is not None]
        assert all(not p.requires_grad for m in modules for p in m.parameters())
        eager = [replay(sample).detach().clone() for sample in samples]
        calls = count_calls(replay, samples[0])
        assert calls["passed"], calls
        original_forward = replay.model.transformer.forward
        replay.model.transformer.forward = replay.original_transformer_forward
        policy = make_reference(replay, c)
        policy_comparisons = []
        for i, sample in enumerate(samples):
            item = difference(eager[i], reference_actions(policy, sample))
            policy_comparisons.append(item)
            print(f"REFERENCE row={row} window={i+1}/{len(samples)} diff={item['max_abs']}", flush=True)
        if hasattr(policy, "close"): policy.close()
        replay.model.transformer.forward = original_forward
        repeat = [difference(replay(samples[i]), eager[i]) for i in reversed(range(len(samples)))]
        if replay.hook is not None: replay.hook.close()
        del policy, replay
        torch.cuda.empty_cache()

        capture = WarningCapture()
        for name in ("torch._dynamo", "torch._inductor"): logging.getLogger(name).addHandler(capture)
        torch._dynamo.reset()
        compiler = Compiler(True)
        compiled = Replay(c, row, compiler, samples)
        for i, sample in enumerate(samples):
            compiled(sample)
            print(f"COMPILE_WARMUP row={row} window={i+1}/{len(samples)} graphs={compiler.graph_count()}", flush=True)
        first = [compiled(sample).detach().clone() for sample in samples]
        before = compiler.graph_count()
        compiled_comparisons = [difference(a, b) for a, b in zip(first, eager)]
        prior_path = old / row / "compile/actions.pt"
        prior = torch.load(prior_path, map_location="cpu", weights_only=True)
        if len(prior) != len(first):
            raise RuntimeError("Prior compiled output count differs from current inputs")
        cross_run = [difference(a, b.cuda()) for a, b in zip(first, prior)]
        compiled_repeat = [difference(compiled(samples[i]), first[i]) for i in reversed(range(len(samples)))]
        after = compiler.graph_count()
        if compiled.hook is not None: compiled.hook.close()
        for name in ("torch._dynamo", "torch._inductor"): logging.getLogger(name).removeHandler(capture)
        finite = all(x["finite"] for x in policy_comparisons + repeat + compiled_comparisons + compiled_repeat)
        reference_equal = all(x["bitwise_equal"] for x in policy_comparisons)
        reset_equal = all(x["bitwise_equal"] for x in repeat + compiled_repeat)
        value = {"row": row, "mode": "compile", "verdict": "RECORDED_INPUT_BENCHMARK_COMPLETE",
            "compiler": compiler.records, "graphs_before_measurement": before, "graphs_after_measurement": after,
            "output_comparisons": [{**x, "bitwise_equal": x["bitwise_equal"]} for x in compiled_comparisons],
            "bitwise_equal_on_all_recorded_inputs": all(x["bitwise_equal"] for x in compiled_comparisons),
            "gripper_sign_changes": sum(x["first5_gripper_changes"] for x in compiled_comparisons)}
        assessment = assess_result(value, log_text="\n".join(capture.messages))
        structural = finite and reference_equal and reset_equal and calls["passed"] and assessment["timing_checks"] == "PASS"
        report = {"row": row, "verdict": "STRUCTURAL_CHECKS_PASS_SR_UNTESTED" if structural else "AUDIT_REVIEW_REQUIRED",
            "input_sha256": contract["sha256"], "windows": len(samples), "queries": len(samples)*4,
            "determinism": determinism, "all_parameters_frozen": True,
            "dtype_parameter_counts": {label: dict(Counter({str(dtype): sum(p.numel() for p in module.parameters() if p.dtype == dtype)
                for dtype in {p.dtype for p in module.parameters()}})) for label, module in [("base", compiled.model)] + ([("bridge", compiled.bridge)] if compiled.bridge is not None else [])},
            "real_policy_vs_replay": policy_comparisons, "eager_reverse_order_repeat": repeat,
            "compile_vs_eager": compiled_comparisons, "compiled_reverse_order_repeat": compiled_repeat,
            "prior_compiled_actions_sha256": sha(prior_path), "prior_compile_vs_current": cross_run,
            "direct_call_counts": calls, "compiler_assessment": assessment,
            "compiler_records": compiler.records, "compiler_warnings": capture.messages,
            "success_rate_validated": False, "timing_measured": False,
            "note": "Hooks/counters are used only in eager audit; no timing claim from instrumented execution."}
        write_json(directory / "audit.json", report)
        print(json.dumps({"row": row, "verdict": report["verdict"], "reference_equal": reference_equal,
                          "repeat_equal": reset_equal, "compile_max_diff": max(x["max_abs"] for x in compiled_comparisons)}), flush=True)
        return structural


def artifact_audit(c, old, output):
    from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_contracts import FROZEN_CONDITION_CHECKPOINT_SHA256
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
        FROZEN_CHECKPOINT_REVISION, FROZEN_GENERATION_CHECKPOINT_SHA256,
        FROZEN_NORM_STATS_SHA256, FROZEN_EXACT_CACHE_MANIFEST_SHA256,
    )
    previous = read_json(old / "provenance.json")
    expected = {"condition_checkpoint": FROZEN_CONDITION_CHECKPOINT_SHA256,
        "generation_checkpoint": FROZEN_GENERATION_CHECKPOINT_SHA256, "norm_stats": FROZEN_NORM_STATS_SHA256}
    actual = {key: sha(c[key]) for key in expected}
    prior_match = {key: sha(c[key]) == value for key, value in previous["inputs"].items() if key in c}
    source = source_identity(c)
    source_differences = {p: {"prior": value, "current": source.get(p)}
                          for p, value in previous["source_files"].items() if p in source and source[p] != value}
    model_assets = model_asset_identity(c)
    blob_check = {p: {"sha256": value, "blob_name": Path(p).resolve().name,
                     "matches": value == Path(p).resolve().name}
                  for p, value in model_assets.items() if len(Path(p).resolve().name) == 64}
    frozen = actual == expected and c["checkpoint_revision"] == FROZEN_CHECKPOINT_REVISION
    frozen &= sha(Path(c["cache"]) / "manifest.json") == FROZEN_EXACT_CACHE_MANIFEST_SHA256
    report = {"frozen_checkpoint_norm_cache_match": frozen, "expected": expected, "actual": actual,
        "prior_benchmark_artifact_matches": prior_match, "model_assets": model_assets,
        "sha256_named_hf_blobs": blob_check, "source_differences_from_prior_benchmark": source_differences,
        "current_source_files": source, "input_contract": verify_recorded_inputs(old),
        "note": "Post-run asset inspection; old reports did not capture all HF bytes. Not proof of unchanged assets before this inspection."}
    write_json(output / "artifact_audit.json", report)
    return frozen and all(prior_match.values()) and bool(blob_check) and all(x["matches"] for x in blob_check.values())


def run_all(c, old, output):
    import os
    from tools.simvla.compile_benchmark import preflight, stop_worker
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Audit output must be new; previous evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    preflight(c)
    previous = read_json(old / "summary.json")
    corrected = {row: {mode: assess_result(value, log_text=read_log(old / row / mode / "run.log"))
                      for mode, value in previous["rows"][row].items()} for row in c["rows"]}
    write_json(output / "previous_result_reassessment.json", corrected)
    write_json(output / "provenance.json", {"source_files": source_identity(c), "command": sys.argv,
        "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": {key: sha(c[key]) for key in ("condition_checkpoint", "generation_checkpoint", "bridge_checkpoint", "norm_stats")},
        "input_contract": verify_recorded_inputs(old), "vla_cache_excluded": True})
    all_ok = True
    for row in c["rows"]:
        directory = output / row
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "audit.json").exists():
            raise RuntimeError("Audit output already exists; use a new directory instead of mixing executions")
        busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if busy: raise RuntimeError("GPU is occupied; refusing concurrent audit")
        write_json(output / "status.json", {"state": "running", "row": row})
        print(f"START row={row} log={directory/'audit.log'}", flush=True)
        with (directory / "audit.log").open("w") as log:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--row", row,
                       "--input", str(old), "--output", str(output)]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "TORCHINDUCTOR_CACHE_DIR": str(old / "compiler_cache")})
            try: code = process.wait(timeout=900)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                stop_worker(process)
                raise
        all_ok &= code == 0
        print(f"FINISH row={row} rc={code}", flush=True)
    write_json(output / "status.json", {"state": "finished", "structural_checks_pass": all_ok,
        "success_rate_validated": False})
    return all_ok


def readout_audit(c, old, output):
    import statistics
    import torch
    from tools.simvla.compile_runtime import Compiler
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    from architectures.simvla.adapters.latentloop.efficient_multirate.generation_hidden import full_generation_step_with_hidden
    configure_strict_torch_determinism(c["seed"])
    torch.set_num_threads(1)
    verify_recorded_inputs(old)
    sample = torch.load(old / "recorded_inputs.pt", map_location="cpu", weights_only=False)[0]
    sample["queries"] = [{k: v.cuda() for k, v in q.items()} for q in sample["queries"]]
    with torch.inference_mode():
        replay = Replay(c, "baseline", Compiler(False), [sample])
        replay.model.transformer.forward = replay.original_transformer_forward
        q = sample["queries"][0]
        condition = replay.model.forward_vlm_efficient(q["image_input"], q["image_mask"], q["input_ids"])["vlm_features"]
        proprio = replay.action.normalize_proprio(q["proprio"])
        def old_step(tau):
            return full_generation_step_with_hidden(replay.model.transformer, condition=condition,
                noisy_action=q["noise"], proprio=proprio, tau=tau, dt=-0.1).velocity
        def new_step(tau):
            return replay.step(condition, q["noise"], proprio, tau)[1]
        parity = []
        times = {"old_hook_and_double_decoder": [], "shared_weights_single_decoder": []}
        calls = {}
        for label, fn in [("old_hook_and_double_decoder", old_step), ("shared_weights_single_decoder", new_step)]:
            counts = []
            handle = replay.model.transformer.action_decoder.register_forward_pre_hook(lambda *_: counts.append(1))
            fn(q["noise"].new_ones(1))
            handle.remove()
            calls[label] = len(counts)
        for index in range(10):
            tau = q["noise"].new_full((1,), 1 - index / 10)
            parity.append(bool(torch.equal(old_step(tau), new_step(tau))))
        for rep in range(40):
            tau = q["noise"].new_full((1,), 1 - (rep % 10) / 10)
            order = [("old_hook_and_double_decoder", old_step), ("shared_weights_single_decoder", new_step)]
            if rep % 2: order.reverse()
            for label, fn in order:
                torch.cuda.synchronize()
                started = time.perf_counter()
                fn(tau)
                torch.cuda.synchronize()
                times[label].append((time.perf_counter() - started) * 1000)
        report = {"velocity_bitwise_equal_all_10_tau": all(parity), "decoder_calls_per_full_step": calls,
            "ms_per_full_action_transformer_step": {label: {"mean": statistics.mean(v), "median": statistics.median(v)} for label, v in times.items()},
            "samples": times, "scope": "single-input isolated full step, not rollout latency; no subtraction from old paper numbers"}
        write_json(output / "readout_overhead.json", report)
        print(json.dumps({k:v for k,v in report.items() if k != "samples"}), flush=True)
        return all(parity) and calls == {"old_hook_and_double_decoder": 2, "shared_weights_single_decoder": 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("all", "worker", "readout", "artifacts"))
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--row")
    args = p.parse_args()
    c = read_json(DEFAULT_CONFIG)
    configure(c)
    try:
        if args.command == "worker" and args.row not in c["rows"]:
            raise ValueError("A configured audit row is required")
        if args.command == "artifacts": passed = artifact_audit(c, args.input, args.output)
        elif args.command == "readout": passed = readout_audit(c, args.input, args.output)
        elif args.command == "worker": passed = audit_row(c, args.input, args.output, args.row)
        else: passed = run_all(c, args.input, args.output)
        return 0 if passed else 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
