#!/usr/bin/env python3
"""Cached-real-input component latency benchmark on one RTX 3090."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from architectures.seer.adapters.latent_bridge.action_protocol import SeerTemporalEnsembler
from architectures.seer.adapters.latent_bridge.checkpoint import load_bridge_checkpoint
from architectures.seer.adapters.latent_bridge.hooks import SeerBoundaryCapture
from architectures.seer.adapters.latent_bridge.layout import SeerTokenLayout
from architectures.seer.adapters.latent_bridge.runtime import (
    SeerRuntimeSpec,
    build_real_libero_batch,
    build_seer_model,
)


def _measure(function, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(values, dtype=np.float64)
    return {
        "warmup_iterations_excluded": warmup,
        "measured_iterations": repeats,
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "std_ms": float(array.std(ddof=1)) if repeats > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vit-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--libero-path", required=True)
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    device = torch.device("cuda:0")
    spec = SeerRuntimeSpec(
        checkpoint=args.checkpoint,
        vit_checkpoint=args.vit_checkpoint,
        dataset_root=args.dataset_root,
        libero_path=args.libero_path,
    )
    model, _ = build_seer_model(spec, device=device)
    inputs, batch_audit = build_real_libero_batch(spec, model, device=device)
    layout = SeerTokenLayout.from_model(model)
    # Load checkpoint metadata on CPU first. Moving the bridge to CUDA before
    # the baseline measurement would contaminate baseline resident/peak memory.
    bridge, bridge_payload = load_bridge_checkpoint(args.bridge_checkpoint)

    capture_layers = (
        ()
        if bridge.config.stable_layer == "ln_f"
        else (int(bridge.config.stable_layer.removeprefix("block_")),)
    )
    with torch.inference_mode(), SeerBoundaryCapture(
        model, layer_indices=capture_layers
    ) as capture:
        full_outputs = model(**inputs, return_action_latent=True)
        capture.require_complete()
        previous = full_outputs["action_latent"][:, -1].detach().clone()
        stable_flat = (
            capture.final_output
            if bridge.config.stable_layer == "ln_f"
            else capture.layer_outputs[bridge.config.stable_layer]
        )
        stable = layout.select(
            stable_flat, timestep=-1, group=bridge.config.stable_token_group
        ).detach().clone()
        state = inputs["state"][:, -1].detach().clone()
        sequence = torch.cat(
            [full_outputs["arm_pred_action"][:, -1], full_outputs["gripper_pred_action"][:, -1]],
            dim=-1,
        )
        executed = (
            SeerTemporalEnsembler(8, 3, 0.01).to(device).step(sequence, 0).float().detach().clone()
        )

    # The extraction hook is intentionally gone for timing. Otherwise it
    # retains an extra transformer output and inflates the baseline peak.
    del full_outputs, stable_flat, sequence
    torch.cuda.synchronize()

    with torch.inference_mode():

        def full_forward():
            return model(**inputs, return_action_latent=True)

        def full_head():
            return model.decode_action_from_latent(previous)

        baseline_resident = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        full = _measure(full_forward, args.warmup, args.repeats)
        full_peak = int(torch.cuda.max_memory_allocated(device))

        torch.cuda.reset_peak_memory_stats(device)
        head = _measure(full_head, args.warmup, args.repeats)
        head_peak = int(torch.cuda.max_memory_allocated(device))

    bridge = bridge.to(device=device, dtype=torch.bfloat16).eval()
    bridge.requires_grad_(False)
    compiled_bridge = torch.compile(bridge, mode="max-autotune")
    bridge_inputs = tuple(
        value.to(torch.bfloat16) for value in (previous, stable, state, executed)
    )
    with torch.inference_mode():
        # Compile and allocator warmup are excluded from measured latency.
        for _ in range(args.warmup):
            compiled_bridge(*bridge_inputs)
        torch.cuda.synchronize()
        bridge_resident_after_compile = int(torch.cuda.memory_allocated(device))

        def bridge_only():
            return compiled_bridge(*bridge_inputs)

        def bridge_and_head():
            predicted = bridge_inputs[0] + compiled_bridge(*bridge_inputs)
            return model.decode_action_from_latent(
                predicted.to(next(model.action_decoder.parameters()).dtype)
            )

        torch.cuda.reset_peak_memory_stats(device)
        bridge_latency = _measure(bridge_only, args.warmup, args.repeats)
        bridge_only_peak = int(torch.cuda.max_memory_allocated(device))

        torch.cuda.reset_peak_memory_stats(device)
        bridge_total = _measure(bridge_and_head, args.warmup, args.repeats)
        bridge_policy_peak = int(torch.cuda.max_memory_allocated(device))

    amortized = {
        f"f{period}": (
            full["mean_ms"] + (period - 1) * bridge_total["mean_ms"]
        ) / period
        for period in (2, 3, 4)
    }

    payload = {
        "status": "PASS",
        "hardware": torch.cuda.get_device_name(device),
        "timing_boundary": (
            "cached preprocessed real LIBERO batch; CUDA synchronized; simulator and host "
            "preprocessing excluded; compile/warmup iterations excluded"
        ),
        "batch": batch_audit,
        "baseline_full_forward_including_action_head": full,
        "shared_action_head_only": head,
        "baseline_non_action_head_mean_ms_by_subtraction": max(
            0.0, full["mean_ms"] - head["mean_ms"]
        ),
        "compiled_bf16_bridge_only": bridge_latency,
        "compiled_bf16_bridge_plus_shared_action_head": bridge_total,
        "latency_per_executed_action_ms": {
            "f1_baseline": full["mean_ms"],
            "bridge_skip_step": bridge_total["mean_ms"],
            "periodic_amortized": amortized,
        },
        "peak_gpu_memory_bytes": {
            "baseline_full_forward": full_peak,
            "shared_action_head_only": head_peak,
            "bridge_only_with_frozen_seer_resident": bridge_only_peak,
            "bridge_plus_shared_action_head_with_frozen_seer_resident": bridge_policy_peak,
        },
        "resident_gpu_memory_bytes": {
            "frozen_seer_and_cached_inputs_before_bridge": baseline_resident,
            "frozen_seer_bridge_and_compiled_runtime": bridge_resident_after_compile,
            "increment_after_loading_and_compiling_bridge": max(
                0, bridge_resident_after_compile - baseline_resident
            ),
        },
        "bridge_stage": bridge_payload["stage"],
        "bridge_epoch": bridge_payload["epoch"],
        "bridge_precision": "bf16",
        "bridge_compile_mode": "max-autotune",
        "baseline_compile_mode": "eager",
        "fairness_caveat": (
            "The official bridge latency contract uses max-autotune while Seer's canonical "
            "baseline implementation remains eager; both statuses are reported explicitly."
        ),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
