#!/usr/bin/env python3
"""One-step feature-bridge forward/backward and optional compile smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from architectures.seer.adapters.latent_bridge.bridge import (
    SeerFeatureBridge,
    SeerFeatureBridgeConfig,
)
from methods.latent_bridge import bridge_distillation_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("full", "small"), required=True)
    parser.add_argument("--stable-seq-len", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    config = SeerFeatureBridgeConfig.from_preset(
        args.preset,
        stable_seq_len=args.stable_seq_len,
        stable_layer="block_00",
        stable_token_group="visual",
    )
    model = SeerFeatureBridge(config).to(device)
    initial_parameter_audit = model.parameter_audit()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    batch = 1
    inputs = (
        torch.randn(batch, 3, 384, device=device),
        torch.randn(batch, args.stable_seq_len, 384, device=device),
        torch.randn(batch, 8, device=device),
        torch.randn(batch, 7, device=device),
    )
    target = torch.randn(batch, 3, 384, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = model.predict_next(*inputs)
        loss, _ = bridge_distillation_loss(prediction.float(), target.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    compiled_max_diff = None
    if args.compile:
        model.eval().to(torch.bfloat16)
        bf16_inputs = tuple(value.to(torch.bfloat16) for value in inputs)
        with torch.no_grad():
            eager = model(*bf16_inputs)
            compiled = torch.compile(model, mode="max-autotune")
            actual = compiled(*bf16_inputs)
            compiled_max_diff = float((eager.float() - actual.float()).abs().max().item())
    payload = {
        "status": "PASS",
        "preset": args.preset,
        "device": str(device),
        "loss": float(loss.item()),
        "output_shape": list(prediction.shape),
        "initial_parameter_audit": initial_parameter_audit,
        "post_update_parameter_audit": model.parameter_audit(),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "bf16_compile_max_abs_diff": compiled_max_diff,
        "compile_mode": "max-autotune" if args.compile else "not_tested",
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
