"""Bounded real-checkpoint interface and latency smoke for SimVLA FastV."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
)

from .encoder import FastVConditionEncoder, FastVForwardConfig
from .provenance import fastv_source_manifest, simvla_fastv_integration_manifest
from .recipe import scientific_contract


ROOT = Path(__file__).resolve().parents[4]


def _configure_upstream() -> Path:
    upstream = Path(
        os.environ.get("SIMVLA_UPSTREAM_ROOT", ROOT / "architectures/simvla/upstream")
    ).expanduser().resolve()
    for path in (ROOT, upstream):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if not (upstream / "models/modeling_smolvlm_vla.py").is_file():
        raise FileNotFoundError(f"SimVLA upstream not found: {upstream}")
    return upstream


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    iterations: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    function()
    _sync(device)
    values: list[float] = []
    output = None
    for _ in range(iterations):
        _sync(device)
        started = time.perf_counter()
        output = function()
        _sync(device)
        values.append((time.perf_counter() - started) * 1000.0)
    assert output is not None
    return output, values


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing smoke output: {output}")
    output.mkdir(parents=True)
    upstream = _configure_upstream()
    device = torch.device(args.device)
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model)
    language = processor.encode_language(
        ["pick up the black bowl and place it on the plate"]
    )
    input_ids = language["input_ids"].to(device)
    image_input = torch.zeros(1, 3, 3, 384, 384, device=device)
    image_mask = torch.tensor([[True, True, False]], device=device)
    fastv = FastVConditionEncoder(
        model,
        FastVForwardConfig(
            prune_layer=2,
            prune_ratio=0.5,
            score_mode="text_mean_first_k",
        ),
    )

    def baseline_call() -> torch.Tensor:
        return model.forward_vlm_efficient(
            image_input, image_mask, input_ids
        )["vlm_features"]

    def fastv_call() -> torch.Tensor:
        return fastv.encode_condition(
            input_ids=input_ids,
            image_input=image_input,
            image_mask=image_mask,
        )

    with torch.inference_mode():
        baseline, baseline_ms = _measure(
            baseline_call, iterations=args.iterations, device=device
        )
        adapted, fastv_ms = _measure(
            fastv_call, iterations=args.iterations, device=device
        )
    debug = fastv.last_debug
    if debug is None:
        raise RuntimeError("FastV smoke did not capture pruning metadata")
    checks = {
        "baseline_shape_is_1x122x960": list(baseline.shape) == [1, 122, 960],
        "fastv_shape_is_1x122x960": list(adapted.shape) == [1, 122, 960],
        "compact_sequence_is_86": debug["compact_sequence_length"] == 86,
        "visual_tokens_are_72": debug["visual_tokens_before"] == 72,
        "visual_tokens_kept_are_36": debug["visual_tokens_kept"] == 36,
        "nonvisual_tokens_kept_are_50": debug["nonvisual_tokens_kept"] == 50,
        "baseline_is_finite": bool(torch.isfinite(baseline).all()),
        "fastv_is_finite": bool(torch.isfinite(adapted).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"real-checkpoint FastV interface smoke failed: {checks}")
    baseline_flat = baseline.float().reshape(1, -1)
    adapted_flat = adapted.float().reshape(1, -1)
    result = {
        "verdict": "SIMVLA_FASTV_REAL_CHECKPOINT_SMOKE_PASS",
        "checks": checks,
        "checkpoint": args.checkpoint,
        "simvla_upstream_root": str(upstream),
        "device": str(device),
        "iterations": args.iterations,
        "baseline_vlm_ms": {
            "mean": float(np.mean(baseline_ms)),
            "values": baseline_ms,
        },
        "fastv_vlm_ms": {
            "mean": float(np.mean(fastv_ms)),
            "values": fastv_ms,
        },
        "observed_vlm_speedup": float(np.mean(baseline_ms) / np.mean(fastv_ms)),
        "condition_cosine": float(
            F.cosine_similarity(baseline_flat, adapted_flat).item()
        ),
        "condition_mean_absolute_difference": float(
            (baseline_flat - adapted_flat).abs().mean().item()
        ),
        "fastv_debug": debug,
        "fastv_source": fastv_source_manifest(),
        "simvla_fastv_integration": simvla_fastv_integration_manifest(),
        "scientific_contract": scientific_contract(),
    }
    (output / "real_checkpoint_smoke.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", required=True)
    value.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    value.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    value.add_argument("--iterations", type=int, default=3)
    value.add_argument("--device", default="cuda")
    return value


def main() -> None:
    args = parser().parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
