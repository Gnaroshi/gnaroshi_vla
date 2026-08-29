#!/usr/bin/env python3
"""Exact real-batch SimVLA condition-hook equivalence with explicit flow noise."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
for path in (ROOT, UPSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter  # noqa: E402
from architectures.simvla.adapters.latentloop.action_adapter import (  # noqa: E402
    ActionNoiseKey,
    explicit_action_noise,
)
from architectures.simvla.adapters.latentloop.source_lock import (  # noqa: E402
    require_empty_output,
    write_source_lock,
)
from methods.latentloop.training.query_cache_dataset import tensor_sha256  # noqa: E402


def _fixed_randn(initial_noise: torch.Tensor, calls: list[dict[str, object]]):
    def _replacement(*shape: int, **kwargs: object) -> torch.Tensor:
        expected = tuple(int(value) for value in shape)
        if expected != tuple(initial_noise.shape):
            raise AssertionError(f"official path requested noise {expected}, expected {tuple(initial_noise.shape)}")
        calls.append({"shape": list(expected), "kwargs": sorted(kwargs)})
        return initial_noise.to(
            device=kwargs.get("device", initial_noise.device),
            dtype=kwargs.get("dtype", initial_noise.dtype),
        ).clone()

    return _replacement


def _rng_snapshot(device: torch.device) -> dict[str, torch.Tensor]:
    state = {"cpu": torch.get_rng_state().clone()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device).clone()
    return state


def _rng_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def main() -> int:
    """Run one explicit-noise real-batch parity check and write exact diffs."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--metas-path", default=str(UPSTREAM / "datasets" / "metas" / "libero_train.json"))
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed-base", type=int, default=20260804)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from datasets import create_smolvlm_dataloader
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    output = require_empty_output(args.output)
    write_source_lock(
        output,
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    previous_cwd = Path.cwd()
    try:
        os.chdir(UPSTREAM)
        loader = create_smolvlm_dataloader(
            batch_size=1,
            metas_path=args.metas_path,
            num_actions=10,
            training=False,
            action_mode="libero_joint",
            num_workers=0,
            image_size=args.image_size,
        )
        raw_batch = next(iter(loader))
    finally:
        os.chdir(previous_cwd)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)
    language = processor.encode_language(raw_batch["language_instruction"])
    device = torch.device(args.device)
    batch = {
        "input_ids": language["input_ids"].to(device),
        "image_input": raw_batch["image_input"].to(device),
        "image_mask": raw_batch["image_mask"].to(device),
        "proprio": raw_batch["proprio"].to(device),
    }
    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device).eval()
    model.action_space.load_norm_stats(args.norm_stats)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    action_adapter = SimVLAActionAdapter(model)
    key = ActionNoiseKey(
        checkpoint=args.checkpoint,
        task_id=0,
        episode_id="real_libero_batch_0000",
        policy_query_index=0,
        seed_base=args.seed_base,
    )
    initial_noise = explicit_action_noise(
        key,
        batch_size=batch["proprio"].shape[0],
        action_horizon=action_adapter.num_actions,
        action_dim=action_adapter.dim_action,
        device=device,
        dtype=batch["proprio"].dtype,
    )
    randn_calls: list[dict[str, object]] = []
    rng_before = _rng_snapshot(device)
    with torch.no_grad(), mock.patch.object(
        torch,
        "randn",
        side_effect=_fixed_randn(initial_noise, randn_calls),
    ):
        official_action = model.generate_actions(
            batch["input_ids"],
            batch["image_input"],
            batch["image_mask"],
            batch["proprio"],
            steps=args.flow_steps,
        )
    rng_after_official = _rng_snapshot(device)
    with torch.no_grad():
        encoded = model.forward_vlm_efficient(
            batch["image_input"],
            batch["image_mask"],
            batch["input_ids"],
        )
        condition = encoded["vlm_features"]
        hooked_action = action_adapter.decode_action_from_condition(
            condition,
            batch["proprio"],
            steps=args.flow_steps,
            initial_noise=initial_noise,
        )
    rng_after_hook = _rng_snapshot(device)
    difference = (official_action - hooked_action).detach().float().cpu()
    exact = bool(torch.equal(official_action, hooked_action))
    result = {
        "checkpoint": args.checkpoint,
        "condition_shape": list(condition.shape),
        "condition_dtype": str(condition.dtype),
        "official_action_shape": list(official_action.shape),
        "hooked_action_shape": list(hooked_action.shape),
        "initial_noise_shape": list(initial_noise.shape),
        "initial_noise_seed": key.seed(),
        "official_initial_noise_hash": tensor_sha256(initial_noise),
        "hook_initial_noise_hash": tensor_sha256(initial_noise),
        "identical_explicit_initial_noise": True,
        "official_torch_randn_calls": randn_calls,
        "official_randn_call_count": len(randn_calls),
        "global_rng_unchanged_by_official": _rng_equal(rng_before, rng_after_official),
        "global_rng_unchanged_by_hook": _rng_equal(rng_after_official, rng_after_hook),
        "mean_abs_action_diff": float(difference.abs().mean().item()),
        "max_abs_action_diff": float(difference.abs().max().item()),
        "tensor_exact_equal": exact,
        "allclose": bool(torch.allclose(official_action, hooked_action)),
        "passed": exact
        and list(condition.shape[1:]) == [122, 960]
        and list(official_action.shape[1:]) == [10, 7]
        and len(randn_calls) == 1
        and _rng_equal(rng_before, rng_after_official)
        and _rng_equal(rng_after_official, rng_after_hook),
    }
    (output / "condition_hook_equivalence_real.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "condition_action_diff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("flat_index", "official", "hooked", "absolute_difference"),
        )
        writer.writeheader()
        for index, (official, hooked) in enumerate(
            zip(official_action.detach().cpu().flatten(), hooked_action.detach().cpu().flatten())
        ):
            writer.writerow(
                {
                    "flat_index": index,
                    "official": float(official.item()),
                    "hooked": float(hooked.item()),
                    "absolute_difference": float(abs((official - hooked).item())),
                }
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
