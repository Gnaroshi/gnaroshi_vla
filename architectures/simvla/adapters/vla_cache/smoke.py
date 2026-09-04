"""Real-checkpoint parity and reuse smoke test for SimVLA VLA-Cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from architectures.simvla.adapters.latentloop.native_v0_runtime import freeze_module
from architectures.simvla.wrappers.dcld_eval.rollout_runner import RealSimVLADCLDPolicy

from .policy import VLACacheSimVLAPolicy
from .recipe import scientific_contract


def _configure_paths() -> None:
    root = Path(__file__).resolve().parents[4]
    upstream = Path(
        os.environ.get("SIMVLA_UPSTREAM_ROOT", root / "architectures/simvla/upstream")
    ).expanduser().resolve()
    for path in (root, upstream):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def run(args: argparse.Namespace) -> dict[str, object]:
    _configure_paths()
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    device = torch.device(args.device)
    model = SmolVLMVLA.from_pretrained(
        args.checkpoint, revision=args.checkpoint_revision
    ).to(device).eval()
    model.action_space.load_norm_stats(args.norm_stats)
    freeze_module(model)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model)
    common = dict(
        model=model,
        processor=processor,
        device=device,
        suite="libero_10",
        task_id=9,
        trial_id=0,
        action_noise_seed_base=20260905,
        log_action_chunks=False,
    )
    cached = VLACacheSimVLAPolicy(enable_reuse=True, **common)
    matched_full = VLACacheSimVLAPolicy(enable_reuse=False, **common)
    full = RealSimVLADCLDPolicy(
        dcld_core=None,
        mode="full",
        refresh_every=1,
        flow_steps=10,
        image_size=384,
        replan_steps=5,
        client_resize_size=224,
        row_name="parity_full",
        paired_action_noise=True,
        **common,
    )
    image0 = np.full((256, 256, 3), 96, dtype=np.uint8)
    image1 = np.full((256, 256, 3), 128, dtype=np.uint8)
    proprio = np.zeros(8, dtype=np.float32)
    prompt = "pick up the object"
    batch = full.preprocess(image0, image1, proprio, prompt)
    with torch.inference_mode():
        expected = full.condition_adapter.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
        observed = cached.vla_cache.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
        matched_full.vla_cache.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
        matched_full.vla_cache.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
    difference = (expected.float() - observed.float()).abs()
    cosine = F.cosine_similarity(
        expected.float().flatten(1), observed.float().flatten(1), dim=1
    ).item()

    changed = image0.copy()
    changed[0, 0, 0] = min(255, int(changed[0, 0, 0]) + 1)
    second_batch = full.preprocess(changed, image1, proprio, prompt)
    with torch.inference_mode():
        cached.vla_cache.encode_condition(
            input_ids=second_batch["input_ids"],
            image_input=second_batch["image_input"],
            image_mask=second_batch["image_mask"],
        )
    second_report = cached.vla_cache.last_report["decoder"]
    matched_full_report = matched_full.vla_cache.last_report["decoder"]
    checks = {
        "first_query_shape_match": list(expected.shape) == list(observed.shape),
        "first_query_max_abs_within_tolerance": float(difference.max()) <= args.atol,
        "first_query_cosine_within_tolerance": cosine >= args.min_cosine,
        "second_query_actual_kv_reuse": bool(second_report["actual_kv_reuse"]),
        "second_query_skips_token_layers": int(second_report["skipped_token_layers"]) > 0,
        "matched_full_has_no_kv_reuse": not bool(matched_full_report["actual_kv_reuse"]),
        "matched_full_skips_no_token_layers": int(matched_full_report["skipped_token_layers"]) == 0,
    }
    result = {
        "verdict": "SIMVLA_VLA_CACHE_REAL_CHECKPOINT_SMOKE_PASS" if all(checks.values()) else "SIMVLA_VLA_CACHE_REAL_CHECKPOINT_SMOKE_FAIL",
        "checks": checks,
        "first_query": {
            "shape": list(expected.shape),
            "mean_abs_diff": float(difference.mean()),
            "max_abs_diff": float(difference.max()),
            "cosine_similarity": cosine,
            "atol": args.atol,
            "min_cosine": args.min_cosine,
        },
        "second_query_decoder": second_report,
        "matched_full_second_query_decoder": matched_full_report,
        "scientific_contract": scientific_contract(),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError(result["verdict"])
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", required=True)
    value.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    value.add_argument("--checkpoint-revision", required=True)
    value.add_argument("--smolvlm-model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    value.add_argument("--norm-stats", required=True)
    value.add_argument("--atol", type=float, default=5e-4)
    value.add_argument("--min-cosine", type=float, default=0.999999)
    value.add_argument("--device", default="cuda")
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True))
