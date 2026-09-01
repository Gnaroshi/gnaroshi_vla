"""Augment the compact SimVLA training cache with Latent Bridge inputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.action_adapter import (
    ActionNoiseKey,
    explicit_action_noise,
)
from architectures.simvla.adapters.latentloop.native_v0_prepare import (
    _official_training_image_inputs,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    load_frozen_simvla,
)
from architectures.simvla.adapters.latentloop.native_v0_training_cache import (
    load_compact_sequence,
    load_raw_rgb_sequence,
    load_training_manifest,
)

from .condition_hook import SimVLAConditionWithStableHook
from .dataset import SIDECAR_SCHEMA, sha256_file
from .provenance import (
    latent_bridge_source_manifest,
    simvla_latent_bridge_integration_manifest,
)


def _git(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def _configure_simvla_upstream() -> Path:
    root = Path(__file__).resolve().parents[4]
    upstream = Path(
        os.environ.get("SIMVLA_UPSTREAM_ROOT", root / "architectures/simvla/upstream")
    ).expanduser().resolve()
    if not (upstream / "models/modeling_smolvlm_vla.py").is_file():
        raise FileNotFoundError(f"SimVLA upstream not found: {upstream}")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    return upstream


def _encode_query(
    *,
    hook: SimVLAConditionWithStableHook,
    processor: Any,
    raw_rgb: torch.Tensor,
    language: str,
    device: torch.device,
) -> Any:
    processed = _official_training_image_inputs(
        raw_rgb,
        image_size=int(getattr(processor, "image_size", 384)),
        num_views=int(getattr(processor, "num_views", 3)),
    )
    processed.update(processor.encode_language([language]))
    return hook.encode(
        input_ids=processed["input_ids"].to(device),
        image_input=processed["image_input"].to(device),
        image_mask=processed["image_mask"].to(device),
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    upstream = _configure_simvla_upstream()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing sidecar output: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    cache_root = Path(args.cache).expanduser().resolve()
    source_manifest = load_training_manifest(cache_root)
    if source_manifest["data_role"] != "official_libero_training_demonstrations":
        raise ValueError("Latent Bridge cache must use official LIBERO training demonstrations")
    if source_manifest["checkpoint"] != args.checkpoint:
        raise ValueError("training cache checkpoint differs from requested checkpoint")
    device = torch.device(args.device)
    model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    source_entries = source_manifest["sequences"]
    if args.max_sequences is not None:
        source_entries = source_entries[: args.max_sequences]
    entries: list[dict[str, Any]] = []
    parity_max = 0.0
    parity_mean_max = 0.0
    parity_cosine_min = 1.0
    with SimVLAConditionWithStableHook(
        model, stable_layer_index=args.stable_layer_index
    ) as hook, torch.inference_mode():
        for index, source_entry in enumerate(tqdm(source_entries, desc="bridge sidecar")):
            source = load_compact_sequence(cache_root, source_entry)
            images = load_raw_rgb_sequence(source)
            stable_features: list[torch.Tensor] = []
            previous_actions: list[torch.Tensor] = []
            for query_offset in range(4):
                extracted = _encode_query(
                    hook=hook,
                    processor=processor,
                    raw_rgb=images[query_offset],
                    language=str(source["language_instruction"]),
                    device=device,
                )
                cached = source["condition_sequence"][query_offset].to(
                    device=device, dtype=extracted.condition.dtype
                )
                absolute = (cached.float() - extracted.condition[0].float()).abs()
                difference = float(absolute.max().item())
                mean_difference = float(absolute.mean().item())
                cosine = float(
                    torch.nn.functional.cosine_similarity(
                        cached.float().flatten(),
                        extracted.condition[0].float().flatten(),
                        dim=0,
                    ).item()
                )
                parity_max = max(parity_max, difference)
                parity_mean_max = max(parity_mean_max, mean_difference)
                parity_cosine_min = min(parity_cosine_min, cosine)
                if (
                    difference > args.parity_max_abs
                    or mean_difference > args.parity_mean_abs
                    or cosine < args.parity_min_cosine
                ):
                    raise RuntimeError(
                        f"cached/full condition parity failed at sequence={index}, q={query_offset}, "
                        f"max_abs={difference}, mean_abs={mean_difference}, cosine={cosine}"
                    )
                noise_key = ActionNoiseKey(
                    checkpoint=args.checkpoint,
                    task_id=int(source["task_id"]),
                    episode_id=str(source["episode_id"]),
                    policy_query_index=int(source["anchor_query_index"]) + query_offset,
                    seed_base=int(
                        source_manifest["metadata"]["action_noise_seed_base"]
                    ),
                )
                noise = explicit_action_noise(
                    noise_key,
                    batch_size=1,
                    action_horizon=10,
                    action_dim=7,
                    device=device,
                    dtype=source["proprio_sequence"].dtype,
                )
                action_chunk = action_adapter.decode_action_from_condition(
                    extracted.condition,
                    source["proprio_sequence"][query_offset : query_offset + 1].to(device),
                    steps=args.flow_steps,
                    initial_noise=noise,
                )
                stable_features.append(
                    extracted.stable[0].detach().to("cpu", torch.bfloat16)
                )
                previous_actions.append(
                    action_chunk[0, 0].detach().to("cpu", torch.float32)
                )
            payload = {
                "schema_version": SIDECAR_SCHEMA,
                "task_id": int(source["task_id"]),
                "episode_id": str(source["episode_id"]),
                "anchor_query_index": int(source["anchor_query_index"]),
                "stable_sequence": torch.stack(stable_features),
                "previous_action_sequence": torch.stack(previous_actions),
                "stable_layer_index": int(args.stable_layer_index),
                "previous_action_semantics": (
                    "first action of deterministic teacher chunk at the same "
                    "policy query"
                ),
            }
            relative = Path("sequences") / f"sequence_{index:07d}.pt"
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
            entries.append(
                {
                    "file": str(relative),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "task_id": payload["task_id"],
                    "episode_id": payload["episode_id"],
                    "anchor_query_index": payload["anchor_query_index"],
                }
            )
    manifest = {
        "schema_version": SIDECAR_SCHEMA,
        "source_cache_root": str(cache_root),
        "source_cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "checkpoint": args.checkpoint,
        "norm_stats": str(Path(args.norm_stats).expanduser().resolve()),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "simvla_upstream_root": str(upstream),
        "simvla_upstream_commit": _git(upstream),
        "latent_bridge_upstream": latent_bridge_source_manifest(),
        "simvla_latent_bridge_integration": simvla_latent_bridge_integration_manifest(),
        "stable_layer_index": int(args.stable_layer_index),
        "condition_shape": [122, 960],
        "action_horizon": 10,
        "execution_horizon": 5,
        "flow_steps": int(args.flow_steps),
        "condition_parity_max_abs": parity_max,
        "condition_parity_max_mean_abs": parity_mean_max,
        "condition_parity_min_cosine": parity_cosine_min,
        "condition_parity_thresholds": {
            "max_abs": args.parity_max_abs,
            "mean_abs": args.parity_mean_abs,
            "min_cosine": args.parity_min_cosine,
        },
        "data_role": "official_libero_training_demonstrations",
        "final_evaluation_episodes_used": False,
        "sequences": entries,
        "total_sequences": len(entries),
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    return {
        "verdict": "SIMVLA_LATENT_BRIDGE_SIDECAR_READY",
        "output": str(output),
        "sequences": len(entries),
        "condition_parity_max_abs": parity_max,
        "condition_parity_max_mean_abs": parity_mean_max,
        "condition_parity_min_cosine": parity_cosine_min,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--cache", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    value.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    value.add_argument("--norm-stats", required=True)
    value.add_argument("--stable-layer-index", type=int, default=10)
    value.add_argument("--flow-steps", type=int, default=10)
    value.add_argument("--max-sequences", type=int)
    value.add_argument("--parity-max-abs", type=float, default=1e-3)
    value.add_argument("--parity-mean-abs", type=float, default=1e-5)
    value.add_argument("--parity-min-cosine", type=float, default=0.99999)
    value.add_argument("--device", default="cuda")
    return value


def main() -> None:
    result = prepare(parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
