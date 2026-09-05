#!/usr/bin/env python3
"""Verify Seer's external action-condition hook on a real converted batch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from architectures.seer.adapters.latent_bridge.action_protocol import SeerTemporalEnsembler
from architectures.seer.adapters.latent_bridge.hooks import SeerBoundaryCapture
from architectures.seer.adapters.latent_bridge.layout import SeerTokenLayout
from architectures.seer.adapters.latent_bridge.runtime import (
    SeerRuntimeSpec,
    build_real_libero_batch,
    build_seer_model,
    validate_runtime_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vit-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", default="libero_10_converted")
    parser.add_argument("--dataset-info", default="")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--libero-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict:
    left_f = left.detach().float()
    right_f = right.detach().float()
    return {
        "shape": list(left.shape),
        "max_abs_diff": float((left_f - right_f).abs().max().item()),
        "mean_abs_diff": float((left_f - right_f).abs().mean().item()),
        "cosine": float(
            F.cosine_similarity(left_f.reshape(1, -1), right_f.reshape(1, -1), dim=-1).item()
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    spec = SeerRuntimeSpec(
        checkpoint=args.checkpoint,
        vit_checkpoint=args.vit_checkpoint,
        dataset_root=args.dataset_root,
        libero_path=args.libero_path,
        checkpoint_sha256=args.checkpoint_sha256,
        dataset_name=args.dataset_name,
        dataset_info_path=args.dataset_info,
    )
    provenance = validate_runtime_spec(spec)
    model, checkpoint = build_seer_model(spec, device=args.device, seed=args.seed)
    inputs, batch_audit = build_real_libero_batch(
        spec, model, device=args.device, sample_index=args.sample_index, seed=args.seed
    )
    layout = SeerTokenLayout.from_model(model)

    with torch.inference_mode(), SeerBoundaryCapture(model) as capture:
        outputs = model(**inputs)
        capture.require_complete()
    arm_original, gripper_original = outputs[:2]
    original_sequence = torch.cat([arm_original, gripper_original], dim=-1)

    final_condition = layout.select(capture.final_output, timestep=-1, group="action")
    hooked_head_input = capture.action_head_input
    if hooked_head_input is None:
        raise RuntimeError("action-head pre-hook is empty")
    expected_all_conditions = layout.as_timesteps(capture.final_output)[
        :, :, layout.per_timestep_slices["action"], :
    ]
    arm_hook, gripper_hook = model.decode_action_from_latent(hooked_head_input)
    hook_sequence = torch.cat([arm_hook, gripper_hook], dim=-1)

    selected_original = original_sequence[:, -1]
    selected_hook = hook_sequence[:, -1]
    original_ensemble = SeerTemporalEnsembler(16, 3, spec.ensembling_temperature).to(args.device)
    hook_ensemble = SeerTemporalEnsembler(16, 3, spec.ensembling_temperature).to(args.device)
    original_env = original_ensemble.step(selected_original, 0)
    hook_env = hook_ensemble.step(selected_hook, 0)

    comparisons = {
        "captured_condition_vs_action_head_input": tensor_metrics(
            expected_all_conditions, hooked_head_input
        ),
        "original_vs_hook_action_sequence": tensor_metrics(original_sequence, hook_sequence),
        "selected_final_condition_vs_head_input": tensor_metrics(
            final_condition, hooked_head_input[:, -1]
        ),
        "original_vs_hook_executed_action": tensor_metrics(original_env, hook_env),
    }
    threshold = 1e-6
    passed = all(item["max_abs_diff"] <= threshold for item in comparisons.values())
    payload = {
        "status": "PASS" if passed else "FAIL",
        "threshold_max_abs_diff": threshold,
        "runtime_spec": spec.to_dict(),
        "provenance": provenance,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_load_audit": model._latent_bridge_checkpoint_load_audit,
        "token_layout": layout.to_dict(),
        "batch": batch_audit,
        "captured_layers": {
            name: list(value.shape) for name, value in capture.layer_outputs.items()
        },
        "final_layer_shape": list(capture.final_output.shape),
        "action_head_input_shape": list(hooked_head_input.shape),
        "comparisons": comparisons,
    }
    (output_dir / "condition_hook_equivalence.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "condition_action_diff.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["comparison", "shape", "max_abs_diff", "mean_abs_diff", "cosine"]
        )
        writer.writeheader()
        for name, values in comparisons.items():
            writer.writerow({"comparison": name, **values, "shape": "x".join(map(str, values["shape"]))})

    audit_lines = [
        "# Real LIBERO Batch Format Audit",
        "",
        f"- Verdict: **{'PASS' if passed else 'FAIL'}**",
        f"- Public Seer checkpoint epoch: `{checkpoint.get('epoch')}`",
        f"- Converted episode: `{batch_audit['episode_id']}`",
        f"- Language instruction: `{batch_audit['language']}`",
        f"- Transformer layout: `{layout.sequence_length} x {layout.tokens_per_timestep} = {layout.flattened_tokens}` tokens",
        f"- Action condition: final `ln_f`, current timestep action slice, shape `[1, {layout.action_tokens}, 384]`",
        "- State input: current end-effector position/orientation (6) plus two gripper positions (2).",
        "- Action input to the bridge: previous executed LIBERO action, shape `[1, 7]`.",
        "- The batch uses Seer's converted-data reader and canonical trajectory-consistent image shifts (RGB 10, wrist 4).",
        "- The original Seer action head is reused; no substitute decoder is involved.",
        "",
        "## Shapes",
        "",
    ]
    audit_lines.extend(
        f"- `{key}`: `{value}`" for key, value in batch_audit["model_input_shapes"].items()
    )
    (output_dir / "real_batch_format_audit.md").write_text(
        "\n".join(audit_lines) + "\n", encoding="utf-8"
    )
    if not passed:
        raise SystemExit("external Seer boundary equivalence failed")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
