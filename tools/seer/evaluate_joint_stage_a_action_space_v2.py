#!/usr/bin/env python3
"""Re-evaluate the selected Stage-A surrogate in a matched action space.

This is an additive instrumentation amendment. It never rewrites the immutable
v1 selection or gate artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from methods.joint_latent_action_surrogate.action_space_audit import (
    action_from_arm_and_logit,
    build_amended_gate,
    distribution,
    per_sample_l1,
)


PROTOCOL = "joint_stage_a_action_space_metric_amendment_v2"
DATASET_NAME = "libero_10_converted"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    raw = checkpoint.get("model_state_dict", checkpoint)
    return {key.removeprefix("module."): value for key, value in raw.items()}


def load_exact_prefixes(
    model: torch.nn.Module, path: Path, prefixes: tuple[str, ...]
) -> int:
    state = {
        key: value
        for key, value in strip_state(path).items()
        if key.startswith(prefixes)
    }
    expected = {key for key in model.state_dict() if key.startswith(prefixes)}
    if set(state) != expected:
        raise RuntimeError(
            f"Checkpoint prefix identity failed for {path}: "
            f"missing={sorted(expected - set(state))[:30]} "
            f"extra={sorted(set(state) - expected)[:30]}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected adapter keys while loading {path}: "
            f"{incompatible.unexpected_keys[:30]}"
        )
    return len(state)


def resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def validate_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract = read_json(args.contract)
    if contract.get("protocol") != PROTOCOL:
        raise RuntimeError(f"Unexpected amendment protocol: {contract.get('protocol')}")
    if contract.get("status") != "PREDECLARED_BEFORE_AMENDED_EVALUATION":
        raise RuntimeError("Metric amendment contract is not frozen for evaluation")
    evaluation = contract["evaluation"]
    if evaluation["action_representation"] != "arm_6d_plus_sigmoid_gripper_logit_1d":
        raise RuntimeError("Contract does not use the matched executed-action representation")
    if evaluation["aggregation"] != "per_example_mean_7d_then_arithmetic_mean":
        raise RuntimeError("Contract aggregation is not representation matched")
    implementation = contract["implementation"]
    implementation_paths = {
        "evaluator_sha256": Path(__file__).resolve(),
        "metric_module_sha256": (
            args.repo_root
            / "methods/joint_latent_action_surrogate/action_space_audit.py"
        ),
    }
    for key, path in implementation_paths.items():
        if not path.is_file() or sha256(path) != implementation[key]:
            raise RuntimeError(f"Amendment implementation identity failed: {path}")
    return contract


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    contract = validate_contract(args)
    campaign = args.campaign_root
    source_lock_path = campaign / "source_lock/source_lock_manifest.json"
    selection_path = campaign / "selection/stage_a_selection.json"
    v1_gate_path = campaign / "offline/stage_a_gates.json"
    for path in (source_lock_path, selection_path, v1_gate_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = contract["immutable_v1_artifacts"]
    identities = {
        "source_lock": (source_lock_path, expected["source_lock_sha256"]),
        "selection": (selection_path, expected["selection_sha256"]),
        "v1_gate": (v1_gate_path, expected["v1_gate_sha256"]),
    }
    for label, (path, digest) in identities.items():
        actual = sha256(path)
        if actual != digest:
            raise RuntimeError(
                f"Immutable v1 {label} changed: expected={digest} actual={actual}"
            )

    source_lock = read_json(source_lock_path)
    if source_lock.get("status") != "PASS":
        raise RuntimeError("The v1 source lock is not PASS")
    selection = read_json(selection_path)
    v1_gate = read_json(v1_gate_path)
    if v1_gate.get("pass") is not False:
        raise RuntimeError("This amendment is only valid for the preserved v1 gate failure")

    selected = selection["selected"]
    checkpoint = resolved(selected["checkpoint"])
    if not checkpoint.is_file() or sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise RuntimeError("Selected Stage-A checkpoint identity failed")
    if selected["checkpoint_sha256"] != expected["selected_checkpoint_sha256"]:
        raise RuntimeError("Contract selected-checkpoint identity differs from v1 selection")

    raw_jsonl = resolved(selected["raw_jsonl"])
    args_snapshot = raw_jsonl.parent / "args_snapshot.json"
    split_manifest = resolved(selected["split_manifest"])
    for path in (raw_jsonl, args_snapshot, split_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    recursive_reference = resolved(contract["recursive_reference"]["path"])
    if not recursive_reference.is_file():
        raise FileNotFoundError(recursive_reference)
    if sha256(recursive_reference) != contract["recursive_reference"]["sha256"]:
        raise RuntimeError("Recursive-reference artifact identity failed")
    reference = read_json(recursive_reference)
    reference_metric = reference["metrics"]["first_token_l1"]
    expected_examples = int(contract["evaluation"]["expected_examples"])
    if int(reference_metric["count"]) != expected_examples:
        raise RuntimeError("Recursive reference does not have the contracted sample count")
    if int(contract["evaluation"]["max_batches"]) * int(
        contract["evaluation"]["batch_size"]
    ) != expected_examples:
        raise RuntimeError("Contracted max_batches and batch_size do not match sample count")

    checkpoints = source_lock["checkpoints"]
    for name in ("teacher", "canonical_adapter", "vit_checkpoint"):
        path = resolved(checkpoints[name]["path"])
        if not path.is_file() or sha256(path) != checkpoints[name]["sha256"]:
            raise RuntimeError(f"Source-locked checkpoint identity failed: {name}")
    if (
        reference["teacher_sha256"] != checkpoints["teacher"]["sha256"]
        or reference["canonical_adapter_sha256"]
        != checkpoints["canonical_adapter"]["sha256"]
    ):
        raise RuntimeError("Recursive reference uses a different teacher or canonical adapter")
    if set(selection["validation_split_identity"]) != {
        reference["validation_split"]["validation_episode_keys_sha256"],
        reference["validation_split"]["selected_indices_sha256"],
    }:
        raise RuntimeError("Recursive reference validation split differs from Stage A")

    return {
        "contract": contract,
        "source_lock_path": source_lock_path,
        "source_lock": source_lock,
        "selection_path": selection_path,
        "selection": selection,
        "v1_gate_path": v1_gate_path,
        "checkpoint": checkpoint,
        "args_snapshot": args_snapshot,
        "split_manifest": split_manifest,
        "recursive_reference_path": recursive_reference,
        "recursive_reference": reference,
    }


def build_runtime(
    args: argparse.Namespace, context: dict[str, Any]
) -> tuple[Namespace, torch.nn.Module, Any, torch.device, dict[str, Any]]:
    repo_root = args.repo_root
    upstream = repo_root / "architectures/seer/upstream"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(upstream))

    import clip  # noqa: PLC0415
    from train import (  # noqa: PLC0415
        _apply_precision_policy,
        _attach_configured_plan_adapter,
        _build_seer_agent_from_args,
        _load_raw_model_checkpoint,
    )
    from utils.data_utils import get_libero_finetune_dataset  # noqa: PLC0415

    parser_args = Namespace(**read_json(context["args_snapshot"]))
    checkpoints = context["source_lock"]["checkpoints"]
    contract_eval = context["contract"]["evaluation"]
    parser_args.world_size = 1
    parser_args.rank = 0
    parser_args.local_rank = 0
    parser_args.batch_size = int(contract_eval["batch_size"])
    parser_args.workers = int(args.workers)
    parser_args.report_to_wandb = False
    parser_args.joint_latent_action_surrogate_mode = "joint"
    parser_args.joint_latent_action_surrogate_stage = "stage_a"
    parser_args.latentloop_comparison_protocol = 0
    parser_args.finetune_from_pretrained_ckpt = checkpoints["teacher"]["path"]
    parser_args.lrnode_init_adapter_ckpt = checkpoints["canonical_adapter"]["path"]
    parser_args.joint_latent_action_surrogate_init_ckpt = str(context["checkpoint"])
    parser_args.vit_checkpoint_path = checkpoints["vit_checkpoint"]["path"]
    parser_args.save_checkpoint_path = str(args.output_dir)
    parser_args.run_name = args.output_dir.name

    device = torch.device(args.device)
    torch.manual_seed(parser_args.seed)
    np.random.seed(parser_args.seed)
    random.seed(parser_args.seed)
    model = _build_seer_agent_from_args(parser_args, device)
    adapter_report = _attach_configured_plan_adapter(model, parser_args)
    base_load = _load_raw_model_checkpoint(
        model, parser_args.finetune_from_pretrained_ckpt
    )
    canonical_keys = load_exact_prefixes(
        model,
        resolved(parser_args.lrnode_init_adapter_ckpt),
        ("lrnode_delta_encoder.", "lrnode_dynamics."),
    )
    joint_keys = load_exact_prefixes(
        model,
        resolved(parser_args.joint_latent_action_surrogate_init_ckpt),
        ("joint_latent_action_surrogate.",),
    )
    model = _apply_precision_policy(model, parser_args).to(device)
    model._init_model_type()
    model.eval()
    model.requires_grad_(False)
    dataset = get_libero_finetune_dataset(
        parser_args, model.image_processor, clip, epoch=0, floor=True
    )
    load_report = {
        "teacher_loaded_keys": len(base_load["loaded"]),
        "teacher_unexpected_keys": base_load["unexpected"],
        "canonical_loaded_keys": canonical_keys,
        "joint_loaded_keys": joint_keys,
        "joint_parameter_report": adapter_report,
    }
    return parser_args, model, dataset, device, load_report


def add_pair_metrics(
    metrics: dict[str, list[float]],
    prefix: str,
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> None:
    metrics[f"{prefix}_horizon_l1"].extend(per_sample_l1(predicted, target))
    metrics[f"{prefix}_first_token_l1"].extend(
        per_sample_l1(predicted[:, :1], target[:, :1])
    )
    metrics[f"{prefix}_first_token_arm_l1"].extend(
        per_sample_l1(predicted[:, :1, :6], target[:, :1, :6])
    )
    metrics[f"{prefix}_first_token_gripper_probability_l1"].extend(
        per_sample_l1(predicted[:, :1, 6:], target[:, :1, 6:])
    )


def evaluate(args: argparse.Namespace, context: dict[str, Any]) -> dict[str, Any]:
    output_dir = args.output_dir
    outputs = [
        output_dir / "metrics.json",
        output_dir / "gate.json",
        output_dir / "per_sample_metrics.jsonl",
        output_dir / "report.md",
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"Refusing to overwrite amended evaluation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    upstream = args.repo_root / "architectures/seer/upstream"
    data_info = upstream / f"data_info/{DATASET_NAME}.json"
    snapshot = read_json(context["args_snapshot"])
    dataset_root = resolved(snapshot["root_dir"])
    runtime_dataset = dataset_root / DATASET_NAME
    for path in (
        data_info,
        runtime_dataset / "episodes",
        runtime_dataset / "meta_info.h5",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    previous_cwd = Path.cwd()
    os.chdir(upstream)
    try:
        parser_args, model, data, device, load_report = build_runtime(args, context)
        split = data.split_manifest
        selected_identities = set(context["selection"]["validation_split_identity"])
        runtime_identities = {
            split["validation_episode_keys_sha256"],
            split["selected_indices_sha256"],
        }
        split_identity_match = runtime_identities == selected_identities
        if not split_identity_match:
            raise RuntimeError("Runtime validation split does not match v1 selection")

        metrics: dict[str, list[float]] = defaultdict(list)
        max_batches = int(context["contract"]["evaluation"]["max_batches"])
        seed = int(parser_args.seed)
        data.set_epoch(0)
        batches = 0
        for batch_index, batch in enumerate(data.dataloader):
            if batch_index >= max_batches:
                break
            images_primary = batch[0].to(device, dtype=torch.float32)
            images_wrist = batch[3].to(device, dtype=torch.float32)
            states = batch[4].to(device, dtype=torch.float32)
            actions = batch[2].to(device, dtype=torch.float32).clone()
            if images_primary.shape[1] < 9:
                raise RuntimeError("Stage-A two-step audit requires at least nine frames")
            text = batch[1].to(device).unsqueeze(1).repeat(1, 10, 1)
            states = torch.cat((states[..., :6], states[..., -2:]), dim=-1)
            actions[..., 6:] = (actions[..., 6:] + 1) // 2
            selected = 6
            with torch.no_grad():
                torch.manual_seed(seed + batch_index * 17)
                anchor_output = model(
                    image_primary=images_primary[:, :7],
                    image_wrist=images_wrist[:, :7],
                    state=states[:, :7],
                    text_token=text[:, :7],
                    action=actions[:, :7],
                    return_action_latent=True,
                    lrnode_compute_loss=False,
                )
                torch.manual_seed(seed + batch_index * 17 + 1)
                teacher_output = model(
                    image_primary=images_primary[:, 1:8],
                    image_wrist=images_wrist[:, 1:8],
                    state=states[:, 1:8],
                    text_token=text[:, 1:8],
                    action=actions[:, 1:8],
                    return_action_latent=True,
                    lrnode_compute_loss=False,
                )
                z_anchor = anchor_output["action_latent"][:, selected]
                z_teacher = teacher_output["action_latent"][:, selected]
                anchor = model.decode_action_diagnostics_from_latent(z_anchor)
                teacher = model.decode_action_diagnostics_from_latent(z_teacher)
                feature = model.lrnode_encode_delta(
                    key_image_primary=images_primary[:, selected],
                    key_image_wrist=images_wrist[:, selected],
                    cur_image_primary=images_primary[:, selected + 1],
                    cur_image_wrist=images_wrist[:, selected + 1],
                    q_key=states[:, selected],
                    q_cur=states[:, selected + 1],
                )
                z_current = model.lrnode_apply_dynamics(
                    z_prev=z_anchor,
                    u_delta=feature,
                    dt=1.0,
                    age=1.0,
                )
                exact = model.decode_action_diagnostics_from_latent(z_current)
                surrogate = model.joint_latent_action_surrogate.surrogate_forward(
                    anchor_arm=anchor["arm"],
                    anchor_gripper_logit=anchor["gripper_logit"],
                    anchor_latent=z_anchor,
                    current_latent=z_current,
                    shared_feature=feature,
                    elapsed=1,
                )

                joint_action = action_from_arm_and_logit(
                    surrogate.arm, surrogate.gripper_logit
                )
                exact_action = action_from_arm_and_logit(
                    exact["arm"], exact["gripper_logit"]
                )
                teacher_action = action_from_arm_and_logit(
                    teacher["arm"], teacher["gripper_logit"]
                )
                hold_action = action_from_arm_and_logit(
                    surrogate.aligned_anchor[..., :6],
                    surrogate.aligned_anchor[..., 6:],
                )

                add_pair_metrics(metrics, "joint_to_exact", joint_action, exact_action)
                add_pair_metrics(metrics, "hold_to_exact", hold_action, exact_action)
                add_pair_metrics(metrics, "joint_to_teacher", joint_action, teacher_action)
                add_pair_metrics(metrics, "exact_to_teacher", exact_action, teacher_action)
                add_pair_metrics(metrics, "hold_to_teacher", hold_action, teacher_action)
                metrics["joint_gripper_probability"].extend(
                    surrogate.gripper_probability.detach().float().reshape(-1).cpu().tolist()
                )
            batches += 1
            if batches == 1 or batches % 10 == 0 or batches == max_batches:
                print(
                    f"[MATCHED ACTION AUDIT] batches={batches}/{max_batches} "
                    f"examples={len(metrics['joint_to_teacher_first_token_l1'])}"
                )

        if batches != max_batches:
            raise RuntimeError(f"Expected {max_batches} batches, evaluated {batches}")
        summaries = {name: distribution(values) for name, values in sorted(metrics.items())}
        expected_examples = int(
            context["contract"]["evaluation"]["expected_examples"]
        )
        primary_metric_count = int(
            summaries["joint_to_teacher_first_token_l1"]["count"]
        )
        if primary_metric_count != expected_examples:
            raise RuntimeError(
                f"Expected {expected_examples} examples, got {primary_metric_count}"
            )

        reference_metric = context["recursive_reference"]["metrics"]["first_token_l1"]
        checkpoint_identity_match = (
            sha256(context["checkpoint"])
            == context["selection"]["selected"]["checkpoint_sha256"]
        )
        gate_core = build_amended_gate(
            metrics=summaries,
            recursive_reference_mean=float(reference_metric["mean"]),
            expected_examples=expected_examples,
            split_identity_match=split_identity_match,
            checkpoint_identity_match=checkpoint_identity_match,
        )
        gripper = np.asarray(metrics["joint_gripper_probability"], dtype=np.float64)
        extra_checks = {
            "gripper_probability_finite": bool(np.isfinite(gripper).all()),
            "gripper_probability_noncollapsed": bool(
                float(gripper.std()) > 1e-4
                and float(gripper.max() - gripper.min()) > 0.05
            ),
        }
        checks = dict(gate_core["checks"])
        checks.update(extra_checks)
        gate = {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "pass": all(checks.values()),
            "checks": checks,
            "summary": gate_core["summary"],
            "interpretation": (
                "ELIGIBLE_FOR_SEPARATE_STAGE_B_V2_PROTOCOL"
                if all(checks.values())
                else "STAGE_A_REJECTED_UNDER_MATCHED_ACTION_SPACE"
            ),
            "does_not_modify_v1_gate": True,
        }
        metrics_payload = {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "action_representation": "[arm_6d, sigmoid(gripper_logit)]",
            "targets": {
                "decoder_fidelity": "exact_action_head(canonical_predicted_latent)",
                "teacher_fidelity": "full_Seer_shifted_context_action",
            },
            "aggregation": "per-example mean over 7D, then arithmetic mean",
            "batches": batches,
            "examples": primary_metric_count,
            "metrics": summaries,
            "recursive_reference_first_token_l1": reference_metric,
        }
        provenance = {
            "schema_version": 2,
            "protocol": PROTOCOL,
            "contract": str(args.contract),
            "contract_sha256": sha256(args.contract),
            "evaluator": str(Path(__file__).resolve()),
            "evaluator_sha256": sha256(Path(__file__).resolve()),
            "source_lock": str(context["source_lock_path"]),
            "source_lock_sha256": sha256(context["source_lock_path"]),
            "v1_selection": str(context["selection_path"]),
            "v1_selection_sha256": sha256(context["selection_path"]),
            "v1_gate": str(context["v1_gate_path"]),
            "v1_gate_sha256": sha256(context["v1_gate_path"]),
            "selected_checkpoint": str(context["checkpoint"]),
            "selected_checkpoint_sha256": sha256(context["checkpoint"]),
            "recursive_reference": str(context["recursive_reference_path"]),
            "recursive_reference_sha256": sha256(
                context["recursive_reference_path"]
            ),
            "runtime_validation_split": split,
            "load_report": load_report,
            "device": str(device),
            "workers": int(args.workers),
        }

        raw_path = output_dir / "per_sample_metrics.jsonl"
        sample_metric_names = sorted(
            name for name, values in metrics.items() if len(values) == expected_examples
        )
        with raw_path.open("w", encoding="utf-8") as handle:
            for index in range(expected_examples):
                row = {"sample_index": index}
                row.update({name: metrics[name][index] for name in sample_metric_names})
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "gate.json").write_text(
            json.dumps(gate, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        summary = gate["summary"]
        report = [
            "# Stage-A Action-Space Metric Amendment v2",
            "",
            "This additive audit preserves the original v1 gate and selection.",
            "",
            "| Comparison | First-token action L1 mean |",
            "|---|---:|",
            f"| Joint surrogate -> exact canonical head | {summary['joint_to_exact_first_token_l1_mean']:.9f} |",
            f"| Hold anchor -> exact canonical head | {summary['hold_to_exact_first_token_l1_mean']:.9f} |",
            f"| Joint surrogate -> full teacher | {summary['joint_to_teacher_first_token_l1_mean']:.9f} |",
            f"| Recursive reference -> full teacher | {summary['recursive_to_teacher_first_token_l1_mean']:.9f} |",
            "",
            f"- Gate: `{'PASS' if gate['pass'] else 'FAIL'}`",
            f"- Interpretation: `{gate['interpretation']}`",
            f"- Examples: `{primary_metric_count}`",
            "- Representation: `[arm_6d, sigmoid(gripper_logit)]`",
            "- Aggregation: per-example 7D mean, then arithmetic mean",
            "- The immutable v1 gate was not overwritten.",
            "",
        ]
        (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
        return gate
    finally:
        os.chdir(previous_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in ("repo_root", "campaign_root", "contract", "output_dir"):
        setattr(args, name, resolved(getattr(args, name)))
    return args


def main() -> None:
    args = parse_args()
    context = preflight(args)
    print(
        "[PREFLIGHT][OK] immutable v1 artifacts, checkpoint identities, "
        "validation split, and matched metric contract"
    )
    if args.preflight_only:
        return
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for the Stage-A matched metric evaluation")
    gate = evaluate(args, context)
    print(f"[DONE] amended Stage-A gate={'PASS' if gate['pass'] else 'FAIL'}")
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
