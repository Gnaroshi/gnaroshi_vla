#!/usr/bin/env python3
"""Real-window loss and gradient parity gate for exact all-age Mode B batching."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from _common import load_local_policy, require_run
from pi05_stage_gate_v2 import verify_stage
import torch

from architectures.openpi.adapters.latentloop.cache_contract_v2 import (
    load_final_evaluation_manifest,
    load_split_contract,
)
from architectures.openpi.adapters.latentloop.losses import LossWeights
from architectures.openpi.adapters.latentloop.streaming_teacher import (
    OnlineV0TeacherSource,
    StreamingTeacherConfig,
)
from architectures.openpi.adapters.latentloop.trainer import (
    LatentLoopTrainer,
    TrainerConfig,
)
from architectures.openpi.adapters.latentloop.transition_core import OpenPIKVLatentLoop


def _gradient_vector(module: torch.nn.Module) -> torch.Tensor:
    values = []
    for parameter in module.parameters():
        if parameter.requires_grad:
            values.append(
                torch.zeros_like(parameter).reshape(-1)
                if parameter.grad is None
                else parameter.grad.detach().reshape(-1)
            )
    return torch.cat(values).float().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--k1-gate", required=True)
    parser.add_argument("--freeze-gate", required=True)
    parser.add_argument("--final-evaluation-manifest", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--loss-weights-gate", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed-base", type=int, default=20260820)
    parser.add_argument("--max-loss-relative-error", type=float, default=1e-3)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.999)
    parser.add_argument("--max-gradient-relative-error", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require_run(args.run, "OPENPI_LATENTLOOP_MODE_B_AUDIT_RUN")

    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite Mode B audit: {root}")
    stage = verify_stage(
        "stage3_v0_streaming",
        args.source_lock,
        [args.k1_gate, args.freeze_gate, args.loss_weights_gate],
        output_candidate=root,
    )
    source_lock = json.loads(Path(args.source_lock).read_text(encoding="utf-8"))
    if (
        Path(source_lock["checkpoint"]["directory"]).resolve()
        != Path(args.checkpoint).resolve()
    ):
        raise RuntimeError("Mode B audit checkpoint differs from the source lock")
    loss_lock = json.loads(Path(args.loss_weights_gate).read_text(encoding="utf-8"))
    if loss_lock.get("action_execution_mode") != "B":
        raise RuntimeError("Mode B audit requires a Mode B loss-weight lock")
    weights = LossWeights(**loss_lock["weights"])
    final_manifest = load_final_evaluation_manifest(args.final_evaluation_manifest)
    _, split_contract = load_split_contract(args.split_contract, final_manifest)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    policy = load_local_policy(args.checkpoint, args.device, flow_steps=10)
    source = OnlineV0TeacherSource.from_openpi(
        policy=policy,
        source_lock=source_lock,
        checkpoint=args.checkpoint,
        final_manifest=final_manifest,
        final_manifest_path=args.final_evaluation_manifest,
        split_contract=split_contract,
        split_contract_path=args.split_contract,
        config=StreamingTeacherConfig(
            noise_seed_base=args.noise_seed_base,
            episode_order_seed=args.seed,
        ),
        device=args.device,
    )
    if source.provenance["source_lock_id"] != stage["source_lock_id"]:
        raise RuntimeError("Mode B audit source and stage gate disagree")
    example = next(source.iter_validation_examples(1))

    torch.manual_seed(args.seed)
    initial_adapter = OpenPIKVLatentLoop()
    initial_state = {
        name: value.detach().cpu()
        for name, value in initial_adapter.state_dict().items()
    }
    del initial_adapter
    root.mkdir(parents=True)
    rows = {}
    gradients = {}
    for mode in ("A", "B"):
        adapter = OpenPIKVLatentLoop()
        adapter.load_state_dict(initial_state, strict=True)
        trainer = LatentLoopTrainer(
            base_model=policy._model,  # noqa: SLF001
            adapter=adapter,
            example_source=source,
            output_dir=root / f"mode_{mode.lower()}",
            config=TrainerConfig(
                variant="v0",
                max_steps=1,
                validation_interval=1,
                save_interval=1,
                validation_examples=1,
                action_execution_mode=mode,
            ),
            weights=weights,
            device=args.device,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        started = time.perf_counter()
        loss, metrics = trainer._forward_example(example)  # noqa: SLF001
        loss.backward()
        elapsed = time.perf_counter() - started
        gradient = _gradient_vector(adapter)
        gradients[mode] = gradient
        rows[mode] = {
            "loss": float(loss.detach().cpu()),
            "gradient_norm": float(torch.linalg.vector_norm(gradient)),
            "action_expert_calls": int(metrics["action_expert_calls"]),
            "cache_rebuild_calls": int(metrics["cache_rebuild_calls"]),
            "action_expert_ms": metrics["action_expert_ms"],
            "cache_rebuild_ms": metrics["cache_rebuild_ms"],
            "elapsed_seconds": elapsed,
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated(torch.device(args.device))
                if torch.cuda.is_available()
                else 0
            ),
            "base_gradients_present": any(
                parameter.grad is not None
                for parameter in policy._model.parameters()  # noqa: SLF001
            ),
        }
        del loss, trainer, adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    loss_relative = abs(rows["B"]["loss"] - rows["A"]["loss"]) / max(
        abs(rows["A"]["loss"]), 1e-12
    )
    gradient_cosine = float(
        torch.nn.functional.cosine_similarity(gradients["B"], gradients["A"], dim=0)
    )
    gradient_relative = float(
        torch.linalg.vector_norm(gradients["B"] - gradients["A"])
        / torch.linalg.vector_norm(gradients["A"]).clamp_min(1e-12)
    )
    passed = (
        math.isfinite(loss_relative)
        and loss_relative <= args.max_loss_relative_error
        and gradient_cosine >= args.min_gradient_cosine
        and gradient_relative <= args.max_gradient_relative_error
        and rows["A"]["action_expert_calls"] == 3
        and rows["B"]["action_expert_calls"] == 1
        and not rows["A"]["base_gradients_present"]
        and not rows["B"]["base_gradients_present"]
    )
    report = {
        "MODE_B_REAL_WINDOW_PARITY_PASS": passed,
        "source_lock_id": stage["source_lock_id"],
        "training_source_id": source.provenance["training_source_id"],
        "rows": rows,
        "loss_relative_error": loss_relative,
        "gradient_cosine": gradient_cosine,
        "gradient_relative_error": gradient_relative,
        "thresholds": {
            "max_loss_relative_error": args.max_loss_relative_error,
            "min_gradient_cosine": args.min_gradient_cosine,
            "max_gradient_relative_error": args.max_gradient_relative_error,
        },
        "scientific_training_started": False,
        "libero_evaluation_started": False,
    }
    (root / "mode_b_real_window_parity.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        raise RuntimeError("Mode B real-window loss/gradient parity failed")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
