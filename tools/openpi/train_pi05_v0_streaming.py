#!/usr/bin/env python3
"""Train or calibrate V0 from an online frozen teacher without a tensor cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _common import DEFAULT_CHECKPOINT
from _common import load_local_policy
from _common import require_run
from pi05_stage_gate_v2 import verify_stage
import torch

from architectures.openpi.adapters.latentloop.cache_contract_v2 import load_final_evaluation_manifest
from architectures.openpi.adapters.latentloop.cache_contract_v2 import load_split_contract
from architectures.openpi.adapters.latentloop.losses import LossWeights
from architectures.openpi.adapters.latentloop.streaming_teacher import OnlineV0TeacherSource
from architectures.openpi.adapters.latentloop.streaming_teacher import StreamingTeacherConfig
from architectures.openpi.adapters.latentloop.trainer import LatentLoopTrainer
from architectures.openpi.adapters.latentloop.trainer import TrainerConfig
from architectures.openpi.adapters.latentloop.transition_core import OpenPIKVLatentLoop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--k1-gate", required=True)
    parser.add_argument("--freeze-gate", required=True)
    parser.add_argument("--final-evaluation-manifest", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--loss-weights-gate")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=1000)
    parser.add_argument("--validation-examples", type=int, default=32)
    parser.add_argument("--action-execution-mode", choices=("A", "B"), default="A")
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--wandb-log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed-base", type=int, default=20260820)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--chunk-weight", type=float, default=1.0)
    parser.add_argument("--executed-weight", type=float, default=1.0)
    parser.add_argument("--gripper-weight", type=float, default=1.0)
    parser.add_argument("--raw-loss-only", action="store_true")
    parser.add_argument("--raw-loss-examples", type=int, default=32)
    parser.add_argument("--wandb-project", default="gnaroshi-openpi-latentloop")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require_run(args.run, "OPENPI_LATENTLOOP_STREAMING_RUN")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.batch_size != 1:
        raise ValueError("streaming V0 currently pins batch-size=1 to bound live teacher state")
    if args.validation_examples < 1 or args.raw_loss_examples < 1:
        raise ValueError("streaming validation and raw-loss limits must be positive")
    if min(args.validation_interval, args.save_interval, args.log_interval) < 1:
        raise ValueError("save, validation, and log intervals must be positive")

    output = Path(args.output).resolve()
    artifacts = [args.k1_gate, args.freeze_gate]
    if args.raw_loss_only:
        stage = "stage3_v0_streaming_raw_loss"
    else:
        if not args.loss_weights_gate:
            raise ValueError("streaming V0 training requires an explicitly approved loss-weight gate")
        stage = "stage3_v0_streaming"
        artifacts.append(args.loss_weights_gate)
    stage_gate = verify_stage(stage, args.source_lock, artifacts, output_candidate=output)
    source_lock = json.loads(Path(args.source_lock).read_text(encoding="utf-8"))
    if Path(source_lock["checkpoint"]["directory"]).resolve() != Path(args.checkpoint).resolve():
        raise ValueError("streaming teacher checkpoint differs from the current source lock")
    final_manifest = load_final_evaluation_manifest(args.final_evaluation_manifest)
    _, split_contract = load_split_contract(args.split_contract, final_manifest)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    policy = load_local_policy(args.checkpoint, args.device, flow_steps=10)
    source_config = StreamingTeacherConfig(
        noise_seed_base=args.noise_seed_base,
        episode_order_seed=args.seed,
    )
    example_source = OnlineV0TeacherSource.from_openpi(
        policy=policy,
        source_lock=source_lock,
        checkpoint=args.checkpoint,
        final_manifest=final_manifest,
        final_manifest_path=args.final_evaluation_manifest,
        split_contract=split_contract,
        split_contract_path=args.split_contract,
        config=source_config,
        device=args.device,
    )
    if example_source.provenance["source_lock_id"] != stage_gate["source_lock_id"]:
        raise RuntimeError("streaming source and stage gate were created under different source locks")

    approved_weights = None
    if not args.raw_loss_only:
        loss_payload = json.loads(Path(args.loss_weights_gate).read_text(encoding="utf-8"))
        if (
            loss_payload.get("V0_STREAMING_LOSS_WEIGHTS_APPROVED") is not True
            or loss_payload.get("training_source_id") != example_source.provenance["training_source_id"]
            or loss_payload.get("action_execution_mode", "A") != args.action_execution_mode
        ):
            raise RuntimeError("streaming V0 loss weights are absent, stale, or for another source")
        approved_weights = loss_payload["weights"]
    weights = LossWeights(
        state=float(approved_weights["state"] if approved_weights else args.state_weight),
        chunk=float(approved_weights["chunk"] if approved_weights else args.chunk_weight),
        executed=float(approved_weights["executed"] if approved_weights else args.executed_weight),
        gripper=float(approved_weights["gripper"] if approved_weights else args.gripper_weight),
        composition=0.0,
    )
    config = TrainerConfig(
        variant="v0",
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        validation_interval=args.validation_interval,
        validation_examples=args.validation_examples,
        action_execution_mode=args.action_execution_mode,
        save_interval=args.save_interval,
        seed=args.seed,
        num_workers=0,
        log_interval=args.log_interval,
        wandb_log_interval=args.wandb_log_interval,
    )
    wandb_run = None
    if args.wandb_mode != "disabled" and not args.raw_loss_only:
        import wandb

        os.environ["WANDB_MODE"] = args.wandb_mode
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or output.name,
            dir=str(output.parent),
            config={
                "trainer": config.__dict__,
                "weights": weights.__dict__,
                "streaming_source": example_source.provenance,
            },
        )
    adapter = OpenPIKVLatentLoop()
    trainer = LatentLoopTrainer(
        base_model=policy._model,  # noqa: SLF001
        adapter=adapter,
        example_source=example_source,
        output_dir=output,
        config=config,
        weights=weights,
        device=args.device,
        wandb_run=wandb_run,
    )
    try:
        result = trainer.raw_loss_calibration(args.raw_loss_examples) if args.raw_loss_only else trainer.train()
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
