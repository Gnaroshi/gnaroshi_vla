#!/usr/bin/env python3
"""User-launched V0/V1 training with frozen pi0.5 and validation-only selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from _common import DEFAULT_CHECKPOINT, load_local_policy, require_run
from architectures.openpi.adapters.latentloop.losses import LossWeights
from architectures.openpi.adapters.latentloop.latent_bridge_baseline import (
    LocalLatentBridgeAdapter,
    official_style_config,
    small_under_19m_config,
)
from architectures.openpi.adapters.latentloop.trainer import LatentLoopTrainer, TrainerConfig
from architectures.openpi.adapters.latentloop.transition_core import OpenPIKVLatentLoop
from pi05_stage_gate_v2 import verify_stage
from validate_pi05_cache_v2 import validate_cache_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--cache-gate", required=True)
    parser.add_argument("--previous-stage-gate")
    parser.add_argument("--final-evaluation-manifest", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--full-cache-inventory", required=True)
    parser.add_argument("--loss-weights-gate")
    parser.add_argument("--variant", choices=("v0", "v1", "latent_bridge"), required=True)
    parser.add_argument("--bridge-size", choices=("official", "small"), default="small")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--wandb-log-interval", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-weight", type=float, default=1.0)
    parser.add_argument("--chunk-weight", type=float, default=1.0)
    parser.add_argument("--executed-weight", type=float, default=1.0)
    parser.add_argument("--gripper-weight", type=float, default=1.0)
    parser.add_argument("--composition-weight", type=float, default=1.0)
    parser.add_argument("--raw-loss-only", action="store_true")
    parser.add_argument("--raw-loss-examples", type=int, default=32)
    parser.add_argument("--wandb-project", default="gnaroshi-openpi-latentloop")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require_run(args.run, "OPENPI_LATENTLOOP_TRAIN_RUN")
    output = Path(args.output).resolve()
    if args.variant == "v0":
        if args.raw_loss_only:
            stage = "stage3_v0_raw_loss"
            stage_artifacts = [args.cache_gate]
        else:
            if not args.loss_weights_gate:
                raise ValueError("V0 training requires --loss-weights-gate after explicit approval")
            stage = "stage3_v0"
            stage_artifacts = [args.cache_gate, args.loss_weights_gate]
    elif args.variant == "v1":
        stage = "stage6_v1"
        if not args.previous_stage_gate:
            raise ValueError("V1 requires --previous-stage-gate with V0_PAIRED_ROW_PASS")
        stage_artifacts = [args.previous_stage_gate]
    else:
        raise RuntimeError(
            "Latent Bridge-style KV baseline training is disabled until official fidelity is established"
        )
    stage_gate = verify_stage(
        stage,
        args.source_lock,
        stage_artifacts,
        output_candidate=output,
    )
    cache_status = validate_cache_v2(
        args.cache,
        source_lock_path=args.source_lock,
        final_manifest_path=args.final_evaluation_manifest,
        split_contract_path=args.split_contract,
        full_cache_inventory_path=args.full_cache_inventory,
        verify_hashes=True,
        require_full=True,
    )
    if not cache_status["FULL_CACHE_SCHEMA_V2_PASS"]:
        raise RuntimeError(f"cache validation failed: {cache_status['errors']}")
    cache_manifest = json.loads(
        (Path(args.cache).resolve() / "pi05_latentloop_cache_manifest.json").read_text(encoding="utf-8")
    )
    if Path(cache_manifest["metadata"]["checkpoint"]).resolve() != Path(args.checkpoint).resolve():
        raise ValueError("training checkpoint does not match the teacher-cache checkpoint")
    if args.save_interval < 1 or args.validation_interval < 1 or args.log_interval < 1:
        raise ValueError("save, validation, and log intervals must be positive")
    if cache_manifest["metadata"].get("source_lock_id") != stage_gate["source_lock_id"]:
        raise RuntimeError("source mismatch: cache was created under another source lock")

    loss_weight_payload = None
    if args.variant == "v0" and not args.raw_loss_only:
        loss_weight_path = Path(args.loss_weights_gate).resolve()
        loss_weight_payload = json.loads(loss_weight_path.read_text(encoding="utf-8"))
        if (
            loss_weight_payload.get("V0_LOSS_WEIGHTS_APPROVED") is not True
            or loss_weight_payload.get("source_lock_id") != stage_gate["source_lock_id"]
            or loss_weight_payload.get("cache_manifest_sha256")
            != cache_status["cache_manifest_sha256"]
        ):
            raise RuntimeError("V0 loss-weight approval is absent, stale, or for another full cache")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    policy = load_local_policy(args.checkpoint, args.device, flow_steps=10)
    if args.variant == "latent_bridge":
        bridge_config = official_style_config() if args.bridge_size == "official" else small_under_19m_config()
        adapter = LocalLatentBridgeAdapter(bridge_config)
    else:
        adapter = OpenPIKVLatentLoop()
    approved_weights = loss_weight_payload["weights"] if loss_weight_payload else None
    weights = LossWeights(
        state=float(approved_weights["state"] if approved_weights else args.state_weight),
        chunk=float(approved_weights["chunk"] if approved_weights else args.chunk_weight),
        executed=float(approved_weights["executed"] if approved_weights else args.executed_weight),
        gripper=float(approved_weights["gripper"] if approved_weights else args.gripper_weight),
        composition=(
            float(approved_weights["composition"])
            if approved_weights
            else (args.composition_weight if args.variant == "v1" else 0.0)
        ),
    )
    config = TrainerConfig(
        variant=args.variant,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        validation_interval=args.validation_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        num_workers=args.num_workers,
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
                "bridge_size": args.bridge_size if args.variant == "latent_bridge" else None,
            },
        )
    trainer = LatentLoopTrainer(
        base_model=policy._model,  # noqa: SLF001
        adapter=adapter,
        cache_root=args.cache,
        output_dir=output,
        config=config,
        weights=weights,
        device=args.device,
        wandb_run=wandb_run,
    )
    try:
        result = (
            trainer.raw_loss_calibration(args.raw_loss_examples)
            if args.raw_loss_only
            else trainer.train()
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
