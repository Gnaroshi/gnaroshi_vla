"""Bounded training entry point for the provisional rewq v0 router head."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from methods.latentloop.modules.rewq_v0 import (
    ComputeCostTable,
    NextAnchorRecoveryTargets,
    RecoverabilityHead,
    RecoveryPrediction,
    fit_recovery_safety_envelope,
    fit_split_conformal_calibration,
    recoverability_loss,
)
from .checkpoint import save_rewq_v0_checkpoint
from .data import CompactRecoveryDataset, load_safe_reference


def _targets(batch: dict[str, Tensor], device: torch.device) -> NextAnchorRecoveryTargets:
    return NextAnchorRecoveryTargets(
        continuous=batch["continuous"].to(device),
        continuous_valid=batch["continuous_valid"].to(device),
        events=batch["events"].to(device),
        mode_valid=batch["mode_valid"].to(device),
    )


def _validation_pass(
    head: RecoverabilityHead,
    dataset: CompactRecoveryDataset,
    *,
    device: torch.device,
    batch_size: int,
    normalization: Tensor,
) -> tuple[RecoveryPrediction, NextAnchorRecoveryTargets, dict[str, float]]:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    continuous_prediction: list[Tensor] = []
    event_logits: list[Tensor] = []
    continuous_target: list[Tensor] = []
    continuous_valid: list[Tensor] = []
    event_target: list[Tensor] = []
    mode_valid: list[Tensor] = []
    loss_sums = {"loss": 0.0, "continuous_q90_pinball": 0.0, "event_bce": 0.0}
    rows = 0
    head.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            prediction = head(features)
            target = _targets(batch, device)
            losses = recoverability_loss(
                prediction,
                target,
                normalization=normalization,
                quantile=head.quantile,
            )
            count = int(features.shape[0])
            rows += count
            for name, value in losses.items():
                loss_sums[name] += float(value) * count
            continuous_prediction.append(prediction.continuous_q90.detach().cpu())
            event_logits.append(prediction.event_logits.detach().cpu())
            continuous_target.append(target.continuous.detach().cpu())
            continuous_valid.append(target.continuous_valid.detach().cpu())
            event_target.append(target.events.detach().cpu())
            mode_valid.append(target.mode_valid.detach().cpu())
    prediction = RecoveryPrediction(
        continuous_q90=torch.cat(continuous_prediction),
        event_logits=torch.cat(event_logits),
    )
    target = NextAnchorRecoveryTargets(
        continuous=torch.cat(continuous_target),
        continuous_valid=torch.cat(continuous_valid),
        events=torch.cat(event_target),
        mode_valid=torch.cat(mode_valid),
    )
    return prediction, target, {
        name: value / max(rows, 1) for name, value in loss_sums.items()
    }


def train_rewq_v0(
    *,
    train_data: str | Path,
    validation_data: str | Path,
    safe_reference: str | Path,
    output: str | Path,
    costs: ComputeCostTable,
    device: torch.device | str,
    max_steps: int = 2_000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 20260827,
) -> dict[str, Any]:
    """Train only the <=100K router head; base VLA, U_C, and U_G remain frozen."""

    if int(max_steps) < 1 or int(max_steps) > 5_000:
        raise ValueError("rewq v0 training is bounded to 1--5000 steps")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    torch.manual_seed(int(seed))
    target_device = torch.device(device)
    train = CompactRecoveryDataset(train_data, expected_split="train")
    validation = CompactRecoveryDataset(
        validation_data, expected_split="checkpoint_validation"
    )
    if train.feature_config != validation.feature_config:
        raise ValueError("train/validation feature contracts differ")
    overlap = set(train.episode_ids) & set(validation.episode_ids)
    if overlap:
        raise ValueError("train/validation episodes overlap")

    reference = load_safe_reference(safe_reference)
    envelope = fit_recovery_safety_envelope(
        reference["continuous"],
        reference["events"],
        provenance=str(Path(safe_reference).expanduser().resolve()),
    )
    normalization = torch.tensor(envelope.continuous_limits).clamp_min(1e-6)
    head = RecoverabilityHead(train.feature_config.input_dim).to(target_device)
    if not head.parameter_audit()["within_parameter_ceiling"]:
        raise RuntimeError("rewq v0 head violates its parameter ceiling")
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    loader = DataLoader(
        train,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    iterator = iter(loader)
    last_losses: dict[str, float] = {}
    head.train()
    for _ in range(int(max_steps)):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        prediction = head(batch["features"].to(target_device))
        losses = recoverability_loss(
            prediction,
            _targets(batch, target_device),
            normalization=normalization.to(target_device),
            quantile=head.quantile,
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()
        last_losses = {
            name: float(value.detach()) for name, value in losses.items()
        }

    validation_prediction, validation_target, validation_losses = _validation_pass(
        head,
        validation,
        device=target_device,
        batch_size=batch_size,
        normalization=normalization,
    )
    conformal = fit_split_conformal_calibration(
        validation_prediction,
        validation_target,
        provenance=str(Path(validation_data).expanduser().resolve()),
    )
    output_root = Path(output).expanduser().resolve()
    checkpoint = save_rewq_v0_checkpoint(
        output_root / "rewq_v0_recoverability.pt",
        head=head,
        feature_config=train.feature_config,
        envelope=envelope,
        conformal=conformal,
        costs=costs,
        metadata={
            "train_data": str(Path(train_data).expanduser().resolve()),
            "validation_data": str(Path(validation_data).expanduser().resolve()),
            "safe_reference": str(Path(safe_reference).expanduser().resolve()),
            "train_episodes": len(set(train.episode_ids)),
            "validation_episodes": len(set(validation.episode_ids)),
            "episode_disjoint": True,
            "max_steps": int(max_steps),
            "seed": int(seed),
        },
    )
    summary = {
        "verdict": "REWQ_V0_RECOVERABILITY_TRAINING_COMPLETE",
        "checkpoint": str(checkpoint),
        "head": head.parameter_audit(),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_validation_episode_overlap": 0,
        "safe_reference_rows": int(reference["continuous"].shape[0]),
        "max_steps": int(max_steps),
        "last_train_losses": last_losses,
        "validation_losses": validation_losses,
        "envelope": envelope.to_dict(),
        "conformal": conformal.to_dict(),
        "costs": costs.to_dict(),
        "frozen_external_modules": ["SimVLA", "U_C", "U_G"],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".training_summary.json.tmp-{os.getpid()}"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_root / "training_summary.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--safe-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--exact-condition-ms", type=float, required=True)
    parser.add_argument("--approximate-condition-ms", type=float, required=True)
    parser.add_argument("--generation-ng3-ms", type=float, required=True)
    parser.add_argument("--generation-ng2-ms", type=float, required=True)
    parser.add_argument("--router-ms", type=float, default=0.0)
    parser.add_argument("--cost-provenance", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = train_rewq_v0(
        train_data=args.train_data,
        validation_data=args.validation_data,
        safe_reference=args.safe_reference,
        output=args.output,
        costs=ComputeCostTable(
            exact_condition_ms=args.exact_condition_ms,
            approximate_condition_ms=args.approximate_condition_ms,
            generation_ng3_ms=args.generation_ng3_ms,
            generation_ng2_ms=args.generation_ng2_ms,
            router_ms=args.router_ms,
            provenance=args.cost_provenance,
        ),
        device=args.device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
