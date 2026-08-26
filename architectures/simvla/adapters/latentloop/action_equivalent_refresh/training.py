"""Compact training path for the frozen SimVLA selective-refresh risk head."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from architectures.simvla.adapters.latentloop.action_equivalent_refresh.checkpoint import (
    save_action_fidelity_checkpoint,
)
from architectures.simvla.adapters.latentloop.action_equivalent_refresh.features import (
    SimVLAActionFidelityFeatureConfig,
)
from methods.latentloop.modules.action_equivalent_refresh import (
    ActionFidelityHead,
    CounterfactualActionTargets,
    action_fidelity_loss,
    fit_exact_call_budget_calibration,
)


DATASET_SCHEMA = "simvla_action_fidelity_compact_dataset_v1"
ALLOWED_SPLITS = ("train", "checkpoint_validation")


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def save_compact_action_fidelity_dataset(
    path: str | Path,
    *,
    split: str,
    features: Tensor,
    targets: CounterfactualActionTargets,
    episode_first_candidates: Tensor,
    episode_ids: list[str],
    feature_config: SimVLAActionFidelityFeatureConfig,
    source_metadata: dict[str, Any],
) -> Path:
    """Persist only cheap runtime features and scalar same-noise labels."""

    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_SPLITS}")
    rows = int(features.shape[0])
    if features.shape != (rows, feature_config.input_dim):
        raise ValueError("features do not match the declared feature contract")
    one_dimensional = (
        targets.arm_normalized_l1,
        targets.direction_cosine_error,
        targets.direction_valid,
        targets.gripper_mismatch,
        episode_first_candidates,
    )
    if any(value.shape != (rows,) for value in one_dimensional):
        raise ValueError("targets and episode_first_candidates must be [N]")
    if len(episode_ids) != rows or not rows:
        raise ValueError("episode_ids must contain one non-empty ID per row")
    if not bool(episode_first_candidates[0]) or not bool(
        episode_first_candidates.any()
    ):
        raise ValueError("compact data must identify each episode's first candidate")
    floating = (
        features,
        targets.arm_normalized_l1,
        targets.direction_cosine_error,
        targets.gripper_mismatch,
    )
    if not all(bool(torch.isfinite(value.float()).all()) for value in floating):
        raise ValueError("compact action-fidelity data contains non-finite values")
    payload = {
        "schema_version": DATASET_SCHEMA,
        "split": split,
        "feature_config": feature_config.to_dict(),
        "features": features.detach().cpu().float(),
        "arm_normalized_l1": targets.arm_normalized_l1.detach().cpu().float(),
        "direction_cosine_error": targets.direction_cosine_error.detach().cpu().float(),
        "direction_valid": targets.direction_valid.detach().cpu().bool(),
        "gripper_mismatch": targets.gripper_mismatch.detach().cpu().float(),
        "episode_first_candidates": episode_first_candidates.detach().cpu().bool(),
        "episode_ids": [str(value) for value in episode_ids],
        "source_metadata": dict(source_metadata),
        "runtime_forbidden_tensors_absent": {
            "exact_condition": True,
            "exact_action_chunk": True,
            "raw_images": True,
        },
    }
    return _atomic_torch_save(Path(path).expanduser().resolve(), payload)


class CompactActionFidelityDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, path: str | Path, *, expected_split: str) -> None:
        payload = torch.load(
            Path(path).expanduser().resolve(), map_location="cpu", weights_only=False
        )
        if payload.get("schema_version") != DATASET_SCHEMA:
            raise ValueError("unsupported compact action-fidelity dataset")
        if payload.get("split") != expected_split:
            raise ValueError("compact dataset split does not match its role")
        if expected_split not in ALLOWED_SPLITS:
            raise ValueError(f"expected_split must be one of {ALLOWED_SPLITS}")
        raw = payload["feature_config"]
        self.feature_config = SimVLAActionFidelityFeatureConfig(
            **{
                key: int(raw[key])
                for key in (
                    "delta_dim",
                    "proprio_dim",
                    "action_dim",
                    "first_r",
                    "num_token_groups",
                    "max_age",
                )
            }
        )
        self.payload = payload
        self.features = payload["features"].float()
        if self.features.ndim != 2 or self.features.shape[1] != self.feature_config.input_dim:
            raise ValueError("stored features violate their dimension contract")
        self.episode_ids = tuple(str(value) for value in payload["episode_ids"])
        if len(self.episode_ids) != len(self.features):
            raise ValueError("stored episode IDs are not row-aligned")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "features": self.features[index],
            "arm_normalized_l1": self.payload["arm_normalized_l1"][index],
            "direction_cosine_error": self.payload["direction_cosine_error"][index],
            "direction_valid": self.payload["direction_valid"][index],
            "gripper_mismatch": self.payload["gripper_mismatch"][index],
            "episode_first_candidates": self.payload[
                "episode_first_candidates"
            ][index],
        }


def _targets(batch: dict[str, Tensor], device: torch.device) -> CounterfactualActionTargets:
    return CounterfactualActionTargets(
        arm_normalized_l1=batch["arm_normalized_l1"].to(device),
        direction_cosine_error=batch["direction_cosine_error"].to(device),
        direction_valid=batch["direction_valid"].to(device),
        gripper_mismatch=batch["gripper_mismatch"].to(device),
    )


def _validation_predictions(
    head: ActionFidelityHead,
    dataset: CompactActionFidelityDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[list[float], list[float], list[bool], dict[str, float]]:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    arm: list[float] = []
    gripper: list[float] = []
    starts: list[bool] = []
    loss_sums = {
        "loss": 0.0,
        "arm_q90_pinball": 0.0,
        "direction_q90_pinball": 0.0,
        "gripper_bce": 0.0,
    }
    rows = 0
    head.eval()
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            prediction = head(features)
            losses = action_fidelity_loss(
                prediction, _targets(batch, device), quantile=head.quantile
            )
            count = int(features.shape[0])
            rows += count
            for name, value in losses.items():
                loss_sums[name] += float(value.item()) * count
            arm.extend(prediction.arm_q90.detach().cpu().tolist())
            gripper.extend(
                prediction.gripper_mismatch_probability.detach().cpu().tolist()
            )
            starts.extend(batch["episode_first_candidates"].bool().tolist())
    return arm, gripper, starts, {
        name: value / max(rows, 1) for name, value in loss_sums.items()
    }


def train_compact_action_fidelity_head(
    *,
    train_data: str | Path,
    validation_data: str | Path,
    output: str | Path,
    device: torch.device | str,
    max_steps: int = 5_000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 20260827,
    target_exact_fraction: float = 1.0 / 3.0,
) -> dict[str, Any]:
    """Train only the small risk head; all VLA/update modules stay external."""

    if int(max_steps) < 1 or int(max_steps) > 5_000:
        raise ValueError("primary risk-head training is bounded to 1--5000 steps")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive")
    torch.manual_seed(int(seed))
    target_device = torch.device(device)
    train = CompactActionFidelityDataset(train_data, expected_split="train")
    validation = CompactActionFidelityDataset(
        validation_data, expected_split="checkpoint_validation"
    )
    if train.feature_config != validation.feature_config:
        raise ValueError("train/validation feature contracts differ")
    overlap = set(train.episode_ids) & set(validation.episode_ids)
    if overlap:
        raise ValueError("train/validation episodes overlap")

    head = ActionFidelityHead(train.feature_config.input_dim).to(target_device)
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
        losses = action_fidelity_loss(
            prediction, _targets(batch, target_device), quantile=head.quantile
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()
        last_losses = {name: float(value.detach().item()) for name, value in losses.items()}

    arm, gripper, starts, validation_losses = _validation_predictions(
        head,
        validation,
        batch_size=batch_size,
        device=target_device,
    )
    calibration = fit_exact_call_budget_calibration(
        arm,
        gripper,
        starts,
        target_exact_fraction=float(target_exact_fraction),
        max_approximate_age=train.feature_config.max_age,
    )
    output_root = Path(output).expanduser().resolve()
    checkpoint = save_action_fidelity_checkpoint(
        output_root / "action_fidelity_head.pt",
        head=head,
        feature_config=train.feature_config,
        calibration=calibration,
        metadata={
            "train_data": str(Path(train_data).expanduser().resolve()),
            "validation_data": str(Path(validation_data).expanduser().resolve()),
            "train_episodes": len(set(train.episode_ids)),
            "validation_episodes": len(set(validation.episode_ids)),
            "episode_disjoint": True,
            "max_steps": int(max_steps),
            "seed": int(seed),
        },
    )
    summary = {
        "verdict": "ACTION_FIDELITY_HEAD_TRAINING_COMPLETE",
        "checkpoint": str(checkpoint),
        "head": head.parameter_audit(),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_validation_episode_overlap": 0,
        "max_steps": int(max_steps),
        "last_train_losses": last_losses,
        "validation_losses": validation_losses,
        "calibration": calibration.to_dict(),
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--target-exact-fraction", type=float, default=1.0 / 3.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = train_compact_action_fidelity_head(
        train_data=args.train_data,
        validation_data=args.validation_data,
        output=args.output,
        device=args.device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        target_exact_fraction=args.target_exact_fraction,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
