"""Canonical Seer model/data construction and causal V1 tuple extraction."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler, Sampler, Subset

from .runtime import attach_v1_transition, load_base_and_v0


class RankStrideSampler(Sampler[int]):
    """Deterministic distributed evaluation sampler without padded duplicates."""

    def __init__(self, size: int, rank: int, world_size: int) -> None:
        self.indices = tuple(range(rank, size, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


@contextmanager
def seer_upstream_context(repo_root: Path) -> Iterator[Path]:
    upstream = repo_root / "architectures/seer/upstream"
    previous = Path.cwd()
    for path in (str(repo_root), str(upstream)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(upstream)
    try:
        yield upstream
    finally:
        os.chdir(previous)


def seed_everything(seed: int, rank: int) -> None:
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def _legacy_args(save_path: Path) -> argparse.Namespace:
    from utils.arguments_utils import get_parser

    values = ["--save_checkpoint_path", str(save_path), "--phase", "finetune"]
    outer = sys.argv
    try:
        sys.argv = ["seer_v1_runtime", *values]
        parser = get_parser()
    finally:
        sys.argv = outer
    return parser.parse_args(values)


def build_seer_args(
    *,
    output_root: Path,
    dataset_root: Path,
    vit_checkpoint: Path,
    libero_path: Path,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    seed: int,
) -> argparse.Namespace:
    args = _legacy_args(output_root)
    overrides = {
        "finetune_type": "libero_finetune",
        "root_dir": str(dataset_root),
        "vit_checkpoint_path": str(vit_checkpoint),
        "libero_path": str(libero_path),
        "world_size": world_size,
        "rank": rank,
        "local_rank": int(os.environ.get("LOCAL_RANK", rank)),
        "batch_size": batch_size,
        "workers": workers,
        "rgb_pad": 10,
        "gripper_pad": 4,
        "traj_cons": True,
        "text_aug": False,
        "sequence_length": 7,
        "window_size": 10,
        "min_window_size": 10,
        "max_window_size": 10,
        "future_steps": 3,
        "action_pred_steps": 3,
        "multi_step_action": 1,
        "num_resampler_query": 6,
        "transformer_layers": 24,
        "obs_pred": True,
        "gripper_width": True,
        "precision": "fp32",
        "bf16_module": "vision_encoder",
        "use_lrnode_latent_update": 1,
        "lrnode_hidden_dim": 256,
        "lrnode_motion_dim": 128,
        "lrnode_fast_encoder_type": "diffcnn",
        "lrnode_gate_init_bias": -4.0,
        "lrnode_use_post_layernorm": 0,
        "latentloop_comparison_protocol": 0,
        "seed": seed,
    }
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def build_frozen_teacher(
    args: argparse.Namespace,
    device: torch.device,
    teacher: Path,
    adapter: Path,
):
    from train import _apply_precision_policy, _build_seer_agent_from_args

    # Every rank must construct the exact same adapter initialization. Dataset
    # sampling is seeded separately with rank offsets.
    seed_everything(args.seed, 0)
    model = _build_seer_agent_from_args(args, device)
    load_report = load_base_and_v0(model, teacher, adapter)
    transition = attach_v1_transition(model)
    model = _apply_precision_policy(model, args).to(device)
    model._init_model_type()
    model.eval()
    return model, transition, load_report


def _episode_ranges(dataset) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    dataset_offset = 0
    for dataset_name, child in zip(dataset.dataset_names, dataset.datasets):
        episode_offset = 0
        for episode_id, size in zip(child.episode_list, child.num_step_per_episode):
            size = int(size)
            start = dataset_offset + episode_offset
            rows.append((f"dataset:{dataset_name}:{episode_id}", start, start + size))
            episode_offset += size
        if episode_offset != len(child):
            raise RuntimeError("dataset episode accounting mismatch")
        dataset_offset += len(child)
    if dataset_offset != len(dataset):
        raise RuntimeError("dataset child accounting mismatch")
    return rows


def build_split_loader(
    *,
    args: argparse.Namespace,
    model,
    split_manifest: Path,
    role: str,
    training: bool,
) -> DataLoader:
    import clip
    from utils.data_utils import get_libero_finetune_dataset

    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    selected = set(manifest["splits"][role]["episode_keys"])
    if not selected:
        raise RuntimeError(f"split role has no episodes: {role}")
    dataset_args = copy.copy(args)
    if not training:
        # Validation and all scientific held-out splits must not consume random
        # image-shift augmentation.
        dataset_args.rgb_pad = -1
        dataset_args.gripper_pad = -1
    full_info = get_libero_finetune_dataset(
        dataset_args, model.image_processor, clip, epoch=0, floor=training
    )
    full_dataset = full_info.dataset
    indices: list[int] = []
    observed: set[str] = set()
    for key, start, stop in _episode_ranges(full_dataset):
        if key in selected:
            indices.extend(range(start, stop))
            observed.add(key)
    missing = sorted(selected - observed)
    if missing:
        raise RuntimeError(f"split references unknown dataset episodes: {missing[:20]}")
    subset = Subset(full_dataset, indices)
    sampler = (
        DistributedSampler(
            subset,
            num_replicas=dataset_args.world_size,
            rank=dataset_args.rank,
            shuffle=True,
            seed=dataset_args.seed,
            drop_last=True,
        )
        if training
        else RankStrideSampler(len(subset), dataset_args.rank, dataset_args.world_size)
    )
    loader = DataLoader(
        subset,
        batch_size=dataset_args.batch_size,
        pin_memory=False,
        num_workers=max(1, dataset_args.workers),
        prefetch_factor=1,
        sampler=sampler,
        persistent_workers=False,
        collate_fn=full_dataset.collator,
        drop_last=training,
    )
    loader.split_role = role
    loader.selected_episode_keys = tuple(sorted(selected))
    return loader


def causal_teacher_tuple(model, batch, interval: int, device: torch.device):
    if interval not in (1, 2, 3):
        raise ValueError("interval must be one of 1,2,3")
    primary = batch[0].to(device=device, dtype=torch.float32)
    wrist = batch[3].to(device=device, dtype=torch.float32)
    raw_state = batch[4].to(device=device, dtype=torch.float32)
    raw_action = batch[2].to(device=device, dtype=torch.float32)
    if primary.shape[1] < 10 or wrist.shape[1] < 10 or raw_state.shape[1] < 10:
        raise RuntimeError("V1 requires a 10-frame Seer window")
    state = torch.cat((raw_state[..., :6], raw_state[..., -2:]), dim=-1)
    model_action = raw_action.clone()
    model_action[..., 6:] = (model_action[..., 6:] + 1) // 2
    text = batch[1].to(device).unsqueeze(1).repeat(1, 10, 1)
    selected = 6
    with torch.no_grad():
        anchor = model(
            image_primary=primary[:, :7],
            image_wrist=wrist[:, :7],
            state=state[:, :7],
            text_token=text[:, :7],
            action=model_action[:, :7],
            return_action_latent=True,
            lrnode_compute_loss=False,
        )["action_latent"][:, selected].detach()
        target = model(
            image_primary=primary[:, interval : 7 + interval],
            image_wrist=wrist[:, interval : 7 + interval],
            state=state[:, interval : 7 + interval],
            text_token=text[:, interval : 7 + interval],
            action=model_action[:, interval : 7 + interval],
            return_action_latent=True,
            lrnode_compute_loss=False,
        )["action_latent"][:, selected].detach()
    return {
        "anchor_latent": anchor,
        "teacher_latent": target,
        "primary_sequence": primary[:, selected : selected + interval + 1],
        "wrist_sequence": wrist[:, selected : selected + interval + 1],
        "state_sequence": state[:, selected : selected + interval + 1],
        # Preserve the demonstrated/executed action domain, including {-1,+1}
        # gripper values; only Seer's teacher-forward action input uses {0,1}.
        "executed_actions": raw_action[:, selected : selected + interval],
        "interval": interval,
    }
