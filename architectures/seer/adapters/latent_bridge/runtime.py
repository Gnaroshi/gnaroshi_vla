"""Canonical public-Seer runtime construction for Latent Bridge audits."""

from __future__ import annotations

import contextlib
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .provenance import (
    PUBLIC_SEER_33_SHA256,
    SEER_VIT_MAE_SHA256,
    require_file_hash,
)


@dataclass(frozen=True)
class SeerRuntimeSpec:
    checkpoint: str
    vit_checkpoint: str
    dataset_root: str
    libero_path: str
    checkpoint_sha256: str = PUBLIC_SEER_33_SHA256
    dataset_name: str = "libero_10_converted"
    dataset_info_path: str = ""
    sequence_length: int = 7
    num_resampler_query: int = 6
    num_obs_token_per_image: int = 9
    action_pred_steps: int = 3
    transformer_layers: int = 24
    hidden_dim: int = 384
    transformer_heads: int = 12
    gripper_width: bool = True
    obs_pred: bool = True
    temporal_ensembling: bool = True
    ensembling_temperature: float = 0.01
    precision: str = "fp32"
    bf16_modules: tuple[str, ...] = ("vision_encoder",)

    def to_dict(self) -> dict:
        return asdict(self)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def seer_upstream_root() -> Path:
    return repository_root() / "architectures/seer/upstream"


@contextlib.contextmanager
def seer_import_context():
    upstream = str(seer_upstream_root())
    old_cwd = os.getcwd()
    inserted = upstream not in sys.path
    if inserted:
        sys.path.insert(0, upstream)
    os.chdir(upstream)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if inserted and sys.path[0] == upstream:
            sys.path.pop(0)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _strip_ddp_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key[len("module.") :] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def validate_runtime_spec(spec: SeerRuntimeSpec) -> dict:
    checkpoint_hash = require_file_hash(
        spec.checkpoint, spec.checkpoint_sha256, "Seer base checkpoint"
    )
    vit_hash = require_file_hash(spec.vit_checkpoint, SEER_VIT_MAE_SHA256, "Seer ViT-MAE")
    dataset = Path(spec.dataset_root) / spec.dataset_name
    metadata = dataset / "meta_info.h5"
    if not metadata.is_file():
        raise FileNotFoundError(
            f"dataset_root must contain {spec.dataset_name}/meta_info.h5; "
            f"missing {metadata}"
        )
    dataset_info = Path(spec.dataset_info_path) if spec.dataset_info_path else None
    if dataset_info is not None and not dataset_info.is_file():
        raise FileNotFoundError(f"missing converted-dataset metadata: {dataset_info}")
    if not (Path(spec.libero_path) / "libero/libero").is_dir():
        raise FileNotFoundError(f"invalid LIBERO repository: {spec.libero_path}")
    return {
        "checkpoint_sha256": checkpoint_hash,
        "vit_checkpoint_sha256": vit_hash,
        "dataset_metadata": str(metadata),
        "dataset_name": spec.dataset_name,
        "dataset_info_path": str(dataset_info) if dataset_info else "",
        "libero_path": str(Path(spec.libero_path).resolve()),
    }


def build_seer_model(spec: SeerRuntimeSpec, *, device: str | torch.device, seed: int = 42):
    validate_runtime_spec(spec)
    seed_everything(seed)
    with seer_import_context():
        from models.seer_model import SeerAgent

        model = SeerAgent(
            finetune_type="libero_finetune",
            clip_device=str(device),
            vit_checkpoint_path=spec.vit_checkpoint,
            sequence_length=spec.sequence_length,
            num_resampler_query=spec.num_resampler_query,
            num_obs_token_per_image=spec.num_obs_token_per_image,
            obs_pred=spec.obs_pred,
            atten_only_obs=False,
            attn_robot_proprio_state=False,
            atten_goal=False,
            atten_goal_state=False,
            mask_l_obs_ratio=0.0,
            calvin_input_image_size=224,
            patch_size=16,
            action_pred_steps=spec.action_pred_steps,
            transformer_layers=spec.transformer_layers,
            hidden_dim=spec.hidden_dim,
            transformer_heads=spec.transformer_heads,
            phase="evaluate",
            gripper_width=spec.gripper_width,
            use_lrnode_latent_update=0,
        )
    checkpoint = torch.load(spec.checkpoint, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"checkpoint lacks model_state_dict: {spec.checkpoint}")
    status = model.load_state_dict(_strip_ddp_prefix(checkpoint["model_state_dict"]), strict=False)
    allowed_missing_prefixes = (
        "vision_encoder.",
        "clip_model.",
        "attention_mask",
        "image_decoder_position_embedding",
    )
    disallowed_missing = [
        key for key in status.missing_keys if not key.startswith(allowed_missing_prefixes)
    ]
    if disallowed_missing or status.unexpected_keys:
        raise RuntimeError(
            "public Seer load contract failed: "
            f"disallowed_missing={disallowed_missing}, unexpected={status.unexpected_keys}"
        )
    model._latent_bridge_checkpoint_load_audit = {
        "allowed_missing_keys": list(status.missing_keys),
        "unexpected_keys": list(status.unexpected_keys),
        "loaded_tensor_count": len(checkpoint["model_state_dict"]),
    }
    model.requires_grad_(False)
    model.eval()
    model = model.to(device)
    if spec.precision == "fp32" and "vision_encoder" in spec.bf16_modules:
        model.vision_encoder.bfloat16()
    model._init_model_type()
    return model, checkpoint


def build_real_libero_batch(
    spec: SeerRuntimeSpec,
    model,
    *,
    device: str | torch.device,
    sample_index: int = 0,
    seed: int = 42,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load one unaugmented batch through Seer's real converted-data pipeline."""

    seed_everything(seed)
    args = SimpleNamespace(
        rgb_pad=10,
        gripper_pad=4,
        traj_cons=True,
        text_aug=False,
        multi_step_action=1,
        root_dir=spec.dataset_root,
        libero_dataset_name=spec.dataset_name,
        libero_dataset_info_path=spec.dataset_info_path,
        image_primary_size=224,
        image_wrist_size=224,
        window_size=spec.sequence_length,
        dif_ws=False,
        min_window_size=spec.sequence_length,
        max_window_size=spec.sequence_length,
        primary_mode="image_primary",
        small_size=0,
        gripper_width=spec.gripper_width,
        load_libero_file="h5",
        batch_size=1,
        world_size=1,
        workers=1,
        rank=0,
        seed=seed,
    )
    with seer_import_context():
        import clip
        from utils.data_utils import get_libero_finetune_dataset

        data = get_libero_finetune_dataset(args, model.image_processor, clip, epoch=0)
        sample = data.dataset[int(sample_index)]
        batch = data.dataset.collator([sample])

    images_primary, text_tokens, _actions, images_wrist, states, _robot_obs = batch
    if spec.gripper_width:
        input_states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
    else:
        input_states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
        input_states[..., 6:] = (input_states[..., 6:] + 1) // 2
    model_dtype = next(model.action_decoder.parameters()).dtype
    inputs = {
        "image_primary": images_primary[:, : spec.sequence_length].to(device=device, dtype=model_dtype),
        "image_wrist": images_wrist[:, : spec.sequence_length].to(device=device, dtype=model_dtype),
        "state": input_states[:, : spec.sequence_length].to(device=device, dtype=model_dtype),
        "text_token": text_tokens.to(device).unsqueeze(1).repeat(1, spec.sequence_length, 1),
        "action": torch.zeros(1, spec.sequence_length, 7, device=device, dtype=model_dtype),
    }
    audit = {
        "sample_index": int(sample_index),
        "episode_id": str(sample["episode_id"]),
        "language": str(sample["lang"]),
        "raw_batch_shapes": [list(tensor.shape) for tensor in batch if torch.is_tensor(tensor)],
        "model_input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "model_input_dtypes": {key: str(value.dtype) for key, value in inputs.items()},
        "augmentation": {"rgb_pad": -1, "gripper_pad": -1, "enabled": False},
    }
    return inputs, audit
