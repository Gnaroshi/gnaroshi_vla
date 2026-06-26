#!/usr/bin/env python3
"""CPU LR-NODE parity checks.

This script isolates Seer common-path parity without depending on real CLIP,
real ViT checkpoint loading, LIBERO, CUDA, or dataloaders.
"""

from __future__ import annotations

import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models.seer_model as seer_model
from models.seer_model import SeerAgent
from utils.train_utils import _preserve_torch_rng, normalize_patchfied_image, patchify


SEED = 1234


class DummyClip(nn.Module):
    def encode_text(self, text_tokens):
        x = text_tokens.float()
        base = torch.linspace(0.0, 1.0, 512, device=x.device, dtype=torch.float32)
        scale = x.sum(dim=-1, keepdim=True).remainder(17.0) / 17.0
        return base.unsqueeze(0).expand(x.shape[0], -1) + scale


def _dummy_clip_load(*_args, **_kwargs):
    return DummyClip(), (lambda image: image)


@contextmanager
def patched_external_loaders():
    old_clip_load = seer_model.clip.load
    old_torch_load = torch.load
    seer_model.clip.load = _dummy_clip_load
    torch.load = lambda *_args, **_kwargs: {"model": {}}
    try:
        yield
    finally:
        seer_model.clip.load = old_clip_load
        torch.load = old_torch_load


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def is_lrnode_name(name):
    return name.startswith("lrnode_delta_encoder.") or name.startswith("lrnode_dynamics.")


def patch_frozen_encoders(model):
    def forward_encoder(self, x, mask_ratio=0.0):
        n = x.shape[0]
        device = x.device
        dtype = x.dtype
        token = torch.linspace(-1.0, 1.0, 197 * 768, device=device, dtype=torch.float32)
        token = token.view(1, 197, 768).expand(n, -1, -1).to(dtype=dtype)
        image_scale = x.float().mean(dim=(1, 2, 3), keepdim=True).view(n, 1, 1).to(dtype=dtype)
        return token + image_scale, None, None

    model.vision_encoder.forward_encoder = MethodType(forward_encoder, model.vision_encoder)
    model.vision_encoder.requires_grad_(False)
    model.clip_model.requires_grad_(False)
    model._init_model_type()


def build_model(use_lrnode):
    model = SeerAgent(
        finetune_type="libero_finetune",
        clip_device="cpu",
        vit_checkpoint_path="dummy_vit.pth",
        sequence_length=3,
        num_resampler_query=2,
        num_obs_token_per_image=4,
        calvin_input_image_size=32,
        patch_size=16,
        action_pred_steps=2,
        obs_pred=True,
        atten_only_obs=False,
        attn_robot_proprio_state=False,
        atten_goal=0,
        atten_goal_state=False,
        mask_l_obs_ratio=0.0,
        transformer_layers=2,
        hidden_dim=32,
        transformer_heads=4,
        phase="finetune",
        gripper_width=True,
        use_lrnode_latent_update=int(use_lrnode),
        lrnode_hidden_dim=64,
        lrnode_motion_dim=16,
        lrnode_fast_encoder_type="diffcnn",
        lrnode_detach_input_latent=1,
        lrnode_detach_teacher_latent=1,
        lrnode_freeze_action_head_for_lrnode=1,
        lrnode_use_post_layernorm=0,
        lrnode_multistep_train=0,
        lrnode_train_max_horizon=2,
        lrnode_log_sanity=1,
        lrnode_gate_init_bias=-4.0,
        lrnode_trace=0,
    )
    patch_frozen_encoders(model)
    return model.float()


def common_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if not is_lrnode_name(k)}


def make_batch():
    set_seed(999)
    bsz, window, h, w = 2, 4, 32, 32
    images_primary = torch.randn(bsz, window, 3, h, w)
    images_wrist = torch.randn(bsz, window, 3, h, w)
    states = torch.randn(bsz, window, 8)
    text_tokens = torch.randint(0, 100, (bsz, window, 77))
    arm = torch.empty(bsz, window, 6).uniform_(-1.0, 1.0)
    grip = torch.randint(0, 2, (bsz, window, 1)).float() * 2.0 - 1.0
    actions = torch.cat([arm, grip], dim=-1)
    return images_primary, images_wrist, states, text_tokens, actions


def label_actions(actions):
    actions = actions.clone()
    actions[..., 6:] = (actions[..., 6:] + 1) // 2
    return torch.cat([actions[:, j : 3 + j].unsqueeze(-2) for j in range(2)], dim=-2)


def base_losses(outputs, batch):
    images_primary, images_wrist, _states, _text_tokens, actions = batch
    arm_pred = outputs["arm_pred_action"] if isinstance(outputs, dict) else outputs[0]
    gripper_pred = outputs["gripper_pred_action"] if isinstance(outputs, dict) else outputs[1]
    image_pred = outputs["image_pred"] if isinstance(outputs, dict) else outputs[2]
    labels = label_actions(actions)

    loss_arm = torch.nn.functional.smooth_l1_loss(arm_pred[:, :3], labels[:, :3, :, :6].detach())
    loss_grip = torch.nn.functional.binary_cross_entropy(gripper_pred[:, :3], labels[:, :3, :, 6:].detach())

    label_image_primary = images_primary[:, 1:4].flatten(0, 1)
    label_image_wrist = images_wrist[:, 1:4].flatten(0, 1)
    label_image_primary = normalize_patchfied_image(patchify(label_image_primary, patch_size=16))
    label_image_wrist = normalize_patchfied_image(patchify(label_image_wrist, patch_size=16))
    image_pred = image_pred.reshape(-1, 3, image_pred.shape[1], image_pred.shape[2], image_pred.shape[3])
    image_pred = image_pred[:, :3].reshape(-1, image_pred.shape[2], image_pred.shape[3], image_pred.shape[4])
    loss_image = 0.5 * (
        torch.nn.functional.mse_loss(image_pred[:, 0], label_image_primary.detach())
        + torch.nn.functional.mse_loss(image_pred[:, 1], label_image_wrist.detach())
    )
    total = loss_arm + 0.01 * loss_grip + 0.1 * loss_image
    return total, loss_arm, loss_grip, loss_image


def baseline_forward(model, batch):
    images_primary, images_wrist, states, text_tokens, actions = batch
    return model(
        image_primary=images_primary[:, :3],
        image_wrist=images_wrist[:, :3],
        state=states[:, :3],
        text_token=text_tokens[:, :3],
        action=actions[:, :3],
        return_action_latent=True,
    )


def lrnode_shifted_forward(model, batch):
    images_primary, images_wrist, states, text_tokens, actions = batch
    selected_step = 2
    with _preserve_torch_rng(None):
        with torch.no_grad():
            teacher_outputs = model(
                image_primary=images_primary[:, 1:4],
                image_wrist=images_wrist[:, 1:4],
                state=states[:, 1:4],
                text_token=text_tokens[:, 1:4],
                action=actions[:, 1:4],
                return_action_latent=True,
                lrnode_compute_loss=False,
            )
    teacher_z = teacher_outputs["action_latent"][:, selected_step]
    return model(
        image_primary=images_primary[:, :3],
        image_wrist=images_wrist[:, :3],
        state=states[:, :3],
        text_token=text_tokens[:, :3],
        action=actions[:, :3],
        return_action_latent=True,
        lrnode_compute_loss=True,
        lrnode_key_image_primary=images_primary[:, selected_step],
        lrnode_key_image_wrist=images_wrist[:, selected_step],
        lrnode_cur_image_primary=images_primary[:, selected_step + 1],
        lrnode_cur_image_wrist=images_wrist[:, selected_step + 1],
        lrnode_q_key=states[:, selected_step],
        lrnode_q_cur=states[:, selected_step + 1],
        lrnode_detach_input_latent=True,
        lrnode_detach_teacher_latent=True,
        lrnode_freeze_action_head_for_lrnode=True,
        lrnode_multistep_train=False,
        lrnode_train_max_horizon=2,
        lrnode_z_teacher_next_external=teacher_z,
        lrnode_selected_step=selected_step,
    )


def lrnode_loss(outputs):
    lr_action = torch.cat([outputs["lrnode_arm_action"], outputs["lrnode_gripper_action"]], dim=-1)
    latent = torch.nn.functional.mse_loss(outputs["lrnode_z_pred_next"], outputs["lrnode_z_teacher_next"])
    action = torch.nn.functional.l1_loss(lr_action, outputs["lrnode_teacher_action"].detach())
    smooth = torch.nn.functional.mse_loss(
        outputs["lrnode_z_pred_next"] - outputs["lrnode_z_prev"].detach(),
        torch.zeros_like(outputs["lrnode_z_pred_next"]),
    )
    return 0.05 * latent + 0.1 * action + 0.001 * smooth


def max_abs_common_grad_diff(base, ours):
    diff_count = 0
    max_abs = 0.0
    checked = 0
    ours_params = dict(ours.named_parameters())
    for name, p_base in base.named_parameters():
        if is_lrnode_name(name) or name not in ours_params:
            continue
        p_ours = ours_params[name]
        g_base = p_base.grad
        g_ours = p_ours.grad
        if g_base is None and g_ours is None:
            continue
        checked += 1
        if (g_base is None) != (g_ours is None):
            diff_count += 1
            max_abs = float("inf")
            continue
        diff = (g_base.detach() - g_ours.detach()).abs().max().item()
        if diff != 0.0:
            diff_count += 1
            max_abs = max(max_abs, float(diff))
    return checked, diff_count, max_abs


def max_abs_common_param_diff(base, ours):
    diff_count = 0
    max_abs = 0.0
    checked = 0
    first_diffs = []
    ours_state = common_state_dict(ours)
    for name, value in common_state_dict(base).items():
        if name not in ours_state:
            continue
        if not torch.is_floating_point(value):
            continue
        checked += 1
        equal = torch.equal(value, ours_state[name])
        if not equal:
            diff_count += 1
            diff_tensor = (value - ours_state[name]).abs()
            diff = float(torch.nan_to_num(diff_tensor, nan=0.0).max().item())
            max_abs = max(max_abs, diff)
            if len(first_diffs) < 5:
                first_diffs.append(
                    {
                        "name": name,
                        "max_absdiff_nan_to_num": diff,
                        "base_has_nan": bool(torch.isnan(value).any().item()),
                        "ours_has_nan": bool(torch.isnan(ours_state[name]).any().item()),
                        "equal_nan": bool(torch.allclose(value, ours_state[name], equal_nan=True)),
                    }
                )
    return checked, diff_count, max_abs, first_diffs


def check_init_parity():
    with patched_external_loaders():
        set_seed()
        base = build_model(use_lrnode=False)
        base_rng = torch.get_rng_state().clone()
        set_seed()
        ours = build_model(use_lrnode=True)
        ours_rng = torch.get_rng_state().clone()
    base_sd = common_state_dict(base)
    ours_sd = common_state_dict(ours)
    unequal = []
    for name, value in base_sd.items():
        if name not in ours_sd:
            unequal.append(name)
            continue
        if value.shape != ours_sd[name].shape or not torch.equal(value, ours_sd[name]):
            unequal.append(name)
    return {
        "common_tensor_count": len(base_sd),
        "unequal_common_tensor_count": len(unequal),
        "rng_equal_after_constructor": bool(torch.equal(base_rng, ours_rng)),
        "first_unequal": unequal[:5],
    }


def check_loss_gradient_update_parity():
    batch = make_batch()
    with patched_external_loaders():
        set_seed()
        base = build_model(use_lrnode=False)
        set_seed()
        ours = build_model(use_lrnode=True)

    base.train()
    ours.train()
    set_seed(2026)
    base_out = baseline_forward(base, batch)
    base_loss, base_arm, base_grip, base_img = base_losses(base_out, batch)
    set_seed(2026)
    ours_out = lrnode_shifted_forward(ours, batch)
    ours_base_loss, ours_arm, ours_grip, ours_img = base_losses(ours_out, batch)
    ours_total = ours_base_loss + lrnode_loss(ours_out)

    output_diffs = {
        "arm": float((base_out["arm_pred_action"] - ours_out["arm_pred_action"]).abs().max().item()),
        "gripper": float((base_out["gripper_pred_action"] - ours_out["gripper_pred_action"]).abs().max().item()),
        "image": float((base_out["image_pred"] - ours_out["image_pred"]).abs().max().item()),
        "latent": float((base_out["action_latent"] - ours_out["action_latent"]).abs().max().item()),
    }

    opt_base = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    opt_ours = torch.optim.AdamW([p for p in ours.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    opt_base.zero_grad()
    opt_ours.zero_grad()
    base_loss.backward()
    ours_total.backward()
    grad_checked, grad_diff_count, grad_max_abs = max_abs_common_grad_diff(base, ours)

    torch.nn.utils.clip_grad_norm_([p for p in base.parameters() if p.requires_grad], 0.1)
    ours_non_lrnode = [p for name, p in ours.named_parameters() if p.requires_grad and not is_lrnode_name(name)]
    ours_lrnode = [p for name, p in ours.named_parameters() if p.requires_grad and is_lrnode_name(name)]
    torch.nn.utils.clip_grad_norm_(ours_non_lrnode, 0.1)
    torch.nn.utils.clip_grad_norm_(ours_lrnode, 0.1)
    opt_base.step()
    opt_ours.step()
    param_checked, param_diff_count, param_max_abs, param_first_diffs = max_abs_common_param_diff(base, ours)
    lrnode_grad_nonzero = sum(
        1
        for name, p in ours.named_parameters()
        if is_lrnode_name(name) and p.grad is not None and float(p.grad.detach().abs().max().item()) > 0.0
    )

    return {
        "base_loss": [float(base_loss.item()), float(base_arm.item()), float(base_grip.item()), float(base_img.item())],
        "ours_base_loss": [
            float(ours_base_loss.item()),
            float(ours_arm.item()),
            float(ours_grip.item()),
            float(ours_img.item()),
        ],
        "base_loss_absdiff": float(abs(base_loss.item() - ours_base_loss.item())),
        "main_output_max_absdiff": output_diffs,
        "common_grad_checked": grad_checked,
        "common_grad_diff_count": grad_diff_count,
        "common_grad_max_absdiff": grad_max_abs,
        "common_param_checked_after_step": param_checked,
        "common_param_diff_count_after_step": param_diff_count,
        "common_param_max_absdiff_after_step": param_max_abs,
        "common_param_first_diffs_after_step": param_first_diffs,
        "lrnode_nonzero_grad_tensor_count": lrnode_grad_nonzero,
    }


def check_eval_full_forward_parity():
    batch = make_batch()
    with patched_external_loaders():
        set_seed()
        base = build_model(use_lrnode=False)
        set_seed()
        ours = build_model(use_lrnode=True)
    base.eval()
    ours.eval()
    with torch.no_grad():
        base_out = baseline_forward(base, batch)
        ours_out = baseline_forward(ours, batch)
    return {
        "eval_arm_max_absdiff": float((base_out["arm_pred_action"] - ours_out["arm_pred_action"]).abs().max().item()),
        "eval_gripper_max_absdiff": float(
            (base_out["gripper_pred_action"] - ours_out["gripper_pred_action"]).abs().max().item()
        ),
        "eval_latent_max_absdiff": float((base_out["action_latent"] - ours_out["action_latent"]).abs().max().item()),
    }


def main():
    report = {
        "init_parity": check_init_parity(),
        "train_shifted_teacher_parity": check_loss_gradient_update_parity(),
        "eval_full_forward_parity": check_eval_full_forward_parity(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
