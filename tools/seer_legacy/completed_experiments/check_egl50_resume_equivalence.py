#!/usr/bin/env python3

import importlib.util
import hashlib
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "architectures/seer/upstream"
LOCKED_MODEL = REPO / "tools/seer/fixtures/egl50_phase1_seer_model.py"
LOCKED_MODEL_SHA256 = "3f72676e3f64ace7396eefaf9324dc70ad9867ec8b3cfb206966d110e0735a7c"
if hashlib.sha256(LOCKED_MODEL.read_bytes()).hexdigest() != LOCKED_MODEL_SHA256:
    raise RuntimeError(f"locked phase-1 Seer fixture hash mismatch: {LOCKED_MODEL}")
sys.path.insert(0, str(UPSTREAM))

import models.seer_model as current_module


spec = importlib.util.spec_from_file_location("locked_seer_model", LOCKED_MODEL)
locked_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(locked_module)


class DummyClip(nn.Module):
    def encode_text(self, text_tokens):
        x = text_tokens.float()
        base = torch.linspace(0.0, 1.0, 512, device=x.device)
        scale = x.sum(dim=-1, keepdim=True).remainder(17.0) / 17.0
        return base.unsqueeze(0).expand(x.shape[0], -1) + scale


@contextmanager
def patched_loaders():
    old_clip_load = current_module.clip.load
    old_torch_load = torch.load
    current_module.clip.load = lambda *_args, **_kwargs: (DummyClip(), lambda image: image)
    torch.load = lambda *_args, **_kwargs: {"model": {}}
    try:
        yield
    finally:
        current_module.clip.load = old_clip_load
        torch.load = old_torch_load


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def patch_encoder(model):
    def forward_encoder(self, x, mask_ratio=0.0):
        n = x.shape[0]
        token = torch.linspace(-1.0, 1.0, 197 * 768, device=x.device)
        token = token.view(1, 197, 768).expand(n, -1, -1).to(dtype=x.dtype)
        scale = x.float().mean(dim=(1, 2, 3), keepdim=True).view(n, 1, 1).to(x.dtype)
        return token + scale, None, None

    model.vision_encoder.forward_encoder = MethodType(forward_encoder, model.vision_encoder)
    model.vision_encoder.requires_grad_(False)
    model.clip_model.requires_grad_(False)
    model._init_model_type()


def build(cls):
    model = cls(
        finetune_type="libero_finetune",
        clip_device="cpu",
        vit_checkpoint_path="dummy.pth",
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
        use_lrnode_latent_update=1,
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
    ).float()
    patch_encoder(model)
    model.eval()
    return model


def maxdiff(left, right):
    return float((left - right).abs().max().item())


def main():
    with patched_loaders():
        seed(1234)
        locked = build(locked_module.SeerAgent)
        seed(1234)
        current = build(current_module.SeerAgent)

    locked_state = locked.state_dict()
    current_state = current.state_dict()
    if locked_state.keys() != current_state.keys():
        raise RuntimeError("state-dict keys differ between locked and current Seer")
    unequal_state = [
        name for name in locked_state if not torch.equal(locked_state[name], current_state[name])
    ]
    if unequal_state:
        raise RuntimeError(f"initial state differs: {unequal_state[:8]}")

    seed(999)
    primary = torch.randn(2, 3, 3, 32, 32)
    wrist = torch.randn(2, 3, 3, 32, 32)
    state = torch.randn(2, 3, 8)
    text = torch.randint(0, 100, (2, 3, 77))
    action = torch.randn(2, 3, 7)
    action[..., 6] = torch.where(action[..., 6] >= 0, 1.0, -1.0)

    kwargs = dict(
        image_primary=primary,
        image_wrist=wrist,
        state=state,
        text_token=text,
        action=action,
        return_action_latent=True,
        lrnode_compute_loss=False,
    )
    with torch.no_grad():
        seed(777)
        old_out = locked(**kwargs)
        seed(777)
        new_out = current(**kwargs)

    full_keys = ("arm_pred_action", "gripper_pred_action", "action_latent")
    full_diffs = {key: maxdiff(old_out[key], new_out[key]) for key in full_keys}
    z_old = old_out["action_latent"][:, -1]
    z_new = new_out["action_latent"][:, -1]
    key_primary = torch.randn(2, 3, 32, 32)
    cur_primary = torch.randn(2, 3, 32, 32)
    key_wrist = torch.randn(2, 3, 32, 32)
    cur_wrist = torch.randn(2, 3, 32, 32)
    q_key = torch.randn(2, 8)
    q_cur = torch.randn(2, 8)
    update_kwargs = dict(
        key_image_primary=key_primary,
        key_image_wrist=key_wrist,
        cur_image_primary=cur_primary,
        cur_image_wrist=cur_wrist,
        q_key=q_key,
        q_cur=q_cur,
        dt=1.0,
        age=2.0,
    )
    with torch.no_grad():
        old_next = locked.lrnode_predict_next_latent(z_prev=z_old, **update_kwargs)
        new_next = current.lrnode_predict_next_latent(z_prev=z_new, **update_kwargs)
        old_arm, old_grip = locked.decode_action_from_latent(old_next)
        new_arm, new_grip = current.decode_action_from_latent(new_next)

    result = {
        "status": "PASS",
        "locked_model": str(LOCKED_MODEL),
        "locked_model_sha256_expected": LOCKED_MODEL_SHA256,
        "state_tensor_count": len(locked_state),
        "unequal_initial_state_count": len(unequal_state),
        "full_forward_max_absdiff": full_diffs,
        "latent_update_max_absdiff": maxdiff(old_next, new_next),
        "decoded_arm_max_absdiff": maxdiff(old_arm, new_arm),
        "decoded_gripper_max_absdiff": maxdiff(old_grip, new_grip),
    }
    values = list(full_diffs.values()) + [
        result["latent_update_max_absdiff"],
        result["decoded_arm_max_absdiff"],
        result["decoded_gripper_max_absdiff"],
    ]
    if any(value != 0.0 for value in values):
        raise RuntimeError(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
