import time
import os
import json
from contextlib import suppress
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from einops import rearrange
from pdb import set_trace
import numpy as np
import torch.distributed as dist
from PIL import Image


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    else:
        cast_dtype = torch.float32
    return cast_dtype

def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress

def get_ckpt_name(args, epoch=-1):
    return f'{epoch}.pth'

def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """

    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

    h = w = imgs.shape[2] // patch_size
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, patch_size, w, patch_size))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, patch_size**2 * 3))

    return x

def normalize_patchfied_image(patchfied_imgs):
    mean = patchfied_imgs.mean(dim=-1, keepdim=True)
    var = patchfied_imgs.var(dim=-1, keepdim=True)
    patchfied_imgs = (patchfied_imgs - mean) / (var + 1.e-6)**.5

    return patchfied_imgs

def module_grad_norm(module):
    total_sq_norm = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        total_sq_norm += float(param.grad.detach().float().norm(2).item() ** 2)
    return total_sq_norm ** 0.5


def _float_item(value):
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    return float(value)


def _tensor_stats(prefix, tensor):
    if tensor is None:
        return {}
    x = tensor.detach().float()
    return {
        f"{prefix}_mean": x.mean(),
        f"{prefix}_std": x.std(unbiased=False),
        f"{prefix}_min": x.min(),
        f"{prefix}_max": x.max(),
        f"{prefix}_norm": x.norm(dim=-1).mean() if x.dim() > 0 else x.abs(),
    }


def _cosine_mean(a, b):
    if a is None or b is None:
        return None
    return F.cosine_similarity(a.detach().float().reshape(-1, a.shape[-1]), b.detach().float().reshape(-1, b.shape[-1]), dim=-1).mean()


def _safe_ratio(num, den, eps=1e-8):
    if torch.is_tensor(num) or torch.is_tensor(den):
        return num / (den + eps)
    return float(num) / (float(den) + eps)


def _scalar_log_dict(metrics):
    out = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.numel() != 1:
                value = value.detach().float().mean()
            out[key] = value.detach().float()
        else:
            out[key] = torch.tensor(float(value))
    return out


def _all_reduce_scalar_dict(metrics, device):
    tensor_metrics = _scalar_log_dict(metrics)
    if not tensor_metrics:
        return {}
    keys = list(tensor_metrics.keys())
    vals = torch.stack([tensor_metrics[k].to(device=device) for k in keys])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        vals = vals / dist.get_world_size()
    return {k: float(v.item()) for k, v in zip(keys, vals)}


def _gather_mean_scalar_dict(metrics):
    local = {}
    for key, value in metrics.items():
        if value is None:
            continue
        try:
            local[key] = _float_item(value)
        except Exception:
            continue
    if not (dist.is_available() and dist.is_initialized()):
        return local
    gathered = [None for _ in range(dist.get_world_size())] if dist.get_rank() == 0 else None
    dist.gather_object(local, gathered, dst=0)
    if dist.get_rank() != 0:
        return {}
    accum = {}
    counts = {}
    for item in gathered:
        if not item:
            continue
        for key, value in item.items():
            accum[key] = accum.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: accum[key] / counts[key] for key in accum}


def _corrcoef(x, y):
    if x is None or y is None:
        return None
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    if x.numel() < 2 or y.numel() < 2 or x.numel() != y.numel():
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) == 0.0:
        return None
    return (x * y).sum() / denom


def _save_lrnode_debug_artifacts(args, global_step, tensors):
    interval = int(getattr(args, "lrnode_debug_artifact_interval", 0))
    if interval <= 0 or getattr(args, "rank", 0) != 0 or global_step % interval != 0:
        return
    out_dir = Path(args.save_checkpoint_path) / args.run_name / "lrnode_debug" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"gs_{global_step:06d}"

    summary = {}
    arrays = {}
    for key, tensor in tensors.items():
        if tensor is None:
            continue
        try:
            x = tensor.detach().float().cpu()
            summary[key] = {
                "shape": list(x.shape),
                "mean": float(x.mean().item()),
                "std": float(x.std(unbiased=False).item()),
                "min": float(x.min().item()),
                "max": float(x.max().item()),
            }
            arrays[key] = x[:4].numpy()
        except Exception:
            continue
    with open(out_dir / f"{stem}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if arrays:
        np.savez_compressed(out_dir / f"{stem}_sample.npz", **arrays)

    # Lightweight image contact sheet for the first primary/wrist key/current pair.
    try:
        imgs = []
        for key in ["key_primary", "cur_primary", "diff_primary", "key_wrist", "cur_wrist", "diff_wrist"]:
            if key not in tensors or tensors[key] is None:
                continue
            arr = tensors[key].detach().float().cpu()
            while arr.dim() > 3:
                arr = arr[0]
            if arr.shape[0] == 3:
                arr = arr.permute(1, 2, 0)
            arr = arr.numpy()
            arr = arr - arr.min()
            arr = arr / (arr.max() + 1e-6)
            imgs.append((arr * 255).astype(np.uint8))
        if imgs:
            h = max(img.shape[0] for img in imgs)
            padded = []
            for img in imgs:
                if img.shape[0] != h:
                    pad = np.zeros((h - img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
                    img = np.concatenate([img, pad], axis=0)
                padded.append(img)
            Image.fromarray(np.concatenate(padded, axis=1)).save(out_dir / f"{stem}_images.png")
    except Exception:
        pass

def train_one_epoch_calvin(
    args,
    model,
    epoch,
    calvin_loader,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    num_batches_per_epoch_calvin = calvin_loader.num_batches
    num_batches_per_epoch = num_batches_per_epoch_calvin
    total_training_steps = num_batches_per_epoch * args.num_epochs
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    model.train()
    if (
        args.rank == 0
        and args.use_lrnode_latent_update
        and args.lrnode_train_latent_distill
        and args.lrnode_query_interval > 2
        and not args.lrnode_multistep_train
    ):
        print(
            "[LR-NODE warning] eval query_interval > 2 but lrnode_multistep_train=0. "
            "The LR-NODE student is only trained for one-step latent updates."
        )

    # setup logging
    step_time_m = (
        AverageMeter()
    )  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both calvin (= 1 batch regardless of gradient accum)
    end = time.time()
    # loop through dataloader
    t = tqdm(
        enumerate(calvin_loader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    )
    t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
    mv_avg_loss = []
    
    for num_steps, batch_calvin in t:
        data_time_m.update(time.time() - end)
        global_step = num_steps + epoch * num_batches_per_epoch

        # images
        images_primary = batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True)
        images_wrist = batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True)
        # text tokens
        text_tokens = batch_calvin[1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, args.window_size, 1)
        
        # states
        states = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        if args.gripper_width:
            input_states = torch.cat([states[..., :6], states[..., -2:]], dim=-1)
        else:
            input_states = torch.cat([states[..., :6], states[..., [-1]]], dim=-1)
            input_states[..., 6:] = (input_states[..., 6:] + 1) // 2
        
        # self_key_point
        self_keypoints = None
        
        # actions
        actions = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        # label. [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        actions[..., 6:] = (actions[..., 6:] + 1) // 2
        input_image_primary = images_primary[:, :args.sequence_length, :]
        input_image_wrist = images_wrist[:, :args.sequence_length, :]
        input_text_token = text_tokens[:, :args.sequence_length, :]
        input_state = input_states[:, :args.sequence_length, :]

        # label action
        label_actions = torch.cat([actions[:, j:args.sequence_length-args.atten_goal+j, :].unsqueeze(-2) for j in range(args.action_pred_steps)], dim=-2) 

        train_lrnode = bool(args.use_lrnode_latent_update and args.lrnode_train_latent_distill)
        lrnode_teacher_target_mode = getattr(args, "lrnode_teacher_target_mode", "shifted_context")
        if lrnode_teacher_target_mode not in {"shifted_context", "adjacent_sequence"}:
            raise RuntimeError(
                f"Unsupported lrnode_teacher_target_mode={lrnode_teacher_target_mode}. "
                "Expected 'shifted_context' or 'adjacent_sequence'."
            )
        base_model = model.module if hasattr(model, "module") else model
        if train_lrnode and not getattr(base_model, "use_lrnode_latent_update", False):
            raise RuntimeError("lrnode_train_latent_distill=1 requires use_lrnode_latent_update=1")

        with autocast():  # image_primary, image_wrist, state, language_instruction
            use_shifted_context_target = train_lrnode and lrnode_teacher_target_mode == "shifted_context"
            model_outputs = model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=actions[:, :args.sequence_length, :],
                return_action_latent=train_lrnode,
                lrnode_compute_loss=train_lrnode and not use_shifted_context_target,
                lrnode_key_image_primary=input_image_primary[:, :-1] if train_lrnode and not use_shifted_context_target else None,
                lrnode_key_image_wrist=input_image_wrist[:, :-1] if train_lrnode and not use_shifted_context_target else None,
                lrnode_cur_image_primary=input_image_primary[:, 1:] if train_lrnode and not use_shifted_context_target else None,
                lrnode_cur_image_wrist=input_image_wrist[:, 1:] if train_lrnode and not use_shifted_context_target else None,
                lrnode_q_key=input_state[:, :-1] if train_lrnode and not use_shifted_context_target else None,
                lrnode_q_cur=input_state[:, 1:] if train_lrnode and not use_shifted_context_target else None,
                lrnode_detach_input_latent=bool(args.lrnode_detach_input_latent),
                lrnode_detach_teacher_latent=bool(args.lrnode_detach_teacher_latent),
                lrnode_freeze_action_head_for_lrnode=bool(args.lrnode_freeze_action_head_for_lrnode),
                lrnode_multistep_train=bool(args.lrnode_multistep_train),
                lrnode_train_max_horizon=int(args.lrnode_train_max_horizon),
            )
            if train_lrnode:
                arm_pred_action = model_outputs["arm_pred_action"]
                gripper_pred_action = model_outputs["gripper_pred_action"]
                image_pred = model_outputs["image_pred"]
                arm_pred_state = model_outputs["arm_pred_state"]
                gripper_pred_state = model_outputs["gripper_pred_state"]
                loss_arm_action = model_outputs["loss_arm_action"]
                action_latent_full = model_outputs["action_latent"]
                if use_shifted_context_target:
                    if bool(args.lrnode_multistep_train):
                        raise RuntimeError(
                            "lrnode_multistep_train=1 is only implemented for "
                            "lrnode_teacher_target_mode=adjacent_sequence."
                        )
                    if images_primary.shape[1] < args.sequence_length + 1:
                        raise RuntimeError(
                            "LR-NODE shifted_context target requires one extra frame beyond "
                            f"sequence_length={args.sequence_length}, got image window length {images_primary.shape[1]}"
                        )
                    if input_states.shape[1] < args.sequence_length + 1:
                        raise RuntimeError(
                            "LR-NODE shifted_context target requires one extra state beyond "
                            f"sequence_length={args.sequence_length}, got state window length {input_states.shape[1]}"
                        )
                    if action_latent_full is None or action_latent_full.dim() != 4:
                        raise RuntimeError(
                            "LR-NODE shifted_context target requires action_latent_full "
                            f"[B, S, action_pred_steps, D], got {None if action_latent_full is None else tuple(action_latent_full.shape)}"
                        )

                    input_image_primary_next = images_primary[:, 1:args.sequence_length + 1, :]
                    input_image_wrist_next = images_wrist[:, 1:args.sequence_length + 1, :]
                    input_text_token_next = text_tokens[:, 1:args.sequence_length + 1, :]
                    input_state_next = input_states[:, 1:args.sequence_length + 1, :]
                    action_next = actions[:, 1:args.sequence_length + 1, :]
                    if input_text_token_next.shape[1] != args.sequence_length:
                        raise RuntimeError(
                            "LR-NODE shifted_context target requires text context length "
                            f"{args.sequence_length}, got {input_text_token_next.shape[1]}"
                        )

                    with torch.no_grad():
                        teacher_outputs_next = model(
                            image_primary=input_image_primary_next,
                            image_wrist=input_image_wrist_next,
                            state=input_state_next,
                            text_token=input_text_token_next,
                            action=action_next,
                            return_action_latent=True,
                            lrnode_compute_loss=False,
                        )
                    if teacher_outputs_next["action_latent"] is None or teacher_outputs_next["action_latent"].dim() != 4:
                        raise RuntimeError(
                            "LR-NODE shifted_context teacher forward did not return action_latent "
                            f"[B, S, action_pred_steps, D], got "
                            f"{None if teacher_outputs_next['action_latent'] is None else tuple(teacher_outputs_next['action_latent'].shape)}"
                        )

                    selected_step = int(getattr(args, "lrnode_context_selected_step", -1))
                    if selected_step < 0:
                        selected_step = args.sequence_length + selected_step
                    if selected_step < 0 or selected_step >= args.sequence_length:
                        raise RuntimeError(
                            f"lrnode_context_selected_step resolves to {selected_step}, "
                            f"but valid range is [0, {args.sequence_length - 1}]"
                        )

                    # Architecture-agnostic teacher-probe target:
                    # C_t is the normal policy context, C_{t+1} is the same policy context shifted by one
                    # environment step. For Seer, we probe the selected action-token latent from both contexts.
                    z_prev = action_latent_full[:, selected_step]
                    z_teacher_next = teacher_outputs_next["action_latent"][:, selected_step]
                    if bool(args.lrnode_detach_input_latent):
                        z_prev = z_prev.detach()
                    if bool(args.lrnode_detach_teacher_latent):
                        z_teacher_next = z_teacher_next.detach()

                    z_pred_next = base_model.lrnode_predict_next_latent(
                        z_prev=z_prev,
                        key_image_primary=input_image_primary[:, selected_step],
                        key_image_wrist=input_image_wrist[:, selected_step],
                        cur_image_primary=input_image_primary_next[:, selected_step],
                        cur_image_wrist=input_image_wrist_next[:, selected_step],
                        q_key=input_state[:, selected_step],
                        q_cur=input_state_next[:, selected_step],
                        dt=1.0,
                        age=1.0,
                    )
                    lrnode_gate = getattr(base_model.lrnode_dynamics, "last_gate", None)
                    lrnode_u_delta = getattr(base_model.lrnode_delta_encoder, "last_u_delta", None)
                    lrnode_dz = getattr(base_model.lrnode_dynamics, "last_dz", None)
                    lrnode_update = getattr(base_model.lrnode_dynamics, "last_update", None)
                    if z_pred_next.shape != z_teacher_next.shape:
                        raise RuntimeError(
                            f"LR-NODE shifted_context latent prediction shape mismatch: "
                            f"pred={tuple(z_pred_next.shape)}, teacher={tuple(z_teacher_next.shape)}"
                        )

                    lrnode_arm_action, lrnode_gripper_action = base_model.decode_lrnode_action_from_latent(
                        z_pred_next,
                        freeze_action_head=bool(args.lrnode_freeze_action_head_for_lrnode),
                    )
                    with torch.no_grad():
                        teacher_arm_action, teacher_gripper_action = base_model.decode_action_from_latent(
                            z_teacher_next.detach()
                        )
                        hold_arm_action, hold_gripper_action = base_model.decode_action_from_latent(
                            z_prev.detach()
                        )
                        lrnode_teacher_action = torch.cat([teacher_arm_action, teacher_gripper_action], dim=-1)
                        lrnode_hold_action = torch.cat([hold_arm_action, hold_gripper_action], dim=-1)
                else:
                    z_prev = model_outputs["lrnode_z_prev"]
                    z_teacher_next = model_outputs["lrnode_z_teacher_next"]
                    z_pred_next = model_outputs["lrnode_z_pred_next"]
                    lrnode_arm_action = model_outputs["lrnode_arm_action"]
                    lrnode_gripper_action = model_outputs["lrnode_gripper_action"]
                    lrnode_teacher_action = model_outputs["lrnode_teacher_action"]
                    lrnode_hold_action = model_outputs["lrnode_hold_action"]
                    lrnode_gate = model_outputs["lrnode_gate"]
                    lrnode_u_delta = model_outputs.get("lrnode_u_delta")
                    lrnode_dz = model_outputs.get("lrnode_dz")
                    lrnode_update = model_outputs.get("lrnode_update")
            else:
                arm_pred_action, gripper_pred_action, image_pred, arm_pred_state, gripper_pred_state, loss_arm_action = model_outputs
                lrnode_u_delta = None
                lrnode_dz = None
                lrnode_update = None
        # loss_action
        if args.loss_action and args.action_pred_steps:
            loss_arm_action = torch.nn.functional.smooth_l1_loss(
                            arm_pred_action[:, :args.sequence_length-args.atten_goal], 
                            label_actions[:, :args.sequence_length-args.atten_goal, :, :6].detach())
            loss_gripper_action = torch.nn.functional.binary_cross_entropy(
                            gripper_pred_action[:, :args.sequence_length-args.atten_goal], 
                            label_actions[:, :args.sequence_length-args.atten_goal, :, 6:].detach())
        else:
            loss_arm_action = torch.tensor([0.0]).to(device_id)
            loss_gripper_action = torch.tensor([0.0]).to(device_id)

        # loss_image 
        if args.loss_image and args.obs_pred:
            label_image_primary = images_primary[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)
            label_image_wrist = images_wrist[:, args.future_steps:args.future_steps+args.sequence_length-args.atten_goal, :].flatten(0, 1)
            label_image_primary = patchify(label_image_primary, patch_size=args.patch_size)
            label_image_wrist = patchify(label_image_wrist, patch_size=args.patch_size)
            label_image_primary = normalize_patchfied_image(label_image_primary)
            label_image_wrist = normalize_patchfied_image(label_image_wrist)
            image_pred = image_pred.reshape(-1, args.sequence_length, image_pred.shape[1], image_pred.shape[2], image_pred.shape[3])
            image_pred = image_pred[:, :args.sequence_length-args.atten_goal]
            image_pred = image_pred.reshape(-1, image_pred.shape[2], image_pred.shape[3], image_pred.shape[4])
            loss_image = 0.5 * (torch.nn.functional.mse_loss(
                            image_pred[:, 0, :, :], 
                            label_image_primary.detach()) + 
                            torch.nn.functional.mse_loss(
                            image_pred[:, 1, :, :], 
                            label_image_wrist.detach()))
        else:
            loss_image = torch.tensor([0.0]).to(device_id)

        loss_lrnode_latent = torch.tensor([0.0]).to(device_id)
        loss_lrnode_action_distill = torch.tensor([0.0]).to(device_id)
        loss_lrnode_smooth = torch.tensor([0.0]).to(device_id)
        loss_lrnode_bc = torch.tensor([0.0]).to(device_id)
        loss_lrnode_hold_latent = torch.tensor([0.0]).to(device_id)
        loss_lrnode_hold_action = torch.tensor([0.0]).to(device_id)
        lrnode_z_prev_mean = torch.tensor([0.0]).to(device_id)
        lrnode_z_prev_std = torch.tensor([0.0]).to(device_id)
        lrnode_z_teacher_mean = torch.tensor([0.0]).to(device_id)
        lrnode_z_teacher_std = torch.tensor([0.0]).to(device_id)
        lrnode_z_pred_mean = torch.tensor([0.0]).to(device_id)
        lrnode_z_pred_std = torch.tensor([0.0]).to(device_id)
        lrnode_gate_mean = torch.tensor([0.0]).to(device_id)
        lrnode_gate_std = torch.tensor([0.0]).to(device_id)
        lrnode_gate_min = torch.tensor([0.0]).to(device_id)
        lrnode_gate_max = torch.tensor([0.0]).to(device_id)
        train_log_metrics = {}
        lrnode_action = None

        if train_lrnode:
            if action_latent_full is None:
                raise RuntimeError("LR-NODE distillation requires action_pred_steps > 0 and action latent output")
            if action_latent_full.dim() != 4:
                raise RuntimeError(
                    f"Expected action_latent_full [B, S, action_pred_steps, D], got {tuple(action_latent_full.shape)}"
                )
            if z_pred_next is not None:
                loss_lrnode_latent = torch.nn.functional.mse_loss(z_pred_next, z_teacher_next)
                lrnode_action = torch.cat([lrnode_arm_action, lrnode_gripper_action], dim=-1)
                if lrnode_teacher_action is None:
                    raise RuntimeError("LR-NODE action distillation requires lrnode_teacher_action")
                if lrnode_action.shape != lrnode_teacher_action.shape:
                    raise RuntimeError(
                        f"LR-NODE action distillation shape mismatch: pred={tuple(lrnode_action.shape)}, "
                        f"teacher={tuple(lrnode_teacher_action.shape)}"
                    )
                loss_lrnode_action_distill = torch.nn.functional.l1_loss(
                    lrnode_action,
                    lrnode_teacher_action.detach(),
                )
                loss_lrnode_hold_latent = torch.nn.functional.mse_loss(
                    z_prev.detach(),
                    z_teacher_next.detach(),
                )
                if lrnode_hold_action is not None:
                    loss_lrnode_hold_action = torch.nn.functional.l1_loss(
                        lrnode_hold_action.detach(),
                        lrnode_teacher_action.detach(),
                    )

                loss_lrnode_smooth = torch.nn.functional.mse_loss(
                    z_pred_next - z_prev.detach(),
                    torch.zeros_like(z_pred_next),
                )

                lrnode_z_prev_mean = z_prev.detach().float().mean()
                lrnode_z_prev_std = z_prev.detach().float().std(unbiased=False)
                lrnode_z_teacher_mean = z_teacher_next.detach().float().mean()
                lrnode_z_teacher_std = z_teacher_next.detach().float().std(unbiased=False)
                lrnode_z_pred_mean = z_pred_next.detach().float().mean()
                lrnode_z_pred_std = z_pred_next.detach().float().std(unbiased=False)
                if lrnode_gate is not None:
                    lrnode_gate_float = lrnode_gate.detach().float()
                    lrnode_gate_mean = lrnode_gate_float.mean()
                    lrnode_gate_std = lrnode_gate_float.std(unbiased=False)
                    lrnode_gate_min = lrnode_gate_float.min()
                    lrnode_gate_max = lrnode_gate_float.max()

                if args.lrnode_bc_weight > 0 and lrnode_teacher_target_mode == "shifted_context":
                    raise RuntimeError(
                        "lrnode_bc_weight > 0 is not supported with "
                        "lrnode_teacher_target_mode=shifted_context. Use latent/action distillation "
                        "or define an explicit shifted-context BC label alignment first."
                    )
                if args.lrnode_bc_weight > 0 and args.lrnode_multistep_train:
                    raise RuntimeError("lrnode_bc_weight > 0 is not supported with lrnode_multistep_train=1 yet")
                if args.lrnode_bc_weight > 0 and label_actions.shape[1] > 1:
                    bc_pair_len = min(lrnode_action.shape[1], label_actions.shape[1] - 1)
                    if bc_pair_len > 0:
                        lrnode_action_for_bc = lrnode_action[:, :bc_pair_len]
                        label_action_for_bc = label_actions[:, 1:1 + bc_pair_len].detach()
                        loss_lrnode_bc_arm = torch.nn.functional.smooth_l1_loss(
                            lrnode_action_for_bc[..., :6],
                            label_action_for_bc[..., :6],
                        )
                        loss_lrnode_bc_gripper = torch.nn.functional.binary_cross_entropy(
                            lrnode_action_for_bc[..., 6:],
                            label_action_for_bc[..., 6:],
                        )
                        loss_lrnode_bc = loss_lrnode_bc_arm + loss_lrnode_bc_gripper

        base_loss_weighted = (
            args.loss_arm_action_ratio * loss_arm_action
            + args.loss_gripper_action_ratio * loss_gripper_action
            + 0.1 * loss_image
        )
        lrnode_loss_weighted = (
            args.lrnode_latent_weight * loss_lrnode_latent
            + args.lrnode_action_distill_weight * loss_lrnode_action_distill
            + args.lrnode_smooth_weight * loss_lrnode_smooth
            + args.lrnode_bc_weight * loss_lrnode_bc
        )
        loss_calvin = base_loss_weighted + lrnode_loss_weighted

        train_log_metrics.update(
            {
                "train/total_loss": loss_calvin,
                "train/base_total_loss_without_lrnode": base_loss_weighted,
                "train/lrnode_total_loss_weighted": lrnode_loss_weighted,
                "train/base/loss_arm_action_raw": loss_arm_action,
                "train/base/loss_arm_action_weighted": args.loss_arm_action_ratio * loss_arm_action,
                "train/base/loss_gripper_action_raw": loss_gripper_action,
                "train/base/loss_gripper_action_weighted": args.loss_gripper_action_ratio * loss_gripper_action,
                "train/base/loss_image_raw": loss_image,
                "train/base/loss_image_weighted": 0.1 * loss_image,
                "train/lrnode/loss_latent_raw": loss_lrnode_latent,
                "train/lrnode/loss_latent_weighted": args.lrnode_latent_weight * loss_lrnode_latent,
                "train/lrnode/loss_action_distill_raw": loss_lrnode_action_distill,
                "train/lrnode/loss_action_distill_weighted": args.lrnode_action_distill_weight * loss_lrnode_action_distill,
                "train/lrnode/loss_bc_raw": loss_lrnode_bc,
                "train/lrnode/loss_bc_weighted": args.lrnode_bc_weight * loss_lrnode_bc,
                "train/lrnode/loss_smooth_raw": loss_lrnode_smooth,
                "train/lrnode/loss_smooth_weighted": args.lrnode_smooth_weight * loss_lrnode_smooth,
                "train/lrnode/target_mode_shifted_context": torch.tensor(
                    float(train_lrnode and lrnode_teacher_target_mode == "shifted_context"),
                    device=device_id,
                ),
                "train/lrnode/target_mode_adjacent_sequence": torch.tensor(
                    float(train_lrnode and lrnode_teacher_target_mode == "adjacent_sequence"),
                    device=device_id,
                ),
            }
        )

        if train_lrnode and z_pred_next is not None:
            eps = 1e-8
            z_prev_det = z_prev.detach().float()
            z_teacher_det = z_teacher_next.detach().float()
            z_pred_float = z_pred_next.float()
            latent_mse_hold = F.mse_loss(z_prev_det, z_teacher_det)
            latent_mse_pred = F.mse_loss(z_pred_float, z_teacher_det)
            train_log_metrics.update(
                {
                    "train/lrnode/latent_mse_hold": latent_mse_hold,
                    "train/lrnode/latent_mse_pred": latent_mse_pred,
                    "train/lrnode/latent_mse_improvement": latent_mse_hold - latent_mse_pred,
                    "train/lrnode/latent_mse_ratio": latent_mse_pred / (latent_mse_hold + eps),
                    "train/lrnode/z_delta_pred_norm": (z_pred_float - z_prev_det).norm(dim=-1).mean(),
                    "train/lrnode/z_delta_teacher_norm": (z_teacher_det - z_prev_det).norm(dim=-1).mean(),
                    "train/lrnode/cos_z_pred_teacher": _cosine_mean(z_pred_float, z_teacher_det),
                    "train/lrnode/cos_z_prev_teacher": _cosine_mean(z_prev_det, z_teacher_det),
                    "train/lrnode/cos_z_pred_prev": _cosine_mean(z_pred_float, z_prev_det),
                }
            )
            train_log_metrics.update(_tensor_stats("train/lrnode/z_prev", z_prev_det))
            train_log_metrics.update(_tensor_stats("train/lrnode/z_teacher", z_teacher_det))
            train_log_metrics.update(_tensor_stats("train/lrnode/z_pred", z_pred_float))

            if lrnode_action is not None and lrnode_teacher_action is not None and lrnode_hold_action is not None:
                action_pred_det = lrnode_action.detach().float()
                action_teacher_det = lrnode_teacher_action.detach().float()
                action_hold_det = lrnode_hold_action.detach().float()
                action_l1_pred = F.l1_loss(action_pred_det, action_teacher_det)
                action_l1_hold = F.l1_loss(action_hold_det, action_teacher_det)
                train_log_metrics.update(
                    {
                        "train/lrnode/action_l1_hold": action_l1_hold,
                        "train/lrnode/action_l1_pred": action_l1_pred,
                        "train/lrnode/action_l1_improvement": action_l1_hold - action_l1_pred,
                        "train/lrnode/action_l1_ratio": action_l1_pred / (action_l1_hold + eps),
                        "train/lrnode/arm_l1_pred": F.l1_loss(action_pred_det[..., :6], action_teacher_det[..., :6]),
                        "train/lrnode/arm_l1_hold": F.l1_loss(action_hold_det[..., :6], action_teacher_det[..., :6]),
                        "train/lrnode/gripper_l1_pred": F.l1_loss(action_pred_det[..., 6:], action_teacher_det[..., 6:]),
                        "train/lrnode/gripper_l1_hold": F.l1_loss(action_hold_det[..., 6:], action_teacher_det[..., 6:]),
                        "train/lrnode/trans_l1_pred": F.l1_loss(action_pred_det[..., :3], action_teacher_det[..., :3]),
                        "train/lrnode/rot_l1_pred": F.l1_loss(action_pred_det[..., 3:6], action_teacher_det[..., 3:6]),
                        "train/lrnode/trans_l1_hold": F.l1_loss(action_hold_det[..., :3], action_teacher_det[..., :3]),
                        "train/lrnode/rot_l1_hold": F.l1_loss(action_hold_det[..., 3:6], action_teacher_det[..., 3:6]),
                    }
                )
                for dim_idx in range(min(6, action_pred_det.shape[-1])):
                    train_log_metrics[f"train/lrnode/arm_dim{dim_idx}_l1_pred"] = F.l1_loss(
                        action_pred_det[..., dim_idx], action_teacher_det[..., dim_idx]
                    )
                train_log_metrics.update(_tensor_stats("train/lrnode/action_teacher", action_teacher_det))
                train_log_metrics.update(_tensor_stats("train/lrnode/action_pred", action_pred_det))
                train_log_metrics.update(_tensor_stats("train/lrnode/action_hold", action_hold_det))

            if z_pred_next.dim() == 4:
                latent_sq = (z_pred_float - z_teacher_det).pow(2)
                latent_hold_sq = (z_prev_det - z_teacher_det).pow(2)
                for local_t in range(min(latent_sq.shape[1], 8)):
                    train_log_metrics[f"train/lrnode/by_local_t/latent_mse_pred_{local_t}_to_{local_t+1}"] = latent_sq[:, local_t].mean()
                    train_log_metrics[f"train/lrnode/by_local_t/latent_mse_hold_{local_t}_to_{local_t+1}"] = latent_hold_sq[:, local_t].mean()
                for token_idx in range(min(latent_sq.shape[-2], args.action_pred_steps)):
                    train_log_metrics[f"train/lrnode/by_token/latent_mse_pred_token{token_idx}"] = latent_sq[..., token_idx, :].mean()
                    train_log_metrics[f"train/lrnode/by_token/latent_mse_hold_token{token_idx}"] = latent_hold_sq[..., token_idx, :].mean()
                if lrnode_action is not None and lrnode_teacher_action is not None and lrnode_hold_action is not None:
                    action_abs = (lrnode_action.detach().float() - lrnode_teacher_action.detach().float()).abs()
                    action_hold_abs = (lrnode_hold_action.detach().float() - lrnode_teacher_action.detach().float()).abs()
                    for local_t in range(min(action_abs.shape[1], 8)):
                        train_log_metrics[f"train/lrnode/by_local_t/action_l1_pred_{local_t}_to_{local_t+1}"] = action_abs[:, local_t].mean()
                        train_log_metrics[f"train/lrnode/by_local_t/action_l1_hold_{local_t}_to_{local_t+1}"] = action_hold_abs[:, local_t].mean()
                    for token_idx in range(min(action_abs.shape[-2], args.action_pred_steps)):
                        train_log_metrics[f"train/lrnode/by_token/action_l1_pred_token{token_idx}"] = action_abs[..., token_idx, :].mean()
                        train_log_metrics[f"train/lrnode/by_token/action_l1_hold_token{token_idx}"] = action_hold_abs[..., token_idx, :].mean()

            if lrnode_teacher_target_mode == "shifted_context":
                diff_step = int(getattr(args, "lrnode_context_selected_step", -1))
                if diff_step < 0:
                    diff_step = args.sequence_length + diff_step
                key_primary_for_debug = images_primary[:, diff_step:diff_step + 1]
                cur_primary_for_debug = images_primary[:, diff_step + 1:diff_step + 2]
                key_wrist_for_debug = images_wrist[:, diff_step:diff_step + 1]
                cur_wrist_for_debug = images_wrist[:, diff_step + 1:diff_step + 2]
                imgdiff_primary = (cur_primary_for_debug - key_primary_for_debug).detach().float()
                imgdiff_wrist = (cur_wrist_for_debug - key_wrist_for_debug).detach().float()
            else:
                key_primary_for_debug = input_image_primary[:, :-1]
                cur_primary_for_debug = input_image_primary[:, 1:]
                key_wrist_for_debug = input_image_wrist[:, :-1]
                cur_wrist_for_debug = input_image_wrist[:, 1:]
                imgdiff_primary = (cur_primary_for_debug - key_primary_for_debug).detach().float()
                imgdiff_wrist = (cur_wrist_for_debug - key_wrist_for_debug).detach().float()
            train_log_metrics.update(
                {
                    "train/lrnode/imgdiff_primary_l1": imgdiff_primary.abs().mean(),
                    "train/lrnode/imgdiff_primary_l2": imgdiff_primary.pow(2).mean().sqrt(),
                    "train/lrnode/imgdiff_primary_max": imgdiff_primary.abs().max(),
                    "train/lrnode/imgdiff_wrist_l1": imgdiff_wrist.abs().mean(),
                    "train/lrnode/imgdiff_wrist_l2": imgdiff_wrist.pow(2).mean().sqrt(),
                    "train/lrnode/imgdiff_wrist_max": imgdiff_wrist.abs().max(),
                }
            )

            if lrnode_u_delta is not None:
                u_det = lrnode_u_delta.detach().float()
                train_log_metrics.update(_tensor_stats("train/lrnode/u_delta", u_det))
                camera_features = getattr(base_model.lrnode_delta_encoder, "last_camera_features", None)
                if camera_features:
                    if len(camera_features) > 0:
                        train_log_metrics["train/lrnode/u_delta_primary_norm"] = camera_features[0].detach().float().norm(dim=-1).mean()
                    if len(camera_features) > 1:
                        train_log_metrics["train/lrnode/u_delta_wrist_norm"] = camera_features[1].detach().float().norm(dim=-1).mean()

            if lrnode_gate is not None:
                gate_det = lrnode_gate.detach().float()
                train_log_metrics.update(
                    {
                        "train/lrnode/gate_mean": gate_det.mean(),
                        "train/lrnode/gate_std": gate_det.std(unbiased=False),
                        "train/lrnode/gate_min": gate_det.min(),
                        "train/lrnode/gate_max": gate_det.max(),
                        "train/lrnode/gate_p10": torch.quantile(gate_det.reshape(-1), 0.10),
                        "train/lrnode/gate_p50": torch.quantile(gate_det.reshape(-1), 0.50),
                        "train/lrnode/gate_p90": torch.quantile(gate_det.reshape(-1), 0.90),
                    }
                )
            if lrnode_dz is not None:
                dz_det = lrnode_dz.detach().float()
                train_log_metrics["train/lrnode/dzdt_norm"] = dz_det.norm(dim=-1).mean()
            if lrnode_update is not None:
                update_det = lrnode_update.detach().float()
                train_log_metrics["train/lrnode/update_norm"] = update_det.norm(dim=-1).mean()
                train_log_metrics["train/lrnode/update_to_latent_norm_ratio"] = (
                    update_det.norm(dim=-1).mean() / (z_prev_det.norm(dim=-1).mean() + eps)
                )
                if lrnode_gate is not None and imgdiff_primary.dim() >= 3:
                    img_pair = imgdiff_primary.flatten(2).abs().mean(dim=-1)
                    update_pair = update_det.norm(dim=-1)
                    while update_pair.dim() > img_pair.dim():
                        update_pair = update_pair.mean(dim=-1)
                    gate_pair = lrnode_gate.detach().float()
                    while gate_pair.dim() > img_pair.dim():
                        gate_pair = gate_pair.mean(dim=-1)
                    train_log_metrics["train/lrnode/corr_imgdiff_gate"] = _corrcoef(img_pair, gate_pair)
                    train_log_metrics["train/lrnode/corr_imgdiff_update_norm"] = _corrcoef(img_pair, update_pair)
                    if lrnode_u_delta is not None:
                        u_pair = lrnode_u_delta.detach().float().norm(dim=-1)
                        train_log_metrics["train/lrnode/corr_u_norm_update_norm"] = _corrcoef(u_pair, update_pair)

            _save_lrnode_debug_artifacts(
                args,
                global_step,
                {
                    "z_prev_sample": z_prev_det,
                    "z_teacher_sample": z_teacher_det,
                    "z_pred_sample": z_pred_float,
                    "action_hold_sample": lrnode_hold_action.detach().float() if lrnode_hold_action is not None else None,
                    "action_teacher_sample": lrnode_teacher_action.detach().float() if lrnode_teacher_action is not None else None,
                    "action_pred_sample": lrnode_action.detach().float() if lrnode_action is not None else None,
                    "gt_action_sample": label_actions.detach().float(),
                    "gate_sample": lrnode_gate.detach().float() if lrnode_gate is not None else None,
                    "u_delta_sample": lrnode_u_delta.detach().float() if lrnode_u_delta is not None else None,
                    "image_diff_norm_sample": imgdiff_primary.flatten(2).abs().mean(dim=-1),
                    "key_primary": key_primary_for_debug,
                    "cur_primary": cur_primary_for_debug,
                    "diff_primary": imgdiff_primary.abs(),
                    "key_wrist": key_wrist_for_debug,
                    "cur_wrist": cur_wrist_for_debug,
                    "diff_wrist": imgdiff_wrist.abs(),
                },
            )

        # gradient_accumulation_steps        
        loss = loss_calvin / args.gradient_accumulation_steps
        loss_arm_action = loss_arm_action / args.gradient_accumulation_steps
        loss_gripper_action = loss_gripper_action / args.gradient_accumulation_steps
        loss_image = loss_image / args.gradient_accumulation_steps
        loss_lrnode_latent = loss_lrnode_latent / args.gradient_accumulation_steps
        loss_lrnode_action_distill = loss_lrnode_action_distill / args.gradient_accumulation_steps
        loss_lrnode_smooth = loss_lrnode_smooth / args.gradient_accumulation_steps
        loss_lrnode_bc = loss_lrnode_bc / args.gradient_accumulation_steps
        mv_avg_loss.append(loss.item())

        ### backward pass ###
        loss.backward()
        lrnode_fast_encoder_grad_norm = 0.0
        lrnode_controlled_node_grad_norm = 0.0
        action_head_grad_norm_total = 0.0
        seer_backbone_grad_norm_total = 0.0
        if train_lrnode and args.lrnode_log_sanity:
            lrnode_fast_encoder_grad_norm = module_grad_norm(base_model.lrnode_delta_encoder)
            lrnode_controlled_node_grad_norm = module_grad_norm(base_model.lrnode_dynamics)
            action_head_grad_norm_total = (
                module_grad_norm(base_model.action_decoder)
                + module_grad_norm(base_model.arm_action_decoder)
                + module_grad_norm(base_model.gripper_action_decoder)
            )
            seer_backbone_grad_norm_total = module_grad_norm(base_model.transformer_backbone)
            train_log_metrics.update(
                {
                    "train/grad/fast_delta_encoder": lrnode_fast_encoder_grad_norm,
                    "train/grad/controlled_node": lrnode_controlled_node_grad_norm,
                    "train/grad/seer_backbone": seer_backbone_grad_norm_total,
                    "train/grad/seer_action_head": action_head_grad_norm_total,
                    "train/grad/action_decoder": module_grad_norm(base_model.action_decoder),
                }
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        reduced_train_log_metrics = (
            _gather_mean_scalar_dict(train_log_metrics)
            if args.report_to_wandb
            else {}
        )

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (
            num_steps == num_batches_per_epoch - 1
        ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                calvin_samples_per_second = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    * args.world_size
                    / step_time_m.val
                )
                calvin_samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps
                    * args.batch_size
                    / step_time_m.val
                )

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "calvin_samples_per_second": calvin_samples_per_second,
                        "calvin_samples_per_second_per_gpu": calvin_samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                )
                step_time_m.reset()
                data_time_m.reset()

                wandb.log(
                    {
                        "loss_calvin": loss.item() * args.gradient_accumulation_steps,
                        "loss_arm_action": loss_arm_action.item() * args.gradient_accumulation_steps,
                        "loss_gripper_action": loss_gripper_action.item() * args.gradient_accumulation_steps,
                        "loss_image": loss_image.item() * args.gradient_accumulation_steps,
                        "loss_lrnode_latent": loss_lrnode_latent.item() * args.gradient_accumulation_steps,
                        "loss_lrnode_action_distill": loss_lrnode_action_distill.item() * args.gradient_accumulation_steps,
                        "loss_lrnode_smooth": loss_lrnode_smooth.item() * args.gradient_accumulation_steps,
                        "loss_lrnode_bc": loss_lrnode_bc.item() * args.gradient_accumulation_steps,
                        "global_step": global_step,
                    },
                )
                if reduced_train_log_metrics:
                    reduced_train_log_metrics["global_step"] = global_step
                    wandb.log(reduced_train_log_metrics)
                if train_lrnode and args.lrnode_log_sanity:
                    wandb.log(
                        {
                            "lrnode_loss_hold_latent": loss_lrnode_hold_latent.item(),
                            "lrnode_loss_pred_latent": loss_lrnode_latent.item() * args.gradient_accumulation_steps,
                            "lrnode_loss_hold_action": loss_lrnode_hold_action.item(),
                            "lrnode_loss_pred_action": loss_lrnode_action_distill.item() * args.gradient_accumulation_steps,
                            "lrnode_z_prev_mean": lrnode_z_prev_mean.item(),
                            "lrnode_z_prev_std": lrnode_z_prev_std.item(),
                            "lrnode_z_teacher_mean": lrnode_z_teacher_mean.item(),
                            "lrnode_z_teacher_std": lrnode_z_teacher_std.item(),
                            "lrnode_z_pred_mean": lrnode_z_pred_mean.item(),
                            "lrnode_z_pred_std": lrnode_z_pred_std.item(),
                            "lrnode_gate_mean": lrnode_gate_mean.item(),
                            "lrnode_gate_std": lrnode_gate_std.item(),
                            "lrnode_gate_min": lrnode_gate_min.item(),
                            "lrnode_gate_max": lrnode_gate_max.item(),
                            "lrnode_fast_encoder_grad_norm": lrnode_fast_encoder_grad_norm,
                            "lrnode_controlled_node_grad_norm": lrnode_controlled_node_grad_norm,
                            "action_head_grad_norm_total": action_head_grad_norm_total,
                            "seer_backbone_grad_norm_total": seer_backbone_grad_norm_total,
                            "global_step": global_step,
                        },
                    )

        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "loss_image": loss_image.item(), "loss_arm_action": loss_arm_action.item(), "loss_gripper_action": loss_gripper_action.item(), "loss_lrnode_latent": loss_lrnode_latent.item(), "loss_lrnode_action_distill": loss_lrnode_action_distill.item()})

        # if args.save_every_iter != -1 and args.save_checkpoint and global_step % args.save_every_iter == 0 and global_step > 0:
                
        #     if args.rank == 0:
        #         import os
        #         if not os.path.exists(f"{args.save_checkpoint_path}/exp/{args.run_name}"):
        #             os.makedirs(f"{args.save_checkpoint_path}/exp/{args.run_name}")

        #         checkpoint_dict = {
        #             "epoch": epoch,
        #             "model_state_dict": get_checkpoint(model),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        #         }

        #         ckpt_name = get_ckpt_name(args, global_step)
        #         ckpt_path = os.path.join(f"{args.save_checkpoint_path}/exp", args.run_name, ckpt_name)
        #         print(f"Saving checkpoint to {ckpt_path}")
        #         torch.save(checkpoint_dict, ckpt_path)
        #         if args.delete_previous_checkpoint:
        #             if epoch > 0:
        #                 os.remove(ckpt_path)

def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict

def get_checkpoint_all_param(model):
    state_dict = model.state_dict()

    return state_dict

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
