"""
1. 3d point cloud 추가했음
"""
import time
from contextlib import suppress
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from einops import rearrange
from pdb import set_trace
import numpy as np
import torch.distributed as dist
# from utils.pcd_viz_utils import maybe_visualize_tcp_points


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
def compute_pixel_motion_mask(scale_tensor: torch.Tensor, threshold: float):
    if threshold <= 0:
        return torch.ones_like(scale_tensor)
    return (scale_tensor.abs() >= threshold).float()

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


def log_pcd_conf_stats_per_step(
    *,
    global_step: int,
    pcd_pts: Optional[torch.Tensor],
    pcd_conf: Optional[torch.Tensor],
    conf_thresh: float,
    seq_eff: int,
    action_pred_steps: int,
    batch_idx_for_log: int = 0,
) -> None:
    """Print per-step confidence statistics for threshold debugging."""
    if pcd_conf is None or pcd_conf.ndim != 4:
        print(f"[pcd_conf][step={global_step}] unavailable", flush=True)
        return

    B, frames, H, W = pcd_conf.shape
    total_points = H * W
    frames_need = min(frames, seq_eff + action_pred_steps)
    if frames_need <= 0 or total_points <= 0:
        print(f"[pcd_conf][step={global_step}] invalid_shape={tuple(pcd_conf.shape)}", flush=True)
        return

    conf = pcd_conf.reshape(B, frames, total_points).float()
    conf_use = conf[:, :frames_need]  # (B, F, P)
    conf_ok = conf_use > float(conf_thresh)
    b_idx = max(0, min(int(batch_idx_for_log), B - 1))
    conf_b = conf_use[b_idx]  # (F, P)
    conf_ok_b = conf_ok[b_idx]  # (F, P)
    qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=conf_b.device)
    qvals_f = torch.quantile(conf_b, qs, dim=-1)  # (5, F)
    q01_f = qvals_f[0].tolist()
    q05_f = qvals_f[1].tolist()
    q50_f = qvals_f[2].tolist()
    q95_f = qvals_f[3].tolist()
    q99_f = qvals_f[4].tolist()
    pass_f = conf_ok_b.float().mean(dim=-1).tolist()

    point_any_ratio = conf_ok.any(dim=1).float().mean().item()
    point_all_ratio = conf_ok.all(dim=1).float().mean().item()
    point_ratio60 = (conf_ok.float().mean(dim=1) >= 0.6).float().mean().item()

    def _fmt(vals):
        return "[" + ",".join(f"{float(v):.4f}" for v in vals) + "]"

    msg = (
        f"[pcd_conf][step={global_step}] shape=({B},{frames},{H},{W}) th={conf_thresh:.4f} "
        f"batch={b_idx} pass_f={_fmt(pass_f)} q01_f={_fmt(q01_f)} q05_f={_fmt(q05_f)} "
        f"q50_f={_fmt(q50_f)} q95_f={_fmt(q95_f)} q99_f={_fmt(q99_f)} "
        f"point_any={point_any_ratio:.4f} point_all={point_all_ratio:.4f} point_ratio60={point_ratio60:.4f}"
    )

    if pcd_pts is not None and pcd_pts.ndim == 5 and frames_need >= 2:
        pts = pcd_pts[..., :3].reshape(B, pcd_pts.shape[1], total_points, 3).float()
        pts_use = pts[:, :frames_need]
        motion = (pts_use[:, 1:] - pts_use[:, :-1]).norm(dim=-1)  # (B, F-1, P)
        score = motion.mean(dim=1)  # (B, P)
        score_mean = score.mean().item()
        score_p95 = torch.quantile(score.reshape(-1), 0.95).item()
        valid_any = conf_ok.any(dim=1)
        valid_all = conf_ok.all(dim=1)
        if valid_any.any():
            score_any = score[valid_any].mean().item()
        else:
            score_any = 0.0
        if valid_all.any():
            score_all = score[valid_all].mean().item()
        else:
            score_all = 0.0
        msg += (
            f" motion_mean={score_mean:.5f} motion_p95={score_p95:.5f} "
            f"motion_any={score_any:.5f} motion_all={score_all:.5f}"
        )

    print(msg, flush=True)


def build_sparse_pcd_targets_motion_topk(
    pcd_pts: Optional[torch.Tensor],
    pcd_conf: Optional[torch.Tensor] = None,
    *,
    seq_eff: int,
    action_pred_steps: int,
    num_sparse_points: int,
    conf_thresh: float = 0.1,
    motion_thresh: float = 0.0,
    conf_quantile: float = 0.5,
    motion_quantile: float = 0.8,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build sparse point inputs/targets by motion-heavy top-k sampling.

    Returns:
        pixel_points_for_model: (B, seq_eff, T, P, 3) or None
        pixel_points_mask: (B, seq_eff, T, P) bool or None
        pcd_delta_gt: (B, seq_eff, T, P, 3) or None
    """
    if pcd_pts is None or pcd_pts.ndim != 5:
        return None, None, None
    if pcd_pts.shape[1] < seq_eff + action_pred_steps:
        return None, None, None

    B, frames, H, W, _ = pcd_pts.shape
    # print(frames, 'frames') #10
    total_points = H * W
    if total_points <= 0:
        return None, None, None

    if num_sparse_points is None or num_sparse_points <= 0:
        sampled_points = total_points
    else:
        sampled_points = min(int(num_sparse_points), total_points)
    # sampled_points = 512
    
    pts = pcd_pts[..., :3].reshape(B, frames, total_points, 3)
    frames_need = seq_eff + action_pred_steps
    pts_use = pts[:, :frames_need]
    delta01 = pts_use[:, 1:] - pts_use[:, :-1] #방향 + 크기 포함
    motion = delta01.norm(dim=-1) #크기만 포함하기 위해 norm 계산
    score = motion.mean(dim=1)

    # Hybrid filter: keep points that satisfy either confidence or motion quantile.
    # This rescues low-confidence but truly moving points.
    if pcd_conf is not None and pcd_conf.ndim == 4:
        conf = pcd_conf.reshape(B, frames, total_points)
        conf_use = conf[:, :frames_need]  # (B, F, P)
        conf_score = conf_use.mean(dim=1)  # (B, P)

        conf_q = max(0.0, min(1.0, float(conf_quantile)))
        motion_q = max(0.0, min(1.0, float(motion_quantile)))
        conf_thr_q = torch.quantile(conf_score.float(), conf_q, dim=-1, keepdim=True)   # (B, 1)
        motion_thr_q = torch.quantile(score.float(), motion_q, dim=-1, keepdim=True)     # (B, 1)

        conf_valid = (conf_score >= conf_thr_q) & (conf_score > float(conf_thresh))
        motion_valid = score >= motion_thr_q
        # valid = conf_valid | motion_valid
        valid = motion_valid
        
        score_masked = score.masked_fill(~valid, float("-inf"))
        no_valid = ~torch.isfinite(score_masked).any(dim=-1, keepdim=True)
        score = torch.where(no_valid, score, score_masked)


    topk_idx = score.topk(sampled_points, dim=-1).indices  
    gather_idx = topk_idx[:, None, :, None].expand(B, frames_need, sampled_points, 3)
    pts_sel = pts_use.gather(dim=2, index=gather_idx)

    start_slices = []
    delta_slices = []
    for j in range(action_pred_steps):
        start = pts_sel[:, j : seq_eff + j]
        end = pts_sel[:, j + 1 : seq_eff + j + 1]
        start_slices.append(start)
        delta_slices.append(end - start)

    pixel_points_for_model = torch.stack(start_slices, dim=2)
    pcd_delta_gt = torch.stack(delta_slices, dim=2)

    if motion_thresh > 0:
        delta_mag = pcd_delta_gt.norm(dim=-1)
        pixel_points_mask = delta_mag > motion_thresh
    else:
        pixel_points_mask = torch.ones(
            (B, seq_eff, action_pred_steps, sampled_points),
            device=pcd_pts.device,
            dtype=torch.bool,
        )
    
    return pixel_points_for_model, pixel_points_mask, pcd_delta_gt

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
    print(args.tcp_loss, args.scale_loss, args.direc_loss, args.tcp_mask, getattr(args, "pcd_conf_thresh", 0.1))
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
        pixel_motion = None
            
        pcd = None
        if len(batch_calvin) > 7 and batch_calvin[7] is not None:
            pcd = {
                "pts3d": batch_calvin[7]["pts3d"].to(device_id, dtype=cast_dtype, non_blocking=True),
                "conf": batch_calvin[7]["conf"].to(device_id, dtype=cast_dtype, non_blocking=True),
            }
        # print(pcd["pts3d"].shape) #16, 10, 224, 224, 3
        
        # print(pixel_motion["points"].shape, '123'*30) #torch.Size([16, 10, 256, 2, 2])
        
        input_image_primary = images_primary[:, :args.sequence_length, :]
        input_image_wrist = images_wrist[:, :args.sequence_length, :]
        input_text_token = text_tokens[:, :args.sequence_length, :]
        input_state = input_states[:, :args.sequence_length, :]
        
        #todo: 3d point cloud 
        #TODO: TCP -POSE ground truth
        valid_pcd = torch.tensor(1.0, device=device_id)
        seq_eff = args.sequence_length - args.atten_goal
        pcd_pts = pcd["pts3d"] if pcd is not None else None
        pcd_conf = pcd["conf"] if pcd is not None else None
        sparse_points = getattr(args, "pcd_sparse_points", 4096)
        pcd_conf_thresh = getattr(args, "pcd_conf_thresh", 0.1)
        pcd_motion_thresh = getattr(args, "pcd_motion_thresh", 0.0)
        pcd_conf_quantile = getattr(args, "pcd_conf_quantile", 0.05)
        pcd_motion_quantile = getattr(args, "pcd_motion_quantile", 0.05)
        # if args.rank == 0:
        #     log_pcd_conf_stats_per_step(
        #         global_step=global_step,
        #         pcd_pts=pcd_pts,
        #         pcd_conf=pcd_conf,
        #         conf_thresh=pcd_conf_thresh,
        #         seq_eff=seq_eff,
        #         action_pred_steps=args.action_pred_steps,
        #         batch_idx_for_log=getattr(args, "pcd_conf_log_batch_idx", 0),
        #     )
        pixel_points_for_model, pixel_points_mask, pcd_delta_gt = build_sparse_pcd_targets_motion_topk(
            pcd_pts,
            pcd_conf,
            seq_eff=seq_eff,
            action_pred_steps=args.action_pred_steps,
            num_sparse_points=sparse_points,
            conf_thresh=pcd_conf_thresh,
            motion_thresh=pcd_motion_thresh,
            conf_quantile=pcd_conf_quantile,
            motion_quantile=pcd_motion_quantile,
        )
        
        
        if states.shape[1] >= seq_eff + args.action_pred_steps:
            tcp_deltas = []
            for j in range(args.action_pred_steps):
                start_pose = states[:, j:seq_eff + j, :6]
                end_pose = states[:, j + 1:seq_eff + j + 1, :6]
                tcp_deltas.append(end_pose[..., :3] - start_pose[..., :3])
            tcp_delta_gt = torch.stack(tcp_deltas, dim=2)
        else:
            tcp_delta_gt = None   
        
        # label action
        label_actions = torch.cat([actions[:, j:args.sequence_length-args.atten_goal+j, :].unsqueeze(-2) for j in range(args.action_pred_steps)], dim=-2) 
        
        
        with autocast():  # image_primary, image_wrist, state, language_instruction
            (
                arm_pred_action,
                gripper_pred_action,
                image_pred,
                arm_pred_state,
                gripper_pred_state,
                loss_arm_action,
                # flow_pred_scale,
                # flow_pred_direction,
                # flow_pred_3d,
                # flow_robot_logits,
                pcd_pred,
                flow_pred_tcp,
                flow_points_weights,
                # points_feat
            ) = model(
                image_primary=input_image_primary,
                image_wrist=input_image_wrist,
                state=input_state,
                text_token=input_text_token,
                action=actions[:, :args.sequence_length, :],
                #TODO: Pixel motion 추가
                pixel_points=pixel_points_for_model,
                pixel_points_mask=pixel_points_mask,
            )
            
        # if args.rank == 0:
        #     maybe_visualize_tcp_points(
        #         args,
        #         global_step=global_step,
        #         pixel_points=pixel_points_for_model,
        #         pixel_points_mask=pixel_points_mask,
        #         point_weights=flow_points_weights,
        #         flow_pred_tcp=flow_pred_tcp,
        #         tcp_delta_gt=tcp_delta_gt,
        #         pcd_delta_gt=pcd_delta_gt,
        #         pcd_pts=pcd_pts,
        #         pcd_conf=pcd_conf,
        #         image_primary=images_primary,
        #         conf_thresh=pcd_conf_thresh,
        #     )
            
            
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

        #todo : loss_flow        
        # loss_flow_scale = torch.tensor(0.0, device=device_id)
        # loss_flow_direction = torch.tensor(0.0, device=device_id)
        loss_tcp_flow = torch.tensor(0.0, device=device_id)
        loss_pcd = torch.tensor(0.0, device=device_id)
        loss_alpha_out = torch.tensor(0.0, device=device_id)
        i = 0
        # if flow_pred_scale is not None and flow_pred_direction is not None and label_scale is not None and motion_mask is not None:
        #     scale_factor = 128 / args.calvin_input_image_size #128이 85.25 임. 수정할것
        #     pred_scale = flow_pred_scale[:, :args.sequence_length-args.atten_goal]
        #     target_scale = label_scale[:, :args.sequence_length-args.atten_goal].detach()
        #     scale_l1 = F.smooth_l1_loss(
        #         pred_scale * scale_factor,
        #         target_scale * scale_factor,
        #         reduction="none",
        #     )

        #     loss_flow_scale = (scale_l1 * motion_mask[:, :args.sequence_length-args.atten_goal]).sum() / valid_motion
        #     pred_dir = F.normalize(flow_pred_direction[:, :args.sequence_length-args.atten_goal], dim=-1)
        #     target_dir = F.normalize(label_direction[:, :args.sequence_length-args.atten_goal].detach(), dim=-1)
        #     cos_sim = (pred_dir * target_dir).sum(dim=-1)
        #     loss_flow_direction = ((1 - cos_sim) * motion_mask[:, :args.sequence_length-args.atten_goal]).sum() / valid_motion
            #TODO: TCP supervision
        if tcp_delta_gt is not None and flow_pred_tcp is not None:
            target_tcp = tcp_delta_gt[:, :seq_eff].detach()
            loss_tcp_flow = F.smooth_l1_loss(flow_pred_tcp[:, :seq_eff], target_tcp, reduction="mean")
            # if flow_points_weights is not None:
            #     alpha = flow_points_weights[:, :seq_eff]               # (B,seq,T,P,1)
            #     m = motion_mask[:, :args.sequence_length-args.atten_goal].float()
            #     # print(m.mean(),'what is mean?')
            #     loss_alpha_out = (alpha * (1.0 - m)).sum(dim=3).mean()  # sum over P -> (B,seq,T,1) -> mean

        if pcd_delta_gt is not None and pcd_pred is not None:
            target_pcd = pcd_delta_gt[:, :seq_eff].detach()
            if pixel_points_mask is not None:
                pcd_error = F.smooth_l1_loss(pcd_pred[:, :seq_eff], target_pcd, reduction="none").mean(dim=-1)
                mask = pixel_points_mask[:, :seq_eff].float()
                valid_count = mask.sum().clamp(min=1.0)
                loss_pcd = (pcd_error * mask).sum() / valid_count
            else:
                loss_pcd = F.smooth_l1_loss(pcd_pred[:, :seq_eff], target_pcd, reduction="mean")
            
            
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
        loss_calvin = (
            args.loss_arm_action_ratio * loss_arm_action
            + args.loss_gripper_action_ratio * loss_gripper_action
            # + loss_flow_scale * args.scale_loss
            # + loss_flow_direction * args.direc_loss
            + loss_pcd * args.tcp_loss
            + loss_tcp_flow * args.tcp_loss
            # + loss_alpha_out * args.tcp_mask #0.0001
            + 0.1 * loss_image
        )
        
        # print(args.loss_arm_action_ratio, valid_motion)
        
        # gradient_accumulation_steps        
        loss = loss_calvin / args.gradient_accumulation_steps
        loss_arm_action = loss_arm_action / args.gradient_accumulation_steps
        loss_gripper_action = loss_gripper_action / args.gradient_accumulation_steps
        # loss_flow_scale = loss_flow_scale / args.gradient_accumulation_steps
        # loss_flow_direction = loss_flow_direction / args.gradient_accumulation_steps
        loss_tcp_flow = loss_tcp_flow / args.gradient_accumulation_steps
        loss_pcd = loss_pcd / args.gradient_accumulation_steps
        loss_image = loss_image / args.gradient_accumulation_steps
        mv_avg_loss.append(loss.item())

        ### backward pass ###
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

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
                        # "loss_flow_scale": loss_flow_scale.item() * args.gradient_accumulation_steps,
                        # "loss_flow_direction": loss_flow_direction.item() * args.gradient_accumulation_steps,
                        "loss_pcd": loss_pcd.item() * args.gradient_accumulation_steps,
                        "loss_tcp_flow": loss_tcp_flow.item() * args.gradient_accumulation_steps,
                        "loss_image": loss_image.item() * args.gradient_accumulation_steps,
                        "global_step": global_step,
                    },
                )

        avg_horizon = min(100, len(mv_avg_loss))
        t.set_postfix({"avg loss": sum(mv_avg_loss[-avg_horizon:]) / avg_horizon, "loss": loss_calvin.item(), "loss_tcp": loss_tcp_flow.item(), "loss_pcd": loss_pcd.item(), "loss_image": loss_image.item(), "loss_arm_action": loss_arm_action.item(), "loss_gripper_action": loss_gripper_action.item()})

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
        
