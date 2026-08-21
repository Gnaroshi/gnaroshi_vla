import os
import json
from pathlib import Path
import random

import clip
import numpy as np
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP

from models.seer_model import SeerAgent
from utils.arguments_utils import get_parser
from utils.distributed_utils import init_distributed_device, world_info_from_env
from utils.eval_utils_libero import eval_one_epoch_libero_ddp
from utils.lrnode_logging_utils import save_lrnode_run_snapshots


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def _save_eval_args_snapshot(args):
    if getattr(args, "rank", 0) != 0:
        return
    log_dir = os.environ.get("LOG_DIR")
    if log_dir:
        out_dir = os.path.join(log_dir, "analysis")
    else:
        ckpt = getattr(args, "resume_from_checkpoint", "")
        ckpt_dir = os.path.dirname(ckpt) if ckpt else os.path.join(os.getcwd(), "eval_analysis", args.run_name)
        out_dir = os.path.join(ckpt_dir, "analysis")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    payload = {k: v for k, v in vars(args).items()}
    out_path = os.path.join(out_dir, "args_snapshot.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    ckpt_tag = os.environ.get("CKPT_TAG", "")
    if ckpt_tag:
        tagged_out_path = os.path.join(out_dir, f"args_snapshot_{ckpt_tag}.json")
        with open(tagged_out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)


def _checkpoint_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict': {checkpoint_path}")
    return checkpoint, checkpoint["model_state_dict"]


def _is_lrnode_adapter_only_state_dict(state_dict):
    keys = list(state_dict.keys())
    if not keys:
        return False
    has_lrnode = any("lrnode_" in key or ".lrnode" in key for key in keys)
    has_core_seer = any(
        marker in key
        for key in keys
        for marker in (
            "transformer_backbone",
            "action_decoder",
            "action_pred_token",
            "perceiver_resampler",
            "image_primary_projector",
            "image_wrist_projector",
        )
    )
    return has_lrnode and not has_core_seer


def _is_lrnode_state_key(key):
    return key.startswith("module.lrnode_delta_encoder.") or key.startswith("module.lrnode_dynamics.")


def _expected_checkpoint_keys(ddp_model, checkpoint_kind):
    if checkpoint_kind == "adapter":
        return {
            name for name, _ in ddp_model.named_parameters()
            if _is_lrnode_state_key(name)
        }
    if checkpoint_kind == "base":
        return {
            name for name, param in ddp_model.named_parameters()
            if param.requires_grad and not _is_lrnode_state_key(name)
        }
    raise ValueError(f"Unknown checkpoint_kind={checkpoint_kind}")


def _assert_checkpoint_contract(ddp_model, state_dict, checkpoint_path, checkpoint_kind):
    expected_keys = _expected_checkpoint_keys(ddp_model, checkpoint_kind)
    actual_keys = set(state_dict.keys())
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint contract failed for kind={checkpoint_kind}: {checkpoint_path}; "
            f"missing={missing[:50]}, unexpected={unexpected[:50]}"
        )
    return len(expected_keys)


def _load_checkpoint_into_model(ddp_model, checkpoint_path, label, rank, checkpoint_kind=None):
    checkpoint, state_dict = _checkpoint_state_dict(checkpoint_path)
    verified_key_count = None
    if checkpoint_kind is not None:
        verified_key_count = _assert_checkpoint_contract(
            ddp_model,
            state_dict,
            checkpoint_path,
            checkpoint_kind,
        )
    ret = ddp_model.load_state_dict(state_dict, False)
    if ret.unexpected_keys:
        raise RuntimeError(
            f"Unexpected checkpoint keys while loading {checkpoint_path}: "
            f"{ret.unexpected_keys[:50]}"
        )
    if rank == 0:
        print(f"[CKPT LOAD:{label}] path={checkpoint_path}")
        print(f"[CKPT LOAD:{label}] epoch={checkpoint.get('epoch', 'NA')}")
        print(f"[CKPT LOAD:{label}] state_dict_keys={len(state_dict)}")
        print(f"[CKPT LOAD:{label}] adapter_only={_is_lrnode_adapter_only_state_dict(state_dict)}")
        if checkpoint_kind is not None:
            print(
                f"[CKPT VERIFY:{label}] contract={checkpoint_kind} "
                f"exact_keys=True verified_key_count={verified_key_count}"
            )
        try:
            print(f"[CKPT LOAD:{label}] missing_keys={len(ret.missing_keys)}")
            if len(ret.missing_keys) > 0:
                print(ret.missing_keys[:50])
            print(f"[CKPT LOAD:{label}] unexpected_keys={len(ret.unexpected_keys)}")
            if len(ret.unexpected_keys) > 0:
                print(ret.unexpected_keys[:50])
        except Exception:
            pass
    return ret, state_dict


@record
def main():
    parser = get_parser(is_eval=True)
    args = parser.parse_args()
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)
    if args.rank == 0:
        print(f"[EVAL ARGS] use_lrnode_latent_update={bool(args.use_lrnode_latent_update)}")
        print(f"[EVAL ARGS] lrnode_eval_skip_full_forward={bool(args.lrnode_eval_skip_full_forward)}")
        print(f"[EVAL ARGS] lrnode_query_interval={args.lrnode_query_interval}")
        print(f"[EVAL ARGS] lrnode_eval_ablation_mode={args.lrnode_eval_ablation_mode}")
        print(f"[EVAL ARGS] lrnode_no_delta_mode={args.lrnode_no_delta_mode}")
        print(f"[EVAL ARGS] lrnode_chunk_token_policy={args.lrnode_chunk_token_policy}")
        print(f"[EVAL ARGS] lrnode_eval_refresh_policy={args.lrnode_eval_refresh_policy}")
        print(
            "[EVAL ARGS] "
            f"lrnode_eval_max_full_forwards_per_episode={args.lrnode_eval_max_full_forwards_per_episode}"
        )
        print(
            "[EVAL ARGS] "
            f"lrnode_eval_profile_full_action_head={bool(args.lrnode_eval_profile_full_action_head)}"
        )
        print(f"[EVAL ARGS] lrnode_hidden_dim={args.lrnode_hidden_dim}")
        print(f"[EVAL ARGS] lrnode_motion_dim={args.lrnode_motion_dim}")
        print(f"[EVAL ARGS] lrnode_fast_encoder_type={args.lrnode_fast_encoder_type}")
        print(f"[EVAL ARGS] lrnode_detach_input_latent={bool(args.lrnode_detach_input_latent)}")
        print(f"[EVAL ARGS] lrnode_freeze_action_head_for_lrnode={bool(args.lrnode_freeze_action_head_for_lrnode)}")
        print(f"[EVAL ARGS] lrnode_use_post_layernorm={bool(args.lrnode_use_post_layernorm)}")
        print(f"[EVAL ARGS] lrnode_multistep_train={bool(args.lrnode_multistep_train)}")
        print(f"[EVAL ARGS] lrnode_train_max_horizon={args.lrnode_train_max_horizon}")
        print(f"[EVAL ARGS] lrnode_gate_init_bias={args.lrnode_gate_init_bias}")
    _save_eval_args_snapshot(args)
    random_seed(args.seed)

    model = SeerAgent(
        finetune_type=args.finetune_type,
        clip_device=device_id,
        vit_checkpoint_path=args.vit_checkpoint_path,
        sequence_length=args.sequence_length,
        num_resampler_query=args.num_resampler_query,
        num_obs_token_per_image=args.num_obs_token_per_image,
        calvin_input_image_size=args.calvin_input_image_size,
        patch_size=args.patch_size,
        action_pred_steps=args.action_pred_steps,
        obs_pred=args.obs_pred,
        atten_only_obs=args.atten_only_obs,
        attn_robot_proprio_state=args.attn_robot_proprio_state,
        atten_goal=args.atten_goal,
        atten_goal_state=args.atten_goal_state,
        mask_l_obs_ratio=args.mask_l_obs_ratio,
        transformer_layers=args.transformer_layers,
        hidden_dim=args.hidden_dim,
        transformer_heads=args.transformer_heads,
        phase=args.phase,
        gripper_width=args.gripper_width,
        use_lrnode_latent_update=args.use_lrnode_latent_update,
        lrnode_hidden_dim=args.lrnode_hidden_dim,
        lrnode_motion_dim=args.lrnode_motion_dim,
        lrnode_fast_encoder_type=args.lrnode_fast_encoder_type,
        lrnode_detach_input_latent=args.lrnode_detach_input_latent,
        lrnode_detach_teacher_latent=args.lrnode_detach_teacher_latent,
        lrnode_freeze_action_head_for_lrnode=args.lrnode_freeze_action_head_for_lrnode,
        lrnode_use_post_layernorm=args.lrnode_use_post_layernorm,
        lrnode_multistep_train=args.lrnode_multistep_train,
        lrnode_train_max_horizon=args.lrnode_train_max_horizon,
        lrnode_log_sanity=args.lrnode_log_sanity,
        lrnode_gate_init_bias=args.lrnode_gate_init_bias,
        lrnode_trace=args.lrnode_trace,
    )

    random_seed(args.seed, args.rank)
    print(f"Start running LIBERO evaluation on rank {args.rank}.")

    device_id = args.rank % torch.cuda.device_count()
    if args.precision in ["bf16", "amp_bfloat16", "amp_bf16"]:
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    elif args.precision == "fp32":
        model = model.float()
        if "vision_encoder" in args.bf16_module:
            model.vision_encoder.bfloat16()
        if "causal_transformer" in args.bf16_module:
            model.transformer_backbone.bfloat16()
        if "image_decoder" in args.bf16_module:
            model.image_decoder.bfloat16()
            model.image_decoder_obs_pred_projector.bfloat16()

    model.clip_model.requires_grad_(False)
    model.vision_encoder.requires_grad_(False)
    model = model.to(device_id)
    model._init_model_type()
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    if (
        bool(args.use_lrnode_latent_update)
        and args.lrnode_train_protocol == "adapter"
        and args.resume_from_checkpoint is not None
        and args.finetune_from_pretrained_ckpt is None
    ):
        _, resume_state_dict = _checkpoint_state_dict(args.resume_from_checkpoint)
        if _is_lrnode_adapter_only_state_dict(resume_state_dict):
            raise ValueError(
                "Adapter-only LR-NODE checkpoint was passed to --resume_from_checkpoint "
                "without --finetune_from_pretrained_ckpt. Eval would leave the frozen Seer/action "
                "head randomly initialized. Pass the baseline full Seer checkpoint via "
                "--finetune_from_pretrained_ckpt, then pass the adapter checkpoint via "
                "--resume_from_checkpoint."
            )

    if args.finetune_from_pretrained_ckpt is not None:
        _load_checkpoint_into_model(
            ddp_model,
            args.finetune_from_pretrained_ckpt,
            "base",
            args.rank,
            checkpoint_kind="base" if args.lrnode_train_protocol == "adapter" else None,
        )

    if args.resume_from_checkpoint is not None:
        resume_checkpoint_kind = None
        if args.lrnode_train_protocol == "adapter":
            _, resume_state_dict = _checkpoint_state_dict(args.resume_from_checkpoint)
            resume_checkpoint_kind = (
                "adapter" if _is_lrnode_adapter_only_state_dict(resume_state_dict) else "base"
            )
        _load_checkpoint_into_model(
            ddp_model,
            args.resume_from_checkpoint,
            "resume_or_adapter",
            args.rank,
            checkpoint_kind=resume_checkpoint_kind,
        )
        if args.rank == 0:
            m = ddp_model.module
            print(f"[EVAL MODEL] use_lrnode_latent_update={getattr(m, 'use_lrnode_latent_update', None)}")
            print(f"[EVAL MODEL] lrnode_hidden_dim={getattr(m, 'lrnode_hidden_dim', None)}")
            print(f"[EVAL MODEL] lrnode_motion_dim={getattr(m, 'lrnode_motion_dim', None)}")
            print(f"[EVAL MODEL] lrnode_fast_encoder_type={getattr(m, 'lrnode_fast_encoder_type', None)}")
            print(f"[EVAL MODEL] lrnode_detach_input_latent={getattr(m, 'lrnode_detach_input_latent', None)}")
            print(f"[EVAL MODEL] lrnode_freeze_action_head_for_lrnode={getattr(m, 'lrnode_freeze_action_head_for_lrnode', None)}")
            print(f"[EVAL MODEL] lrnode_use_post_layernorm={getattr(m, 'lrnode_use_post_layernorm', None)}")
            print(f"[EVAL MODEL] lrnode_multistep_train={getattr(m, 'lrnode_multistep_train', None)}")
            print(f"[EVAL MODEL] lrnode_train_max_horizon={getattr(m, 'lrnode_train_max_horizon', None)}")
            print(f"[EVAL MODEL] lrnode_gate_init_bias={getattr(m, 'lrnode_gate_init_bias', None)}")
            print(f"[EVAL MODEL] action_pred_steps={getattr(m, 'action_pred_steps', None)}")

    if args.rank == 0:
        log_dir = os.environ.get("LOG_DIR")
        if log_dir:
            analysis_dir = os.path.join(log_dir, "analysis")
        else:
            ckpt = getattr(args, "resume_from_checkpoint", "")
            ckpt_dir = os.path.dirname(ckpt) if ckpt else os.path.join(os.getcwd(), "eval_analysis", args.run_name)
            analysis_dir = os.path.join(ckpt_dir, "analysis")
        save_lrnode_run_snapshots(args, ddp_model, analysis_dir, repo_dir=os.getcwd())

    ddp_model.eval()
    if args.finetune_type == "libero_10":
        eval_one_epoch_libero_ddp(
            args=args,
            model=ddp_model,
            image_processor=model.image_processor,
            tokenizer=clip,
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    main()
