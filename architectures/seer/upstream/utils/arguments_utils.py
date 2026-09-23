import argparse
import copy
import glob
import os
import random
from collections import OrderedDict
import numpy as np
import yaml
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size

def get_parser(is_eval=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default="RobotFlamingo",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    # Sum of gradient optimization batch size
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-4, type=float)  # 1e-4
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument(
        "--calvin_dataset",
        type=str,
        default='/mnt/petrelfs/share_data/robomani/calvin_data/task_ABCD_D',
        help="path to calvin_dataset",
    )
    parser.add_argument("--warmup_epochs", default=1, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    # hot fix for torch.distributed.launch
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32", "bf16_and_fp32"],
        default="fp32",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=16)
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )
    parser.add_argument('--rgb_pad', type=int, default=-1)
    parser.add_argument('--gripper_pad', type=int, default=-1)
    parser.add_argument(
        "--traj_cons",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--text_aug",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--residual",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--dif_ws",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--partial_data",
        default=False,
        action="store_true"
    )
    # data
    parser.add_argument("--save_every_iter", type=int, default=-1)
    parser.add_argument("--min_window_size", type=int, default=12)
    parser.add_argument("--max_window_size", type=int, default=24)
    parser.add_argument("--multi_step_action", type=int, default=1, help="multiple step action prediction")
    # ceph
    parser.add_argument("--data_in_ceph",default=False, action="store_true")
    # oxe
    parser.add_argument("--root_dir", type=str, default="s3://real_data")
    parser.add_argument("--image_primary_size", type=int, default=200)
    parser.add_argument("--image_wrist_size", type=int, default=84)
    parser.add_argument("--finetune_type", type=str, default="",)   
    # save checkpoint
    parser.add_argument("--start_save_checkpoint", default=-1, type=int)
    parser.add_argument("--save_checkpoint", default=False, action="store_true")
    parser.add_argument("--save_checkpoint_path", required=True, type=str)
    parser.add_argument("--save_checkpoint_seq", type=int, default=1)
    # if validate
    parser.add_argument("--validation", default=False, action="store_true")
    # bf16 module
    parser.add_argument("--bf16_module", type=str, default="")
    # model structure 
    parser.add_argument("--sequence_length", type=int, default=10)
    # for image prediction
    parser.add_argument("--future_steps", type=int, default=3)
    parser.add_argument("--num_resampler_query", type=int, default=9)
    parser.add_argument("--num_obs_token_per_image", type=int, default=9)
    parser.add_argument("--calvin_input_image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    # droid
    parser.add_argument("--primary_mode", type=str, default="image_primary")
    parser.add_argument("--small_size", type=int, default=0)
    parser.add_argument("--dataset_info", type=str, default="droid_success")
    # pretrain
    parser.add_argument("--finetune_from_pretrained_ckpt", type=str, default=None)
    # loss
    parser.add_argument("--loss_arm_action_ratio", type=float, default=1.0)
    parser.add_argument("--loss_gripper_action_ratio", type=float, default=0.01)   
    # action_pred_steps
    parser.add_argument("--action_pred_steps", type=int, default=1)
    # obs_pred
    parser.add_argument("--obs_pred", default=False, action="store_true")
    parser.add_argument("--atten_only_obs", default=False, action="store_true")
    parser.add_argument("--attn_robot_proprio_state", default=False, action="store_true")
    parser.add_argument("--atten_goal", default=0, type=int)
    parser.add_argument("--atten_goal_state", default=False, action="store_true")
    # action mask ratio
    parser.add_argument("--mask_l_obs_ratio", default=0.00, type=float)
    # reset during finetuning
    parser.add_argument("--reset_action_token", default=False, action="store_true")
    parser.add_argument("--reset_obs_token", default=False, action="store_true")
    parser.add_argument("--reset_mask_token", default=False, action="store_true")
    parser.add_argument("--reset_image_decoder", default=False, action="store_true")
    parser.add_argument("--reset_action_decoder", default=False, action="store_true")
    # loss
    parser.add_argument("--loss_action", default=False, action="store_true")
    parser.add_argument("--loss_image", default=False, action="store_true")

    # Seer-only teacher distillation control. Disabled by default.
    parser.add_argument(
        "--seer_distill_teacher_ckpt",
        type=str,
        default=None,
        help=(
            "Optional frozen Seer teacher checkpoint for Seer-only distillation controls. "
            "This is separate from LR-NODE and is used to test whether teacher KD itself, "
            "without the LR-NODE skip module, changes full-Seer K=1 performance."
        ),
    )
    parser.add_argument("--seer_distill_action_weight", type=float, default=0.0)
    parser.add_argument("--seer_distill_latent_weight", type=float, default=0.0)
    parser.add_argument(
        "--seer_distill_teacher_eval_mode",
        type=int,
        default=1,
        help="When 1, keep the Seer distillation teacher in eval mode during training.",
    )

    # LR-NODE latent update. Disabled by default.
    parser.add_argument("--use_lrnode_latent_update", type=int, default=0)
    parser.add_argument("--lrnode_train_latent_distill", type=int, default=0)
    parser.add_argument("--lrnode_eval_skip_full_forward", type=int, default=0)
    parser.add_argument(
        "--lrnode_teacher_target_mode",
        type=str,
        default="shifted_context",
        choices=["shifted_context", "adjacent_sequence"],
        help=(
            "How LR-NODE teacher latents are built. 'shifted_context' runs the teacher policy on "
            "the next normal policy context C_{t+1}; 'adjacent_sequence' keeps the older in-window "
            "z_full[:, t] -> z_full[:, t+1] target."
        ),
    )
    parser.add_argument(
        "--lrnode_context_selected_step",
        type=int,
        default=-1,
        help=(
            "Context token index used for shifted_context teacher probing. -1 means the last "
            "policy-context timestep, matching the steady-state eval cache refresh point."
        ),
    )
    parser.add_argument(
        "--lrnode_train_protocol",
        type=str,
        default="joint",
        choices=["joint", "adapter"],
        help=(
            "LR-NODE training protocol. 'joint' trains normal Seer losses plus LR-NODE auxiliary "
            "losses. 'adapter' freezes the existing Seer/action head and trains only LR-NODE modules."
        ),
    )
    parser.add_argument(
        "--lrnode_freeze_seer_for_adapter",
        type=int,
        default=0,
        help="When 1, freeze all non-LR-NODE parameters before optimizer construction.",
    )
    parser.add_argument(
        "--lrnode_assert_only_lrnode_trainable",
        type=int,
        default=0,
        help="When 1, fail if any non-LR-NODE parameter remains trainable.",
    )
    parser.add_argument("--lrnode_query_interval", type=int, default=1)
    parser.add_argument("--lrnode_latent_weight", type=float, default=1.0)
    parser.add_argument("--lrnode_action_distill_weight", type=float, default=0.5)
    parser.add_argument("--lrnode_bc_weight", type=float, default=0.0)
    parser.add_argument("--lrnode_smooth_weight", type=float, default=0.01)
    parser.add_argument("--lrnode_hidden_dim", type=int, default=256)
    parser.add_argument("--lrnode_motion_dim", type=int, default=128)
    parser.add_argument("--lrnode_fast_encoder_type", type=str, default="diffcnn")
    parser.add_argument("--lrnode_detach_input_latent", type=int, default=1)
    parser.add_argument("--lrnode_detach_teacher_latent", type=int, default=1)
    parser.add_argument("--lrnode_freeze_action_head_for_lrnode", type=int, default=1)
    parser.add_argument("--lrnode_use_post_layernorm", type=int, default=0)
    parser.add_argument("--lrnode_multistep_train", type=int, default=0)
    parser.add_argument("--lrnode_train_max_horizon", type=int, default=2)
    parser.add_argument("--lrnode_log_sanity", type=int, default=1)
    parser.add_argument("--lrnode_gate_init_bias", type=float, default=-4.0)
    parser.add_argument("--lrnode_trace", type=int, default=0)
    parser.add_argument("--lrnode_debug_artifact_interval", type=int, default=0)
    parser.add_argument("--lrnode_eval_step_log", type=int, default=0)
    parser.add_argument(
        "--lrnode_eval_shadow_full_forward",
        "--lrnode_shadow_full_forward",
        dest="lrnode_eval_shadow_full_forward",
        type=int,
        default=0,
        help=(
            "Run a logging-only full Seer forward on LR-NODE skip steps. "
            "The shadow path has separate counters/ensemble state and restores RNG state."
        ),
    )
    parser.add_argument("--lrnode_mechanism_trace", type=int, default=0)
    parser.add_argument("--lrnode_trace_save_latents", type=int, default=0)
    parser.add_argument("--lrnode_trace_episode_limit", type=int, default=0)
    parser.add_argument("--lrnode_trace_output_dir", type=str, default="")
    parser.add_argument(
        "--lrnode_counterfactual_mode",
        type=str,
        default="standard",
        choices=[
            "standard",
            "full_arm_full_gripper",
            "lr_arm_lr_gripper",
            "lr_arm_full_gripper",
            "full_arm_lr_gripper",
            "latent_fusion",
            "matched_random",
        ],
        help=(
            "Diagnostic execution intervention. All non-standard modes require "
            "shadow full-forward and are disabled by default."
        ),
    )
    parser.add_argument(
        "--lrnode_counterfactual_mix_stage",
        type=str,
        default="pre_ensemble",
        choices=["pre_ensemble"],
        help="Arm/gripper branch mixing occurs on raw action-token sequences before ensembling.",
    )
    parser.add_argument("--lrnode_latent_fusion_alpha", type=float, default=0.0)
    parser.add_argument(
        "--lrnode_latent_fusion_mode",
        type=str,
        default="every_step",
        choices=["every_step", "soft_reset_only"],
    )
    parser.add_argument(
        "--lrnode_every_step_filter_mode",
        type=str,
        default="off",
        choices=[
            "off",
            "raw_full",
            "recurrent_prior",
            "fixed_filter",
            "full_latent_ema",
        ],
        help=(
            "Default-off mechanism experiment that runs full Seer every step and "
            "selects the cached/executed latent with an explicit prediction-correction rule."
        ),
    )
    parser.add_argument(
        "--lrnode_every_step_filter_alpha",
        type=float,
        default=0.5,
        help="Fixed-filter correction weight on the current full-Seer latent.",
    )
    parser.add_argument(
        "--lrnode_every_step_filter_beta",
        type=float,
        default=0.5,
        help="Full-latent EMA weight on the current full-Seer latent.",
    )
    parser.add_argument(
        "--lrnode_every_step_filter_diagnostics",
        type=int,
        default=0,
        help=(
            "When 1, decode non-executed prior/full branches for action diagnostics. "
            "RNG is restored and diagnostic latency is reported separately."
        ),
    )
    parser.add_argument("--lrnode_matched_random_seed", type=int, default=20260724)
    parser.add_argument(
        "--lrnode_matched_random_norm_mode",
        type=str,
        default="per_token",
        choices=["per_token", "global"],
    )
    parser.add_argument(
        "--lrnode_eval_profile_full_action_head",
        type=int,
        default=0,
        help=(
            "When 1, profile the existing Seer action head inside full-forward calls. "
            "This adds CUDA synchronization and should be used for latency studies."
        ),
    )
    parser.add_argument(
        "--lrnode_eval_ablation_mode",
        type=str,
        default="stepwise",
        choices=["stepwise", "hold_action", "hold_latent", "seer_token_chunk", "no_delta"],
        help=(
            "Eval-only intervention ablation for skipped LR-NODE steps. 'stepwise' is the existing "
            "delta-conditioned fixed-Euler latent update path."
        ),
    )
    parser.add_argument(
        "--latentloop_segment_grid_enable",
        type=int,
        default=0,
        help=(
            "Default-off segment-length/feedback-density protocol. When enabled, "
            "lrnode_query_interval is the segment length L and only the current-"
            "observation feature is controlled by latentloop_feedback_schedule."
        ),
    )
    parser.add_argument(
        "--latentloop_feedback_schedule",
        type=str,
        default="dense",
        choices=["dense", "alternate", "none"],
        help=(
            "Predeclared intermediate-step feedback mask. The updater and shared "
            "Seer action head still run at every intermediate step."
        ),
    )
    parser.add_argument(
        "--latentloop_same_input_stochasticity_repeats",
        type=int,
        default=0,
        help=(
            "Default-off diagnostic: repeat one fixed preprocessed full-Seer input "
            "and record maximum latent/raw/executed-action differences."
        ),
    )
    parser.add_argument(
        "--latentloop_same_input_stochasticity_output",
        type=str,
        default="",
        help="Optional JSON path for the same-input stochasticity diagnostic.",
    )
    parser.add_argument(
        "--lrnode_no_delta_mode",
        type=str,
        default="zero",
        choices=["zero", "learned_constant", "previous"],
        help="No-delta ablation variant. Only 'zero' is implemented.",
    )
    parser.add_argument(
        "--lrnode_chunk_token_policy",
        type=str,
        default="skip_only",
        choices=["skip_only"],
        help=(
            "Seer action-token chunk ablation policy. 'skip_only' executes the normal full-step "
            "action at refresh steps and cached Seer action tokens only on skipped steps."
        ),
    )
    parser.add_argument(
        "--lrnode_eval_refresh_policy",
        type=str,
        default="periodic",
        choices=["periodic", "first_only", "fixed_budget"],
        help=(
            "Full-Seer refresh policy during LR-NODE eval. 'periodic' is the existing K-step "
            "refresh; 'first_only' runs full Seer only once at episode start; 'fixed_budget' "
            "uses lrnode_eval_max_full_forwards_per_episode full queries spread across the episode."
        ),
    )
    parser.add_argument(
        "--lrnode_eval_max_full_forwards_per_episode",
        type=int,
        default=1,
        help="Episode-level full-Seer query budget for first_only/fixed_budget LR-NODE eval policies.",
    )

    # LatentLoop cross-query plan-continuation study. Every option is inert by
    # default so the established Seer/LatentLoop protocol remains unchanged.
    parser.add_argument("--latentloop_plan_trace", type=int, default=0)
    parser.add_argument("--latentloop_plan_trace_save_latents", type=int, default=0)
    parser.add_argument("--latentloop_plan_trace_output_dir", type=str, default="")
    parser.add_argument("--latentloop_plan_trace_row_id", type=str, default="")
    parser.add_argument("--latentloop_plan_trace_paired_group", type=str, default="")
    parser.add_argument(
        "--latentloop_feedback_source",
        type=str,
        default="current",
        choices=["current", "time_shifted"],
        help=(
            "Feature consumed by skipped-step latent updates. 'time_shifted' "
            "uses the previous intermediate step's feature while still encoding "
            "and staging the current observation for the next step."
        ),
    )
    parser.add_argument(
        "--latentloop_plan_adapter_mode",
        type=str,
        default="off",
        choices=["off", "action_correction", "anchor_bridge"],
    )
    parser.add_argument("--latentloop_plan_adapter_hidden_dim", type=int, default=0)
    parser.add_argument(
        "--latentloop_plan_parameter_match_tolerance", type=float, default=0.05
    )
    parser.add_argument("--latentloop_plan_arm_weight", type=float, default=1.0)
    parser.add_argument("--latentloop_plan_gripper_weight", type=float, default=1.0)
    parser.add_argument("--latentloop_plan_latent_weight", type=float, default=1.0)
    # Source-locked Q1/Q2 comparison protocol. The master switch is off by
    # default; these options cannot alter canonical Seer/LatentLoop runs.
    parser.add_argument("--latentloop_comparison_protocol", type=int, default=0)
    parser.add_argument(
        "--latentloop_comparison_offset_schedule",
        type=str,
        default="adjacent",
        choices=["adjacent", "cyclic_k4"],
    )
    parser.add_argument(
        "--latentloop_comparison_split_role",
        type=str,
        default="full",
        choices=["full", "train", "validation"],
    )
    parser.add_argument(
        "--latentloop_comparison_validation_fraction", type=float, default=0.05
    )
    parser.add_argument(
        "--latentloop_comparison_validation_seed", type=int, default=20260805
    )
    parser.add_argument(
        "--latentloop_comparison_target_microbatches",
        type=int,
        default=0,
        help="Optional exact global dataloader-microbatch budget; zero disables it.",
    )
    parser.add_argument(
        "--latentloop_comparison_warmup_microbatches", type=int, default=0
    )
    parser.add_argument(
        "--latentloop_comparison_checkpoint_microbatches", type=int, default=0
    )
    parser.add_argument("--latentloop_action_arm_weight", type=float, default=0.0)
    parser.add_argument("--latentloop_action_gripper_weight", type=float, default=0.0)
    parser.add_argument("--latentloop_action_exec_weight", type=float, default=0.0)
    parser.add_argument("--latentloop_action_reg_weight", type=float, default=0.0)
    parser.add_argument(
        "--latentloop_nonrecurrent_latent_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--latentloop_nonrecurrent_action_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--latentloop_nonrecurrent_smooth_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--latentloop_comparison_selection_metric",
        type=str,
        default="validation_total_loss",
        choices=["validation_total_loss"],
    )
    parser.add_argument(
        "--lrnode_init_adapter_ckpt",
        type=str,
        default=None,
        help=(
            "Optional model-only LatentLoop adapter initialization. Unlike resume, "
            "optimizer/scheduler state is never loaded."
        ),
    )
    parser.add_argument("--latentloop_cqpc_weight", type=float, default=0.0)
    parser.add_argument("--latentloop_cqpc_gamma", type=float, default=0.0)
    parser.add_argument("--latentloop_cqpc_arm_weight", type=float, default=1.0)
    parser.add_argument("--latentloop_cqpc_gripper_weight", type=float, default=1.0)
    parser.add_argument("--latentloop_cqpc_log_teacher_disagreement", type=int, default=0)

    # Joint Latent-Anchored Action Surrogate. Every option is inert unless the
    # explicit mode is joint or wide.
    parser.add_argument(
        "--joint_latent_action_surrogate_mode",
        type=str,
        default="off",
        choices=["off", "joint", "wide"],
    )
    parser.add_argument(
        "--joint_latent_action_surrogate_stage",
        type=str,
        default="off",
        choices=["off", "stage_a", "stage_b"],
    )
    parser.add_argument(
        "--joint_latent_action_surrogate_hidden_dim", type=int, default=192
    )
    parser.add_argument(
        "--joint_latent_action_surrogate_parameter_match_tolerance",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--joint_latent_action_surrogate_pretrained_lr_scale",
        type=float,
        default=0.0,
        help="Required in (0,1) for Stage B; ignored in Stage A.",
    )
    parser.add_argument(
        "--joint_latent_action_surrogate_init_ckpt", type=str, default=None
    )
    parser.add_argument(
        "--joint_latent_action_surrogate_calibration_only", type=int, default=0
    )
    parser.add_argument(
        "--joint_latent_action_surrogate_target_microbatches", type=int, default=0
    )
    parser.add_argument(
        "--joint_latent_action_surrogate_warmup_microbatches", type=int, default=0
    )
    parser.add_argument(
        "--joint_latent_action_surrogate_checkpoint_microbatches", type=int, default=0
    )
    parser.add_argument("--joint_latent_weight", type=float, default=0.0)
    parser.add_argument("--joint_latent_action_weight", type=float, default=0.0)
    parser.add_argument("--joint_surrogate_weight", type=float, default=0.0)
    parser.add_argument("--joint_executed_token_weight", type=float, default=0.0)
    parser.add_argument("--joint_tail_weight", type=float, default=0.0)
    parser.add_argument("--joint_gripper_weight", type=float, default=0.0)
    parser.add_argument("--joint_residual_weight", type=float, default=0.0)
    parser.add_argument("--joint_error_trace", type=int, default=0)
    parser.add_argument("--joint_error_trace_output_dir", type=str, default="")
    parser.add_argument("--joint_force_exact_action_head", type=int, default=0)

    # Default-off Seer horizon-provenance-aware hierarchical correction.
    parser.add_argument(
        "--latentloop_hierarchical_mode",
        type=str,
        default="off",
        choices=[
            "off",
            "full_seer",
            "pure_latentloop",
            "pure_action_correction",
            "hybrid",
        ],
    )
    parser.add_argument("--latentloop_hierarchical_full_interval", type=int, default=8)
    parser.add_argument(
        "--latentloop_hierarchical_regeneration_interval", type=int, default=3
    )
    parser.add_argument(
        "--latentloop_hierarchical_action_checkpoint", type=str, default=""
    )
    parser.add_argument("--latentloop_hierarchical_trace", type=int, default=0)
    parser.add_argument(
        "--latentloop_hierarchical_trace_output_dir", type=str, default=""
    )
    parser.add_argument("--latentloop_hierarchical_assert_invariants", type=int, default=1)
    
    # calvin
    parser.add_argument("--except_lang", default=False, action="store_true")
    # gpt2
    parser.add_argument("--transformer_layers", default=12, type=int)
    parser.add_argument("--hidden_dim", default=384, type=int)
    parser.add_argument("--transformer_heads", default=12, type=int)
    # pretrain, finetune, evaluate
    parser.add_argument('--phase', required=True, help='pretrain, finetune, evaluate')
    # libero 
    parser.add_argument(
        "--libero_path",
        default=os.environ.get("LIBERO_PATH", ""),
        help="Path to the LIBERO repository. Defaults to the LIBERO_PATH environment variable.",
    )
    parser.add_argument("--libero_img_size", default=128, type=int)
    parser.add_argument("--libero_eval_max_steps", default=600, type=int)
    parser.add_argument("--gripper_width", default=False, action="store_true")
    parser.add_argument("--load_libero_file", type=str, default="h5")
    parser.add_argument("--eval_libero_ensembling", default=False, action="store_true")
    parser.add_argument("--ensembling_temp", default=0.01, type=float)
    # real
    parser.add_argument("--real_dataset_names", type=str)
    parser.add_argument("--use_aug_data", default=False, action="store_true")
    parser.add_argument("--real_eval_max_steps", default=600, type=int)
    # preprocess
    parser.add_argument("--max_rel_pos", type=float, default=0.02)
    parser.add_argument("--max_rel_orn", type=float, default=0.05)
    parser.add_argument("--magic_scaling_factor_pos", type=float, default=1.0)
    parser.add_argument("--magic_scaling_factor_orn", type=float, default=1.0)
    # for eval
    if is_eval:
        parser.add_argument("--calvin_conf_path", type=str, help="path to calvin configuration file")
        parser.add_argument("--future_act_len", default=-1, type=int)
        parser.add_argument(
            "--visualize",
            default=False,
            action="store_true"
        )
        parser.add_argument(
            "--reset",
            default=False,
            action="store_true"
        )
        parser.add_argument(
            "--diverse_inst",
            default=False,
            action="store_true"
        )
        parser.add_argument("--save_video", default=False, action="store_true")
        parser.add_argument("--save_video_all_ranks", default=False, action="store_true")
        parser.add_argument("--video_fps", type=int, default=20)
        parser.add_argument("--video_stride", type=int, default=1)
        parser.add_argument("--pad_length", type=int, default=-1)
    parser.add_argument("--window_size", type=int, default=13)
    parser.add_argument("--vit_checkpoint_path", type=str)
    args = parser.parse_args()

    return parser

    # if args.dataloading_type == "seer":
    #     if args.phase == "pretrain":
    #         if args.finetune_type == "calvin":
    #             args.window_size = args.sequence_length + args.future_steps 
    #         else:
    #             args.window_size = args.sequence_length
    #     elif args.phase == "finetune":
    #         args.window_size = args.sequence_length + args.future_steps
