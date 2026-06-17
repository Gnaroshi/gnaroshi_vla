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
    parser.add_argument("--lrnode_eval_shadow_full_forward", type=int, default=0)
    
    # calvin
    parser.add_argument("--except_lang", default=False, action="store_true")
    # gpt2
    parser.add_argument("--transformer_layers", default=12, type=int)
    parser.add_argument("--hidden_dim", default=384, type=int)
    parser.add_argument("--transformer_heads", default=12, type=int)
    # pretrain, finetune, evaluate
    parser.add_argument('--phase', required=True, help='pretrain, finetune, evaluate')
    # libero 
    # parser.add_argument("--libero_path", default="/ailab/user/tianyang/Code/LIBERO")
    parser.add_argument("--libero_path", default="/home/mingyujung/private/LIBERO")
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
