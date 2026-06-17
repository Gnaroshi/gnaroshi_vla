import json
import os
import subprocess
from pathlib import Path


LRNODE_FLAG_KEYS = [
    "use_lrnode_latent_update",
    "lrnode_train_latent_distill",
    "lrnode_eval_skip_full_forward",
    "lrnode_train_protocol",
    "lrnode_freeze_seer_for_adapter",
    "lrnode_assert_only_lrnode_trainable",
    "lrnode_query_interval",
    "lrnode_detach_input_latent",
    "lrnode_detach_teacher_latent",
    "lrnode_freeze_action_head_for_lrnode",
    "lrnode_use_post_layernorm",
    "lrnode_multistep_train",
    "lrnode_train_max_horizon",
    "lrnode_latent_weight",
    "lrnode_action_distill_weight",
    "lrnode_bc_weight",
    "lrnode_smooth_weight",
    "lrnode_hidden_dim",
    "lrnode_motion_dim",
    "lrnode_fast_encoder_type",
    "lrnode_gate_init_bias",
    "lrnode_trace",
    "lrnode_log_sanity",
    "lrnode_debug_artifact_interval",
    "lrnode_eval_step_log",
    "lrnode_eval_shadow_full_forward",
]


def _json_default(value):
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    return str(value)


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def collect_git_snapshot(repo_dir):
    snapshot = {"repo_dir": os.path.abspath(repo_dir)}
    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "status_short": ["git", "status", "--short"],
        "diff_stat": ["git", "diff", "--stat"],
    }
    for key, cmd in commands.items():
        try:
            ret = subprocess.run(
                cmd,
                cwd=repo_dir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            snapshot[key] = ret.stdout.strip()
            if ret.returncode != 0:
                snapshot[f"{key}_error"] = ret.stderr.strip()
        except Exception as exc:
            snapshot[f"{key}_error"] = repr(exc)
    return snapshot


def collect_model_trainable_params(model):
    base = model.module if hasattr(model, "module") else model
    groups = {}
    total = 0
    trainable = 0
    for name, param in base.named_parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
        prefix = name.split(".", 1)[0]
        item = groups.setdefault(prefix, {"total": 0, "trainable": 0})
        item["total"] += count
        if param.requires_grad:
            item["trainable"] += count
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "by_top_level_module": groups,
    }


def collect_freeze_status(model):
    base = model.module if hasattr(model, "module") else model
    status = {}
    for module_name in [
        "clip_model",
        "vision_encoder",
        "transformer_backbone",
        "action_decoder",
        "arm_action_decoder",
        "gripper_action_decoder",
        "lrnode_delta_encoder",
        "lrnode_dynamics",
    ]:
        module = getattr(base, module_name, None)
        if module is None:
            status[module_name] = {"present": False}
            continue
        params = list(module.parameters())
        status[module_name] = {
            "present": True,
            "num_params": int(sum(p.numel() for p in params)),
            "num_trainable_params": int(sum(p.numel() for p in params if p.requires_grad)),
            "all_frozen": bool(params and all(not p.requires_grad for p in params)),
            "any_trainable": bool(any(p.requires_grad for p in params)),
        }
    return status


def collect_loss_weights(args):
    keys = [
        "loss_arm_action_ratio",
        "loss_gripper_action_ratio",
        "loss_action",
        "loss_image",
        "obs_pred",
        "lrnode_train_protocol",
        "lrnode_freeze_seer_for_adapter",
        "lrnode_latent_weight",
        "lrnode_action_distill_weight",
        "lrnode_bc_weight",
        "lrnode_smooth_weight",
    ]
    return {key: getattr(args, key, None) for key in keys}


def collect_lrnode_flags(args):
    return {key: getattr(args, key, None) for key in LRNODE_FLAG_KEYS if hasattr(args, key)}


def save_lrnode_run_snapshots(args, model, output_dir, repo_dir=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_json(os.path.join(output_dir, "args_snapshot.json"), {k: v for k, v in vars(args).items()})
    write_json(os.path.join(output_dir, "lrnode_flags_snapshot.json"), collect_lrnode_flags(args))
    write_json(os.path.join(output_dir, "loss_weights_snapshot.json"), collect_loss_weights(args))
    write_json(os.path.join(output_dir, "model_trainable_params.json"), collect_model_trainable_params(model))
    write_json(os.path.join(output_dir, "freeze_status_snapshot.json"), collect_freeze_status(model))
    if repo_dir is None:
        repo_dir = os.getcwd()
    write_json(os.path.join(output_dir, "git_snapshot.json"), collect_git_snapshot(repo_dir))
