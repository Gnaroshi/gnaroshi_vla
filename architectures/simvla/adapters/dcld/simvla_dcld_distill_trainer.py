"""Distill-only trainer skeleton for SimVLA DCLD.

This file intentionally does not run full training. It provides the freeze and
optimizer safety machinery that a cache-backed DCLD trainer should use.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from methods.dcld.modules import DCLDCore, DeltaObservation
from methods.dcld.training import DCLDLossWeights, compute_dcld_losses

ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))


@dataclass
class SimVLADCLDDistillConfig:
    checkpoint_id: str = "YuankaiLuo/SimVLA-LIBERO"
    teacher_cache: str | None = None
    norm_stats_path: str | None = None
    output_dir: str = "runs/simvla_dcld_distill"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_batches: int = 2
    condition_weight: float = 0.05
    action_weight: float = 0.1
    smooth_weight: float = 0.001
    cosine_weight: float = 0.05
    dynamics_type: str = "dense"
    delta_dim: int = 512
    hidden_dim: int = 1024
    rank_dim: int = 64
    gate_mode: str = "dense"
    gate_bias: float = -4.0
    use_post_layernorm: bool = False
    log_interval: int = 0
    save_interval: int = 10000
    wandb_project: str = "gnaroshi-simvla-dcld"
    wandb_name: str | None = None
    wandb_mode: str | None = None
    wandb_log_interval: int = 1000
    debug_env_var: str = "SIMVLA_DCLD_DEBUG"


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def unfreeze_module(module: nn.Module) -> None:
    module.train()
    for param in module.parameters():
        param.requires_grad_(True)


def named_trainable_parameters(prefix: str, module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(f"{prefix}.{name}", param) for name, param in module.named_parameters() if param.requires_grad]


class SimVLADCLDDistillTrainer:
    """Frozen-teacher, DCLD-only trainer scaffold."""

    def __init__(
        self,
        *,
        teacher_model: nn.Module,
        dcld_core: DCLDCore,
        output_dir: str | Path,
        optional_trainable_adapters: Sequence[tuple[str, nn.Module]] = (),
        loss_weights: DCLDLossWeights | None = None,
    ) -> None:
        self.teacher_model = teacher_model
        self.dcld_core = dcld_core
        self.optional_trainable_adapters = list(optional_trainable_adapters)
        self.output_dir = Path(output_dir)
        self.loss_weights = loss_weights or DCLDLossWeights()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_id: str,
        dcld_core: DCLDCore,
        output_dir: str | Path,
        norm_stats_path: str | None = None,
        device: str | torch.device = "cuda",
        optional_trainable_adapters: Sequence[tuple[str, nn.Module]] = (),
    ) -> "SimVLADCLDDistillTrainer":
        from models.modeling_smolvlm_vla import SmolVLMVLA

        teacher_model = SmolVLMVLA.from_pretrained(checkpoint_id)
        teacher_model.to(device)
        teacher_model.eval()
        if norm_stats_path and Path(norm_stats_path).exists():
            teacher_model.action_space.load_norm_stats(norm_stats_path)
        return cls(
            teacher_model=teacher_model,
            dcld_core=dcld_core.to(device),
            output_dir=output_dir,
            optional_trainable_adapters=optional_trainable_adapters,
        )

    def prepare_frozen_teacher(self) -> None:
        freeze_module(self.teacher_model)
        unfreeze_module(self.dcld_core)
        for _, module in self.optional_trainable_adapters:
            unfreeze_module(module)

    def trainable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        params = named_trainable_parameters("dcld_core", self.dcld_core)
        for name, module in self.optional_trainable_adapters:
            params.extend(named_trainable_parameters(name, module))
        return params

    def assert_only_dcld_trainable(self) -> None:
        teacher_trainable = [name for name, param in self.teacher_model.named_parameters() if param.requires_grad]
        if teacher_trainable:
            preview = teacher_trainable[:20]
            raise RuntimeError(f"Teacher SimVLA has trainable parameters: {preview}")
        if not self.trainable_named_parameters():
            raise RuntimeError("No DCLD parameters are trainable")

    def build_optimizer(self, *, learning_rate: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
        self.prepare_frozen_teacher()
        self.assert_only_dcld_trainable()
        named_params = self.trainable_named_parameters()
        optimizer = torch.optim.AdamW(
            [{"name": "dcld", "params": [param for _, param in named_params]}],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.save_freeze_snapshots(optimizer_param_names=[name for name, _ in named_params])
        return optimizer

    def save_freeze_snapshots(self, optimizer_param_names: Sequence[str]) -> None:
        trainable = [
            {"name": name, "shape": list(param.shape), "numel": int(param.numel())}
            for name, param in self.trainable_named_parameters()
        ]
        freeze_status = {
            "teacher_total_params": int(sum(p.numel() for p in self.teacher_model.parameters())),
            "teacher_trainable_params": int(sum(p.numel() for p in self.teacher_model.parameters() if p.requires_grad)),
            "dcld_total_params": int(sum(p.numel() for p in self.dcld_core.parameters())),
            "dcld_trainable_params": int(sum(p.numel() for p in self.dcld_core.parameters() if p.requires_grad)),
            "optional_trainable_adapters": [name for name, _ in self.optional_trainable_adapters],
        }
        with (self.output_dir / "model_trainable_params.json").open("w", encoding="utf-8") as f:
            json.dump(trainable, f, indent=2, sort_keys=True)
        with (self.output_dir / "freeze_status_snapshot.json").open("w", encoding="utf-8") as f:
            json.dump(freeze_status, f, indent=2, sort_keys=True)
        with (self.output_dir / "optimizer_param_names.txt").open("w", encoding="utf-8") as f:
            for name in optimizer_param_names:
                f.write(f"{name}\n")

    def smoke_train_cached_batches(
        self,
        batches: Iterable[dict],
        optimizer: torch.optim.Optimizer,
        *,
        max_batches: int = 2,
    ) -> list[dict[str, float]]:
        if os.environ.get("SIMVLA_DCLD_DEBUG") != "1":
            raise RuntimeError("Set SIMVLA_DCLD_DEBUG=1 to run smoke_train_cached_batches")
        self.prepare_frozen_teacher()
        self.assert_only_dcld_trainable()
        logs: list[dict[str, float]] = []
        for idx, batch in enumerate(batches):
            if idx >= max_batches:
                break
            latent_prev = batch["latent_prev"]
            target_condition = batch["target_condition"]
            delta_obs = batch["delta_obs"]
            if not isinstance(delta_obs, DeltaObservation):
                raise TypeError("batch['delta_obs'] must be a DeltaObservation")
            pred = self.dcld_core.update_latent(latent_prev, delta_obs).latent
            loss_dict = compute_dcld_losses(pred, target_condition, weights=self.loss_weights)
            optimizer.zero_grad(set_to_none=True)
            loss_dict["total"].backward()
            optimizer.step()
            logs.append({name: float(value.detach().cpu().item()) for name, value in loss_dict.items()})
        return logs


def write_config_snapshot(path: str | Path, config: SimVLADCLDDistillConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2, sort_keys=True)


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:
        return "unknown"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def run_cmd(args: list[str], cwd: str | Path = ROOT) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def init_wandb_run(args: Any, config: SimVLADCLDDistillConfig, output_dir: Path) -> Any | None:
    if args.disable_wandb:
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"[DCLD TRAIN] wandb is unavailable, continuing without W&B: {exc}", flush=True)
        return None

    mode = args.wandb_mode or os.environ.get("WANDB_MODE")
    if mode == "disabled":
        return None
    project = args.wandb_project or os.environ.get("WANDB_PROJECT") or "gnaroshi-simvla-dcld"
    name = args.wandb_name or os.environ.get("WANDB_NAME") or output_dir.name
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
    init_kwargs: dict[str, Any] = {
        "project": project,
        "name": name,
        "config": asdict(config),
        "dir": str(output_dir),
    }
    if mode:
        init_kwargs["mode"] = mode
    if entity:
        init_kwargs["entity"] = entity
    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"[DCLD TRAIN] wandb init failed, continuing without W&B: {exc}", flush=True)
        return None


def wandb_payload_from_step(step_log: dict[str, Any]) -> dict[str, float | int | bool]:
    payload: dict[str, float | int | bool] = {}
    for key, value in step_log.items():
        if isinstance(value, bool):
            payload[f"train/{key}"] = value
        elif isinstance(value, (int, float)):
            payload[f"train/{key}"] = value
    if step_log.get("checkpoint_path"):
        payload["train/checkpoint_saved"] = 1
    return payload


def tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach()
    stats: dict[str, Any] = {
        "shape": list(y.shape),
        "dtype": str(y.dtype),
        "device": str(y.device),
    }
    if y.numel() == 0:
        return stats
    if not torch.is_floating_point(y):
        y = y.float()
    else:
        y = y.float()
    stats.update(
        {
            "min": float(y.min().item()),
            "max": float(y.max().item()),
            "mean": float(y.mean().item()),
            "std": float(y.std(unbiased=False).item()) if y.numel() > 1 else 0.0,
            "norm": float(y.norm().item()),
            "has_nan": bool(torch.isnan(y).any().item()),
        }
    )
    return stats


def grad_summary(teacher: nn.Module, dcld_core: nn.Module) -> dict[str, Any]:
    teacher_grad_nonzero = 0
    teacher_grad_tensors = 0
    for param in teacher.parameters():
        if param.grad is not None:
            teacher_grad_tensors += 1
            if torch.any(param.grad.detach() != 0):
                teacher_grad_nonzero += 1

    dcld_grad_nonzero = 0
    dcld_grad_tensors = 0
    dcld_grad_norm_sq = 0.0
    fast_grad_nonzero = 0
    updater_grad_nonzero = 0
    for name, param in dcld_core.named_parameters():
        if param.grad is None:
            continue
        dcld_grad_tensors += 1
        grad = param.grad.detach().float()
        dcld_grad_norm_sq += float(torch.sum(grad * grad).item())
        nonzero = bool(torch.any(grad != 0).item())
        if nonzero:
            dcld_grad_nonzero += 1
            if name.startswith("delta_encoder"):
                fast_grad_nonzero += 1
            if name.startswith("dynamics"):
                updater_grad_nonzero += 1

    return {
        "teacher_grad_tensors": teacher_grad_tensors,
        "teacher_grad_nonzero_tensors": teacher_grad_nonzero,
        "dcld_grad_tensors": dcld_grad_tensors,
        "dcld_grad_nonzero_tensors": dcld_grad_nonzero,
        "dcld_grad_l2_norm": dcld_grad_norm_sq**0.5,
        "fast_visual_delta_encoder_grad_nonzero_tensors": fast_grad_nonzero,
        "dcld_updater_grad_nonzero_tensors": updater_grad_nonzero,
    }


def compute_cache_backed_losses(
    *,
    c_prev: torch.Tensor,
    c_pred: torch.Tensor,
    c_teacher: torch.Tensor,
    pred_action: torch.Tensor,
    teacher_action: torch.Tensor,
    weights: DCLDLossWeights,
) -> dict[str, torch.Tensor]:
    condition = F.mse_loss(c_pred, c_teacher.detach())
    cosine = 1.0 - F.cosine_similarity(
        c_pred.flatten(start_dim=1),
        c_teacher.detach().flatten(start_dim=1),
        dim=-1,
    ).mean()
    action = F.l1_loss(pred_action, teacher_action.detach())
    smooth = F.mse_loss(c_pred, c_prev.detach())
    total = (
        weights.condition_mse * condition
        + weights.condition_cosine * cosine
        + weights.action_l1 * action
        + weights.smoothness * smooth
    )
    return {
        "condition_mse": condition,
        "condition_cosine": cosine,
        "action_l1": action,
        "smoothness": smooth,
        "total": total,
    }


def _prepend_batch(first_batch: dict[str, Any], rest: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    yield first_batch
    yield from rest


def run_cache_backed_distill(args: Any) -> int:
    from models.modeling_smolvlm_vla import SmolVLMVLA

    from .simvla_action_adapter import SimVLAActionAdapter
    from .simvla_teacher_cache import iter_transition_batches

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    debug_mode = bool(args.debug or os.environ.get("SIMVLA_DCLD_DEBUG") == "1")
    max_batches = int(args.max_batches if args.max_batches is not None else (2 if debug_mode else args.max_steps))
    if not debug_mode and max_batches <= 0:
        raise RuntimeError("Set --max-steps for production training or --debug for smoke mode")

    config = SimVLADCLDDistillConfig(
        checkpoint_id=args.checkpoint,
        teacher_cache=args.teacher_cache,
        norm_stats_path=args.norm_stats,
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_batches=max_batches,
        condition_weight=args.condition_weight,
        action_weight=args.action_weight,
        smooth_weight=args.smooth_weight,
        cosine_weight=args.cosine_weight,
        dynamics_type=args.dynamics_type,
        delta_dim=args.delta_dim,
        hidden_dim=args.hidden_dim,
        rank_dim=args.rank_dim,
        gate_mode=args.gate_mode,
        gate_bias=args.gate_bias,
        use_post_layernorm=args.use_post_layernorm,
        log_interval=max(0, int(args.log_interval)),
        save_interval=int(args.save_interval),
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_mode=args.wandb_mode or os.environ.get("WANDB_MODE"),
        wandb_log_interval=max(1, int(args.wandb_log_interval)),
    )
    write_config_snapshot(output_dir / "args_snapshot.json", config)
    write_json(
        output_dir / "git_snapshot.json",
        {
            "root_head": run_cmd(["git", "rev-parse", "HEAD"]),
            "root_status_short": run_cmd(["git", "status", "--short"]),
            "simvla_upstream_head": run_cmd(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"]),
            "simvla_upstream_status_short": run_cmd(["git", "-C", str(UPSTREAM), "status", "--short"]),
        },
    )
    weights = DCLDLossWeights(
        condition_mse=args.condition_weight,
        condition_cosine=args.cosine_weight,
        action_l1=args.action_weight,
        smoothness=args.smooth_weight,
    )
    write_json(output_dir / "loss_weights_snapshot.json", asdict(weights))

    transition_iter = iter_transition_batches(args.teacher_cache, device=device, max_batches=max_batches)
    first_batch = next(transition_iter)
    latent_dim = int(first_batch["c_prev"].shape[-1])
    dcld_core = DCLDCore(
        latent_dim=latent_dim,
        delta_dim=args.delta_dim,
        hidden_dim=args.hidden_dim,
        dynamics_type=args.dynamics_type,
        rank_dim=args.rank_dim,
        gate_mode=args.gate_mode,
        gate_bias=args.gate_bias,
        use_post_layernorm=args.use_post_layernorm,
    ).to(device)
    with torch.no_grad():
        # Initialize LazyLinear layers before freeze/optimizer safety checks.
        _ = dcld_core.update_latent(first_batch["c_prev"], first_batch["delta_obs"], dt=1.0, age=1.0)

    teacher = SmolVLMVLA.from_pretrained(args.checkpoint).to(device)
    teacher.eval()
    if args.norm_stats and Path(args.norm_stats).exists():
        teacher.action_space.load_norm_stats(str(args.norm_stats))

    trainer = SimVLADCLDDistillTrainer(
        teacher_model=teacher,
        dcld_core=dcld_core,
        output_dir=output_dir,
        loss_weights=weights,
    )
    optimizer = trainer.build_optimizer(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    action_adapter = SimVLAActionAdapter(teacher)
    wandb_run = init_wandb_run(args, config, output_dir)

    progress_path = output_dir / "train_progress.jsonl"
    latest_metrics_path = output_dir / "latest_metrics.json"
    progress_path.write_text("", encoding="utf-8")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest_checkpoint.txt"
    log_interval = max(0, int(args.log_interval))
    save_interval = int(args.save_interval)
    wandb_log_interval = max(1, int(args.wandb_log_interval))
    progress_bar = None
    try:
        from tqdm.auto import tqdm

        progress_bar = tqdm(
            total=max_batches,
            desc="DCLD distill",
            dynamic_ncols=True,
            mininterval=float(args.tqdm_mininterval),
            disable=bool(args.no_tqdm or not sys.stderr.isatty()),
        )
    except Exception as exc:
        print(f"[DCLD TRAIN] tqdm is unavailable, using sparse event logs only: {exc}", flush=True)

    def emit_event(message: str) -> None:
        if progress_bar is not None and not progress_bar.disable:
            progress_bar.write(message)
        else:
            print(message, flush=True)

    def persist_step_event(step_log: dict[str, Any]) -> None:
        logs.append(step_log)
        append_jsonl(progress_path, step_log)
        write_json(latest_metrics_path, step_log)

    def log_step_to_wandb(step_log: dict[str, Any]) -> None:
        if wandb_run is None:
            return
        try:
            wandb_run.log(wandb_payload_from_step(step_log), step=int(step_log["step"]))
            if step_log.get("checkpoint_path"):
                wandb_run.summary["latest_checkpoint"] = str(step_log["checkpoint_path"])
                wandb_run.summary["latest_checkpoint_step"] = int(step_log["step"])
        except Exception as exc:
            emit_event(f"[DCLD TRAIN] wandb log failed, continuing: {exc}")

    logs: list[dict[str, Any]] = []
    last_batch = first_batch
    last_step_log: dict[str, Any] | None = None
    steps_run = 0
    train_start_time = time.time()

    def checkpoint_payload(step_no: int) -> dict[str, Any]:
        return {
            "checkpoint_type": "simvla_dcld_only",
            "step": int(step_no),
            "dcld_state_dict": dcld_core.state_dict(),
            "config": {
                "latent_dim": latent_dim,
                "dynamics_type": args.dynamics_type,
                "gate_bias": args.gate_bias,
                "delta_dim": dcld_core.delta_dim,
                "hidden_dim": dcld_core.hidden_dim,
                "rank_dim": dcld_core.rank_dim,
                "gate_mode": dcld_core.gate_mode,
                "use_post_layernorm": dcld_core.use_post_layernorm,
                "visual_input": "raw_libero_rgb_[0,1]",
                "condition_shape": list(last_batch["c_prev"].shape),
            },
        }

    def save_dcld_checkpoint(path: Path, step_no: int) -> None:
        torch.save(checkpoint_payload(step_no), path)
        latest_path.write_text(str(path) + "\n", encoding="utf-8")

    for step, batch in enumerate(_prepend_batch(first_batch, transition_iter)):
        if step >= max_batches:
            break
        step_no = step + 1
        steps_run = step_no
        last_batch = batch
        c_prev = batch["c_prev"]
        c_teacher = batch["c_teacher"]
        delta_obs = batch["delta_obs"]
        teacher_action = batch["action_teacher"]

        update = dcld_core.update_latent(c_prev, delta_obs, dt=1.0, age=1.0)
        pred_action = action_adapter.decode_action_from_condition(
            update.latent,
            batch["proprio"],
            steps=args.action_steps,
            deterministic=True,
            requires_grad=True,
        )
        loss_dict = compute_cache_backed_losses(
            c_prev=c_prev,
            c_pred=update.latent,
            c_teacher=c_teacher,
            pred_action=pred_action,
            teacher_action=teacher_action,
            weights=weights,
        )
        optimizer.zero_grad(set_to_none=True)
        loss_dict["total"].backward()
        optimizer.step()

        with torch.no_grad():
            hold_action = action_adapter.decode_action_from_condition(
                c_prev,
                batch["proprio"],
                steps=args.action_steps,
                deterministic=True,
            )
            condition_mse_hold = F.mse_loss(c_prev, c_teacher).item()
            condition_mse_pred = F.mse_loss(update.latent.detach(), c_teacher).item()
            action_l1_hold = F.l1_loss(hold_action, teacher_action).item()
            action_l1_pred = F.l1_loss(pred_action.detach(), teacher_action).item()
            update_norm = update.dynamics.update.detach().flatten(start_dim=1).norm(dim=-1).mean()
            c_norm = c_prev.detach().flatten(start_dim=1).norm(dim=-1).mean()

        gate = update.dynamics.gate.detach().float()
        elapsed_seconds = time.time() - train_start_time
        seconds_per_step = elapsed_seconds / max(1, step_no)
        remaining_steps = max(0, max_batches - step_no)
        eta_seconds = seconds_per_step * remaining_steps
        progress_pct = 100.0 * step_no / max(1, max_batches)
        step_log = {
            "step": int(step_no),
            "max_steps": int(max_batches),
            "progress_pct": float(progress_pct),
            "elapsed_seconds": float(elapsed_seconds),
            "seconds_per_step": float(seconds_per_step),
            "eta_seconds": float(eta_seconds),
            "total": float(loss_dict["total"].detach().item()),
            "condition_mse": float(loss_dict["condition_mse"].detach().item()),
            "condition_cosine": float(loss_dict["condition_cosine"].detach().item()),
            "action_l1": float(loss_dict["action_l1"].detach().item()),
            "smoothness": float(loss_dict["smoothness"].detach().item()),
            "condition_mse_hold": float(condition_mse_hold),
            "condition_mse_pred": float(condition_mse_pred),
            "condition_mse_improvement": float(condition_mse_hold - condition_mse_pred),
            "action_l1_hold": float(action_l1_hold),
            "action_l1_pred": float(action_l1_pred),
            "gate_mean": float(gate.mean().item()),
            "gate_std": float(gate.std(unbiased=False).item()),
            "gate_min": float(gate.min().item()),
            "gate_max": float(gate.max().item()),
            "u_delta_norm": float(update.delta_feature.detach().norm(dim=-1).mean().item()),
            "update_norm": float(update_norm.item()),
            "update_to_condition_norm_ratio": float((update_norm / (c_norm + 1e-8)).item()),
        }
        last_step_log = step_log
        is_checkpoint_step = save_interval > 0 and step_no % save_interval == 0
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{step_log['total']:.4g}",
                cond=f"{step_log['condition_mse']:.4g}",
                act=f"{step_log['action_l1']:.4g}",
                gate=f"{step_log['gate_mean']:.3g}",
                refresh=False,
            )
        if is_checkpoint_step:
            periodic_ckpt = ckpt_dir / f"dcld_step_{step_no:06d}.pt"
            save_dcld_checkpoint(periodic_ckpt, step_no)
            step_log["event"] = "checkpoint"
            step_log["checkpoint_path"] = str(periodic_ckpt)
            emit_event(
                "[DCLD TRAIN] "
                f"checkpoint step {step_no}/{max_batches} ({progress_pct:.2f}%) "
                f"loss={step_log['total']:.6g} "
                f"cond={step_log['condition_mse']:.6g} "
                f"act={step_log['action_l1']:.6g} "
                f"elapsed={format_duration(elapsed_seconds)} "
                f"eta={format_duration(eta_seconds)} "
                f"path={periodic_ckpt}"
            )
        should_persist_log = (
            debug_mode
            or step_no == 1
            or is_checkpoint_step
            or step_no == max_batches
            or (log_interval > 0 and step_no % log_interval == 0)
        )
        if should_persist_log:
            step_log.setdefault("event", "periodic" if log_interval > 0 and step_no % log_interval == 0 else "metric")
            persist_step_event(step_log)
        should_wandb_log = (
            debug_mode
            or step_no == 1
            or is_checkpoint_step
            or step_no == max_batches
            or step_no % wandb_log_interval == 0
        )
        if should_wandb_log:
            log_step_to_wandb(step_log)

    if progress_bar is not None:
        progress_bar.close()

    grad = grad_summary(teacher, dcld_core)
    gradient_sanity = {
        **grad,
        "teacher_trainable_params": int(sum(p.numel() for p in teacher.parameters() if p.requires_grad)),
        "dcld_trainable_params": int(sum(p.numel() for p in dcld_core.parameters() if p.requires_grad)),
        "passed": grad["teacher_grad_nonzero_tensors"] == 0
        and grad["dcld_grad_nonzero_tensors"] > 0
        and grad["fast_visual_delta_encoder_grad_nonzero_tensors"] > 0
        and grad["dcld_updater_grad_nonzero_tensors"] > 0,
    }
    write_json(output_dir / "gradient_sanity.json", gradient_sanity)
    train_metrics = {
        "debug_mode": debug_mode,
        "max_batches": max_batches,
        "steps_run": steps_run,
        "log_interval": log_interval,
        "save_interval": save_interval,
        "progress_path": str(progress_path),
        "latest_metrics_path": str(latest_metrics_path),
        "logged_steps": logs,
        "gradient_sanity": gradient_sanity,
        "trainable_param_count": int(sum(p.numel() for p in dcld_core.parameters() if p.requires_grad)),
        "frozen_param_count": int(sum(p.numel() for p in teacher.parameters() if not p.requires_grad)),
    }
    write_json(output_dir / "train_smoke_metrics.json", train_metrics)
    write_json(output_dir / "train_metrics.json", train_metrics)

    ckpt_name = "dcld_smoke_latest.pt" if debug_mode else f"dcld_step_{steps_run:06d}.pt"
    ckpt_path = ckpt_dir / ckpt_name
    with torch.no_grad():
        reference_out = dcld_core.update_latent(last_batch["c_prev"], last_batch["delta_obs"]).latent.detach()
    save_dcld_checkpoint(ckpt_path, steps_run)
    final_log = dict(last_step_log or {"step": int(steps_run), "max_steps": int(max_batches)})
    final_log["event"] = "final_checkpoint"
    final_log["checkpoint_path"] = str(ckpt_path)
    persist_step_event(final_log)
    log_step_to_wandb(final_log)
    emit_event(f"[DCLD TRAIN] saved final checkpoint: {ckpt_path}")

    reload_core = DCLDCore(
        latent_dim=latent_dim,
        delta_dim=args.delta_dim,
        hidden_dim=args.hidden_dim,
        dynamics_type=args.dynamics_type,
        rank_dim=args.rank_dim,
        gate_mode=args.gate_mode,
        gate_bias=args.gate_bias,
        use_post_layernorm=args.use_post_layernorm,
    ).to(device)
    with torch.no_grad():
        _ = reload_core.update_latent(last_batch["c_prev"], last_batch["delta_obs"])
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    reload_core.load_state_dict(state["dcld_state_dict"])
    with torch.no_grad():
        reload_out = reload_core.update_latent(last_batch["c_prev"], last_batch["delta_obs"]).latent.detach()
    max_abs_diff = float((reference_out - reload_out).abs().max().item())
    reload_check = {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_type": state.get("checkpoint_type"),
        "loaded": True,
        "max_abs_diff_same_input": max_abs_diff,
        "allclose_1e_6": bool(torch.allclose(reference_out, reload_out, atol=1e-6, rtol=1e-6)),
        "config": state.get("config"),
    }
    write_json(output_dir / "checkpoint_reload_check.json", reload_check)
    write_text(
        output_dir / "cache_backed_trainer_report.md",
        "\n".join(
            [
                "# Cache-Backed DCLD Trainer Report",
                "",
                f"- teacher_cache: `{args.teacher_cache}`",
                f"- output: `{output_dir}`",
                f"- debug_mode: `{debug_mode}`",
                f"- batches_run: `{steps_run}`",
                f"- progress_jsonl: `{progress_path}`",
                f"- latest_metrics: `{latest_metrics_path}`",
                f"- teacher_trainable_params: `{gradient_sanity['teacher_trainable_params']}`",
                f"- dcld_trainable_params: `{gradient_sanity['dcld_trainable_params']}`",
                f"- gradient_sanity_passed: `{gradient_sanity['passed']}`",
                f"- checkpoint: `{ckpt_path}`",
                f"- checkpoint_reload_allclose_1e_6: `{reload_check['allclose_1e_6']}`",
                "",
                "The checkpoint stores only DCLD/adaptor state, not the SimVLA teacher.",
            ]
        ),
    )
    if wandb_run is not None:
        wandb_run.summary["steps_run"] = steps_run
        wandb_run.summary["final_checkpoint"] = str(ckpt_path)
        wandb_run.summary["checkpoint_reload_allclose_1e_6"] = reload_check["allclose_1e_6"]
        wandb_run.finish()
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--condition-weight", type=float, default=0.05)
    parser.add_argument("--action-weight", type=float, default=0.1)
    parser.add_argument("--smooth-weight", type=float, default=0.001)
    parser.add_argument("--cosine-weight", type=float, default=0.05)
    parser.add_argument("--dynamics-type", choices=["dense", "low_rank"], default="dense")
    parser.add_argument("--delta-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--rank-dim", type=int, default=64)
    parser.add_argument("--gate-mode", choices=["dense", "scalar", "token"], default="dense")
    parser.add_argument("--gate-bias", type=float, default=-4.0)
    parser.add_argument("--use-post-layernorm", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--action-steps", type=int, default=2)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Sparse JSONL/stdout event interval. 0 logs only first step, checkpoints, and final.",
    )
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--tqdm-mininterval", type=float, default=1.0)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "gnaroshi-simvla-dcld"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE"))
    parser.add_argument("--wandb-log-interval", type=int, default=1000)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return run_cache_backed_distill(args)


if __name__ == "__main__":
    raise SystemExit(main())
