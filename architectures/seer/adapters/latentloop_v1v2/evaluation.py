"""Source-locked online evaluation hook for LatentLoop V1 fixed-K."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data_runtime import build_frozen_teacher, build_seer_args, seer_upstream_context
from .runtime import load_v1_checkpoint, sha256_file


def _validate_episode_manifest(path: Path, tasks: int, episodes_per_task: int, seed: int) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != tasks * episodes_per_task:
        raise RuntimeError(f"episode manifest has {len(rows)} rows, expected {tasks * episodes_per_task}")
    keys = {f"{int(row['task_id'])}:{int(row['episode_id'])}:{int(row['seed'])}" for row in rows}
    expected = {
        f"{task}:{episode}:{seed}"
        for task in range(tasks)
        for episode in range(episodes_per_task)
    }
    if keys != expected:
        raise RuntimeError("episode manifest differs from the fixed task/episode/seed grid")
    return keys


def _validate_saved_episode_keys(path: Path, expected: set[str]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    observed = {
        f"{int(row['task_id'])}:{int(row['episode_id'])}:{int(row['seed'])}" for row in rows
    }
    if len(rows) != len(expected) or observed != expected:
        raise RuntimeError("saved online episode keys differ from the canonical manifest")


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _wrapper_class(base_wrapper):
    class V1FixedModelWrapper(base_wrapper):
        """Replace only the locked V0 skip update; preserve Seer execution."""

        def _update_from_lrnode_cache(
            self,
            image_x,
            gripper,
            state,
            use_zero_delta=False,
            compute_hold_action=False,
            commit_cache=True,
            decode_action=True,
            timestep=None,
        ):
            if use_zero_delta:
                raise RuntimeError("V1 fixed-K scientific row does not support zero-delta ablation")
            if compute_hold_action:
                raise RuntimeError("V1 fixed-K scientific row does not enable shadow hold diagnostics")
            if self.lrnode_cached_latent is None:
                raise RuntimeError("V1 skip requested before Full Seer initialized the cache")
            if self.lrnode_cached_env_action is None:
                raise RuntimeError("V1 skip requires the action actually executed at the preceding step")
            if timestep is None:
                raise ValueError("V1 observation-conditioned update requires timestep")
            base_model = self._base_model()
            adapter = base_model.latentloop_plan_adapter
            if str(getattr(adapter, "mode", "")) != "v1_transition":
                raise RuntimeError("the attached online adapter is not V1 transition")
            age = self.lrnode_cached_age + 1
            executed_action = torch.as_tensor(
                self.lrnode_cached_env_action,
                device=self.lrnode_cached_latent.device,
                dtype=self.lrnode_cached_latent.dtype,
            ).reshape(1, 1, 7)

            _sync_cuda()
            fast_start = time.perf_counter()
            observation_delta = adapter._delta(
                self.lrnode_cached_image_primary[:, 0],
                self.lrnode_cached_image_wrist[:, 0],
                image_x[:, 0],
                gripper[:, 0],
                self.lrnode_cached_state[:, 0],
                state[:, 0],
            )
            feature = adapter.conditioner(observation_delta, executed_action, 1)
            _sync_cuda()
            fast_ms = (time.perf_counter() - fast_start) * 1000.0

            _sync_cuda()
            update_start = time.perf_counter()
            z_next = adapter.dynamics(
                self.lrnode_cached_latent,
                feature,
                dt=1.0,
                age=float(age),
            )
            _sync_cuda()
            update_ms = (time.perf_counter() - update_start) * 1000.0

            action_seq = None
            diagnostics = None
            if decode_action:
                _sync_cuda()
                head_start = time.perf_counter()
                diagnostics = base_model.decode_action_diagnostics_from_latent(z_next)
                action_seq = torch.cat(
                    (diagnostics["arm"], diagnostics["gripper_probability"]), dim=-1
                )
                _sync_cuda()
                head_ms = (time.perf_counter() - head_start) * 1000.0
            else:
                head_ms = 0.0

            update = getattr(adapter.dynamics, "last_update", None)
            gate = getattr(adapter.dynamics, "last_gate", None)
            primary_change = (
                image_x[:, 0].detach().float()
                - self.lrnode_cached_image_primary[:, 0].detach().float()
            ).abs()
            wrist_change = (
                gripper[:, 0].detach().float()
                - self.lrnode_cached_image_wrist[:, 0].detach().float()
            ).abs()
            proprio_change = (
                state[:, 0].detach().float()
                - self.lrnode_cached_state[:, 0].detach().float()
            )
            debug: dict[str, Any] = {
                "cache_age": age,
                "skip_age": age,
                "feature_source_step": int(timestep),
                "feedback_source": "current",
                "time_shift_initialized_with_zero": 0,
                "fast_encoder_called": 1,
                "lrnode_update_called": 1,
                "action_head_called": int(decode_action),
                "observation_conditioned_update_called": 1,
                "zero_feature_update_called": 0,
                "observation_cache_advanced": int(commit_cache),
                "fast_encoder_ms": fast_ms,
                "node_update_ms": update_ms,
                "action_head_ms": head_ms,
                "gate_mean": float(gate.detach().float().mean().item()) if gate is not None else 0.0,
                "gate_max": float(gate.detach().float().max().item()) if gate is not None else 0.0,
                "u_delta_norm": float(feature.detach().float().norm(dim=-1).mean().item()),
                "image_diff_primary_l1": float(primary_change.mean().item()),
                "image_diff_wrist_l1": float(wrist_change.mean().item()),
                "proprio_delta_l2": float(proprio_change.norm(dim=-1).mean().item()),
                "update_norm": float(update.detach().float().norm(dim=-1).mean().item()) if update is not None else 0.0,
                "z_norm": float(z_next.detach().float().norm(dim=-1).mean().item()),
                "z_pred": z_next.detach(),
                "z_prev": self.lrnode_cached_latent.detach(),
                "u_delta": feature.detach(),
                "lrnode_update": None if update is None else update.detach(),
                "lrnode_gate": None if gate is None else gate.detach(),
                "v1_executed_action_condition": executed_action.detach(),
                "v1_direct_transition_called": 0,
            }
            if action_seq is not None and diagnostics is not None:
                debug.update(
                    action_pred=action_seq.detach(),
                    action_pred_gripper_logit=diagnostics["gripper_logit"].detach(),
                    action_pred_gripper_probability=diagnostics[
                        "gripper_probability"
                    ].detach(),
                )
            if commit_cache:
                self.lrnode_cached_latent = z_next.detach()
                self.lrnode_cached_image_primary = image_x.detach()
                self.lrnode_cached_image_wrist = gripper.detach()
                self.lrnode_cached_state = state.detach()
                self.lrnode_cached_age = age
            return action_seq, debug

        def step(self, obs, goal, timestep, frames=None, video_stride: int = 1):
            action = super().step(
                obs, goal, timestep, frames=frames, video_stride=video_stride
            )
            # Base V0 stores this only after Full steps. V1 conditions every
            # transition on the command that the environment actually received.
            self._cache_executed_env_action(action)
            return action

        def get_lrnode_stats(self):
            payload = super().get_lrnode_stats()
            payload.update(
                v1_transition_calls=int(self.lrnode_update_calls),
                v1_direct_transition_calls=0,
                v1_action_generator_calls=int(self.num_policy_steps),
                v1_executed_action_conditioning=True,
            )
            return payload

    V1FixedModelWrapper.__name__ = "V1FixedModelWrapper"
    return V1FixedModelWrapper


def run_evaluation(args) -> None:
    if args.mode != "v1_fixed":
        raise RuntimeError("the reviewed V1 runtime handles only fixed-K V1")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4 or local_rank not in range(4):
        raise RuntimeError("V1 evaluation requires exactly four visible GPUs")
    expected_episode_keys = _validate_episode_manifest(
        args.episode_manifest, args.tasks, args.episodes_per_task, args.seed
    )
    os.environ.update(
        LIBERO_GL_BACKEND="osmesa",
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        LIBERO_GL_REQUIRE_ACTUAL="1",
        EVAL_CONTROL_HZ="20",
        EVAL_NUM_TASKS=str(args.tasks),
        EVAL_NUM_EPISODES_PER_TASK=str(args.episodes_per_task),
        EVAL_SCALE_MAX_STEPS_WITH_HZ="1",
        LOG_DIR=str(args.output_root),
        RUN_NAME="latentloop_v1_fixed_k4",
        CKPT_TAG="teacher33_v1_selected",
        BASELINE_CKPT_ID="33",
        OURS_CKPT_ID="v1_selected",
        SAVE_VIDEO="0",
        SAVE_VIDEO_SUCC="0",
        SAVE_VIDEO_FAIL="0",
        SAVE_VIDEO_ALL_RANKS="0",
    )
    repo_root = Path(__file__).resolve().parents[4]
    for path in (str(repo_root), str(args.libero_path)):
        if path not in sys.path:
            sys.path.insert(0, path)
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    output_exists = torch.tensor(
        [int(args.output_root.exists())], device=device, dtype=torch.int32
    )
    dist.all_reduce(output_exists, op=dist.ReduceOp.MAX)
    if int(output_exists.item()):
        raise FileExistsError(args.output_root)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = False
    try:
        with seer_upstream_context(repo_root):
            import clip
            from utils import eval_utils_libero

            seer_args = build_seer_args(
                output_root=args.output_root,
                dataset_root=repo_root,
                vit_checkpoint=args.vit_checkpoint,
                libero_path=args.libero_path,
                batch_size=64,
                workers=16,
                rank=rank,
                world_size=world_size,
                seed=args.seed,
            )
            overrides = {
                "phase": "evaluate",
                "finetune_type": "libero_10",
                "sequence_length": 7,
                "libero_eval_max_steps": args.max_steps,
                "eval_libero_ensembling": True,
                "ensembling_temp": args.temporal_ensemble_temperature,
                "use_lrnode_latent_update": 1,
                "lrnode_eval_skip_full_forward": 1,
                "lrnode_query_interval": args.query_interval,
                "lrnode_train_protocol": "adapter",
                "lrnode_eval_step_log": 1,
                "lrnode_eval_shadow_full_forward": 0,
                "lrnode_eval_profile_full_action_head": 1,
                "lrnode_eval_refresh_policy": "periodic",
                "lrnode_eval_ablation_mode": "stepwise",
                "lrnode_mechanism_trace": 0,
                "lrnode_trace_save_latents": 0,
                "latentloop_segment_grid_enable": 0,
                "latentloop_feedback_source": "current",
                "latentloop_plan_adapter_mode": "v1_transition",
                "precision": "fp32",
                "bf16_module": "vision_encoder",
                "rank": rank,
                "world_size": world_size,
                "local_rank": local_rank,
            }
            for name, value in overrides.items():
                setattr(seer_args, name, value)
            model, _, load_report = build_frozen_teacher(
                seer_args, device, args.teacher, args.adapter_init
            )
            v1_load = load_v1_checkpoint(model, args.checkpoint)
            model.eval()
            model._init_model_type()
            ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
            eval_utils_libero.ModelWrapper = _wrapper_class(eval_utils_libero.ModelWrapper)
            if rank == 0:
                args.output_root.mkdir(parents=True, exist_ok=False)
                contract = {
                    "schema_version": 1,
                    "status": "V1_FIXED_K4_EVAL_STARTED",
                    "teacher_sha256": load_report["teacher_sha256"],
                    "v0_adapter_sha256": load_report["v0_adapter_sha256"],
                    "v1_checkpoint_sha256": v1_load["checkpoint_sha256"],
                    "episode_manifest_sha256": sha256_file(args.episode_manifest),
                    "world_size": world_size,
                    "query_interval": args.query_interval,
                    "executed_action_conditioning": True,
                    "direct_path_online": False,
                    "existing_action_generator_shared": True,
                }
                (args.output_root / "v1_eval_contract.json").write_text(
                    json.dumps(contract, indent=2) + "\n", encoding="utf-8"
                )
            dist.barrier()
            eval_utils_libero.eval_one_epoch_libero_ddp(
                args=seer_args,
                model=ddp_model,
                image_processor=model.image_processor,
                tokenizer=clip,
            )
        dist.barrier()
        if rank == 0:
            episode_csv = args.output_root / "analysis/eval_episode_metrics.csv"
            summary = args.output_root / "analysis/eval_summary.json"
            if not episode_csv.is_file() or not summary.is_file():
                raise RuntimeError("V1 evaluation did not produce canonical analysis artifacts")
            _validate_saved_episode_keys(episode_csv, expected_episode_keys)
            completion = {
                "status": "V1_FIXED_K4_EVAL_COMPLETE",
                "episodes": len(expected_episode_keys),
                "episode_manifest_match": True,
                "summary_sha256": sha256_file(summary),
                "episode_metrics_sha256": sha256_file(episode_csv),
            }
            (args.output_root / "completion.json").write_text(
                json.dumps(completion, indent=2) + "\n", encoding="utf-8"
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
