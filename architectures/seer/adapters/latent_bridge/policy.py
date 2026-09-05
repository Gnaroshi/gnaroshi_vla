"""External Seer policy wrapper for official-style Latent Bridge inference."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_bridge_checkpoint, validate_bridge_runtime_provenance
from .dataset import BridgeTransition, StreamingBridgeTransitionWriter
from .hooks import SeerBoundaryCapture
from .layout import SeerTokenLayout
from .provenance import expected_base_checkpoint_sha256, require_file_hash, sha256_file
from .rendering import configured_renderer_backend


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError(f"{name} must be a boolean flag, got {value!r}")
    return value in {"1", "true"}


class LatentBridgePolicyMixin:
    """Replace only Seer's skipped full-forward path.

    Full-refresh calls, preprocessing, temporal ensembling, and environment
    action execution stay in the upstream ``ModelWrapper`` implementation.
    """

    def _latent_bridge_initialize(self) -> None:
        self._latent_bridge_renderer = configured_renderer_backend()
        checkpoint_path = Path(os.environ["SEER_LATENT_BRIDGE_CHECKPOINT"])
        bridge, payload = load_bridge_checkpoint(checkpoint_path, map_location="cpu")
        self._latent_bridge_provenance = validate_bridge_runtime_provenance(payload)
        self._latent_bridge_base_checkpoint_sha256 = require_file_hash(
            os.environ["SEER_LATENT_BRIDGE_BASE_CHECKPOINT"],
            expected_base_checkpoint_sha256(),
            "Latent Bridge runtime base Seer checkpoint",
        )
        self._latent_bridge_payload = payload
        self._latent_bridge_config = bridge.config
        self._latent_bridge_layout = SeerTokenLayout.from_model(self._base_model())
        if bridge.config.target_seq_len != self._latent_bridge_layout.action_tokens:
            raise RuntimeError("bridge target length does not match Seer action-token count")

        self._latent_bridge_precision = os.environ.get(
            "SEER_LATENT_BRIDGE_PRECISION", "bf16"
        ).lower()
        if self._latent_bridge_precision not in {"bf16", "fp32"}:
            raise ValueError("SEER_LATENT_BRIDGE_PRECISION must be bf16 or fp32")
        self._latent_bridge_dtype = (
            torch.bfloat16 if self._latent_bridge_precision == "bf16" else torch.float32
        )
        bridge = bridge.to(device=self.device, dtype=self._latent_bridge_dtype).eval()
        bridge.requires_grad_(False)
        self._latent_bridge_eager = bridge
        self._latent_bridge_compile = _env_flag("SEER_LATENT_BRIDGE_COMPILE", "1")
        self._latent_bridge_compile_mode = (
            "max-autotune" if self._latent_bridge_compile else "disabled"
        )
        self._latent_bridge = (
            torch.compile(bridge, mode="max-autotune")
            if self._latent_bridge_compile
            else bridge
        )
        self._latent_bridge_compile_warmup_seconds = 0.0
        if self._latent_bridge_compile:
            dummy = (
                torch.zeros(1, bridge.config.target_seq_len, bridge.config.feature_dim,
                            device=self.device, dtype=self._latent_bridge_dtype),
                torch.zeros(1, bridge.config.stable_seq_len, bridge.config.feature_dim,
                            device=self.device, dtype=self._latent_bridge_dtype),
                torch.zeros(1, bridge.config.state_dim, device=self.device,
                            dtype=self._latent_bridge_dtype),
                torch.zeros(1, bridge.config.action_dim, device=self.device,
                            dtype=self._latent_bridge_dtype),
            )
            self._sync_cuda()
            warmup_t0 = time.perf_counter()
            # Match the evaluator's no_grad context so torch.compile does not
            # recompile on the first timed bridge call.
            with torch.no_grad():
                for _ in range(3):
                    self._latent_bridge(*dummy)
            self._sync_cuda()
            self._latent_bridge_compile_warmup_seconds = time.perf_counter() - warmup_t0

        capture_layers = (
            ()
            if bridge.config.stable_layer == "ln_f"
            else (int(bridge.config.stable_layer.removeprefix("block_")),)
        )
        self._latent_bridge_capture = SeerBoundaryCapture(
            self.model, layer_indices=capture_layers
        )
        self._latent_bridge_capture.__enter__()
        self._latent_bridge_stable_context = None
        self._latent_bridge_previous_executed_action = None
        self._latent_bridge_pending_input = None
        self._latent_bridge_episode_transitions = []
        self._latent_bridge_dagger_writer = None
        dagger_output = os.environ.get("SEER_LATENT_BRIDGE_DAGGER_OUTPUT", "").strip()
        if dagger_output:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            root = Path(dagger_output)
            root.mkdir(parents=True, exist_ok=True)
            self._latent_bridge_dagger_writer = StreamingBridgeTransitionWriter(
                root / f"dagger_transitions_rank{rank}.h5"
            )

        self._latent_bridge_calls = 0
        self._latent_bridge_latency_ms = 0.0
        self._latent_bridge_peak_memory_bytes = 0

        # Set these only after the base constructor's model-resident LR-NODE
        # guard has completed. The frozen public Seer intentionally has no such
        # modules; all bridge behavior lives in this external wrapper.
        self.use_lrnode_latent_update = True
        self.lrnode_eval_skip_full_forward = True

    def reset(self):
        result = super().reset()
        self._latent_bridge_stable_context = None
        self._latent_bridge_previous_executed_action = None
        self._latent_bridge_pending_input = None
        self._latent_bridge_episode_transitions = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return result

    def _cache_full_forward_state(
        self,
        action_latent,
        selected_step,
        image_x,
        gripper,
        state,
        action_tokens=None,
        timestep=None,
    ):
        if action_latent is None:
            raise RuntimeError("Latent Bridge full refresh requires Seer action conditions")
        self._latent_bridge_capture.require_complete()
        config = self._latent_bridge_config
        if config.stable_layer == "ln_f":
            flat = self._latent_bridge_capture.final_output
        else:
            try:
                flat = self._latent_bridge_capture.layer_outputs[config.stable_layer]
            except KeyError as exc:
                raise RuntimeError(f"stable layer was not captured: {config.stable_layer}") from exc
        stable = self._latent_bridge_layout.select(
            flat, timestep=selected_step, group=config.stable_token_group
        )
        if stable.shape[1] != config.stable_seq_len:
            raise RuntimeError(
                f"stable sequence mismatch: checkpoint={config.stable_seq_len}, runtime={stable.shape[1]}"
            )
        self._latent_bridge_stable_context = stable.detach()
        self.lrnode_cached_action_tokens = (
            action_tokens.detach() if action_tokens is not None else None
        )
        self.lrnode_cached_image_primary = image_x.detach()
        self.lrnode_cached_image_wrist = gripper.detach()
        self.lrnode_cached_state = state.detach()
        self.lrnode_cached_latent = action_latent[:, selected_step].detach()
        self.lrnode_cached_age = 0
        self.lrnode_last_full_timestep = int(timestep) if timestep is not None else None

    def _cache_executed_env_action(self, action):
        value = np.asarray(action, dtype=np.float32).copy()
        self.lrnode_cached_env_action = value
        self._latent_bridge_previous_executed_action = value

    def _update_from_lrnode_cache(
        self,
        image_x,
        gripper,
        state,
        use_zero_delta=False,
        compute_hold_action=False,
    ):
        if use_zero_delta:
            raise RuntimeError("Latent Bridge does not implement the LR-NODE no-delta ablation")
        if self.lrnode_cached_latent is None or self._latent_bridge_stable_context is None:
            raise RuntimeError("bridge update requested before a full-refresh cache exists")
        if self._latent_bridge_previous_executed_action is None:
            raise RuntimeError("bridge update requested before an executed action exists")

        z_previous = self.lrnode_cached_latent.detach()
        stable = self._latent_bridge_stable_context.detach()
        current_state = state[:, -1].detach()
        previous_action = torch.as_tensor(
            self._latent_bridge_previous_executed_action,
            device=z_previous.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        self._latent_bridge_pending_input = {
            "previous_condition": z_previous.float().cpu().numpy()[0],
            "stable_context": stable.float().cpu().numpy()[0],
            "current_state": current_state.float().cpu().numpy()[0],
            "previous_executed_action": previous_action.float().cpu().numpy()[0],
        }

        bridge_inputs = tuple(
            tensor.to(dtype=self._latent_bridge_dtype)
            for tensor in (z_previous, stable, current_state, previous_action)
        )
        self._sync_cuda()
        bridge_t0 = time.perf_counter()
        predicted = bridge_inputs[0] + self._latent_bridge(*bridge_inputs)
        self._sync_cuda()
        bridge_ms = (time.perf_counter() - bridge_t0) * 1000.0
        action_head_dtype = next(self._base_model().action_decoder.parameters()).dtype
        predicted = predicted.to(dtype=action_head_dtype)

        self._sync_cuda()
        head_t0 = time.perf_counter()
        arm_action, gripper_action = self._base_model().decode_action_from_latent(predicted)
        self._sync_cuda()
        action_head_ms = (time.perf_counter() - head_t0) * 1000.0
        action_sequence = torch.cat([arm_action, gripper_action], dim=-1)

        update_norm = float((predicted - z_previous).float().norm(dim=-1).mean().item())
        debug = {
            "cache_age": self.lrnode_cached_age + 1,
            "skip_age": self.lrnode_cached_age + 1,
            "fast_encoder_called": 0,
            "lrnode_update_called": 1,
            "action_head_called": 1,
            "fast_encoder_ms": 0.0,
            "node_update_ms": bridge_ms,
            "action_head_ms": action_head_ms,
            "gate_mean": 0.0,
            "gate_max": 0.0,
            "u_delta_norm": update_norm,
            "image_diff_primary_l1": 0.0,
            "image_diff_wrist_l1": 0.0,
            "update_norm": update_norm,
            "z_norm": float(predicted.float().norm(dim=-1).mean().item()),
            "z_pred": predicted.detach(),
            "z_hold": z_previous.detach(),
            "action_pred": action_sequence.detach(),
        }
        if compute_hold_action:
            hold_arm, hold_gripper = self._base_model().decode_action_from_latent(z_previous)
            debug["action_hold"] = torch.cat([hold_arm, hold_gripper], dim=-1).detach()

        self.lrnode_cached_latent = predicted.detach()
        self.lrnode_cached_age += 1
        self._latent_bridge_calls += 1
        self._latent_bridge_latency_ms += bridge_ms
        if torch.cuda.is_available():
            self._latent_bridge_peak_memory_bytes = max(
                self._latent_bridge_peak_memory_bytes,
                int(torch.cuda.max_memory_allocated()),
            )
        return action_sequence, debug

    def step(self, obs, goal, timestep, frames=None, video_stride: int = 1):
        self._latent_bridge_capture.clear()
        action = super().step(obs, goal, timestep, frames=frames, video_stride=video_stride)
        record = self.current_step_records[-1]
        if record.get("mode") == "stepwise" and self._latent_bridge_dagger_writer is not None:
            if not self.lrnode_eval_shadow_full_forward:
                raise RuntimeError("DAgger collection requires shadow full forwarding")
            self._latent_bridge_capture.require_complete()
            selected_step = self._selected_step(len(self.img_queue))
            target = self._latent_bridge_layout.select(
                self._latent_bridge_capture.final_output,
                timestep=selected_step,
                group="action",
            )
            if self._latent_bridge_pending_input is None:
                raise RuntimeError("missing bridge input for DAgger transition")
            item = self._latent_bridge_pending_input
            self._latent_bridge_episode_transitions.append(
                BridgeTransition(
                    previous_condition=item["previous_condition"],
                    target_condition=target.detach().float().cpu().numpy()[0],
                    stable_context=item["stable_context"],
                    current_state=item["current_state"],
                    previous_executed_action=item["previous_executed_action"],
                    episode_id="pending",
                    task_id=-1,
                    step=int(timestep),
                    success=-1,
                    source="R1_dagger_f3",
                )
            )
        self._latent_bridge_previous_executed_action = np.asarray(action, dtype=np.float32).copy()
        return action

    def finish_episode(self, task, env, success, steps, args):
        metrics = super().finish_episode(task, env, success, steps, args)
        if self._latent_bridge_dagger_writer is not None:
            episode_id = (
                f"task{int(env.task_id):02d}_trial{int(env.exp_id):03d}_seed{int(args.seed)}"
            )
            for transition in self._latent_bridge_episode_transitions:
                transition.episode_id = episode_id
                transition.task_id = int(env.task_id)
                transition.success = int(success)
                self._latent_bridge_dagger_writer.append(transition)
        self._latent_bridge_episode_transitions = []
        return metrics

    def get_lrnode_stats(self):
        stats = super().get_lrnode_stats()
        stats.update(
            {
                "method": "seer_latent_bridge",
                "bridge_calls": int(self._latent_bridge_calls),
                "avg_bridge_latency_sec": (
                    self._latent_bridge_latency_ms / self._latent_bridge_calls / 1000.0
                    if self._latent_bridge_calls
                    else 0.0
                ),
                "peak_gpu_memory_bytes": int(self._latent_bridge_peak_memory_bytes),
                "bridge_parameters": int(
                    sum(parameter.numel() for parameter in self._latent_bridge_eager.parameters())
                ),
                "bridge_precision": self._latent_bridge_precision,
                "bridge_compile": bool(self._latent_bridge_compile),
                "bridge_compile_mode": self._latent_bridge_compile_mode,
                "bridge_compile_warmup_seconds_excluded": (
                    self._latent_bridge_compile_warmup_seconds
                ),
                "stable_layer": self._latent_bridge_config.stable_layer,
                "stable_token_group": self._latent_bridge_config.stable_token_group,
                "base_checkpoint_sha256": self._latent_bridge_base_checkpoint_sha256,
                "refresh_period": int(self.lrnode_query_interval),
                "action_protocol": (
                    "three-token prediction with temporal ensembling; one executed action"
                ),
                "checkpoint_provenance": self._latent_bridge_provenance,
                "renderer_backend": self._latent_bridge_renderer,
                "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            }
        )
        if self._latent_bridge_dagger_writer is not None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            self._latent_bridge_dagger_writer.close(
                {
                    "stage": "R1_DAgger",
                    "refresh_period": int(self.lrnode_query_interval),
                    "rank": int(rank),
                    "bridge_checkpoint_stage": self._latent_bridge_payload["stage"],
                    "bridge_checkpoint_epoch": self._latent_bridge_payload["epoch"],
                    "bridge_checkpoint_sha256": sha256_file(
                        os.environ["SEER_LATENT_BRIDGE_CHECKPOINT"]
                    ),
                    "base_checkpoint_sha256": self._latent_bridge_base_checkpoint_sha256,
                    "stable_layer": self._latent_bridge_config.stable_layer,
                    "stable_token_group": self._latent_bridge_config.stable_token_group,
                    "renderer": self._latent_bridge_renderer,
                    "action_protocol": "three-token prediction with temporal ensembling; one executed action",
                }
            )
        return stats


def build_latent_bridge_wrapper(base_wrapper):
    class LatentBridgeModelWrapper(LatentBridgePolicyMixin, base_wrapper):
        def __init__(self, *args, **kwargs):
            kwargs["use_lrnode_latent_update"] = 0
            kwargs["lrnode_eval_skip_full_forward"] = 0
            kwargs["lrnode_eval_ablation_mode"] = "stepwise"
            super().__init__(*args, **kwargs)
            self.lrnode_query_interval = int(os.environ["SEER_LATENT_BRIDGE_REFRESH_PERIOD"])
            if self.lrnode_query_interval < 2:
                raise ValueError("Latent Bridge wrapper requires refresh period >= 2")
            self.lrnode_eval_shadow_full_forward = bool(
                os.environ.get("SEER_LATENT_BRIDGE_DAGGER_OUTPUT", "").strip()
            )
            self.lrnode_eval_profile_full_action_head = True
            setattr(self._base_model(), "profile_full_action_head", True)
            self._latent_bridge_initialize()

    return LatentBridgeModelWrapper
