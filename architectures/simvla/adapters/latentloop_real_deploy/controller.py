"""SimVLA baseline and LatentLoop controller for the copied UR5e runtime."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from .bootstrap import configure_model_imports
from .contracts import DeploymentContract


def encode_robot_state(
    robot_state: Mapping[str, Any],
    encoding: str,
    tcp_orientation: str,
    gripper_max_opening_m: float = 0.04,
) -> np.ndarray:
    pose = np.asarray(robot_state["pose6d"], dtype=np.float32).reshape(-1)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("robot_state.pose6d must be a finite six-vector")
    if tcp_orientation == "axis_angle_radians":
        orientation = np.asarray(
            robot_state["tcp_rotvec"], dtype=np.float32
        ).reshape(-1)
        if orientation.shape != (3,) or not np.isfinite(orientation).all():
            raise ValueError("robot_state.tcp_rotvec must be a finite three-vector")
        pose = np.concatenate((pose[:3], orientation))
    elif tcp_orientation != "euler_xyz_radians":
        raise ValueError(f"Unsupported TCP orientation: {tcp_orientation}")
    position = np.asarray(
        robot_state["gripper_position"], dtype=np.float32
    ).reshape(-1)
    open_state = np.asarray(
        robot_state["gripper_open_state"], dtype=np.float32
    ).reshape(-1)
    if position.shape != (1,) or open_state.shape != (1,):
        raise ValueError("gripper state and position must each be scalar arrays")
    if encoding != "opposed_finger_positions":
        raise ValueError(f"Unsupported state encoding: {encoding}")
    opening = (1.0 - float(np.clip(position[0], 0.0, 1.0))) * float(
        gripper_max_opening_m
    )
    state = np.concatenate((pose, [opening, -opening]))
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("encoded SimVLA proprioception must be a finite eight-vector")
    return state.astype(np.float32, copy=False)


def convert_model_action(
    action: np.ndarray,
    *,
    clip_abs: float,
    positive_gripper_means: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("SimVLA action must be a finite seven-vector")
    if float(np.max(np.abs(value[:6]))) > float(clip_abs):
        raise RuntimeError(
            "SimVLA pose action exceeded the reviewed normalized action bound: "
            f"max={float(np.max(np.abs(value[:6]))):.6f} bound={float(clip_abs):.6f}"
        )
    # The real gripper consumes only the sign (open/close). Flow decoding can
    # overshoot the supervised {-1, +1} targets slightly, so saturate this
    # categorical channel while retaining the hard bound for Cartesian motion.
    gripper = float(np.clip(value[6], -float(clip_abs), float(clip_abs)))
    if positive_gripper_means == "close":
        gripper = -gripper
    elif positive_gripper_means != "open":
        raise ValueError(f"Unsupported gripper convention: {positive_gripper_means}")
    return value[:3].copy(), value[3:6].copy(), gripper


class SimVLARealController:
    """Legacy-GUI-compatible controller with a fresh H=10 chunk every R=5 actions."""

    def __init__(
        self,
        *,
        contract: DeploymentContract,
        deployment_method: str,
        policy: Any,
        device: torch.device,
    ) -> None:
        if deployment_method not in {
            "baseline",
            "condition_loop",
            "latentloop",
            "vla_cache_full",
            "vla_cache",
        }:
            raise ValueError(
                "deployment_method must be baseline, condition_loop, latentloop, "
                "vla_cache_full, or vla_cache"
            )
        self.contract = contract
        self.deployment_method = deployment_method
        self.policy = policy
        self.device = device
        self.use_ensembling = False
        self.rollout_index = -1
        self.session_dir: Path | None = None
        self.step_records: list[dict[str, Any]] = []
        self.control_command_monotonic_s: list[float] = []
        self.args = SimpleNamespace(
            resume_from_checkpoint=str(
                contract.artifacts["real_action_transformer"].path
            ),
            vit_checkpoint_path=None,
            real_eval_max_steps=int(contract.runtime["max_steps"]),
            eval_frame_stride=1,
            eval_frame_offset=0,
            skip_action_blend_ratio=1.0,
            skip_action_blend_offset=0,
            skip_action_direct=False,
        )
        self.reset(write_previous=False)

    @classmethod
    def from_contract(
        cls,
        contract: DeploymentContract,
        *,
        deployment_method: str,
        device: str | torch.device,
    ) -> "SimVLARealController":
        configure_model_imports()
        from architectures.simvla.adapters.latentloop.native_v0_runtime import freeze_module
        from architectures.simvla.adapters.real_world_training.model_io import (
            load_exact_official_model,
        )
        from architectures.simvla.adapters.real_world_training.updater_io import (
            load_real_updater,
        )
        from .policy import (
            ConditionLoopSimVLARealPolicy,
            FullSimVLARealPolicy,
            LatentLoopSimVLARealPolicy,
        )

        target_device = torch.device(device)
        base_directory = contract.artifacts["official_base_model_directory"].path
        processor_directory = contract.artifacts["processor_directory"].path
        model, processor, loading_report = load_exact_official_model(
            model_directory=base_directory,
            processor_directory=processor_directory,
            norm_stats=contract.artifacts["norm_stats"].path,
            real_action_checkpoint=contract.artifacts["real_action_transformer"].path,
            device=target_device,
            freeze_vlm=True,
            freeze_action_transformer=True,
        )
        freeze_module(model)

        if str(model.action_mode) != str(contract.policy["action_mode"]):
            raise ValueError(
                f"Model action mode {model.action_mode!r} does not match deployment contract"
            )
        if int(model.num_actions) != int(contract.policy["action_horizon"]):
            raise ValueError("Model action horizon does not match H=10 contract")

        common = {
            "model": model,
            "processor": processor,
            "device": target_device,
            "suite": "real_world",
            "task_id": 0,
            "trial_id": 0,
            "action_noise_seed_base": int(contract.runtime["seed"]),
        }
        if deployment_method == "baseline":
            policy = FullSimVLARealPolicy(**common)
        elif deployment_method in {"condition_loop", "latentloop"}:
            baseline_sha256 = contract.artifacts["real_action_transformer"].sha256
            condition, _ = load_real_updater(
                contract.artifacts["condition_updater"].path,
                kind="condition",
                device=target_device,
                expected_baseline_sha256=baseline_sha256,
            )
            condition_common = {
                "adapter": condition,
                "checkpoint_id": str(
                    contract.artifacts["real_action_transformer"].path
                ),
                **common,
            }
            if deployment_method == "condition_loop":
                policy = ConditionLoopSimVLARealPolicy(**condition_common)
            else:
                generation, _ = load_real_updater(
                    contract.artifacts["generation_updater"].path,
                    kind="generation",
                    device=target_device,
                    expected_baseline_sha256=baseline_sha256,
                )
                policy = LatentLoopSimVLARealPolicy(
                    generation_updater=generation,
                    **condition_common,
                )
        else:
            from architectures.simvla.adapters.vla_cache.policy import (
                VLACacheSimVLARealPolicy,
            )

            policy = VLACacheSimVLARealPolicy(
                enable_reuse=deployment_method == "vla_cache",
                **common,
            )
        from architectures.simvla.adapters.real_world_training.dataset import (
            build_real_image_transform,
        )

        # The action head and both updaters were trained from the FP32 condition
        # cache produced with bicubic 224-to-384 resizing. Keep every real-world
        # deployment method on that exact shared preprocessing implementation.
        policy.image_transform = build_real_image_transform(training=False)
        controller = cls(
            contract=contract,
            deployment_method=deployment_method,
            policy=policy,
            device=target_device,
        )
        controller.exact_initialization = loading_report
        return controller

    def attach_session_dir(self, path: str | Path) -> None:
        self.session_dir = Path(path).expanduser().resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def reset(self, write_previous: bool = True) -> None:
        if write_previous and self.step_records and self.session_dir is not None:
            self.write_runtime_summary()
        self.rollout_index += 1
        self.policy.task_id = 0
        self.policy.trial_id = self.rollout_index
        self.policy.reset()
        self.step_records = []
        self.control_command_monotonic_s = []

    def record_control_command(self, timestamp: float | None = None) -> None:
        self.control_command_monotonic_s.append(
            time.perf_counter() if timestamp is None else float(timestamp)
        )

    @property
    def needs_policy_query(self) -> bool:
        return not bool(self.policy.action_queue)

    @torch.inference_mode()
    def forward(
        self,
        observation: Mapping[str, Any],
        *,
        include_info: bool = False,
        timestep: int = 0,
        record_step: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float, Any]:
        started = time.perf_counter()
        images = observation.get("color_image")
        if not isinstance(images, (list, tuple)) or len(images) < 2:
            raise ValueError("color_image must contain exterior and wrist RGB images")
        state = encode_robot_state(
            observation["robot_state"],
            str(self.contract.state["encoding"]),
            str(self.contract.state["tcp_orientation"]),
            float(self.contract.state["gripper_max_opening_m"]),
        )
        output = self.policy.act(
            np.asarray(images[0]),
            np.asarray(images[1]),
            state,
            str(observation["language_instruction"]),
        )
        target_pos, target_euler, target_gripper = convert_model_action(
            output.action,
            clip_abs=float(self.contract.action["clip_abs"]),
            positive_gripper_means=str(
                self.contract.action["model_positive_gripper_means"]
            ),
        )
        record = {
            "rollout_index": self.rollout_index,
            "timestep": int(timestep),
            "deployment_method": self.deployment_method,
            "policy_ms": (time.perf_counter() - started) * 1000.0,
            "action": output.action.tolist(),
            **output.info,
        }
        if record_step:
            self.step_records.append(record)
            if self.session_dir is not None:
                path = self.session_dir / f"policy_steps_rollout_{self.rollout_index:03d}.jsonl"
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        terminal = {"is_terminal": -1.0, "record": record} if include_info else -1.0
        return target_pos, target_euler, target_gripper, terminal

    def deployment_metadata(self) -> dict[str, Any]:
        metadata = {
            "deployment_id": self.contract.deployment_id,
            "deployment_method": self.deployment_method,
            "manifest": str(self.contract.path),
            "exact_initialization": getattr(self, "exact_initialization", None),
            "policy_contract": dict(self.contract.policy),
            "state_contract": dict(self.contract.state),
            "action_contract": dict(self.contract.action),
            "artifact_sha256": {
                name: artifact.sha256
                for name, artifact in self.contract.artifacts.items()
                if artifact.sha256
            },
        }
        cache_runtime = getattr(self.policy, "vla_cache", None)
        if cache_runtime is not None:
            metadata["vla_cache_contract"] = cache_runtime.config.to_dict()
            metadata["vla_cache_reuse_enabled"] = bool(cache_runtime.enable_reuse)
        return metadata

    def runtime_summary(self) -> dict[str, Any]:
        latency = np.asarray(
            [record["policy_ms"] for record in self.step_records], dtype=np.float64
        )
        command_times = np.asarray(self.control_command_monotonic_s, dtype=np.float64)
        command_intervals_ms = (
            np.diff(command_times) * 1000.0
            if command_times.size > 1
            else np.asarray([], dtype=np.float64)
        )
        control_period_ms = 1000.0 / float(
            self.contract.runtime["control_frequency_hz"]
        )
        return {
            **self.deployment_metadata(),
            "rollout_index": self.rollout_index,
            "steps": len(self.step_records),
            "policy_latency_ms": {
                "mean": float(latency.mean()) if latency.size else 0.0,
                "p95": float(np.percentile(latency, 95)) if latency.size else 0.0,
                "max": float(latency.max()) if latency.size else 0.0,
            },
            "policy_metrics": self.policy.metrics.summary(),
            "policy_counters": dict(self.policy.metrics.counters),
            "query_trace": list(getattr(self.policy, "query_trace", [])),
            "control_period_ms": control_period_ms,
            "control_commands": int(command_times.size),
            "control_command_interval_ms": {
                "mean": float(command_intervals_ms.mean())
                if command_intervals_ms.size
                else 0.0,
                "p95": float(np.percentile(command_intervals_ms, 95))
                if command_intervals_ms.size
                else 0.0,
                "max": float(command_intervals_ms.max())
                if command_intervals_ms.size
                else 0.0,
                "deadline_misses": int(
                    np.count_nonzero(command_intervals_ms > control_period_ms)
                ),
            },
        }

    def write_runtime_summary(self) -> Path | None:
        if self.session_dir is None:
            return None
        path = self.session_dir / f"deployment_runtime_rollout_{self.rollout_index:03d}.json"
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(self.runtime_summary(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return path
