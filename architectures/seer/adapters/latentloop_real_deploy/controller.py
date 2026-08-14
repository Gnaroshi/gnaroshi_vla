"""Real-world controller for full Seer and periodic LatentLoop inference."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import clip
import numpy as np
import torch
from PIL import Image as PILImage


REPO_ROOT = Path(__file__).resolve().parents[4]
SEER_UPSTREAM_ROOT = REPO_ROOT / "architectures" / "seer" / "upstream"
if str(SEER_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SEER_UPSTREAM_ROOT))

from models.seer_model import SeerAgent  # noqa: E402
from utils.arguments_utils import get_parser  # noqa: E402
from utils.data_utils import preprocess_image, preprocess_text_calvin  # noqa: E402
from utils.distributed_utils import init_distributed_device, world_info_from_env  # noqa: E402
from utils.train_utils import get_cast_dtype  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def should_use_latentloop(timestep: int, query_interval: int, has_cache: bool) -> bool:
    if query_interval < 1:
        raise ValueError(f"query_interval must be positive, got {query_interval}")
    return bool(has_cache and timestep % query_interval != 0)


def remove_ddp_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize training checkpoints for the single-GPU inference model."""

    normalized = {}
    for key, value in state.items():
        normalized_key = key.removeprefix("module.")
        if normalized_key in normalized:
            raise ValueError(f"Checkpoint key collision after DDP prefix removal: {key}")
        normalized[normalized_key] = value
    return normalized


def is_allowed_teacher_missing_key(key: str) -> bool:
    """Allow only state intentionally rebuilt outside the real-world checkpoint."""

    return (
        key in {"attention_mask", "image_decoder_position_embedding"}
        or key.startswith("vision_encoder.")
        or key.startswith("clip_model.")
        or key.startswith("lrnode_")
    )


def temporal_ensemble_probability(
    action_sequence: torch.Tensor,
    timestep: int,
    buffer: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    """Preserve Seer's legacy probability-domain temporal ensemble exactly."""

    if action_sequence.ndim != 3 or action_sequence.shape[0] != 1:
        raise ValueError(f"Expected action sequence [1, H, A], got {tuple(action_sequence.shape)}")
    horizon = int(action_sequence.shape[1])
    buffer[timestep : timestep + 1, timestep : timestep + horizon] = action_sequence
    candidates = buffer[:, timestep]
    candidates = candidates[torch.all(candidates != 0, dim=1)]
    if len(candidates) == 0:
        raise RuntimeError(f"Temporal ensemble has no populated action at timestep {timestep}")
    weights = np.exp(-float(temperature) * np.arange(len(candidates)))
    weights /= weights.sum()
    weight_tensor = torch.from_numpy(weights).to(candidates.device).unsqueeze(1)
    return (candidates * weight_tensor).sum(dim=0, keepdim=True), int(len(candidates))


def load_artifact_profile(
    manifest_path: Path,
    teacher_id: int,
    adapter_id: int | None,
) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    teacher = manifest.get("teachers", {}).get(str(teacher_id))
    if teacher is None:
        raise KeyError(f"Teacher {teacher_id} is not defined in {manifest_path}")
    adapter = None
    if adapter_id is not None:
        adapter = teacher.get("adapters", {}).get(str(adapter_id))
        if adapter is None:
            raise KeyError(
                f"Adapter {adapter_id} for teacher {teacher_id} is not defined in {manifest_path}"
            )
    return {
        "schema_version": manifest.get("schema_version"),
        "task": manifest.get("task"),
        "latentloop_architecture": manifest["latentloop_architecture"],
        "vit": manifest["vit"],
        "teacher": teacher,
        "adapter": adapter,
        "teacher_id": int(teacher_id),
        "adapter_id": int(adapter_id) if adapter_id is not None else None,
    }


class LatentLoopSeerController:
    def __init__(
        self,
        *,
        deployment_method: str,
        deployment_control_freq: float,
        adapter_checkpoint: str | None,
        artifact_manifest: str,
        teacher_id: int,
        adapter_id: int | None,
        deployment_profile: str,
    ):
        parser = get_parser(is_eval=True)
        args = parser.parse_args()
        args.local_rank, args.rank, args.world_size = world_info_from_env()
        args.device_id = init_distributed_device(args)
        self.args = args
        self.device_id = args.device_id
        self.deployment_method = str(deployment_method).strip().lower()
        self.target_control_hz = float(deployment_control_freq)
        self.adapter_checkpoint = (
            Path(adapter_checkpoint).expanduser().resolve() if adapter_checkpoint else None
        )
        self.manifest_path = Path(artifact_manifest).expanduser().resolve()
        self.teacher_checkpoint = Path(args.resume_from_checkpoint).expanduser().resolve()
        self.vit_checkpoint = Path(args.vit_checkpoint_path).expanduser().resolve()
        self.teacher_id = int(teacher_id)
        self.adapter_id = int(adapter_id) if adapter_id is not None else None
        self.deployment_profile = self._safe_profile(deployment_profile)
        self.query_interval = int(args.lrnode_query_interval)
        self.use_ensembling = bool(args.eval_libero_ensembling)
        self.ensembling_temp = float(args.ensembling_temp)
        self.history_len = int(args.sequence_length)
        self.action_pred_steps = int(args.action_pred_steps)
        self.real_eval_max_steps = int(args.real_eval_max_steps)
        self.gripper_width = bool(args.gripper_width)
        self.session_dir: Path | None = None
        self.rollout_index = -1
        self._rollout_complete = False

        self._validate_cli_contract()
        self.artifact_profile = load_artifact_profile(
            self.manifest_path, self.teacher_id, self.adapter_id
        )
        self._verify_architecture_contract()
        self.artifact_hashes = self._verify_artifacts()
        self.random_seed(args.seed, args.rank)
        self._setup_model()

        self.cast_dtype = get_cast_dtype(args.precision)
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=clip)
        self.image_process_fn = functools.partial(
            preprocess_image, image_processor=self.model.image_processor
        )
        self.reset(write_previous=False)

    @staticmethod
    def _safe_profile(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
        safe = safe.strip("._")
        if not safe:
            raise ValueError("deployment_profile must contain a filename-safe character")
        return safe

    def _validate_cli_contract(self) -> None:
        required_files = {
            "teacher checkpoint": self.teacher_checkpoint,
            "ViT checkpoint": self.vit_checkpoint,
            "artifact manifest": self.manifest_path,
        }
        if self.deployment_method == "latentloop" and self.adapter_checkpoint is not None:
            required_files["adapter checkpoint"] = self.adapter_checkpoint
        missing = [f"{name}: {path}" for name, path in required_files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing deployment artifacts:\n" + "\n".join(missing))
        if self.deployment_method not in {"baseline", "latentloop"}:
            raise ValueError(
                "deployment_method must be 'baseline' or 'latentloop', got "
                f"{self.deployment_method!r}"
            )
        if not np.isfinite(self.target_control_hz) or self.target_control_hz <= 0:
            raise ValueError(
                f"deployment_control_freq must be positive, got {self.target_control_hz}"
            )
        if self.deployment_method == "baseline":
            if self.adapter_checkpoint is not None or self.adapter_id is not None:
                raise ValueError("Baseline deploy must not load a LatentLoop adapter")
            if bool(self.args.use_lrnode_latent_update):
                raise ValueError("Baseline deploy requires --use_lrnode_latent_update 0")
            if bool(self.args.lrnode_eval_skip_full_forward):
                raise ValueError("Baseline deploy requires --lrnode_eval_skip_full_forward 0")
            if self.query_interval != 1:
                raise ValueError("Baseline deploy requires lrnode_query_interval=1")
        else:
            if self.adapter_checkpoint is None or self.adapter_id is None:
                raise ValueError("LatentLoop deploy requires a teacher-specific adapter")
            if not bool(self.args.use_lrnode_latent_update):
                raise ValueError("LatentLoop deploy requires --use_lrnode_latent_update 1")
            if not bool(self.args.lrnode_eval_skip_full_forward):
                raise ValueError("LatentLoop deploy requires --lrnode_eval_skip_full_forward 1")
        if self.query_interval < 1:
            raise ValueError(f"lrnode_query_interval must be positive, got {self.query_interval}")
        if self.args.phase != "evaluate":
            raise ValueError(f"LatentLoop deploy requires --phase evaluate, got {self.args.phase!r}")
        if self.args.finetune_type != "real":
            raise ValueError(
                f"LatentLoop real deploy requires --finetune_type real, got {self.args.finetune_type!r}"
            )
        if self.action_pred_steps != 3:
            raise ValueError(
                f"The trained basketball protocol requires action_pred_steps=3, got {self.action_pred_steps}"
            )
        if self.history_len != 7:
            raise ValueError(
                f"The trained basketball protocol requires sequence_length=7, got {self.history_len}"
            )

    def _verify_one_artifact(self, path: Path, spec: Mapping[str, Any], label: str) -> str:
        if path.name != spec["filename"]:
            raise ValueError(
                f"{label} filename mismatch: expected {spec['filename']!r}, got {path.name!r}"
            )
        actual_size = path.stat().st_size
        if actual_size != int(spec["size_bytes"]):
            raise ValueError(
                f"{label} size mismatch: expected {spec['size_bytes']}, got {actual_size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != spec["sha256"]:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {spec['sha256']}, got {actual_hash}"
            )
        return actual_hash

    def _verify_architecture_contract(self) -> None:
        if self.deployment_method == "baseline":
            return
        expected = self.artifact_profile["latentloop_architecture"]
        actual = {
            "hidden_dim": int(self.args.lrnode_hidden_dim),
            "motion_dim": int(self.args.lrnode_motion_dim),
            "fast_encoder_type": self.args.lrnode_fast_encoder_type,
            "detach_input_latent": bool(self.args.lrnode_detach_input_latent),
            "detach_teacher_latent": bool(self.args.lrnode_detach_teacher_latent),
            "freeze_action_head": bool(self.args.lrnode_freeze_action_head_for_lrnode),
            "use_post_layernorm": bool(self.args.lrnode_use_post_layernorm),
            "gate_init_bias": float(self.args.lrnode_gate_init_bias),
        }
        if actual != expected:
            raise ValueError(
                "LatentLoop architecture does not match the training manifest: "
                f"expected={expected}, actual={actual}"
            )

    def _verify_artifacts(self) -> dict[str, str]:
        verified = {
            "teacher": self._verify_one_artifact(
                self.teacher_checkpoint, self.artifact_profile["teacher"], "teacher"
            ),
            "vit": self._verify_one_artifact(
                self.vit_checkpoint, self.artifact_profile["vit"], "ViT"
            ),
        }
        if self.deployment_method == "latentloop":
            verified["adapter"] = self._verify_one_artifact(
                self.adapter_checkpoint, self.artifact_profile["adapter"], "adapter"
            )
        return verified

    @staticmethod
    def random_seed(seed: int, rank: int = 0) -> None:
        torch.manual_seed(seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + rank)
            torch.cuda.manual_seed_all(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _setup_model(self) -> None:
        args = self.args
        self.model = SeerAgent(
            finetune_type=args.finetune_type,
            clip_device=self.device_id,
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

        if args.precision in {"bf16", "amp_bfloat16", "amp_bf16"}:
            self.model = self.model.bfloat16()
        elif args.precision == "fp16":
            self.model = self.model.half()
        elif args.precision == "fp32":
            self.model = self.model.float()
            if "vision_encoder" in args.bf16_module:
                self.model.vision_encoder.bfloat16()
            if "causal_transformer" in args.bf16_module:
                self.model.transformer_backbone.bfloat16()
            if "image_decoder" in args.bf16_module:
                self.model.image_decoder.bfloat16()
                self.model.image_decoder_obs_pred_projector.bfloat16()

        self.model.requires_grad_(False)
        self.model = self.model.to(self.device_id)
        self.model._init_model_type()
        self.model.profile_full_action_head = bool(args.lrnode_eval_profile_full_action_head)
        # Deployment is single-process inference. Wrapping a fully frozen model in
        # DDP is unnecessary and is rejected by the inference host's PyTorch 2.2.
        self.inference_model = self.model

        teacher_payload = torch.load(self.teacher_checkpoint, map_location="cpu")
        teacher_epoch = int(teacher_payload.get("epoch", -1))
        expected_teacher_epoch = int(self.artifact_profile["teacher"]["checkpoint_epoch"])
        if teacher_epoch != expected_teacher_epoch:
            raise ValueError(
                f"Teacher epoch mismatch: manifest={expected_teacher_epoch}, checkpoint={teacher_epoch}"
            )
        teacher_state_raw = teacher_payload.get("model_state_dict")
        if not isinstance(teacher_state_raw, Mapping):
            raise TypeError("Teacher checkpoint has no model_state_dict mapping")
        teacher_state = remove_ddp_prefix(teacher_state_raw)
        teacher_result = self.inference_model.load_state_dict(teacher_state, strict=False)
        disallowed_missing = [
            key
            for key in teacher_result.missing_keys
            if not is_allowed_teacher_missing_key(key)
        ]
        if disallowed_missing or teacher_result.unexpected_keys:
            raise RuntimeError(
                "Teacher checkpoint is incompatible with the deployment model: "
                f"missing={disallowed_missing}, unexpected={teacher_result.unexpected_keys}"
            )
        self.teacher_rebuilt_keys = sorted(teacher_result.missing_keys)

        if self.deployment_method == "latentloop":
            adapter_payload = torch.load(self.adapter_checkpoint, map_location="cpu")
            adapter_epoch = int(adapter_payload.get("epoch", -1))
            expected_adapter_epoch = int(
                self.artifact_profile["adapter"]["checkpoint_epoch"]
            )
            if adapter_epoch != expected_adapter_epoch:
                raise ValueError(
                    "Adapter epoch mismatch: "
                    f"manifest={expected_adapter_epoch}, checkpoint={adapter_epoch}"
                )
            adapter_state_raw = adapter_payload.get("model_state_dict")
            if not isinstance(adapter_state_raw, Mapping):
                raise TypeError("Adapter checkpoint has no model_state_dict mapping")
            adapter_state = remove_ddp_prefix(adapter_state_raw)
            expected_adapter_keys = {
                key for key in self.inference_model.state_dict() if key.startswith("lrnode_")
            }
            actual_adapter_keys = set(adapter_state)
            if actual_adapter_keys != expected_adapter_keys:
                raise RuntimeError(
                    "Adapter state must contain exactly the LatentLoop parameters: "
                    f"missing={sorted(expected_adapter_keys - actual_adapter_keys)}, "
                    f"unexpected={sorted(actual_adapter_keys - expected_adapter_keys)}"
                )
            self.inference_model.load_state_dict(adapter_state, strict=False)
        self.inference_model.eval()

        print(
            f"[{self.deployment_method} deploy] loaded "
            f"teacher={self.teacher_id} adapter={self.adapter_id} "
            f"K={self.query_interval} profile={self.deployment_profile}"
        )

    def attach_session_dir(self, session_dir: str) -> None:
        self.session_dir = Path(session_dir).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.write_runtime_summary()

    def reset(self, write_previous: bool = True) -> None:
        if (
            write_previous
            and not self._rollout_complete
            and hasattr(self, "step_records")
            and self.step_records
        ):
            self.write_runtime_summary()
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.cached_latent = None
        self.cached_image_primary = None
        self.cached_image_wrist = None
        self.cached_state = None
        self.cached_age = 0
        self.step_records: list[dict[str, Any]] = []
        self.full_forward_calls = 0
        self.latentloop_update_calls = 0
        self.full_forward_latency_sum_ms = 0.0
        self.latentloop_latency_sum_ms = 0.0
        self.policy_latency_sum_ms = 0.0
        self.control_command_monotonic_s: list[float] = []
        self.rollout_index += 1
        self._rollout_complete = False
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                [
                    self.real_eval_max_steps,
                    self.real_eval_max_steps + self.action_pred_steps,
                    7,
                ],
                device=self.device_id,
            )
        else:
            self.all_time_actions = None

    def mark_rollout_complete(self) -> None:
        """Seal metrics before the legacy GUI starts the next warm-up pass."""

        self.write_runtime_summary()
        self._rollout_complete = True

    def record_control_command(self, completed_monotonic_s: float) -> None:
        """Record when a robot command completes for achieved-rate accounting."""

        if self.rollout_index > 0 and not self._rollout_complete:
            self.control_command_monotonic_s.append(float(completed_monotonic_s))

    @staticmethod
    def _cuda_sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _selected_step(self, num_step: int) -> int:
        return num_step - 1 if num_step < self.history_len else -1

    def _preprocess(self, obs_dict: Mapping[str, Any]):
        image_x = PILImage.fromarray(obs_dict["color_image"][0]).convert("RGB")
        image_x = self.image_process_fn([image_x]).unsqueeze(1).to(dtype=self.cast_dtype)
        gripper_x = PILImage.fromarray(obs_dict["color_image"][1]).convert("RGB")
        gripper_x = self.image_process_fn([gripper_x]).unsqueeze(1).to(dtype=self.cast_dtype)
        text_x = self.text_process_fn([obs_dict["language_instruction"]]).unsqueeze(1)

        pose = np.asarray(obs_dict["robot_state"]["pose6d"], dtype=np.float32).reshape(-1)
        gripper_state = np.asarray(
            obs_dict["robot_state"]["gripper_open_state"], dtype=np.float32
        ).reshape(-1)
        if self.gripper_width:
            gripper_position = np.asarray(
                obs_dict["robot_state"]["gripper_position"], dtype=np.float32
            ).reshape(-1)
            state_np = np.concatenate([pose, gripper_position, gripper_position])
        else:
            state_np = np.concatenate([pose, gripper_state])
        expected_dim = 8 if self.gripper_width else 7
        if state_np.shape != (expected_dim,):
            raise RuntimeError(
                f"Expected real proprio shape ({expected_dim},), got {state_np.shape}"
            )
        state_x = torch.from_numpy(state_np).to(dtype=self.cast_dtype).unsqueeze(0).unsqueeze(0)
        return image_x, gripper_x, text_x, state_x

    def _prepare_context(self, image_x, gripper_x, text_x, state_x):
        image_x = image_x.to(self.device_id)
        gripper_x = gripper_x.to(self.device_id)
        text_x = text_x.to(self.device_id)
        state_x = state_x.to(self.device_id)
        self.img_queue.append(image_x)
        self.gripper_queue.append(gripper_x)
        self.state_queue.append(state_x)
        if not self.text_queue:
            for _ in range(self.history_len):
                self.text_queue.append(text_x)

        image_primary = torch.cat(list(self.img_queue), dim=1)
        image_wrist = torch.cat(list(self.gripper_queue), dim=1)
        state = torch.cat(list(self.state_queue), dim=1)
        text = torch.cat(list(self.text_queue), dim=1)
        num_step = int(image_primary.shape[1])
        if num_step < self.history_len:
            pad = self.history_len - num_step
            input_primary = torch.cat(
                [image_primary, image_primary[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1
            )
            input_wrist = torch.cat(
                [image_wrist, image_wrist[:, -1:].repeat(1, pad, 1, 1, 1)], dim=1
            )
            input_state = torch.cat([state, state[:, -1:].repeat(1, pad, 1)], dim=1)
        else:
            input_primary = image_primary
            input_wrist = image_wrist
            input_state = state
        return (
            image_x,
            gripper_x,
            state_x,
            input_primary,
            input_wrist,
            input_state,
            text,
            num_step,
        )

    def _full_forward(
        self,
        input_primary,
        input_wrist,
        input_state,
        input_text,
        current_primary,
        current_wrist,
        current_state,
        num_step: int,
    ):
        self._cuda_sync()
        started = time.perf_counter()
        outputs = self.inference_model(
            image_primary=input_primary,
            image_wrist=input_wrist,
            state=input_state,
            text_token=input_text,
            action=torch.zeros(1, self.history_len, 7, device=input_state.device),
            return_action_latent=True,
        )
        self._cuda_sync()
        full_ms = (time.perf_counter() - started) * 1000.0
        selected_step = self._selected_step(num_step)
        action_sequence = torch.cat(
            [
                outputs["arm_pred_action"][:, selected_step],
                outputs["gripper_pred_action"][:, selected_step],
            ],
            dim=-1,
        )
        action_latent = outputs["action_latent"]
        if action_latent.ndim != 4:
            raise RuntimeError(
                f"Expected full action latent [B,S,H,D], got {tuple(action_latent.shape)}"
            )
        self.cached_latent = action_latent[:, selected_step].detach()
        self.cached_image_primary = current_primary.detach()
        self.cached_image_wrist = current_wrist.detach()
        self.cached_state = current_state.detach()
        self.cached_age = 0
        self.full_forward_calls += 1
        self.full_forward_latency_sum_ms += full_ms
        return action_sequence, {
            "mode": "full",
            "cache_age": 0,
            "full_forward_ms": full_ms,
            "full_action_head_ms": float(
                getattr(self.model, "last_full_action_head_ms", 0.0)
            ),
            "fast_encoder_ms": 0.0,
            "latent_update_ms": 0.0,
            "action_head_ms": 0.0,
            "gate_mean": 0.0,
            "update_norm": 0.0,
            "u_delta_norm": 0.0,
        }

    def _latentloop_forward(self, current_primary, current_wrist, current_state):
        if self.cached_latent is None:
            raise RuntimeError("LatentLoop update requested before a full latent was cached")
        age = self.cached_age + 1

        self._cuda_sync()
        started = time.perf_counter()
        u_delta = self.model.lrnode_encode_delta(
            key_image_primary=self.cached_image_primary[:, 0],
            key_image_wrist=self.cached_image_wrist[:, 0],
            cur_image_primary=current_primary[:, 0],
            cur_image_wrist=current_wrist[:, 0],
            q_key=self.cached_state[:, 0],
            q_cur=current_state[:, 0],
        )
        self._cuda_sync()
        fast_ms = (time.perf_counter() - started) * 1000.0

        self._cuda_sync()
        started = time.perf_counter()
        z_next = self.model.lrnode_apply_dynamics(
            z_prev=self.cached_latent,
            u_delta=u_delta,
            dt=1.0,
            age=float(age),
        )
        self._cuda_sync()
        update_ms = (time.perf_counter() - started) * 1000.0

        self._cuda_sync()
        started = time.perf_counter()
        arm_action, gripper_action = self.model.decode_action_from_latent(z_next)
        self._cuda_sync()
        head_ms = (time.perf_counter() - started) * 1000.0
        action_sequence = torch.cat([arm_action, gripper_action], dim=-1)

        gate = getattr(self.model.lrnode_dynamics, "last_gate", None)
        update = getattr(self.model.lrnode_dynamics, "last_update", None)
        self.cached_latent = z_next.detach()
        self.cached_image_primary = current_primary.detach()
        self.cached_image_wrist = current_wrist.detach()
        self.cached_state = current_state.detach()
        self.cached_age = age
        latentloop_ms = fast_ms + update_ms + head_ms
        self.latentloop_update_calls += 1
        self.latentloop_latency_sum_ms += latentloop_ms
        return action_sequence, {
            "mode": "latentloop",
            "cache_age": age,
            "full_forward_ms": 0.0,
            "full_action_head_ms": 0.0,
            "fast_encoder_ms": fast_ms,
            "latent_update_ms": update_ms,
            "action_head_ms": head_ms,
            "gate_mean": (
                float(gate.detach().float().mean().item()) if gate is not None else 0.0
            ),
            "update_norm": (
                float(update.detach().float().norm(dim=-1).mean().item())
                if update is not None
                else 0.0
            ),
            "u_delta_norm": float(u_delta.detach().float().norm(dim=-1).mean().item()),
        }

    def _to_environment_action(self, action_sequence: torch.Tensor, timestep: int):
        if self.use_ensembling:
            action_probability, candidate_count = temporal_ensemble_probability(
                action_sequence,
                timestep,
                self.all_time_actions,
                self.ensembling_temp,
            )
        else:
            action_probability = action_sequence[:, 0]
            candidate_count = 1
        action = torch.cat(
            [action_probability[:, :6], action_probability[:, 6:] > 0.5], dim=-1
        )
        action[:, -1] = (action[:, -1] - 0.5) * 2
        action_np = action.detach().cpu().numpy()[-1]
        if action_np.shape != (7,):
            raise RuntimeError(f"Expected real action shape (7,), got {action_np.shape}")
        return action_np, candidate_count

    def forward(
        self,
        obs_dict: Mapping[str, Any],
        include_info: bool = False,
        timestep: int = 0,
        record_step: bool = True,
    ):
        policy_started = time.perf_counter()
        image_x, gripper_x, text_x, state_x = self._preprocess(obs_dict)
        with torch.no_grad():
            (
                current_primary,
                current_wrist,
                current_state,
                input_primary,
                input_wrist,
                input_state,
                input_text,
                num_step,
            ) = self._prepare_context(image_x, gripper_x, text_x, state_x)

            use_latentloop = self.deployment_method == "latentloop" and should_use_latentloop(
                int(timestep), self.query_interval, self.cached_latent is not None
            )
            if use_latentloop:
                action_sequence, record = self._latentloop_forward(
                    current_primary, current_wrist, current_state
                )
            else:
                action_sequence, record = self._full_forward(
                    input_primary,
                    input_wrist,
                    input_state,
                    input_text,
                    current_primary,
                    current_wrist,
                    current_state,
                    num_step,
                )
            action, candidate_count = self._to_environment_action(action_sequence, int(timestep))

        self._cuda_sync()
        policy_ms = (time.perf_counter() - policy_started) * 1000.0
        self.policy_latency_sum_ms += policy_ms
        record.update(
            {
                "timestep": int(timestep),
                "deployment_method": self.deployment_method,
                "query_interval": self.query_interval,
                "policy_ms": policy_ms,
                "temporal_ensemble_enabled": int(self.use_ensembling),
                "ensemble_candidate_count": candidate_count,
                "full_forward_calls": self.full_forward_calls,
                "latentloop_update_calls": self.latentloop_update_calls,
                "action": [float(value) for value in action],
            }
        )
        # The preserved GUI performs model warmup before its first reset. Keep
        # rollout zero out of deploy metrics; the first real rollout is index 1.
        if record_step and self.rollout_index > 0 and not self._rollout_complete:
            self.step_records.append(record)
            self._append_step_record(record)
            if bool(self.args.lrnode_eval_step_log):
                print("[LatentLoop step] " + json.dumps(record, sort_keys=True))

        target_pos = action[:3]
        target_euler = action[3:6]
        target_gripper = action[6]
        info = record if include_info else -1.0
        return target_pos, target_euler, target_gripper, info

    def _append_step_record(self, record: Mapping[str, Any]) -> None:
        if self.session_dir is None:
            return
        path = self.session_dir / f"policy_steps_rollout_{self.rollout_index:03d}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def runtime_summary(self) -> dict[str, Any]:
        total_steps = self.full_forward_calls + self.latentloop_update_calls
        command_times = np.asarray(self.control_command_monotonic_s, dtype=np.float64)
        periods_ms = np.diff(command_times) * 1000.0
        target_period_ms = 1000.0 / self.target_control_hz
        return {
            **self.deployment_metadata(),
            "rollout_index": self.rollout_index,
            "policy_steps": total_steps,
            "full_forward_calls": self.full_forward_calls,
            "latentloop_update_calls": self.latentloop_update_calls,
            "query_reduction": (
                1.0 - self.full_forward_calls / total_steps if total_steps else 0.0
            ),
            "effective_query_interval": (
                total_steps / self.full_forward_calls if self.full_forward_calls else 0.0
            ),
            "average_full_forward_ms": (
                self.full_forward_latency_sum_ms / self.full_forward_calls
                if self.full_forward_calls
                else 0.0
            ),
            "average_latentloop_ms": (
                self.latentloop_latency_sum_ms / self.latentloop_update_calls
                if self.latentloop_update_calls
                else 0.0
            ),
            "average_policy_ms": (
                self.policy_latency_sum_ms / total_steps if total_steps else 0.0
            ),
            "target_control_hz": self.target_control_hz,
            "target_control_period_ms": target_period_ms,
            "measured_control_period_count": int(periods_ms.size),
            "average_control_period_ms": (
                float(periods_ms.mean()) if periods_ms.size else 0.0
            ),
            "p95_control_period_ms": (
                float(np.percentile(periods_ms, 95)) if periods_ms.size else 0.0
            ),
            "maximum_control_period_ms": (
                float(periods_ms.max()) if periods_ms.size else 0.0
            ),
            "achieved_control_hz": (
                float(1000.0 / periods_ms.mean())
                if periods_ms.size and periods_ms.mean() > 0
                else 0.0
            ),
            "strict_deadline_miss_count": (
                int(np.count_nonzero(periods_ms > target_period_ms))
                if periods_ms.size
                else 0
            ),
            "strict_deadline_miss_rate": (
                float(np.mean(periods_ms > target_period_ms)) if periods_ms.size else 0.0
            ),
        }

    def write_runtime_summary(self) -> None:
        if self.session_dir is None:
            return
        path = self.session_dir / f"deployment_runtime_rollout_{self.rollout_index:03d}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.runtime_summary(), handle, indent=2, sort_keys=True)

    def deployment_metadata(self) -> dict[str, Any]:
        return {
            "method": "LatentLoop" if self.deployment_method == "latentloop" else "Seer baseline",
            "deployment_method": self.deployment_method,
            "deployment_profile": self.deployment_profile,
            "teacher_id": self.teacher_id,
            "adapter_id": self.adapter_id,
            "query_interval": self.query_interval,
            "teacher_checkpoint": str(self.teacher_checkpoint),
            "adapter_checkpoint": (
                str(self.adapter_checkpoint) if self.adapter_checkpoint is not None else None
            ),
            "vit_checkpoint": str(self.vit_checkpoint),
            "artifact_manifest": str(self.manifest_path),
            "artifact_sha256": dict(self.artifact_hashes),
            "latentloop_architecture": (
                dict(self.artifact_profile["latentloop_architecture"])
                if self.deployment_method == "latentloop"
                else None
            ),
            "teacher_rebuilt_key_count": len(self.teacher_rebuilt_keys),
            "action_pred_steps": self.action_pred_steps,
            "sequence_length": self.history_len,
            "temporal_ensemble_enabled": bool(self.use_ensembling),
            "temporal_ensemble_temperature": self.ensembling_temp,
            "target_control_hz": self.target_control_hz,
            "target_control_period_ms": 1000.0 / self.target_control_hz,
        }

    def run_synthetic_preflight(self, instruction: str) -> dict[str, Any]:
        """Exercise one complete scheduling cycle without opening physical hardware."""

        records = []
        for timestep in range(self.query_interval + 1):
            primary = np.full((480, 640, 3), timestep * 3, dtype=np.uint8)
            wrist = np.full((480, 640, 3), timestep * 5, dtype=np.uint8)
            pose6d = np.zeros(6, dtype=np.float32)
            pose6d[0] = timestep * 1e-3
            observation = {
                "color_image": [primary, wrist],
                "language_instruction": instruction,
                "robot_state": {
                    "pose6d": pose6d,
                    "gripper_open_state": np.array([1.0], dtype=np.float32),
                    "gripper_position": np.array([0.0], dtype=np.float32),
                },
            }
            target_pos, target_euler, target_gripper, record = self.forward(
                observation,
                include_info=True,
                timestep=timestep,
                record_step=False,
            )
            action = np.concatenate(
                [
                    np.asarray(target_pos).reshape(-1),
                    np.asarray(target_euler).reshape(-1),
                    np.asarray([target_gripper]).reshape(-1),
                ]
            )
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(
                    f"Synthetic preflight produced invalid action at step {timestep}: {action}"
                )
            records.append(record)

        expected_modes = [
            "latentloop"
            if self.deployment_method == "latentloop"
            and timestep % self.query_interval != 0
            else "full"
            for timestep in range(self.query_interval + 1)
        ]
        actual_modes = [record["mode"] for record in records]
        if actual_modes != expected_modes:
            raise RuntimeError(
                f"Synthetic preflight schedule mismatch: expected={expected_modes}, "
                f"actual={actual_modes}"
            )
        result = {
            "steps": len(records),
            "modes": actual_modes,
            "cache_ages": [record["cache_age"] for record in records],
            "full_forward_calls": self.full_forward_calls,
            "latentloop_update_calls": self.latentloop_update_calls,
            "all_actions_finite": True,
        }
        self.reset(write_previous=False)
        return result
