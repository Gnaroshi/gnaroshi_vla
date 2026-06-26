import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ['MUJOCO_GL'] = 'osmesa'

from pathlib import Path
import copy
import io
import distutils.dir_util
import json
import csv
import numpy as np
import re
import time
import torch
import torch.nn.functional as F
from torch.distributed import gather
from collections import deque
import functools
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

from utils.data_utils import preprocess_image, preprocess_text_calvin
from utils.train_utils import get_cast_dtype

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

# libero
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
from pdb import set_trace


def quaternion_to_euler(q):
    rot = R.from_quat(q)
    euler = rot.as_euler('xyz', degrees=False)

    return euler


benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}


def _safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:200]


def _is_rank0() -> bool:
    try:
        return (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)
    except Exception:
        return True


def _env_flag(name: str, default: str = "0") -> bool:
    return bool(int(os.environ.get(name, default)))


def _eval_control_hz() -> float:
    value = float(os.environ.get("EVAL_CONTROL_HZ", os.environ.get("LIBERO_CONTROL_HZ", "20")))
    if value <= 0:
        raise ValueError(f"EVAL_CONTROL_HZ must be positive, got {value}")
    return value


def _base_control_hz() -> float:
    value = float(os.environ.get("EVAL_BASE_CONTROL_HZ", "20"))
    if value <= 0:
        raise ValueError(f"EVAL_BASE_CONTROL_HZ must be positive, got {value}")
    return value


def _scaled_step_count(base_steps: int, control_hz: float) -> int:
    if not _env_flag("EVAL_SCALE_MAX_STEPS_WITH_HZ", "1"):
        return int(base_steps)
    return max(1, int(round(float(base_steps) * control_hz / _base_control_hz())))


def _settle_steps(control_hz: float) -> int:
    base_settle_steps = int(os.environ.get("EVAL_BASE_SETTLE_STEPS", "5"))
    if not _env_flag("EVAL_SCALE_SETTLE_STEPS_WITH_HZ", os.environ.get("EVAL_SCALE_MAX_STEPS_WITH_HZ", "1")):
        return base_settle_steps
    return max(1, int(round(float(base_settle_steps) * control_hz / _base_control_hz())))


def _env_horizon(eval_max_steps: int, settle_steps: int) -> int:
    requested = int(os.environ.get("EVAL_ENV_HORIZON", "0"))
    if requested > 0:
        return requested
    return max(1000, int(eval_max_steps) + int(settle_steps) + 10)


def save_episode_video(frames, out_path: str, fps: int = 20):
    """Save eval frames to mp4, falling back to gif if ffmpeg/libx264 is unavailable."""
    if frames is None or len(frames) == 0 or imageio is None:
        return None
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    fixed = []
    for frame in frames:
        try:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            fixed.append(arr)
        except Exception:
            continue
    if not fixed:
        return None

    try:
        with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as writer:
            for frame in fixed:
                writer.append_data(frame)
        return out_path
    except Exception:
        try:
            gif_path = os.path.splitext(out_path)[0] + ".gif"
            imageio.mimsave(gif_path, fixed, fps=fps)
            return gif_path
        except Exception:
            return None


class ModelWrapper:
    def __init__(self, model, tokenizer, image_processor, cast_dtype, history_len=10,
                 use_ensembling=False, ensembling_temp=0.01, libero_eval_max_steps=600, action_pred_steps=3,
                 gripper_width=False, use_lrnode_latent_update=0, lrnode_eval_skip_full_forward=0,
                 lrnode_query_interval=1, lrnode_eval_step_log=0, lrnode_eval_shadow_full_forward=0,
                 lrnode_eval_refresh_policy="periodic", lrnode_eval_max_full_forwards_per_episode=1):
        super().__init__()
        self.model = model
        self.cast_type = cast_dtype
        self.text_process_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
        self.image_process_fn = functools.partial(preprocess_image, image_processor=image_processor)
        self.action_hist_queue = []
        self.history_len = history_len
        self.libero_eval_max_steps = libero_eval_max_steps
        self.action_pred_steps = action_pred_steps
        self.device = "cuda"
        self.use_ensembling = use_ensembling
        self.ensembling_temp = ensembling_temp
        self.img_queue = deque(maxlen=history_len)
        self.gripper_queue = deque(maxlen=history_len)
        self.state_queue = deque(maxlen=history_len)
        self.mask_queue = deque(maxlen=history_len)
        self.text_queue = deque(maxlen=history_len)
        self.act_queue = deque(maxlen=history_len - 1)
        self.cnt = 0
        self.gripper_width = gripper_width
        self.use_lrnode_latent_update = bool(use_lrnode_latent_update)
        self.lrnode_eval_skip_full_forward = bool(lrnode_eval_skip_full_forward)
        self.lrnode_query_interval = max(1, int(lrnode_query_interval))
        self.lrnode_eval_step_log = bool(lrnode_eval_step_log)
        self.lrnode_eval_shadow_full_forward = bool(lrnode_eval_shadow_full_forward)
        self.lrnode_eval_refresh_policy = str(lrnode_eval_refresh_policy)
        if self.lrnode_eval_refresh_policy not in {"periodic", "first_only", "fixed_budget"}:
            raise ValueError(f"Unknown lrnode_eval_refresh_policy={self.lrnode_eval_refresh_policy}")
        self.lrnode_eval_max_full_forwards_per_episode = max(
            1, int(lrnode_eval_max_full_forwards_per_episode)
        )
        self.lrnode_episode_full_forward_calls = 0
        self.lrnode_cached_latent = None
        self.lrnode_cached_image_primary = None
        self.lrnode_cached_image_wrist = None
        self.lrnode_cached_state = None
        self.lrnode_cached_age = 0
        self.full_forward_calls = 0
        self.lrnode_update_calls = 0
        self.full_forward_latency_sum = 0.0
        self.lrnode_latency_sum = 0.0
        self.fast_encoder_latency_sum = 0.0
        self.node_update_latency_sum = 0.0
        self.action_head_latency_sum = 0.0
        self.policy_step_latency_sum = 0.0
        self.env_step_latency_sum = 0.0
        self.num_policy_steps = 0
        self.shadow_full_forward_calls = 0
        self.shadow_full_forward_latency_sum = 0.0
        self.shadow_latent_mse_sum = 0.0
        self.shadow_latent_cos_sum = 0.0
        self.shadow_action_l1_sum = 0.0
        self.shadow_action_l2_sum = 0.0
        self.shadow_action_hold_l1_sum = 0.0
        self.shadow_age_stats = {}
        self.current_step_records = []
        self.episode_metrics = []
        self.current_episode_start_time = None
        self.last_action = None
        self.last_action_delta = None
        if self.lrnode_eval_skip_full_forward:
            base_model = self._base_model()
            if not getattr(base_model, "use_lrnode_latent_update", False):
                raise RuntimeError("lrnode_eval_skip_full_forward=1 requires use_lrnode_latent_update=1")
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                [
                    self.libero_eval_max_steps,
                    self.libero_eval_max_steps + self.action_pred_steps,
                    7,
                ]
            ).to(self.device)

    def reset(self):
        self.img_queue = deque(maxlen=self.history_len)
        self.gripper_queue = deque(maxlen=self.history_len)
        self.state_queue = deque(maxlen=self.history_len)
        self.mask_queue = deque(maxlen=self.history_len)
        self.text_queue = deque(maxlen=self.history_len)
        self.act_queue = deque(maxlen=self.history_len - 1)
        self.gripper_state = np.array([-1.0])
        self.lrnode_cached_latent = None
        self.lrnode_cached_image_primary = None
        self.lrnode_cached_image_wrist = None
        self.lrnode_cached_state = None
        self.lrnode_cached_age = 0
        self.lrnode_episode_full_forward_calls = 0
        self.current_step_records = []
        self.current_episode_start_time = time.perf_counter()
        self.last_action = None
        self.last_action_delta = None
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                [
                    self.libero_eval_max_steps,
                    self.libero_eval_max_steps + self.action_pred_steps,
                    7,
                ]
            ).to(self.device)

        self.cnt += 1

    def _base_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _sync_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _selected_step(self, num_step):
        if num_step < self.history_len:
            return num_step - 1
        return -1

    def _action_sequence_to_env_action(self, action_seq, timestep):
        if action_seq.dim() != 3 or action_seq.shape[0] != 1 or action_seq.shape[-1] != 7:
            raise RuntimeError(f"Expected action sequence [1, action_pred_steps, 7], got {tuple(action_seq.shape)}")

        if self.use_ensembling:
            self.all_time_actions[timestep:timestep + 1, timestep:timestep + self.action_pred_steps] = action_seq
            actions_for_curr_step = self.all_time_actions[:, timestep]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = self.ensembling_temp
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
        else:
            action = action_seq[:, 0]

        action = torch.concat((action[:, :6], action[:, 6:] > 0.5), dim=-1)
        action[:, -1] = (action[:, -1] - 0.5) * 2
        action = action.detach().cpu().numpy()[-1]
        if action.shape != (7,):
            raise RuntimeError(f"LIBERO action must have shape (7,), got {action.shape}")
        return action

    def _should_use_lrnode(self, timestep):
        if not (
            self.use_lrnode_latent_update
            and self.lrnode_eval_skip_full_forward
            and self.lrnode_cached_latent is not None
        ):
            return False

        if self.lrnode_eval_refresh_policy == "first_only":
            return True

        if self.lrnode_eval_refresh_policy == "fixed_budget":
            if self.lrnode_episode_full_forward_calls >= self.lrnode_eval_max_full_forwards_per_episode:
                return True
            stride = max(
                1,
                int(np.ceil(float(self.libero_eval_max_steps) / self.lrnode_eval_max_full_forwards_per_episode)),
            )
            return timestep % stride != 0

        return timestep % self.lrnode_query_interval != 0

    def _full_refresh_reason(self, timestep):
        if self.lrnode_cached_latent is None:
            return "cache_empty"
        if not (self.use_lrnode_latent_update and self.lrnode_eval_skip_full_forward):
            return "normal_full"
        if self.lrnode_eval_refresh_policy == "periodic":
            return "scheduled_periodic"
        if self.lrnode_eval_refresh_policy == "fixed_budget":
            return "scheduled_fixed_budget"
        return "forced_full"

    def _cache_full_forward_state(self, action_latent, selected_step, image_x, gripper, state):
        if not self.use_lrnode_latent_update:
            return
        if action_latent is None or action_latent.dim() != 4:
            raise RuntimeError(f"Expected full action latent [B, S, action_pred_steps, D], got {type(action_latent)}")
        self.lrnode_cached_latent = action_latent[:, selected_step].detach()
        self.lrnode_cached_image_primary = image_x.detach()
        self.lrnode_cached_image_wrist = gripper.detach()
        self.lrnode_cached_state = state.detach()
        self.lrnode_cached_age = 0

    def _update_from_lrnode_cache(self, image_x, gripper, state):
        base_model = self._base_model()
        if self.lrnode_cached_latent is None:
            raise RuntimeError("LR-NODE skip requested before a full-forward latent was cached")

        age = self.lrnode_cached_age + 1
        z_prev = self.lrnode_cached_latent
        self._sync_cuda()
        t_fast = time.perf_counter()
        u_delta = base_model.lrnode_encode_delta(
            key_image_primary=self.lrnode_cached_image_primary[:, 0],
            key_image_wrist=self.lrnode_cached_image_wrist[:, 0],
            cur_image_primary=image_x[:, 0],
            cur_image_wrist=gripper[:, 0],
            q_key=self.lrnode_cached_state[:, 0],
            q_cur=state[:, 0],
        )
        self._sync_cuda()
        fast_encoder_ms = (time.perf_counter() - t_fast) * 1000.0

        self._sync_cuda()
        t_node = time.perf_counter()
        z_next = base_model.lrnode_apply_dynamics(
            z_prev=z_prev,
            u_delta=u_delta,
            dt=1.0,
            age=float(age),
        )
        self._sync_cuda()
        node_update_ms = (time.perf_counter() - t_node) * 1000.0

        self._sync_cuda()
        t_head = time.perf_counter()
        arm_action, gripper_action = base_model.decode_action_from_latent(z_next)
        with torch.no_grad():
            hold_arm_action, hold_gripper_action = base_model.decode_action_from_latent(z_prev.detach())
        self._sync_cuda()
        action_head_ms = (time.perf_counter() - t_head) * 1000.0

        action_seq = torch.concat((arm_action, gripper_action), dim=-1)
        hold_action_seq = torch.concat((hold_arm_action, hold_gripper_action), dim=-1)
        update = getattr(base_model.lrnode_dynamics, "last_update", None)
        gate = getattr(base_model.lrnode_dynamics, "last_gate", None)
        imgdiff_primary = (image_x[:, 0].detach().float() - self.lrnode_cached_image_primary[:, 0].detach().float()).abs()
        imgdiff_wrist = (gripper[:, 0].detach().float() - self.lrnode_cached_image_wrist[:, 0].detach().float()).abs()
        debug = {
            "cache_age": age,
            "fast_encoder_ms": fast_encoder_ms,
            "node_update_ms": node_update_ms,
            "action_head_ms": action_head_ms,
            "gate_mean": float(gate.detach().float().mean().item()) if gate is not None else 0.0,
            "gate_max": float(gate.detach().float().max().item()) if gate is not None else 0.0,
            "u_delta_norm": float(u_delta.detach().float().norm(dim=-1).mean().item()),
            "image_diff_primary_l1": float(imgdiff_primary.mean().item()),
            "image_diff_wrist_l1": float(imgdiff_wrist.mean().item()),
            "update_norm": float(update.detach().float().norm(dim=-1).mean().item()) if update is not None else 0.0,
            "z_norm": float(z_next.detach().float().norm(dim=-1).mean().item()),
            "z_pred": z_next.detach(),
            "z_hold": z_prev.detach(),
            "action_pred": action_seq.detach(),
            "action_hold": hold_action_seq.detach(),
        }
        self.lrnode_cached_latent = z_next.detach()
        self.lrnode_cached_image_primary = image_x.detach()
        self.lrnode_cached_image_wrist = gripper.detach()
        self.lrnode_cached_state = state.detach()
        self.lrnode_cached_age = age
        return action_seq, debug

    def get_lrnode_stats(self):
        total_calls = self.full_forward_calls + self.lrnode_update_calls
        avg_full_latency = self.full_forward_latency_sum / self.full_forward_calls if self.full_forward_calls else 0.0
        avg_lrnode_latency = self.lrnode_latency_sum / self.lrnode_update_calls if self.lrnode_update_calls else 0.0
        query_reduction = self.lrnode_update_calls / total_calls if total_calls else 0.0
        full_query_reduction_ratio = 1.0 - (self.full_forward_calls / self.num_policy_steps) if self.num_policy_steps else 0.0
        effective_query_interval = self.num_policy_steps / self.full_forward_calls if self.full_forward_calls else 0.0
        return {
            "num_env_steps": self.num_policy_steps,
            "full_forward_calls": self.full_forward_calls,
            "lrnode_update_calls": self.lrnode_update_calls,
            "num_fallback_full_calls": 0,
            "refresh_policy": self.lrnode_eval_refresh_policy,
            "max_full_forwards_per_episode": int(self.lrnode_eval_max_full_forwards_per_episode),
            "avg_full_forward_latency_sec": avg_full_latency,
            "avg_lrnode_latency_sec": avg_lrnode_latency,
            "avg_fast_encoder_latency_sec": (
                self.fast_encoder_latency_sum / self.lrnode_update_calls / 1000.0
                if self.lrnode_update_calls else 0.0
            ),
            "avg_node_update_latency_sec": (
                self.node_update_latency_sum / self.lrnode_update_calls / 1000.0
                if self.lrnode_update_calls else 0.0
            ),
            "avg_action_head_latency_sec": (
                self.action_head_latency_sum / max(1, self.lrnode_update_calls) / 1000.0
                if self.lrnode_update_calls else 0.0
            ),
            "avg_policy_step_latency_sec": (
                self.policy_step_latency_sum / self.num_policy_steps / 1000.0
                if self.num_policy_steps else 0.0
            ),
            "avg_env_step_latency_sec": (
                self.env_step_latency_sum / self.num_policy_steps / 1000.0
                if self.num_policy_steps else 0.0
            ),
            "effective_query_reduction": query_reduction,
            "full_query_reduction_ratio": full_query_reduction_ratio,
            "effective_query_interval": effective_query_interval,
            "shadow_full_forward_calls": self.shadow_full_forward_calls,
            "shadow_avg_full_forward_latency_sec": (
                self.shadow_full_forward_latency_sum / self.shadow_full_forward_calls / 1000.0
                if self.shadow_full_forward_calls else 0.0
            ),
            "shadow_latent_mse": self.shadow_latent_mse_sum / self.shadow_full_forward_calls
            if self.shadow_full_forward_calls else 0.0,
            "shadow_latent_cos": self.shadow_latent_cos_sum / self.shadow_full_forward_calls
            if self.shadow_full_forward_calls else 0.0,
            "shadow_action_l1": self.shadow_action_l1_sum / self.shadow_full_forward_calls
            if self.shadow_full_forward_calls else 0.0,
            "shadow_action_l2": self.shadow_action_l2_sum / self.shadow_full_forward_calls
            if self.shadow_full_forward_calls else 0.0,
            "shadow_action_hold_l1": self.shadow_action_hold_l1_sum / self.shadow_full_forward_calls
            if self.shadow_full_forward_calls else 0.0,
            "shadow_by_age": self.shadow_age_stats,
        }

    def record_env_step_ms(self, env_step_ms):
        self.env_step_latency_sum += float(env_step_ms)
        if self.current_step_records:
            self.current_step_records[-1]["env_step_ms"] = float(env_step_ms)

    def finish_episode(self, task, env, success, steps, args):
        records = list(self.current_step_records)

        def values(key):
            return [float(r.get(key, 0.0)) for r in records if r.get(key, None) is not None]

        def mean(key):
            vals = values(key)
            return float(np.mean(vals)) if vals else 0.0

        def percentile(key, q):
            vals = values(key)
            return float(np.percentile(vals, q)) if vals else 0.0

        full_count = sum(1 for r in records if r.get("mode") == "full")
        update_count = sum(1 for r in records if r.get("mode") == "lrnode_update")
        hold_count = sum(1 for r in records if r.get("mode") == "hold")
        episode_wallclock = (
            time.perf_counter() - self.current_episode_start_time
            if self.current_episode_start_time is not None else 0.0
        )
        control_hz = _eval_control_hz()
        settle_steps = _settle_steps(control_hz)
        metrics = {
            "episode_id": int(getattr(env, "exp_id", 0)),
            "task_id": int(getattr(env, "task_id", -1)),
            "task_name": getattr(task, "name", getattr(env, "task_name", "")),
            "seed": int(getattr(args, "seed", 0)),
            "success": int(success),
            "num_steps": int(steps),
            "control_hz": float(control_hz),
            "base_control_hz": float(_base_control_hz()),
            "eval_max_steps": int(args.libero_eval_max_steps),
            "settle_steps": int(settle_steps),
            "env_horizon": int(_env_horizon(args.libero_eval_max_steps, settle_steps)),
            "scale_max_steps_with_hz": int(_env_flag("EVAL_SCALE_MAX_STEPS_WITH_HZ", "1")),
            "lrnode_enabled": int(self.use_lrnode_latent_update),
            "eval_skip_full_forward": int(self.lrnode_eval_skip_full_forward),
            "query_interval": int(self.lrnode_query_interval),
            "refresh_policy": self.lrnode_eval_refresh_policy,
            "max_full_forwards_per_episode": int(self.lrnode_eval_max_full_forwards_per_episode),
            "mode_full_count": int(full_count),
            "mode_update_count": int(update_count),
            "mode_hold_count": int(hold_count),
            "full_forward_ratio": full_count / max(1, len(records)),
            "skip_ratio": update_count / max(1, len(records)),
            "avg_full_forward_ms": mean("full_forward_ms"),
            "avg_fast_encoder_ms": mean("fast_encoder_ms"),
            "avg_node_update_ms": mean("node_update_ms"),
            "avg_action_head_ms": mean("action_head_ms"),
            "avg_policy_step_ms": mean("total_policy_ms"),
            "avg_env_step_ms": mean("env_step_ms"),
            "episode_wallclock_sec": float(episode_wallclock),
            "avg_gate": mean("gate_mean"),
            "max_gate": max(values("gate_max") or [0.0]),
            "avg_image_diff_primary": mean("image_diff_primary_l1"),
            "avg_image_diff_wrist": mean("image_diff_wrist_l1"),
            "avg_update_norm": mean("update_norm"),
            "avg_action_norm": mean("action_norm"),
            "avg_action_delta_l2": mean("action_delta_norm"),
            "p95_action_delta_l2": percentile("action_delta_norm", 95),
            "avg_action_jerk": mean("action_jerk"),
            "p95_action_jerk": percentile("action_jerk", 95),
            "arm_action_jerk": mean("arm_action_jerk"),
            "trans_action_jerk": mean("trans_action_jerk"),
            "rot_action_jerk": mean("rot_action_jerk"),
            "gripper_switch_rate": mean("gripper_switch"),
            "failure_episode_id": int(getattr(env, "exp_id", 0)) if not success else "",
            "step_failed_or_timeout": int(steps) if not success else "",
            "avg_gate_before_failure": mean("gate_mean") if not success else "",
            "max_gate_before_failure": max(values("gate_max") or [0.0]) if not success else "",
            "max_image_diff_before_failure": max(values("image_diff_primary_l1") or [0.0]) if not success else "",
            "max_action_jerk_before_failure": max(values("action_jerk") or [0.0]) if not success else "",
            "cache_age_at_failure": int(records[-1].get("cache_age", 0)) if records and not success else "",
            "last_full_forward_step": max([int(r["timestep"]) for r in records if r.get("mode") == "full"] or [-1]),
        }
        self.episode_metrics.append(metrics)
        if self.lrnode_eval_step_log:
            self._save_step_log(task, env, args, records)
        return metrics

    def _save_step_log(self, task, env, args, records):
        log_dir = os.environ.get("LOG_DIR")
        if not log_dir or not records:
            return
        out_dir = Path(log_dir) / "analysis" / "eval_step_logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
        ckpt_tag = os.environ.get("CKPT_TAG", "ckpt")
        task_name = _safe_name(getattr(task, "name", "task"))
        path = out_dir / f"{ckpt_tag}_{task_name}_exp{int(getattr(env, 'exp_id', 0))}_rank{rank}.csv"
        keys = sorted(set().union(*(r.keys() for r in records)))
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

    def step(self, obs, goal, timestep, frames=None, video_stride: int = 1):
        policy_t0 = time.perf_counter()
        preprocess_t0 = time.perf_counter()
        step_record = {
            "timestep": int(timestep),
            "mode": "full",
            "cache_age": int(self.lrnode_cached_age),
            "query_interval": int(self.lrnode_query_interval),
            "refresh_policy": self.lrnode_eval_refresh_policy,
            "max_full_forwards_per_episode": int(self.lrnode_eval_max_full_forwards_per_episode),
            "full_forward_ms": 0.0,
            "fast_encoder_ms": 0.0,
            "node_update_ms": 0.0,
            "action_head_ms": 0.0,
            "env_step_ms": 0.0,
            "gate_mean": 0.0,
            "gate_max": 0.0,
            "u_delta_norm": 0.0,
            "image_diff_primary_l1": 0.0,
            "image_diff_wrist_l1": 0.0,
            "update_norm": 0.0,
            "z_norm": 0.0,
            "full_refresh_reason": "",
        }
        # preprocess image
        image = obs["agentview_image"]
        if frames is not None and (video_stride <= 1 or (timestep % int(video_stride) == 0)):
            try:
                primary = np.array(image, copy=True)
                wrist_raw = (
                    np.array(obs["robot0_eye_in_hand_image"], copy=True)
                    if "robot0_eye_in_hand_image" in obs
                    else None
                )
                if (
                    wrist_raw is not None
                    and isinstance(wrist_raw, np.ndarray)
                    and wrist_raw.ndim == 3
                    and primary.ndim == 3
                    and primary.shape[0] == wrist_raw.shape[0]
                    and primary.shape[2] == wrist_raw.shape[2]
                ):
                    frames.append(np.concatenate([primary, wrist_raw], axis=1))
                else:
                    frames.append(primary)
            except Exception:
                pass
        image = Image.fromarray(image)
        image_x = self.image_process_fn([image])
        # expand image dimension
        image_x = image_x.unsqueeze(1).to(dtype=self.cast_type)

        gripper = obs["robot0_eye_in_hand_image"]
        gripper = Image.fromarray(gripper)
        gripper = self.image_process_fn([gripper])
        # expand image dimension
        gripper = gripper.unsqueeze(1).to(dtype=self.cast_type)

        # expand text dimension
        text_x = self.text_process_fn([goal])
        text_x = text_x.unsqueeze(1)
        state_pos = obs["robot0_eef_pos"]
        state_ori = quaternion_to_euler(obs["robot0_eef_quat"])

        if not self.gripper_width:
            state = torch.from_numpy(np.concatenate([state_pos, state_ori, self.gripper_state])).to(
                dtype=self.cast_type).unsqueeze(0).unsqueeze(0)  # [1, 1, 7]
        else:
            state = torch.from_numpy(np.concatenate([state_pos, state_ori, obs['robot0_gripper_qpos']])).to(
                dtype=self.cast_type).unsqueeze(0).unsqueeze(0)  # [1, 1, 8]

        with torch.no_grad():
            device = 'cuda'
            image_x = image_x.to(device)
            text_x = text_x.to(device)
            gripper = gripper.to(device)
            state = state.to(device)

            self.img_queue.append(
                image_x)  # TODO find out how the policy completes the 5 sub-tasks. the obs of the later task will be appended after the former?
            self.gripper_queue.append(gripper)
            self.state_queue.append(state)
            if len(self.text_queue) == 0 and text_x is not None:  # the instruction does not change
                self.text_queue.append(text_x)
                for _ in range(self.model.module.sequence_length - 1):
                    self.text_queue.append(text_x)

            image_primary = torch.cat(list(self.img_queue), dim=1)
            image_wrist = torch.cat(list(self.gripper_queue), dim=1)
            state = torch.cat(list(self.state_queue), dim=1)
            input_text_token = torch.cat(list(self.text_queue), dim=1)

            num_step = image_primary.shape[1]
            if num_step < self.history_len:  # padding
                input_image_primary = torch.cat(
                    [image_primary, image_primary[:, -1].repeat(1, self.history_len - num_step, 1, 1, 1)], dim=1)
                input_image_wrist = torch.cat(
                    [image_wrist, image_wrist[:, -1].repeat(1, self.history_len - num_step, 1, 1, 1)], dim=1)
                input_state = torch.cat([state, state[:, -1].repeat(1, self.history_len - num_step, 1)], dim=1)
            else:
                input_image_primary = image_primary
                input_image_wrist = image_wrist
                input_state = state

            preprocess_ms = (time.perf_counter() - preprocess_t0) * 1000.0
            step_record["preprocess_ms"] = preprocess_ms

            if self._should_use_lrnode(timestep):
                self._sync_cuda()
                t0 = time.perf_counter()
                action_seq, lrnode_debug = self._update_from_lrnode_cache(image_x, gripper, state)
                self._sync_cuda()
                lrnode_ms = (time.perf_counter() - t0) * 1000.0
                self.lrnode_latency_sum += lrnode_ms / 1000.0
                self.lrnode_update_calls += 1
                self.fast_encoder_latency_sum += float(lrnode_debug.get("fast_encoder_ms", 0.0))
                self.node_update_latency_sum += float(lrnode_debug.get("node_update_ms", 0.0))
                self.action_head_latency_sum += float(lrnode_debug.get("action_head_ms", 0.0))
                step_record.update(
                    {
                        "mode": "lrnode_update",
                        "cache_age": int(lrnode_debug.get("cache_age", 0)),
                        "fast_encoder_ms": float(lrnode_debug.get("fast_encoder_ms", 0.0)),
                        "node_update_ms": float(lrnode_debug.get("node_update_ms", 0.0)),
                        "action_head_ms": float(lrnode_debug.get("action_head_ms", 0.0)),
                        "total_policy_ms": lrnode_ms + preprocess_ms,
                        "gate_mean": float(lrnode_debug.get("gate_mean", 0.0)),
                        "gate_max": float(lrnode_debug.get("gate_max", 0.0)),
                        "u_delta_norm": float(lrnode_debug.get("u_delta_norm", 0.0)),
                        "image_diff_primary_l1": float(lrnode_debug.get("image_diff_primary_l1", 0.0)),
                        "image_diff_wrist_l1": float(lrnode_debug.get("image_diff_wrist_l1", 0.0)),
                        "update_norm": float(lrnode_debug.get("update_norm", 0.0)),
                        "z_norm": float(lrnode_debug.get("z_norm", 0.0)),
                    }
                )
                if self.lrnode_eval_shadow_full_forward:
                    selected_step = self._selected_step(num_step)
                    self._sync_cuda()
                    shadow_t0 = time.perf_counter()
                    shadow_outputs = self.model(
                        image_primary=input_image_primary,
                        image_wrist=input_image_wrist,
                        state=input_state,
                        text_token=input_text_token,
                        action=torch.zeros(1, self.history_len, 7).to(input_state.device),
                        return_action_latent=True,
                    )
                    self._sync_cuda()
                    shadow_ms = (time.perf_counter() - shadow_t0) * 1000.0
                    shadow_arm = shadow_outputs["arm_pred_action"][:, selected_step]
                    shadow_gripper = shadow_outputs["gripper_pred_action"][:, selected_step]
                    shadow_action = torch.cat([shadow_arm, shadow_gripper], dim=-1).detach().float()
                    shadow_latent = shadow_outputs["action_latent"][:, selected_step].detach().float()
                    pred_action = lrnode_debug["action_pred"].detach().float()
                    hold_action = lrnode_debug["action_hold"].detach().float()
                    pred_latent = lrnode_debug["z_pred"].detach().float()
                    latent_mse = F.mse_loss(pred_latent, shadow_latent).item()
                    latent_cos = F.cosine_similarity(
                        pred_latent.reshape(-1, pred_latent.shape[-1]),
                        shadow_latent.reshape(-1, shadow_latent.shape[-1]),
                        dim=-1,
                    ).mean().item()
                    action_l1 = F.l1_loss(pred_action, shadow_action).item()
                    action_l2 = torch.sqrt(F.mse_loss(pred_action, shadow_action)).item()
                    action_hold_l1 = F.l1_loss(hold_action, shadow_action).item()
                    improvement = action_hold_l1 - action_l1
                    age_key = f"age{int(lrnode_debug.get('cache_age', 0))}"
                    age_stats = self.shadow_age_stats.setdefault(
                        age_key,
                        {"count": 0, "latent_mse_sum": 0.0, "action_l1_sum": 0.0, "action_hold_l1_sum": 0.0},
                    )
                    age_stats["count"] += 1
                    age_stats["latent_mse_sum"] += latent_mse
                    age_stats["action_l1_sum"] += action_l1
                    age_stats["action_hold_l1_sum"] += action_hold_l1
                    self.shadow_full_forward_calls += 1
                    self.shadow_full_forward_latency_sum += shadow_ms
                    self.shadow_latent_mse_sum += latent_mse
                    self.shadow_latent_cos_sum += latent_cos
                    self.shadow_action_l1_sum += action_l1
                    self.shadow_action_l2_sum += action_l2
                    self.shadow_action_hold_l1_sum += action_hold_l1
                    step_record.update(
                        {
                            "shadow_full_forward_ms": shadow_ms,
                            "shadow_latent_mse": latent_mse,
                            "shadow_latent_cos": latent_cos,
                            "shadow_action_l1": action_l1,
                            "shadow_action_l2": action_l2,
                            "shadow_action_hold_l1": action_hold_l1,
                            "shadow_pred_vs_hold_improvement": improvement,
                        }
                    )
            else:
                self._sync_cuda()
                t0 = time.perf_counter()
                model_outputs = self.model(
                    image_primary=input_image_primary,
                    image_wrist=input_image_wrist,
                    state=input_state,
                    text_token=input_text_token,
                    action=torch.zeros(1, self.history_len, 7).to(input_state.device),
                    return_action_latent=self.use_lrnode_latent_update,
                )
                self._sync_cuda()
                full_ms = (time.perf_counter() - t0) * 1000.0
                self.full_forward_latency_sum += full_ms / 1000.0
                self.full_forward_calls += 1
                self.lrnode_episode_full_forward_calls += 1
                step_record.update(
                    {
                        "mode": "full",
                        "full_forward_ms": full_ms,
                        "full_refresh_reason": self._full_refresh_reason(timestep),
                    }
                )

                if self.use_lrnode_latent_update:
                    arm_action = model_outputs["arm_pred_action"]
                    gripper_action = model_outputs["gripper_pred_action"]
                    action_latent = model_outputs["action_latent"]
                else:
                    arm_action, gripper_action, _, _, _, _ = model_outputs
                    action_latent = None
                selected_step = self._selected_step(num_step)
                action_seq = torch.concat((arm_action[:, selected_step], gripper_action[:, selected_step]), dim=-1)
                self._cache_full_forward_state(action_latent, selected_step, image_x, gripper, state)

            action = self._action_sequence_to_env_action(action_seq, timestep)
            action_float = np.asarray(action, dtype=np.float32)
            action_delta = np.zeros_like(action_float) if self.last_action is None else action_float - self.last_action
            action_jerk = (
                np.zeros_like(action_float)
                if self.last_action_delta is None
                else action_delta - self.last_action_delta
            )
            step_record.update(
                {
                    "action_norm": float(np.linalg.norm(action_float)),
                    "action_delta_norm": float(np.linalg.norm(action_delta)),
                    "action_jerk": float(np.linalg.norm(action_jerk)),
                    "arm_action_jerk": float(np.linalg.norm(action_jerk[:6])),
                    "trans_action_jerk": float(np.linalg.norm(action_jerk[:3])),
                    "rot_action_jerk": float(np.linalg.norm(action_jerk[3:6])),
                    "gripper_switch": float(
                        0.0 if self.last_action is None else abs(float(action_float[-1] != self.last_action[-1]))
                    ),
                }
            )
            self.last_action = action_float.copy()
            self.last_action_delta = action_delta.copy()
            total_policy_ms = (time.perf_counter() - policy_t0) * 1000.0
            step_record["total_policy_ms"] = total_policy_ms
            self.policy_step_latency_sum += total_policy_ms
            self.num_policy_steps += 1
            self.current_step_records.append(step_record)

        self.gripper_state = np.array([action[-1]])
        return action


def evaluate_libero_task(task, env, obs, args, model):
    steps = 0
    success = 0
    model.reset()
    goal = task.language

    save_video_flag = bool(int(os.environ.get("SAVE_VIDEO", "0"))) or bool(getattr(args, "save_video", False))
    save_video_all_ranks = (
        bool(int(os.environ.get("SAVE_VIDEO_ALL_RANKS", "0")))
        or bool(getattr(args, "save_video_all_ranks", False))
    )
    save_video_succ = bool(int(os.environ.get("SAVE_VIDEO_SUCC", "1")))
    save_video_fail = bool(int(os.environ.get("SAVE_VIDEO_FAIL", "1")))
    video_fps = int(os.environ.get("VIDEO_FPS", getattr(args, "video_fps", 20)))
    video_stride = max(1, int(os.environ.get("VIDEO_STRIDE", getattr(args, "video_stride", 1))))

    run_name = os.environ.get("RUN_NAME", getattr(args, "run_name", "run"))
    base_dir = os.environ.get("LOG_DIR")
    if base_dir is None:
        base_video_dir = os.path.join(os.getcwd(), "eval_videos", _safe_name(run_name))
    else:
        base_video_dir = os.path.join(base_dir, "eval_videos", _safe_name(run_name))

    ckpt_tag = os.environ.get("CKPT_TAG", "").strip()
    if ckpt_tag:
        base_video_dir = os.path.join(base_video_dir, ckpt_tag)

    do_collect_video = save_video_flag and (_is_rank0() or save_video_all_ranks)
    frames = [] if do_collect_video else None
    with torch.no_grad():
        while steps < args.libero_eval_max_steps:  # default
            action = model.step(obs, goal, steps, frames=frames, video_stride=video_stride)
            steps += 1

            env_t0 = time.perf_counter()
            obs, reward, done, info = env.step(action)
            model.record_env_step_ms((time.perf_counter() - env_t0) * 1000.0)
            if done:
                success = 1
                break

    if frames is not None and len(frames) > 0:
        should_save = (success == 1 and save_video_succ) or (success == 0 and save_video_fail)
        if should_save:
            split_dir = "success" if success == 1 else "fail"
            video_dir = os.path.join(base_video_dir, split_dir)
            Path(video_dir).mkdir(parents=True, exist_ok=True)

            task_tag = _safe_name(getattr(task, "name", "task"))
            exp_id = int(getattr(env, "exp_id", 0))
            try:
                rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
            except Exception:
                rank = 0
            out_name = f"{task_tag}_exp{exp_id}_succ{success}_seed{getattr(args, 'seed', 0)}_rank{rank}.mp4"
            if ckpt_tag:
                out_name = f"{ckpt_tag}_{out_name}"
            out_path = os.path.join(video_dir, out_name)
            saved = save_episode_video(frames, out_path, fps=video_fps)
            if saved is not None and (_is_rank0() or save_video_all_ranks):
                print(f"[VIDEO] saved: {saved} ({len(frames)} frames)")
    episode_metrics = model.finish_episode(task, env, success, steps, args)
    env.close()
    return success, episode_metrics


def evaluate_policy_ddp(args, model):
    pass
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.finetune_type]()
    device_num = int(torch.distributed.get_world_size())
    device_id = torch.distributed.get_rank()
    results = []
    local_episode_metrics = []
    control_hz = _eval_control_hz()
    settle_steps = _settle_steps(control_hz)
    env_horizon = _env_horizon(args.libero_eval_max_steps, settle_steps)
    if "libero" in args.finetune_type:
        if args.finetune_type == "libero_10":
            global num_eval_episodes
            global task_num
            num_eval_episodes = 20
            task_num = 10

            NUM_SEQUENCES = num_eval_episodes * task_num
            eval_sequences = list(range(NUM_SEQUENCES))
            interval_len = int(np.ceil(NUM_SEQUENCES / device_num))
            eval_sequences = eval_sequences[
                device_id * interval_len:min((device_id + 1) * interval_len, NUM_SEQUENCES)
            ]
            eval_sequence_ids = list(eval_sequences)
            eval_sequences = tqdm(eval_sequence_ids)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    for eval_id in eval_sequences:
        task_id = eval_id // num_eval_episodes
        exp_id = eval_id % num_eval_episodes
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(f"{args.libero_path}/libero/libero/bddl_files", task.problem_folder,
                                      task.bddl_file)
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": args.libero_img_size,
            "camera_widths": args.libero_img_size,
            "render_gpu_device_id": device_id,
            "control_freq": int(round(control_hz)),
            "horizon": env_horizon,
        }
        print("device_id :", device_id)
        print(
            f"[LIBERO ENV] control_freq={env_args['control_freq']}, "
            f"eval_max_steps={args.libero_eval_max_steps}, "
            f"settle_steps={settle_steps}, horizon={env_horizon}"
        )
        env = OffScreenRenderEnv(**env_args)
        env.exp_id = exp_id
        env.task_id = task_id
        env.task_name = task_name
        env.task_suite_name = args.finetune_type
        env.reset()
        env.seed(args.seed)

        # set initial state
        init_states_path = os.path.join(
            f"{args.libero_path}/libero/libero/init_files", task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        init_state = init_states[exp_id]
        obs = env.set_init_state(init_state)

        for _ in range(settle_steps):  # simulate the physics without any actions
            env.step(np.zeros(7))

        result, episode_metrics = evaluate_libero_task(task, env, obs, args, model)
        results.append(result)
        local_episode_metrics.append(episode_metrics)
        print("rank", torch.distributed.get_rank(), "results :", results)

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    res_tup = [(res, eval_seq) for res, eval_seq in zip(results, eval_sequence_ids)]
    all_res_tup = [copy.deepcopy(res_tup) for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(res_tup, all_res_tup, dst=0)
    local_lrnode_stats = model.get_lrnode_stats()
    all_lrnode_stats = [None for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(local_lrnode_stats, all_lrnode_stats, dst=0)
    all_episode_metrics = [None for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(local_episode_metrics, all_episode_metrics, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_tup_list.sort(key=lambda x: x[1])
        episode_metrics_list = merge_multi_list(all_episode_metrics)
        print_and_save(res_tup_list, task_suite)
        save_eval_json(args, res_tup_list, task_suite, all_lrnode_stats, episode_metrics_list)


def print_and_save(result_list, task_suite):
    for j in range(task_num):
        this_result_list = result_list[j * num_eval_episodes: (j + 1) * num_eval_episodes]
        print("this_result_list :", this_result_list)
        this_result_list = np.array(this_result_list)
        avg_success = np.mean(this_result_list, axis=0)[0]
        task = task_suite.get_task(j)
        task_name = task.name
        print(f"Success rates for task {j} {task_name}:")
        print(f"{avg_success * 100:.1f}%")


def merge_lrnode_stats(stats_list):
    merged = {
        "num_env_steps": 0,
        "full_forward_calls": 0,
        "lrnode_update_calls": 0,
        "num_fallback_full_calls": 0,
        "full_forward_latency_sum": 0.0,
        "lrnode_latency_sum": 0.0,
        "fast_encoder_latency_sum": 0.0,
        "node_update_latency_sum": 0.0,
        "action_head_latency_sum": 0.0,
        "policy_step_latency_sum": 0.0,
        "env_step_latency_sum": 0.0,
        "shadow_full_forward_calls": 0,
        "shadow_full_forward_latency_sum": 0.0,
        "shadow_latent_mse_sum": 0.0,
        "shadow_latent_cos_sum": 0.0,
        "shadow_action_l1_sum": 0.0,
        "shadow_action_l2_sum": 0.0,
        "shadow_action_hold_l1_sum": 0.0,
        "shadow_by_age": {},
    }
    for item in stats_list:
        if item is None:
            continue
        env_steps = int(item.get("num_env_steps", 0))
        full_calls = int(item.get("full_forward_calls", 0))
        lrnode_calls = int(item.get("lrnode_update_calls", 0))
        shadow_calls = int(item.get("shadow_full_forward_calls", 0))
        merged["num_env_steps"] += env_steps
        merged["full_forward_calls"] += full_calls
        merged["lrnode_update_calls"] += lrnode_calls
        merged["num_fallback_full_calls"] += int(item.get("num_fallback_full_calls", 0))
        merged["full_forward_latency_sum"] += float(item.get("avg_full_forward_latency_sec", 0.0)) * full_calls
        merged["lrnode_latency_sum"] += float(item.get("avg_lrnode_latency_sec", 0.0)) * lrnode_calls
        merged["fast_encoder_latency_sum"] += float(item.get("avg_fast_encoder_latency_sec", 0.0)) * lrnode_calls
        merged["node_update_latency_sum"] += float(item.get("avg_node_update_latency_sec", 0.0)) * lrnode_calls
        merged["action_head_latency_sum"] += float(item.get("avg_action_head_latency_sec", 0.0)) * lrnode_calls
        merged["policy_step_latency_sum"] += float(item.get("avg_policy_step_latency_sec", 0.0)) * env_steps
        merged["env_step_latency_sum"] += float(item.get("avg_env_step_latency_sec", 0.0)) * env_steps
        merged["shadow_full_forward_calls"] += shadow_calls
        merged["shadow_full_forward_latency_sum"] += float(item.get("shadow_avg_full_forward_latency_sec", 0.0)) * shadow_calls
        merged["shadow_latent_mse_sum"] += float(item.get("shadow_latent_mse", 0.0)) * shadow_calls
        merged["shadow_latent_cos_sum"] += float(item.get("shadow_latent_cos", 0.0)) * shadow_calls
        merged["shadow_action_l1_sum"] += float(item.get("shadow_action_l1", 0.0)) * shadow_calls
        merged["shadow_action_l2_sum"] += float(item.get("shadow_action_l2", 0.0)) * shadow_calls
        merged["shadow_action_hold_l1_sum"] += float(item.get("shadow_action_hold_l1", 0.0)) * shadow_calls
        for age_key, age_item in item.get("shadow_by_age", {}).items():
            target = merged["shadow_by_age"].setdefault(
                age_key,
                {"count": 0, "latent_mse_sum": 0.0, "action_l1_sum": 0.0, "action_hold_l1_sum": 0.0},
            )
            target["count"] += int(age_item.get("count", 0))
            target["latent_mse_sum"] += float(age_item.get("latent_mse_sum", 0.0))
            target["action_l1_sum"] += float(age_item.get("action_l1_sum", 0.0))
            target["action_hold_l1_sum"] += float(age_item.get("action_hold_l1_sum", 0.0))

    total_calls = merged["full_forward_calls"] + merged["lrnode_update_calls"]
    merged["avg_full_forward_latency_sec"] = (
        merged["full_forward_latency_sum"] / merged["full_forward_calls"]
        if merged["full_forward_calls"]
        else 0.0
    )
    merged["avg_lrnode_latency_sec"] = (
        merged["lrnode_latency_sum"] / merged["lrnode_update_calls"]
        if merged["lrnode_update_calls"]
        else 0.0
    )
    merged["avg_fast_encoder_latency_sec"] = (
        merged["fast_encoder_latency_sum"] / merged["lrnode_update_calls"]
        if merged["lrnode_update_calls"]
        else 0.0
    )
    merged["avg_node_update_latency_sec"] = (
        merged["node_update_latency_sum"] / merged["lrnode_update_calls"]
        if merged["lrnode_update_calls"]
        else 0.0
    )
    merged["avg_action_head_latency_sec"] = (
        merged["action_head_latency_sum"] / merged["lrnode_update_calls"]
        if merged["lrnode_update_calls"]
        else 0.0
    )
    merged["avg_policy_step_latency_sec"] = (
        merged["policy_step_latency_sum"] / merged["num_env_steps"]
        if merged["num_env_steps"]
        else 0.0
    )
    merged["avg_env_step_latency_sec"] = (
        merged["env_step_latency_sum"] / merged["num_env_steps"]
        if merged["num_env_steps"]
        else 0.0
    )
    merged["effective_query_reduction"] = merged["lrnode_update_calls"] / total_calls if total_calls else 0.0
    merged["full_query_reduction_ratio"] = (
        1.0 - (merged["full_forward_calls"] / merged["num_env_steps"])
        if merged["num_env_steps"]
        else 0.0
    )
    merged["effective_query_interval"] = (
        merged["num_env_steps"] / merged["full_forward_calls"]
        if merged["full_forward_calls"]
        else 0.0
    )
    merged["shadow_avg_full_forward_latency_sec"] = (
        merged["shadow_full_forward_latency_sum"] / merged["shadow_full_forward_calls"]
        if merged["shadow_full_forward_calls"]
        else 0.0
    )
    for key in ["latent_mse", "latent_cos", "action_l1", "action_l2", "action_hold_l1"]:
        merged[f"shadow_{key}"] = (
            merged[f"shadow_{key}_sum"] / merged["shadow_full_forward_calls"]
            if merged["shadow_full_forward_calls"]
            else 0.0
        )
    for age_key, age_item in merged["shadow_by_age"].items():
        count = max(1, int(age_item.get("count", 0)))
        age_item["latent_mse"] = age_item["latent_mse_sum"] / count
        age_item["action_l1"] = age_item["action_l1_sum"] / count
        age_item["action_hold_l1"] = age_item["action_hold_l1_sum"] / count
    return merged


def _write_episode_metrics_csv(path, episode_metrics):
    if not episode_metrics:
        return
    keys = sorted(set().union(*(item.keys() for item in episode_metrics)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(episode_metrics)


def _profile_values(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _write_latency_profile(path, episode_metrics):
    key_map = {
        "full_forward_model_ms": "avg_full_forward_ms",
        "fast_delta_encoder_ms": "avg_fast_encoder_ms",
        "node_update_ms": "avg_node_update_ms",
        "action_head_ms": "avg_action_head_ms",
        "policy_total_ms": "avg_policy_step_ms",
        "env_step_ms": "avg_env_step_ms",
        "e2e_step_ms": "avg_policy_step_ms",
        "action_delta_l2": "avg_action_delta_l2",
        "action_jerk_l2": "avg_action_jerk",
    }
    profile = {}
    for out_key, metric_key in key_map.items():
        profile[out_key] = _profile_values([item.get(metric_key, 0.0) for item in episode_metrics])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def save_eval_json(args, result_list, task_suite, lrnode_stats_list, episode_metrics=None):
    log_dir = os.environ.get("LOG_DIR")
    if log_dir:
        output_dir = os.path.join(log_dir, "analysis")
    else:
        output_dir = os.path.join(os.getcwd(), "evaluate")
    os.makedirs(output_dir, exist_ok=True)
    valid_results = [int(item[0]) for item in result_list if int(item[0]) in [0, 1]]
    success_rate = float(np.mean(valid_results)) if valid_results else 0.0
    lrnode_stats = merge_lrnode_stats(lrnode_stats_list)
    episode_metrics = episode_metrics or []
    control_hz = _eval_control_hz()
    settle_steps = _settle_steps(control_hz)
    env_horizon = _env_horizon(args.libero_eval_max_steps, settle_steps)
    query_interval = max(1, int(args.lrnode_query_interval))
    nominal_full_query_hz = control_hz / query_interval if bool(args.lrnode_eval_skip_full_forward) else control_hz
    nominal_lrnode_update_hz = max(0.0, control_hz - nominal_full_query_hz)
    num_env_steps = int(lrnode_stats.get("num_env_steps", 0))
    if num_env_steps > 0:
        effective_full_query_hz = (
            control_hz * float(lrnode_stats.get("full_forward_calls", 0)) / float(num_env_steps)
        )
        effective_lrnode_update_hz = (
            control_hz * float(lrnode_stats.get("lrnode_update_calls", 0)) / float(num_env_steps)
        )
    else:
        effective_full_query_hz = nominal_full_query_hz
        effective_lrnode_update_hz = nominal_lrnode_update_hz

    task_results = []
    for j in range(task_num):
        this_result_list = result_list[j * num_eval_episodes: (j + 1) * num_eval_episodes]
        task = task_suite.get_task(j)
        values = [int(item[0]) for item in this_result_list if int(item[0]) in [0, 1]]
        task_results.append(
            {
                "task_id": j,
                "task_name": task.name,
                "success_rate": float(np.mean(values)) if values else 0.0,
                "num_episodes": len(values),
            }
        )

    payload = {
        "run_name": args.run_name,
        "suite": args.finetune_type,
        "success_rate": success_rate,
        "environment": {
            "control_freq": int(round(control_hz)),
            "control_hz": control_hz,
            "base_control_hz": _base_control_hz(),
            "eval_max_steps": int(args.libero_eval_max_steps),
            "settle_steps": int(settle_steps),
            "env_horizon": int(env_horizon),
            "scale_max_steps_with_hz": _env_flag("EVAL_SCALE_MAX_STEPS_WITH_HZ", "1"),
            "scale_settle_steps_with_hz": _env_flag(
                "EVAL_SCALE_SETTLE_STEPS_WITH_HZ",
                os.environ.get("EVAL_SCALE_MAX_STEPS_WITH_HZ", "1"),
            ),
        },
        "lrnode": {
            "enabled": bool(args.use_lrnode_latent_update),
            "eval_skip_full_forward": bool(args.lrnode_eval_skip_full_forward),
            "query_interval": query_interval,
            "eval_refresh_policy": getattr(args, "lrnode_eval_refresh_policy", "periodic"),
            "max_full_forwards_per_episode": int(
                getattr(args, "lrnode_eval_max_full_forwards_per_episode", 1)
            ),
            "control_hz": control_hz,
            "effective_action_hz": control_hz,
            "nominal_full_query_hz": nominal_full_query_hz,
            "nominal_lrnode_update_hz": nominal_lrnode_update_hz,
            "effective_full_query_hz": effective_full_query_hz,
            "effective_lrnode_update_hz": effective_lrnode_update_hz,
            "detach_input_latent": bool(args.lrnode_detach_input_latent),
            "detach_teacher_latent": bool(args.lrnode_detach_teacher_latent),
            "freeze_action_head_for_lrnode": bool(args.lrnode_freeze_action_head_for_lrnode),
            "use_post_layernorm": bool(args.lrnode_use_post_layernorm),
            "multistep_train": bool(args.lrnode_multistep_train),
            "train_max_horizon": int(args.lrnode_train_max_horizon),
            **lrnode_stats,
        },
        "query_reduction": {
            "num_env_steps": int(lrnode_stats.get("num_env_steps", 0)),
            "num_full_forward_calls": int(lrnode_stats.get("full_forward_calls", 0)),
            "num_lrnode_update_calls": int(lrnode_stats.get("lrnode_update_calls", 0)),
            "num_fallback_full_calls": int(lrnode_stats.get("num_fallback_full_calls", 0)),
            "full_query_reduction_ratio": float(lrnode_stats.get("full_query_reduction_ratio", 0.0)),
            "effective_query_interval": float(lrnode_stats.get("effective_query_interval", 0.0)),
        },
        "shadow_full_forward": {
            "enabled": bool(getattr(args, "lrnode_eval_shadow_full_forward", 0)),
            "calls": int(lrnode_stats.get("shadow_full_forward_calls", 0)),
            "latent_mse": float(lrnode_stats.get("shadow_latent_mse", 0.0)),
            "latent_cos": float(lrnode_stats.get("shadow_latent_cos", 0.0)),
            "action_l1": float(lrnode_stats.get("shadow_action_l1", 0.0)),
            "action_l2": float(lrnode_stats.get("shadow_action_l2", 0.0)),
            "action_hold_l1": float(lrnode_stats.get("shadow_action_hold_l1", 0.0)),
            "pred_vs_hold_improvement": float(lrnode_stats.get("shadow_action_hold_l1", 0.0))
            - float(lrnode_stats.get("shadow_action_l1", 0.0)),
            "by_cache_age": lrnode_stats.get("shadow_by_age", {}),
        },
        "action_smoothness": {
            "action_delta_l2_mean": float(np.mean([m.get("avg_action_delta_l2", 0.0) for m in episode_metrics]))
            if episode_metrics else 0.0,
            "action_delta_l2_p95": float(np.mean([m.get("p95_action_delta_l2", 0.0) for m in episode_metrics]))
            if episode_metrics else 0.0,
            "action_jerk_l2_mean": float(np.mean([m.get("avg_action_jerk", 0.0) for m in episode_metrics]))
            if episode_metrics else 0.0,
            "action_jerk_l2_p95": float(np.mean([m.get("p95_action_jerk", 0.0) for m in episode_metrics]))
            if episode_metrics else 0.0,
            "arm_action_jerk": float(np.mean([m.get("arm_action_jerk", 0.0) for m in episode_metrics]))
            if episode_metrics else 0.0,
            "gripper_switch_rate": float(np.mean([m.get("gripper_switch_rate", 0.0) for m in episode_metrics]))
            if episode_metrics else 0.0,
            "trans_action_jerk": float(np.mean([m.get("trans_action_jerk", 0.0) for m in episode_metrics]))
            if episode_metrics else 0.0,
            "rot_action_jerk": float(np.mean([m.get("rot_action_jerk", 0.0) for m in episode_metrics]))
            if episode_metrics else 0.0,
        },
        "video": {
            "enabled": bool(int(os.environ.get("SAVE_VIDEO", "0"))) or bool(getattr(args, "save_video", False)),
            "all_ranks": (
                bool(int(os.environ.get("SAVE_VIDEO_ALL_RANKS", "0")))
                or bool(getattr(args, "save_video_all_ranks", False))
            ),
            "save_success": bool(int(os.environ.get("SAVE_VIDEO_SUCC", "1"))),
            "save_fail": bool(int(os.environ.get("SAVE_VIDEO_FAIL", "1"))),
            "fps": int(os.environ.get("VIDEO_FPS", getattr(args, "video_fps", 20))),
            "stride": int(os.environ.get("VIDEO_STRIDE", getattr(args, "video_stride", 1))),
        },
        "task_results": task_results,
    }
    safe_run_name = args.run_name.replace("/", "_")
    ckpt_tag = os.environ.get("CKPT_TAG", "").strip()
    tag = f"_{ckpt_tag}" if ckpt_tag else ""
    json_path = os.path.join(output_dir, f"{safe_run_name}_{args.finetune_type}{tag}_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    summary_path = os.path.join(output_dir, "eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    episode_csv_path = os.path.join(output_dir, "eval_episode_metrics.csv")
    _write_episode_metrics_csv(episode_csv_path, episode_metrics)
    latency_profile_path = os.path.join(output_dir, "eval_latency_profile.json")
    _write_latency_profile(latency_profile_path, episode_metrics)

    print(f"[LR-NODE eval] success_rate: {success_rate * 100:.1f}%")
    print(f"[LR-NODE eval] control_hz: {control_hz:.2f}")
    print(f"[LR-NODE eval] effective_action_hz: {control_hz:.2f}")
    print(f"[LR-NODE eval] effective_full_query_hz: {effective_full_query_hz:.2f}")
    print(f"[LR-NODE eval] effective_lrnode_update_hz: {effective_lrnode_update_hz:.2f}")
    print(f"[LR-NODE eval] full_forward_calls: {lrnode_stats['full_forward_calls']}")
    print(f"[LR-NODE eval] lrnode_update_calls: {lrnode_stats['lrnode_update_calls']}")
    print(f"[LR-NODE eval] avg_full_forward_latency_sec: {lrnode_stats['avg_full_forward_latency_sec']:.6f}")
    print(f"[LR-NODE eval] avg_lrnode_latency_sec: {lrnode_stats['avg_lrnode_latency_sec']:.6f}")
    print(f"[LR-NODE eval] effective_query_reduction: {lrnode_stats['effective_query_reduction'] * 100:.1f}%")
    print(f"[LR-NODE eval] full_query_reduction_ratio: {lrnode_stats['full_query_reduction_ratio'] * 100:.1f}%")
    print(f"[LR-NODE eval] effective_query_interval: {lrnode_stats['effective_query_interval']:.3f}")
    print(f"[LR-NODE eval] saved json: {json_path}")
    print(f"[LR-NODE eval] saved summary: {summary_path}")
    print(f"[LR-NODE eval] saved episode csv: {episode_csv_path}")
    print(f"[LR-NODE eval] saved latency profile: {latency_profile_path}")


def eval_one_epoch_libero_ddp(args, model, image_processor, tokenizer):
    cast_dtype = get_cast_dtype(args.precision)
    hist_len = args.sequence_length
    control_hz = _eval_control_hz()
    base_eval_max_steps = int(args.libero_eval_max_steps)
    scaled_eval_max_steps = _scaled_step_count(base_eval_max_steps, control_hz)
    if _is_rank0():
        print(
            f"[LIBERO ENV] EVAL_CONTROL_HZ={control_hz:.2f}, "
            f"base_control_hz={_base_control_hz():.2f}, "
            f"base_eval_max_steps={base_eval_max_steps}, "
            f"actual_eval_max_steps={scaled_eval_max_steps}, "
            f"scale_max_steps_with_hz={_env_flag('EVAL_SCALE_MAX_STEPS_WITH_HZ', '1')}"
        )
    args.libero_eval_max_steps = scaled_eval_max_steps
    wrapped_model = ModelWrapper(
        model,
        tokenizer,
        image_processor,
        cast_dtype,
        history_len=hist_len,
        use_ensembling=args.eval_libero_ensembling,
        ensembling_temp=args.ensembling_temp,
        libero_eval_max_steps=args.libero_eval_max_steps,
        action_pred_steps=args.action_pred_steps,
        gripper_width=args.gripper_width,
        use_lrnode_latent_update=args.use_lrnode_latent_update,
        lrnode_eval_skip_full_forward=args.lrnode_eval_skip_full_forward,
        lrnode_query_interval=args.lrnode_query_interval,
        lrnode_eval_step_log=args.lrnode_eval_step_log,
        lrnode_eval_shadow_full_forward=args.lrnode_eval_shadow_full_forward,
        lrnode_eval_refresh_policy=args.lrnode_eval_refresh_policy,
        lrnode_eval_max_full_forwards_per_episode=args.lrnode_eval_max_full_forwards_per_episode)
    evaluate_policy_ddp(args, wrapped_model)
