import sys, os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


def _configure_libero_renderer_backend():
    explicit_backend = os.environ.get("LIBERO_GL_BACKEND", "").strip().lower()
    mujoco_gl = os.environ.get("MUJOCO_GL", "").strip().lower()
    pyopengl_platform = os.environ.get("PYOPENGL_PLATFORM", "").strip().lower()

    if explicit_backend:
        backend = explicit_backend
        configured_by = "LIBERO_GL_BACKEND"
    else:
        configured = {value for value in (mujoco_gl, pyopengl_platform) if value}
        if len(configured) > 1:
            raise RuntimeError(
                "Conflicting renderer settings before LIBERO import: "
                f"MUJOCO_GL={mujoco_gl!r}, PYOPENGL_PLATFORM={pyopengl_platform!r}. "
                "Set LIBERO_GL_BACKEND explicitly to 'osmesa' or 'egl'."
            )
        backend = next(iter(configured), "osmesa")
        configured_by = (
            "MUJOCO_GL/PYOPENGL_PLATFORM" if configured else "default_osmesa"
        )

    if backend not in {"osmesa", "egl"}:
        raise ValueError(
            f"Unsupported LIBERO_GL_BACKEND={backend!r}; expected 'osmesa' or 'egl'."
        )

    # Renderer selection must happen before robosuite, MuJoCo, or PyOpenGL imports.
    os.environ["LIBERO_GL_BACKEND"] = backend
    os.environ["MUJOCO_GL"] = backend
    os.environ["PYOPENGL_PLATFORM"] = backend
    return {
        "schema_version": 1,
        "requested_backend": backend,
        "effective_backend": backend,
        "configured_by": configured_by,
        "mujoco_gl": backend,
        "pyopengl_platform": backend,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "render_gpu_device_id": None,
        "actual_gl_vendor": None,
        "actual_gl_renderer": None,
        "actual_gl_version": None,
        "backend_classification": "unverified",
        "actual_context_verified": False,
        "verification_method": "configured_environment_only",
    }


_RENDERER_BACKEND_METADATA = _configure_libero_renderer_backend()

import copy
import fcntl
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
from utils.lrnode_mechanism_utils import (
    action_second_differences,
    capture_rng_state,
    classify_transition,
    counterfactual_requires_skip_shadow,
    deterministic_step_seed,
    extract_simulator_signals,
    fuse_latents,
    gripper_summary,
    matched_random_latent,
    mix_action_tokens,
    preserve_rng_state,
    rng_states_equal,
    save_trace_episode,
    temporal_ensemble_probability,
)
from utils.train_utils import get_cast_dtype
from architectures.seer.adapters.latentloop_plan_continuation.trace_adapter import (
    save_plan_trace_episode,
)
from architectures.seer.adapters.latentloop_horizon_regeneration import (
    save_hierarchical_trace,
)
from methods.latentloop_horizon_regeneration import (
    PROVENANCE_LABELS,
    ExecutionLevel,
    ExecutionMode,
    HierarchicalSchedule,
    HorizonProvenance,
    LevelCallCounts,
    assert_level_call_contract,
)
from methods.latentloop_plan_continuation.action_correction import shift_action_horizon
from methods.latentloop_plan_continuation.feedback_source import FeedbackFeatureBuffer

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


def get_renderer_backend_metadata():
    return copy.deepcopy(_RENDERER_BACKEND_METADATA)


def _renderer_gpu_device_id(local_device_id: int) -> int:
    """Map a torch local rank to the physical EGL device expected by this robosuite."""
    if _RENDERER_BACKEND_METADATA["effective_backend"] != "egl":
        return int(local_device_id)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return int(local_device_id)
    try:
        physical_ids = [int(item.strip()) for item in visible.split(",")]
    except ValueError as exc:
        raise RuntimeError(
            "This robosuite EGL backend requires numeric CUDA_VISIBLE_DEVICES entries; "
            f"got {visible!r}."
        ) from exc
    if not 0 <= int(local_device_id) < len(physical_ids):
        raise RuntimeError(
            f"Local render device {local_device_id} is outside CUDA_VISIBLE_DEVICES={visible!r}."
        )
    return physical_ids[int(local_device_id)]


def _decode_gl_string(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def verify_renderer_backend(env, render_gpu_device_id: int):
    """Query the active OpenGL context and reject software fallback for strict EGL runs."""
    context = getattr(env.sim, "_render_context_offscreen", None)
    if context is None:
        raise RuntimeError("LIBERO environment has no offscreen render context to verify.")
    gl_context = getattr(context, "gl_ctx", None)
    if gl_context is None or not hasattr(gl_context, "make_current"):
        raise RuntimeError("LIBERO offscreen context does not expose an active GL context.")
    gl_context.make_current()

    from OpenGL import GL

    vendor = _decode_gl_string(GL.glGetString(GL.GL_VENDOR))
    renderer = _decode_gl_string(GL.glGetString(GL.GL_RENDERER))
    version = _decode_gl_string(GL.glGetString(GL.GL_VERSION))
    combined = " ".join(value or "" for value in (vendor, renderer)).lower()
    is_software = any(
        token in combined
        for token in ("llvmpipe", "softpipe", "software rasterizer", "mesa/x.org")
    )
    requested = _RENDERER_BACKEND_METADATA["requested_backend"]
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if is_software:
        classification = "software_osmesa"
    elif requested == "egl":
        classification = "hardware_egl"
    else:
        classification = "nonsoftware_context"

    _RENDERER_BACKEND_METADATA.update(
        {
            "render_gpu_device_id": int(render_gpu_device_id),
            "process_rank": int(rank),
            "actual_gl_vendor": vendor,
            "actual_gl_renderer": renderer,
            "actual_gl_version": version,
            "backend_classification": classification,
            "actual_context_verified": bool(vendor and renderer and version),
            "verification_method": "active_opengl_context_glGetString",
        }
    )

    strict = os.environ.get("LIBERO_GL_REQUIRE_ACTUAL", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if strict and not _RENDERER_BACKEND_METADATA["actual_context_verified"]:
        raise RuntimeError(
            "Strict renderer verification failed: GL_VENDOR/GL_RENDERER/GL_VERSION "
            f"were not all available: {get_renderer_backend_metadata()}"
        )
    if strict and requested == "egl" and is_software:
        raise RuntimeError(
            "Strict EGL verification detected a software OpenGL renderer: "
            f"{get_renderer_backend_metadata()}"
        )

    if not getattr(verify_renderer_backend, "_printed_ranks", None):
        verify_renderer_backend._printed_ranks = set()
    if rank not in verify_renderer_backend._printed_ranks:
        print(
            "[LIBERO RENDERER] "
            f"rank={rank} backend={requested} render_gpu_device_id={render_gpu_device_id} "
            f"vendor={vendor!r} renderer={renderer!r} version={version!r} "
            f"classification={classification}"
        )
        verify_renderer_backend._printed_ranks.add(rank)
    return get_renderer_backend_metadata()


def _atomic_write_json(path: Path, payload) -> None:
    if isinstance(payload, dict) and "renderer_backend" not in payload:
        payload = {**payload, "renderer_backend": get_renderer_backend_metadata()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp_path, path)


def _write_live_eval_progress(
    args,
    local_results,
    local_eval_ids,
    total_sequences: int,
    local_assigned: int,
    last_eval_id: int,
) -> None:
    """Persist asynchronous completed-episode SR without synchronizing DDP ranks."""
    log_dir = os.environ.get("LOG_DIR")
    if not log_dir:
        return

    try:
        rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
        world_size = int(torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1
        analysis_dir = Path(log_dir) / "analysis"
        task_progress = {}
        for eval_id, success in zip(local_eval_ids, local_results):
            task_id = int(eval_id) // int(num_eval_episodes)
            item = task_progress.setdefault(str(task_id), {"completed": 0, "successes": 0})
            item["completed"] += 1
            item["successes"] += int(bool(success))

        local_completed = len(local_results)
        local_successes = sum(int(bool(value)) for value in local_results)
        rank_payload = {
            "schema_version": 1,
            "status": "rank_complete" if local_completed == local_assigned else "running",
            "run_name": str(getattr(args, "run_name", os.environ.get("RUN_NAME", ""))),
            "checkpoint_tag": os.environ.get("CKPT_TAG", ""),
            "rank": rank,
            "world_size": world_size,
            "total_sequences": int(total_sequences),
            "local_assigned": int(local_assigned),
            "local_completed": local_completed,
            "local_successes": local_successes,
            "local_completed_success_rate": (
                float(local_successes) / local_completed if local_completed else 0.0
            ),
            "last_eval_id": int(last_eval_id),
            "last_task_id": int(last_eval_id) // int(num_eval_episodes),
            "last_episode_id": int(last_eval_id) % int(num_eval_episodes),
            "last_success": bool(local_results[-1]),
            "task_progress": task_progress,
            "updated_at_unix_s": time.time(),
        }
        rank_path = analysis_dir / f"eval_progress_rank{rank}.json"
        _atomic_write_json(rank_path, rank_payload)

        # File locking avoids competing aggregate writers while keeping the
        # policy evaluation itself free of per-episode DDP collectives.
        analysis_dir.mkdir(parents=True, exist_ok=True)
        lock_path = analysis_dir / ".eval_progress.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            rank_payloads = []
            for rank_id in range(world_size):
                path = analysis_dir / f"eval_progress_rank{rank_id}.json"
                if not path.is_file():
                    continue
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if int(payload.get("total_sequences", -1)) != int(total_sequences):
                    continue
                rank_payloads.append(payload)

            completed = sum(int(item["local_completed"]) for item in rank_payloads)
            successes = sum(int(item["local_successes"]) for item in rank_payloads)
            aggregate_tasks = {}
            for payload in rank_payloads:
                for task_id, item in payload.get("task_progress", {}).items():
                    target = aggregate_tasks.setdefault(task_id, {"completed": 0, "successes": 0})
                    target["completed"] += int(item["completed"])
                    target["successes"] += int(item["successes"])
            for item in aggregate_tasks.values():
                item["completed_success_rate"] = (
                    float(item["successes"]) / item["completed"] if item["completed"] else 0.0
                )

            aggregate = {
                "schema_version": 1,
                "status": "complete" if completed == int(total_sequences) else "running",
                "metric_scope": "completed_episodes_only",
                "total_sequences": int(total_sequences),
                "completed": completed,
                "successes": successes,
                "completed_success_rate": float(successes) / completed if completed else 0.0,
                "ranks_reporting": len(rank_payloads),
                "world_size": world_size,
                "task_progress": aggregate_tasks,
                "updated_at_unix_s": time.time(),
            }
            _atomic_write_json(analysis_dir / "eval_progress.json", aggregate)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

        print(
            f"[EVAL PROGRESS] completed={completed}/{total_sequences} "
            f"successes={successes} completed_sr={100.0 * aggregate['completed_success_rate']:.2f}% "
            f"ranks={len(rank_payloads)}/{world_size} last_rank={rank} "
            f"last_task={rank_payload['last_task_id']} last_episode={rank_payload['last_episode_id']} "
            f"last_success={int(rank_payload['last_success'])}",
            flush=True,
        )
    except Exception as exc:
        # Progress reporting must never invalidate an otherwise valid rollout.
        print(f"[EVAL PROGRESS][WARN] failed to update live progress: {exc}", flush=True)


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
                 lrnode_eval_profile_full_action_head=0,
                 lrnode_eval_refresh_policy="periodic", lrnode_eval_max_full_forwards_per_episode=1,
                 lrnode_eval_ablation_mode="stepwise", lrnode_no_delta_mode="zero",
                 lrnode_chunk_token_policy="skip_only", lrnode_mechanism_trace=0,
                 lrnode_trace_save_latents=0, lrnode_trace_episode_limit=0,
                 lrnode_trace_output_dir="", lrnode_counterfactual_mode="standard",
                 lrnode_counterfactual_mix_stage="pre_ensemble",
                 lrnode_latent_fusion_alpha=0.0, lrnode_latent_fusion_mode="every_step",
                 lrnode_matched_random_seed=20260724,
                 lrnode_matched_random_norm_mode="per_token",
                 lrnode_every_step_filter_mode="off",
                 lrnode_every_step_filter_alpha=0.5,
                 lrnode_every_step_filter_beta=0.5,
                 lrnode_every_step_filter_diagnostics=0,
                 latentloop_segment_grid_enable=0,
                 latentloop_feedback_schedule="dense",
                 latentloop_same_input_stochasticity_repeats=0,
                 latentloop_same_input_stochasticity_output="",
                 latentloop_plan_trace=0,
                 latentloop_plan_trace_save_latents=0,
                 latentloop_plan_trace_output_dir="",
                 latentloop_plan_trace_row_id="",
                 latentloop_plan_trace_paired_group="",
                 latentloop_feedback_source="current",
                 latentloop_plan_adapter_mode="off",
                 joint_latent_action_surrogate_mode="off",
                 joint_error_trace=0,
                 joint_error_trace_output_dir="",
                 joint_force_exact_action_head=0,
                 latentloop_hierarchical_mode="off",
                 latentloop_hierarchical_full_interval=8,
                 latentloop_hierarchical_regeneration_interval=3,
                 latentloop_hierarchical_trace=0,
                 latentloop_hierarchical_trace_output_dir="",
                 latentloop_hierarchical_assert_invariants=1,
                 evaluation_seed=0):
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
        self.lrnode_eval_profile_full_action_head = bool(lrnode_eval_profile_full_action_head)
        self.latentloop_plan_trace = bool(latentloop_plan_trace)
        self.latentloop_plan_trace_save_latents = bool(
            latentloop_plan_trace_save_latents
        )
        self.latentloop_plan_trace_output_dir = str(
            latentloop_plan_trace_output_dir
        ).strip()
        self.latentloop_plan_trace_row_id = str(latentloop_plan_trace_row_id).strip()
        self.latentloop_plan_trace_paired_group = str(
            latentloop_plan_trace_paired_group
        ).strip()
        self.lrnode_mechanism_trace = bool(lrnode_mechanism_trace)
        self.lrnode_trace_save_latents = bool(lrnode_trace_save_latents)
        self.lrnode_trace_episode_limit = max(0, int(lrnode_trace_episode_limit))
        self.lrnode_trace_output_dir = str(lrnode_trace_output_dir).strip()
        self.lrnode_counterfactual_mode = str(lrnode_counterfactual_mode)
        self.lrnode_counterfactual_mix_stage = str(lrnode_counterfactual_mix_stage)
        self.lrnode_latent_fusion_alpha = float(lrnode_latent_fusion_alpha)
        self.lrnode_latent_fusion_mode = str(lrnode_latent_fusion_mode)
        self.lrnode_matched_random_seed = int(lrnode_matched_random_seed)
        self.lrnode_matched_random_norm_mode = str(lrnode_matched_random_norm_mode)
        self.lrnode_every_step_filter_mode = str(lrnode_every_step_filter_mode)
        self.lrnode_every_step_filter_alpha = float(lrnode_every_step_filter_alpha)
        self.lrnode_every_step_filter_beta = float(lrnode_every_step_filter_beta)
        self.lrnode_every_step_filter_diagnostics = bool(
            lrnode_every_step_filter_diagnostics
        )
        self.latentloop_segment_grid_enable = bool(
            latentloop_segment_grid_enable
        )
        self.latentloop_feedback_schedule = str(
            latentloop_feedback_schedule
        )
        self.latentloop_segment_executor = None
        self.latentloop_same_input_stochasticity_repeats = int(
            latentloop_same_input_stochasticity_repeats
        )
        self.latentloop_same_input_stochasticity_output = str(
            latentloop_same_input_stochasticity_output
        ).strip()
        self.latentloop_same_input_stochasticity_done = False
        self.latentloop_feedback_source = str(latentloop_feedback_source)
        self.latentloop_feedback_buffer = FeedbackFeatureBuffer(
            self.latentloop_feedback_source
        )
        self.latentloop_plan_adapter_mode = str(latentloop_plan_adapter_mode)
        self.joint_latent_action_surrogate_mode = str(
            joint_latent_action_surrogate_mode
        )
        if self.joint_latent_action_surrogate_mode not in {"off", "joint", "wide"}:
            raise ValueError("Unknown joint_latent_action_surrogate_mode")
        if (
            self.joint_latent_action_surrogate_mode != "off"
            and str(latentloop_hierarchical_mode) != "off"
        ):
            raise ValueError("Joint mode cannot be combined with a legacy hierarchy")
        self.joint_error_trace = bool(joint_error_trace)
        self.joint_error_trace_output_dir = str(joint_error_trace_output_dir).strip()
        self.joint_force_exact_action_head = bool(joint_force_exact_action_head)
        if self.joint_force_exact_action_head and self.joint_latent_action_surrogate_mode != "joint":
            raise ValueError("joint_force_exact_action_head requires joint mode")
        joint_hierarchy_enabled = (
            self.joint_latent_action_surrogate_mode != "off"
            and self.lrnode_eval_skip_full_forward
        )
        self.latentloop_hierarchical_mode = (
            ("pure_latentloop" if self.joint_force_exact_action_head else "hybrid")
            if joint_hierarchy_enabled
            and self.joint_latent_action_surrogate_mode == "joint"
            else (
                "pure_latentloop"
                if joint_hierarchy_enabled
                and self.joint_latent_action_surrogate_mode == "wide"
                else str(latentloop_hierarchical_mode)
            )
        )
        self.latentloop_hierarchical_schedule = HierarchicalSchedule(
            self.latentloop_hierarchical_mode,
            full_interval=latentloop_hierarchical_full_interval,
            regeneration_interval=latentloop_hierarchical_regeneration_interval,
        )
        self.latentloop_hierarchical_trace = bool(latentloop_hierarchical_trace)
        self.latentloop_hierarchical_trace_output_dir = str(
            latentloop_hierarchical_trace_output_dir
        ).strip()
        self.latentloop_hierarchical_assert_invariants = bool(
            latentloop_hierarchical_assert_invariants
        )
        self.evaluation_seed = int(evaluation_seed)
        if self.latentloop_hierarchical_schedule.enabled:
            if int(action_pred_steps) != 3:
                raise ValueError("The source-locked hierarchical protocol requires P=3")
            if (
                self.latentloop_hierarchical_mode != "full_seer"
                and self.lrnode_query_interval
                != self.latentloop_hierarchical_schedule.full_interval
            ):
                raise ValueError(
                    "lrnode_query_interval must equal latentloop_hierarchical_full_interval"
                )
            if str(lrnode_eval_refresh_policy) != "periodic":
                raise ValueError("Hierarchical execution requires periodic full refresh")
            if bool(latentloop_segment_grid_enable):
                raise ValueError("Hierarchical execution cannot be combined with segment grid")
            if str(lrnode_eval_ablation_mode) != "stepwise":
                raise ValueError("Hierarchical execution requires stepwise ablation mode")
            if str(lrnode_counterfactual_mode) != "standard":
                raise ValueError("Hierarchical execution cannot alter action counterfactuals")
            if str(lrnode_every_step_filter_mode) != "off":
                raise ValueError("Hierarchical execution cannot use every-step latent filters")
            if self.latentloop_feedback_source != "current":
                raise ValueError("Hierarchical execution requires current observation feedback")
            if self.lrnode_eval_shadow_full_forward or self.lrnode_mechanism_trace:
                raise ValueError(
                    "Hierarchical execution uses its dedicated deterministic trace; "
                    "disable shadow full-forward and legacy mechanism tracing"
                )
            needs_action_adapter = self.latentloop_hierarchical_mode in {
                "pure_action_correction",
                "hybrid",
            } and self.joint_latent_action_surrogate_mode == "off"
            if needs_action_adapter and self.latentloop_plan_adapter_mode != "action_correction":
                raise ValueError(
                    f"{self.latentloop_hierarchical_mode} requires action_correction adapter"
                )
        if self.latentloop_plan_adapter_mode != "off":
            base_model = self._base_model()
            if not hasattr(base_model, "latentloop_plan_adapter"):
                raise RuntimeError(
                    "Plan adapter mode was requested but the model has no attached adapter"
                )
            attached_mode = str(base_model.latentloop_plan_adapter.mode)
            if attached_mode != self.latentloop_plan_adapter_mode:
                raise RuntimeError(
                    f"Attached plan adapter mode={attached_mode}, requested="
                    f"{self.latentloop_plan_adapter_mode}"
                )
            if not (self.use_lrnode_latent_update and self.lrnode_eval_skip_full_forward):
                raise ValueError(
                    "Plan adapter evaluation requires the existing periodic skip schedule"
                )
        if self.joint_latent_action_surrogate_mode != "off":
            base_model = self._base_model()
            if not hasattr(base_model, "joint_latent_action_surrogate"):
                raise RuntimeError("Joint mode requested without an attached joint adapter")
            attached_mode = str(base_model.joint_latent_action_surrogate.mode)
            if attached_mode != self.joint_latent_action_surrogate_mode:
                raise RuntimeError(
                    f"Attached joint mode={attached_mode}, requested={self.joint_latent_action_surrogate_mode}"
                )
            if self.latentloop_plan_adapter_mode != "off":
                raise ValueError("Joint mode cannot use a legacy plan adapter")
        if self.latentloop_feedback_source == "time_shifted":
            if not (self.use_lrnode_latent_update and self.lrnode_eval_skip_full_forward):
                raise ValueError("time-shifted feedback requires the LatentLoop skip path")
            if self.latentloop_plan_adapter_mode != "off":
                raise ValueError(
                    "time-shifted feedback isolates the recurrent LatentLoop updater and "
                    "cannot be combined with a matched baseline adapter"
                )
        if self.latentloop_same_input_stochasticity_repeats < 0:
            raise ValueError(
                "latentloop_same_input_stochasticity_repeats must be non-negative"
            )
        self.lrnode_every_step_filter = None
        if self.lrnode_every_step_filter_mode != "off":
            workspace_root = str(Path(__file__).resolve().parents[4])
            if workspace_root not in sys.path:
                sys.path.insert(0, workspace_root)
            from architectures.seer.adapters.latent_prediction_correction import (
                SeerLatentFilterAdapter,
            )

            self.lrnode_every_step_filter = SeerLatentFilterAdapter(
                mode=self.lrnode_every_step_filter_mode,
                alpha=self.lrnode_every_step_filter_alpha,
                beta=self.lrnode_every_step_filter_beta,
            )
            if not self.use_lrnode_latent_update:
                raise ValueError(
                    "lrnode_every_step_filter_mode requires "
                    "use_lrnode_latent_update=1"
                )
            if self.lrnode_eval_skip_full_forward:
                raise ValueError(
                    "Every-step latent filtering requires "
                    "lrnode_eval_skip_full_forward=0"
                )
            if self.lrnode_query_interval != 1:
                raise ValueError(
                    "Every-step latent filtering requires lrnode_query_interval=1"
                )
            if self.lrnode_counterfactual_mode != "standard":
                raise ValueError(
                    "Every-step latent filtering cannot be combined with "
                    "lrnode_counterfactual_mode"
                )
            if self.lrnode_eval_shadow_full_forward:
                raise ValueError(
                    "Every-step latent filtering already executes full Seer every step; "
                    "set lrnode_eval_shadow_full_forward=0"
                )
        valid_counterfactual_modes = {
            "standard",
            "full_arm_full_gripper",
            "lr_arm_lr_gripper",
            "lr_arm_full_gripper",
            "full_arm_lr_gripper",
            "latent_fusion",
            "matched_random",
        }
        if self.lrnode_counterfactual_mode not in valid_counterfactual_modes:
            raise ValueError(
                f"Unknown lrnode_counterfactual_mode={self.lrnode_counterfactual_mode}; "
                f"expected one of {sorted(valid_counterfactual_modes)}"
            )
        if self.lrnode_counterfactual_mix_stage != "pre_ensemble":
            raise ValueError("Only pre_ensemble arm/gripper mixing is implemented")
        if not 0.0 <= self.lrnode_latent_fusion_alpha <= 1.0:
            raise ValueError("lrnode_latent_fusion_alpha must be in [0, 1]")
        if self.lrnode_latent_fusion_mode not in {"every_step", "soft_reset_only"}:
            raise ValueError("lrnode_latent_fusion_mode must be every_step or soft_reset_only")
        if self.lrnode_matched_random_norm_mode not in {"per_token", "global"}:
            raise ValueError("lrnode_matched_random_norm_mode must be per_token or global")
        if self.lrnode_counterfactual_mode != "standard" and not self.lrnode_eval_shadow_full_forward:
            raise ValueError(
                "Non-standard lrnode_counterfactual_mode requires lrnode_shadow_full_forward=1"
            )
        self.lrnode_eval_refresh_policy = str(lrnode_eval_refresh_policy)
        if self.lrnode_eval_refresh_policy not in {"periodic", "first_only", "fixed_budget"}:
            raise ValueError(f"Unknown lrnode_eval_refresh_policy={self.lrnode_eval_refresh_policy}")
        self.lrnode_eval_max_full_forwards_per_episode = max(
            1, int(lrnode_eval_max_full_forwards_per_episode)
        )
        self.lrnode_eval_ablation_mode = str(lrnode_eval_ablation_mode)
        valid_ablation_modes = {"stepwise", "hold_action", "hold_latent", "seer_token_chunk", "no_delta"}
        if self.lrnode_eval_ablation_mode not in valid_ablation_modes:
            raise ValueError(
                f"Unknown lrnode_eval_ablation_mode={self.lrnode_eval_ablation_mode}; "
                f"expected one of {sorted(valid_ablation_modes)}"
            )
        if (
            self.lrnode_counterfactual_mode != "standard"
            and self.lrnode_eval_ablation_mode != "stepwise"
        ):
            raise ValueError(
                "Counterfactual mechanism modes require lrnode_eval_ablation_mode=stepwise"
            )
        self.lrnode_no_delta_mode = str(lrnode_no_delta_mode)
        if self.lrnode_no_delta_mode not in {"zero", "learned_constant", "previous"}:
            raise ValueError(f"Unknown lrnode_no_delta_mode={self.lrnode_no_delta_mode}")
        if self.lrnode_eval_ablation_mode == "no_delta" and self.lrnode_no_delta_mode != "zero":
            raise NotImplementedError(
                "lrnode_eval_ablation_mode=no_delta currently implements only "
                "lrnode_no_delta_mode=zero"
            )
        self.lrnode_chunk_token_policy = str(lrnode_chunk_token_policy)
        if self.lrnode_chunk_token_policy != "skip_only":
            raise NotImplementedError(
                "Only lrnode_chunk_token_policy=skip_only is implemented for "
                "seer_token_chunk ablation"
            )
        if self.latentloop_segment_grid_enable:
            if not (
                self.use_lrnode_latent_update
                and self.lrnode_eval_skip_full_forward
            ):
                raise ValueError(
                    "latentloop_segment_grid_enable=1 requires the existing "
                    "LatentLoop skip path"
                )
            if self.lrnode_eval_refresh_policy != "periodic":
                raise ValueError(
                    "LatentLoop segment-grid execution requires periodic refresh"
                )
            if self.lrnode_eval_ablation_mode != "stepwise":
                raise ValueError(
                    "Feedback schedules require lrnode_eval_ablation_mode=stepwise; "
                    "hold and replay baselines run with the segment-grid flag off"
                )
            if self.lrnode_every_step_filter is not None:
                raise ValueError(
                    "LatentLoop segment-grid execution cannot be combined with "
                    "every-step latent filtering"
                )
            if self.lrnode_counterfactual_mode != "standard":
                raise ValueError(
                    "LatentLoop segment-grid execution cannot be combined with "
                    "counterfactual execution modes"
                )
            workspace_root = str(Path(__file__).resolve().parents[4])
            if workspace_root not in sys.path:
                sys.path.insert(0, workspace_root)
            from architectures.seer.adapters.latentloop_segment_grid import (
                LatentLoopSegmentExecutor,
            )

            self.latentloop_segment_executor = LatentLoopSegmentExecutor(
                segment_length=self.lrnode_query_interval,
                feedback_schedule=self.latentloop_feedback_schedule,
            )
        self.lrnode_episode_full_forward_calls = 0
        self.lrnode_cached_latent = None
        self.lrnode_cached_image_primary = None
        self.lrnode_cached_image_wrist = None
        self.lrnode_cached_state = None
        self.lrnode_cached_action_tokens = None
        self.lrnode_cached_action_arm = None
        self.lrnode_cached_gripper_logit = None
        self.lrnode_cached_env_action = None
        self.lrnode_cached_age = 0
        self.lrnode_last_full_timestep = None
        self.hierarchical_regeneration_age = 0
        self.hierarchical_latent_cache_version = 0
        self.hierarchical_action_cache_generation = -1
        self.hierarchical_action_cache_input_generation = -1
        self.hierarchical_provenance = HorizonProvenance(self.action_pred_steps)
        self.hierarchical_provenance_buffer = None
        self.hierarchical_provenance_presence = None
        self.hierarchical_level0_calls = 0
        self.hierarchical_level1_calls = 0
        self.hierarchical_level2_calls = 0
        self.hierarchical_latent_updater_calls = 0
        self.hierarchical_action_correction_calls = 0
        self.hierarchical_action_head_calls = 0
        self.joint_surrogate_calls = 0
        self.joint_surrogate_latency_sum = 0.0
        self.joint_diagnostic_action_head_calls = 0
        self.joint_diagnostic_full_forward_calls = 0
        self.joint_diagnostic_full_forward_latency_sum = 0.0
        self.joint_anchor_arm = None
        self.joint_anchor_gripper_logit = None
        self.joint_anchor_latent = None
        self.joint_anchor_timestep = None
        self.joint_anchor_generation = -1
        self.hierarchical_action_correction_latency_sum = 0.0
        self.full_forward_calls = 0
        self.lrnode_update_calls = 0
        self.fast_encoder_calls = 0
        self.action_head_calls = 0
        self.hold_action_steps = 0
        self.hold_latent_steps = 0
        self.chunk_token_steps = 0
        self.no_delta_steps = 0
        self.observation_conditioned_update_calls = 0
        self.zero_feature_update_calls = 0
        self.observation_cache_advance_calls = 0
        self.full_forward_latency_sum = 0.0
        self.full_action_head_latency_sum = 0.0
        self.full_non_action_head_latency_sum = 0.0
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
        self.counterfactual_arm_lr_steps = 0
        self.counterfactual_arm_full_steps = 0
        self.counterfactual_gripper_lr_steps = 0
        self.counterfactual_gripper_full_steps = 0
        self.counterfactual_latent_fusion_steps = 0
        self.counterfactual_matched_random_steps = 0
        self.every_step_filter_prior_calls = 0
        self.every_step_filter_fusion_calls = 0
        self.every_step_filter_action_head_calls = 0
        self.every_step_filter_diagnostic_action_head_calls = 0
        self.every_step_filter_prior_latency_sum = 0.0
        self.every_step_filter_fusion_latency_sum = 0.0
        self.every_step_filter_action_head_latency_sum = 0.0
        self.every_step_filter_diagnostic_latency_sum = 0.0
        self.every_step_filter_rng_checks = 0
        self.every_step_filter_rng_failures = 0
        self.current_step_records = []
        self.current_trace_scalars = []
        self.current_trace_tensors = []
        self.current_plan_trace_scalars = []
        self.current_plan_trace_tensors = []
        self.previous_raw_primary = None
        self.previous_raw_wrist = None
        self.previous_raw_proprio = None
        self.latentloop_anchor_latent = None
        self.latentloop_anchor_image_primary = None
        self.latentloop_anchor_image_wrist = None
        self.latentloop_anchor_state = None
        self.episode_metrics = []
        self.current_episode_start_time = None
        self.current_task_id = -1
        self.current_task_name = ""
        self.current_episode_id = -1
        self.trace_episode_count = 0
        self.previous_step_was_full = None
        self.last_action = None
        self.last_action_delta = None
        if self.lrnode_every_step_filter is not None:
            self.lrnode_every_step_filter.reset()
        setattr(self._base_model(), "profile_full_action_head", self.lrnode_eval_profile_full_action_head)
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
            self.shadow_all_time_actions = torch.zeros_like(self.all_time_actions)
            if self.latentloop_hierarchical_schedule.enabled:
                self.hierarchical_provenance_buffer = np.zeros(
                    (
                        self.libero_eval_max_steps,
                        self.libero_eval_max_steps + self.action_pred_steps,
                        len(PROVENANCE_LABELS),
                    ),
                    dtype=np.float64,
                )
                self.hierarchical_provenance_presence = np.zeros(
                    (
                        self.libero_eval_max_steps,
                        self.libero_eval_max_steps + self.action_pred_steps,
                    ),
                    dtype=bool,
                )
        else:
            self.shadow_all_time_actions = None

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
        self.lrnode_cached_action_tokens = None
        self.lrnode_cached_action_arm = None
        self.lrnode_cached_gripper_logit = None
        self.lrnode_cached_env_action = None
        self.lrnode_cached_age = 0
        self.lrnode_last_full_timestep = None
        self.hierarchical_regeneration_age = 0
        self.hierarchical_latent_cache_version = 0
        self.hierarchical_action_cache_generation = -1
        self.hierarchical_action_cache_input_generation = -1
        self.hierarchical_provenance = HorizonProvenance(self.action_pred_steps)
        self.joint_anchor_arm = None
        self.joint_anchor_gripper_logit = None
        self.joint_anchor_latent = None
        self.joint_anchor_timestep = None
        self.joint_anchor_generation = -1
        self.lrnode_episode_full_forward_calls = 0
        self.current_step_records = []
        self.current_trace_scalars = []
        self.current_trace_tensors = []
        self.current_plan_trace_scalars = []
        self.current_plan_trace_tensors = []
        self.previous_raw_primary = None
        self.previous_raw_wrist = None
        self.previous_raw_proprio = None
        self.latentloop_anchor_latent = None
        self.latentloop_anchor_image_primary = None
        self.latentloop_anchor_image_wrist = None
        self.latentloop_anchor_state = None
        self.latentloop_feedback_buffer.reset()
        self.current_episode_start_time = time.perf_counter()
        self.previous_step_was_full = None
        self.last_action = None
        self.last_action_delta = None
        if self.lrnode_every_step_filter is not None:
            self.lrnode_every_step_filter.reset()
        if self.use_ensembling:
            self.all_time_actions = torch.zeros(
                [
                    self.libero_eval_max_steps,
                    self.libero_eval_max_steps + self.action_pred_steps,
                    7,
                ]
            ).to(self.device)
            self.shadow_all_time_actions = torch.zeros_like(self.all_time_actions)
            if self.latentloop_hierarchical_schedule.enabled:
                self.hierarchical_provenance_buffer = np.zeros(
                    (
                        self.libero_eval_max_steps,
                        self.libero_eval_max_steps + self.action_pred_steps,
                        len(PROVENANCE_LABELS),
                    ),
                    dtype=np.float64,
                )
                self.hierarchical_provenance_presence = np.zeros(
                    (
                        self.libero_eval_max_steps,
                        self.libero_eval_max_steps + self.action_pred_steps,
                    ),
                    dtype=bool,
                )
        else:
            self.shadow_all_time_actions = None

        self.cnt += 1

    def set_episode_context(self, task, env):
        self.current_task_id = int(getattr(env, "task_id", -1))
        self.current_task_name = str(getattr(task, "name", getattr(env, "task_name", "")))
        self.current_episode_id = int(getattr(env, "exp_id", -1))

    def _base_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _sync_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _selected_step(self, num_step):
        if num_step < self.history_len:
            return num_step - 1
        return -1

    def _action_sequence_to_probability_action(self, action_seq, timestep, ensemble_buffer):
        if action_seq.dim() != 3 or action_seq.shape[0] != 1 or action_seq.shape[-1] != 7:
            raise RuntimeError(f"Expected action sequence [1, action_pred_steps, 7], got {tuple(action_seq.shape)}")

        if self.use_ensembling:
            if ensemble_buffer is None:
                raise RuntimeError("Temporal ensembling requires an explicit branch-local buffer")
            action, candidate_count = temporal_ensemble_probability(
                action_seq,
                timestep,
                ensemble_buffer,
                self.ensembling_temp,
            )
        else:
            action = action_seq[:, 0]
            candidate_count = 1
        return action, candidate_count

    def _threshold_probability_action(self, action):
        action = torch.concat((action[:, :6], action[:, 6:] > 0.5), dim=-1)
        action[:, -1] = (action[:, -1] - 0.5) * 2
        action = action.detach().cpu().numpy()[-1]
        if action.shape != (7,):
            raise RuntimeError(f"LIBERO action must have shape (7,), got {action.shape}")
        return action

    def _action_sequence_to_env_action(self, action_seq, timestep):
        action_probability, candidate_count = self._action_sequence_to_probability_action(
            action_seq,
            timestep,
            self.all_time_actions if self.use_ensembling else None,
        )
        return (
            self._threshold_probability_action(action_probability),
            action_probability.detach(),
            candidate_count,
        )

    def _shadow_action_sequence_to_env_action(self, action_seq, timestep):
        action_probability, candidate_count = self._action_sequence_to_probability_action(
            action_seq,
            timestep,
            self.shadow_all_time_actions if self.use_ensembling else None,
        )
        return (
            self._threshold_probability_action(action_probability),
            action_probability.detach(),
            candidate_count,
        )

    def _raw_action_token_to_env_action(self, action_token):
        if action_token.dim() == 3:
            if action_token.shape[0] != 1 or action_token.shape[1] != 1 or action_token.shape[-1] != 7:
                raise RuntimeError(f"Expected action token [1, 1, 7], got {tuple(action_token.shape)}")
            action_token = action_token[:, 0]
        if action_token.dim() != 2 or action_token.shape[0] != 1 or action_token.shape[-1] != 7:
            raise RuntimeError(f"Expected action token [1, 7], got {tuple(action_token.shape)}")
        action = torch.concat((action_token[:, :6], action_token[:, 6:] > 0.5), dim=-1)
        action[:, -1] = (action[:, -1] - 0.5) * 2
        action = action.detach().cpu().numpy()[-1]
        if action.shape != (7,):
            raise RuntimeError(f"LIBERO action must have shape (7,), got {action.shape}")
        return action

    def _run_same_input_stochasticity_sanity(
        self,
        input_image_primary,
        input_image_wrist,
        input_state,
        input_text_token,
        selected_step,
        timestep,
    ):
        """Repeat one fixed preprocessed full-Seer input without changing rollout RNG."""

        repeats = self.latentloop_same_input_stochasticity_repeats
        if repeats <= 0 or self.latentloop_same_input_stochasticity_done:
            return None
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
            raise RuntimeError(
                "Same-input stochasticity sanity requires NODE_NUM=1 so one fixed "
                "input produces exactly one artifact"
            )

        workspace_root = str(Path(__file__).resolve().parents[4])
        if workspace_root not in sys.path:
            sys.path.insert(0, workspace_root)
        from methods.latentloop_segment_grid.serialization import atomic_write_json
        from methods.latentloop_segment_grid.stochasticity import (
            array_fingerprint,
            summarize_repeated_outputs,
        )

        latents = []
        raw_actions = []
        executed_actions = []
        with preserve_rng_state(include_cuda=True):
            for _ in range(repeats):
                outputs = self.model(
                    image_primary=input_image_primary,
                    image_wrist=input_image_wrist,
                    state=input_state,
                    text_token=input_text_token,
                    action=torch.zeros(
                        1, self.history_len, 7, device=input_state.device
                    ),
                    return_action_latent=True,
                )
                if not isinstance(outputs, dict):
                    raise RuntimeError(
                        "return_action_latent=True must return the Seer output dictionary"
                    )
                latent = outputs["action_latent"][:, selected_step]
                raw_action = torch.cat(
                    [
                        outputs["arm_pred_action"][:, selected_step],
                        outputs["gripper_pred_action"][:, selected_step],
                    ],
                    dim=-1,
                )
                if self.use_ensembling:
                    ensemble_buffer = torch.zeros_like(self.all_time_actions)
                else:
                    ensemble_buffer = None
                probability, _ = self._action_sequence_to_probability_action(
                    raw_action,
                    timestep,
                    ensemble_buffer,
                )
                executed = self._threshold_probability_action(probability)
                latents.append(latent.detach().float().cpu().numpy())
                raw_actions.append(raw_action.detach().float().cpu().numpy())
                executed_actions.append(np.asarray(executed, dtype=np.float32))

        payload = {
            "schema_version": 1,
            "diagnostic": "same_preprocessed_input_full_seer_stochasticity",
            "repeat_count": repeats,
            "model_training": bool(self.model.training),
            "fixed_timestep": int(timestep),
            "selected_context_step": int(selected_step),
            "temporal_ensemble_enabled": int(self.use_ensembling),
            "input_shapes": {
                "image_primary": list(input_image_primary.shape),
                "image_wrist": list(input_image_wrist.shape),
                "state": list(input_state.shape),
                "text_token": list(input_text_token.shape),
            },
            "input_sha256": {
                "image_primary": array_fingerprint(
                    input_image_primary.detach().cpu().numpy()
                ),
                "image_wrist": array_fingerprint(
                    input_image_wrist.detach().cpu().numpy()
                ),
                "state": array_fingerprint(input_state.detach().cpu().numpy()),
                "text_token": array_fingerprint(
                    input_text_token.detach().cpu().numpy()
                ),
            },
            **summarize_repeated_outputs(
                latents, raw_actions, executed_actions
            ),
        }
        output_path = self.latentloop_same_input_stochasticity_output
        if not output_path:
            log_dir = os.environ.get("LOG_DIR")
            if not log_dir:
                raise RuntimeError(
                    "Set latentloop_same_input_stochasticity_output or LOG_DIR"
                )
            output_path = str(
                Path(log_dir) / "analysis" / "same_input_stochasticity.json"
            )
        atomic_write_json(Path(output_path), payload, refuse_overwrite=True)
        self.latentloop_same_input_stochasticity_done = True
        print(f"[LatentLoop] same-input stochasticity saved: {output_path}")
        return payload

    def _should_use_lrnode(self, timestep):
        if not (
            self.use_lrnode_latent_update
            and self.lrnode_eval_skip_full_forward
            and self.lrnode_cached_latent is not None
        ):
            return False

        if self.latentloop_hierarchical_schedule.enabled:
            return (
                self.latentloop_hierarchical_schedule.level(
                    timestep,
                    has_latent_cache=self.lrnode_cached_latent is not None,
                )
                != ExecutionLevel.FULL_SEER
            )

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
        if self.latentloop_hierarchical_schedule.enabled:
            return "hierarchical_level_2"
        if self.lrnode_eval_refresh_policy == "periodic":
            return "scheduled_periodic"
        if self.lrnode_eval_refresh_policy == "fixed_budget":
            return "scheduled_fixed_budget"
        return "forced_full"

    def _cache_full_forward_state(
        self,
        action_latent,
        selected_step,
        image_x,
        gripper,
        state,
        action_tokens=None,
        action_arm=None,
        gripper_logit=None,
        timestep=None,
    ):
        if not self.use_lrnode_latent_update:
            return
        if action_latent is None or action_latent.dim() != 4:
            raise RuntimeError(f"Expected full action latent [B, S, action_pred_steps, D], got {type(action_latent)}")
        if action_tokens is not None:
            if action_tokens.dim() != 3 or action_tokens.shape[0] != 1 or action_tokens.shape[-1] != 7:
                raise RuntimeError(f"Expected cached action tokens [1, action_pred_steps, 7], got {tuple(action_tokens.shape)}")
            self.lrnode_cached_action_tokens = action_tokens.detach()
        if action_arm is not None:
            self.lrnode_cached_action_arm = action_arm.detach()
        if gripper_logit is not None:
            self.lrnode_cached_gripper_logit = gripper_logit.detach()
        self.lrnode_cached_image_primary = image_x.detach()
        self.lrnode_cached_image_wrist = gripper.detach()
        self.lrnode_cached_state = state.detach()
        self.lrnode_cached_latent = action_latent[:, selected_step].detach()
        self.lrnode_cached_age = 0
        self.lrnode_last_full_timestep = int(timestep) if timestep is not None else None
        if self.latentloop_hierarchical_schedule.enabled:
            self.hierarchical_regeneration_age = 0
            self.hierarchical_latent_cache_version += 1
            self.hierarchical_provenance.reset_full()
            self.hierarchical_action_cache_generation = self.hierarchical_provenance.generation
            self.hierarchical_action_cache_input_generation = (
                self.hierarchical_action_cache_generation
            )
        if self.joint_latent_action_surrogate_mode == "joint":
            if action_arm is None or gripper_logit is None:
                raise RuntimeError("Joint full refresh requires exact action diagnostics")
            self.joint_anchor_arm = action_arm.detach().clone()
            self.joint_anchor_gripper_logit = gripper_logit.detach().clone()
            self.joint_anchor_latent = self.lrnode_cached_latent.detach().clone()
            self.joint_anchor_timestep = int(timestep)
            self.joint_anchor_generation = int(
                self.hierarchical_action_cache_generation
            )
        self.latentloop_feedback_buffer.reset()
        if self.latentloop_plan_adapter_mode == "anchor_bridge":
            self.latentloop_anchor_latent = self.lrnode_cached_latent
            self.latentloop_anchor_image_primary = image_x.detach()
            self.latentloop_anchor_image_wrist = gripper.detach()
            self.latentloop_anchor_state = state.detach()

    def _cache_executed_env_action(self, action):
        if self.use_lrnode_latent_update:
            self.lrnode_cached_env_action = np.asarray(action, dtype=np.float32).copy()

    def _zero_u_delta_like_latent(self, z_prev):
        base_model = self._base_model()
        motion_dim = int(getattr(base_model, "lrnode_motion_dim", 0))
        if motion_dim <= 0:
            raise RuntimeError("Cannot infer LR-NODE motion dimension for no_delta ablation")
        if z_prev.dim() >= 3 and z_prev.shape[-2] == self.action_pred_steps:
            leading_shape = tuple(z_prev.shape[:-2])
        else:
            leading_shape = tuple(z_prev.shape[:-1])
        return torch.zeros(
            leading_shape + (motion_dim,),
            device=z_prev.device,
            dtype=z_prev.dtype,
        )

    def _update_from_lrnode_cache(
        self,
        image_x,
        gripper,
        state,
        use_zero_delta=False,
        compute_hold_action=False,
        commit_cache=True,
        decode_action=True,
        require_action_diagnostics=False,
        timestep=None,
    ):
        base_model = self._base_model()
        if self.lrnode_cached_latent is None:
            raise RuntimeError("LR-NODE skip requested before a full-forward latent was cached")

        age = self.lrnode_cached_age + 1
        z_prev = self.lrnode_cached_latent
        if use_zero_delta:
            u_delta = self._zero_u_delta_like_latent(z_prev)
            fast_encoder_ms = 0.0
            fast_encoder_called = 0
        else:
            self._sync_cuda()
            t_fast = time.perf_counter()
            current_u_delta = base_model.lrnode_encode_delta(
                key_image_primary=self.lrnode_cached_image_primary[:, 0],
                key_image_wrist=self.lrnode_cached_image_wrist[:, 0],
                cur_image_primary=image_x[:, 0],
                cur_image_wrist=gripper[:, 0],
                q_key=self.lrnode_cached_state[:, 0],
                q_cur=state[:, 0],
            )
            self._sync_cuda()
            fast_encoder_ms = (time.perf_counter() - t_fast) * 1000.0
            fast_encoder_called = 1
            if timestep is None:
                raise ValueError("Observation-conditioned updates require timestep")
            if self.latentloop_feedback_source == "current":
                u_delta = current_u_delta
                feature_source_step = int(timestep)
                time_shift_initialized_with_zero = 0
            else:
                feedback = self.latentloop_feedback_buffer.select(
                    current_u_delta, int(timestep)
                )
                u_delta = feedback.feature
                feature_source_step = int(feedback.source_step)
                time_shift_initialized_with_zero = int(
                    feedback.initialized_with_zero
                )

        if use_zero_delta:
            feature_source_step = -1
            time_shift_initialized_with_zero = 0

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

        action_diagnostics = None
        action_seq = None
        if decode_action:
            self._sync_cuda()
            t_head = time.perf_counter()
            if (
                require_action_diagnostics
                or
                self.lrnode_mechanism_trace
                or self.latentloop_plan_trace
                or self.lrnode_eval_shadow_full_forward
            ):
                action_diagnostics = base_model.decode_action_diagnostics_from_latent(z_next)
                arm_action = action_diagnostics["arm"]
                gripper_action = action_diagnostics["gripper_probability"]
            else:
                arm_action, gripper_action = base_model.decode_action_from_latent(z_next)
            self._sync_cuda()
            action_head_ms = (time.perf_counter() - t_head) * 1000.0
            action_seq = torch.concat((arm_action, gripper_action), dim=-1)
        else:
            action_head_ms = 0.0
        hold_action_seq = None
        if compute_hold_action:
            if not decode_action:
                raise ValueError("compute_hold_action=True requires decode_action=True")
            with torch.no_grad():
                hold_arm_action, hold_gripper_action = base_model.decode_action_from_latent(z_prev.detach())
            hold_action_seq = torch.concat((hold_arm_action, hold_gripper_action), dim=-1)
        update = getattr(base_model.lrnode_dynamics, "last_update", None)
        gate = getattr(base_model.lrnode_dynamics, "last_gate", None)
        imgdiff_primary = (image_x[:, 0].detach().float() - self.lrnode_cached_image_primary[:, 0].detach().float()).abs()
        imgdiff_wrist = (gripper[:, 0].detach().float() - self.lrnode_cached_image_wrist[:, 0].detach().float()).abs()
        proprio_delta = (
            state[:, 0].detach().float() - self.lrnode_cached_state[:, 0].detach().float()
        )
        debug = {
            "cache_age": age,
            "skip_age": age,
            "fast_encoder_called": fast_encoder_called,
            "lrnode_update_called": 1,
            "latent_updater_called": 1,
            "action_correction_called": 0,
            "action_correction_residual_norm": 0.0,
            "observation_conditioned_update_called": int(not use_zero_delta),
            "zero_feature_update_called": int(use_zero_delta),
            "action_head_called": int(decode_action),
            "fast_encoder_ms": fast_encoder_ms,
            "node_update_ms": node_update_ms,
            "action_head_ms": action_head_ms,
            "gate_mean": float(gate.detach().float().mean().item()) if gate is not None else 0.0,
            "gate_max": float(gate.detach().float().max().item()) if gate is not None else 0.0,
            "u_delta_norm": float(u_delta.detach().float().norm(dim=-1).mean().item()),
            "feature_source_step": feature_source_step,
            "feedback_source": self.latentloop_feedback_source,
            "time_shift_initialized_with_zero": time_shift_initialized_with_zero,
            "image_diff_primary_l1": float(imgdiff_primary.mean().item()),
            "image_diff_wrist_l1": float(imgdiff_wrist.mean().item()),
            "proprio_delta_l2": float(proprio_delta.norm(dim=-1).mean().item()),
            "update_norm": float(update.detach().float().norm(dim=-1).mean().item()) if update is not None else 0.0,
            "z_norm": float(z_next.detach().float().norm(dim=-1).mean().item()),
            "z_pred": z_next.detach(),
            "z_hold": z_prev.detach(),
            "z_prev": z_prev.detach(),
            "u_delta": u_delta.detach(),
            "lrnode_update": update.detach() if update is not None else None,
            "lrnode_gate": gate.detach() if gate is not None else None,
            "observation_cache_advanced": int(commit_cache),
        }
        if action_seq is not None:
            debug["action_pred"] = action_seq.detach()
        if action_diagnostics is not None:
            debug["action_pred_arm"] = action_diagnostics["arm"].detach()
            debug["action_pred_gripper_logit"] = action_diagnostics["gripper_logit"].detach()
            debug["action_pred_gripper_probability"] = action_diagnostics[
                "gripper_probability"
            ].detach()
        if hold_action_seq is not None:
            debug["action_hold"] = hold_action_seq.detach()
        if commit_cache:
            self.lrnode_cached_latent = z_next.detach()
            self.lrnode_cached_image_primary = image_x.detach()
            self.lrnode_cached_image_wrist = gripper.detach()
            self.lrnode_cached_state = state.detach()
            self.lrnode_cached_age = age
            if self.latentloop_hierarchical_schedule.enabled:
                self.hierarchical_latent_cache_version += 1
        return action_seq, debug

    def _update_action_correction_cache(self, image_x, gripper, state, timestep):
        """Run the matched full-horizon action-space correction baseline."""

        if self.lrnode_cached_action_arm is None or self.lrnode_cached_gripper_logit is None:
            raise RuntimeError("Action correction requires a cached full-Seer raw horizon")
        adapter = self._base_model().latentloop_plan_adapter
        age = self.lrnode_cached_age + 1
        self._sync_cuda()
        encoder_t0 = time.perf_counter()
        feature = adapter.encode_delta(
            self.lrnode_cached_image_primary[:, 0],
            self.lrnode_cached_image_wrist[:, 0],
            image_x[:, 0],
            gripper[:, 0],
            self.lrnode_cached_state[:, 0],
            state[:, 0],
        )
        self._sync_cuda()
        encoder_ms = (time.perf_counter() - encoder_t0) * 1000.0
        predictor_t0 = time.perf_counter()
        output = adapter.forward_from_feature(
            self.lrnode_cached_action_arm,
            self.lrnode_cached_gripper_logit,
            feature,
            age=float(age),
        )
        self._sync_cuda()
        predictor_ms = (time.perf_counter() - predictor_t0) * 1000.0
        action_seq = torch.cat([output.arm, output.gripper_probability], dim=-1)
        proprio_delta = state[:, 0].float() - self.lrnode_cached_state[:, 0].float()
        primary_change = (
            image_x[:, 0].float() - self.lrnode_cached_image_primary[:, 0].float()
        ).abs().mean()
        wrist_change = (
            gripper[:, 0].float() - self.lrnode_cached_image_wrist[:, 0].float()
        ).abs().mean()
        self.lrnode_cached_action_arm = output.arm.detach()
        self.lrnode_cached_gripper_logit = output.gripper_logit.detach()
        self.lrnode_cached_action_tokens = action_seq.detach()
        self.lrnode_cached_image_primary = image_x.detach()
        self.lrnode_cached_image_wrist = gripper.detach()
        self.lrnode_cached_state = state.detach()
        self.lrnode_cached_age = age
        return action_seq, {
            "cache_age": age,
            "skip_age": age,
            "feature_source_step": int(timestep),
            "feedback_source": "current",
            "fast_encoder_called": 1,
            "lrnode_update_called": 1,
            "latent_updater_called": 0,
            "action_correction_called": 1,
            "action_correction_residual_norm": float(
                torch.cat(
                    [output.arm_residual, output.gripper_logit_residual], dim=-1
                ).float().norm(dim=-1).mean().item()
            ),
            "action_head_called": 0,
            "observation_conditioned_update_called": 1,
            "zero_feature_update_called": 0,
            "observation_cache_advanced": 1,
            "fast_encoder_ms": encoder_ms,
            "node_update_ms": predictor_ms,
            "action_head_ms": 0.0,
            "u_delta_norm": float(feature.float().norm(dim=-1).mean().item()),
            "image_diff_primary_l1": float(primary_change.item()),
            "image_diff_wrist_l1": float(wrist_change.item()),
            "proprio_delta_l2": float(proprio_delta.norm(dim=-1).mean().item()),
            "update_norm": float(output.arm_residual.float().norm(dim=-1).mean().item()),
            "z_norm": 0.0,
            "action_pred": action_seq.detach(),
            "action_pred_gripper_logit": output.gripper_logit.detach(),
            "action_pred_gripper_probability": output.gripper_probability.detach(),
            "u_delta": feature.detach(),
        }

    def _update_hybrid_level0(self, image_x, gripper, state, timestep):
        """Advance both caches without invoking the Seer action head."""

        if self.lrnode_cached_action_arm is None or self.lrnode_cached_gripper_logit is None:
            raise RuntimeError("Hybrid Level 0 requires a complete cached action horizon")
        previous_primary = self.lrnode_cached_image_primary
        previous_wrist = self.lrnode_cached_image_wrist
        previous_state = self.lrnode_cached_state
        previous_arm = self.lrnode_cached_action_arm
        previous_gripper_logit = self.lrnode_cached_gripper_logit
        input_generation = self.hierarchical_action_cache_generation

        _, latent_debug = self._update_from_lrnode_cache(
            image_x,
            gripper,
            state,
            use_zero_delta=False,
            compute_hold_action=False,
            commit_cache=True,
            decode_action=False,
            timestep=timestep,
        )

        adapter = self._base_model().latentloop_plan_adapter
        correction_age = self.hierarchical_regeneration_age + 1
        self._sync_cuda()
        encoder_t0 = time.perf_counter()
        action_feature = adapter.encode_delta(
            previous_primary[:, 0],
            previous_wrist[:, 0],
            image_x[:, 0],
            gripper[:, 0],
            previous_state[:, 0],
            state[:, 0],
        )
        self._sync_cuda()
        correction_encoder_ms = (time.perf_counter() - encoder_t0) * 1000.0
        correction_t0 = time.perf_counter()
        output = adapter.forward_from_feature(
            previous_arm,
            previous_gripper_logit,
            action_feature,
            age=float(correction_age),
        )
        self._sync_cuda()
        correction_ms = (time.perf_counter() - correction_t0) * 1000.0
        action_seq = torch.cat([output.arm, output.gripper_probability], dim=-1)
        self.lrnode_cached_action_arm = output.arm.detach()
        self.lrnode_cached_gripper_logit = output.gripper_logit.detach()
        self.lrnode_cached_action_tokens = action_seq.detach()
        self.hierarchical_regeneration_age = correction_age
        self.hierarchical_action_cache_input_generation = input_generation

        residual = torch.cat(
            [output.arm_residual, output.gripper_logit_residual], dim=-1
        )
        latent_debug.update(
            {
                "fast_encoder_called": int(latent_debug.get("fast_encoder_called", 0)) + 1,
                "fast_encoder_ms": float(latent_debug.get("fast_encoder_ms", 0.0))
                + correction_encoder_ms,
                "action_correction_called": 1,
                "action_correction_ms": correction_ms,
                "action_correction_residual_norm": float(
                    residual.float().norm(dim=-1).mean().item()
                ),
                "action_pred": action_seq.detach(),
                "action_pred_arm": output.arm.detach(),
                "action_pred_gripper_logit": output.gripper_logit.detach(),
                "action_pred_gripper_probability": output.gripper_probability.detach(),
                "action_correction_u_delta": action_feature.detach(),
                "action_cache_input_generation": int(input_generation),
            }
        )
        return action_seq, latent_debug

    def _regenerate_horizon_from_latent(self, image_x, gripper, state, timestep):
        require_diagnostics = (
            self.latentloop_hierarchical_mode == "hybrid"
            or self.joint_latent_action_surrogate_mode == "joint"
        )
        action_seq, debug = self._update_from_lrnode_cache(
            image_x,
            gripper,
            state,
            use_zero_delta=False,
            compute_hold_action=False,
            commit_cache=True,
            decode_action=True,
            require_action_diagnostics=require_diagnostics,
            timestep=timestep,
        )
        self.lrnode_cached_action_tokens = action_seq.detach()
        if require_diagnostics:
            self.lrnode_cached_action_arm = debug["action_pred_arm"].detach()
            self.lrnode_cached_gripper_logit = debug[
                "action_pred_gripper_logit"
            ].detach()
        self.hierarchical_regeneration_age = 0
        self.hierarchical_provenance.reset_regenerated()
        self.hierarchical_action_cache_generation = self.hierarchical_provenance.generation
        self.hierarchical_action_cache_input_generation = (
            self.hierarchical_action_cache_generation
        )
        debug.update(
            {
                "action_cache_input_generation": int(
                    self.hierarchical_action_cache_input_generation
                ),
                "action_correction_called": 0,
                "action_correction_residual_norm": 0.0,
            }
        )
        return action_seq, debug

    def _joint_exact_regeneration(self, image_x, gripper, state, timestep):
        action_seq, debug = self._regenerate_horizon_from_latent(
            image_x, gripper, state, timestep
        )
        if "action_pred_arm" not in debug or "action_pred_gripper_logit" not in debug:
            raise RuntimeError("Exact joint regeneration did not return diagnostics")
        self.joint_anchor_arm = debug["action_pred_arm"].detach().clone()
        self.joint_anchor_gripper_logit = debug[
            "action_pred_gripper_logit"
        ].detach().clone()
        self.joint_anchor_latent = self.lrnode_cached_latent.detach().clone()
        self.joint_anchor_timestep = int(timestep)
        self.joint_anchor_generation = int(self.hierarchical_action_cache_generation)
        debug.update(
            {
                "joint_anchor_replaced_by_exact": 1,
                "joint_surrogate_called": 0,
                "joint_surrogate_ms": 0.0,
                "joint_action_surrogate_error": 0.0,
                "joint_executed_token_error": 0.0,
                "joint_tail_token_error": 0.0,
            }
        )
        return action_seq, debug

    def _joint_fast_surrogate(self, image_x, gripper, state, timestep):
        if any(
            value is None
            for value in (
                self.joint_anchor_arm,
                self.joint_anchor_gripper_logit,
                self.joint_anchor_latent,
                self.joint_anchor_timestep,
            )
        ):
            raise RuntimeError("Joint fast step requested before exact anchor initialization")
        input_generation = int(self.joint_anchor_generation)
        _, debug = self._update_from_lrnode_cache(
            image_x,
            gripper,
            state,
            use_zero_delta=False,
            compute_hold_action=False,
            commit_cache=True,
            decode_action=False,
            timestep=timestep,
        )
        elapsed = int(timestep) - int(self.joint_anchor_timestep)
        if elapsed not in {1, 2}:
            raise RuntimeError(
                f"Primary K_G=3 joint schedule expected anchor age 1 or 2, got {elapsed}"
            )
        adapter = self._base_model().joint_latent_action_surrogate
        self._sync_cuda()
        surrogate_t0 = time.perf_counter()
        output = adapter.surrogate_forward(
            anchor_arm=self.joint_anchor_arm,
            anchor_gripper_logit=self.joint_anchor_gripper_logit,
            anchor_latent=self.joint_anchor_latent,
            current_latent=self.lrnode_cached_latent,
            shared_feature=debug["u_delta"],
            elapsed=elapsed,
        )
        self._sync_cuda()
        surrogate_ms = (time.perf_counter() - surrogate_t0) * 1000.0
        action_seq = output.horizon
        self.lrnode_cached_action_tokens = action_seq.detach()
        self.hierarchical_regeneration_age = elapsed
        self.hierarchical_action_cache_input_generation = input_generation
        action_error = 0.0
        executed_token_error = 0.0
        tail_token_error = 0.0
        diagnostic_ms = 0.0
        if self.joint_error_trace:
            self._sync_cuda()
            diagnostic_t0 = time.perf_counter()
            with torch.no_grad():
                exact = self._base_model().decode_action_diagnostics_from_latent(
                    self.lrnode_cached_latent
                )
            self._sync_cuda()
            diagnostic_ms = (time.perf_counter() - diagnostic_t0) * 1000.0
            exact_horizon = torch.cat(
                [exact["arm"], exact["gripper_probability"]], dim=-1
            )
            action_error = float(
                F.l1_loss(action_seq.float(), exact_horizon.float()).item()
            )
            absolute_error = (action_seq.float() - exact_horizon.float()).abs()
            executed_token_error = float(absolute_error[..., 0, :].mean().item())
            invalid_tail = (~output.valid_mask.bool()).expand_as(absolute_error)
            tail_token_error = float(
                absolute_error[invalid_tail].mean().item()
                if bool(invalid_tail.any())
                else 0.0
            )
            self.joint_diagnostic_action_head_calls += 1
        debug.update(
            {
                "action_pred": action_seq.detach(),
                "action_pred_arm": output.arm.detach(),
                "action_pred_gripper_logit": output.gripper_logit.detach(),
                "action_pred_gripper_probability": output.gripper_probability.detach(),
                "action_correction_called": 1,
                "action_correction_ms": surrogate_ms,
                "action_correction_residual_norm": float(
                    output.residual.detach().float().norm(dim=-1).mean().item()
                ),
                "joint_surrogate_called": 1,
                "joint_surrogate_ms": surrogate_ms,
                "joint_anchor_elapsed": elapsed,
                "joint_anchor_generation": input_generation,
                "joint_anchor_replaced_by_exact": 0,
                "joint_valid_token_fraction": float(
                    output.valid_mask.float().mean().item()
                ),
                "joint_action_surrogate_error": action_error,
                "joint_executed_token_error": executed_token_error,
                "joint_tail_token_error": tail_token_error,
                "joint_diagnostic_action_head_ms": diagnostic_ms,
                "action_cache_input_generation": input_generation,
            }
        )
        self.joint_surrogate_calls += 1
        self.joint_surrogate_latency_sum += surrogate_ms
        return action_seq, debug

    def _wide_exact_update(self, image_x, gripper, state, timestep):
        _, debug = self._update_from_lrnode_cache(
            image_x,
            gripper,
            state,
            use_zero_delta=False,
            compute_hold_action=False,
            commit_cache=True,
            decode_action=False,
            timestep=timestep,
        )
        adapter = self._base_model().joint_latent_action_surrogate
        z_wide = adapter.apply_wide_capacity(
            self.lrnode_cached_latent,
            debug["u_delta"],
            age=float(self.lrnode_cached_age),
        )
        self.lrnode_cached_latent = z_wide.detach()
        self._sync_cuda()
        head_t0 = time.perf_counter()
        diagnostics = self._base_model().decode_action_diagnostics_from_latent(z_wide)
        self._sync_cuda()
        head_ms = (time.perf_counter() - head_t0) * 1000.0
        action_seq = torch.cat(
            [diagnostics["arm"], diagnostics["gripper_probability"]], dim=-1
        )
        self.lrnode_cached_action_tokens = action_seq.detach()
        self.hierarchical_regeneration_age = 0
        self.hierarchical_provenance.reset_regenerated()
        self.hierarchical_action_cache_generation = self.hierarchical_provenance.generation
        self.hierarchical_action_cache_input_generation = (
            self.hierarchical_action_cache_generation
        )
        debug.update(
            {
                "action_pred": action_seq.detach(),
                "action_pred_arm": diagnostics["arm"].detach(),
                "action_pred_gripper_logit": diagnostics["gripper_logit"].detach(),
                "action_pred_gripper_probability": diagnostics[
                    "gripper_probability"
                ].detach(),
                "action_head_called": 1,
                "action_head_ms": head_ms,
                "joint_surrogate_called": 0,
                "joint_surrogate_ms": 0.0,
                "action_cache_input_generation": int(
                    self.hierarchical_action_cache_input_generation
                ),
            }
        )
        return action_seq, debug

    def _update_anchor_bridge_cache(self, image_x, gripper, state, timestep):
        """Predict from the fixed segment anchor without recursive latent input."""

        if self.latentloop_anchor_latent is None:
            raise RuntimeError("Anchor bridge requires a full-refresh segment anchor")
        adapter = self._base_model().latentloop_plan_adapter
        age = self.lrnode_cached_age + 1
        self._sync_cuda()
        encoder_t0 = time.perf_counter()
        feature = adapter.encode_anchor_to_current(
            self.latentloop_anchor_image_primary[:, 0],
            self.latentloop_anchor_image_wrist[:, 0],
            image_x[:, 0],
            gripper[:, 0],
            self.latentloop_anchor_state[:, 0],
            state[:, 0],
        )
        self._sync_cuda()
        encoder_ms = (time.perf_counter() - encoder_t0) * 1000.0
        predictor_t0 = time.perf_counter()
        output = adapter.forward_from_feature(
            self.latentloop_anchor_latent, feature, age=float(age)
        )
        self._sync_cuda()
        predictor_ms = (time.perf_counter() - predictor_t0) * 1000.0
        head_t0 = time.perf_counter()
        diagnostics = self._base_model().decode_action_diagnostics_from_latent(
            output.latent
        )
        self._sync_cuda()
        head_ms = (time.perf_counter() - head_t0) * 1000.0
        action_seq = torch.cat(
            [diagnostics["arm"], diagnostics["gripper_probability"]], dim=-1
        )
        proprio_delta = (
            state[:, 0].float() - self.latentloop_anchor_state[:, 0].float()
        )
        primary_change = (
            image_x[:, 0].float()
            - self.latentloop_anchor_image_primary[:, 0].float()
        ).abs().mean()
        wrist_change = (
            gripper[:, 0].float()
            - self.latentloop_anchor_image_wrist[:, 0].float()
        ).abs().mean()
        self.lrnode_cached_latent = output.latent.detach()
        self.lrnode_cached_image_primary = image_x.detach()
        self.lrnode_cached_image_wrist = gripper.detach()
        self.lrnode_cached_state = state.detach()
        self.lrnode_cached_age = age
        return action_seq, {
            "cache_age": age,
            "skip_age": age,
            "feature_source_step": int(timestep),
            "feedback_source": "current",
            "fast_encoder_called": 1,
            "lrnode_update_called": 1,
            "action_head_called": 1,
            "observation_conditioned_update_called": 1,
            "zero_feature_update_called": 0,
            "observation_cache_advanced": 1,
            "fast_encoder_ms": encoder_ms,
            "node_update_ms": predictor_ms,
            "action_head_ms": head_ms,
            "u_delta_norm": float(feature.float().norm(dim=-1).mean().item()),
            "image_diff_primary_l1": float(primary_change.item()),
            "image_diff_wrist_l1": float(wrist_change.item()),
            "proprio_delta_l2": float(proprio_delta.norm(dim=-1).mean().item()),
            "update_norm": float(output.residual.float().norm(dim=-1).mean().item()),
            "z_norm": float(output.latent.float().norm(dim=-1).mean().item()),
            "z_pred": output.latent.detach(),
            "lrnode_update": output.residual.detach(),
            "lrnode_gate": output.gate.detach(),
            "action_pred": action_seq.detach(),
            "action_pred_gripper_logit": diagnostics["gripper_logit"].detach(),
            "action_pred_gripper_probability": diagnostics[
                "gripper_probability"
            ].detach(),
            "u_delta": feature.detach(),
        }

    def _decode_from_cached_latent(self):
        base_model = self._base_model()
        if self.lrnode_cached_latent is None:
            raise RuntimeError("hold_latent ablation requested before a full-forward latent was cached")

        age = self.lrnode_cached_age + 1
        z_prev = self.lrnode_cached_latent
        self._sync_cuda()
        t_head = time.perf_counter()
        diagnostics = base_model.decode_action_diagnostics_from_latent(z_prev)
        arm_action = diagnostics["arm"]
        gripper_action = diagnostics["gripper_probability"]
        self._sync_cuda()
        action_head_ms = (time.perf_counter() - t_head) * 1000.0
        action_seq = torch.concat((arm_action, gripper_action), dim=-1)
        self.lrnode_cached_age = age
        return action_seq, {
            "cache_age": age,
            "skip_age": age,
            "action_head_called": 1,
            "lrnode_update_called": 0,
            "fast_encoder_called": 0,
            "fast_encoder_ms": 0.0,
            "node_update_ms": 0.0,
            "action_head_ms": action_head_ms,
            "gate_mean": 0.0,
            "gate_max": 0.0,
            "u_delta_norm": 0.0,
            "feedback_source": self.latentloop_feedback_source,
            "time_shift_initialized_with_zero": 0,
            "image_diff_primary_l1": 0.0,
            "image_diff_wrist_l1": 0.0,
            "update_norm": 0.0,
            "z_norm": float(z_prev.detach().float().norm(dim=-1).mean().item()),
            "feature_source_step": -1,
            "action_pred": action_seq.detach(),
            "action_pred_gripper_logit": diagnostics["gripper_logit"].detach(),
            "action_pred_gripper_probability": gripper_action.detach(),
        }

    def _run_shadow_full_forward(
        self,
        input_image_primary,
        input_image_wrist,
        input_state,
        input_text_token,
        selected_step,
    ):
        base_model = self._base_model()
        previous_profile = bool(getattr(base_model, "profile_full_action_head", False))
        previous_head_ms = float(getattr(base_model, "last_full_action_head_ms", 0.0))
        self._sync_cuda()
        shadow_t0 = time.perf_counter()
        try:
            with preserve_rng_state(include_cuda=True):
                base_model.profile_full_action_head = False
                shadow_outputs = self.model(
                    image_primary=input_image_primary,
                    image_wrist=input_image_wrist,
                    state=input_state,
                    text_token=input_text_token,
                    action=torch.zeros(1, self.history_len, 7).to(input_state.device),
                    return_action_latent=True,
                )
                shadow_latent = shadow_outputs["action_latent"][:, selected_step].detach()
                shadow_diagnostics = base_model.decode_action_diagnostics_from_latent(
                    shadow_latent
                )
        finally:
            base_model.profile_full_action_head = previous_profile
            base_model.last_full_action_head_ms = previous_head_ms
        self._sync_cuda()
        shadow_ms = (time.perf_counter() - shadow_t0) * 1000.0
        shadow_action = torch.cat(
            [
                shadow_diagnostics["arm"],
                shadow_diagnostics["gripper_probability"],
            ],
            dim=-1,
        ).detach()
        return {
            "latent": shadow_latent,
            "action": shadow_action,
            "arm": shadow_diagnostics["arm"].detach(),
            "gripper_logit": shadow_diagnostics["gripper_logit"].detach(),
            "gripper_probability": shadow_diagnostics["gripper_probability"].detach(),
            "latency_ms": shadow_ms,
        }

    def _decode_diagnostic_latent(self, latent):
        diagnostics = self._base_model().decode_action_diagnostics_from_latent(latent)
        return (
            torch.cat(
                [diagnostics["arm"], diagnostics["gripper_probability"]],
                dim=-1,
            ),
            diagnostics,
        )

    def _apply_skip_counterfactual(
        self,
        timestep,
        lr_action_seq,
        lrnode_debug,
        shadow,
    ):
        mode = self.lrnode_counterfactual_mode
        arm_source = "lr"
        gripper_source = "lr"
        executed_latent = lrnode_debug.get("z_pred")
        extra = {}

        if mode in {
            "full_arm_full_gripper",
            "lr_arm_lr_gripper",
            "lr_arm_full_gripper",
            "full_arm_lr_gripper",
        }:
            action_seq, arm_source, gripper_source = mix_action_tokens(
                lr_action_seq,
                shadow["action"],
                mode,
            )
        elif mode == "latent_fusion" and self.lrnode_latent_fusion_mode == "every_step":
            executed_latent = fuse_latents(
                lrnode_debug["z_pred"],
                shadow["latent"],
                self.lrnode_latent_fusion_alpha,
            )
            action_seq, diagnostics = self._decode_diagnostic_latent(executed_latent)
            self.lrnode_cached_latent = executed_latent.detach()
            self.counterfactual_latent_fusion_steps += 1
            extra = {
                "counterfactual_gripper_logit": diagnostics["gripper_logit"].detach(),
                "counterfactual_gripper_probability": diagnostics[
                    "gripper_probability"
                ].detach(),
            }
            arm_source = "fusion"
            gripper_source = "fusion"
        elif mode == "matched_random":
            seed = deterministic_step_seed(
                self.lrnode_matched_random_seed,
                self.current_task_id,
                self.current_episode_id,
                timestep,
            )
            executed_latent, learned_delta, random_delta = matched_random_latent(
                lrnode_debug["z_pred"],
                shadow["latent"],
                seed=seed,
                norm_mode=self.lrnode_matched_random_norm_mode,
            )
            action_seq, diagnostics = self._decode_diagnostic_latent(executed_latent)
            self.lrnode_cached_latent = executed_latent.detach()
            self.counterfactual_matched_random_steps += 1
            extra = {
                "matched_random_seed": seed,
                "learned_delta": learned_delta.detach(),
                "random_delta": random_delta.detach(),
                "learned_delta_norm": float(learned_delta.detach().float().norm().item()),
                "random_delta_norm": float(random_delta.detach().float().norm().item()),
                "counterfactual_gripper_logit": diagnostics["gripper_logit"].detach(),
                "counterfactual_gripper_probability": diagnostics[
                    "gripper_probability"
                ].detach(),
            }
            arm_source = "random"
            gripper_source = "random"
        else:
            action_seq = lr_action_seq

        if arm_source == "lr":
            self.counterfactual_arm_lr_steps += 1
        elif arm_source == "full":
            self.counterfactual_arm_full_steps += 1
        if gripper_source == "lr":
            self.counterfactual_gripper_lr_steps += 1
        elif gripper_source == "full":
            self.counterfactual_gripper_full_steps += 1
        return action_seq, executed_latent, arm_source, gripper_source, extra

    def _trace_enabled_for_current_episode(self):
        return self.lrnode_mechanism_trace and (
            self.lrnode_trace_episode_limit == 0
            or self.trace_episode_count < self.lrnode_trace_episode_limit
        )

    def _trace_output_path(self):
        if self.lrnode_trace_output_dir:
            return Path(self.lrnode_trace_output_dir)
        log_dir = os.environ.get("LOG_DIR")
        if not log_dir:
            raise RuntimeError(
                "lrnode_mechanism_trace requires lrnode_trace_output_dir or LOG_DIR"
            )
        return Path(log_dir) / "analysis" / "mechanism_trace"

    def _hierarchical_trace_output_path(self):
        if self.latentloop_hierarchical_trace_output_dir:
            return Path(self.latentloop_hierarchical_trace_output_dir)
        log_dir = os.environ.get("LOG_DIR")
        if not log_dir:
            raise RuntimeError(
                "hierarchical trace requires latentloop_hierarchical_trace_output_dir "
                "or LOG_DIR"
            )
        return Path(log_dir) / "analysis" / "hierarchical_trace"

    def _hierarchical_executed_provenance(self, action_seq, timestep, candidate_count):
        horizon = self.hierarchical_provenance.snapshot()
        if action_seq.shape != (1, self.action_pred_steps, 7):
            raise RuntimeError(
                "Hierarchical provenance expects action horizon [1,P,7], got "
                f"{tuple(action_seq.shape)}"
            )
        if self.use_ensembling:
            if self.hierarchical_provenance_buffer is None:
                raise RuntimeError("Missing hierarchical provenance ensemble buffer")
            presence = torch.all(action_seq[0].detach().cpu() != 0, dim=-1).numpy()
            start = int(timestep)
            stop = start + self.action_pred_steps
            self.hierarchical_provenance_buffer[start, start:stop] = horizon
            self.hierarchical_provenance_presence[start, start:stop] = presence
            mask = self.hierarchical_provenance_presence[:, start]
            candidates = self.hierarchical_provenance_buffer[:, start][mask]
            if int(candidates.shape[0]) != int(candidate_count):
                raise RuntimeError(
                    "Provenance sidecar diverged from canonical temporal ensemble: "
                    f"provenance={candidates.shape[0]}, action={candidate_count}"
                )
            weights = np.exp(-float(self.ensembling_temp) * np.arange(len(candidates)))
            weights /= weights.sum()
            executed = (candidates * weights[:, None]).sum(axis=0)
        else:
            executed = horizon[0]
            if int(candidate_count) != 1:
                raise RuntimeError("Non-ensemble action must have one provenance candidate")
        executed /= executed.sum()
        return horizon, executed

    def _assert_hierarchical_step(
        self,
        level,
        step_record,
        *,
        latent_version_before,
        action_generation_before,
    ):
        if not self.latentloop_hierarchical_assert_invariants:
            return
        counts = LevelCallCounts(
            full_seer=int(step_record["hierarchical_step_full_seer_calls"]),
            action_head=int(step_record["hierarchical_step_action_head_calls"]),
            latent_updater=int(step_record["hierarchical_step_latent_updater_calls"]),
            action_correction=int(step_record["hierarchical_step_action_correction_calls"]),
        )
        assert_level_call_contract(
            int(level), counts, mode=self.latentloop_hierarchical_mode
        )
        if level != ExecutionLevel.FULL_SEER and (
            self.latentloop_hierarchical_mode != "pure_action_correction"
            and self.hierarchical_latent_cache_version != latent_version_before + 1
        ):
            raise RuntimeError("Latent cache did not advance exactly once on a skip step")
        if level == ExecutionLevel.HORIZON_REGENERATION:
            provenance = self.hierarchical_provenance.snapshot()
            regenerated = provenance[:, PROVENANCE_LABELS.index("regenerated")]
            if not np.all(regenerated == 1.0):
                raise RuntimeError("Level 1 did not replace all token provenance")
            if self.hierarchical_regeneration_age != 0:
                raise RuntimeError("Level 1 did not reset regeneration age")
            if self.hierarchical_action_cache_generation <= action_generation_before:
                raise RuntimeError("Level 1 did not create a new action-cache generation")
        if level == ExecutionLevel.ACTION_CORRECTION:
            if self.hierarchical_action_cache_input_generation != action_generation_before:
                raise RuntimeError(
                    "Action correction did not consume the most recent complete horizon"
                )
            if self.hierarchical_action_cache_generation != action_generation_before:
                raise RuntimeError("Level 0 must not create a complete-horizon generation")

    def _plan_trace_output_path(self):
        if self.latentloop_plan_trace_output_dir:
            return Path(self.latentloop_plan_trace_output_dir)
        log_dir = os.environ.get("LOG_DIR")
        if not log_dir:
            raise RuntimeError(
                "latentloop_plan_trace requires latentloop_plan_trace_output_dir or LOG_DIR"
            )
        return Path(log_dir) / "analysis" / "plan_trace"

    def _cached_chunk_token_action(self, timestep):
        if self.lrnode_cached_action_tokens is None:
            raise RuntimeError("seer_token_chunk ablation requested before full action tokens were cached")
        if self.lrnode_last_full_timestep is None:
            raise RuntimeError("seer_token_chunk ablation requested before last full timestep was cached")

        skip_age = int(timestep) - int(self.lrnode_last_full_timestep)
        token_idx = skip_age - 1
        num_tokens = int(self.lrnode_cached_action_tokens.shape[1])
        if token_idx < 0:
            raise RuntimeError(
                f"Invalid seer_token_chunk token_idx={token_idx}; "
                f"timestep={timestep}, last_full_timestep={self.lrnode_last_full_timestep}"
            )
        if token_idx >= num_tokens:
            raise RuntimeError(
                "seer_token_chunk skip-only policy requires token_idx < action_pred_steps; "
                f"got token_idx={token_idx}, action_pred_steps={num_tokens}. "
                "For this MVP use K <= action_pred_steps + 1."
            )
        action_token = self.lrnode_cached_action_tokens[:, token_idx:token_idx + 1]
        replay_horizon = self.lrnode_cached_action_tokens
        replay_gripper_logit = self.lrnode_cached_gripper_logit
        for _ in range(skip_age):
            replay_horizon = shift_action_horizon(replay_horizon)
            if replay_gripper_logit is not None:
                replay_gripper_logit = shift_action_horizon(replay_gripper_logit)
        self.lrnode_cached_age = skip_age
        action = self._raw_action_token_to_env_action(action_token)
        return action, {
            "cache_age": skip_age,
            "skip_age": skip_age,
            "token_idx_used": token_idx,
            "feature_source_step": -1,
            "action_head_called": 0,
            "lrnode_update_called": 0,
            "fast_encoder_called": 0,
            "fast_encoder_ms": 0.0,
            "node_update_ms": 0.0,
            "action_head_ms": 0.0,
            "gate_mean": 0.0,
            "gate_max": 0.0,
            "u_delta_norm": 0.0,
            "image_diff_primary_l1": 0.0,
            "image_diff_wrist_l1": 0.0,
            "update_norm": 0.0,
            "z_norm": 0.0,
            "action_pred": replay_horizon.detach(),
            "action_pred_gripper_logit": (
                None if replay_gripper_logit is None else replay_gripper_logit.detach()
            ),
            "action_pred_gripper_probability": replay_horizon[..., 6:].detach(),
        }

    def get_lrnode_stats(self):
        total_calls = self.full_forward_calls + self.lrnode_update_calls
        avg_full_latency = self.full_forward_latency_sum / self.full_forward_calls if self.full_forward_calls else 0.0
        avg_full_action_head_latency = (
            self.full_action_head_latency_sum / self.full_forward_calls if self.full_forward_calls else 0.0
        )
        avg_full_non_action_head_latency = (
            self.full_non_action_head_latency_sum / self.full_forward_calls if self.full_forward_calls else 0.0
        )
        avg_lrnode_latency = self.lrnode_latency_sum / self.lrnode_update_calls if self.lrnode_update_calls else 0.0
        query_reduction = (
            0.0
            if self.lrnode_every_step_filter is not None
            else self.lrnode_update_calls / total_calls if total_calls else 0.0
        )
        full_query_reduction_ratio = 1.0 - (self.full_forward_calls / self.num_policy_steps) if self.num_policy_steps else 0.0
        effective_query_interval = self.num_policy_steps / self.full_forward_calls if self.full_forward_calls else 0.0
        return {
            "num_env_steps": self.num_policy_steps,
            "full_forward_calls": self.full_forward_calls,
            "lrnode_update_calls": self.lrnode_update_calls,
            "fast_encoder_calls": self.fast_encoder_calls,
            "action_head_calls": self.action_head_calls,
            "hold_action_steps": self.hold_action_steps,
            "hold_latent_steps": self.hold_latent_steps,
            "chunk_token_steps": self.chunk_token_steps,
            "no_delta_steps": self.no_delta_steps,
            "latentloop_segment_grid_enabled": int(
                self.latentloop_segment_executor is not None
            ),
            "segment_length": int(self.lrnode_query_interval),
            "feedback_schedule": (
                self.latentloop_feedback_schedule
                if self.latentloop_segment_executor is not None
                else "not_applicable"
            ),
            "planned_feedback_density": (
                None
                if self.latentloop_segment_executor is None
                else self.latentloop_segment_executor.plan.density
            ),
            "observation_conditioned_update_calls": (
                self.observation_conditioned_update_calls
            ),
            "zero_feature_update_calls": self.zero_feature_update_calls,
            "observation_cache_advance_calls": self.observation_cache_advance_calls,
            "num_fallback_full_calls": 0,
            "refresh_policy": self.lrnode_eval_refresh_policy,
            "max_full_forwards_per_episode": int(self.lrnode_eval_max_full_forwards_per_episode),
            "eval_ablation_mode": self.lrnode_eval_ablation_mode,
            "no_delta_mode": self.lrnode_no_delta_mode,
            "chunk_token_policy": self.lrnode_chunk_token_policy,
            "avg_full_forward_latency_sec": avg_full_latency,
            "avg_full_action_head_latency_sec": avg_full_action_head_latency,
            "avg_full_non_action_head_latency_sec": avg_full_non_action_head_latency,
            "avg_lrnode_latency_sec": avg_lrnode_latency,
            "avg_fast_encoder_latency_sec": (
                self.fast_encoder_latency_sum / self.fast_encoder_calls / 1000.0
                if self.fast_encoder_calls else 0.0
            ),
            "avg_node_update_latency_sec": (
                self.node_update_latency_sum / self.lrnode_update_calls / 1000.0
                if self.lrnode_update_calls else 0.0
            ),
            "avg_action_head_latency_sec": (
                self.action_head_latency_sum / self.action_head_calls / 1000.0
                if self.action_head_calls else 0.0
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
            "mechanism_trace_enabled": int(self.lrnode_mechanism_trace),
            "counterfactual_mode": self.lrnode_counterfactual_mode,
            "counterfactual_mix_stage": self.lrnode_counterfactual_mix_stage,
            "counterfactual_arm_lr_steps": self.counterfactual_arm_lr_steps,
            "counterfactual_arm_full_steps": self.counterfactual_arm_full_steps,
            "counterfactual_gripper_lr_steps": self.counterfactual_gripper_lr_steps,
            "counterfactual_gripper_full_steps": self.counterfactual_gripper_full_steps,
            "counterfactual_latent_fusion_steps": self.counterfactual_latent_fusion_steps,
            "counterfactual_matched_random_steps": self.counterfactual_matched_random_steps,
            "every_step_filter_mode": self.lrnode_every_step_filter_mode,
            "every_step_filter_alpha": self.lrnode_every_step_filter_alpha,
            "every_step_filter_beta": self.lrnode_every_step_filter_beta,
            "every_step_filter_diagnostics": int(
                self.lrnode_every_step_filter_diagnostics
            ),
            "every_step_filter_prior_calls": self.every_step_filter_prior_calls,
            "every_step_filter_fusion_calls": self.every_step_filter_fusion_calls,
            "every_step_filter_action_head_calls": (
                self.every_step_filter_action_head_calls
            ),
            "every_step_filter_diagnostic_action_head_calls": (
                self.every_step_filter_diagnostic_action_head_calls
            ),
            "avg_every_step_filter_prior_latency_sec": (
                self.every_step_filter_prior_latency_sum
                / self.every_step_filter_prior_calls
                if self.every_step_filter_prior_calls else 0.0
            ),
            "avg_every_step_filter_fusion_latency_sec": (
                self.every_step_filter_fusion_latency_sum
                / max(1, self.num_policy_steps)
            ),
            "avg_every_step_filter_action_head_latency_sec": (
                self.every_step_filter_action_head_latency_sum
                / self.every_step_filter_action_head_calls
                if self.every_step_filter_action_head_calls else 0.0
            ),
            "avg_every_step_filter_diagnostic_latency_sec": (
                self.every_step_filter_diagnostic_latency_sum
                / max(1, self.num_policy_steps)
            ),
            "every_step_filter_rng_checks": self.every_step_filter_rng_checks,
            "every_step_filter_rng_failures": self.every_step_filter_rng_failures,
            "query_reduction_claim_allowed": int(
                self.lrnode_every_step_filter is None
            ),
            "full_forward_calls_per_policy_step": (
                self.full_forward_calls / self.num_policy_steps
                if self.num_policy_steps else 0.0
            ),
            "hierarchical_mode": self.latentloop_hierarchical_mode,
            "hierarchical_full_interval": int(
                self.latentloop_hierarchical_schedule.full_interval
            ),
            "hierarchical_regeneration_interval": int(
                self.latentloop_hierarchical_schedule.regeneration_interval
            ),
            "hierarchical_level0_calls": self.hierarchical_level0_calls,
            "hierarchical_level1_calls": self.hierarchical_level1_calls,
            "hierarchical_level2_calls": self.hierarchical_level2_calls,
            "hierarchical_latent_updater_calls": (
                self.hierarchical_latent_updater_calls
            ),
            "hierarchical_action_correction_calls": (
                self.hierarchical_action_correction_calls
            ),
            "hierarchical_action_head_calls": self.hierarchical_action_head_calls,
            "joint_latent_action_surrogate_mode": (
                self.joint_latent_action_surrogate_mode
            ),
            "joint_surrogate_calls": self.joint_surrogate_calls,
            "joint_diagnostic_action_head_calls": (
                self.joint_diagnostic_action_head_calls
            ),
            "joint_diagnostic_full_forward_calls": (
                self.joint_diagnostic_full_forward_calls
            ),
            "joint_avg_diagnostic_full_forward_latency_sec": (
                self.joint_diagnostic_full_forward_latency_sum
                / self.joint_diagnostic_full_forward_calls
                / 1000.0
                if self.joint_diagnostic_full_forward_calls
                else 0.0
            ),
            "joint_avg_surrogate_latency_sec": (
                self.joint_surrogate_latency_sum
                / self.joint_surrogate_calls
                / 1000.0
                if self.joint_surrogate_calls
                else 0.0
            ),
            "hierarchical_avg_action_correction_latency_sec": (
                self.hierarchical_action_correction_latency_sum
                / self.hierarchical_action_correction_calls
                / 1000.0
                if self.hierarchical_action_correction_calls
                else 0.0
            ),
            "fully_synthetic_horizons_prevented": (
                self.hierarchical_provenance.fully_synthetic_horizons_prevented
            ),
        }

    def record_env_step_ms(self, env_step_ms, observation=None, reward=None, done=None, info=None):
        self.env_step_latency_sum += float(env_step_ms)
        if self.current_step_records:
            self.current_step_records[-1]["env_step_ms"] = float(env_step_ms)
            signals = extract_simulator_signals(observation, info)
            self.current_step_records[-1]["reward"] = "" if reward is None else float(reward)
            self.current_step_records[-1]["done"] = "" if done is None else int(bool(done))
            self.current_step_records[-1]["simulator_signals_json"] = json.dumps(
                signals,
                sort_keys=True,
            )
            if self.current_trace_scalars:
                self.current_trace_scalars[-1].update(
                    {
                        "env_step_ms": float(env_step_ms),
                        "reward": "" if reward is None else float(reward),
                        "done": "" if done is None else int(bool(done)),
                        "simulator_signals_json": json.dumps(signals, sort_keys=True),
                    }
                )
            if self.current_plan_trace_scalars:
                self.current_plan_trace_scalars[-1].update(
                    {
                        "env_step_ms": float(env_step_ms),
                        "reward": "" if reward is None else float(reward),
                        "done": "" if done is None else int(bool(done)),
                        "simulator_signals_json": json.dumps(signals, sort_keys=True),
                    }
                )

    def finish_episode(self, task, env, success, steps, args):
        records = list(self.current_step_records)

        def values(key):
            return [
                float(r.get(key, 0.0))
                for r in records
                if r.get(key, None) not in {None, ""}
            ]

        def mean(key):
            vals = values(key)
            return float(np.mean(vals)) if vals else 0.0

        def percentile(key, q):
            vals = values(key)
            return float(np.percentile(vals, q)) if vals else 0.0

        full_count = sum(1 for r in records if r.get("mode") == "full")
        if self.lrnode_every_step_filter is not None and full_count != len(records):
            raise RuntimeError(
                "Every-step latent filtering requires exactly one full Seer call "
                f"per policy step; full_steps={full_count}, policy_steps={len(records)}"
            )
        if self.lrnode_every_step_filter is not None and any(
            int(r.get("full_forward_called", 0)) != 1 for r in records
        ):
            raise RuntimeError(
                "Every-step latent filter trace contains a step without a full forward"
            )
        stepwise_count = sum(1 for r in records if r.get("mode") in {"lrnode_update", "stepwise"})
        no_delta_count = sum(1 for r in records if r.get("mode") == "no_delta")
        update_count = stepwise_count + no_delta_count
        hold_action_count = sum(1 for r in records if r.get("mode") == "hold_action")
        hold_latent_count = sum(1 for r in records if r.get("mode") == "hold_latent")
        chunk_token_count = sum(1 for r in records if r.get("mode") == "seer_token_chunk")
        hold_count = hold_action_count + hold_latent_count
        skip_step_count = max(0, len(records) - full_count)
        if self.latentloop_segment_executor is not None:
            invalid_full_offsets = [
                int(record.get("segment_offset", -1))
                for record in records
                if record.get("mode") == "full"
                and int(record.get("segment_offset", -1)) != 0
            ]
            if invalid_full_offsets:
                raise RuntimeError(
                    "LatentLoop full Seer call occurred outside segment offset 0: "
                    f"{invalid_full_offsets}"
                )
            if update_count != skip_step_count:
                raise RuntimeError(
                    "LatentLoop updater must execute at every intermediate step: "
                    f"updates={update_count}, intermediate_steps={skip_step_count}"
                )
            cache_advances = sum(
                int(record.get("observation_cache_advanced", 0))
                for record in records
            )
            if cache_advances != len(records):
                raise RuntimeError(
                    "LatentLoop observation cache must advance at every environment "
                    f"step: advances={cache_advances}, steps={len(records)}"
                )
        episode_wallclock = (
            time.perf_counter() - self.current_episode_start_time
            if self.current_episode_start_time is not None else 0.0
        )
        action_rows = [
            [float(record.get(f"action_{index}", 0.0)) for index in range(7)]
            for record in records
        ]
        actions = np.asarray(action_rows, dtype=np.float64).reshape(-1, 7)
        continuity = action_second_differences(actions) if len(actions) else {
            "translation": np.asarray([]),
            "rotation": np.asarray([]),
            "arm": np.asarray([]),
        }
        gripper_metrics = gripper_summary(actions) if len(actions) else {
            "gripper_switch_count": 0.0,
            "gripper_switches_per_100_steps": 0.0,
            "gripper_close_count": 0.0,
            "gripper_open_count": 0.0,
            "gripper_reverse_within_1_count": 0.0,
            "gripper_reverse_within_2_count": 0.0,
            "gripper_reverse_within_5_count": 0.0,
            "gripper_reverse_within_1_per_100_steps": 0.0,
            "gripper_reverse_within_2_per_100_steps": 0.0,
            "gripper_reverse_within_5_per_100_steps": 0.0,
        }
        feedback_records = [
            record
            for record in records
            if int(record.get("latentloop_segment_grid_enabled", 0)) == 1
            and record.get("feedback_mask", "") != ""
        ]
        episode_feedback_mask = [
            int(record["feedback_mask"]) for record in feedback_records
        ]
        observation_conditioned_count = sum(
            int(record.get("observation_conditioned_update_called", 0))
            for record in records
        )
        zero_feature_count = sum(
            int(record.get("zero_feature_update_called", 0))
            for record in records
        )
        feedback_denominator = observation_conditioned_count + zero_feature_count
        actual_feedback_density = (
            float(observation_conditioned_count) / float(feedback_denominator)
            if feedback_denominator
            else None
        )
        offset_metrics = {}
        for offset in sorted(
            {
                int(record["segment_offset"])
                for record in feedback_records
            }
        ):
            offset_rows = [
                record
                for record in feedback_records
                if int(record["segment_offset"]) == offset
            ]

            def offset_mean(key):
                offset_values = [
                    float(record[key])
                    for record in offset_rows
                    if record.get(key, "") not in ("", None)
                ]
                return float(np.mean(offset_values)) if offset_values else None

            offset_metrics[str(offset)] = {
                "count": len(offset_rows),
                "feedback_density": float(
                    np.mean([int(record["feedback_mask"]) for record in offset_rows])
                ),
                "update_norm": offset_mean("update_norm"),
                "gate_mean": offset_mean("gate_mean"),
                "latent_norm": offset_mean("z_norm"),
                "current_observation_feature_norm": offset_mean("u_delta_norm"),
                "shadow_latent_mse": offset_mean("shadow_latent_mse"),
                "shadow_latent_cos": offset_mean("shadow_latent_cos"),
            }
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
            "every_step_filter_mode": self.lrnode_every_step_filter_mode,
            "every_step_filter_alpha": self.lrnode_every_step_filter_alpha,
            "every_step_filter_beta": self.lrnode_every_step_filter_beta,
            "every_step_filter_diagnostics": int(
                self.lrnode_every_step_filter_diagnostics
            ),
            "query_reduction_claim_allowed": int(
                self.lrnode_every_step_filter is None
            ),
            "eval_skip_full_forward": int(self.lrnode_eval_skip_full_forward),
            "query_interval": int(self.lrnode_query_interval),
            "segment_length": int(self.lrnode_query_interval),
            "latentloop_segment_grid_enabled": int(
                self.latentloop_segment_executor is not None
            ),
            "feedback_schedule": (
                self.latentloop_feedback_schedule
                if self.latentloop_segment_executor is not None
                else "not_applicable"
            ),
            "planned_feedback_density": (
                ""
                if self.latentloop_segment_executor is None
                or self.latentloop_segment_executor.plan.density is None
                else float(self.latentloop_segment_executor.plan.density)
            ),
            "actual_feedback_density": (
                "" if actual_feedback_density is None else actual_feedback_density
            ),
            "feedback_mask_json": json.dumps(episode_feedback_mask),
            "feedback_offsets_json": json.dumps(
                [int(record["segment_offset"]) for record in feedback_records]
            ),
            "segment_offset_metrics_json": json.dumps(
                offset_metrics, sort_keys=True
            ),
            "observation_conditioned_updater_calls": int(
                observation_conditioned_count
            ),
            "zero_feature_updater_calls": int(zero_feature_count),
            "observation_cache_advance_calls": int(
                sum(
                    int(record.get("observation_cache_advanced", 0))
                    for record in records
                )
            ),
            "ablation_mode": self.lrnode_eval_ablation_mode,
            "no_delta_mode": self.lrnode_no_delta_mode,
            "chunk_token_policy": self.lrnode_chunk_token_policy,
            "refresh_policy": self.lrnode_eval_refresh_policy,
            "max_full_forwards_per_episode": int(self.lrnode_eval_max_full_forwards_per_episode),
            "mode_full_count": int(full_count),
            "mode_update_count": int(update_count),
            "mode_hold_count": int(hold_count),
            "mode_stepwise_count": int(stepwise_count),
            "mode_hold_action_count": int(hold_action_count),
            "mode_hold_latent_count": int(hold_latent_count),
            "mode_chunk_token_count": int(chunk_token_count),
            "mode_no_delta_count": int(no_delta_count),
            "mode_skip_step_count": int(skip_step_count),
            "full_forward_ratio": full_count / max(1, len(records)),
            "skip_ratio": update_count / max(1, len(records)),
            "nonfull_skip_ratio": skip_step_count / max(1, len(records)),
            "max_cache_age": int(max(values("cache_age") or [0.0])),
            "avg_full_forward_ms": mean("full_forward_ms"),
            "avg_full_action_head_ms": mean("full_action_head_ms"),
            "avg_full_non_action_head_ms": mean("full_non_action_head_ms"),
            "avg_fast_encoder_ms": mean("fast_encoder_ms"),
            "avg_node_update_ms": mean("node_update_ms"),
            "avg_action_head_ms": mean("action_head_ms"),
            "avg_filter_prior_ms": mean("filter_prior_ms"),
            "avg_filter_fusion_ms": mean("filter_fusion_ms"),
            "avg_filter_action_head_ms": mean("filter_action_head_ms"),
            "avg_filter_diagnostic_ms": mean("filter_diagnostic_ms"),
            "avg_policy_step_ms": mean("total_policy_ms"),
            "avg_protocol_policy_ms": mean("protocol_policy_ms"),
            "avg_causal_executed_policy_ms": mean(
                "causal_executed_policy_ms"
            ),
            "avg_diagnostic_full_forward_ms": mean(
                "diagnostic_full_forward_ms"
            ),
            "avg_total_diagnostic_only_ms": mean("total_diagnostic_only_ms"),
            "avg_env_step_ms": mean("env_step_ms"),
            "joint_surrogate_calls": int(
                sum(int(record.get("joint_surrogate_called", 0)) for record in records)
            ),
            "joint_exact_anchor_writes": int(
                sum(
                    int(record.get("joint_anchor_replaced_by_exact", 0))
                    for record in records
                )
            ),
            "avg_joint_surrogate_ms": mean("joint_surrogate_ms"),
            "joint_surrogate_latency_p50_ms": percentile("joint_surrogate_ms", 50),
            "joint_surrogate_latency_p95_ms": percentile("joint_surrogate_ms", 95),
            "joint_surrogate_latency_p99_ms": percentile("joint_surrogate_ms", 99),
            "joint_latent_error_p50": percentile("joint_latent_error", 50),
            "joint_latent_error_p90": percentile("joint_latent_error", 90),
            "joint_latent_error_p95": percentile("joint_latent_error", 95),
            "joint_latent_error_p99": percentile("joint_latent_error", 99),
            "joint_action_surrogate_error_p50": percentile(
                "joint_action_surrogate_error", 50
            ),
            "joint_action_surrogate_error_p90": percentile(
                "joint_action_surrogate_error", 90
            ),
            "joint_action_surrogate_error_p95": percentile(
                "joint_action_surrogate_error", 95
            ),
            "joint_action_surrogate_error_p99": percentile(
                "joint_action_surrogate_error", 99
            ),
            "joint_action_surrogate_error_before_regeneration": max(
                [
                    float(record["joint_action_surrogate_error"])
                    for record in records
                    if record.get("joint_action_surrogate_error") not in {None, ""}
                    and int(record.get("joint_anchor_elapsed", 0) or 0) == 2
                ]
                or [0.0]
            ),
            "joint_action_surrogate_error_after_regeneration": max(
                [
                    float(record["joint_action_surrogate_error"])
                    for record in records
                    if record.get("joint_action_surrogate_error") not in {None, ""}
                    and int(record.get("joint_anchor_replaced_by_exact", 0)) == 1
                ]
                or [0.0]
            ),
            "joint_latent_error_after_full_refresh": max(
                [
                    float(record["joint_latent_error"])
                    for record in records
                    if record.get("joint_latent_error") not in {None, ""}
                    and int(record.get("full_refresh_age", 0) or 0) == 0
                ]
                or [0.0]
            ),
            "episode_wallclock_sec": float(episode_wallclock),
            "avg_gate": mean("gate_mean"),
            "max_gate": max(values("gate_max") or [0.0]),
            "avg_image_diff_primary": mean("image_diff_primary_l1"),
            "avg_image_diff_wrist": mean("image_diff_wrist_l1"),
            "avg_update_norm": mean("update_norm"),
            "avg_latent_prior_vs_full_l2": mean(
                "latent_prior_vs_full_l2"
            ),
            "avg_latent_prior_vs_full_cosine": mean(
                "latent_prior_vs_full_cosine"
            ),
            "avg_latent_filter_vs_full_l2": mean(
                "latent_filter_vs_full_l2"
            ),
            "avg_latent_correction_l2": mean("latent_correction_l2"),
            "recurrent_prior_path_length": sum(
                values("latent_prior_step_l2")
            ),
            "full_latent_path_length": sum(values("latent_full_step_l2")),
            "filtered_latent_path_length": sum(
                values("latent_filter_step_l2")
            ),
            "latent_prior_second_difference_mean": mean(
                "latent_prior_second_difference_l2"
            ),
            "latent_full_second_difference_mean": mean(
                "latent_full_second_difference_l2"
            ),
            "latent_filter_second_difference_mean": mean(
                "latent_filter_second_difference_l2"
            ),
            "filter_diagnostics_rng_failures": sum(
                1
                for record in records
                if not int(record.get("filter_diagnostics_rng_preserved", 1))
            ),
            **{
                f"avg_latent_token{token_index}_{name}": mean(
                    f"latent_token{token_index}_{name}"
                )
                for token_index in range(self.action_pred_steps)
                for name in (
                    "full_norm",
                    "prior_norm",
                    "filter_norm",
                    "prior_vs_full_l2",
                    "filter_vs_full_l2",
                )
            },
            "avg_action_norm": mean("action_norm"),
            "avg_action_delta_l2": mean("action_delta_norm"),
            "p95_action_delta_l2": percentile("action_delta_norm", 95),
            "avg_action_jerk": mean("action_jerk"),
            "p95_action_jerk": percentile("action_jerk", 95),
            "max_action_jerk": max(values("action_jerk") or [0.0]),
            "mean_action_jerk": mean("action_jerk"),
            "arm_action_jerk": mean("arm_action_jerk"),
            "trans_action_jerk": mean("trans_action_jerk"),
            "rot_action_jerk": mean("rot_action_jerk"),
            "gripper_switch_rate": mean("gripper_switch"),
            "arm_jerk_normalized_mean": float(np.mean(continuity["arm"])) if len(actions) else 0.0,
            "translation_jerk_normalized_mean": (
                float(np.mean(continuity["translation"])) if len(actions) else 0.0
            ),
            "translation_jerk_normalized_p95": (
                float(np.percentile(continuity["translation"], 95))
                if len(actions) else 0.0
            ),
            "rotation_jerk_normalized_mean": (
                float(np.mean(continuity["rotation"])) if len(actions) else 0.0
            ),
            "rotation_jerk_normalized_p95": (
                float(np.percentile(continuity["rotation"], 95))
                if len(actions) else 0.0
            ),
            **gripper_metrics,
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
        if self._trace_enabled_for_current_episode() and self.current_trace_scalars:
            for row in self.current_trace_scalars:
                row["episode_success"] = int(success)
            rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
            ckpt_tag = _safe_name(os.environ.get("CKPT_TAG", "ckpt"))
            episode_key = (
                f"{ckpt_tag}_task{int(getattr(env, 'task_id', -1)):02d}_"
                f"episode{int(getattr(env, 'exp_id', 0)):03d}_rank{rank}"
            )
            save_trace_episode(
                self._trace_output_path(),
                episode_key,
                self.current_trace_scalars,
                self.current_trace_tensors,
                {
                    "checkpoint_id": os.environ.get("BASELINE_CKPT_ID", ""),
                    "adapter_id": os.environ.get("OURS_CKPT_ID", ""),
                    "checkpoint_tag": os.environ.get("CKPT_TAG", ""),
                    "task_id": int(getattr(env, "task_id", -1)),
                    "task_name": getattr(task, "name", getattr(env, "task_name", "")),
                    "episode_id": int(getattr(env, "exp_id", 0)),
                    "seed": int(getattr(args, "seed", 0)),
                    "success": int(success),
                    "steps": int(steps),
                    "query_interval": int(self.lrnode_query_interval),
                    "temporal_ensemble": int(self.use_ensembling),
                    "counterfactual_mode": self.lrnode_counterfactual_mode,
                    "counterfactual_mix_stage": self.lrnode_counterfactual_mix_stage,
                    "every_step_filter_mode": self.lrnode_every_step_filter_mode,
                    "every_step_filter_alpha": self.lrnode_every_step_filter_alpha,
                    "every_step_filter_beta": self.lrnode_every_step_filter_beta,
                    "every_step_filter_diagnostics": int(
                        self.lrnode_every_step_filter_diagnostics
                    ),
                },
            )
            self.trace_episode_count += 1
        if self.latentloop_plan_trace and self.current_plan_trace_scalars:
            for row in self.current_plan_trace_scalars:
                row["episode_success"] = int(success)
            rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
            episode_key = (
                f"{_safe_name(self.latentloop_plan_trace_row_id or 'row')}_"
                f"task{int(getattr(env, 'task_id', -1)):02d}_"
                f"episode{int(getattr(env, 'exp_id', 0)):03d}_rank{rank}"
            )
            save_plan_trace_episode(
                self._plan_trace_output_path(),
                episode_key,
                self.current_plan_trace_scalars,
                self.current_plan_trace_tensors,
                {
                    "row_id": self.latentloop_plan_trace_row_id,
                    "paired_group": self.latentloop_plan_trace_paired_group,
                    "task_id": int(getattr(env, "task_id", -1)),
                    "task_name": getattr(task, "name", getattr(env, "task_name", "")),
                    "episode_id": int(getattr(env, "exp_id", 0)),
                    "success": int(success),
                    "steps": int(steps),
                    "query_interval": int(self.lrnode_query_interval),
                    "feedback_source": self.latentloop_feedback_source,
                    "plan_adapter_mode": self.latentloop_plan_adapter_mode,
                    "temporal_ensemble": int(self.use_ensembling),
                },
            )
        if self.latentloop_hierarchical_trace and records:
            hierarchical_rows = [dict(row) for row in records]
            for row in hierarchical_rows:
                row["episode_success"] = int(success)
                row["seed"] = int(getattr(args, "seed", 0))
            rank = int(torch.distributed.get_rank()) if torch.distributed.is_initialized() else 0
            episode_key = (
                f"{_safe_name(os.environ.get('CKPT_TAG', 'row'))}_"
                f"task{int(getattr(env, 'task_id', -1)):02d}_"
                f"episode{int(getattr(env, 'exp_id', 0)):03d}_rank{rank}"
            )
            save_hierarchical_trace(
                self._hierarchical_trace_output_path(),
                episode_key,
                hierarchical_rows,
                {
                    "mode": self.latentloop_hierarchical_mode,
                    "full_interval": int(
                        self.latentloop_hierarchical_schedule.full_interval
                    ),
                    "regeneration_interval": int(
                        self.latentloop_hierarchical_schedule.regeneration_interval
                    ),
                    "action_pred_steps": int(self.action_pred_steps),
                    "task_id": int(getattr(env, "task_id", -1)),
                    "task_name": getattr(task, "name", getattr(env, "task_name", "")),
                    "episode_id": int(getattr(env, "exp_id", 0)),
                    "seed": int(getattr(args, "seed", 0)),
                    "success": int(success),
                    "steps": int(steps),
                    "temporal_ensemble": int(self.use_ensembling),
                    "provenance_labels": list(PROVENANCE_LABELS),
                    "invariants_asserted": int(
                        self.latentloop_hierarchical_assert_invariants
                    ),
                },
            )
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
        every_step_filter = self.lrnode_every_step_filter
        previous_cached_latent = None
        if (
            self.lrnode_cached_latent is not None
            and (
                self.lrnode_mechanism_trace
                or self.lrnode_counterfactual_mode != "standard"
                or every_step_filter is not None
            )
        ):
            previous_cached_latent = self.lrnode_cached_latent.detach().clone()
        shadow = None
        lrnode_debug = {}
        z_prior = None
        z_full = None
        z_filter = None
        full_action_seq = None
        prior_action_seq = None
        filter_action_seq = None
        executed_probability = None
        shadow_executed_probability = None
        shadow_env_action = None
        ensemble_candidate_count = 0
        shadow_ensemble_candidate_count = 0
        shadow_diagnostic_ms = 0.0
        joint_latent_diagnostic_ms = 0.0
        refresh_diagnostic_ms = 0.0
        filter_diagnostic_ms = 0.0
        shadow_ensemble_ms = 0.0
        arm_source = "full"
        gripper_source = "full"
        hierarchical_level = None
        hierarchical_latent_version_before = self.hierarchical_latent_cache_version
        hierarchical_action_generation_before = self.hierarchical_action_cache_generation
        if self.latentloop_hierarchical_schedule.enabled:
            hierarchical_level = self.latentloop_hierarchical_schedule.level(
                timestep,
                has_latent_cache=self.lrnode_cached_latent is not None,
            )
        segment_decision = None
        if self.latentloop_segment_executor is not None:
            segment_decision = self.latentloop_segment_executor.decision(
                timestep,
                has_latent_cache=self.lrnode_cached_latent is not None,
            )
        step_record = {
            "timestep": int(timestep),
            "mode": "full",
            "cache_age": int(self.lrnode_cached_age),
            "skip_age": 0,
            "token_idx_used": "",
            "query_interval": int(self.lrnode_query_interval),
            "latentloop_segment_grid_enabled": int(
                self.latentloop_segment_executor is not None
            ),
            "segment_length": int(self.lrnode_query_interval),
            "segment_offset": (
                "" if segment_decision is None else int(segment_decision.segment_offset)
            ),
            "feedback_schedule": (
                "not_applicable"
                if segment_decision is None
                else self.latentloop_feedback_schedule
            ),
            "feedback_mask": (
                ""
                if segment_decision is None
                or segment_decision.feedback_enabled is None
                else int(segment_decision.feedback_enabled)
            ),
            "planned_feedback_density": (
                ""
                if segment_decision is None
                or segment_decision.planned_feedback_density is None
                else float(segment_decision.planned_feedback_density)
            ),
            "observation_conditioned_update_called": 0,
            "zero_feature_update_called": 0,
            "observation_cache_advanced": 0,
            "ablation_mode": self.lrnode_eval_ablation_mode,
            "no_delta_mode": self.lrnode_no_delta_mode,
            "chunk_token_policy": self.lrnode_chunk_token_policy,
            "refresh_policy": self.lrnode_eval_refresh_policy,
            "max_full_forwards_per_episode": int(self.lrnode_eval_max_full_forwards_per_episode),
            "full_forward_called": 0,
            "lrnode_update_called": 0,
            "fast_encoder_called": 0,
            "action_head_called": 0,
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
            "task_id": int(self.current_task_id),
            "task_name": self.current_task_name,
            "episode_id": int(self.current_episode_id),
            "seed": self.evaluation_seed,
            "checkpoint_id": os.environ.get("BASELINE_CKPT_ID", ""),
            "adapter_id": os.environ.get("OURS_CKPT_ID", ""),
            "counterfactual_mode": self.lrnode_counterfactual_mode,
            "counterfactual_mix_stage": self.lrnode_counterfactual_mix_stage,
            "temporal_ensemble_enabled": int(self.use_ensembling),
            "every_step_filter_mode": self.lrnode_every_step_filter_mode,
            "every_step_filter_alpha": self.lrnode_every_step_filter_alpha,
            "every_step_filter_beta": self.lrnode_every_step_filter_beta,
            "every_step_filter_diagnostics": int(
                self.lrnode_every_step_filter_diagnostics
            ),
            "filter_prior_called": 0,
            "filter_prior_ms": 0.0,
            "filter_fusion_ms": 0.0,
            "filter_action_head_called": 0,
            "filter_action_head_ms": 0.0,
            "filter_diagnostic_action_head_calls": 0,
            "filter_diagnostic_ms": 0.0,
            "filter_diagnostics_rng_preserved": 1,
            "query_reduction_claim_allowed": int(every_step_filter is None),
            "hierarchical_mode": self.latentloop_hierarchical_mode,
            "hierarchical_level": (
                "" if hierarchical_level is None else int(hierarchical_level)
            ),
            "full_refresh_age": int(self.lrnode_cached_age),
            "action_head_regeneration_age": int(self.hierarchical_regeneration_age),
            "latent_cache_source": "full_seer",
            "action_horizon_cache_source": "full_seer",
            "executed_action_source": "",
            "hierarchical_step_full_seer_calls": 0,
            "hierarchical_step_action_head_calls": 0,
            "hierarchical_step_latent_updater_calls": 0,
            "hierarchical_step_action_correction_calls": 0,
            "action_correction_residual_norm": 0.0,
            "action_cache_generation": int(self.hierarchical_action_cache_generation),
            "action_cache_input_generation": int(
                self.hierarchical_action_cache_input_generation
            ),
            "fully_synthetic_horizons_prevented": int(
                self.hierarchical_provenance.fully_synthetic_horizons_prevented
            ),
            "joint_latent_error": "",
            "joint_latent_diagnostic_full_forward_ms": 0.0,
        }
        # preprocess image
        image = obs["agentview_image"]
        raw_primary_for_trace = None
        raw_wrist_for_trace = None
        if self.latentloop_plan_trace:
            raw_primary_for_trace = np.asarray(image, dtype=np.float32)
            raw_wrist_for_trace = np.asarray(
                obs["robot0_eye_in_hand_image"], dtype=np.float32
            )
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
        raw_proprio_for_trace = None
        if self.latentloop_plan_trace:
            raw_proprio_for_trace = state.detach().float().cpu().numpy()[0, 0]
            step_record["primary_raw_change_l1"] = (
                0.0
                if self.previous_raw_primary is None
                else float(np.mean(np.abs(raw_primary_for_trace - self.previous_raw_primary)))
            )
            step_record["wrist_raw_change_l1"] = (
                0.0
                if self.previous_raw_wrist is None
                else float(np.mean(np.abs(raw_wrist_for_trace - self.previous_raw_wrist)))
            )
            step_record["proprio_delta_l2"] = (
                0.0
                if self.previous_raw_proprio is None
                else float(np.linalg.norm(raw_proprio_for_trace - self.previous_raw_proprio))
            )

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

            self._run_same_input_stochasticity_sanity(
                input_image_primary,
                input_image_wrist,
                input_state,
                input_text_token,
                self._selected_step(num_step),
                timestep,
            )

            direct_env_action = None
            use_lrnode_step = self._should_use_lrnode(timestep)
            if hierarchical_level is not None:
                expected_skip = hierarchical_level != ExecutionLevel.FULL_SEER
                if use_lrnode_step != expected_skip:
                    raise RuntimeError(
                        "Hierarchical schedule disagrees with Seer skip routing: "
                        f"timestep={timestep}, level={int(hierarchical_level)}, "
                        f"skip={use_lrnode_step}"
                    )
            if segment_decision is not None:
                expected_skip = not segment_decision.full_refresh
                if use_lrnode_step != expected_skip:
                    raise RuntimeError(
                        "LatentLoop segment schedule disagrees with the existing "
                        f"periodic skip path at timestep={timestep}: "
                        f"expected_skip={expected_skip}, actual_skip={use_lrnode_step}"
                    )
            diagnostic_refresh_debug = None
            if (
                every_step_filter is not None
                and every_step_filter.requires_recurrent_prior(
                    has_previous=self.lrnode_cached_latent is not None
                )
            ):
                prior_t0 = time.perf_counter()
                with preserve_rng_state(include_cuda=True):
                    _, lrnode_debug = self._update_from_lrnode_cache(
                        image_x,
                        gripper,
                        state,
                        use_zero_delta=False,
                        compute_hold_action=False,
                        commit_cache=False,
                        decode_action=False,
                        timestep=timestep,
                    )
                prior_ms = (time.perf_counter() - prior_t0) * 1000.0
                z_prior = lrnode_debug["z_pred"].detach()
                self.lrnode_update_calls += 1
                self.fast_encoder_calls += int(
                    lrnode_debug.get("fast_encoder_called", 0)
                )
                self.lrnode_latency_sum += prior_ms / 1000.0
                self.fast_encoder_latency_sum += float(
                    lrnode_debug.get("fast_encoder_ms", 0.0)
                )
                self.node_update_latency_sum += float(
                    lrnode_debug.get("node_update_ms", 0.0)
                )
                self.every_step_filter_prior_calls += 1
                self.every_step_filter_prior_latency_sum += prior_ms / 1000.0
                step_record.update(
                    {
                        "lrnode_update_called": 1,
                        "fast_encoder_called": int(
                            lrnode_debug.get("fast_encoder_called", 0)
                        ),
                        "fast_encoder_ms": float(
                            lrnode_debug.get("fast_encoder_ms", 0.0)
                        ),
                        "node_update_ms": float(
                            lrnode_debug.get("node_update_ms", 0.0)
                        ),
                        "filter_prior_called": 1,
                        "filter_prior_ms": prior_ms,
                        "cache_age": int(lrnode_debug.get("cache_age", 1)),
                        "u_delta_norm": float(
                            lrnode_debug.get("u_delta_norm", 0.0)
                        ),
                        "image_diff_primary_l1": float(
                            lrnode_debug.get("image_diff_primary_l1", 0.0)
                        ),
                        "image_diff_wrist_l1": float(
                            lrnode_debug.get("image_diff_wrist_l1", 0.0)
                        ),
                        "update_norm": float(
                            lrnode_debug.get("update_norm", 0.0)
                        ),
                        "gate_mean": float(lrnode_debug.get("gate_mean", 0.0)),
                        "gate_max": float(lrnode_debug.get("gate_max", 0.0)),
                    }
                )
            if (
                every_step_filter is None
                and
                not use_lrnode_step
                and self.lrnode_cached_latent is not None
                and (
                    self.lrnode_mechanism_trace
                    or self.lrnode_counterfactual_mode != "standard"
                )
            ):
                refresh_diagnostic_t0 = time.perf_counter()
                with preserve_rng_state(include_cuda=True):
                    _, diagnostic_refresh_debug = self._update_from_lrnode_cache(
                        image_x,
                        gripper,
                        state,
                        use_zero_delta=False,
                        compute_hold_action=False,
                        commit_cache=False,
                        timestep=timestep,
                    )
                refresh_diagnostic_ms = (
                    time.perf_counter() - refresh_diagnostic_t0
                ) * 1000.0
            if use_lrnode_step:
                self._sync_cuda()
                t0 = time.perf_counter()
                mode = self.lrnode_eval_ablation_mode
                if hierarchical_level is not None:
                    if self.joint_latent_action_surrogate_mode == "wide":
                        mode = "joint_wide_exact_head"
                        action_seq, lrnode_debug = self._wide_exact_update(
                            image_x, gripper, state, timestep
                        )
                    elif (
                        self.joint_latent_action_surrogate_mode == "joint"
                        and hierarchical_level == ExecutionLevel.HORIZON_REGENERATION
                    ):
                        mode = "joint_exact_action_head_regeneration"
                        action_seq, lrnode_debug = self._joint_exact_regeneration(
                            image_x, gripper, state, timestep
                        )
                    elif self.joint_latent_action_surrogate_mode == "joint":
                        mode = "joint_anchor_relative_surrogate"
                        action_seq, lrnode_debug = self._joint_fast_surrogate(
                            image_x, gripper, state, timestep
                        )
                        self.hierarchical_provenance.correct()
                    elif hierarchical_level == ExecutionLevel.HORIZON_REGENERATION:
                        mode = "hierarchical_level_1_regeneration"
                        action_seq, lrnode_debug = self._regenerate_horizon_from_latent(
                            image_x, gripper, state, timestep
                        )
                    elif self.latentloop_hierarchical_mode == "pure_action_correction":
                        mode = "hierarchical_level_0_action_endpoint"
                        self.hierarchical_action_cache_input_generation = (
                            self.hierarchical_action_cache_generation
                        )
                        action_seq, lrnode_debug = self._update_action_correction_cache(
                            image_x, gripper, state, timestep
                        )
                        self.hierarchical_regeneration_age += 1
                        self.hierarchical_provenance.correct()
                    else:
                        mode = "hierarchical_level_0_hybrid"
                        action_seq, lrnode_debug = self._update_hybrid_level0(
                            image_x, gripper, state, timestep
                        )
                        self.hierarchical_provenance.correct()
                elif self.latentloop_plan_adapter_mode == "action_correction":
                    mode = "action_correction"
                    action_seq, lrnode_debug = self._update_action_correction_cache(
                        image_x, gripper, state, timestep
                    )
                elif self.latentloop_plan_adapter_mode == "anchor_bridge":
                    mode = "anchor_bridge"
                    action_seq, lrnode_debug = self._update_anchor_bridge_cache(
                        image_x, gripper, state, timestep
                    )
                elif mode == "hold_action":
                    if self.lrnode_cached_env_action is None:
                        raise RuntimeError("hold_action ablation requested before an executed full-step action was cached")
                    self.lrnode_cached_age += 1
                    direct_env_action = self.lrnode_cached_env_action.copy()
                    action_seq = None
                    lrnode_debug = {
                        "cache_age": int(self.lrnode_cached_age),
                        "skip_age": int(self.lrnode_cached_age),
                        "fast_encoder_called": 0,
                        "lrnode_update_called": 0,
                        "action_head_called": 0,
                        "fast_encoder_ms": 0.0,
                        "node_update_ms": 0.0,
                        "action_head_ms": 0.0,
                        "gate_mean": 0.0,
                        "gate_max": 0.0,
                        "u_delta_norm": 0.0,
                        "image_diff_primary_l1": 0.0,
                        "image_diff_wrist_l1": 0.0,
                        "update_norm": 0.0,
                        "z_norm": 0.0,
                        "feature_source_step": -1,
                        "action_pred": self.lrnode_cached_action_tokens,
                        "action_pred_gripper_logit": self.lrnode_cached_gripper_logit,
                        "action_pred_gripper_probability": (
                            None
                            if self.lrnode_cached_action_tokens is None
                            else self.lrnode_cached_action_tokens[..., 6:]
                        ),
                    }
                elif mode == "hold_latent":
                    action_seq, lrnode_debug = self._decode_from_cached_latent()
                elif mode == "seer_token_chunk":
                    direct_env_action, lrnode_debug = self._cached_chunk_token_action(timestep)
                    action_seq = None
                else:
                    use_zero_delta = mode == "no_delta"
                    if segment_decision is not None:
                        if segment_decision.feedback_enabled is None:
                            raise RuntimeError(
                                "Intermediate LatentLoop step has no feedback decision"
                            )
                        use_zero_delta = segment_decision.use_zero_feature
                    action_seq, lrnode_debug = self._update_from_lrnode_cache(
                        image_x,
                        gripper,
                        state,
                        use_zero_delta=use_zero_delta,
                        compute_hold_action=self.lrnode_eval_shadow_full_forward,
                        timestep=timestep,
                    )
                self._sync_cuda()
                skip_ms = (time.perf_counter() - t0) * 1000.0

                lrnode_update_called = int(lrnode_debug.get("lrnode_update_called", 0))
                fast_encoder_called = int(lrnode_debug.get("fast_encoder_called", 0))
                action_head_called = int(lrnode_debug.get("action_head_called", 0))
                latent_updater_called = int(
                    lrnode_debug.get("latent_updater_called", lrnode_update_called)
                )
                action_correction_called = int(
                    lrnode_debug.get("action_correction_called", 0)
                )
                self.lrnode_update_calls += lrnode_update_called
                self.fast_encoder_calls += fast_encoder_called
                self.action_head_calls += action_head_called
                if hierarchical_level is not None:
                    self.hierarchical_latent_updater_calls += latent_updater_called
                    self.hierarchical_action_correction_calls += action_correction_called
                    self.hierarchical_action_head_calls += action_head_called
                    self.hierarchical_action_correction_latency_sum += float(
                        lrnode_debug.get("action_correction_ms", 0.0)
                    )
                    if hierarchical_level == ExecutionLevel.ACTION_CORRECTION:
                        self.hierarchical_level0_calls += 1
                    elif hierarchical_level == ExecutionLevel.HORIZON_REGENERATION:
                        self.hierarchical_level1_calls += 1
                observation_conditioned_called = int(
                    lrnode_debug.get("observation_conditioned_update_called", 0)
                )
                zero_feature_called = int(
                    lrnode_debug.get("zero_feature_update_called", 0)
                )
                observation_cache_advanced = int(
                    lrnode_debug.get("observation_cache_advanced", 0)
                )
                self.observation_conditioned_update_calls += (
                    observation_conditioned_called
                )
                self.zero_feature_update_calls += zero_feature_called
                self.observation_cache_advance_calls += observation_cache_advanced
                joint_diagnostic_ms = float(
                    lrnode_debug.get("joint_diagnostic_action_head_ms", 0.0)
                )
                if lrnode_update_called:
                    self.lrnode_latency_sum += max(
                        0.0, skip_ms - joint_diagnostic_ms
                    ) / 1000.0
                self.fast_encoder_latency_sum += float(lrnode_debug.get("fast_encoder_ms", 0.0))
                self.node_update_latency_sum += float(lrnode_debug.get("node_update_ms", 0.0))
                self.action_head_latency_sum += float(lrnode_debug.get("action_head_ms", 0.0))
                if mode == "hold_action":
                    self.hold_action_steps += 1
                elif mode == "hold_latent":
                    self.hold_latent_steps += 1
                elif mode == "seer_token_chunk":
                    self.chunk_token_steps += 1
                elif mode == "no_delta":
                    self.no_delta_steps += 1
                step_record.update(
                    {
                        "mode": mode,
                        "cache_age": int(lrnode_debug.get("cache_age", 0)),
                        "skip_age": int(lrnode_debug.get("skip_age", lrnode_debug.get("cache_age", 0))),
                        "token_idx_used": lrnode_debug.get("token_idx_used", ""),
                        "lrnode_update_called": lrnode_update_called,
                        "fast_encoder_called": fast_encoder_called,
                        "action_head_called": action_head_called,
                        "hierarchical_step_action_head_calls": action_head_called,
                        "hierarchical_step_latent_updater_calls": latent_updater_called,
                        "hierarchical_step_action_correction_calls": action_correction_called,
                        "latent_cache_source": (
                            "unchanged_endpoint"
                            if self.latentloop_hierarchical_mode
                            == "pure_action_correction"
                            else "canonical_latentloop"
                        ),
                        "action_horizon_cache_source": (
                            "regenerated"
                            if hierarchical_level
                            == ExecutionLevel.HORIZON_REGENERATION
                            else "shifted_corrected"
                        ),
                        "full_refresh_age": int(lrnode_debug.get("cache_age", 0)),
                        "action_head_regeneration_age": int(
                            self.hierarchical_regeneration_age
                        ),
                        "action_correction_ms": float(
                            lrnode_debug.get("action_correction_ms", 0.0)
                        ),
                        "action_correction_residual_norm": float(
                            lrnode_debug.get("action_correction_residual_norm", 0.0)
                        ),
                        "joint_surrogate_called": int(
                            lrnode_debug.get("joint_surrogate_called", 0)
                        ),
                        "joint_surrogate_ms": float(
                            lrnode_debug.get("joint_surrogate_ms", 0.0)
                        ),
                        "joint_anchor_elapsed": lrnode_debug.get(
                            "joint_anchor_elapsed", ""
                        ),
                        "joint_anchor_generation": lrnode_debug.get(
                            "joint_anchor_generation", ""
                        ),
                        "joint_anchor_replaced_by_exact": int(
                            lrnode_debug.get("joint_anchor_replaced_by_exact", 0)
                        ),
                        "joint_valid_token_fraction": lrnode_debug.get(
                            "joint_valid_token_fraction", ""
                        ),
                        "joint_action_surrogate_error": lrnode_debug.get(
                            "joint_action_surrogate_error", ""
                        ),
                        "joint_executed_token_error": lrnode_debug.get(
                            "joint_executed_token_error", ""
                        ),
                        "joint_tail_token_error": lrnode_debug.get(
                            "joint_tail_token_error", ""
                        ),
                        "joint_diagnostic_action_head_ms": float(
                            lrnode_debug.get("joint_diagnostic_action_head_ms", 0.0)
                        ),
                        "action_cache_generation": int(
                            self.hierarchical_action_cache_generation
                        ),
                        "action_cache_input_generation": int(
                            self.hierarchical_action_cache_input_generation
                        ),
                        "fully_synthetic_horizons_prevented": int(
                            self.hierarchical_provenance.fully_synthetic_horizons_prevented
                        ),
                        "observation_conditioned_update_called": (
                            observation_conditioned_called
                        ),
                        "zero_feature_update_called": zero_feature_called,
                        "observation_cache_advanced": observation_cache_advanced,
                        "fast_encoder_ms": float(lrnode_debug.get("fast_encoder_ms", 0.0)),
                        "node_update_ms": float(lrnode_debug.get("node_update_ms", 0.0)),
                        "action_head_ms": float(lrnode_debug.get("action_head_ms", 0.0)),
                        "total_policy_ms": skip_ms + preprocess_ms,
                        "gate_mean": float(lrnode_debug.get("gate_mean", 0.0)),
                        "gate_max": float(lrnode_debug.get("gate_max", 0.0)),
                        "u_delta_norm": float(lrnode_debug.get("u_delta_norm", 0.0)),
                        "feature_source_step": int(
                            lrnode_debug.get("feature_source_step", timestep)
                        ),
                        "feedback_source": lrnode_debug.get(
                            "feedback_source", self.latentloop_feedback_source
                        ),
                        "time_shift_initialized_with_zero": int(
                            lrnode_debug.get("time_shift_initialized_with_zero", 0)
                        ),
                        "image_diff_primary_l1": float(lrnode_debug.get("image_diff_primary_l1", 0.0)),
                        "image_diff_wrist_l1": float(lrnode_debug.get("image_diff_wrist_l1", 0.0)),
                        "proprio_delta_l2": float(
                            lrnode_debug.get(
                                "proprio_delta_l2",
                                step_record.get("proprio_delta_l2", 0.0),
                            )
                        ),
                        "update_norm": float(lrnode_debug.get("update_norm", 0.0)),
                        "z_norm": float(lrnode_debug.get("z_norm", 0.0)),
                    }
                )
                if (
                    self.joint_error_trace
                    and self.joint_latent_action_surrogate_mode != "off"
                    and lrnode_debug.get("z_pred") is not None
                ):
                    joint_shadow = self._run_shadow_full_forward(
                        input_image_primary,
                        input_image_wrist,
                        input_state,
                        input_text_token,
                        self._selected_step(num_step),
                    )
                    predicted_latent = lrnode_debug["z_pred"].detach().float()
                    exact_latent = joint_shadow["latent"].detach().float()
                    joint_latent_error = torch.sqrt(
                        F.mse_loss(predicted_latent, exact_latent)
                    ).item()
                    joint_latent_diagnostic_ms = float(joint_shadow["latency_ms"])
                    self.joint_diagnostic_full_forward_calls += 1
                    self.joint_diagnostic_full_forward_latency_sum += (
                        joint_latent_diagnostic_ms
                    )
                    step_record.update(
                        {
                            "joint_latent_error": joint_latent_error,
                            "joint_latent_error_full_refresh_age": int(
                                lrnode_debug.get("cache_age", 0)
                            ),
                            "joint_latent_diagnostic_full_forward_ms": (
                                joint_latent_diagnostic_ms
                            ),
                        }
                    )
                if self.lrnode_eval_shadow_full_forward and "action_pred" in lrnode_debug:
                    shadow_diagnostic_t0 = time.perf_counter()
                    selected_step = self._selected_step(num_step)
                    shadow = self._run_shadow_full_forward(
                        input_image_primary,
                        input_image_wrist,
                        input_state,
                        input_text_token,
                        selected_step,
                    )
                    pred_action = lrnode_debug["action_pred"].detach().float()
                    hold_action = lrnode_debug["action_hold"].detach().float()
                    pred_latent = lrnode_debug["z_pred"].detach().float()
                    shadow_latent = shadow["latent"].detach().float()
                    shadow_action = shadow["action"].detach().float()
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
                    self.shadow_full_forward_latency_sum += shadow["latency_ms"]
                    self.shadow_latent_mse_sum += latent_mse
                    self.shadow_latent_cos_sum += latent_cos
                    self.shadow_action_l1_sum += action_l1
                    self.shadow_action_l2_sum += action_l2
                    self.shadow_action_hold_l1_sum += action_hold_l1
                    step_record.update(
                        {
                            "shadow_full_forward_ms": shadow["latency_ms"],
                            "shadow_latent_mse": latent_mse,
                            "shadow_latent_cos": latent_cos,
                            "shadow_action_l1": action_l1,
                            "shadow_action_l2": action_l2,
                            "shadow_action_hold_l1": action_hold_l1,
                            "shadow_pred_vs_hold_improvement": improvement,
                        }
                    )
                    (
                        action_seq,
                        executed_latent,
                        arm_source,
                        gripper_source,
                        counterfactual_extra,
                    ) = self._apply_skip_counterfactual(
                        timestep,
                        action_seq,
                        lrnode_debug,
                        shadow,
                    )
                    lrnode_debug["executed_latent"] = (
                        None if executed_latent is None else executed_latent.detach()
                    )
                    lrnode_debug.update(counterfactual_extra)
                    step_record["arm_source"] = arm_source
                    step_record["gripper_source"] = gripper_source
                    step_record["learned_delta_norm"] = float(
                        counterfactual_extra.get("learned_delta_norm", 0.0)
                    )
                    step_record["random_delta_norm"] = float(
                        counterfactual_extra.get("random_delta_norm", 0.0)
                    )
                    shadow_diagnostic_ms = (
                        time.perf_counter() - shadow_diagnostic_t0
                    ) * 1000.0
            else:
                self._sync_cuda()
                t0 = time.perf_counter()
                model_outputs = self.model(
                    image_primary=input_image_primary,
                    image_wrist=input_image_wrist,
                    state=input_state,
                    text_token=input_text_token,
                    action=torch.zeros(1, self.history_len, 7).to(input_state.device),
                    return_action_latent=(
                        self.use_lrnode_latent_update or self.latentloop_plan_trace
                    ),
                )
                self._sync_cuda()
                full_ms = (time.perf_counter() - t0) * 1000.0
                full_action_head_ms = (
                    float(getattr(self._base_model(), "last_full_action_head_ms", 0.0))
                    if self.lrnode_eval_profile_full_action_head else 0.0
                )
                full_non_action_head_ms = max(0.0, full_ms - full_action_head_ms)
                self.full_forward_latency_sum += full_ms / 1000.0
                self.full_action_head_latency_sum += full_action_head_ms / 1000.0
                self.full_non_action_head_latency_sum += full_non_action_head_ms / 1000.0
                self.full_forward_calls += 1
                self.lrnode_episode_full_forward_calls += 1
                step_record.update(
                    {
                        "mode": "full",
                        "full_forward_called": 1,
                        "full_forward_ms": full_ms,
                        "full_action_head_ms": full_action_head_ms,
                        "full_non_action_head_ms": full_non_action_head_ms,
                        "full_refresh_reason": self._full_refresh_reason(timestep),
                    }
                )

                if self.use_lrnode_latent_update or self.latentloop_plan_trace:
                    arm_action = model_outputs["arm_pred_action"]
                    gripper_action = model_outputs["gripper_pred_action"]
                    action_latent = model_outputs["action_latent"]
                else:
                    arm_action, gripper_action, _, _, _, _ = model_outputs
                    action_latent = None
                selected_step = self._selected_step(num_step)
                action_seq = torch.concat((arm_action[:, selected_step], gripper_action[:, selected_step]), dim=-1)
                cache_latent = None if action_latent is None else action_latent[:, selected_step].detach()
                full_action_seq = action_seq.detach()
                z_full = cache_latent
                full_diagnostics = None
                if (
                    every_step_filter is None
                    and action_latent is not None
                    and (
                    self.lrnode_eval_shadow_full_forward or self.lrnode_mechanism_trace
                    or self.latentloop_plan_trace
                    or self.latentloop_plan_adapter_mode != "off"
                    or self.joint_latent_action_surrogate_mode != "off"
                    )
                ):
                    full_diagnostics = {
                        "arm": arm_action[:, selected_step],
                        "gripper_logit": model_outputs["action_gripper_logit"][
                            :, selected_step
                        ],
                        "gripper_probability": gripper_action[:, selected_step],
                    }
                    shadow = {
                        "latent": cache_latent,
                        "action": torch.cat(
                            [
                                full_diagnostics["arm"],
                                full_diagnostics["gripper_probability"],
                            ],
                            dim=-1,
                        ).detach(),
                        "arm": full_diagnostics["arm"].detach(),
                        "gripper_logit": full_diagnostics["gripper_logit"].detach(),
                        "gripper_probability": full_diagnostics[
                            "gripper_probability"
                        ].detach(),
                        "latency_ms": 0.0,
                    }
                    lrnode_debug.update(
                        {
                            "action_pred": shadow["action"],
                            "action_pred_gripper_logit": shadow["gripper_logit"],
                            "action_pred_gripper_probability": shadow[
                                "gripper_probability"
                            ],
                            "z_pred": cache_latent,
                            "feature_source_step": int(timestep),
                        }
                    )
                if every_step_filter is not None:
                    if cache_latent is None:
                        raise RuntimeError(
                            "Every-step latent filtering requires a full Seer action latent"
                        )
                    self._sync_cuda()
                    fusion_t0 = time.perf_counter()
                    filter_selection = every_step_filter.select(
                        z_full=cache_latent,
                        z_previous=previous_cached_latent,
                        z_prior=z_prior,
                    )
                    self._sync_cuda()
                    fusion_ms = (time.perf_counter() - fusion_t0) * 1000.0
                    z_filter = filter_selection.latent
                    cache_latent = z_filter
                    step_record["filter_fusion_ms"] = fusion_ms
                    self.every_step_filter_fusion_latency_sum += fusion_ms / 1000.0
                    if (
                        not filter_selection.initialized_from_full
                        and not filter_selection.reused_full_action
                        and self.lrnode_every_step_filter_mode
                        in {"fixed_filter", "full_latent_ema"}
                    ):
                        self.every_step_filter_fusion_calls += 1

                    if filter_selection.reused_full_action:
                        action_seq = full_action_seq
                        filter_action_seq = full_action_seq
                    else:
                        self._sync_cuda()
                        filter_head_t0 = time.perf_counter()
                        filter_arm, filter_gripper = (
                            self._base_model().decode_action_from_latent(z_filter)
                        )
                        self._sync_cuda()
                        filter_head_ms = (
                            time.perf_counter() - filter_head_t0
                        ) * 1000.0
                        action_seq = torch.cat(
                            [filter_arm, filter_gripper],
                            dim=-1,
                        )
                        filter_action_seq = action_seq.detach()
                        self.every_step_filter_action_head_calls += 1
                        self.every_step_filter_action_head_latency_sum += (
                            filter_head_ms / 1000.0
                        )
                        step_record.update(
                            {
                                "filter_action_head_called": 1,
                                "filter_action_head_ms": filter_head_ms,
                            }
                        )

                    if z_prior is not None and z_filter is z_prior:
                        prior_action_seq = filter_action_seq

                    self._sync_cuda()
                    diagnostic_t0 = time.perf_counter()
                    rng_before = capture_rng_state(include_cuda=True)
                    diagnostic_head_calls = 0
                    with preserve_rng_state(include_cuda=True):
                        latent_metrics = every_step_filter.diagnostics(
                            selection=filter_selection,
                            z_full=z_full,
                            z_prior=z_prior,
                        )
                        filter_diagnostics = None
                        full_filter_diagnostics = None
                        prior_diagnostics = None
                        if self.lrnode_every_step_filter_diagnostics:
                            full_filter_diagnostics = (
                                self._base_model().decode_action_diagnostics_from_latent(
                                    z_full
                                )
                            )
                            diagnostic_head_calls += 1
                            filter_diagnostics = (
                                full_filter_diagnostics
                                if filter_selection.reused_full_action
                                else self._base_model().decode_action_diagnostics_from_latent(
                                    z_filter
                                )
                            )
                            if not filter_selection.reused_full_action:
                                diagnostic_head_calls += 1
                            if z_prior is not None:
                                if z_filter is z_prior:
                                    prior_diagnostics = filter_diagnostics
                                else:
                                    prior_diagnostics = (
                                        self._base_model().decode_action_diagnostics_from_latent(
                                            z_prior
                                        )
                                    )
                                    diagnostic_head_calls += 1
                        latent_metrics[
                            "full_raw_first_token_gripper_probability"
                        ] = float(
                            full_action_seq[0, 0, 6].detach().float().item()
                        )
                        latent_metrics[
                            "filter_raw_first_token_gripper_probability"
                        ] = float(
                            filter_action_seq[0, 0, 6].detach().float().item()
                        )
                        prior_probability_source = (
                            prior_diagnostics["gripper_probability"]
                            if prior_diagnostics is not None
                            else prior_action_seq
                        )
                        if prior_probability_source is not None:
                            latent_metrics[
                                "prior_raw_first_token_gripper_probability"
                            ] = float(
                                prior_probability_source[0, 0, -1]
                                .detach()
                                .float()
                                .item()
                            )
                        if full_filter_diagnostics is not None:
                            latent_metrics[
                                "full_raw_first_token_gripper_logit"
                            ] = float(
                                full_filter_diagnostics["gripper_logit"][
                                    0, 0, 0
                                ]
                                .detach()
                                .float()
                                .item()
                            )
                        if filter_diagnostics is not None:
                            latent_metrics[
                                "filter_raw_first_token_gripper_logit"
                            ] = float(
                                filter_diagnostics["gripper_logit"][0, 0, 0]
                                .detach()
                                .float()
                                .item()
                            )
                        if prior_diagnostics is not None:
                            latent_metrics[
                                "prior_raw_first_token_gripper_logit"
                            ] = float(
                                prior_diagnostics["gripper_logit"][0, 0, 0]
                                .detach()
                                .float()
                                .item()
                            )
                    self._sync_cuda()
                    rng_after = capture_rng_state(include_cuda=True)
                    rng_preserved = rng_states_equal(rng_before, rng_after)
                    filter_diagnostic_ms = (
                        time.perf_counter() - diagnostic_t0
                    ) * 1000.0
                    self.every_step_filter_diagnostic_latency_sum += (
                        filter_diagnostic_ms / 1000.0
                    )
                    self.every_step_filter_diagnostic_action_head_calls += (
                        diagnostic_head_calls
                    )
                    self.every_step_filter_rng_checks += 1
                    self.every_step_filter_rng_failures += int(not rng_preserved)
                    step_record.update(latent_metrics)
                    step_record.update(
                        {
                            "filter_diagnostic_action_head_calls": diagnostic_head_calls,
                            "filter_diagnostic_ms": filter_diagnostic_ms,
                            "filter_diagnostics_rng_preserved": int(rng_preserved),
                            "executed_latent_source": self.lrnode_every_step_filter_mode,
                            "full_forward_role": (
                                "executed_candidate"
                                if (
                                    filter_selection.initialized_from_full
                                    or self.lrnode_every_step_filter_mode == "raw_full"
                                    or (
                                        self.lrnode_every_step_filter_mode == "fixed_filter"
                                        and self.lrnode_every_step_filter_alpha > 0.0
                                    )
                                    or (
                                        self.lrnode_every_step_filter_mode == "full_latent_ema"
                                        and self.lrnode_every_step_filter_beta > 0.0
                                    )
                                )
                                else "diagnostic_only"
                            ),
                        }
                    )
                    arm_source = self.lrnode_every_step_filter_mode
                    gripper_source = self.lrnode_every_step_filter_mode
                    lrnode_debug.update(
                        {
                            "z_prior": z_prior,
                            "z_filter": z_filter.detach(),
                            "executed_latent": z_filter.detach(),
                            "action_filter": filter_action_seq,
                            "action_full": full_action_seq,
                        }
                    )
                    if prior_action_seq is not None:
                        lrnode_debug["action_prior"] = prior_action_seq.detach()
                    if self.lrnode_every_step_filter_diagnostics:
                        if prior_diagnostics is not None:
                            prior_action_seq = torch.cat(
                                [
                                    prior_diagnostics["arm"],
                                    prior_diagnostics["gripper_probability"],
                                ],
                                dim=-1,
                            ).detach()
                            lrnode_debug["action_prior"] = prior_action_seq
                        if filter_diagnostics is not None:
                            lrnode_debug[
                                "counterfactual_gripper_logit"
                            ] = filter_diagnostics["gripper_logit"].detach()
                            lrnode_debug[
                                "counterfactual_gripper_probability"
                            ] = filter_diagnostics[
                                "gripper_probability"
                            ].detach()
                        if full_filter_diagnostics is not None:
                            lrnode_debug[
                                "full_gripper_logit"
                            ] = full_filter_diagnostics[
                                "gripper_logit"
                            ].detach()
                    step_record["arm_source"] = arm_source
                    step_record["gripper_source"] = gripper_source
                if diagnostic_refresh_debug is not None and cache_latent is not None:
                    z_lr_candidate = diagnostic_refresh_debug["z_pred"].detach()
                    if self.lrnode_counterfactual_mode in {
                        "full_arm_full_gripper",
                        "lr_arm_lr_gripper",
                        "lr_arm_full_gripper",
                        "full_arm_lr_gripper",
                    }:
                        action_seq, arm_source, gripper_source = mix_action_tokens(
                            diagnostic_refresh_debug["action_pred"],
                            shadow["action"],
                            self.lrnode_counterfactual_mode,
                        )
                        if arm_source == "lr":
                            self.counterfactual_arm_lr_steps += 1
                        else:
                            self.counterfactual_arm_full_steps += 1
                        if gripper_source == "lr":
                            self.counterfactual_gripper_lr_steps += 1
                        else:
                            self.counterfactual_gripper_full_steps += 1
                    elif self.lrnode_counterfactual_mode == "latent_fusion":
                        cache_latent = fuse_latents(
                            z_lr_candidate,
                            cache_latent,
                            self.lrnode_latent_fusion_alpha,
                        )
                        action_seq, fused_diagnostics = self._decode_diagnostic_latent(
                            cache_latent
                        )
                        diagnostic_refresh_debug[
                            "counterfactual_gripper_logit"
                        ] = fused_diagnostics["gripper_logit"].detach()
                        diagnostic_refresh_debug[
                            "counterfactual_gripper_probability"
                        ] = fused_diagnostics["gripper_probability"].detach()
                        self.counterfactual_latent_fusion_steps += 1
                        arm_source = "fusion"
                        gripper_source = "fusion"
                    elif self.lrnode_counterfactual_mode == "matched_random":
                        seed = deterministic_step_seed(
                            self.lrnode_matched_random_seed,
                            self.current_task_id,
                            self.current_episode_id,
                            timestep,
                        )
                        cache_latent, learned_delta, random_delta = matched_random_latent(
                            z_lr_candidate,
                            cache_latent,
                            seed=seed,
                            norm_mode=self.lrnode_matched_random_norm_mode,
                        )
                        action_seq, random_diagnostics = self._decode_diagnostic_latent(
                            cache_latent
                        )
                        diagnostic_refresh_debug.update(
                            {
                                "matched_random_seed": seed,
                                "learned_delta": learned_delta.detach(),
                                "random_delta": random_delta.detach(),
                                "learned_delta_norm": float(
                                    learned_delta.detach().float().norm().item()
                                ),
                                "random_delta_norm": float(
                                    random_delta.detach().float().norm().item()
                                ),
                                "counterfactual_gripper_logit": random_diagnostics[
                                    "gripper_logit"
                                ].detach(),
                                "counterfactual_gripper_probability": random_diagnostics[
                                    "gripper_probability"
                                ].detach(),
                            }
                        )
                        self.counterfactual_matched_random_steps += 1
                        arm_source = "random"
                        gripper_source = "random"
                    diagnostic_refresh_debug["executed_latent"] = cache_latent.detach()
                    lrnode_debug = diagnostic_refresh_debug
                    step_record["arm_source"] = arm_source
                    step_record["gripper_source"] = gripper_source
                    step_record["learned_delta_norm"] = float(
                        diagnostic_refresh_debug.get("learned_delta_norm", 0.0)
                    )
                    step_record["random_delta_norm"] = float(
                        diagnostic_refresh_debug.get("random_delta_norm", 0.0)
                    )
                self._cache_full_forward_state(
                    action_latent,
                    selected_step,
                    image_x,
                    gripper,
                    state,
                    action_tokens=action_seq,
                    action_arm=(
                        None if full_diagnostics is None else full_diagnostics["arm"]
                    ),
                    gripper_logit=(
                        None
                        if full_diagnostics is None
                        else full_diagnostics["gripper_logit"]
                    ),
                    timestep=timestep,
                )
                if self.joint_latent_action_surrogate_mode != "off":
                    step_record.update(
                        {
                            "joint_anchor_replaced_by_exact": int(
                                self.joint_latent_action_surrogate_mode == "joint"
                            ),
                            "joint_action_surrogate_error": 0.0,
                            "joint_executed_token_error": 0.0,
                            "joint_tail_token_error": 0.0,
                            "joint_latent_error": 0.0,
                            "joint_latent_error_full_refresh_age": 0,
                        }
                    )
                if hierarchical_level is not None:
                    full_step_action_head_calls = 1
                    self.hierarchical_level2_calls += 1
                    self.hierarchical_action_head_calls += full_step_action_head_calls
                    step_record.update(
                        {
                            "hierarchical_step_full_seer_calls": 1,
                            "hierarchical_step_action_head_calls": (
                                full_step_action_head_calls
                            ),
                            "hierarchical_step_latent_updater_calls": 0,
                            "hierarchical_step_action_correction_calls": 0,
                            "full_refresh_age": 0,
                            "action_head_regeneration_age": 0,
                            "latent_cache_source": "full_seer",
                            "action_horizon_cache_source": "full_origin",
                            "action_cache_generation": int(
                                self.hierarchical_action_cache_generation
                            ),
                            "action_cache_input_generation": int(
                                self.hierarchical_action_cache_input_generation
                            ),
                            "fully_synthetic_horizons_prevented": int(
                                self.hierarchical_provenance.fully_synthetic_horizons_prevented
                            ),
                        }
                    )
                if self.latentloop_segment_executor is not None:
                    step_record["observation_cache_advanced"] = 1
                    self.observation_cache_advance_calls += 1
                if cache_latent is not None:
                    self.lrnode_cached_latent = cache_latent.detach()

            if shadow is not None:
                shadow_ensemble_t0 = time.perf_counter()
                (
                    shadow_env_action,
                    shadow_executed_probability,
                    shadow_ensemble_candidate_count,
                ) = self._shadow_action_sequence_to_env_action(
                    shadow["action"],
                    timestep,
                )
                shadow_ensemble_ms = (
                    time.perf_counter() - shadow_ensemble_t0
                ) * 1000.0
            if direct_env_action is None:
                (
                    action,
                    executed_probability,
                    ensemble_candidate_count,
                ) = self._action_sequence_to_env_action(action_seq, timestep)
            else:
                action = np.asarray(direct_env_action, dtype=np.float32)
            if step_record.get("mode") == "full":
                self._cache_executed_env_action(action)
            action_float = np.asarray(action, dtype=np.float32)
            if hierarchical_level is not None:
                horizon_provenance, executed_provenance = (
                    self._hierarchical_executed_provenance(
                        action_seq,
                        timestep,
                        ensemble_candidate_count,
                    )
                )
                dominant_index = int(np.argmax(executed_provenance))
                step_record["executed_action_source"] = PROVENANCE_LABELS[
                    dominant_index
                ]
                step_record["token_provenance_weights_json"] = json.dumps(
                    horizon_provenance.tolist(), separators=(",", ":")
                )
                for provenance_index, provenance_name in enumerate(PROVENANCE_LABELS):
                    step_record[
                        f"executed_provenance_{provenance_name}"
                    ] = float(executed_provenance[provenance_index])
                    for token_index in range(self.action_pred_steps):
                        step_record[
                            f"token{token_index}_provenance_{provenance_name}"
                        ] = float(
                            horizon_provenance[token_index, provenance_index]
                        )
                if executed_probability is not None:
                    probability_values = (
                        executed_probability.detach().float().cpu().numpy()[0]
                    )
                    for action_index, value in enumerate(probability_values):
                        step_record[f"action_before_threshold_{action_index}"] = float(
                            value
                        )
                for action_index, value in enumerate(action_float):
                    step_record[f"action_after_threshold_{action_index}"] = float(value)
                self._assert_hierarchical_step(
                    hierarchical_level,
                    step_record,
                    latent_version_before=hierarchical_latent_version_before,
                    action_generation_before=hierarchical_action_generation_before,
                )
            action_delta = np.zeros_like(action_float) if self.last_action is None else action_float - self.last_action
            action_jerk = (
                np.zeros_like(action_float)
                if self.last_action_delta is None
                else action_delta - self.last_action_delta
            )
            transition_type = classify_transition(
                self.previous_step_was_full,
                step_record.get("mode") == "full",
            )
            arm_jerk = action_jerk[:6]
            executed_gripper_probability = (
                float(executed_probability[0, 6].detach().float().item())
                if executed_probability is not None
                else float((action_float[-1] + 1.0) / 2.0)
            )
            executed_gripper_logit = ""
            if gripper_source == "full" and shadow is not None:
                executed_gripper_logit = float(
                    shadow["gripper_logit"][0, 0, 0].detach().float().item()
                )
            elif "counterfactual_gripper_logit" in lrnode_debug:
                executed_gripper_logit = float(
                    lrnode_debug["counterfactual_gripper_logit"][0, 0, 0]
                    .detach()
                    .float()
                    .item()
                )
            elif lrnode_debug.get("action_pred_gripper_logit") is not None:
                executed_gripper_logit = float(
                    lrnode_debug["action_pred_gripper_logit"][0, 0, 0]
                    .detach()
                    .float()
                    .item()
                )
            elif shadow is not None and step_record.get("mode") == "full":
                executed_gripper_logit = float(
                    shadow["gripper_logit"][0, 0, 0].detach().float().item()
                )
            step_record.update(
                {
                    "action_norm": float(np.linalg.norm(action_float)),
                    "action_delta_norm": float(np.linalg.norm(action_delta)),
                    "action_delta_l2": float(np.linalg.norm(action_delta)),
                    "action_jerk": float(np.linalg.norm(arm_jerk)),
                    "action_jerk_l2": float(np.linalg.norm(arm_jerk)),
                    "arm_action_jerk": float(np.linalg.norm(arm_jerk)),
                    "trans_action_jerk": float(np.linalg.norm(action_jerk[:3])),
                    "rot_action_jerk": float(np.linalg.norm(action_jerk[3:6])),
                    "gripper_switch": float(
                        0.0 if self.last_action is None else abs(float(action_float[-1] != self.last_action[-1]))
                    ),
                    "gripper_probability": executed_gripper_probability,
                    "gripper_logit": executed_gripper_logit,
                    "gripper_thresholded": float(action_float[-1]),
                    "ensemble_candidate_count": int(ensemble_candidate_count),
                    "shadow_ensemble_candidate_count": int(
                        shadow_ensemble_candidate_count
                    ),
                    "transition_type": transition_type,
                    "arm_source": step_record.get("arm_source", arm_source),
                    "gripper_source": step_record.get("gripper_source", gripper_source),
                    **{
                        f"action_{index}": float(action_float[index])
                        for index in range(7)
                    },
                }
            )
            self.last_action = action_float.copy()
            self.last_action_delta = action_delta.copy()
            self.previous_step_was_full = step_record.get("mode") == "full"
            policy_wall_ms = (time.perf_counter() - policy_t0) * 1000.0
            is_full_step = step_record.get("mode") == "full"
            requires_skip_shadow = (
                not is_full_step
                and counterfactual_requires_skip_shadow(
                    self.lrnode_counterfactual_mode,
                    self.lrnode_latent_fusion_mode,
                )
            )
            if not requires_skip_shadow:
                logging_only_shadow_ms = (
                    shadow_diagnostic_ms
                    + shadow_ensemble_ms
                    + filter_diagnostic_ms
                    + float(
                        step_record.get("joint_diagnostic_action_head_ms", 0.0)
                    )
                    + joint_latent_diagnostic_ms
                )
                if self.lrnode_counterfactual_mode == "standard":
                    logging_only_shadow_ms += refresh_diagnostic_ms
            else:
                required_shadow_forward_ms = (
                    0.0 if shadow is None else float(shadow.get("latency_ms", 0.0))
                )
                logging_only_shadow_ms = (
                    max(0.0, shadow_diagnostic_ms - required_shadow_forward_ms)
                    + shadow_ensemble_ms
                    + filter_diagnostic_ms
                    + float(
                        step_record.get("joint_diagnostic_action_head_ms", 0.0)
                    )
                    + joint_latent_diagnostic_ms
                )
            executed_policy_ms = max(0.0, policy_wall_ms - logging_only_shadow_ms)
            diagnostic_full_forward_ms = (
                float(step_record.get("full_forward_ms", 0.0))
                if step_record.get("full_forward_role") == "diagnostic_only"
                else 0.0
            )
            causal_executed_policy_ms = max(
                0.0,
                executed_policy_ms - diagnostic_full_forward_ms,
            )
            step_record["policy_wall_ms"] = policy_wall_ms
            step_record["shadow_diagnostic_ms"] = shadow_diagnostic_ms
            step_record["refresh_diagnostic_ms"] = refresh_diagnostic_ms
            step_record["filter_diagnostic_ms"] = filter_diagnostic_ms
            step_record["shadow_ensemble_ms"] = shadow_ensemble_ms
            step_record["logging_only_shadow_ms"] = logging_only_shadow_ms
            step_record["total_policy_ms"] = executed_policy_ms
            step_record["protocol_policy_ms"] = executed_policy_ms
            step_record["diagnostic_full_forward_ms"] = diagnostic_full_forward_ms
            step_record["causal_executed_policy_ms"] = causal_executed_policy_ms
            step_record["total_diagnostic_only_ms"] = (
                logging_only_shadow_ms + diagnostic_full_forward_ms
            )
            self.policy_step_latency_sum += executed_policy_ms
            self.num_policy_steps += 1
            self.current_step_records.append(step_record)
            if self._trace_enabled_for_current_episode():
                trace_scalar = dict(step_record)
                trace_scalar.update(
                    {
                        "full_refresh_flag": int(step_record.get("mode") == "full"),
                        "shadow_action_available": int(shadow is not None),
                        "shadow_gripper_probability": (
                            ""
                            if shadow is None
                            else float(
                                shadow["gripper_probability"][0, 0, 0]
                                .detach()
                                .float()
                                .item()
                            )
                        ),
                        "shadow_gripper_logit": (
                            ""
                            if shadow is None
                            else float(
                                shadow["gripper_logit"][0, 0, 0]
                                .detach()
                                .float()
                                .item()
                            )
                        ),
                    }
                )
                trace_tensors = {
                    "z_previous": previous_cached_latent,
                    "z_lr": lrnode_debug.get(
                        "z_pred",
                        None if shadow is None else shadow["latent"],
                    ),
                    "z_prior": z_prior,
                    "z_full": z_full if z_full is not None else (
                        None if shadow is None else shadow["latent"]
                    ),
                    "z_filter": z_filter,
                    "z_executed": lrnode_debug.get(
                        "executed_latent",
                        self.lrnode_cached_latent,
                    ),
                    "lr_gate": lrnode_debug.get("lrnode_gate"),
                    "lr_update": lrnode_debug.get("lrnode_update"),
                    "u_delta": lrnode_debug.get("u_delta"),
                    "a_lr_raw": lrnode_debug.get(
                        "action_pred",
                        None if shadow is None else shadow["action"],
                    ),
                    "a_prior_raw": prior_action_seq,
                    "a_full_raw": full_action_seq if full_action_seq is not None else (
                        None if shadow is None else shadow["action"]
                    ),
                    "a_filter_raw": filter_action_seq,
                    "a_executed_raw": action_seq,
                    "a_executed_final": action_float,
                    "a_lr_executed": action_float,
                    "a_full_shadow_executed": shadow_env_action,
                    "a_lr_post_ensemble": executed_probability,
                    "a_full_post_ensemble": shadow_executed_probability,
                    "learned_delta": lrnode_debug.get("learned_delta"),
                    "random_delta": lrnode_debug.get("random_delta"),
                }
                if not self.lrnode_trace_save_latents:
                    for key in (
                        "z_previous",
                        "z_lr",
                        "z_prior",
                        "z_full",
                        "z_filter",
                        "z_executed",
                    ):
                        trace_tensors.pop(key, None)
                self.current_trace_scalars.append(trace_scalar)
                self.current_trace_tensors.append(trace_tensors)

            if self.latentloop_plan_trace:
                raw_horizon = lrnode_debug.get("action_pred", action_seq)
                raw_logit = lrnode_debug.get("action_pred_gripper_logit")
                raw_probability = lrnode_debug.get(
                    "action_pred_gripper_probability"
                )
                if raw_horizon is None:
                    raise RuntimeError(
                        "Plan trace requires a raw P-token horizon before ensembling"
                    )
                if raw_probability is None:
                    raw_probability = raw_horizon[..., 6:]
                if raw_logit is None:
                    raise RuntimeError(
                        "Plan trace requires the pre-sigmoid gripper logit"
                    )
                if raw_horizon.shape != (1, self.action_pred_steps, 7):
                    raise RuntimeError(
                        "Plan trace raw horizon must be [1,P,7], got "
                        f"{tuple(raw_horizon.shape)}"
                    )
                if direct_env_action is not None:
                    post_probability = torch.as_tensor(action_float).float()
                    post_probability[-1] = (post_probability[-1] + 1.0) / 2.0
                else:
                    post_probability = executed_probability.detach().float().cpu()[0]
                plan_scalar = {
                    "row_id": self.latentloop_plan_trace_row_id,
                    "paired_group": self.latentloop_plan_trace_paired_group,
                    "task_id": int(self.current_task_id),
                    "episode_id": int(self.current_episode_id),
                    "timestep": int(timestep),
                    "mode": step_record.get("mode", ""),
                    "cache_age": int(step_record.get("cache_age", 0)),
                    "feature_source_step": int(
                        step_record.get("feature_source_step", timestep)
                    ),
                    "full_refresh_flag": int(step_record.get("mode") == "full"),
                    "primary_raw_change_l1": float(
                        step_record.get("primary_raw_change_l1", 0.0)
                    ),
                    "wrist_raw_change_l1": float(
                        step_record.get("wrist_raw_change_l1", 0.0)
                    ),
                    "primary_preprocessed_change_l1": float(
                        step_record.get("image_diff_primary_l1", 0.0)
                    ),
                    "wrist_preprocessed_change_l1": float(
                        step_record.get("image_diff_wrist_l1", 0.0)
                    ),
                    "proprio_delta_l2": float(
                        step_record.get("proprio_delta_l2", 0.0)
                    ),
                    "u_delta_norm": float(step_record.get("u_delta_norm", 0.0)),
                    "feedback_source": step_record.get(
                        "feedback_source", self.latentloop_feedback_source
                    ),
                    "time_shift_initialized_with_zero": int(
                        step_record.get("time_shift_initialized_with_zero", 0)
                    ),
                    "replay_execution_token_idx": step_record.get(
                        "token_idx_used", ""
                    ),
                }
                plan_tensors = {
                    "raw_action_arm": raw_horizon.detach().float().cpu()[0, :, :6],
                    "raw_gripper_logit": raw_logit.detach().float().cpu()[0],
                    "raw_gripper_probability": raw_probability.detach().float().cpu()[0],
                    "raw_gripper_thresholded": (
                        raw_probability.detach().float().cpu()[0] > 0.5
                    ).float(),
                    "post_ensemble_probability": post_probability,
                    "executed_action": torch.as_tensor(action_float).float(),
                    "proprio_delta": torch.as_tensor(
                        np.zeros_like(raw_proprio_for_trace)
                        if self.previous_raw_proprio is None
                        else raw_proprio_for_trace - self.previous_raw_proprio
                    ).float(),
                    "u_delta": lrnode_debug.get("u_delta"),
                    "action_latent": lrnode_debug.get(
                        "executed_latent", lrnode_debug.get("z_pred")
                    ),
                }
                if not self.latentloop_plan_trace_save_latents:
                    plan_tensors.pop("action_latent", None)
                self.current_plan_trace_scalars.append(plan_scalar)
                self.current_plan_trace_tensors.append(plan_tensors)

        if self.latentloop_plan_trace:
            self.previous_raw_primary = raw_primary_for_trace.copy()
            self.previous_raw_wrist = raw_wrist_for_trace.copy()
            self.previous_raw_proprio = raw_proprio_for_trace.copy()
        self.gripper_state = np.array([action[-1]])
        return action


def evaluate_libero_task(task, env, obs, args, model):
    steps = 0
    success = 0
    model.reset()
    model.set_episode_context(task, env)
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
            model.record_env_step_ms(
                (time.perf_counter() - env_t0) * 1000.0,
                observation=obs,
                reward=reward,
                done=done,
                info=info,
            )
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
            num_eval_episodes = int(os.environ.get("EVAL_NUM_EPISODES_PER_TASK", "20"))
            task_num = int(os.environ.get("EVAL_NUM_TASKS", "10"))
            if num_eval_episodes <= 0:
                raise ValueError(f"EVAL_NUM_EPISODES_PER_TASK must be positive, got {num_eval_episodes}")
            if task_num <= 0 or task_num > 10:
                raise ValueError(f"EVAL_NUM_TASKS must be in [1, 10], got {task_num}")

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
            "render_gpu_device_id": _renderer_gpu_device_id(device_id),
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
        verify_renderer_backend(env, env_args["render_gpu_device_id"])
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
        _write_live_eval_progress(
            args=args,
            local_results=results,
            local_eval_ids=eval_sequence_ids[:len(results)],
            total_sequences=NUM_SEQUENCES,
            local_assigned=len(eval_sequence_ids),
            last_eval_id=eval_id,
        )

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
    local_renderer_metadata = get_renderer_backend_metadata()
    all_renderer_metadata = [None for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(local_renderer_metadata, all_renderer_metadata, dst=0)
    all_episode_metrics = [None for _ in range(device_num)] if torch.distributed.get_rank() == 0 else None
    torch.distributed.gather_object(local_episode_metrics, all_episode_metrics, dst=0)

    if torch.distributed.get_rank() == 0:
        res_tup_list = merge_multi_list(all_res_tup)
        res_tup_list.sort(key=lambda x: x[1])
        episode_metrics_list = merge_multi_list(all_episode_metrics)
        print_and_save(res_tup_list, task_suite)
        save_eval_json(
            args,
            res_tup_list,
            task_suite,
            all_lrnode_stats,
            episode_metrics_list,
            all_renderer_metadata,
        )


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
        "fast_encoder_calls": 0,
        "action_head_calls": 0,
        "hold_action_steps": 0,
        "hold_latent_steps": 0,
        "chunk_token_steps": 0,
        "no_delta_steps": 0,
        "latentloop_segment_grid_enabled": 0,
        "segment_length": 1,
        "feedback_schedule": "not_applicable",
        "planned_feedback_density": None,
        "observation_conditioned_update_calls": 0,
        "zero_feature_update_calls": 0,
        "observation_cache_advance_calls": 0,
        "num_fallback_full_calls": 0,
        "full_forward_latency_sum": 0.0,
        "full_action_head_latency_sum": 0.0,
        "full_non_action_head_latency_sum": 0.0,
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
        "counterfactual_arm_lr_steps": 0,
        "counterfactual_arm_full_steps": 0,
        "counterfactual_gripper_lr_steps": 0,
        "counterfactual_gripper_full_steps": 0,
        "counterfactual_latent_fusion_steps": 0,
        "counterfactual_matched_random_steps": 0,
        "counterfactual_mode": "standard",
        "counterfactual_mix_stage": "pre_ensemble",
        "every_step_filter_mode": "off",
        "every_step_filter_alpha": 0.5,
        "every_step_filter_beta": 0.5,
        "every_step_filter_diagnostics": 0,
        "every_step_filter_prior_calls": 0,
        "every_step_filter_fusion_calls": 0,
        "every_step_filter_action_head_calls": 0,
        "every_step_filter_diagnostic_action_head_calls": 0,
        "every_step_filter_prior_latency_sum": 0.0,
        "every_step_filter_fusion_latency_sum": 0.0,
        "every_step_filter_action_head_latency_sum": 0.0,
        "every_step_filter_diagnostic_latency_sum": 0.0,
        "every_step_filter_rng_checks": 0,
        "every_step_filter_rng_failures": 0,
        "query_reduction_claim_allowed": 1,
        "hierarchical_mode": "off",
        "hierarchical_full_interval": 8,
        "hierarchical_regeneration_interval": 3,
        "hierarchical_level0_calls": 0,
        "hierarchical_level1_calls": 0,
        "hierarchical_level2_calls": 0,
        "hierarchical_latent_updater_calls": 0,
        "hierarchical_action_correction_calls": 0,
        "hierarchical_action_head_calls": 0,
        "hierarchical_action_correction_latency_sum": 0.0,
        "joint_latent_action_surrogate_mode": "off",
        "joint_surrogate_calls": 0,
        "joint_diagnostic_action_head_calls": 0,
        "joint_diagnostic_full_forward_calls": 0,
        "joint_diagnostic_full_forward_latency_sum": 0.0,
        "joint_surrogate_latency_sum": 0.0,
        "fully_synthetic_horizons_prevented": 0,
    }
    for item in stats_list:
        if item is None:
            continue
        env_steps = int(item.get("num_env_steps", 0))
        full_calls = int(item.get("full_forward_calls", 0))
        lrnode_calls = int(item.get("lrnode_update_calls", 0))
        fast_encoder_calls = int(item.get("fast_encoder_calls", 0))
        action_head_calls = int(item.get("action_head_calls", 0))
        shadow_calls = int(item.get("shadow_full_forward_calls", 0))
        filter_prior_calls = int(
            item.get("every_step_filter_prior_calls", 0)
        )
        filter_action_head_calls = int(
            item.get("every_step_filter_action_head_calls", 0)
        )
        merged["num_env_steps"] += env_steps
        merged["full_forward_calls"] += full_calls
        merged["lrnode_update_calls"] += lrnode_calls
        merged["fast_encoder_calls"] += fast_encoder_calls
        merged["action_head_calls"] += action_head_calls
        merged["hierarchical_mode"] = item.get(
            "hierarchical_mode", merged["hierarchical_mode"]
        )
        merged["hierarchical_full_interval"] = int(
            item.get(
                "hierarchical_full_interval",
                merged["hierarchical_full_interval"],
            )
        )
        merged["hierarchical_regeneration_interval"] = int(
            item.get(
                "hierarchical_regeneration_interval",
                merged["hierarchical_regeneration_interval"],
            )
        )
        merged["joint_latent_action_surrogate_mode"] = item.get(
            "joint_latent_action_surrogate_mode",
            merged["joint_latent_action_surrogate_mode"],
        )
        joint_surrogate_calls = int(item.get("joint_surrogate_calls", 0))
        merged["joint_surrogate_calls"] += joint_surrogate_calls
        merged["joint_diagnostic_action_head_calls"] += int(
            item.get("joint_diagnostic_action_head_calls", 0)
        )
        joint_diagnostic_full_calls = int(
            item.get("joint_diagnostic_full_forward_calls", 0)
        )
        merged["joint_diagnostic_full_forward_calls"] += (
            joint_diagnostic_full_calls
        )
        merged["joint_diagnostic_full_forward_latency_sum"] += (
            float(item.get("joint_avg_diagnostic_full_forward_latency_sec", 0.0))
            * joint_diagnostic_full_calls
        )
        merged["joint_surrogate_latency_sum"] += (
            float(item.get("joint_avg_surrogate_latency_sec", 0.0))
            * joint_surrogate_calls
        )
        for key in (
            "hierarchical_level0_calls",
            "hierarchical_level1_calls",
            "hierarchical_level2_calls",
            "hierarchical_latent_updater_calls",
            "hierarchical_action_correction_calls",
            "hierarchical_action_head_calls",
            "fully_synthetic_horizons_prevented",
        ):
            merged[key] += int(item.get(key, 0))
        merged["hierarchical_action_correction_latency_sum"] += (
            float(item.get("hierarchical_avg_action_correction_latency_sec", 0.0))
            * int(item.get("hierarchical_action_correction_calls", 0))
        )
        merged["hold_action_steps"] += int(item.get("hold_action_steps", 0))
        merged["hold_latent_steps"] += int(item.get("hold_latent_steps", 0))
        merged["chunk_token_steps"] += int(item.get("chunk_token_steps", 0))
        merged["no_delta_steps"] += int(item.get("no_delta_steps", 0))
        merged["latentloop_segment_grid_enabled"] = max(
            merged["latentloop_segment_grid_enabled"],
            int(item.get("latentloop_segment_grid_enabled", 0)),
        )
        merged["segment_length"] = int(
            item.get("segment_length", merged["segment_length"])
        )
        merged["feedback_schedule"] = item.get(
            "feedback_schedule", merged["feedback_schedule"]
        )
        if item.get("planned_feedback_density") is not None:
            merged["planned_feedback_density"] = float(
                item["planned_feedback_density"]
            )
        merged["observation_conditioned_update_calls"] += int(
            item.get("observation_conditioned_update_calls", 0)
        )
        merged["zero_feature_update_calls"] += int(
            item.get("zero_feature_update_calls", 0)
        )
        merged["observation_cache_advance_calls"] += int(
            item.get("observation_cache_advance_calls", 0)
        )
        merged["every_step_filter_mode"] = item.get(
            "every_step_filter_mode",
            merged["every_step_filter_mode"],
        )
        merged["every_step_filter_alpha"] = float(
            item.get(
                "every_step_filter_alpha",
                merged["every_step_filter_alpha"],
            )
        )
        merged["every_step_filter_beta"] = float(
            item.get(
                "every_step_filter_beta",
                merged["every_step_filter_beta"],
            )
        )
        merged["every_step_filter_diagnostics"] = int(
            item.get("every_step_filter_diagnostics", 0)
        )
        merged["every_step_filter_prior_calls"] += filter_prior_calls
        merged["every_step_filter_fusion_calls"] += int(
            item.get("every_step_filter_fusion_calls", 0)
        )
        merged["every_step_filter_action_head_calls"] += (
            filter_action_head_calls
        )
        merged["every_step_filter_diagnostic_action_head_calls"] += int(
            item.get("every_step_filter_diagnostic_action_head_calls", 0)
        )
        merged["every_step_filter_rng_checks"] += int(
            item.get("every_step_filter_rng_checks", 0)
        )
        merged["every_step_filter_rng_failures"] += int(
            item.get("every_step_filter_rng_failures", 0)
        )
        merged["query_reduction_claim_allowed"] = min(
            merged["query_reduction_claim_allowed"],
            int(item.get("query_reduction_claim_allowed", 1)),
        )
        for key in (
            "counterfactual_arm_lr_steps",
            "counterfactual_arm_full_steps",
            "counterfactual_gripper_lr_steps",
            "counterfactual_gripper_full_steps",
            "counterfactual_latent_fusion_steps",
            "counterfactual_matched_random_steps",
        ):
            merged[key] += int(item.get(key, 0))
        merged["counterfactual_mode"] = item.get(
            "counterfactual_mode",
            merged["counterfactual_mode"],
        )
        merged["counterfactual_mix_stage"] = item.get(
            "counterfactual_mix_stage",
            merged["counterfactual_mix_stage"],
        )
        merged["num_fallback_full_calls"] += int(item.get("num_fallback_full_calls", 0))
        merged["full_forward_latency_sum"] += float(item.get("avg_full_forward_latency_sec", 0.0)) * full_calls
        merged["full_action_head_latency_sum"] += (
            float(item.get("avg_full_action_head_latency_sec", 0.0)) * full_calls
        )
        merged["full_non_action_head_latency_sum"] += (
            float(item.get("avg_full_non_action_head_latency_sec", 0.0)) * full_calls
        )
        merged["lrnode_latency_sum"] += float(item.get("avg_lrnode_latency_sec", 0.0)) * lrnode_calls
        merged["fast_encoder_latency_sum"] += float(item.get("avg_fast_encoder_latency_sec", 0.0)) * fast_encoder_calls
        merged["node_update_latency_sum"] += float(item.get("avg_node_update_latency_sec", 0.0)) * lrnode_calls
        merged["action_head_latency_sum"] += float(item.get("avg_action_head_latency_sec", 0.0)) * action_head_calls
        merged["policy_step_latency_sum"] += float(item.get("avg_policy_step_latency_sec", 0.0)) * env_steps
        merged["env_step_latency_sum"] += float(item.get("avg_env_step_latency_sec", 0.0)) * env_steps
        merged["every_step_filter_prior_latency_sum"] += (
            float(item.get("avg_every_step_filter_prior_latency_sec", 0.0))
            * filter_prior_calls
        )
        merged["every_step_filter_fusion_latency_sum"] += (
            float(item.get("avg_every_step_filter_fusion_latency_sec", 0.0))
            * env_steps
        )
        merged["every_step_filter_action_head_latency_sum"] += (
            float(
                item.get(
                    "avg_every_step_filter_action_head_latency_sec",
                    0.0,
                )
            )
            * filter_action_head_calls
        )
        merged["every_step_filter_diagnostic_latency_sum"] += (
            float(
                item.get(
                    "avg_every_step_filter_diagnostic_latency_sec",
                    0.0,
                )
            )
            * env_steps
        )
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
    # action_head_calls is the legacy skip-path counter. Every full Seer call
    # also executes the shared action head once.
    merged["skip_action_head_calls"] = merged["action_head_calls"]
    merged["total_action_head_calls"] = (
        merged["full_forward_calls"]
        + merged["skip_action_head_calls"]
        + merged["every_step_filter_action_head_calls"]
    )
    merged["avg_full_forward_latency_sec"] = (
        merged["full_forward_latency_sum"] / merged["full_forward_calls"]
        if merged["full_forward_calls"]
        else 0.0
    )
    merged["avg_full_action_head_latency_sec"] = (
        merged["full_action_head_latency_sum"] / merged["full_forward_calls"]
        if merged["full_forward_calls"]
        else 0.0
    )
    merged["avg_full_non_action_head_latency_sec"] = (
        merged["full_non_action_head_latency_sum"] / merged["full_forward_calls"]
        if merged["full_forward_calls"]
        else 0.0
    )
    merged["avg_lrnode_latency_sec"] = (
        merged["lrnode_latency_sum"] / merged["lrnode_update_calls"]
        if merged["lrnode_update_calls"]
        else 0.0
    )
    merged["avg_fast_encoder_latency_sec"] = (
        merged["fast_encoder_latency_sum"] / merged["fast_encoder_calls"]
        if merged["fast_encoder_calls"]
        else 0.0
    )
    merged["avg_node_update_latency_sec"] = (
        merged["node_update_latency_sum"] / merged["lrnode_update_calls"]
        if merged["lrnode_update_calls"]
        else 0.0
    )
    merged["avg_action_head_latency_sec"] = (
        merged["action_head_latency_sum"] / merged["action_head_calls"]
        if merged["action_head_calls"]
        else 0.0
    )
    merged["joint_avg_surrogate_latency_sec"] = (
        merged["joint_surrogate_latency_sum"] / merged["joint_surrogate_calls"]
        if merged["joint_surrogate_calls"]
        else 0.0
    )
    merged["joint_avg_diagnostic_full_forward_latency_sec"] = (
        merged["joint_diagnostic_full_forward_latency_sum"]
        / merged["joint_diagnostic_full_forward_calls"]
        if merged["joint_diagnostic_full_forward_calls"]
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
    merged["effective_query_reduction"] = (
        merged["lrnode_update_calls"] / total_calls
        if total_calls and merged["query_reduction_claim_allowed"]
        else 0.0
    )
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
    merged["full_forward_calls_per_policy_step"] = (
        merged["full_forward_calls"] / merged["num_env_steps"]
        if merged["num_env_steps"] else 0.0
    )
    feedback_call_count = (
        merged["observation_conditioned_update_calls"]
        + merged["zero_feature_update_calls"]
    )
    merged["actual_feedback_density"] = (
        merged["observation_conditioned_update_calls"] / feedback_call_count
        if feedback_call_count else None
    )
    merged["avg_every_step_filter_prior_latency_sec"] = (
        merged["every_step_filter_prior_latency_sum"]
        / merged["every_step_filter_prior_calls"]
        if merged["every_step_filter_prior_calls"] else 0.0
    )
    merged["avg_every_step_filter_fusion_latency_sec"] = (
        merged["every_step_filter_fusion_latency_sum"]
        / merged["num_env_steps"]
        if merged["num_env_steps"] else 0.0
    )
    merged["avg_every_step_filter_action_head_latency_sec"] = (
        merged["every_step_filter_action_head_latency_sum"]
        / merged["every_step_filter_action_head_calls"]
        if merged["every_step_filter_action_head_calls"] else 0.0
    )
    merged["avg_every_step_filter_diagnostic_latency_sec"] = (
        merged["every_step_filter_diagnostic_latency_sum"]
        / merged["num_env_steps"]
        if merged["num_env_steps"] else 0.0
    )
    merged["hierarchical_avg_action_correction_latency_sec"] = (
        merged["hierarchical_action_correction_latency_sum"]
        / merged["hierarchical_action_correction_calls"]
        if merged["hierarchical_action_correction_calls"]
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
        "full_action_head_ms": "avg_full_action_head_ms",
        "full_non_action_head_ms": "avg_full_non_action_head_ms",
        "fast_delta_encoder_ms": "avg_fast_encoder_ms",
        "node_update_ms": "avg_node_update_ms",
        "skip_action_head_ms": "avg_action_head_ms",
        "joint_surrogate_ms": "avg_joint_surrogate_ms",
        "filter_prior_ms": "avg_filter_prior_ms",
        "filter_fusion_ms": "avg_filter_fusion_ms",
        "filter_action_head_ms": "avg_filter_action_head_ms",
        "filter_diagnostic_ms": "avg_filter_diagnostic_ms",
        "policy_total_ms": "avg_policy_step_ms",
        "protocol_policy_ms": "avg_protocol_policy_ms",
        "causal_executed_policy_ms": "avg_causal_executed_policy_ms",
        "diagnostic_full_forward_ms": "avg_diagnostic_full_forward_ms",
        "diagnostic_only_total_ms": "avg_total_diagnostic_only_ms",
        "env_step_ms": "avg_env_step_ms",
        "e2e_step_ms": "avg_policy_step_ms",
        "action_delta_l2": "avg_action_delta_l2",
        "action_jerk_l2": "avg_action_jerk",
    }
    profile = {}
    for out_key, metric_key in key_map.items():
        profile[out_key] = _profile_values([item.get(metric_key, 0.0) for item in episode_metrics])
    _atomic_write_json(Path(path), profile)


def save_eval_json(
    args,
    result_list,
    task_suite,
    lrnode_stats_list,
    episode_metrics=None,
    renderer_metadata_list=None,
):
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
    renderer_metadata_list = [
        item for item in (renderer_metadata_list or []) if item is not None
    ]
    renderer_metadata_list.sort(key=lambda item: int(item.get("process_rank", -1)))
    renderer_backend = get_renderer_backend_metadata()
    renderer_backend["rank_contexts"] = renderer_metadata_list
    renderer_backend["all_ranks_actual_context_verified"] = bool(
        renderer_metadata_list
        and len(renderer_metadata_list) == torch.distributed.get_world_size()
        and all(item.get("actual_context_verified") for item in renderer_metadata_list)
    )
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
        effective_action_head_hz = (
            control_hz * float(lrnode_stats.get("total_action_head_calls", 0)) / float(num_env_steps)
        )
    else:
        effective_full_query_hz = nominal_full_query_hz
        effective_lrnode_update_hz = nominal_lrnode_update_hz
        effective_action_head_hz = 0.0

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
        "lrnode_eval_ablation_mode": getattr(args, "lrnode_eval_ablation_mode", "stepwise"),
        "lrnode_no_delta_mode": getattr(args, "lrnode_no_delta_mode", "zero"),
        "lrnode_chunk_token_policy": getattr(args, "lrnode_chunk_token_policy", "skip_only"),
        "lrnode_query_interval": query_interval,
        "segment_length": int(lrnode_stats.get("segment_length", query_interval)),
        "feedback_schedule": lrnode_stats.get(
            "feedback_schedule", "not_applicable"
        ),
        "actual_feedback_density": lrnode_stats.get(
            "actual_feedback_density"
        ),
        "control_freq": int(round(control_hz)),
        "num_env_steps": int(lrnode_stats.get("num_env_steps", 0)),
        "num_full_forward_calls": int(lrnode_stats.get("full_forward_calls", 0)),
        "num_lrnode_update_calls": int(lrnode_stats.get("lrnode_update_calls", 0)),
        "num_fast_encoder_calls": int(lrnode_stats.get("fast_encoder_calls", 0)),
        "num_action_head_calls": int(lrnode_stats.get("action_head_calls", 0)),
        "num_skip_action_head_calls": int(lrnode_stats.get("skip_action_head_calls", 0)),
        "num_total_action_head_calls": int(lrnode_stats.get("total_action_head_calls", 0)),
        "num_filter_action_head_calls": int(
            lrnode_stats.get("every_step_filter_action_head_calls", 0)
        ),
        "num_hold_action_steps": int(lrnode_stats.get("hold_action_steps", 0)),
        "num_hold_latent_steps": int(lrnode_stats.get("hold_latent_steps", 0)),
        "num_chunk_token_steps": int(lrnode_stats.get("chunk_token_steps", 0)),
        "num_no_delta_steps": int(lrnode_stats.get("no_delta_steps", 0)),
        "num_observation_conditioned_updater_calls": int(
            lrnode_stats.get("observation_conditioned_update_calls", 0)
        ),
        "num_zero_feature_updater_calls": int(
            lrnode_stats.get("zero_feature_update_calls", 0)
        ),
        "full_query_reduction_ratio": float(lrnode_stats.get("full_query_reduction_ratio", 0.0)),
        "effective_full_query_hz": effective_full_query_hz,
        "effective_lrnode_update_hz": effective_lrnode_update_hz,
        "effective_action_head_hz": effective_action_head_hz,
        "avg_policy_step_latency_ms": float(lrnode_stats.get("avg_policy_step_latency_sec", 0.0)) * 1000.0,
        "avg_full_forward_latency_ms": float(lrnode_stats.get("avg_full_forward_latency_sec", 0.0)) * 1000.0,
        "avg_full_action_head_latency_ms": (
            float(lrnode_stats.get("avg_full_action_head_latency_sec", 0.0)) * 1000.0
        ),
        "avg_full_non_action_head_latency_ms": (
            float(lrnode_stats.get("avg_full_non_action_head_latency_sec", 0.0)) * 1000.0
        ),
        "avg_lrnode_latency_ms": float(lrnode_stats.get("avg_lrnode_latency_sec", 0.0)) * 1000.0,
        "avg_fast_encoder_latency_ms": float(lrnode_stats.get("avg_fast_encoder_latency_sec", 0.0)) * 1000.0,
        "avg_action_head_latency_ms": float(lrnode_stats.get("avg_action_head_latency_sec", 0.0)) * 1000.0,
        "avg_skip_action_head_latency_ms": float(lrnode_stats.get("avg_action_head_latency_sec", 0.0)) * 1000.0,
        "avg_filter_prior_latency_ms": (
            float(
                lrnode_stats.get(
                    "avg_every_step_filter_prior_latency_sec",
                    0.0,
                )
            )
            * 1000.0
        ),
        "avg_filter_fusion_latency_ms": (
            float(
                lrnode_stats.get(
                    "avg_every_step_filter_fusion_latency_sec",
                    0.0,
                )
            )
            * 1000.0
        ),
        "avg_filter_action_head_latency_ms": (
            float(
                lrnode_stats.get(
                    "avg_every_step_filter_action_head_latency_sec",
                    0.0,
                )
            )
            * 1000.0
        ),
        "avg_filter_diagnostic_latency_ms": (
            float(
                lrnode_stats.get(
                    "avg_every_step_filter_diagnostic_latency_sec",
                    0.0,
                )
            )
            * 1000.0
        ),
        "query_reduction_claim_allowed": bool(
            lrnode_stats.get("query_reduction_claim_allowed", 1)
        ),
        "action_delta_l2_mean": float(np.mean([m.get("avg_action_delta_l2", 0.0) for m in episode_metrics]))
        if episode_metrics else 0.0,
        "action_delta_l2_p95": float(np.mean([m.get("p95_action_delta_l2", 0.0) for m in episode_metrics]))
        if episode_metrics else 0.0,
        "action_jerk_l2_mean": float(np.mean([m.get("avg_action_jerk", 0.0) for m in episode_metrics]))
        if episode_metrics else 0.0,
        "action_jerk_l2_p95": float(np.mean([m.get("p95_action_jerk", 0.0) for m in episode_metrics]))
        if episode_metrics else 0.0,
        "gripper_switch_rate": float(np.mean([m.get("gripper_switch_rate", 0.0) for m in episode_metrics]))
        if episode_metrics else 0.0,
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
            "eval_ablation_mode": getattr(args, "lrnode_eval_ablation_mode", "stepwise"),
            "no_delta_mode": getattr(args, "lrnode_no_delta_mode", "zero"),
            "chunk_token_policy": getattr(args, "lrnode_chunk_token_policy", "skip_only"),
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
            "effective_action_head_hz": effective_action_head_hz,
            "profile_full_action_head": bool(getattr(args, "lrnode_eval_profile_full_action_head", 0)),
            "segment_grid": {
                "enabled": bool(
                    getattr(args, "latentloop_segment_grid_enable", 0)
                ),
                "segment_length": int(
                    lrnode_stats.get("segment_length", query_interval)
                ),
                "feedback_schedule": lrnode_stats.get(
                    "feedback_schedule", "not_applicable"
                ),
                "planned_feedback_density": lrnode_stats.get(
                    "planned_feedback_density"
                ),
                "actual_feedback_density": lrnode_stats.get(
                    "actual_feedback_density"
                ),
                "observation_conditioned_updater_calls": int(
                    lrnode_stats.get(
                        "observation_conditioned_update_calls", 0
                    )
                ),
                "zero_feature_updater_calls": int(
                    lrnode_stats.get("zero_feature_update_calls", 0)
                ),
                "observation_cache_advance_calls": int(
                    lrnode_stats.get("observation_cache_advance_calls", 0)
                ),
            },
            "every_step_filter": {
                "mode": getattr(args, "lrnode_every_step_filter_mode", "off"),
                "alpha": float(
                    getattr(args, "lrnode_every_step_filter_alpha", 0.5)
                ),
                "beta": float(
                    getattr(args, "lrnode_every_step_filter_beta", 0.5)
                ),
                "diagnostics": bool(
                    getattr(
                        args,
                        "lrnode_every_step_filter_diagnostics",
                        0,
                    )
                ),
                "full_forward_every_step": (
                    getattr(args, "lrnode_every_step_filter_mode", "off")
                    != "off"
                ),
                "query_reduction_claim_allowed": bool(
                    lrnode_stats.get("query_reduction_claim_allowed", 1)
                ),
            },
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
            "num_fast_encoder_calls": int(lrnode_stats.get("fast_encoder_calls", 0)),
            "num_action_head_calls": int(lrnode_stats.get("action_head_calls", 0)),
            "num_skip_action_head_calls": int(lrnode_stats.get("skip_action_head_calls", 0)),
            "num_total_action_head_calls": int(lrnode_stats.get("total_action_head_calls", 0)),
            "num_hold_action_steps": int(lrnode_stats.get("hold_action_steps", 0)),
            "num_hold_latent_steps": int(lrnode_stats.get("hold_latent_steps", 0)),
            "num_chunk_token_steps": int(lrnode_stats.get("chunk_token_steps", 0)),
            "num_no_delta_steps": int(lrnode_stats.get("no_delta_steps", 0)),
            "num_observation_conditioned_updater_calls": int(
                lrnode_stats.get("observation_conditioned_update_calls", 0)
            ),
            "num_zero_feature_updater_calls": int(
                lrnode_stats.get("zero_feature_update_calls", 0)
            ),
            "num_fallback_full_calls": int(lrnode_stats.get("num_fallback_full_calls", 0)),
            "full_query_reduction_ratio": float(lrnode_stats.get("full_query_reduction_ratio", 0.0)),
            "effective_query_interval": float(lrnode_stats.get("effective_query_interval", 0.0)),
            "query_reduction_claim_allowed": bool(
                lrnode_stats.get("query_reduction_claim_allowed", 1)
            ),
            "full_forward_calls_per_policy_step": float(
                lrnode_stats.get("full_forward_calls_per_policy_step", 0.0)
            ),
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
        "renderer_backend": renderer_backend,
    }
    safe_run_name = args.run_name.replace("/", "_")
    ckpt_tag = os.environ.get("CKPT_TAG", "").strip()
    tag = f"_{ckpt_tag}" if ckpt_tag else ""
    json_path = os.path.join(output_dir, f"{safe_run_name}_{args.finetune_type}{tag}_eval.json")
    _atomic_write_json(Path(json_path), payload)
    summary_path = os.path.join(output_dir, "eval_summary.json")
    _atomic_write_json(Path(summary_path), payload)
    episode_csv_path = os.path.join(output_dir, "eval_episode_metrics.csv")
    _write_episode_metrics_csv(episode_csv_path, episode_metrics)
    latency_profile_path = os.path.join(output_dir, "eval_latency_profile.json")
    _write_latency_profile(latency_profile_path, episode_metrics)

    print(f"[LR-NODE eval] success_rate: {success_rate * 100:.1f}%")
    print(f"[LR-NODE eval] control_hz: {control_hz:.2f}")
    print(f"[LR-NODE eval] ablation_mode: {getattr(args, 'lrnode_eval_ablation_mode', 'stepwise')}")
    print(f"[LR-NODE eval] effective_action_hz: {control_hz:.2f}")
    print(f"[LR-NODE eval] effective_full_query_hz: {effective_full_query_hz:.2f}")
    print(f"[LR-NODE eval] effective_lrnode_update_hz: {effective_lrnode_update_hz:.2f}")
    print(f"[LR-NODE eval] effective_action_head_hz: {effective_action_head_hz:.2f}")
    print(f"[LR-NODE eval] full_forward_calls: {lrnode_stats['full_forward_calls']}")
    print(f"[LR-NODE eval] lrnode_update_calls: {lrnode_stats['lrnode_update_calls']}")
    print(f"[LR-NODE eval] fast_encoder_calls: {lrnode_stats.get('fast_encoder_calls', 0)}")
    print(f"[LR-NODE eval] skip_action_head_calls: {lrnode_stats.get('skip_action_head_calls', 0)}")
    print(f"[LR-NODE eval] total_action_head_calls: {lrnode_stats.get('total_action_head_calls', 0)}")
    print(f"[LR-NODE eval] hold_action_steps: {lrnode_stats.get('hold_action_steps', 0)}")
    print(f"[LR-NODE eval] hold_latent_steps: {lrnode_stats.get('hold_latent_steps', 0)}")
    print(f"[LR-NODE eval] chunk_token_steps: {lrnode_stats.get('chunk_token_steps', 0)}")
    print(f"[LR-NODE eval] no_delta_steps: {lrnode_stats.get('no_delta_steps', 0)}")
    print(f"[LR-NODE eval] avg_full_forward_latency_sec: {lrnode_stats['avg_full_forward_latency_sec']:.6f}")
    print(
        "[LR-NODE eval] avg_full_action_head_latency_sec: "
        f"{lrnode_stats.get('avg_full_action_head_latency_sec', 0.0):.6f}"
    )
    print(
        "[LR-NODE eval] avg_full_non_action_head_latency_sec: "
        f"{lrnode_stats.get('avg_full_non_action_head_latency_sec', 0.0):.6f}"
    )
    print(f"[LR-NODE eval] avg_lrnode_latency_sec: {lrnode_stats['avg_lrnode_latency_sec']:.6f}")
    print(f"[LR-NODE eval] avg_skip_action_head_latency_sec: {lrnode_stats['avg_action_head_latency_sec']:.6f}")
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
        lrnode_eval_profile_full_action_head=args.lrnode_eval_profile_full_action_head,
        lrnode_eval_refresh_policy=args.lrnode_eval_refresh_policy,
        lrnode_eval_max_full_forwards_per_episode=args.lrnode_eval_max_full_forwards_per_episode,
        lrnode_eval_ablation_mode=args.lrnode_eval_ablation_mode,
        lrnode_no_delta_mode=args.lrnode_no_delta_mode,
        lrnode_chunk_token_policy=args.lrnode_chunk_token_policy,
        lrnode_mechanism_trace=args.lrnode_mechanism_trace,
        lrnode_trace_save_latents=args.lrnode_trace_save_latents,
        lrnode_trace_episode_limit=args.lrnode_trace_episode_limit,
        lrnode_trace_output_dir=args.lrnode_trace_output_dir,
        lrnode_counterfactual_mode=args.lrnode_counterfactual_mode,
        lrnode_counterfactual_mix_stage=args.lrnode_counterfactual_mix_stage,
        lrnode_latent_fusion_alpha=args.lrnode_latent_fusion_alpha,
        lrnode_latent_fusion_mode=args.lrnode_latent_fusion_mode,
        lrnode_matched_random_seed=args.lrnode_matched_random_seed,
        lrnode_matched_random_norm_mode=args.lrnode_matched_random_norm_mode,
        lrnode_every_step_filter_mode=args.lrnode_every_step_filter_mode,
        lrnode_every_step_filter_alpha=args.lrnode_every_step_filter_alpha,
        lrnode_every_step_filter_beta=args.lrnode_every_step_filter_beta,
        lrnode_every_step_filter_diagnostics=args.lrnode_every_step_filter_diagnostics,
        latentloop_segment_grid_enable=args.latentloop_segment_grid_enable,
        latentloop_feedback_schedule=args.latentloop_feedback_schedule,
        latentloop_same_input_stochasticity_repeats=(
            args.latentloop_same_input_stochasticity_repeats
        ),
        latentloop_same_input_stochasticity_output=(
            args.latentloop_same_input_stochasticity_output
        ),
        latentloop_plan_trace=args.latentloop_plan_trace,
        latentloop_plan_trace_save_latents=args.latentloop_plan_trace_save_latents,
        latentloop_plan_trace_output_dir=args.latentloop_plan_trace_output_dir,
        latentloop_plan_trace_row_id=args.latentloop_plan_trace_row_id,
        latentloop_plan_trace_paired_group=args.latentloop_plan_trace_paired_group,
        latentloop_feedback_source=args.latentloop_feedback_source,
        latentloop_plan_adapter_mode=args.latentloop_plan_adapter_mode,
        joint_latent_action_surrogate_mode=(
            args.joint_latent_action_surrogate_mode
        ),
        joint_error_trace=args.joint_error_trace,
        joint_error_trace_output_dir=args.joint_error_trace_output_dir,
        joint_force_exact_action_head=args.joint_force_exact_action_head,
        latentloop_hierarchical_mode=args.latentloop_hierarchical_mode,
        latentloop_hierarchical_full_interval=(
            args.latentloop_hierarchical_full_interval
        ),
        latentloop_hierarchical_regeneration_interval=(
            args.latentloop_hierarchical_regeneration_interval
        ),
        latentloop_hierarchical_trace=args.latentloop_hierarchical_trace,
        latentloop_hierarchical_trace_output_dir=(
            args.latentloop_hierarchical_trace_output_dir
        ),
        latentloop_hierarchical_assert_invariants=(
            args.latentloop_hierarchical_assert_invariants
        ),
        evaluation_seed=args.seed,
    )
    evaluate_policy_ddp(args, wrapped_model)
