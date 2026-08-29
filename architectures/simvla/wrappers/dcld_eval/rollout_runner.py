#!/usr/bin/env python3
"""Planning and real-smoke backend for SimVLA DCLD LIBERO eval wrappers."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional runtime dependency
    imageio = None

ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
LIBERO_EVAL = UPSTREAM / "evaluation" / "libero"
LIBERO_ROOT = LIBERO_EVAL / "LIBERO"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))
if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter, SimVLAConditionAdapter, SimVLADCLDEvalWrapper  # noqa: E402
from architectures.simvla.adapters.dcld.simvla_delta_obs_adapter import raw_rgb_to_tensor  # noqa: E402
from methods.dcld.modules import DCLDCore, DCLDMode, DeltaObservation  # noqa: E402
from .invariants import assert_mode_invariants  # noqa: E402


REQUIRED_COUNTERS = [
    "num_env_steps",
    "num_policy_queries",
    "num_full_vlm_calls",
    "num_dcld_updates",
    "num_fast_encoder_calls",
    "num_action_transformer_calls",
    "num_action_queue_steps",
    "num_hold_action_steps",
    "num_hold_condition_steps",
    "num_native_chunk_steps",
    "num_no_delta_steps",
]


LATENCY_FIELDS = [
    "VLM_encoder_ms",
    "action_transformer_ms",
    "FastEncoder_ms",
    "DCLD_update_ms",
    "policy_total_ms",
    "env_step_ms",
]


class _DryRunModel:
    """Only used to instantiate the wrapper in dry-run mode."""

    pass


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


def command_output(args: list[str], *, cwd: Path = ROOT) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def module_location(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__file__", "")) or None
    except Exception:
        return None


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def tensor_action_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a_cpu = a.detach().cpu().float()
    b_cpu = b.detach().cpu().float()
    if tuple(a_cpu.shape) != tuple(b_cpu.shape):
        return {
            "shape_a": list(a_cpu.shape),
            "shape_b": list(b_cpu.shape),
            "shape_match": False,
            "mean_abs_diff": None,
            "max_abs_diff": None,
            "l2_diff": None,
            "allclose_1e_5": False,
            "allclose_1e_4": False,
            "allclose_1e_3": False,
            "allclose_1e_2": False,
        }
    diff = (a_cpu - b_cpu).abs()
    return {
        "shape_a": list(a_cpu.shape),
        "shape_b": list(b_cpu.shape),
        "shape_match": True,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "l2_diff": float(torch.linalg.vector_norm((a_cpu - b_cpu).reshape(-1)).item()),
        "allclose_1e_5": bool(torch.allclose(a_cpu, b_cpu, atol=1e-5, rtol=1e-5)),
        "allclose_1e_4": bool(torch.allclose(a_cpu, b_cpu, atol=1e-4, rtol=1e-4)),
        "allclose_1e_3": bool(torch.allclose(a_cpu, b_cpu, atol=1e-3, rtol=1e-3)),
        "allclose_1e_2": bool(torch.allclose(a_cpu, b_cpu, atol=1e-2, rtol=1e-2)),
    }


def collect_environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    cuda_device_name = None
    if torch.cuda.is_available():
        try:
            cuda_device_name = torch.cuda.get_device_name(torch.device(args.device))
        except Exception:
            cuda_device_name = torch.cuda.get_device_name(0)
    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        import mujoco as _mujoco
    except Exception:
        _mujoco = None
    try:
        import robosuite as _robosuite
    except Exception:
        _robosuite = None
    try:
        import libero as _libero
    except Exception:
        _libero = None

    return {
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": cuda_device_name,
        "mujoco_version": getattr(_mujoco, "__version__", None) if _mujoco is not None else package_version("mujoco"),
        "mujoco_location": getattr(_mujoco, "__file__", None) if _mujoco is not None else module_location("mujoco"),
        "robosuite_version": getattr(_robosuite, "__version__", None) if _robosuite is not None else package_version("robosuite"),
        "robosuite_location": getattr(_robosuite, "__file__", None) if _robosuite is not None else module_location("robosuite"),
        "libero_version": getattr(_libero, "__version__", None) if _libero is not None else package_version("libero"),
        "libero_location": getattr(_libero, "__file__", None) if _libero is not None else module_location("libero"),
        "transformers_version": package_version("transformers"),
        "accelerate_version": package_version("accelerate"),
        "numpy_version": getattr(_np, "__version__", None) if _np is not None else package_version("numpy"),
        "checkpoint": args.checkpoint,
        "dcld_checkpoint": args.dcld_checkpoint,
        "root_git_commit": command_output(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "root_git_status_short": command_output(["git", "status", "--short"], cwd=ROOT),
        "simvla_upstream_git_commit": command_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], cwd=ROOT),
        "simvla_upstream_git_status_short": command_output(["git", "-C", str(UPSTREAM), "status", "--short"], cwd=ROOT),
        "wrapper_file": str(Path(__file__).resolve()),
        "task_order": args.task_order,
        "replan_steps": args.replan_steps,
        "client_resize_size": args.client_resize_size,
        "flow_steps": args.flow_steps,
        "paired_action_noise": bool(args.paired_action_noise),
        "action_noise_seed_base": int(args.action_noise_seed_base),
    }


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


def parse_csv_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def parse_csv_strings(text: str) -> list[str]:
    return [item for item in text.split(",") if item]


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((quat[:3] * 2.0 * math.acos(float(quat[3]))) / den).astype(np.float32)


def latency_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ["mean", "std", "p50", "p90", "p95", "p99", "min", "max"]}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def l2_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p95": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "p95": float(np.percentile(arr, 95))}


def latency_mean(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics["latency"].get(name, {}).get("mean")
    return None if value is None else float(value)


def latency_p95(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics["latency"].get(name, {}).get("p95")
    return None if value is None else float(value)


def miss_rate(values: list[float], budget_ms: float) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr > budget_ms))


def video_frame_from_obs(obs: dict[str, Any]) -> np.ndarray:
    image0 = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    image1 = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    if image0.shape[:2] != image1.shape[:2]:
        image1 = np.asarray(Image.fromarray(image1.astype(np.uint8)).resize((image0.shape[1], image0.shape[0])))
    return np.concatenate([image0, image1], axis=1).astype(np.uint8)


def safe_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:160]


def save_episode_video(frames: list[np.ndarray], path: Path, fps: int) -> str | None:
    if not frames or imageio is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, fps=int(fps))
    return str(path)


def resize_with_pad_uint8(image: np.ndarray, size: int) -> np.ndarray:
    """Local equivalent of the official client-side resize-with-pad step."""

    if size <= 0:
        return np.ascontiguousarray(image.astype(np.uint8))
    img = Image.fromarray(image.astype(np.uint8))
    old_w, old_h = img.size
    scale = min(float(size) / max(old_w, 1), float(size) / max(old_h, 1))
    new_w = max(1, int(round(old_w * scale)))
    new_h = max(1, int(round(old_h * scale)))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size))
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))


def base_row(
    *,
    protocol: str,
    row_name: str,
    mode: str,
    k: int,
    control_hz: int,
    checkpoint: str,
    dcld_checkpoint: str | None,
    suite: str,
    replan_steps: int = 5,
) -> dict[str, Any]:
    full_vlm_reduction = 0.0 if mode == "full" else max(0.0, 1.0 - 1.0 / max(1, k))
    policy_query_hz = control_hz / max(1, replan_steps)
    dcld_active = mode not in {"full", "hold_action", "hold_condition", "native_action_chunk"}
    if row_name == "baseline_full_k1":
        row_semantics = "wrapper_full_k1_seeded" if mode == "full" else "wrapper_nonofficial_baseline"
    elif row_name == "ours_full_k1":
        row_semantics = "ours_full_k1_seeded_dcld_inactive" if mode == "full" else "ours_nonfull"
    elif dcld_active:
        row_semantics = f"ours_dcld_k{k}_active"
    else:
        row_semantics = f"wrapper_ablation_{mode}_k{k}"
    return {
        "protocol": protocol,
        "row_name": row_name,
        "row_semantics": row_semantics,
        "is_official_upstream_baseline": False,
        "dcld_active": dcld_active,
        "suite": suite,
        "mode": mode,
        "k": k,
        "control_hz": control_hz,
        "replan_steps": replan_steps,
        "checkpoint": checkpoint,
        "dcld_checkpoint": dcld_checkpoint or "",
        "success_rate": "",
        "task_wise_success": "",
        "num_episodes": 0,
        "num_env_steps": 0,
        "full_vlm_call_reduction": full_vlm_reduction,
        "effective_full_vlm_hz": policy_query_hz / max(1, k) if mode != "full" else policy_query_hz,
        "effective_dcld_update_hz": 0 if mode in {"full", "hold_action", "hold_condition", "native_action_chunk"} else policy_query_hz * full_vlm_reduction,
        "effective_action_transformer_hz": 0 if mode in {"hold_action", "native_action_chunk"} else policy_query_hz,
        "avg_policy_latency_ms": "",
        "VLM_encoder_latency_ms": "",
        "action_transformer_latency_ms": "",
        "FastEncoder_latency_ms": "",
        "DCLD_update_latency_ms": "",
        "action_delta_l2_mean": "",
        "action_delta_l2_p95": "",
        "action_jerk_l2_mean": "",
        "action_jerk_l2_p95": "",
        "gripper_switch_rate": "",
        "miss_rate_20hz": "",
        "miss_rate_40hz": "",
        "miss_rate_60hz": "",
        "miss_rate_80hz": "",
        **{name: 0 for name in REQUIRED_COUNTERS},
    }


def planned_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replan_steps = int(getattr(args, "replan_steps", 5))
    if args.protocol == "qred20":
        rows.append(base_row(protocol=args.protocol, row_name="baseline_full_k1", mode="full", k=1, control_hz=20, checkpoint=args.checkpoint, dcld_checkpoint="", suite=args.suite, replan_steps=replan_steps))
        rows.append(base_row(protocol=args.protocol, row_name="ours_full_k1", mode="full", k=1, control_hz=20, checkpoint=args.checkpoint, dcld_checkpoint=args.dcld_checkpoint, suite=args.suite, replan_steps=replan_steps))
        for k in parse_csv_ints(args.k_list):
            if k == 1:
                continue
            rows.append(base_row(protocol=args.protocol, row_name=f"ours_stepwise_dcld_k{k}", mode="stepwise_dcld", k=k, control_hz=20, checkpoint=args.checkpoint, dcld_checkpoint=args.dcld_checkpoint, suite=args.suite, replan_steps=replan_steps))
    elif args.protocol == "k4_causal":
        modes = parse_csv_strings(args.modes)
        for mode in modes:
            row_name = "baseline_k1_full" if mode == "full" else f"ours_k{args.k}_{mode}"
            k = 1 if mode == "full" else int(args.k)
            rows.append(base_row(protocol=args.protocol, row_name=row_name, mode=mode, k=k, control_hz=20, checkpoint=args.checkpoint, dcld_checkpoint=args.dcld_checkpoint, suite=args.suite, replan_steps=replan_steps))
    elif args.protocol == "hzup20q":
        for item in parse_csv_strings(args.rows):
            hz_text, k_text = item.split(":", 1)
            hz = int(hz_text)
            k = int(k_text)
            rows.append(base_row(protocol=args.protocol, row_name=f"{hz}hz_k1_full_upper", mode="full", k=1, control_hz=hz, checkpoint=args.checkpoint, dcld_checkpoint="", suite=args.suite, replan_steps=replan_steps))
            if k > 1:
                rows.append(base_row(protocol=args.protocol, row_name=f"{hz}hz_k{k}_stepwise_dcld", mode="stepwise_dcld", k=k, control_hz=hz, checkpoint=args.checkpoint, dcld_checkpoint=args.dcld_checkpoint, suite=args.suite, replan_steps=replan_steps))
    else:
        raise ValueError(f"Unknown protocol: {args.protocol}")
    return rows


def instantiate_wrappers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    instantiated = []
    for row in rows:
        mode = row["mode"]
        dcld = DCLDCore(latent_dim=960, gate_bias=-4.0)
        wrapper = SimVLADCLDEvalWrapper(_DryRunModel(), dcld, refresh_every=max(1, int(row["k"])), mode=mode, action_steps=10)
        instantiated.append(
            {
                "row_name": row["row_name"],
                "mode": wrapper.mode,
                "refresh_every": wrapper.scheduler.refresh_every,
                "action_steps": wrapper.action_steps,
            }
        )
    return {"instantiated_wrappers": instantiated, "count": len(instantiated)}


def qred20_k1_equivalence_plan(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(row["row_name"]): row for row in rows}
    baseline = by_name.get("baseline_full_k1")
    ours = by_name.get("ours_full_k1")
    stepwise_k1 = by_name.get("ours_stepwise_dcld_k1")
    checks = {
        "baseline_full_k1_exists": baseline is not None,
        "ours_full_k1_exists": ours is not None,
        "no_stepwise_dcld_k1_row": stepwise_k1 is None,
        "baseline_mode_full_k1": bool(baseline and baseline["mode"] == "full" and int(baseline["k"]) == 1),
        "ours_mode_full_k1": bool(ours and ours["mode"] == "full" and int(ours["k"]) == 1),
        "baseline_has_no_dcld_checkpoint": bool(baseline and not baseline.get("dcld_checkpoint")),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "interpretation": (
            "QRED20 k=1 is represented by full-mode rows. In full mode, RealSimVLADCLDPolicy.act "
            "uses the official-style action queue and refreshes the full SimVLA condition whenever the "
            "queue is refilled. Observed counters after eval must show zero DCLD updates for "
            "baseline_full_k1 and ours_full_k1, and full VLM calls must match policy-query count."
        ),
        "baseline_full_k1": baseline,
        "ours_full_k1": ours,
    }


def write_qred20_k1_equivalence_plan(out: Path, rows: list[dict[str, Any]]) -> None:
    if not any(row.get("protocol") == "qred20" for row in rows):
        return
    report = qred20_k1_equivalence_plan(rows)
    write_json(out / "qred20_k1_baseline_equivalence_plan.json", report)
    write_text(
        out / "qred20_k1_baseline_equivalence_plan.md",
        "\n".join(
            [
                "# QRED20 K=1 Baseline Equivalence Plan",
                "",
                f"- status: `{report['status']}`",
                f"- baseline_full_k1_exists: `{report['checks']['baseline_full_k1_exists']}`",
                f"- ours_full_k1_exists: `{report['checks']['ours_full_k1_exists']}`",
                f"- no_stepwise_dcld_k1_row: `{report['checks']['no_stepwise_dcld_k1_row']}`",
                f"- baseline_mode_full_k1: `{report['checks']['baseline_mode_full_k1']}`",
                f"- ours_mode_full_k1: `{report['checks']['ours_mode_full_k1']}`",
                f"- baseline_has_no_dcld_checkpoint: `{report['checks']['baseline_has_no_dcld_checkpoint']}`",
                "",
                report["interpretation"],
            ]
        ),
    )


def write_outputs(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = planned_rows(args)
    instantiation = instantiate_wrappers(rows)
    write_qred20_k1_equivalence_plan(out, rows)

    summary = {
        "status": "dry_run_passed",
        "scope": "wrapper CLI/backend row planning and wrapper instantiation only; no LIBERO benchmark episodes executed",
        "protocol": args.protocol,
        "checkpoint": args.checkpoint,
        "dcld_checkpoint": args.dcld_checkpoint,
        "suite": args.suite,
        "num_rows": len(rows),
        "required_counters": REQUIRED_COUNTERS,
        **instantiation,
    }
    latency_profile = {
        "status": "dry_run_schema",
        "latency_fields": [
            "avg_policy_latency_ms",
            "VLM_encoder_latency_ms",
            "action_transformer_latency_ms",
            "FastEncoder_latency_ms",
            "DCLD_update_latency_ms",
        ],
        "stats": ["mean", "std", "p50", "p90", "p95", "p99", "min", "max"],
    }

    write_json(out / "eval_smoke_summary.json", summary)
    write_json(out / "eval_smoke_latency_profile.json", latency_profile)
    with (out / "eval_smoke_episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_json(out / f"{args.protocol}_planned_rows.json", rows)


@dataclass
class EvalStepOutput:
    action: np.ndarray
    info: dict[str, Any]


@dataclass
class RealEvalMetrics:
    counters: collections.Counter[str] = field(default_factory=collections.Counter)
    latencies: dict[str, list[float]] = field(default_factory=lambda: {name: [] for name in LATENCY_FIELDS})
    action_delta_l2: list[float] = field(default_factory=list)
    action_jerk_l2: list[float] = field(default_factory=list)
    gripper_switches: int = 0
    gripper_steps: int = 0
    last_action: np.ndarray | None = None
    prev_delta: np.ndarray | None = None

    def observe_action(self, action: np.ndarray) -> None:
        if self.last_action is not None:
            delta = action - self.last_action
            self.action_delta_l2.append(float(np.linalg.norm(delta)))
            if self.prev_delta is not None:
                self.action_jerk_l2.append(float(np.linalg.norm(delta - self.prev_delta)))
            self.prev_delta = delta
            if np.sign(action[-1]) != np.sign(self.last_action[-1]):
                self.gripper_switches += 1
            self.gripper_steps += 1
        self.last_action = action.copy()

    def summary(self) -> dict[str, Any]:
        out = {name: latency_stats(values) for name, values in self.latencies.items()}
        action_delta = l2_stats(self.action_delta_l2)
        action_jerk = l2_stats(self.action_jerk_l2)
        return {
            "latency": out,
            "action_delta_l2_mean": action_delta["mean"],
            "action_delta_l2_p95": action_delta["p95"],
            "action_jerk_l2_mean": action_jerk["mean"],
            "action_jerk_l2_p95": action_jerk["p95"],
            "gripper_switch_rate": float(self.gripper_switches / max(self.gripper_steps, 1)),
        }


class RealSimVLADCLDPolicy:
    """Stateful per-episode SimVLA+DCLD policy used by real LIBERO smoke eval."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        dcld_core: DCLDCore | None,
        mode: str,
        refresh_every: int,
        flow_steps: int,
        image_size: int,
        replan_steps: int,
        client_resize_size: int,
        device: torch.device,
        suite: str = "",
        row_name: str = "",
        task_id: int = -1,
        trial_id: int = -1,
        paired_action_noise: bool = False,
        action_noise_seed_base: int = 20260708,
        log_action_chunks: bool = False,
    ) -> None:
        self.model = model
        self.processor = processor
        self.dcld_core = dcld_core
        self.mode = mode
        self.row_name = row_name
        self.suite = suite
        self.task_id = int(task_id)
        self.trial_id = int(trial_id)
        self.refresh_every = max(1, int(refresh_every))
        self.flow_steps = int(flow_steps)
        self.image_size = int(image_size)
        self.replan_steps = max(1, int(replan_steps))
        self.client_resize_size = int(client_resize_size)
        self.device = device
        self.paired_action_noise = bool(paired_action_noise)
        self.action_noise_seed_base = int(action_noise_seed_base)
        self.log_action_chunks = bool(log_action_chunks)
        self.action_adapter = SimVLAActionAdapter(model)
        self.condition_adapter = SimVLAConditionAdapter(model, self.action_adapter)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.metrics = RealEvalMetrics()
        self.reset()

    def reset(self) -> None:
        self.step_index = 0
        self.query_index = 0
        self.cached_condition: torch.Tensor | None = None
        self.cached_raw_rgb: torch.Tensor | None = None
        self.cached_proprio: torch.Tensor | None = None
        self.cached_action_chunk: torch.Tensor | None = None
        self.cached_executed_action: torch.Tensor | None = None
        self.action_queue: collections.deque[tuple[torch.Tensor, str]] = collections.deque()
        self.action_chunk_records: list[dict[str, Any]] = []
        self.metrics = RealEvalMetrics()

    def preprocess(self, image0: np.ndarray, image1: np.ndarray, proprio: np.ndarray, prompt: str) -> dict[str, torch.Tensor]:
        if self.client_resize_size > 0:
            image0 = resize_with_pad_uint8(image0, self.client_resize_size)
            image1 = resize_with_pad_uint8(image1, self.client_resize_size)
        img0_t = self.image_transform(Image.fromarray(image0.astype(np.uint8)))
        img1_t = self.image_transform(Image.fromarray(image1.astype(np.uint8)))
        padding = torch.zeros_like(img0_t)
        image_input = torch.stack([img0_t, img1_t, padding], dim=0).unsqueeze(0).to(self.device)
        image_mask = torch.tensor([[True, True, False]], device=self.device)
        lang = self.processor.encode_language([prompt])
        input_ids = lang["input_ids"].to(self.device)
        proprio_t = torch.as_tensor(proprio, dtype=torch.float32, device=self.device).reshape(1, -1)[:, :8]
        raw_rgb = raw_rgb_to_tensor(np.stack([image0, image1], axis=0), device=self.device)
        return {
            "image_input": image_input,
            "image_mask": image_mask,
            "input_ids": input_ids,
            "proprio": proprio_t,
            "raw_rgb": raw_rgb,
        }

    def _action_noise_seed(self, policy_query_index: int) -> int:
        return stable_seed(
            self.action_noise_seed_base,
            self.suite,
            self.task_id,
            self.trial_id,
            int(policy_query_index),
            self.flow_steps,
            self.action_adapter.num_actions,
            self.action_adapter.dim_action,
        )

    def _paired_initial_noise(self, condition: torch.Tensor, proprio: torch.Tensor, policy_query_index: int) -> tuple[torch.Tensor | None, int | None]:
        if not self.paired_action_noise:
            return None, None
        seed = self._action_noise_seed(policy_query_index)
        generator_device = proprio.device if proprio.device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(seed)
        noise = self.action_adapter.sample_initial_action_noise(
            condition.shape[0],
            device=proprio.device,
            dtype=proprio.dtype,
            generator=generator,
            deterministic=False,
        )
        return noise, seed

    def _decode(self, condition: torch.Tensor, proprio: torch.Tensor, *, policy_query_index: int) -> tuple[torch.Tensor, int | None]:
        t0 = time.perf_counter()
        initial_noise, seed = self._paired_initial_noise(condition, proprio, policy_query_index)
        action_out = self.action_adapter.decode_action_from_condition(
            condition,
            proprio,
            steps=self.flow_steps,
            initial_noise=initial_noise,
            deterministic=False,
            return_debug=True,
        )
        self.metrics.latencies["action_transformer_ms"].append((time.perf_counter() - t0) * 1000.0)
        self.metrics.counters["num_action_transformer_calls"] += int(action_out.debug.get("iterations", 0))
        return action_out.action, seed

    def _full_refresh(self, batch: dict[str, torch.Tensor], *, policy_query_index: int) -> tuple[torch.Tensor, torch.Tensor, int | None]:
        t0 = time.perf_counter()
        condition = self.condition_adapter.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
        self.metrics.latencies["VLM_encoder_ms"].append((time.perf_counter() - t0) * 1000.0)
        self.metrics.counters["num_full_vlm_calls"] += 1
        action_chunk, seed = self._decode(condition, batch["proprio"], policy_query_index=policy_query_index)
        self.cached_condition = condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action_chunk.detach()
        return condition, action_chunk, seed

    def _dcld_condition(self, batch: dict[str, torch.Tensor], dcld_mode: str, age: int) -> torch.Tensor:
        if self.dcld_core is None:
            raise RuntimeError(f"Mode {self.mode} requires a DCLD checkpoint/core")
        if self.cached_condition is None or self.cached_raw_rgb is None or self.cached_proprio is None:
            raise RuntimeError("DCLD update requires cached full-refresh state")
        obs = DeltaObservation(
            key_images=self.cached_raw_rgb,
            cur_images=batch["raw_rgb"],
            key_proprio=self.cached_proprio,
            cur_proprio=batch["proprio"],
            age=age,
            metadata={"mode": dcld_mode},
        )
        if dcld_mode == DCLDMode.NO_DELTA:
            t_fast = None
        else:
            t_fast = time.perf_counter()
        delta_feature = self.dcld_core.encode_delta(obs, latent_ref=self.cached_condition, mode=dcld_mode)
        if t_fast is not None:
            self.metrics.latencies["FastEncoder_ms"].append((time.perf_counter() - t_fast) * 1000.0)
            self.metrics.counters["num_fast_encoder_calls"] += 1
        t_dyn = time.perf_counter()
        dynamics = self.dcld_core.dynamics(self.cached_condition, delta_feature, dt=1.0, age=float(age))
        self.metrics.latencies["DCLD_update_ms"].append((time.perf_counter() - t_dyn) * 1000.0)
        self.metrics.counters["num_dcld_updates"] += 1
        if dcld_mode == DCLDMode.NO_DELTA:
            self.metrics.counters["num_no_delta_steps"] += 1
        self.cached_condition = dynamics.latent.detach()
        return dynamics.latent

    def _refill_action_queue(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self.metrics.counters["num_policy_queries"] += 1
        refresh = self.mode == "full" or self.cached_condition is None or self.query_index % self.refresh_every == 0
        age = self.query_index % self.refresh_every
        policy_query_index = int(self.query_index)
        action_noise_seed = None

        if refresh:
            _, action_chunk, action_noise_seed = self._full_refresh(batch, policy_query_index=policy_query_index)
            refreshed = True
            queue_mode = "full_refresh"
        elif self.mode == "hold_action":
            if self.cached_executed_action is None:
                raise RuntimeError("hold_action requires cached executed action")
            action_chunk = self.cached_executed_action.reshape(1, 1, -1).repeat(1, self.replan_steps, 1)
            refreshed = False
            queue_mode = "hold_action"
        elif self.mode == "native_action_chunk":
            if self.cached_action_chunk is None:
                raise RuntimeError("native_action_chunk requires cached action chunk")
            start = min(age * self.replan_steps, max(0, self.cached_action_chunk.shape[1] - 1))
            end = min(start + self.replan_steps, self.cached_action_chunk.shape[1])
            action_chunk = self.cached_action_chunk[:, start:end]
            if action_chunk.shape[1] < self.replan_steps:
                pad = action_chunk[:, -1:].repeat(1, self.replan_steps - action_chunk.shape[1], 1)
                action_chunk = torch.cat([action_chunk, pad], dim=1)
            refreshed = False
            queue_mode = "native_action_chunk"
        elif self.mode == "hold_condition":
            if self.cached_condition is None:
                raise RuntimeError("hold_condition requires cached condition")
            action_chunk, action_noise_seed = self._decode(self.cached_condition, batch["proprio"], policy_query_index=policy_query_index)
            refreshed = False
            queue_mode = "hold_condition"
        else:
            dcld_mode = DCLDMode.REAL_DELTA if self.mode == "stepwise_dcld" else self.mode
            condition = self._dcld_condition(batch, dcld_mode=dcld_mode, age=age)
            action_chunk, action_noise_seed = self._decode(condition, batch["proprio"], policy_query_index=policy_query_index)
            refreshed = False
            queue_mode = str(dcld_mode)

        if self.log_action_chunks:
            dcld_called = queue_mode not in {"full_refresh", "hold_action", "hold_condition", "native_action_chunk"}
            self.action_chunk_records.append(
                {
                    "suite": self.suite,
                    "task_id": self.task_id,
                    "trial_id": self.trial_id,
                    "episode_step_index": int(self.step_index),
                    "policy_query_index": policy_query_index,
                    "row_name": self.row_name,
                    "mode": self.mode,
                    "k": self.refresh_every,
                    "queue_mode": queue_mode,
                    "refreshed": bool(refreshed),
                    "full_vlm_called": bool(refreshed and queue_mode == "full_refresh"),
                    "dcld_called": bool(dcld_called),
                    "fast_encoder_called": bool(dcld_called and queue_mode != DCLDMode.NO_DELTA),
                    "paired_action_noise": bool(self.paired_action_noise),
                    "action_noise_seed": action_noise_seed,
                    "action_chunk_shape": list(action_chunk.shape),
                    "action_chunk": action_chunk.detach().cpu().float(),
                }
            )

        self.action_queue.clear()
        for action in action_chunk[0, : self.replan_steps]:
            self.action_queue.append((action.detach(), queue_mode))
        self.query_index += 1
        return {"refreshed": refreshed, "age": age, "queue_mode": queue_mode, "action_noise_seed": action_noise_seed}

    def act(self, image0: np.ndarray, image1: np.ndarray, proprio: np.ndarray, prompt: str) -> EvalStepOutput:
        total_t0 = time.perf_counter()
        self.metrics.counters["num_env_steps"] += 1
        refill_info: dict[str, Any] = {"refreshed": False, "age": self.query_index % self.refresh_every, "queue_mode": "queued"}
        if not self.action_queue:
            batch = self.preprocess(image0, image1, proprio, prompt)
            refill_info = self._refill_action_queue(batch)

        queued_action, action_source = self.action_queue.popleft()
        action = queued_action.reshape(1, -1)
        self.metrics.counters["num_action_queue_steps"] += 1
        if action_source == "hold_action":
            self.metrics.counters["num_hold_action_steps"] += 1
        elif action_source == "native_action_chunk":
            self.metrics.counters["num_native_chunk_steps"] += 1
        elif action_source == "hold_condition":
            self.metrics.counters["num_hold_condition_steps"] += 1
        self.cached_executed_action = action.detach().reshape(-1)
        self.metrics.latencies["policy_total_ms"].append((time.perf_counter() - total_t0) * 1000.0)
        action_np = action.detach().cpu().numpy()[0].astype(np.float32)
        self.metrics.observe_action(action_np)
        self.step_index += 1
        return EvalStepOutput(
            action=action_np,
            info={
                "mode": self.mode,
                "refreshed": bool(refill_info["refreshed"]),
                "age": int(refill_info["age"]),
                "queue_mode": str(refill_info["queue_mode"]),
                "replan_steps": self.replan_steps,
                "counters": dict(self.metrics.counters),
            },
        )


def load_dcld_core_for_eval(path: str, *, device: torch.device) -> tuple[DCLDCore, dict[str, Any]]:
    state = torch.load(path, map_location=device, weights_only=False)
    config = dict(state.get("config", {}))
    latent_dim = int(config.get("latent_dim", 960))
    gate_bias = float(config.get("gate_bias", -4.0))
    delta_dim = int(config.get("delta_dim", 512))
    hidden_dim = int(config.get("hidden_dim", 1024))
    dynamics_type = str(config.get("dynamics_type", "dense"))
    rank_dim = int(config.get("rank_dim", 64))
    gate_mode = str(config.get("gate_mode", "dense" if dynamics_type == "dense" else "scalar"))
    use_post_layernorm = bool(config.get("use_post_layernorm", False))
    core = DCLDCore(
        latent_dim=latent_dim,
        delta_dim=delta_dim,
        hidden_dim=hidden_dim,
        dynamics_type=dynamics_type,
        rank_dim=rank_dim,
        gate_mode=gate_mode,
        gate_bias=gate_bias,
        use_post_layernorm=use_post_layernorm,
    ).to(device)
    try:
        core.load_state_dict(state["dcld_state_dict"], strict=True)
    except RuntimeError:
        dummy_latent = torch.zeros(1, 122, latent_dim, device=device)
        dummy_obs = DeltaObservation(
            key_images=torch.zeros(1, 2, 128, 128, 3, device=device),
            cur_images=torch.zeros(1, 2, 128, 128, 3, device=device),
            key_proprio=torch.zeros(1, 8, device=device),
            cur_proprio=torch.zeros(1, 8, device=device),
        )
        with torch.no_grad():
            _ = core.update_latent(dummy_latent, dummy_obs)
        core.load_state_dict(state["dcld_state_dict"], strict=True)
    core.eval()
    return core, {"checkpoint_path": path, "checkpoint_type": state.get("checkpoint_type"), "config": config}


def get_libero_env(task: Any, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def build_env_obs(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image0 = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    image1 = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate(
        [
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]
    ).astype(np.float32)
    return image0, image1, state


def paired_action_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("suite"),
        record.get("task_id"),
        record.get("trial_id"),
        record.get("policy_query_index"),
    )


def write_k1_action_diff_report(out: Path, action_chunk_records: list[dict[str, Any]], episode_rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_records = {
        paired_action_key(record): record
        for record in action_chunk_records
        if record.get("row_name") == "baseline_full_k1"
    }
    ours_records = {
        paired_action_key(record): record
        for record in action_chunk_records
        if record.get("row_name") == "ours_full_k1"
    }
    keys = sorted(set(baseline_records) | set(ours_records), key=lambda item: tuple(str(x) for x in item))
    diff_rows: list[dict[str, Any]] = []
    for key in keys:
        base = baseline_records.get(key)
        ours = ours_records.get(key)
        row: dict[str, Any] = {
            "suite": key[0],
            "task_id": key[1],
            "trial_id": key[2],
            "policy_query_index": key[3],
            "baseline_record_exists": base is not None,
            "ours_record_exists": ours is not None,
        }
        if base is None or ours is None:
            row.update(
                {
                    "shape_match": False,
                    "mean_abs_diff": None,
                    "max_abs_diff": None,
                    "l2_diff": None,
                    "allclose_1e_5": False,
                    "allclose_1e_4": False,
                    "allclose_1e_3": False,
                    "allclose_1e_2": False,
                    "baseline_action_noise_seed": None if base is None else base.get("action_noise_seed"),
                    "ours_action_noise_seed": None if ours is None else ours.get("action_noise_seed"),
                    "seed_match": False,
                    "baseline_full_vlm_called": None if base is None else base.get("full_vlm_called"),
                    "ours_full_vlm_called": None if ours is None else ours.get("full_vlm_called"),
                    "baseline_dcld_called": None if base is None else base.get("dcld_called"),
                    "ours_dcld_called": None if ours is None else ours.get("dcld_called"),
                    "baseline_fast_encoder_called": None if base is None else base.get("fast_encoder_called"),
                    "ours_fast_encoder_called": None if ours is None else ours.get("fast_encoder_called"),
                }
            )
            diff_rows.append(row)
            continue
        diff = tensor_action_diff(base["action_chunk"], ours["action_chunk"])
        row.update(diff)
        row.update(
            {
                "baseline_action_noise_seed": base.get("action_noise_seed"),
                "ours_action_noise_seed": ours.get("action_noise_seed"),
                "seed_match": base.get("action_noise_seed") == ours.get("action_noise_seed"),
                "baseline_episode_step_index": base.get("episode_step_index"),
                "ours_episode_step_index": ours.get("episode_step_index"),
                "baseline_full_vlm_called": base.get("full_vlm_called"),
                "ours_full_vlm_called": ours.get("full_vlm_called"),
                "baseline_dcld_called": base.get("dcld_called"),
                "ours_dcld_called": ours.get("dcld_called"),
                "baseline_fast_encoder_called": base.get("fast_encoder_called"),
                "ours_fast_encoder_called": ours.get("fast_encoder_called"),
                "baseline_queue_mode": base.get("queue_mode"),
                "ours_queue_mode": ours.get("queue_mode"),
            }
        )
        diff_rows.append(row)

    diff_fieldnames = sorted({key for row in diff_rows for key in row.keys()})
    diff_csv = out / "k1_action_diff.csv"
    with diff_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=diff_fieldnames)
        writer.writeheader()
        writer.writerows(diff_rows)

    outcomes: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in episode_rows:
        if row.get("row_name") not in {"baseline_full_k1", "ours_full_k1"}:
            continue
        key = (row.get("suite"), row.get("task_id"), row.get("episode"))
        outcomes.setdefault(key, {})[str(row["row_name"])] = row
    outcome_rows: list[dict[str, Any]] = []
    for key, pair in sorted(outcomes.items(), key=lambda item: tuple(str(x) for x in item[0])):
        base = pair.get("baseline_full_k1")
        ours = pair.get("ours_full_k1")
        outcome_rows.append(
            {
                "suite": key[0],
                "task_id": key[1],
                "trial_id": key[2],
                "baseline_exists": base is not None,
                "ours_exists": ours is not None,
                "baseline_success": None if base is None else bool(base.get("success")),
                "ours_success": None if ours is None else bool(ours.get("success")),
                "success_match": bool(base is not None and ours is not None and bool(base.get("success")) == bool(ours.get("success"))),
                "baseline_policy_steps": None if base is None else int(base.get("policy_steps", 0)),
                "ours_policy_steps": None if ours is None else int(ours.get("policy_steps", 0)),
                "policy_steps_match": bool(base is not None and ours is not None and int(base.get("policy_steps", -1)) == int(ours.get("policy_steps", -2))),
            }
        )
    outcome_fieldnames = sorted({key for row in outcome_rows for key in row.keys()}) if outcome_rows else []
    outcome_csv = out / "k1_episode_outcome_diff.csv"
    with outcome_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=outcome_fieldnames)
        writer.writeheader()
        writer.writerows(outcome_rows)

    comparable_rows = [row for row in diff_rows if row.get("baseline_record_exists") and row.get("ours_record_exists")]
    allclose_1e_5_count = sum(1 for row in comparable_rows if row.get("allclose_1e_5"))
    missing_pairs = len(diff_rows) - len(comparable_rows)
    seed_mismatch_count = sum(1 for row in comparable_rows if not row.get("seed_match"))
    success_match_count = sum(1 for row in outcome_rows if row.get("success_match"))
    summary = {
        "num_action_chunk_records": len(action_chunk_records),
        "num_baseline_records": len(baseline_records),
        "num_ours_records": len(ours_records),
        "num_diff_rows": len(diff_rows),
        "num_comparable_action_chunks": len(comparable_rows),
        "num_missing_action_chunk_pairs": missing_pairs,
        "num_seed_mismatches": seed_mismatch_count,
        "num_allclose_1e_5": allclose_1e_5_count,
        "all_action_chunks_allclose_1e_5": bool(comparable_rows and allclose_1e_5_count == len(comparable_rows) and missing_pairs == 0),
        "max_abs_diff_max": None if not comparable_rows else max(float(row.get("max_abs_diff") or 0.0) for row in comparable_rows),
        "mean_abs_diff_max": None if not comparable_rows else max(float(row.get("mean_abs_diff") or 0.0) for row in comparable_rows),
        "num_episode_pairs": len(outcome_rows),
        "num_success_matches": success_match_count,
        "all_success_outcomes_match": bool(outcome_rows and success_match_count == len(outcome_rows)),
        "action_diff_csv": str(diff_csv),
        "episode_outcome_diff_csv": str(outcome_csv),
    }
    write_json(out / "k1_action_diff_summary.json", summary)
    write_text(
        out / "k1_action_diff_summary.md",
        "\n".join(
            [
                "# K1 Paired Action Diff Summary",
                "",
                f"- action_chunk_records: `{summary['num_action_chunk_records']}`",
                f"- comparable_action_chunks: `{summary['num_comparable_action_chunks']}`",
                f"- missing_action_chunk_pairs: `{summary['num_missing_action_chunk_pairs']}`",
                f"- seed_mismatches: `{summary['num_seed_mismatches']}`",
                f"- all_action_chunks_allclose_1e_5: `{summary['all_action_chunks_allclose_1e_5']}`",
                f"- max_abs_diff_max: `{summary['max_abs_diff_max']}`",
                f"- episode_pairs: `{summary['num_episode_pairs']}`",
                f"- all_success_outcomes_match: `{summary['all_success_outcomes_match']}`",
                f"- action_diff_csv: `{diff_csv}`",
                f"- episode_outcome_diff_csv: `{outcome_csv}`",
            ]
        ),
    )
    return summary


def run_real_eval(args: argparse.Namespace) -> None:
    from libero.libero import benchmark
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tiny_dir = out / "tiny_eval_smoke"
    tiny_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("LIBERO_ROOT", str(LIBERO_ROOT))
    device = torch.device(args.device)
    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device)
    model.eval()
    if args.norm_stats and Path(args.norm_stats).exists():
        model.action_space.load_norm_stats(args.norm_stats)
    for param in model.parameters():
        param.requires_grad_(False)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)
    environment_metadata = collect_environment_metadata(args)
    write_json(out / "environment_metadata.json", environment_metadata)

    dcld_info: dict[str, Any] | None = None
    dcld_template: DCLDCore | None = None
    eval_rows = planned_rows(args)
    write_qred20_k1_equivalence_plan(out, eval_rows)
    if any(str(row["mode"]) != "full" for row in eval_rows):
        if not args.dcld_checkpoint or not Path(args.dcld_checkpoint).exists():
            blocker = out / "BLOCKER_no_dcld_checkpoint_for_eval_smoke.md"
            write_text(blocker, f"# Blocker\n\nDCLD checkpoint not found: `{args.dcld_checkpoint}`")
            raise FileNotFoundError(f"DCLD checkpoint not found: {args.dcld_checkpoint}")
        dcld_template, dcld_info = load_dcld_core_for_eval(args.dcld_checkpoint, device=device)

    write_json(
        tiny_dir / "args_snapshot.json",
        {key: str(value) for key, value in vars(args).items()},
    )
    write_json(
        tiny_dir / "git_snapshot.json",
        {
            "root_status_short": os.popen(f"git -C {ROOT} status --short").read(),
            "simvla_upstream_status_short": os.popen(f"git -C {UPSTREAM} status --short").read(),
        },
    )
    write_text(
        tiny_dir / "command.sh",
        " ".join([sys.executable, __file__, *sys.argv[1:]]),
    )
    if dcld_info is not None:
        write_json(out / "dcld_checkpoint_eval_loading_report.json", dcld_info)
        write_text(
            out / "dcld_checkpoint_eval_loading_report.md",
            "\n".join(
                [
                    "# DCLD Checkpoint Eval Loading Report",
                    "",
                    f"- checkpoint: `{dcld_info['checkpoint_path']}`",
                    f"- checkpoint_type: `{dcld_info.get('checkpoint_type')}`",
                    f"- config: `{dcld_info.get('config')}`",
                    "- status: `loaded`",
                ]
            ),
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    if args.task_order == "official_reverse":
        task_ids = list(range(task_suite.n_tasks - 1, -1, -1))[: args.max_tasks]
    else:
        task_ids = list(range(task_suite.n_tasks))[: args.max_tasks]
    video_root = Path(args.video_dir) if args.video_dir else out / "eval_videos"
    video_config = {
        "enabled": bool(args.save_video),
        "video_root": str(video_root),
        "fps": int(args.video_fps),
        "stride": max(1, int(args.video_stride)),
        "max_per_row": int(args.video_max_per_row),
        "imageio_available": imageio is not None,
    }
    write_json(out / "video_config.json", video_config)
    total_episodes = len(eval_rows) * len(task_ids) * int(args.num_trials)
    completed_episodes = 0
    global_successes = 0
    eval_start_time = time.time()
    progress_path = out / "eval_progress.jsonl"
    live_summary_path = out / "eval_live_summary.json"
    partial_csv_path = out / "eval_partial_episode_metrics.csv"
    progress_path.write_text("", encoding="utf-8")
    if partial_csv_path.exists():
        partial_csv_path.unlink()
    partial_fieldnames: list[str] | None = None
    progress_bar = None
    try:
        from tqdm.auto import tqdm

        progress_bar = tqdm(
            total=total_episodes,
            desc=f"{args.protocol} eval",
            dynamic_ncols=True,
            mininterval=float(args.tqdm_mininterval),
            disable=bool(args.no_tqdm or not sys.stderr.isatty()),
        )
    except Exception as exc:
        print(f"[DCLD EVAL] tqdm is unavailable, using event logs only: {exc}", flush=True)

    def emit(message: str) -> None:
        if progress_bar is not None and not progress_bar.disable:
            progress_bar.write(message)
        else:
            print(message, flush=True)

    def append_episode_row(row: dict[str, Any]) -> None:
        nonlocal partial_fieldnames
        if partial_fieldnames is None:
            partial_fieldnames = sorted(row.keys())
            with partial_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=partial_fieldnames)
                writer.writeheader()
        with partial_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=partial_fieldnames)
            writer.writerow(row)

    def write_live_summary(event: dict[str, Any]) -> None:
        write_json(
            live_summary_path,
            {
                "status": "running",
                "protocol": args.protocol,
                "output": str(out),
                "completed_episodes": completed_episodes,
                "total_episodes": total_episodes,
                "successes_so_far": global_successes,
                "success_rate_so_far": float(global_successes / max(completed_episodes, 1)),
                "elapsed_seconds": float(time.time() - eval_start_time),
                "latest_event": event,
                "progress_jsonl": str(progress_path),
                "partial_episode_metrics_csv": str(partial_csv_path),
            },
        )

    emit(
        "[DCLD EVAL] "
        f"starting protocol={args.protocol} rows={len(eval_rows)} tasks={len(task_ids)} "
        f"trials={args.num_trials} total_episodes={total_episodes} output={out}"
    )
    rows: list[dict[str, Any]] = []
    mode_summaries: dict[str, Any] = {}
    all_latency: dict[str, Any] = {}
    action_chunk_records: list[dict[str, Any]] = []
    action_diff_episode_counts: collections.Counter[str] = collections.Counter()
    video_counts: collections.Counter[str] = collections.Counter()
    video_paths: list[str] = []

    for row_index, row_cfg in enumerate(eval_rows, start=1):
        mode = str(row_cfg["mode"])
        row_name = str(row_cfg["row_name"])
        refresh_every = int(row_cfg["k"])
        row_start_time = time.time()
        mode_rows = []
        mode_counters = collections.Counter()
        mode_latency = {name: [] for name in LATENCY_FIELDS}
        successes = 0
        episodes = 0
        row_start_event = {
            "event": "row_start",
            "row_index": row_index,
            "num_rows": len(eval_rows),
            "row_name": row_name,
            "mode": mode,
            "k": refresh_every,
            "completed_episodes": completed_episodes,
            "total_episodes": total_episodes,
            "elapsed_seconds": float(time.time() - eval_start_time),
        }
        append_jsonl(progress_path, row_start_event)
        write_live_summary(row_start_event)
        emit(f"[DCLD EVAL] row {row_index}/{len(eval_rows)} start: {row_name} mode={mode} k={refresh_every}")
        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, args.resolution, args.seed)
            try:
                for ep in range(args.num_trials):
                    episode_wall_start = time.time()
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    env.reset()
                    obs = env.set_init_state(initial_states[ep % len(initial_states)])
                    should_record_video = bool(args.save_video) and video_counts[row_name] < int(args.video_max_per_row)
                    video_frames: list[np.ndarray] = []
                    video_step_index = 0

                    def maybe_record_frame(current_obs: dict[str, Any]) -> None:
                        nonlocal video_step_index
                        if not should_record_video:
                            return
                        if video_step_index % max(1, int(args.video_stride)) == 0:
                            video_frames.append(video_frame_from_obs(current_obs))
                        video_step_index += 1

                    maybe_record_frame(obs)
                    dcld_core = None
                    if dcld_template is not None:
                        dcld_core, _ = load_dcld_core_for_eval(args.dcld_checkpoint, device=device)
                    should_log_action_chunks = bool(
                        args.log_action_diff_for_k1
                        and row_name in {"baseline_full_k1", "ours_full_k1"}
                        and action_diff_episode_counts[row_name] < max(0, int(args.action_diff_max_episodes))
                    )
                    policy = RealSimVLADCLDPolicy(
                        model=model,
                        processor=processor,
                        dcld_core=dcld_core,
                        mode=mode,
                        refresh_every=refresh_every,
                        flow_steps=args.flow_steps,
                        image_size=args.image_size,
                        replan_steps=args.replan_steps,
                        client_resize_size=args.client_resize_size,
                        device=device,
                        suite=args.suite,
                        row_name=row_name,
                        task_id=task_id,
                        trial_id=ep,
                        paired_action_noise=args.paired_action_noise,
                        action_noise_seed_base=args.action_noise_seed_base,
                        log_action_chunks=should_log_action_chunks,
                    )
                    done = False
                    policy_steps = 0
                    env_steps = 0
                    for _ in range(args.num_wait_steps):
                        t_env = time.perf_counter()
                        obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
                        policy.metrics.latencies["env_step_ms"].append((time.perf_counter() - t_env) * 1000.0)
                        env_steps += 1
                        maybe_record_frame(obs)
                    while policy_steps < args.max_policy_steps:
                        image0, image1, state = build_env_obs(obs)
                        step_out = policy.act(image0, image1, state, task_description)
                        t_env = time.perf_counter()
                        obs, reward, done, info = env.step(step_out.action.tolist())
                        policy.metrics.latencies["env_step_ms"].append((time.perf_counter() - t_env) * 1000.0)
                        env_steps += 1
                        policy_steps += 1
                        maybe_record_frame(obs)
                        if done:
                            successes += 1
                            break
                    if should_log_action_chunks:
                        action_chunk_records.extend(policy.action_chunk_records)
                        action_diff_episode_counts[row_name] += 1
                    episodes += 1
                    for key, value in policy.metrics.counters.items():
                        mode_counters[key] += value
                    for key, values in policy.metrics.latencies.items():
                        mode_latency[key].extend(values)
                    metrics = policy.metrics.summary()
                    latency = metrics["latency"]
                    episode_wall_time = time.time() - episode_wall_start
                    cuda_peak_allocated_mb = None
                    cuda_peak_reserved_mb = None
                    if device.type == "cuda":
                        cuda_peak_allocated_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
                        cuda_peak_reserved_mb = float(torch.cuda.max_memory_reserved(device) / (1024**2))
                    video_path = ""
                    if should_record_video:
                        suffix = "success" if done else "fail"
                        video_name = f"{safe_stem(row_name)}_task{task_id:02d}_ep{ep:02d}_{suffix}.mp4"
                        saved_video = save_episode_video(
                            video_frames,
                            video_root / safe_stem(row_name) / video_name,
                            fps=int(args.video_fps),
                        )
                        if saved_video:
                            video_path = saved_video
                            video_counts[row_name] += 1
                            video_paths.append(saved_video)
                    row = {
                        "mode": mode,
                        "row_name": row_name,
                        "row_semantics": row_cfg.get("row_semantics"),
                        "is_official_upstream_baseline": bool(row_cfg.get("is_official_upstream_baseline", False)),
                        "dcld_active": bool(row_cfg.get("dcld_active", False)),
                        "k": refresh_every,
                        "control_hz": row_cfg.get("control_hz"),
                        "suite": args.suite,
                        "task_id": task_id,
                        "task_description": task_description,
                        "episode": ep,
                        "success": bool(done),
                        "policy_steps": policy_steps,
                        "env_steps": env_steps,
                        "episode_wall_time_seconds": float(episode_wall_time),
                        "cuda_peak_allocated_mb": cuda_peak_allocated_mb,
                        "cuda_peak_reserved_mb": cuda_peak_reserved_mb,
                        **{name: int(policy.metrics.counters.get(name, 0)) for name in REQUIRED_COUNTERS},
                        "full_vlm_calls_per_env_step": float(policy.metrics.counters.get("num_full_vlm_calls", 0) / max(policy.metrics.counters.get("num_env_steps", 0), 1)),
                        "full_vlm_calls_per_policy_query": float(policy.metrics.counters.get("num_full_vlm_calls", 0) / max(policy.metrics.counters.get("num_policy_queries", 0), 1)),
                        "dcld_updates_per_env_step": float(policy.metrics.counters.get("num_dcld_updates", 0) / max(policy.metrics.counters.get("num_env_steps", 0), 1)),
                        "dcld_updates_per_policy_query": float(policy.metrics.counters.get("num_dcld_updates", 0) / max(policy.metrics.counters.get("num_policy_queries", 0), 1)),
                        "actual_full_vlm_call_reduction": float(1.0 - policy.metrics.counters.get("num_full_vlm_calls", 0) / max(policy.metrics.counters.get("num_policy_queries", 0), 1)),
                        "avg_policy_latency_ms": latency_mean(metrics, "policy_total_ms"),
                        "policy_latency_p95_ms": latency_p95(metrics, "policy_total_ms"),
                        "VLM_encoder_latency_ms": latency_mean(metrics, "VLM_encoder_ms"),
                        "VLM_encoder_latency_p95_ms": latency_p95(metrics, "VLM_encoder_ms"),
                        "action_transformer_latency_ms": latency_mean(metrics, "action_transformer_ms"),
                        "action_transformer_latency_p95_ms": latency_p95(metrics, "action_transformer_ms"),
                        "FastEncoder_latency_ms": latency_mean(metrics, "FastEncoder_ms"),
                        "FastEncoder_latency_p95_ms": latency_p95(metrics, "FastEncoder_ms"),
                        "DCLD_update_latency_ms": latency_mean(metrics, "DCLD_update_ms"),
                        "DCLD_update_latency_p95_ms": latency_p95(metrics, "DCLD_update_ms"),
                        "env_step_latency_ms": latency_mean(metrics, "env_step_ms"),
                        "env_step_latency_p95_ms": latency_p95(metrics, "env_step_ms"),
                        "miss_rate_20hz": miss_rate(policy.metrics.latencies["policy_total_ms"], 50.0),
                        "miss_rate_40hz": miss_rate(policy.metrics.latencies["policy_total_ms"], 25.0),
                        "miss_rate_60hz": miss_rate(policy.metrics.latencies["policy_total_ms"], 1000.0 / 60.0),
                        "miss_rate_80hz": miss_rate(policy.metrics.latencies["policy_total_ms"], 12.5),
                        "action_delta_l2_mean": metrics["action_delta_l2_mean"],
                        "action_delta_l2_p95": metrics["action_delta_l2_p95"],
                        "action_jerk_l2_mean": metrics["action_jerk_l2_mean"],
                        "action_jerk_l2_p95": metrics["action_jerk_l2_p95"],
                        "gripper_switch_rate": metrics["gripper_switch_rate"],
                        "video_path": video_path,
                    }
                    mode_rows.append(row)
                    rows.append(row)
                    completed_episodes += 1
                    global_successes += int(bool(done))
                    elapsed_seconds = time.time() - eval_start_time
                    seconds_per_episode = elapsed_seconds / max(1, completed_episodes)
                    remaining_episodes = max(0, total_episodes - completed_episodes)
                    eta_seconds = seconds_per_episode * remaining_episodes
                    append_episode_row(row)
                    episode_event = {
                        "event": "episode_done",
                        "row_index": row_index,
                        "num_rows": len(eval_rows),
                        "row_name": row_name,
                        "mode": mode,
                        "k": refresh_every,
                        "task_id": task_id,
                        "episode": ep,
                        "success": bool(done),
                        "policy_steps": int(policy_steps),
                        "env_steps": int(env_steps),
                        "video_path": video_path,
                        "completed_episodes": completed_episodes,
                        "total_episodes": total_episodes,
                        "row_successes": successes,
                        "row_episodes": episodes,
                        "global_successes": global_successes,
                        "global_success_rate": float(global_successes / max(1, completed_episodes)),
                        "elapsed_seconds": float(elapsed_seconds),
                        "seconds_per_episode": float(seconds_per_episode),
                        "eta_seconds": float(eta_seconds),
                    }
                    append_jsonl(progress_path, episode_event)
                    write_live_summary(episode_event)
                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            row=row_name,
                            task=task_id,
                            ep=ep,
                            succ=int(bool(done)),
                            rate=f"{global_successes / max(1, completed_episodes):.3f}",
                            eta=format_duration(eta_seconds),
                            refresh=False,
                        )
                    elif args.eval_print_interval > 0 and completed_episodes % args.eval_print_interval == 0:
                        emit(
                            "[DCLD EVAL] "
                            f"episode {completed_episodes}/{total_episodes} row={row_name} "
                            f"task={task_id} ep={ep} success={int(bool(done))} "
                            f"rate={global_successes / max(1, completed_episodes):.3f} "
                            f"elapsed={format_duration(elapsed_seconds)} eta={format_duration(eta_seconds)}"
                        )
            finally:
                env.close()
        row_elapsed = time.time() - row_start_time
        row_latency_stats = {key: latency_stats(values) for key, values in mode_latency.items()}
        policy_latency_values = mode_latency["policy_total_ms"]
        mode_env_steps = int(mode_counters.get("num_env_steps", 0))
        mode_policy_queries = int(mode_counters.get("num_policy_queries", 0))
        mode_full_vlm_calls = int(mode_counters.get("num_full_vlm_calls", 0))
        mode_dcld_updates = int(mode_counters.get("num_dcld_updates", 0))
        mode_summaries[row_name] = {
            "mode": mode,
            "k": refresh_every,
            "control_hz": row_cfg.get("control_hz"),
            "success_rate": float(successes / max(episodes, 1)),
            "successes": successes,
            "episodes": episodes,
            "counters": dict(mode_counters),
            "full_vlm_calls_per_env_step": float(mode_full_vlm_calls / max(mode_env_steps, 1)),
            "full_vlm_calls_per_policy_query": float(mode_full_vlm_calls / max(mode_policy_queries, 1)),
            "dcld_updates_per_env_step": float(mode_dcld_updates / max(mode_env_steps, 1)),
            "dcld_updates_per_policy_query": float(mode_dcld_updates / max(mode_policy_queries, 1)),
            "actual_full_vlm_call_reduction": float(1.0 - mode_full_vlm_calls / max(mode_policy_queries, 1)),
            "avg_policy_latency_ms": row_latency_stats["policy_total_ms"]["mean"],
            "policy_latency_p95_ms": row_latency_stats["policy_total_ms"]["p95"],
            "avg_vlm_encoder_latency_ms": row_latency_stats["VLM_encoder_ms"]["mean"],
            "avg_action_transformer_latency_ms": row_latency_stats["action_transformer_ms"]["mean"],
            "avg_fast_encoder_latency_ms": row_latency_stats["FastEncoder_ms"]["mean"],
            "avg_dcld_update_latency_ms": row_latency_stats["DCLD_update_ms"]["mean"],
            "miss_rate_20hz": miss_rate(policy_latency_values, 50.0),
            "miss_rate_40hz": miss_rate(policy_latency_values, 25.0),
            "miss_rate_60hz": miss_rate(policy_latency_values, 1000.0 / 60.0),
            "miss_rate_80hz": miss_rate(policy_latency_values, 12.5),
            "saved_videos": int(video_counts[row_name]),
            "completed": episodes >= max(1, args.max_tasks * args.num_trials),
        }
        assert_mode_invariants(
            row_name=row_name,
            mode=mode,
            refresh_every=refresh_every,
            counters=mode_counters,
        )
        all_latency[row_name] = row_latency_stats
        row_done_event = {
            "event": "row_done",
            "row_index": row_index,
            "num_rows": len(eval_rows),
            "row_name": row_name,
            "mode": mode,
            "k": refresh_every,
            "successes": successes,
            "episodes": episodes,
            "success_rate": float(successes / max(episodes, 1)),
            "elapsed_seconds": float(time.time() - eval_start_time),
            "row_elapsed_seconds": float(row_elapsed),
            "completed_episodes": completed_episodes,
            "total_episodes": total_episodes,
        }
        append_jsonl(progress_path, row_done_event)
        write_live_summary(row_done_event)
        emit(
            "[DCLD EVAL] "
            f"row {row_index}/{len(eval_rows)} done: {row_name} "
            f"success={successes}/{episodes} ({successes / max(episodes, 1):.3f}) "
            f"row_time={format_duration(row_elapsed)}"
        )

    if progress_bar is not None:
        progress_bar.close()

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (tiny_dir / "eval_smoke_episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counters = {mode: summary["counters"] for mode, summary in mode_summaries.items()}
    k1_action_diff_summary = None
    if args.log_action_diff_for_k1:
        k1_action_diff_summary = write_k1_action_diff_report(out, action_chunk_records, rows)
    qred20_observed_equivalence: dict[str, Any] | None = None
    if args.protocol == "qred20":
        baseline_summary = mode_summaries.get("baseline_full_k1", {})
        ours_summary = mode_summaries.get("ours_full_k1", {})
        baseline_counters = baseline_summary.get("counters", {})
        ours_counters = ours_summary.get("counters", {})
        observed_checks = {
            "baseline_num_dcld_updates_zero": int(baseline_counters.get("num_dcld_updates", 0)) == 0,
            "baseline_num_fast_encoder_calls_zero": int(baseline_counters.get("num_fast_encoder_calls", 0)) == 0,
            "ours_num_dcld_updates_zero": int(ours_counters.get("num_dcld_updates", 0)) == 0,
            "ours_num_fast_encoder_calls_zero": int(ours_counters.get("num_fast_encoder_calls", 0)) == 0,
            "baseline_full_vlm_every_policy_query": int(baseline_counters.get("num_full_vlm_calls", -2))
            == int(baseline_counters.get("num_policy_queries", -1)),
            "ours_full_vlm_every_policy_query": int(ours_counters.get("num_full_vlm_calls", -2))
            == int(ours_counters.get("num_policy_queries", -1)),
            "baseline_action_queue_used": int(baseline_counters.get("num_action_queue_steps", -2))
            == int(baseline_counters.get("num_env_steps", -1)),
            "ours_action_queue_used": int(ours_counters.get("num_action_queue_steps", -2))
            == int(ours_counters.get("num_env_steps", -1)),
        }
        qred20_observed_equivalence = {
            "status": "passed" if all(observed_checks.values()) else "failed",
            "checks": observed_checks,
            "baseline_full_k1_counters": baseline_summary.get("counters", {}),
            "ours_full_k1_counters": ours_summary.get("counters", {}),
            "paired_action_noise": bool(args.paired_action_noise),
            "k1_action_diff_summary": k1_action_diff_summary,
            "interpretation": (
                "A passed status verifies that both QRED20 k=1 full-mode rows avoided the DCLD/FastEncoder path "
                "and refreshed the full SimVLA condition at every official-style policy query/action-queue refill. "
                "When paired_action_noise is enabled, k1 action diff logs must also show shared seeds and allclose action chunks."
            ),
        }
        write_json(out / "qred20_k1_baseline_equivalence_observed.json", qred20_observed_equivalence)
        write_text(
            out / "qred20_k1_baseline_equivalence_observed.md",
            "\n".join(
                [
                    "# QRED20 K=1 Baseline Equivalence Observed Check",
                    "",
                    f"- status: `{qred20_observed_equivalence['status']}`",
                    f"- baseline_num_dcld_updates_zero: `{observed_checks['baseline_num_dcld_updates_zero']}`",
                    f"- baseline_num_fast_encoder_calls_zero: `{observed_checks['baseline_num_fast_encoder_calls_zero']}`",
                    f"- ours_num_dcld_updates_zero: `{observed_checks['ours_num_dcld_updates_zero']}`",
                    f"- ours_num_fast_encoder_calls_zero: `{observed_checks['ours_num_fast_encoder_calls_zero']}`",
                    f"- baseline_full_vlm_every_policy_query: `{observed_checks['baseline_full_vlm_every_policy_query']}`",
                    f"- ours_full_vlm_every_policy_query: `{observed_checks['ours_full_vlm_every_policy_query']}`",
                    f"- baseline_action_queue_used: `{observed_checks['baseline_action_queue_used']}`",
                    f"- ours_action_queue_used: `{observed_checks['ours_action_queue_used']}`",
                    "",
                    qred20_observed_equivalence["interpretation"],
                ]
            ),
        )
    summary = {
        "status": "passed" if all(item["completed"] for item in mode_summaries.values()) else "failed",
        "scope": "tiny real LIBERO rollout smoke; not full benchmark",
        "protocol": args.protocol,
        "suite": args.suite,
        "num_trials": args.num_trials,
        "max_tasks": args.max_tasks,
        "max_policy_steps": args.max_policy_steps,
        "replan_steps": args.replan_steps,
        "task_order": args.task_order,
        "client_resize_size": args.client_resize_size,
        "paired_action_noise": bool(args.paired_action_noise),
        "action_noise_seed_base": int(args.action_noise_seed_base),
        "log_action_diff_for_k1": bool(args.log_action_diff_for_k1),
        "rows": [
            {
                "row_name": row["row_name"],
                "row_semantics": row.get("row_semantics"),
                "is_official_upstream_baseline": row.get("is_official_upstream_baseline"),
                "dcld_active": row.get("dcld_active"),
                "mode": row["mode"],
                "k": row["k"],
                "control_hz": row["control_hz"],
            }
            for row in eval_rows
        ],
        "modes": sorted({str(row["mode"]) for row in eval_rows}),
        "mode_summaries": mode_summaries,
        "completed_episodes": completed_episodes,
        "total_episodes": total_episodes,
        "successes": global_successes,
        "success_rate": float(global_successes / max(completed_episodes, 1)),
        "elapsed_seconds": float(time.time() - eval_start_time),
        "progress_jsonl": str(progress_path),
        "partial_episode_metrics_csv": str(partial_csv_path),
        "video": {
            **video_config,
            "saved_video_count": int(sum(video_counts.values())),
            "saved_videos_by_row": dict(video_counts),
            "video_paths": video_paths,
        },
        "environment_metadata": environment_metadata,
        "k1_action_diff_summary": k1_action_diff_summary,
        "qred20_k1_baseline_equivalence_observed": qred20_observed_equivalence,
    }
    write_json(tiny_dir / "eval_smoke_summary.json", summary)
    write_json(tiny_dir / "eval_smoke_latency_profile.json", all_latency)
    write_json(tiny_dir / "eval_smoke_counters.json", counters)
    # Also mirror at wrapper output root for wrapper-level callers.
    write_json(out / "eval_smoke_summary.json", summary)
    write_json(out / "eval_smoke_latency_profile.json", all_latency)
    write_json(out / "eval_smoke_counters.json", counters)
    with (out / "eval_smoke_episode_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, choices=["qred20", "k4_causal", "hzup20q"])
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--dcld-checkpoint", default="")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--output", required=True)
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--k-list", default="1,2,3,4")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--modes", default="full,stepwise_dcld,hold_action,hold_condition,native_action_chunk,no_delta,shuffled_delta,proprio_only,image_only")
    parser.add_argument("--rows", default="20:1,40:2,60:3,80:4")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--max-policy-steps", type=int, default=2)
    parser.add_argument("--num-wait-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--paired-action-noise", action="store_true")
    parser.add_argument("--action-noise-seed-base", type=int, default=20260708)
    parser.add_argument("--log-action-diff-for-k1", action="store_true")
    parser.add_argument("--action-diff-max-episodes", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--client-resize-size", type=int, default=224)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-order", choices=["official_reverse", "ascending"], default="official_reverse")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-tqdm", action="store_true")
    parser.add_argument("--tqdm-mininterval", type=float, default=1.0)
    parser.add_argument("--eval-print-interval", type=int, default=1)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", default="")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-max-per-row", type=int, default=2)
    args = parser.parse_args()
    if args.run:
        run_real_eval(args)
    else:
        write_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
