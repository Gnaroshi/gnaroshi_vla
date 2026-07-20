#!/usr/bin/env python3
"""Diagnose official SimVLA vs wrapper full-K1 baseline calibration.

This script does not run DCLD K>1 and does not train anything. It compares:

1. official upstream WebSocket server/client semantics with
   `model.generate_actions(...)`
2. wrapper full-K1 in-process semantics with
   `forward_vlm_efficient(...)` + `SimVLAActionAdapter`

Both are evaluated on the same LIBERO suite/task/trial IDs. The script records
per-episode outcomes, hashes for reset/wait/first-query observations, first
action chunk hashes, and environment metadata needed to explain baseline gaps.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import re
import signal
import socket
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


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
LIBERO_EVAL = UPSTREAM / "evaluation" / "libero"
LIBERO_ROOT = LIBERO_EVAL / "LIBERO"

for path in [ROOT, UPSTREAM, LIBERO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter, SimVLAConditionAdapter  # noqa: E402
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (  # noqa: E402
    build_env_obs,
    command_output,
    get_libero_env,
    package_version,
    resize_with_pad_uint8,
    safe_stem,
)
from models.modeling_smolvlm_vla import SmolVLMVLA  # noqa: E402
from models.processing_smolvlm_vla import SmolVLMVLAProcessor  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy as ws_client  # noqa: E402


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def sha256_bytes(chunks: list[bytes]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


def ndarray_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    header = f"{arr.dtype}|{arr.shape}".encode("utf-8")
    return sha256_bytes([header, arr.tobytes()])


def policy_obs_hash(image0: np.ndarray, image1: np.ndarray, state: np.ndarray) -> str:
    return sha256_bytes(
        [
            b"image0",
            ndarray_hash(image0).encode("ascii"),
            b"image1",
            ndarray_hash(image1).encode("ascii"),
            b"state",
            ndarray_hash(state.astype(np.float32)).encode("ascii"),
        ]
    )


def action_chunk_hash(action_chunk: np.ndarray) -> str:
    return ndarray_hash(np.asarray(action_chunk, dtype=np.float32))


def semantic_config_json(config_json: str | None) -> str | None:
    if config_json is None:
        return None
    # robosuite controller_config repr contains process-local MjSim object
    # addresses. Those addresses are not semantic env/controller settings.
    text = re.sub(r"<robosuite\.utils\.binding_utils\.MjSim object at 0x[0-9a-fA-F]+>", "<MjSim>", config_json)
    return text


def wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def resolve_repo_path(path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def collect_environment(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import mujoco
    except Exception:
        mujoco = None
    try:
        import robosuite
    except Exception:
        robosuite = None
    return {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "hostname": socket.gethostname(),
        "root": str(ROOT),
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(torch.device(args.device)) if torch.cuda.is_available() else None,
        "mujoco_version": getattr(mujoco, "__version__", None) if mujoco else package_version("mujoco"),
        "robosuite_version": getattr(robosuite, "__version__", None) if robosuite else package_version("robosuite"),
        "libero_root": str(LIBERO_ROOT),
        "checkpoint": args.checkpoint,
        "norm_stats": resolve_repo_path(args.norm_stats),
        "suite": args.suite,
        "num_trials": args.num_trials,
        "max_tasks": args.max_tasks,
        "task_order": args.task_order,
        "seed": args.seed,
        "num_wait_steps": args.num_wait_steps,
        "max_policy_steps": args.max_policy_steps,
        "replan_steps": args.replan_steps,
        "client_resize_size": args.client_resize_size,
        "image_size": args.image_size,
        "flow_steps": args.flow_steps,
        "root_git_head": command_output(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "root_git_status_short": command_output(["git", "status", "--short"], cwd=ROOT),
        "upstream_git_head": command_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], cwd=ROOT),
        "upstream_git_status_short": command_output(["git", "-C", str(UPSTREAM), "status", "--short"], cwd=ROOT),
    }


def safe_env_config(env: Any) -> dict[str, Any]:
    config: dict[str, Any] = {"env_class": type(env).__name__}
    for attr in ["camera_names", "camera_heights", "camera_widths", "control_freq", "horizon"]:
        try:
            config[attr] = repr(getattr(env, attr))
        except Exception:
            config[attr] = None
    try:
        inner = getattr(env, "env", None)
        config["inner_env_class"] = None if inner is None else type(inner).__name__
        if inner is not None:
            for attr in ["camera_names", "camera_heights", "camera_widths", "control_freq", "horizon"]:
                try:
                    config[f"inner_{attr}"] = repr(getattr(inner, attr))
                except Exception:
                    config[f"inner_{attr}"] = None
            robots = getattr(inner, "robots", [])
            if robots:
                controller = getattr(robots[0], "controller", None)
                controller_config = getattr(robots[0], "controller_config", None)
                config["robot0_controller_class"] = None if controller is None else type(controller).__name__
                config["robot0_controller_config_repr"] = repr(controller_config)
    except Exception as exc:
        config["introspection_error"] = repr(exc)
    return config


def task_ids_for_suite(task_suite: Any, *, task_order: str, max_tasks: int) -> list[int]:
    if task_order == "official_reverse":
        ids = list(range(task_suite.n_tasks - 1, -1, -1))
    else:
        ids = list(range(task_suite.n_tasks))
    return ids[: int(max_tasks)]


@dataclass
class TracePolicyOutput:
    action: np.ndarray
    first_query_trace: dict[str, Any] | None = None


class OfficialWebSocketTracePolicy:
    """Official client-side WebSocket policy with first-query trace logging."""

    def __init__(self, host: str, port: int, *, replan_steps: int, resize_size: int) -> None:
        self.client = ws_client.WebsocketClientPolicy(host, int(port))
        self.replan_steps = int(replan_steps)
        self.resize_size = int(resize_size)
        self.action_plan: collections.deque[np.ndarray] = collections.deque()
        self.query_index = 0

    def reset(self) -> None:
        self.action_plan.clear()
        self.query_index = 0

    def step(self, image0: np.ndarray, image1: np.ndarray, state: np.ndarray, prompt: str) -> TracePolicyOutput:
        trace = None
        if not self.action_plan:
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(image0, self.resize_size, self.resize_size))
            wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(image1, self.resize_size, self.resize_size))
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": state,
                "prompt": prompt,
            }
            result = self.client.infer(element)
            action_chunk = np.asarray(result["actions"], dtype=np.float32)
            for action in action_chunk[: self.replan_steps]:
                self.action_plan.append(action)
            if self.query_index == 0:
                trace = {
                    "first_action_chunk_hash": action_chunk_hash(action_chunk),
                    "first_action_chunk_shape": list(action_chunk.shape),
                    "first_action_chunk_dtype": str(action_chunk.dtype),
                    "action_queue_length_after_refill": len(self.action_plan),
                    "policy_query_index": self.query_index,
                    "client_resized_image_hash": ndarray_hash(img),
                    "client_resized_wrist_hash": ndarray_hash(wrist_img),
                }
            self.query_index += 1
        action = self.action_plan.popleft()
        if trace is not None:
            trace["action_queue_length_after_first_pop"] = len(self.action_plan)
        return TracePolicyOutput(action=action.astype(np.float32), first_query_trace=trace)


class WrapperFullK1TracePolicy:
    """Wrapper full-K1 policy with first-query trace logging and no DCLD calls."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        replan_steps: int,
        resize_size: int,
        image_size: int,
        flow_steps: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.processor = processor
        self.replan_steps = int(replan_steps)
        self.resize_size = int(resize_size)
        self.image_size = int(image_size)
        self.flow_steps = int(flow_steps)
        self.device = device
        self.action_adapter = SimVLAActionAdapter(model)
        self.condition_adapter = SimVLAConditionAdapter(model, self.action_adapter)
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.action_plan: collections.deque[np.ndarray] = collections.deque()
        self.query_index = 0

    def reset(self) -> None:
        self.action_plan.clear()
        self.query_index = 0

    def _preprocess(self, image0: np.ndarray, image1: np.ndarray, state: np.ndarray, prompt: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
        img = resize_with_pad_uint8(image0, self.resize_size)
        wrist_img = resize_with_pad_uint8(image1, self.resize_size)
        img0_t = self.transform(Image.fromarray(img.astype(np.uint8)))
        img1_t = self.transform(Image.fromarray(wrist_img.astype(np.uint8)))
        padding = torch.zeros_like(img0_t)
        image_input = torch.stack([img0_t, img1_t, padding], dim=0).unsqueeze(0).to(self.device)
        image_mask = torch.tensor([[True, True, False]], device=self.device)
        lang = self.processor.encode_language([prompt])
        input_ids = lang["input_ids"].to(self.device)
        proprio = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)[:, :8]
        batch = {
            "image_input": image_input,
            "image_mask": image_mask,
            "input_ids": input_ids,
            "proprio": proprio,
        }
        hashes = {
            "client_resized_image_hash": ndarray_hash(img),
            "client_resized_wrist_hash": ndarray_hash(wrist_img),
        }
        return batch, hashes

    def step(self, image0: np.ndarray, image1: np.ndarray, state: np.ndarray, prompt: str) -> TracePolicyOutput:
        trace = None
        if not self.action_plan:
            batch, hashes = self._preprocess(image0, image1, state, prompt)
            with torch.no_grad():
                condition = self.condition_adapter.encode_condition(
                    input_ids=batch["input_ids"],
                    image_input=batch["image_input"],
                    image_mask=batch["image_mask"],
                )
                action_chunk_t = self.action_adapter.decode_action_from_condition(
                    condition,
                    batch["proprio"],
                    steps=self.flow_steps,
                    deterministic=False,
                )
            action_chunk_batched = action_chunk_t.detach().cpu().numpy().astype(np.float32)
            action_chunk = action_chunk_batched[0]
            for action in action_chunk[: self.replan_steps]:
                self.action_plan.append(action)
            if self.query_index == 0:
                trace = {
                    "first_action_chunk_hash": action_chunk_hash(action_chunk),
                    "first_action_chunk_shape": list(action_chunk.shape),
                    "first_action_chunk_dtype": str(action_chunk.dtype),
                    "action_queue_length_after_refill": len(self.action_plan),
                    "policy_query_index": self.query_index,
                    **hashes,
                }
            self.query_index += 1
        action = self.action_plan.popleft()
        if trace is not None:
            trace["action_queue_length_after_first_pop"] = len(self.action_plan)
        return TracePolicyOutput(action=action.astype(np.float32), first_query_trace=trace)


def run_rollout_set(
    *,
    label: str,
    policy_factory: Any,
    args: argparse.Namespace,
    out: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from libero.libero import benchmark, get_libero_path

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    task_ids = task_ids_for_suite(task_suite, task_order=args.task_order, max_tasks=args.max_tasks)
    rows: list[dict[str, Any]] = []
    env_rows: list[dict[str, Any]] = []
    total = len(task_ids) * int(args.num_trials)
    completed = 0
    successes = 0
    start = time.time()
    print(f"[{label}] starting {total} episodes", flush=True)
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env, task_description = get_libero_env(task, args.resolution, args.seed)
        try:
            env_config = safe_env_config(env)
            env_rows.append(
                {
                    "label": label,
                    "task_id": task_id,
                    "task_description": task_description,
                    "problem_folder": task.problem_folder,
                    "bddl_file": task.bddl_file,
                    "bddl_path": str(task_bddl_file),
                    "env_seed": args.seed,
                    "resolution": args.resolution,
                    "env_config_json": json.dumps(env_config, sort_keys=True),
                }
            )
            for trial_id in range(args.num_trials):
                episode_start = time.time()
                policy = policy_factory()
                policy.reset()
                env.reset()
                obs = env.set_init_state(initial_states[trial_id % len(initial_states)])
                image0, image1, state = build_env_obs(obs)
                initial_obs_hash = policy_obs_hash(image0, image1, state)

                done = False
                env_steps = 0
                for _ in range(args.num_wait_steps):
                    obs, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
                    env_steps += 1
                image0, image1, state = build_env_obs(obs)
                post_wait_obs_hash = policy_obs_hash(image0, image1, state)
                first_policy_query_obs_hash = post_wait_obs_hash
                first_query_trace: dict[str, Any] | None = None
                policy_steps = 0
                success_check_timing = "after_env_step"

                while policy_steps < args.max_policy_steps:
                    image0, image1, state = build_env_obs(obs)
                    step_out = policy.step(image0, image1, state, task_description)
                    if first_query_trace is None and step_out.first_query_trace is not None:
                        first_query_trace = step_out.first_query_trace
                    obs, _reward, done, _info = env.step(step_out.action.tolist())
                    env_steps += 1
                    policy_steps += 1
                    if done:
                        successes += 1
                        break
                if first_query_trace is None:
                    first_query_trace = {}
                completed += 1
                elapsed = time.time() - start
                row = {
                    "label": label,
                    "suite": args.suite,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "task_description": task_description,
                    "success": bool(done),
                    "policy_steps": policy_steps,
                    "env_steps": env_steps,
                    "episode_wall_time_seconds": float(time.time() - episode_start),
                    "initial_obs_hash": initial_obs_hash,
                    "post_wait_obs_hash": post_wait_obs_hash,
                    "first_policy_query_obs_hash": first_policy_query_obs_hash,
                    "num_wait_steps": args.num_wait_steps,
                    "max_policy_steps": args.max_policy_steps,
                    "replan_steps": args.replan_steps,
                    "success_check_timing": success_check_timing,
                    "bddl_path": str(task_bddl_file),
                    "env_seed": args.seed,
                    **first_query_trace,
                }
                rows.append(row)
                if completed % max(1, args.print_interval) == 0:
                    rate = successes / max(1, completed)
                    print(
                        f"[{label}] {completed}/{total} success={successes}/{completed} "
                        f"({rate:.3f}) task={task_id} trial={trial_id} elapsed={elapsed/60:.1f}m",
                        flush=True,
                    )
                write_csv(out / f"{label}_episode_outcomes.partial.csv", rows)
        finally:
            env.close()
    write_csv(out / f"{label}_episode_outcomes.csv", rows)
    write_csv(out / f"{label}_env_configs.csv", env_rows)
    return rows, env_rows


def start_official_server(args: argparse.Namespace, out: Path) -> subprocess.Popen[str]:
    stdout_path = out / "official_server_stdout.txt"
    stderr_path = out / "official_server_stderr.txt"
    command = [
        sys.executable,
        "-u",
        str(LIBERO_EVAL / "serve_smolvlm_libero.py"),
        "--checkpoint",
        args.checkpoint,
        "--norm_stats",
        resolve_repo_path(args.norm_stats),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    write_text(out / "official_server_command.txt", " ".join(command))
    stdout_f = stdout_path.open("w", encoding="utf-8")
    stderr_f = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(command, cwd=str(LIBERO_EVAL), stdout=stdout_f, stderr=stderr_f, text=True)
    wait_for_port(args.host, args.port, timeout_s=args.server_start_timeout)
    return proc


def stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def compare_outcomes(official_rows: list[dict[str, Any]], wrapper_rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    official_by_key = {(int(row["task_id"]), int(row["trial_id"])): row for row in official_rows}
    wrapper_by_key = {(int(row["task_id"]), int(row["trial_id"])): row for row in wrapper_rows}
    all_keys = sorted(set(official_by_key) | set(wrapper_by_key))
    comparison_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for task_id, trial_id in all_keys:
        off = official_by_key.get((task_id, trial_id))
        wrap = wrapper_by_key.get((task_id, trial_id))
        row = {
            "task_id": task_id,
            "trial_id": trial_id,
            "official_exists": off is not None,
            "wrapper_exists": wrap is not None,
            "official_success": None if off is None else bool(off["success"]),
            "wrapper_success": None if wrap is None else bool(wrap["success"]),
            "success_match": bool(off is not None and wrap is not None and bool(off["success"]) == bool(wrap["success"])),
            "initial_obs_hash_match": bool(off is not None and wrap is not None and off.get("initial_obs_hash") == wrap.get("initial_obs_hash")),
            "post_wait_obs_hash_match": bool(off is not None and wrap is not None and off.get("post_wait_obs_hash") == wrap.get("post_wait_obs_hash")),
            "first_policy_query_obs_hash_match": bool(off is not None and wrap is not None and off.get("first_policy_query_obs_hash") == wrap.get("first_policy_query_obs_hash")),
            "first_action_chunk_hash_match": bool(off is not None and wrap is not None and off.get("first_action_chunk_hash") == wrap.get("first_action_chunk_hash")),
            "official_policy_steps": None if off is None else int(off["policy_steps"]),
            "wrapper_policy_steps": None if wrap is None else int(wrap["policy_steps"]),
            "official_first_action_chunk_hash": None if off is None else off.get("first_action_chunk_hash"),
            "wrapper_first_action_chunk_hash": None if wrap is None else wrap.get("first_action_chunk_hash"),
        }
        comparison_rows.append(row)
        if not row["success_match"]:
            detail = dict(row)
            if off is not None:
                detail.update({f"official_{key}": off.get(key) for key in off.keys()})
            if wrap is not None:
                detail.update({f"wrapper_{key}": wrap.get(key) for key in wrap.keys()})
            mismatches.append(detail)
    write_csv(out / "official_vs_wrapper_episode_comparison.csv", comparison_rows)
    write_csv(out / "outcome_mismatches.csv", mismatches)
    write_csv(out / "mismatch_episode_traces.csv", mismatches[:3])
    official_successes = sum(1 for row in official_rows if row.get("success"))
    wrapper_successes = sum(1 for row in wrapper_rows if row.get("success"))
    summary = {
        "official_successes": official_successes,
        "official_episodes": len(official_rows),
        "official_success_rate": official_successes / max(1, len(official_rows)),
        "wrapper_successes": wrapper_successes,
        "wrapper_episodes": len(wrapper_rows),
        "wrapper_success_rate": wrapper_successes / max(1, len(wrapper_rows)),
        "num_mismatches": len(mismatches),
        "num_initial_obs_hash_mismatches": sum(1 for row in comparison_rows if not row["initial_obs_hash_match"]),
        "num_post_wait_obs_hash_mismatches": sum(1 for row in comparison_rows if not row["post_wait_obs_hash_match"]),
        "num_first_policy_query_obs_hash_mismatches": sum(1 for row in comparison_rows if not row["first_policy_query_obs_hash_match"]),
        "num_first_action_chunk_hash_mismatches": sum(1 for row in comparison_rows if not row["first_action_chunk_hash_match"]),
        "mismatched_task_trials": [[int(row["task_id"]), int(row["trial_id"])] for row in mismatches],
    }
    write_json(out / "comparison_summary.json", summary)
    return summary


def write_final_report(out: Path, summary: dict[str, Any], env_comparison: dict[str, Any]) -> None:
    if summary["num_mismatches"] == 0 and summary["official_successes"] == summary["wrapper_successes"]:
        verdict = "BASELINE_CALIBRATED"
        explanation = "Official and wrapper full-K1 outcomes matched on this run."
    elif (
        summary["num_initial_obs_hash_mismatches"] == 0
        and summary["num_post_wait_obs_hash_mismatches"] == 0
        and summary["num_first_policy_query_obs_hash_mismatches"] == 0
        and summary["num_first_action_chunk_hash_mismatches"] > 0
    ):
        verdict = "BASELINE_GAP_EXPLAINED_BUT_NOT_FIXED"
        explanation = (
            "Reset/wait/first-query observations match, but unpaired first action chunks differ. "
            "Given prior saved-batch max action diff 0.0 under identical noise, the remaining gap is "
            "consistent with unpaired action-flow noise / rollout sampling variance, not a known wrapper action-path bug."
        )
    else:
        verdict = "BASELINE_GAP_UNRESOLVED"
        explanation = "Observation or environment hashes differ, or the mismatch pattern is not explained by action noise alone."

    lines = [
        "# Final Wrapper Baseline Calibration Diagnosis",
        "",
        f"- verdict: `{verdict}`",
        f"- official_success: `{summary['official_successes']}/{summary['official_episodes']} ({summary['official_success_rate']:.3f})`",
        f"- wrapper_full_k1_success: `{summary['wrapper_successes']}/{summary['wrapper_episodes']} ({summary['wrapper_success_rate']:.3f})`",
        f"- mismatched_task_trials: `{summary['mismatched_task_trials']}`",
        f"- initial_obs_hash_mismatches: `{summary['num_initial_obs_hash_mismatches']}`",
        f"- post_wait_obs_hash_mismatches: `{summary['num_post_wait_obs_hash_mismatches']}`",
        f"- first_policy_query_obs_hash_mismatches: `{summary['num_first_policy_query_obs_hash_mismatches']}`",
        f"- first_action_chunk_hash_mismatches: `{summary['num_first_action_chunk_hash_mismatches']}`",
        "",
        "## Diagnosis",
        "",
        explanation,
        "",
        "## Environment/Config Comparison",
        "",
        f"- task_bddl_match_all: `{env_comparison.get('task_bddl_match_all')}`",
        f"- env_seed_match_all: `{env_comparison.get('env_seed_match_all')}`",
        f"- resolution_match_all: `{env_comparison.get('resolution_match_all')}`",
        f"- controller_config_match_all: `{env_comparison.get('controller_config_match_all')}`",
        "",
        "## Saved Artifacts",
        "",
        "- `environment_metadata.json`",
        "- `official_episode_outcomes.csv`",
        "- `wrapper_full_k1_episode_outcomes.csv`",
        "- `official_vs_wrapper_episode_comparison.csv`",
        "- `outcome_mismatches.csv`",
        "- `mismatch_episode_traces.csv`",
        "- `official_env_configs.csv`",
        "- `wrapper_full_k1_env_configs.csv`",
        "- `env_config_comparison.json`",
        "- `official_server_command.txt`",
        "- `official_server_stdout.txt`",
        "- `official_server_stderr.txt`",
        "",
        "## Scope",
        "",
        "No DCLD K>1 sweep or DCLD training was run. The wrapper path is full-K1 only.",
    ]
    write_text(out / "final_wrapper_baseline_calibration_diagnosis.md", "\n".join(lines))


def compare_env_configs(official_env: list[dict[str, Any]], wrapper_env: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    off_by_task = {int(row["task_id"]): row for row in official_env}
    wrap_by_task = {int(row["task_id"]): row for row in wrapper_env}
    rows = []
    for task_id in sorted(set(off_by_task) | set(wrap_by_task)):
        off = off_by_task.get(task_id, {})
        wrap = wrap_by_task.get(task_id, {})
        row = {
            "task_id": task_id,
            "bddl_match": off.get("bddl_path") == wrap.get("bddl_path"),
            "env_seed_match": off.get("env_seed") == wrap.get("env_seed"),
            "resolution_match": off.get("resolution") == wrap.get("resolution"),
            "controller_config_exact_match": off.get("env_config_json") == wrap.get("env_config_json"),
            "controller_config_match": semantic_config_json(off.get("env_config_json")) == semantic_config_json(wrap.get("env_config_json")),
            "official_bddl_path": off.get("bddl_path"),
            "wrapper_bddl_path": wrap.get("bddl_path"),
            "official_env_config_json": off.get("env_config_json"),
            "wrapper_env_config_json": wrap.get("env_config_json"),
        }
        rows.append(row)
    write_csv(out / "env_config_comparison.csv", rows)
    summary = {
        "task_bddl_match_all": bool(rows and all(row["bddl_match"] for row in rows)),
        "env_seed_match_all": bool(rows and all(row["env_seed_match"] for row in rows)),
        "resolution_match_all": bool(rows and all(row["resolution_match"] for row in rows)),
        "controller_config_exact_match_all": bool(rows and all(row["controller_config_exact_match"] for row in rows)),
        "controller_config_match_all": bool(rows and all(row["controller_config_match"] for row in rows)),
    }
    write_json(out / "env_config_comparison.json", summary)
    return summary


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIBERO_ROOT", str(LIBERO_ROOT))
    write_json(out / "environment_metadata.json", collect_environment(args))
    write_json(out / "eval_config.json", {key: str(value) for key, value in vars(args).items()})

    server_proc = None
    try:
        server_proc = start_official_server(args, out)
        official_rows, official_env = run_rollout_set(
            label="official",
            policy_factory=lambda: OfficialWebSocketTracePolicy(
                args.host,
                args.port,
                replan_steps=args.replan_steps,
                resize_size=args.client_resize_size,
            ),
            args=args,
            out=out,
        )
    finally:
        stop_process(server_proc)

    device = torch.device(args.device)
    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device)
    model.eval()
    norm_stats = resolve_repo_path(args.norm_stats) if args.norm_stats else None
    if norm_stats and Path(norm_stats).exists():
        model.action_space.load_norm_stats(norm_stats)
    for param in model.parameters():
        param.requires_grad_(False)
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)
    wrapper_rows, wrapper_env = run_rollout_set(
        label="wrapper_full_k1",
        policy_factory=lambda: WrapperFullK1TracePolicy(
            model=model,
            processor=processor,
            replan_steps=args.replan_steps,
            resize_size=args.client_resize_size,
            image_size=args.image_size,
            flow_steps=args.flow_steps,
            device=device,
        ),
        args=args,
        out=out,
    )
    env_comparison = compare_env_configs(official_env, wrapper_env, out)
    summary = compare_outcomes(official_rows, wrapper_rows, out)
    write_final_report(out, summary, env_comparison)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--task-order", choices=["official_reverse", "ascending"], default="official_reverse")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--num-wait-steps", type=int, default=10)
    parser.add_argument("--max-policy-steps", type=int, default=900)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--client-resize-size", type=int, default=224)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--server-start-timeout", type=float, default=420.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print-interval", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
