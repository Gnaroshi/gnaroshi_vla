"""Fixed K_C=2 x N_G={10,3} EGL evaluation for SimVLA LatentLoop."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    load_generation_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FROZEN_CHECKPOINT_REVISION,
    FROZEN_EXACT_CACHE_MANIFEST_SHA256,
    FROZEN_GENERATION_CHECKPOINT_SHA256,
    FROZEN_GENERATION_SOURCE_SHA256,
    FROZEN_NORM_STATS_SHA256,
    FROZEN_ROOT_COMMIT,
    FROZEN_UPSTREAM_COMMIT,
    atomic_write_json,
    load_json,
    require_egl_preflight,
    runtime_versions,
    sha256_file,
    validate_manifest_identity,
    validate_sd1_shard,
    verify_file_hashes,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_policy import (
    RealSimVLAGenerationPolicy,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_contracts import (
    COMBINED_ROW,
    CONDITION_ROW,
    FROZEN_CONDITION_CHECKPOINT_SHA256,
    FROZEN_CONDITION_SOURCE_SHA256,
    ROWS,
)
from architectures.simvla.adapters.latentloop.native_v0_long_eval import _trajectory_metrics
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    load_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.native_v0_policy import (
    RealSimVLANativeV0Policy,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
)
from architectures.simvla.wrappers.dcld_eval import rollout_runner as rollout_runner_runtime
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    build_env_obs,
    get_libero_env,
    save_episode_video,
    video_frame_from_obs,
)
from methods.latentloop.modules.simvla_generation_loop import (
    SimVLAGenerationLoop,
)


ROOT = Path(__file__).resolve().parents[5]
UPSTREAM = Path(
    os.environ.get("SIMVLA_UPSTREAM_ROOT", ROOT / "architectures" / "simvla" / "upstream")
).expanduser().resolve()

def _ensure_generation_latency_schema() -> None:
    if "generation_loop_ms" not in rollout_runner_runtime.LATENCY_FIELDS:
        rollout_runner_runtime.LATENCY_FIELDS.append("generation_loop_ms")


class SynchronizedConditionK_C2Policy(RealSimVLANativeV0Policy):
    """Use one learned condition update between full VLM condition refreshes."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.mode = CONDITION_ROW
        self.row_name = CONDITION_ROW
        self.refresh_every = 2

    def _refill_action_queue(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        self.metrics.counters["num_policy_queries"] += 1
        query = int(self.query_index)
        age = query % 2
        if age == 0:
            _, action_chunk, seed = self._full_refresh(
                batch, policy_query_index=query
            )
            source = "full_refresh"
            refreshed = True
        else:
            _, action_chunk, seed = self._v0_update(
                batch, age=1, policy_query_index=query
            )
            source = "native_v0"
            refreshed = False
        self.action_queue.clear()
        for action in action_chunk[0, :5]:
            self.action_queue.append((action.detach(), source))
        self.query_trace.append(
            {
                "policy_query_index": query,
                "age": age,
                "source": source,
                "full_vlm_called": refreshed,
                "condition_updater_called": not refreshed,
                "action_noise_seed": seed,
                "action_horizon": 10,
                "execution_horizon": 5,
                "k_c": 2,
            }
        )
        if self.log_action_chunks:
            self.action_chunk_records.append(
                {
                    "suite": self.suite,
                    "task_id": self.task_id,
                    "trial_id": self.trial_id,
                    "episode_step_index": int(self.step_index),
                    "policy_query_index": query,
                    "row_name": self.row_name,
                    "mode": self.mode,
                    "k": 2,
                    "queue_mode": source,
                    "refreshed": bool(refreshed),
                    "full_vlm_called": bool(refreshed),
                    "condition_updater_called": bool(not refreshed),
                    "paired_action_noise": True,
                    "action_noise_seed": seed,
                    "action_chunk_shape": list(action_chunk.shape),
                    "action_chunk": action_chunk.detach().cpu().float(),
                }
            )
        self.query_index += 1
        queries = int(self.metrics.counters["num_policy_queries"])
        full_calls = int(self.metrics.counters["num_full_vlm_calls"])
        update_calls = int(self.metrics.counters["num_condition_updater_calls"])
        if full_calls != (queries + 1) // 2 or update_calls != queries // 2:
            raise RuntimeError("K_C=2 condition-call counter drift")
        return {
            "refreshed": refreshed,
            "age": age,
            "queue_mode": source,
            "action_noise_seed": seed,
        }


class SynchronizedCombinedK_C2N_G3Policy(SynchronizedConditionK_C2Policy):
    """Uncoupled fixed 2x2 row: K_C=2 condition loop and N_G=3 generation loop."""

    def __init__(self, *, generation_updater: Any, **kwargs: Any) -> None:
        _ensure_generation_latency_schema()
        super().__init__(**kwargs)
        self.mode = COMBINED_ROW
        self.row_name = COMBINED_ROW
        self.n_g = 3
        self.full_step_indices = (0, 4, 8)
        self.generation_loop = SimVLAGenerationLoop(
            generation_updater.eval(), self.model.transformer.action_decoder
        ).to(self.device)
        self.generation_loop.eval()
        self.metrics.latencies.setdefault("generation_loop_ms", [])

    def _decode(
        self,
        condition: torch.Tensor,
        proprio: torch.Tensor,
        *,
        policy_query_index: int,
    ) -> tuple[torch.Tensor, int | None]:
        # The first combined experiment is deliberately uncoupled: c_j is zero.
        return RealSimVLAGenerationPolicy._decode(
            self,
            condition,
            proprio,
            policy_query_index=policy_query_index,
        )


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(root), *args), text=True, stderr=subprocess.STDOUT
    ).strip()


def _verify_provenance(args: argparse.Namespace) -> dict[str, Any]:
    bundle = Path(args.bundle_root).expanduser().resolve()
    condition_checkpoint = Path(args.condition_checkpoint).expanduser().resolve()
    fixed_source_lock_path = Path(args.fixed_2x2_source_lock).expanduser().resolve()
    source_lock_path = bundle / "metadata" / "source_lock.json"
    source_lock = load_json(source_lock_path)
    transfer_manifest = load_json(bundle / "transfer_manifest.json")
    fixed_source_lock = load_json(fixed_source_lock_path)
    failures: list[str] = []

    if source_lock.get("combined_sha256") != FROZEN_GENERATION_SOURCE_SHA256:
        failures.append("source combined SHA-256 changed")
    if source_lock.get("root_commit") != FROZEN_ROOT_COMMIT:
        failures.append("Generation root commit changed")
    if source_lock.get("simvla_upstream_commit") != FROZEN_UPSTREAM_COMMIT:
        failures.append("SimVLA upstream commit changed")
    if source_lock.get("checkpoint_revision") != FROZEN_CHECKPOINT_REVISION:
        failures.append("SimVLA checkpoint revision changed")
    if source_lock.get("norm_stats_sha256") != FROZEN_NORM_STATS_SHA256:
        failures.append("source lock norm-statistics SHA-256 changed")
    if (
        source_lock.get("exact_cache_manifest_sha256")
        != FROZEN_EXACT_CACHE_MANIFEST_SHA256
    ):
        failures.append("exact-cache manifest SHA-256 changed")

    try:
        root_commit = _git(ROOT, "rev-parse", "HEAD")
    except Exception as exc:
        root_commit = f"ERROR: {exc}"
    try:
        upstream_commit = _git(UPSTREAM, "rev-parse", "HEAD")
    except Exception as exc:
        upstream_commit = f"ERROR: {exc}"
    if fixed_source_lock.get("root_commit") != root_commit:
        failures.append(
            "fixed 2x2 worktree commit mismatch: "
            f"{root_commit} != {fixed_source_lock.get('root_commit')}"
        )
    if fixed_source_lock.get("base_generation_root_commit") != FROZEN_ROOT_COMMIT:
        failures.append("fixed 2x2 base Generation commit changed")
    if fixed_source_lock.get("upstream_commit") != FROZEN_UPSTREAM_COMMIT:
        failures.append("fixed 2x2 upstream commit changed")
    if upstream_commit != FROZEN_UPSTREAM_COMMIT:
        failures.append(f"upstream commit mismatch: {upstream_commit}")

    file_report = verify_file_hashes(ROOT, source_lock["relevant_file_sha256"])
    if file_report["verdict"] != "FILE_HASHES_PASS":
        failures.append("one or more locked Generation source files changed")
    control_file_report = verify_file_hashes(
        ROOT, transfer_manifest.get("control_file_sha256", {})
    )
    if not transfer_manifest.get("control_file_sha256"):
        failures.append("transfer manifest does not lock control evaluator files")
    elif control_file_report["verdict"] != "FILE_HASHES_PASS":
        failures.append("one or more control evaluator files changed")

    fixed_file_report = verify_file_hashes(
        ROOT, fixed_source_lock.get("file_sha256", {})
    )
    if not fixed_source_lock.get("file_sha256"):
        failures.append("fixed 2x2 source lock is empty")
    elif fixed_file_report["verdict"] != "FILE_HASHES_PASS":
        failures.append("one or more fixed 2x2 source files changed")
    if (
        fixed_source_lock.get("generation_source_combined_sha256")
        != FROZEN_GENERATION_SOURCE_SHA256
    ):
        failures.append("fixed 2x2 Generation source identity changed")
    if (
        fixed_source_lock.get("condition_source_combined_sha256")
        != FROZEN_CONDITION_SOURCE_SHA256
    ):
        failures.append("fixed 2x2 Condition source identity changed")

    checkpoint = bundle / "checkpoint" / "generation_step_030000.pt"
    norm_stats = bundle / "norm" / "libero_norm_official_32700d0.json"
    cache_manifest = bundle / "exact_cache_contract" / "manifest.json"
    observed_artifact_hashes = {
        "checkpoint": sha256_file(checkpoint) if checkpoint.is_file() else None,
        "condition_checkpoint": (
            sha256_file(condition_checkpoint)
            if condition_checkpoint.is_file()
            else None
        ),
        "norm_stats": sha256_file(norm_stats) if norm_stats.is_file() else None,
        "exact_cache_manifest": (
            sha256_file(cache_manifest) if cache_manifest.is_file() else None
        ),
    }
    expected_artifact_hashes = {
        "checkpoint": FROZEN_GENERATION_CHECKPOINT_SHA256,
        "condition_checkpoint": FROZEN_CONDITION_CHECKPOINT_SHA256,
        "norm_stats": FROZEN_NORM_STATS_SHA256,
        "exact_cache_manifest": FROZEN_EXACT_CACHE_MANIFEST_SHA256,
    }
    for name, expected in expected_artifact_hashes.items():
        if observed_artifact_hashes[name] != expected:
            failures.append(
                f"{name} SHA-256 mismatch: {observed_artifact_hashes[name]} != {expected}"
            )

    current_runtime = runtime_versions()
    expected_runtime = source_lock.get("environment", {})
    runtime_mismatches = {
        key: {"expected": expected_runtime.get(key), "observed": current_runtime.get(key)}
        for key in expected_runtime
        if current_runtime.get(key) != expected_runtime.get(key)
    }
    classification = str(args.classification)
    if runtime_mismatches and classification != "HOST_LOCAL_EGL_DIAGNOSTIC":
        failures.append(
            "runtime differs from frozen rb2 contract outside HOST_LOCAL_EGL_DIAGNOSTIC"
        )

    report = {
        "verdict": "FROZEN_PROVENANCE_PASS" if not failures else "FROZEN_PROVENANCE_FAIL",
        "classification": classification,
        "paper_runtime_match": not runtime_mismatches,
        "source_combined_sha256": FROZEN_GENERATION_SOURCE_SHA256,
        "root_commit": root_commit,
        "upstream_commit": upstream_commit,
        "checkpoint": str(checkpoint),
        "condition_checkpoint": str(condition_checkpoint),
        "norm_stats": str(norm_stats),
        "exact_cache_manifest": str(cache_manifest),
        "expected_artifact_hashes": expected_artifact_hashes,
        "observed_artifact_hashes": observed_artifact_hashes,
        "expected_runtime": expected_runtime,
        "current_runtime": current_runtime,
        "runtime_mismatches": runtime_mismatches,
        "locked_file_report": file_report,
        "control_file_report": control_file_report,
        "fixed_2x2_source_lock": str(fixed_source_lock_path),
        "fixed_2x2_file_report": fixed_file_report,
        "failures": failures,
    }
    if failures:
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    return report


def _parse_task_ids(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError("task IDs must be a non-empty unique CSV")
    if any(value < 0 or value > 9 for value in values):
        raise ValueError("LIBERO task IDs must be in [0,9]")
    return values


def _make_policy(
    *,
    row: str,
    model: Any,
    processor: Any,
    condition_updater: Any,
    generation_updater: Any,
    device: torch.device,
    suite: str,
    task_id: int,
    trial_id: int,
    action_noise_seed_base: int,
) -> Any:
    common = {
        "model": model,
        "processor": processor,
        "adapter": condition_updater,
        "checkpoint_id": DEFAULT_CHECKPOINT,
        "device": device,
        "suite": suite,
        "task_id": task_id,
        "trial_id": trial_id,
        "action_noise_seed_base": action_noise_seed_base,
        "client_resize_size": 224,
        "image_size": 384,
        "flow_steps": 10,
        "log_action_chunks": True,
    }
    if row == CONDITION_ROW:
        return SynchronizedConditionK_C2Policy(**common)
    if row == COMBINED_ROW:
        if generation_updater is None:
            raise RuntimeError("combined row requires the frozen Generation updater")
        return SynchronizedCombinedK_C2N_G3Policy(
            **common,
            generation_updater=generation_updater,
        )
    raise ValueError(f"unsupported row: {row}")


def _validate_fixed_2x2_counters(
    row: str,
    *,
    policy_queries: int,
    full_vlm_calls: int,
    condition_updater_calls: int,
    full_action_transformer_calls: int,
    generation_loop_updates: int,
) -> dict[str, Any]:
    if row not in ROWS:
        raise ValueError(f"unknown fixed 2x2 row: {row}")
    queries = int(policy_queries)
    expected = {
        "full_vlm_calls": (queries + 1) // 2,
        "condition_updater_calls": queries // 2,
        "full_action_transformer_calls": queries * (10 if row == CONDITION_ROW else 3),
        "generation_loop_updates": queries * (0 if row == CONDITION_ROW else 7),
        "integration_updates": queries * 10,
    }
    observed = {
        "full_vlm_calls": int(full_vlm_calls),
        "condition_updater_calls": int(condition_updater_calls),
        "full_action_transformer_calls": int(full_action_transformer_calls),
        "generation_loop_updates": int(generation_loop_updates),
        "integration_updates": int(full_action_transformer_calls)
        + int(generation_loop_updates),
    }
    checks = {name: observed[name] == value for name, value in expected.items()}
    return {
        "verdict": "FIXED_2X2_COUNTER_PASS" if all(checks.values()) else "FIXED_2X2_COUNTER_FAIL",
        "row": row,
        "policy_queries": queries,
        "expected": expected,
        "observed": observed,
        "checks": checks,
    }


def _gripper_metrics(actions: Sequence[np.ndarray]) -> dict[str, float | int]:
    if len(actions) < 2:
        return {"gripper_switches": 0, "gripper_switch_rate": 0.0}
    array = np.asarray(actions, dtype=np.float32)
    switches = int(np.count_nonzero((array[1:, 6] >= 0) != (array[:-1, 6] >= 0)))
    return {
        "gripper_switches": switches,
        "gripper_switch_rate": float(switches / max(1, len(actions) - 1)),
    }


def _save_action_chunks(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        np.savez_compressed(
            path,
            task_id=np.empty(0, dtype=np.int16),
            trial_id=np.empty(0, dtype=np.int16),
            policy_query_index=np.empty(0, dtype=np.int32),
            action_noise_seed=np.empty(0, dtype=np.uint64),
            action_chunk=np.empty((0, 10, 7), dtype=np.float32),
        )
        return
    np.savez_compressed(
        path,
        task_id=np.asarray([item["task_id"] for item in records], dtype=np.int16),
        trial_id=np.asarray([item["trial_id"] for item in records], dtype=np.int16),
        policy_query_index=np.asarray(
            [item["policy_query_index"] for item in records], dtype=np.int32
        ),
        action_noise_seed=np.asarray(
            [item["action_noise_seed"] for item in records], dtype=np.uint64
        ),
        action_chunk=np.stack([item["action_chunk"] for item in records]).astype(
            np.float32
        ),
    )


def evaluate_shard(args: argparse.Namespace) -> dict[str, Any]:
    if args.row not in ROWS:
        raise ValueError(f"row must be one of {ROWS}")
    physical_gpu_id = int(args.physical_gpu_id)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu_id):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal exactly one physical GPU ID")
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(physical_gpu_id):
        raise RuntimeError("MUJOCO_EGL_DEVICE_ID must equal the physical GPU ID")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("scientific Generation control evaluation is EGL-only")
    task_ids = _parse_task_ids(args.task_ids)
    if args.classification == "HOST_LOCAL_EGL_DIAGNOSTIC":
        shard_contract = validate_sd1_shard(physical_gpu_id, task_ids)
        if shard_contract["verdict"] != "SD1_SHARD_PASS":
            raise RuntimeError(json.dumps(shard_contract, indent=2, sort_keys=True))
    else:
        shard_contract = {
            "verdict": "CONFIRMATORY_SHARD_PASS",
            "physical_gpu_id": physical_gpu_id,
            "task_ids": list(task_ids),
        }

    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    provenance = _verify_provenance(args)
    preflight = require_egl_preflight(args.egl_preflight, physical_gpu_id)
    manifest = load_json(args.manifest)
    manifest_report = validate_manifest_identity(
        manifest,
        expected_manifest_sha256=args.expected_manifest_sha256 or None,
    )
    if manifest_report["verdict"] != "EPISODE_MANIFEST_PASS":
        raise RuntimeError(json.dumps(manifest_report, indent=2, sort_keys=True))
    parity_gate = load_json(args.fixed_2x2_parity_gate)
    parity_checks = {
        "verdict": parity_gate.get("verdict") == "FIXED_2X2_PARITY_PASS",
        "manifest": parity_gate.get("manifest_sha256")
        == manifest.get("manifest_sha256"),
        "condition_source": parity_gate.get("condition_source_combined_sha256")
        == FROZEN_CONDITION_SOURCE_SHA256,
        "generation_source": parity_gate.get("generation_source_combined_sha256")
        == FROZEN_GENERATION_SOURCE_SHA256,
        "condition_checkpoint": parity_gate.get("condition_checkpoint_sha256")
        == FROZEN_CONDITION_CHECKPOINT_SHA256,
        "generation_checkpoint": parity_gate.get("generation_checkpoint_sha256")
        == FROZEN_GENERATION_CHECKPOINT_SHA256,
    }
    if not all(parity_checks.values()):
        raise RuntimeError(f"fixed 2x2 parity gate failed: {parity_checks}")
    renderer_mismatches = {
        name: {"expected": value, "observed": os.environ.get(name)}
        for name, value in manifest.get("renderer", {}).items()
        if os.environ.get(name) != value
    }
    if renderer_mismatches:
        raise RuntimeError(
            "runtime does not match immutable renderer/determinism manifest: "
            + json.dumps(renderer_mismatches, sort_keys=True)
        )
    specs_by_task: dict[int, list[dict[str, Any]]] = {}
    for task_id in task_ids:
        specs = [item for item in manifest["episodes"] if int(item["task_id"]) == task_id]
        if len(specs) != int(manifest["trials_per_task"]):
            raise RuntimeError(f"task {task_id} does not have exactly 50 manifest episodes")
        specs_by_task[task_id] = sorted(specs, key=lambda item: int(item["trial_id"]))

    output.mkdir(parents=True)
    atomic_write_json(output / "frozen_provenance.json", provenance)
    atomic_write_json(output / "egl_preflight.json", preflight)
    atomic_write_json(output / "manifest_validation.json", manifest_report)
    atomic_write_json(output / "fixed_2x2_parity_gate.json", parity_gate)
    atomic_write_json(
        output / "renderer_runtime_contract.json",
        {
            "verdict": "RENDERER_RUNTIME_CONTRACT_PASS",
            "expected": manifest.get("renderer", {}),
            "observed": {
                name: os.environ.get(name) for name in manifest.get("renderer", {})
            },
        },
    )
    atomic_write_json(output / "host_shard_contract.json", shard_contract)

    configure_strict_torch_determinism(int(manifest["determinism_seed"]))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("shard requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    bundle = Path(args.bundle_root).expanduser().resolve()
    norm_stats = bundle / "norm" / "libero_norm_official_32700d0.json"
    generation_checkpoint = bundle / "checkpoint" / "generation_step_030000.pt"
    model, processor, _ = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    freeze_module(model)
    condition_updater, condition_payload = load_native_v0_checkpoint(
        args.condition_checkpoint,
        device=device,
        require_final_150k=True,
    )
    if int(condition_payload.get("global_optimizer_step", -1)) != 150_000:
        raise RuntimeError("Condition V0 checkpoint is not optimizer step 150,000")
    condition_source = condition_payload.get("source_lock", {})
    if condition_source.get("combined_sha256") != FROZEN_CONDITION_SOURCE_SHA256:
        raise RuntimeError("Condition V0 checkpoint source lock changed")
    if condition_source.get("norm_stats_sha256") != FROZEN_NORM_STATS_SHA256:
        raise RuntimeError("Condition V0 checkpoint norm-statistics lock changed")
    if condition_source.get("checkpoint", {}).get("revision") != FROZEN_CHECKPOINT_REVISION:
        raise RuntimeError("Condition V0 checkpoint SimVLA revision changed")
    freeze_module(condition_updater)

    generation_updater = None
    generation_payload = None
    if args.row == COMBINED_ROW:
        generation_updater, generation_payload = load_generation_checkpoint(
            generation_checkpoint, device=device
        )
        checkpoint_source = generation_payload["source_lock"]["combined_sha256"]
        if checkpoint_source != FROZEN_GENERATION_SOURCE_SHA256:
            raise RuntimeError("Generation checkpoint source lock changed")
        if int(generation_payload["optimizer_step"]) != 30_000:
            raise RuntimeError("Generation checkpoint is not optimizer step 30,000")
        freeze_module(generation_updater)

    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[str(manifest["suite"])]()
    episode_rows: list[dict[str, Any]] = []
    action_chunk_rows: list[dict[str, Any]] = []
    assigned_total = len(task_ids) * int(manifest["trials_per_task"])
    progress = tqdm(
        total=assigned_total,
        desc=f"{args.row} gpu{physical_gpu_id}",
        dynamic_ncols=True,
        mininterval=float(args.tqdm_mininterval),
    )
    shard_started = time.perf_counter()
    for task_id in reversed(task_ids):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, prompt = get_libero_env(
            task, int(manifest["environment_resolution"]), int(manifest["environment_seed"])
        )
        try:
            for spec in specs_by_task[task_id]:
                trial_id = int(spec["trial_id"])
                episode_started = time.perf_counter()
                env.reset()
                observation = env.set_init_state(
                    initial_states[int(spec["init_state_index"]) % len(initial_states)]
                )
                environment_ms = 0.0
                for _ in range(int(manifest["num_wait_steps"])):
                    started = time.perf_counter()
                    observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
                    environment_ms += (time.perf_counter() - started) * 1000.0

                policy = _make_policy(
                    row=args.row,
                    model=model,
                    processor=processor,
                    condition_updater=condition_updater,
                    generation_updater=generation_updater,
                    device=device,
                    suite=str(manifest["suite"]),
                    task_id=task_id,
                    trial_id=trial_id,
                    action_noise_seed_base=int(manifest["action_noise_seed_base"]),
                )
                actions: list[np.ndarray] = []
                policy_ms: list[float] = []
                query_policy_ms: list[float] = []
                frames: list[np.ndarray] = []
                success = False
                for action_index in range(int(manifest["max_policy_actions"])):
                    if args.save_video and action_index % int(args.video_stride) == 0:
                        frames.append(video_frame_from_obs(observation))
                    image0, image1, proprio = build_env_obs(observation)
                    is_policy_query = not policy.action_queue
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    step = policy.act(image0, image1, proprio, prompt)
                    torch.cuda.synchronize(device)
                    elapsed_policy_ms = (time.perf_counter() - started) * 1000.0
                    policy_ms.append(elapsed_policy_ms)
                    if is_policy_query:
                        query_policy_ms.append(elapsed_policy_ms)
                    started = time.perf_counter()
                    observation, _, done, _ = env.step(step.action.tolist())
                    environment_ms += (time.perf_counter() - started) * 1000.0
                    actions.append(step.action.copy())
                    if done:
                        success = True
                        break

                counters = policy.metrics.counters
                policy_queries = int(counters.get("num_policy_queries", 0))
                full_calls = int(counters.get("num_action_transformer_calls", 0))
                generation_updates = int(
                    counters.get("num_generation_decoder_only_steps", 0)
                )
                condition_updates = int(
                    counters.get("num_condition_updater_calls", 0)
                )
                integration_updates = full_calls + generation_updates
                counter_gate = _validate_fixed_2x2_counters(
                    args.row,
                    policy_queries=policy_queries,
                    full_vlm_calls=int(counters.get("num_full_vlm_calls", 0)),
                    condition_updater_calls=condition_updates,
                    full_action_transformer_calls=full_calls,
                    generation_loop_updates=generation_updates,
                )
                if counter_gate["verdict"] != "FIXED_2X2_COUNTER_PASS":
                    raise RuntimeError(json.dumps(counter_gate, indent=2, sort_keys=True))

                trajectory = _trajectory_metrics(actions)
                gripper = _gripper_metrics(actions)
                action_transformer_ms = policy.metrics.latencies.get(
                    "action_transformer_ms", []
                )
                vlm_encoder_ms = policy.metrics.latencies.get("VLM_encoder_ms", [])
                condition_updater_ms = policy.metrics.latencies.get(
                    "condition_updater_ms", []
                )
                generation_loop_ms = policy.metrics.latencies.get(
                    "generation_loop_ms", []
                )
                row = {
                    "row": args.row,
                    "classification": args.classification,
                    "inference_seed": args.inference_seed,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "init_state_index": int(spec["init_state_index"]),
                    "success": int(success),
                    "episode_length": len(actions),
                    "num_policy_queries": policy_queries,
                    "num_full_vlm_calls": int(counters.get("num_full_vlm_calls", 0)),
                    "num_condition_updater_calls": condition_updates,
                    "num_full_action_transformer_evaluations": full_calls,
                    "num_generation_loop_updates": generation_updates,
                    "num_integration_updates": integration_updates,
                    "num_action_queue_steps": int(counters.get("num_action_queue_steps", 0)),
                    "effective_k_c": float(
                        policy_queries / max(1, int(counters.get("num_full_vlm_calls", 0)))
                    ),
                    "latency_per_policy_query_ms": float(np.mean(query_policy_ms)),
                    "latency_per_executed_action_ms": float(np.mean(policy_ms)),
                    "model_vlm_encoder_per_query_ms": float(np.mean(vlm_encoder_ms)),
                    "model_action_generation_per_query_ms": float(
                        np.mean(action_transformer_ms)
                    ),
                    "model_condition_updater_per_update_ms": (
                        float(np.mean(condition_updater_ms))
                        if condition_updater_ms
                        else 0.0
                    ),
                    "model_generation_loop_per_query_ms": (
                        float(np.mean(generation_loop_ms))
                        if generation_loop_ms
                        else 0.0
                    ),
                    "policy_wall_time_seconds": float(sum(policy_ms) / 1000.0),
                    "environment_wall_time_seconds": float(environment_ms / 1000.0),
                    "episode_wall_time_seconds": float(
                        time.perf_counter() - episode_started
                    ),
                    "normalized_second_difference": trajectory[
                        "normalized_second_difference"
                    ],
                    "short_reversal": trajectory["short_reversal"],
                    **gripper,
                    "counter_gate": counter_gate["verdict"],
                }
                episode_rows.append(row)
                with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")

                for record in policy.action_chunk_records:
                    action_chunk_rows.append(
                        {
                            "task_id": int(record["task_id"]),
                            "trial_id": int(record["trial_id"]),
                            "policy_query_index": int(record["policy_query_index"]),
                            "action_noise_seed": int(record["action_noise_seed"]),
                            "action_chunk": record["action_chunk"].numpy()[0],
                        }
                    )

                if args.save_video and (not args.video_failures_only or not success):
                    video_root = output / "videos" / f"task_{task_id:02d}"
                    existing = len(list(video_root.glob("*.mp4")))
                    if existing < int(args.video_max_per_task):
                        suffix = "success" if success else "failure"
                        save_episode_video(
                            frames,
                            video_root / f"trial_{trial_id:03d}_{suffix}.mp4",
                            10,
                        )
                progress.update(1)
                progress.set_postfix(
                    successes=sum(int(item["success"]) for item in episode_rows),
                    sr=f"{np.mean([item['success'] for item in episode_rows]) * 100:.1f}%",
                )
        finally:
            env.close()
    progress.close()

    with (output / "episode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    _save_action_chunks(output / "action_chunks.npz", action_chunk_rows)
    summary = {
        "verdict": "FIXED_2X2_SHARD_PASS",
        "row": args.row,
        "classification": args.classification,
        "inference_seed": args.inference_seed,
        "physical_gpu_id": physical_gpu_id,
        "task_ids": list(task_ids),
        "episodes": len(episode_rows),
        "successes": sum(int(item["success"]) for item in episode_rows),
        "success_rate": float(np.mean([item["success"] for item in episode_rows])),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_combined_sha256": {
            "condition": FROZEN_CONDITION_SOURCE_SHA256,
            "generation": FROZEN_GENERATION_SOURCE_SHA256,
        },
        "condition_checkpoint_sha256": FROZEN_CONDITION_CHECKPOINT_SHA256,
        "generation_checkpoint_sha256": FROZEN_GENERATION_CHECKPOINT_SHA256,
        "paper_runtime_match": provenance["paper_runtime_match"],
        "all_episode_counter_gates_pass": all(
            item["counter_gate"] == "FIXED_2X2_COUNTER_PASS"
            for item in episode_rows
        ),
        "total_policy_queries": sum(
            int(item["num_policy_queries"]) for item in episode_rows
        ),
        "total_full_vlm_calls": sum(
            int(item["num_full_vlm_calls"]) for item in episode_rows
        ),
        "total_condition_updater_calls": sum(
            int(item["num_condition_updater_calls"]) for item in episode_rows
        ),
        "total_full_action_transformer_evaluations": sum(
            int(item["num_full_action_transformer_evaluations"])
            for item in episode_rows
        ),
        "total_generation_loop_updates": sum(
            int(item["num_generation_loop_updates"]) for item in episode_rows
        ),
        "elapsed_seconds": float(time.perf_counter() - shard_started),
        "action_chunk_records": len(action_chunk_rows),
    }
    atomic_write_json(output / "shard_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", choices=ROWS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", default="")
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--condition-checkpoint", required=True)
    parser.add_argument("--fixed-2x2-source-lock", required=True)
    parser.add_argument("--fixed-2x2-parity-gate", required=True)
    parser.add_argument("--egl-preflight", required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument(
        "--classification",
        choices=("HOST_LOCAL_EGL_DIAGNOSTIC", "RB2_CONFIRMATORY_EGL"),
        required=True,
    )
    parser.add_argument(
        "--inference-seed", choices=("seed01", "seed02", "seed03"), required=True
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--tqdm-mininterval", type=float, default=1.0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-failures-only", action="store_true")
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--video-max-per-task", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = evaluate_shard(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
