"""Manifest-locked rb2 EGL evaluation for selected stability checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    condition_update_with_code,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.fixed_2x2_eval import (
    SynchronizedConditionK_CPolicy,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_hidden import (
    full_generation_step_with_hidden,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
)
from architectures.simvla.adapters.latentloop.native_v0_long_eval import (
    RealSimVLANativeV0Policy,
    _trajectory_metrics,
)
from architectures.simvla.adapters.latentloop.stability_alignment.checkpoint import (
    load_modules_from_checkpoint,
)
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    BUNDLE_SCHEMA,
    GENERATION_NG3_FULL_INDICES,
    atomic_write_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    build_env_obs,
    get_libero_env,
)
from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop


MODES = ("nfe10", "naive_nfe3", "learned_ng3")


class StabilityAlignedPolicy(SynchronizedConditionK_CPolicy):
    def __init__(self, *, generation_updater: Any, generation_mode: str, **kwargs: Any) -> None:
        if generation_mode not in MODES:
            raise ValueError(f"generation_mode must be one of {MODES}")
        k_c = int(kwargs.get("k_c", 2))
        if k_c == 8:
            # The source control class intentionally limits its fixed study to
            # K_C<=4.  Reuse its unchanged queue/update implementation while
            # bypassing only that constructor guard for the conditional K_C=8
            # checkpoint, whose updater has independently verified age support.
            row_name = kwargs.pop("row_name", None)
            kwargs.pop("k_c", None)
            RealSimVLANativeV0Policy.__init__(self, **kwargs)
            self.k_c = 8
            self.mode = row_name or "condition_kc8_ng10"
            self.row_name = self.mode
            self.refresh_every = 8
        else:
            super().__init__(**kwargs)
        self.generation_mode = generation_mode
        self.row_name = f"selected_kc{self.k_c}_{generation_mode}"
        self.mode = self.row_name
        self.generation_loop = SimVLAGenerationLoop(
            generation_updater, self.model.transformer.action_decoder
        ).to(self.device).eval()
        self._active_code: torch.Tensor | None = None
        self.condition_change_code_norms: list[float] = []
        self.metrics.latencies.setdefault("generation_loop_ms", [])

    def reset(self) -> None:
        super().reset()
        self._active_code = None
        self.condition_change_code_norms = []

    def _full_refresh(self, batch: dict[str, torch.Tensor], *, policy_query_index: int):
        self._active_code = None
        return super()._full_refresh(batch, policy_query_index=policy_query_index)

    def _v0_update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        age: int,
        policy_query_index: int,
    ):
        if self.cached_condition is None or self.cached_raw_rgb is None or self.cached_proprio is None:
            raise RuntimeError("stability update requires preceding query state")
        if self.condition_layout is None:
            raise RuntimeError("stability update requires full-refresh token layout")
        max_age = int(self.native_v0.condition_updater.max_age)
        if age < 1 or age > max_age or age >= self.k_c:
            raise ValueError("Condition age is outside the selected K_C schedule")
        pair = NativeV0ObservationPair(
            previous_images=self.cached_raw_rgb,
            current_images=batch["raw_rgb"],
            previous_proprio=self.cached_proprio,
            current_proprio=batch["proprio"],
        )
        self._sync()
        started = time.perf_counter()
        with torch.no_grad():
            exposed = condition_update_with_code(
                self.native_v0,
                self.cached_condition,
                pair,
                valid_mask=self.condition_layout.valid_mask,
                group_ids=self.condition_layout.group_ids,
                age=age,
            )
        self._sync()
        self.metrics.latencies.setdefault("condition_updater_ms", []).append(
            (time.perf_counter() - started) * 1000.0
        )
        self.metrics.counters["num_condition_updater_calls"] += 1
        self.metrics.counters["num_observation_encoder_calls"] += 1
        code = exposed.condition_change_code.detach()
        if not bool((code.float().norm(dim=-1) > 0).all()):
            raise RuntimeError("online Condition update produced zero c_j")
        self._active_code = code
        self.condition_change_code_norms.extend(
            code.float().norm(dim=-1).cpu().tolist()
        )
        action, seed = self._decode(
            exposed.update.condition,
            batch["proprio"],
            policy_query_index=policy_query_index,
        )
        self.cached_condition = exposed.update.condition.detach()
        self.cached_raw_rgb = batch["raw_rgb"].detach()
        self.cached_proprio = batch["proprio"].detach()
        self.cached_action_chunk = action.detach()
        return exposed.update.condition, action, seed

    def _decode(
        self,
        condition: torch.Tensor,
        proprio: torch.Tensor,
        *,
        policy_query_index: int,
    ):
        if self.generation_mode == "nfe10":
            return super()._decode(
                condition, proprio, policy_query_index=policy_query_index
            )
        initial_noise, seed = self._paired_initial_noise(
            condition, proprio, policy_query_index
        )
        if initial_noise is None:
            raise RuntimeError("selected evaluation requires explicit paired noise")
        self._sync()
        started = time.perf_counter()
        if self.generation_mode == "naive_nfe3":
            decoded = self.action_adapter.decode_action_from_condition(
                condition,
                proprio,
                steps=3,
                initial_noise=initial_noise,
                return_debug=True,
            )
            if int(decoded.debug.get("iterations", -1)) != 3:
                raise RuntimeError("naive NFE3 did not execute exactly three source steps")
            action = decoded.action
            full_calls = 3
            learned_steps = 0
        else:
            normalized = self.action_adapter.normalize_proprio(proprio)
            # The validated 30K Generation parent was trained and evaluated on
            # the zero-code lane.  S50/S150 changes only U_C; real c_j remains
            # diagnostic until the separately gated optional joint branch.
            code = condition.new_zeros((condition.shape[0], 128))

            def full_step(noisy_action: torch.Tensor, tau: torch.Tensor):
                output = full_generation_step_with_hidden(
                    self.model.transformer,
                    condition=condition,
                    noisy_action=noisy_action,
                    proprio=normalized,
                    tau=tau,
                    dt=-0.1,
                )
                return output.action_hidden, output.velocity

            with torch.no_grad():
                trace = self.generation_loop(
                    initial_noise,
                    full_step=full_step,
                    full_step_indices=GENERATION_NG3_FULL_INDICES,
                    proprio=normalized,
                    condition=condition,
                    condition_valid_mask=None,
                    condition_change_code=code,
                )
                action = self.action_adapter.action_space.postprocess(
                    trace.final_noisy_action
                )
            full_calls = 3
            learned_steps = 7
        self._sync()
        elapsed = (time.perf_counter() - started) * 1000.0
        self.metrics.latencies["action_transformer_ms"].append(elapsed)
        if self.generation_mode == "learned_ng3":
            self.metrics.latencies["generation_loop_ms"].append(elapsed)
        self.metrics.counters["num_action_transformer_calls"] += full_calls
        self.metrics.counters["num_action_transformer_decodes"] += 1
        self.metrics.counters["num_generation_decoder_only_steps"] += learned_steps
        return action, seed


def _renderer_contract(manifest: dict[str, Any]) -> None:
    required = {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
    }
    mismatches = {
        key: {"expected": value, "observed": os.environ.get(key)}
        for key, value in required.items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"EGL contract failed: {mismatches}")
    for key in (
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "PYTHONHASHSEED",
    ):
        expected = manifest.get("renderer", {}).get(key)
        if expected is not None and os.environ.get(key) != expected:
            raise RuntimeError(f"manifest renderer field changed: {key}")


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _load_partial_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    identities = [(int(row["task_id"]), int(row["trial_id"])) for row in rows]
    if len(identities) != len(set(identities)):
        raise RuntimeError("partial online output contains duplicate episodes")
    return rows


def _resume_contract(output: Path, expected: dict[str, Any]) -> list[dict[str, Any]]:
    summary = output / "row_summary.json"
    if summary.is_file():
        payload = load_json(summary)
        if (
            payload.get("verdict") == "RB2_STABILITY_ROW_COMPLETE"
            and int(payload.get("episodes", -1)) == 500
            and payload.get("manifest_sha256") == expected["manifest_sha256"]
            and payload.get("checkpoint_sha256") == expected["checkpoint_sha256"]
            and int(payload.get("k_c", -1)) == expected["k_c"]
            and payload.get("generation_mode") == expected["generation_mode"]
        ):
            return []
        raise RuntimeError("existing row summary is incompatible")
    contract_path = output / "run_contract.json"
    if not contract_path.is_file():
        raise RuntimeError("existing output lacks a resumable run contract")
    observed = load_json(contract_path)
    locked = (
        "renderer",
        "manifest_sha256",
        "checkpoint_sha256",
        "k_c",
        "generation_mode",
        "generation_condition_change_code",
        "condition_change_code_diagnostic_only",
        "h",
        "r",
        "additional_inference_seed",
        "diagnostic_only",
        "offline_gate_passed",
    )
    mismatches = {
        key: {"expected": expected.get(key), "observed": observed.get(key)}
        for key in locked
        if expected.get(key) != observed.get(key)
    }
    if mismatches:
        raise RuntimeError(f"partial output contract changed: {mismatches}")
    return _load_partial_rows(output / "progress.jsonl")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id):
        raise RuntimeError("exactly the requested physical GPU must be visible")
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(args.physical_gpu_id):
        raise RuntimeError("MUJOCO_EGL_DEVICE_ID must equal the physical GPU ID")
    output = Path(args.output).expanduser().resolve()
    bundle = Path(args.bundle).expanduser().resolve()
    if args.diagnostic_only and args.k_c == 8:
        raise RuntimeError("diagnostic-only evaluation supports only K_C=2,3,4")
    readiness_name = (
        "READY_SHORT_DIAGNOSTIC_FOR_RB2.json"
        if args.diagnostic_only
        else (
            "READY_KC8_FOR_RB2.json"
            if args.k_c == 8
            else "READY_SHORT_FOR_RB2.json"
        )
    )
    expected_verdict = (
        "READY_SHORT_DIAGNOSTIC_FOR_RB2"
        if args.diagnostic_only
        else ("READY_KC8_FOR_RB2" if args.k_c == 8 else "READY_SHORT_FOR_RB2")
    )
    ready = load_json(bundle / readiness_name)
    if ready.get("schema_version") != BUNDLE_SCHEMA or ready.get("verdict") != expected_verdict:
        raise RuntimeError(f"bundle readiness failed: {readiness_name}")
    if args.diagnostic_only:
        if not ready.get("diagnostic_only") or ready.get("offline_gate_passed") is not False:
            raise RuntimeError("diagnostic bundle must preserve the failed offline verdict")
    checkpoint = bundle / ready["checkpoint"]
    if sha256_file(checkpoint) != ready["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint hash changed")
    if not args.diagnostic_only and args.k_c == 3 and not ready.get("kc3_offline_ready"):
        raise RuntimeError("K_C=3 was not offline-ready on sd1")
    if not args.diagnostic_only and args.k_c == 4 and not ready.get("kc4_offline_ready"):
        raise RuntimeError("K_C=4 was not offline-ready on sd1")
    if args.k_c == 8 and not ready.get("kc8_offline_ready"):
        raise RuntimeError("K_C=8 was not offline-ready on sd1")
    manifest = load_json(args.manifest)
    copied_manifest = dict(manifest)
    manifest_digest = copied_manifest.pop("manifest_sha256", None)
    if manifest_digest is None or canonical_sha256(copied_manifest) != manifest_digest:
        raise RuntimeError("online episode manifest hash mismatch")
    _renderer_contract(manifest)
    if int(manifest.get("trials_per_task", manifest.get("episodes_per_task", -1))) != 50:
        raise RuntimeError("online manifest is not 50 episodes/task")
    if str(manifest.get("suite")) != "libero_10":
        raise RuntimeError("online manifest is not LIBERO-Long")
    run_contract = {
        "hostname": socket.gethostname(),
        "renderer": "egl",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": ready["checkpoint_sha256"],
        "k_c": args.k_c,
        "generation_mode": args.generation_mode,
        "h": 10,
        "r": 5,
        "generation_condition_change_code": "zero",
        "condition_change_code_diagnostic_only": True,
        "additional_inference_seed": False,
        "diagnostic_only": bool(args.diagnostic_only),
        "offline_gate_passed": bool(ready.get("offline_gate_passed", True)),
    }
    rows: list[dict[str, Any]] = []
    if output.exists():
        rows = _resume_contract(output, run_contract)
        if (output / "row_summary.json").is_file():
            return load_json(output / "row_summary.json")
    else:
        output.mkdir(parents=True)
        atomic_write_json(output / "run_contract.json", run_contract)
    configure_strict_torch_determinism(int(manifest["determinism_seed"]))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model, processor, _ = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=bundle / "libero_norm.json",
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    freeze_module(model)
    modules, payload = load_modules_from_checkpoint(checkpoint, device=device)
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    episodes_by_task = {
        task_id: sorted(
            [item for item in manifest["episodes"] if int(item["task_id"]) == task_id],
            key=lambda item: int(item["trial_id"]),
        )
        for task_id in range(10)
    }
    if any(len(value) != 50 for value in episodes_by_task.values()):
        raise RuntimeError("manifest does not contain 10x50 episodes")
    completed = {(int(row["task_id"]), int(row["trial_id"])) for row in rows}
    progress = tqdm(
        total=500,
        initial=len(rows),
        desc=f"K_C={args.k_c} {args.generation_mode}",
        dynamic_ncols=True,
    )
    for task_id in reversed(range(10)):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        env, prompt = get_libero_env(
            task,
            int(manifest["environment_resolution"]),
            int(manifest["environment_seed"]),
        )
        try:
            for spec in episodes_by_task[task_id]:
                trial_id = int(spec["trial_id"])
                if (task_id, trial_id) in completed:
                    continue
                observation = env.reset()
                observation = env.set_init_state(
                    initial_states[int(spec["init_state_index"]) % len(initial_states)]
                )
                for _ in range(int(manifest["num_wait_steps"])):
                    observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
                policy = StabilityAlignedPolicy(
                    model=model,
                    processor=processor,
                    adapter=modules.condition,
                    generation_updater=modules.generation,
                    generation_mode=args.generation_mode,
                    checkpoint_id=args.checkpoint,
                    device=device,
                    suite="libero_10",
                    task_id=task_id,
                    trial_id=trial_id,
                    action_noise_seed_base=int(manifest["action_noise_seed_base"]),
                    k_c=args.k_c,
                    row_name=(
                        f"diagnostic_s50_10k_kc{args.k_c}_{args.generation_mode}"
                        if args.diagnostic_only
                        else f"selected_kc{args.k_c}_{args.generation_mode}"
                    ),
                )
                policy_query_ms: list[float] = []
                success = False
                action_count = 0
                executed_gripper: list[float] = []
                executed_actions: list[np.ndarray] = []
                for _ in range(int(manifest["max_policy_actions"])):
                    image0, image1, proprio = build_env_obs(observation)
                    query = not policy.action_queue
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    step = policy.act(image0, image1, proprio, prompt)
                    torch.cuda.synchronize(device)
                    if query:
                        policy_query_ms.append((time.perf_counter() - started) * 1000.0)
                    observation, _, done, _ = env.step(step.action.tolist())
                    executed_gripper.append(float(step.action[6]))
                    executed_actions.append(np.asarray(step.action, dtype=np.float32))
                    action_count += 1
                    if done:
                        success = True
                        break
                counters = policy.metrics.counters
                trajectory = _trajectory_metrics(executed_actions)
                query_count = int(counters.get("num_policy_queries", 0))
                expected_vlm = (query_count + args.k_c - 1) // args.k_c
                expected_transformer = query_count * (10 if args.generation_mode == "nfe10" else 3)
                if int(counters.get("num_full_vlm_calls", 0)) != expected_vlm:
                    raise RuntimeError("Condition call counter drift")
                if int(counters.get("num_action_transformer_calls", 0)) != expected_transformer:
                    raise RuntimeError("Generation call counter drift")
                row = {
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "success": int(success),
                    "episode_length": action_count,
                    "policy_queries": query_count,
                    "full_vlm_calls": int(counters.get("num_full_vlm_calls", 0)),
                    "condition_updates": int(counters.get("num_condition_updater_calls", 0)),
                    "transformer_evaluations": int(counters.get("num_action_transformer_calls", 0)),
                    "learned_generation_updates": int(counters.get("num_generation_decoder_only_steps", 0)),
                    "policy_query_latency_ms": _mean(policy_query_ms),
                    "vlm_latency_ms": _mean(policy.metrics.latencies.get("VLM_encoder_ms", [])),
                    "condition_latency_ms": _mean(policy.metrics.latencies.get("condition_updater_ms", [])),
                    "generation_latency_ms": _mean(policy.metrics.latencies.get("action_transformer_ms", [])),
                    "gripper_switches": int(
                        sum(
                            (executed_gripper[index] >= 0.0)
                            != (executed_gripper[index - 1] >= 0.0)
                            for index in range(1, len(executed_gripper))
                        )
                    ),
                    "normalized_second_difference": trajectory[
                        "normalized_second_difference"
                    ],
                    "short_reversal": trajectory["short_reversal"],
                    "switch_disagreement": trajectory["switch_disagreement"],
                }
                rows.append(row)
                with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                progress.update(1)
                progress.set_postfix(success=f"{sum(item['success'] for item in rows)}/{len(rows)}")
        finally:
            env.close()
    progress.close()
    rows.sort(key=lambda row: (int(row["task_id"]), int(row["trial_id"])))
    if len(rows) != 500:
        raise RuntimeError(f"selected online row incomplete: {len(rows)}/500")
    with (output / "episode_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    successes = sum(int(row["success"]) for row in rows)
    result = {
        "verdict": (
            "RB2_STABILITY_DIAGNOSTIC_ROW_COMPLETE"
            if args.diagnostic_only
            else "RB2_STABILITY_ROW_COMPLETE"
        ),
        "classification": "DIAGNOSTIC_ONLY" if args.diagnostic_only else "GATED_EVALUATION",
        "diagnostic_only": bool(args.diagnostic_only),
        "offline_gate_passed": bool(ready.get("offline_gate_passed", True)),
        "row": (
            f"diagnostic_s50_10k_kc{args.k_c}_{args.generation_mode}"
            if args.diagnostic_only
            else f"selected_kc{args.k_c}_{args.generation_mode}"
        ),
        "k_c": args.k_c,
        "generation_mode": args.generation_mode,
        "generation_condition_change_code": "zero",
        "condition_change_code_diagnostic_only": True,
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": ready["checkpoint_sha256"],
        "mean_policy_query_latency_ms": _mean([row["policy_query_latency_ms"] for row in rows]),
        "mean_vlm_latency_ms": _mean([row["vlm_latency_ms"] for row in rows]),
        "mean_condition_latency_ms": _mean([row["condition_latency_ms"] for row in rows]),
        "mean_generation_latency_ms": _mean([row["generation_latency_ms"] for row in rows]),
        "mean_gripper_switches_per_episode": _mean(
            [float(row["gripper_switches"]) for row in rows]
        ),
        "switch_disagreement_p95": float(
            np.quantile([float(row["switch_disagreement"]) for row in rows], 0.95)
        ),
        "per_task_success_rate": {
            str(task): sum(row["success"] for row in rows if row["task_id"] == task) / 50.0
            for task in range(10)
        },
    }
    atomic_write_json(output / "row_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--k-c", type=int, choices=(2, 3, 4, 8), required=True)
    parser.add_argument("--generation-mode", choices=MODES, required=True)
    parser.add_argument("--physical-gpu-id", type=int, default=0)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--diagnostic-only", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
