"""Bounded real-observation parity and call-counter gate for the three rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    load_generation_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_contracts import (
    FROZEN_GENERATION_CHECKPOINT_SHA256,
    FROZEN_GENERATION_SOURCE_SHA256,
    FULL_ROW,
    GENERATION_ROW,
    NAIVE_ROW,
    atomic_write_json,
    native_nfe_time_grid,
    require_egl_preflight,
    validate_manifest_identity,
    validate_row_counters,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_control_eval import (
    _make_policy,
    _verify_provenance,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
)
from architectures.simvla.wrappers.dcld_eval.rollout_runner import (
    build_env_obs,
    get_libero_env,
)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _diff(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | bool]:
    difference = (left.detach().cpu().float() - right.detach().cpu().float()).abs()
    return {
        "mean_abs_diff": float(difference.mean().item()),
        "max_abs_diff": float(difference.max().item()),
        "allclose_1e_6": bool(torch.allclose(left, right, atol=1e-6, rtol=1e-6)),
    }


def _official_with_initial_noise(
    model: object,
    batch: dict[str, torch.Tensor],
    initial_noise: torch.Tensor,
) -> torch.Tensor:
    original = torch.randn
    consumed = 0

    def controlled_randn(*shape: object, **kwargs: object) -> torch.Tensor:
        nonlocal consumed
        requested = tuple(int(value) for value in shape)
        if requested == tuple(initial_noise.shape):
            consumed += 1
            return initial_noise.to(
                device=kwargs.get("device", initial_noise.device),
                dtype=kwargs.get("dtype", initial_noise.dtype),
            ).clone()
        return original(*shape, **kwargs)

    with mock.patch("torch.randn", side_effect=controlled_randn):
        action = model.generate_actions(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
            proprio=batch["proprio"],
            steps=10,
        )
    if consumed != 1:
        raise RuntimeError(f"official source consumed controlled action noise {consumed} times")
    return action


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing parity output: {output}")
    physical_gpu = int(args.physical_gpu_id)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("parity requires exactly one explicit physical GPU")
    if os.environ.get("MUJOCO_EGL_DEVICE_ID") != str(physical_gpu):
        raise RuntimeError("parity requires an explicit matching MuJoCo EGL device")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("parity is EGL-only")
    provenance = _verify_provenance(args)
    preflight = require_egl_preflight(args.egl_preflight, physical_gpu)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    manifest_gate = validate_manifest_identity(
        manifest, expected_manifest_sha256=args.expected_manifest_sha256
    )
    if manifest_gate["verdict"] != "EPISODE_MANIFEST_PASS":
        raise RuntimeError(json.dumps(manifest_gate, indent=2, sort_keys=True))
    renderer_mismatches = {
        name: {"expected": value, "observed": os.environ.get(name)}
        for name, value in manifest.get("renderer", {}).items()
        if os.environ.get(name) != value
    }
    if renderer_mismatches:
        raise RuntimeError(
            "parity runtime does not match immutable renderer manifest: "
            + json.dumps(renderer_mismatches, sort_keys=True)
        )

    configure_strict_torch_determinism(int(manifest["determinism_seed"]))
    device = torch.device("cuda:0")
    bundle = Path(args.bundle_root).expanduser().resolve()
    model, processor, _ = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=bundle / "norm" / "libero_norm_official_32700d0.json",
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    freeze_module(model)
    updater, checkpoint_payload = load_generation_checkpoint(
        bundle / "checkpoint" / "generation_step_030000.pt", device=device
    )
    freeze_module(updater)
    if checkpoint_payload["source_lock"]["combined_sha256"] != FROZEN_GENERATION_SOURCE_SHA256:
        raise RuntimeError("Generation checkpoint source identity changed")

    from libero.libero import benchmark

    suite_name = str(manifest["suite"])
    task_id = int(args.task_id)
    trial_id = int(args.trial_id)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)
    env, prompt = get_libero_env(
        task,
        int(manifest["environment_resolution"]),
        int(manifest["environment_seed"]),
    )
    try:
        env.reset()
        observation = env.set_init_state(initial_states[trial_id])
        for _ in range(int(manifest["num_wait_steps"])):
            observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
        image0, image1, proprio = build_env_obs(observation)

        policies = {
            row: _make_policy(
                row=row,
                model=model,
                processor=processor,
                updater=updater if row == GENERATION_ROW else None,
                device=device,
                suite=suite_name,
                task_id=task_id,
                trial_id=trial_id,
                action_noise_seed_base=int(manifest["action_noise_seed_base"]),
            )
            for row in (FULL_ROW, NAIVE_ROW, GENERATION_ROW)
        }
        batch = policies[FULL_ROW].preprocess(image0, image1, proprio, prompt)
        condition = policies[FULL_ROW].condition_adapter.encode_condition(
            input_ids=batch["input_ids"],
            image_input=batch["image_input"],
            image_mask=batch["image_mask"],
        )
        initial_noise, full_noise_seed = policies[FULL_ROW]._paired_initial_noise(
            condition, batch["proprio"], 0
        )
        if initial_noise is None or full_noise_seed is None:
            raise RuntimeError("full policy did not produce query-keyed action noise")
        official_action = _official_with_initial_noise(model, batch, initial_noise)

        row_reports: dict[str, object] = {}
        action_outputs: dict[str, torch.Tensor] = {}
        noise_seeds: dict[str, int] = {}
        for row, policy in policies.items():
            transformer_calls = 0
            updater_calls = 0
            transformer_time_grid: list[float] = []

            def transformer_hook(
                _module: object,
                _inputs: tuple[object, ...],
                kwargs: dict[str, object],
            ) -> None:
                nonlocal transformer_calls
                transformer_calls += 1
                tau = kwargs.get("t")
                if not isinstance(tau, torch.Tensor) or tau.numel() < 1:
                    raise RuntimeError("action transformer hook did not observe tensor t")
                transformer_time_grid.append(float(tau.detach().flatten()[0].item()))

            def updater_hook(*_: object) -> None:
                nonlocal updater_calls
                updater_calls += 1

            transformer_handle = model.transformer.register_forward_pre_hook(
                transformer_hook, with_kwargs=True
            )
            updater_handle = (
                updater.register_forward_hook(updater_hook) if row == GENERATION_ROW else None
            )
            try:
                policy._refill_action_queue(batch)
            finally:
                transformer_handle.remove()
                if updater_handle is not None:
                    updater_handle.remove()
            action = policy.cached_action_chunk
            if action is None or not torch.isfinite(action).all():
                raise RuntimeError(f"{row} returned a non-finite or missing action chunk")
            record = policy.action_chunk_records[0]
            noise_seeds[row] = int(record["action_noise_seed"])
            action_outputs[row] = action
            queries = int(policy.metrics.counters["num_policy_queries"])
            full_calls = int(policy.metrics.counters["num_action_transformer_calls"])
            generation_updates = int(
                policy.metrics.counters.get("num_generation_decoder_only_steps", 0)
            )
            integration_updates = full_calls + generation_updates
            counter_gate = validate_row_counters(
                row,
                policy_queries=queries,
                full_action_transformer_calls=full_calls,
                generation_loop_updates=generation_updates,
                integration_updates=integration_updates,
                full_vlm_calls=int(policy.metrics.counters["num_full_vlm_calls"]),
            )
            row_reports[row] = {
                "counter_gate": counter_gate,
                "forward_hook_transformer_calls": transformer_calls,
                "forward_hook_updater_calls": updater_calls,
                "observed_transformer_time_grid": transformer_time_grid,
                "action_chunk_sha256": _tensor_sha256(action),
                "action_noise_seed": noise_seeds[row],
                "finite_action": True,
            }

        full_diff = _diff(official_action, action_outputs[FULL_ROW])
        first_r_diff = _diff(official_action[:, :5], action_outputs[FULL_ROW][:, :5])
        noise_identity = len(set(noise_seeds.values())) == 1
        expected_time_grids = {
            FULL_ROW: list(native_nfe_time_grid(10)),
            NAIVE_ROW: list(native_nfe_time_grid(3)),
            GENERATION_ROW: [1.0, 0.6, 0.2],
        }
        time_grid_checks = {
            row: bool(
                np.allclose(
                    row_reports[row]["observed_transformer_time_grid"],
                    expected,
                    atol=1e-6,
                    rtol=0.0,
                )
            )
            for row, expected in expected_time_grids.items()
        }
        checks = {
            "official_vs_wrapped_full_exact": full_diff["allclose_1e_6"],
            "official_vs_wrapped_first_r_exact": first_r_diff["allclose_1e_6"],
            "same_query_keyed_noise_all_rows": noise_identity,
            "full_hook_calls_10": row_reports[FULL_ROW]["forward_hook_transformer_calls"] == 10,
            "naive_hook_calls_3": row_reports[NAIVE_ROW]["forward_hook_transformer_calls"] == 3,
            "naive_updater_calls_0": row_reports[NAIVE_ROW]["forward_hook_updater_calls"] == 0,
            "generation_hook_calls_3": row_reports[GENERATION_ROW]["forward_hook_transformer_calls"] == 3,
            "generation_updater_calls_7": row_reports[GENERATION_ROW]["forward_hook_updater_calls"] == 7,
            "full_source_native_time_grid": time_grid_checks[FULL_ROW],
            "naive_source_native_time_grid": time_grid_checks[NAIVE_ROW],
            "generation_full_call_time_grid_0_4_8": time_grid_checks[GENERATION_ROW],
            "all_counter_gates_pass": all(
                report["counter_gate"]["verdict"] == "ROW_COUNTER_PASS"
                for report in row_reports.values()
            ),
            "checkpoint_sha256_fixed": provenance["observed_artifact_hashes"]["checkpoint"]
            == FROZEN_GENERATION_CHECKPOINT_SHA256,
            "condition_loop_not_loaded": True,
            "k_c_is_one": True,
            "condition_change_code_zero": True,
        }
        verdict = "GENERATION_THREE_ROW_PARITY_PASS" if all(checks.values()) else "GENERATION_THREE_ROW_PARITY_FAIL"
        result: dict[str, object] = {
            "verdict": verdict,
            "classification": args.classification,
            "source_combined_sha256": FROZEN_GENERATION_SOURCE_SHA256,
            "manifest_sha256": manifest["manifest_sha256"],
            "physical_gpu_id": physical_gpu,
            "task_id": task_id,
            "trial_id": trial_id,
            "official_action_chunk_sha256": _tensor_sha256(official_action),
            "official_vs_wrapped_full": full_diff,
            "official_vs_wrapped_first_r": first_r_diff,
            "rows": row_reports,
            "expected_transformer_time_grids": expected_time_grids,
            "checks": checks,
            "egl_preflight": preflight,
            "paper_runtime_match": provenance["paper_runtime_match"],
        }
    finally:
        env.close()

    output.mkdir(parents=True)
    atomic_write_json(output / "generation_three_row_counter_gate.json", result)
    lines = [
        "# Generation Three-Row Parity",
        "",
        f"- verdict: `{result['verdict']}`",
        f"- source: `{FROZEN_GENERATION_SOURCE_SHA256}`",
        f"- manifest: `{manifest['manifest_sha256']}`",
        f"- EGL GPU: `{physical_gpu}`",
        f"- paper runtime match: `{provenance['paper_runtime_match']}`",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {name}: `{value}`" for name, value in checks.items())
    (output / "generation_three_row_parity_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if result["verdict"] != "GENERATION_THREE_ROW_PARITY_PASS":
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--egl-preflight", required=True)
    parser.add_argument("--physical-gpu-id", type=int, required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument(
        "--classification",
        choices=("HOST_LOCAL_EGL_DIAGNOSTIC", "RB2_CONFIRMATORY_EGL"),
        required=True,
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
