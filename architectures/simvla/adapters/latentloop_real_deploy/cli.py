"""Command-line entry point for staged SimVLA real-world deployment checks."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=("source-preflight", "artifact-preflight", "read-only-profile", "live"),
    )
    parser.add_argument("--manifest")
    parser.add_argument(
        "--method",
        choices=(
            "baseline",
            "condition_loop",
            "latentloop",
            "vla_cache_full",
            "vla_cache",
        ),
        default="latentloop",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    parser.add_argument("--steps", type=int, default=0)
    return parser.parse_args()


def _write_json(path: str | Path | None, filename: str, payload: dict[str, Any]) -> None:
    if path is None:
        return
    output = Path(path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    target = output / filename
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _configure_deterministic_noise(seed: int) -> None:
    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _load_controller(args: argparse.Namespace):
    if not args.manifest:
        raise ValueError(f"--manifest is required for {args.mode}")
    from .contracts import load_deployment_contract
    from .controller import SimVLARealController

    contract = load_deployment_contract(args.manifest, verify_artifacts=True)
    _configure_deterministic_noise(int(contract.runtime["seed"]))
    controller = SimVLARealController.from_contract(
        contract, deployment_method=args.method, device=args.device
    )
    return contract, controller


def _artifact_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    contract, controller = _load_controller(args)
    cameras = contract.hardware["cameras"]
    height = int(cameras["height"])
    width = int(cameras["width"])
    exterior = np.full((height, width, 3), 96, dtype=np.uint8)
    wrist = np.full((height, width, 3), 160, dtype=np.uint8)
    state = np.asarray(contract.state["preflight_vector"], dtype=np.float32)
    robot_state = {
        "pose6d": state[:6],
        "tcp_rotvec": state[3:6],
        "gripper_open_state": state[6:7],
        "gripper_position": state[7:8],
    }
    steps = max(int(args.steps or 11), 11)
    actions = []
    instruction = str(contract.runtime["instructions"][0])
    for _ in range(steps):
        _, _, _, info = controller.forward(
            {
                "color_image": [exterior, wrist],
                "robot_state": robot_state,
                "language_instruction": instruction,
            },
            include_info=True,
            timestep=len(actions),
        )
        actions.append(info["record"]["action"])

    counters = dict(controller.policy.metrics.counters)
    queries = (steps + 4) // 5
    expected = {
        "num_policy_queries": queries,
        "num_action_queue_steps": steps,
        "num_action_transformer_calls": queries
        * (3 if args.method == "latentloop" else 10),
    }
    if args.method == "baseline":
        expected.update(
            {"num_full_vlm_calls": queries, "num_condition_updater_calls": 0}
        )
    elif args.method in {"condition_loop", "latentloop"}:
        expected.update(
            {
                "num_full_vlm_calls": (queries + 1) // 2,
                "num_condition_updater_calls": queries // 2,
            }
        )
    else:
        expected.update(
            {
                "num_vlm_queries": queries,
                "num_vla_cache_anchor_queries": 1,
                "num_vla_cache_nonanchor_queries": queries - 1,
            }
        )
        if args.method == "vla_cache":
            expected["num_actual_kv_reuse_queries"] = queries - 1
    mismatch = {
        key: {"observed": int(counters.get(key, 0)), "expected": value}
        for key, value in expected.items()
        if int(counters.get(key, 0)) != value
    }
    if args.method == "vla_cache" and int(
        counters.get("skipped_text_token_layers", 0)
    ) <= 0:
        mismatch["skipped_text_token_layers"] = {
            "observed": int(counters.get("skipped_text_token_layers", 0)),
            "required": "positive",
        }
    if args.method == "vla_cache_full" and int(
        counters.get("skipped_text_token_layers", 0)
    ) != 0:
        mismatch["skipped_text_token_layers"] = {
            "observed": int(counters.get("skipped_text_token_layers", 0)),
            "required": 0,
        }
    if mismatch:
        raise RuntimeError(f"Deployment schedule preflight failed: {mismatch}")
    payload = {
        "verdict": "ARTIFACT_PREFLIGHT_PASS",
        "deployment": controller.deployment_metadata(),
        "steps": steps,
        "expected_counters": expected,
        "observed_counters": counters,
        "actions_finite": bool(np.isfinite(np.asarray(actions)).all()),
        "robot_hardware_initialized": False,
        "robot_command_issued": False,
    }
    _write_json(args.output, "artifact_preflight.json", payload)
    return payload


def _read_only_profile(args: argparse.Namespace) -> dict[str, Any]:
    contract, controller = _load_controller(args)
    from .hardware import build_deploy_config
    from .runtime import ReadOnlyDeployEnvironment, run_read_only_profile

    if not args.output:
        raise ValueError("--output is required for read-only-profile")
    steps = int(args.steps or contract.runtime["max_steps"])
    if steps < 11:
        raise ValueError("read-only-profile requires at least 11 steps")
    cfg = build_deploy_config(contract)
    cfg.enable_rollout_media = False
    cfg.enable_observer_media = False
    env = ReadOnlyDeployEnvironment(cfg)
    try:
        return run_read_only_profile(
            controller=controller, env=env, output=args.output, steps=steps
        )
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    from .source_lock import verify_source_snapshots

    source = verify_source_snapshots()
    if args.mode == "source-preflight":
        _write_json(args.output, "source_preflight.json", source)
        print(json.dumps(source, indent=2, sort_keys=True))
        return
    if args.mode == "artifact-preflight":
        result = _artifact_preflight(args)
    elif args.mode == "read-only-profile":
        result = _read_only_profile(args)
    else:
        if not args.manifest:
            raise ValueError("--manifest is required for live")
        from .contracts import load_deployment_contract, require_live_authorization

        contract = load_deployment_contract(args.manifest, verify_artifacts=True)
        require_live_authorization(contract)
        _, controller = _load_controller(args)
        from .deploy_gui import run_live_gui

        run_live_gui(controller=controller)
        return
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
