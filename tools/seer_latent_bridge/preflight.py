#!/usr/bin/env python3
"""Fail-closed provenance, shape, parameter, and recipe preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from architectures.seer.adapters.latent_bridge.bridge import (
    SeerFeatureBridge,
    SeerFeatureBridgeConfig,
)
from architectures.seer.adapters.latent_bridge.layout import SeerTokenLayout
from architectures.seer.adapters.latent_bridge.provenance import (
    sha256_file,
    verify_official_source,
)
from architectures.seer.adapters.latent_bridge.runtime import (
    SeerRuntimeSpec,
    validate_runtime_spec,
)
from methods.latent_bridge import TrainingContract


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--official-source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vit-checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--libero-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--visible-devices", default="4,5,6,7")
    return parser.parse_args()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {output}")
    upstream_diff = git(repo, "status", "--short", "--", "architectures/seer/upstream")
    if upstream_diff:
        raise RuntimeError(f"Seer upstream has worktree modifications:\n{upstream_diff}")
    runtime = SeerRuntimeSpec(
        checkpoint=args.checkpoint,
        vit_checkpoint=args.vit_checkpoint,
        dataset_root=args.dataset_root,
        libero_path=args.libero_path,
    )
    runtime_provenance = validate_runtime_spec(runtime)
    official = verify_official_source(args.official_source)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model_state_dict"]
    key = "module.obs_tokens" if "module.obs_tokens" in state else "obs_tokens"
    if key not in state or tuple(state[key].shape) != (1, 1, 18, 384):
        raise RuntimeError("public33 observation-token contract changed")

    # These values were independently verified from the public checkpoint and
    # real-batch hook; preflight makes accidental CLI drift fail closed.
    layout = SeerTokenLayout(
        sequence_length=7,
        resampler_queries_per_camera=6,
        action_tokens=3,
        observation_prediction_tokens=18,
    )
    if layout.tokens_per_timestep != 37 or layout.flattened_tokens != 259:
        raise AssertionError("canonical Seer token layout is inconsistent")
    parameters = {}
    for preset in ("full", "small"):
        config = SeerFeatureBridgeConfig.from_preset(
            preset,
            stable_seq_len=layout.primary_tokens + layout.wrist_tokens,
            stable_layer="block_00",
            stable_token_group="visual",
        )
        parameters[preset] = SeerFeatureBridge(config).parameter_audit()
    recipes = {
        "R0": TrainingContract("R0", 200, 3e-4, 2, 4, 8),
        "R1": TrainingContract("R1", 100, 3e-5, 2, 4, 8),
    }
    for recipe in recipes.values():
        recipe.validate()
    payload = {
        "status": "SEER_LATENT_BRIDGE_PREFLIGHT_PASS",
        "repo": {
            "path": str(repo),
            "branch": git(repo, "branch", "--show-current"),
            "commit": git(repo, "rev-parse", "HEAD"),
            "dirty_state": git(repo, "status", "--short"),
            "seer_upstream_worktree_diff": "",
        },
        "official_source": official,
        "runtime": runtime.to_dict(),
        "runtime_provenance": runtime_provenance,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_tensor_count": len(state),
        "token_layout": layout.to_dict(),
        "bridge_parameter_audit": parameters,
        "training_contracts": {
            name: {**recipe.__dict__, "effective_batch": recipe.effective_batch}
            for name, recipe in recipes.items()
        },
        "allowed_visible_devices": args.visible_devices,
        "source_files": {
            str(path.relative_to(repo)): sha256_file(path)
            for path in sorted((repo / "architectures/seer/adapters/latent_bridge").glob("*.py"))
        },
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
