#!/usr/bin/env python3
"""Verify cacheless V0 evaluation while allowing additive checkout drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import torch

from architectures.openpi.adapters.latentloop.cache_contract_v2 import (
    SUITES,
    load_final_evaluation_manifest,
)
from source_lock_v2 import environment_identity, git_identity


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "architectures" / "openpi" / "upstream"
EVALUATION_FILES = (
    "architectures/openpi/adapters/latentloop/online_policy.py",
    "architectures/openpi/adapters/latentloop/policy_io.py",
    "architectures/openpi/adapters/latentloop/prefix_kv_hook.py",
    "architectures/openpi/adapters/latentloop/recurrent_policy.py",
    "architectures/openpi/adapters/latentloop/serialization.py",
    "architectures/openpi/adapters/latentloop/transition_core.py",
    "architectures/openpi/wrappers/eval_pi05_v0_streaming_suite.sh",
    "architectures/openpi/wrappers/run_pi05_v0_streaming_eval_4gpu.sh",
    "tools/openpi/aggregate_pi05_v0_streaming_eval.py",
    "tools/openpi/evaluate_pi05_latentloop_client.py",
    "tools/openpi/serve_pi05_v0_streaming.py",
    "tools/openpi/verify_pi05_v0_streaming_eval.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_value(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def git_is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repository,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def verify_git_identity(expected: dict[str, Any], repository: Path, label: str) -> None:
    observed = git_identity(repository)
    comparable = {
        key: value for key, value in expected.items() if key != "expected_head"
    }
    if observed != comparable:
        raise RuntimeError(f"{label} git identity changed since V0 training")


def verify_source_manifest(expected: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, str] = {}
    mismatches = []
    for relative, expected_hash in expected["files"].items():
        path = ROOT / relative
        if not path.is_file():
            mismatches.append({"path": relative, "reason": "missing"})
            continue
        actual_hash = sha256_file(path)
        observed[relative] = actual_hash
        if actual_hash != expected_hash:
            mismatches.append(
                {
                    "path": relative,
                    "reason": "hash",
                    "expected": expected_hash,
                    "actual": actual_hash,
                }
            )
    if mismatches:
        raise RuntimeError(
            f"source-locked V0 files changed: {json.dumps(mismatches, sort_keys=True)}"
        )
    if canonical_hash(observed) != expected["combined_sha256"]:
        raise RuntimeError("source-locked V0 combined hash changed")

    expected_names = set(expected["files"])
    additive = set()
    for relative_root in (
        "methods/variable_time_latentloop",
        "architectures/openpi/adapters/latentloop",
        "architectures/openpi/wrappers",
        "tools/openpi",
    ):
        for path in (ROOT / relative_root).rglob("*"):
            if (
                path.is_file()
                and path.suffix in {".py", ".sh"}
                and "__pycache__" not in path.parts
            ):
                relative = str(path.relative_to(ROOT))
                if relative not in expected_names:
                    additive.add(relative)
    return {
        "verification_mode": "locked_file_subset_exact_additive_files_allowed",
        "locked_file_count": len(observed),
        "locked_combined_sha256": expected["combined_sha256"],
        "additive_source_file_count": len(additive),
        "additive_source_files": sorted(additive),
    }


def verify_named_manifest(lock: dict[str, Any], key: str) -> dict[str, Any]:
    expected = lock[key]
    observed = {
        relative: sha256_file(ROOT / relative) for relative in expected["files"]
    }
    if (
        observed != expected["files"]
        or canonical_hash(observed) != expected["combined_sha256"]
    ):
        raise RuntimeError(f"{key} manifest changed since V0 training")
    return {"file_count": len(observed), "combined_sha256": expected["combined_sha256"]}


def verify_source_lock(
    source_lock_path: str | Path,
    checkpoint: str | Path,
    norm_stats: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(source_lock_path).resolve()
    lock = load_json(path)
    if lock.get("schema_version") != 2 or lock.get("source_lock_v2_pass") is not True:
        raise RuntimeError("training source lock is not completed schema v2")
    id_payload = {
        key: value
        for key, value in lock.items()
        if key not in {"source_lock_id", "source_lock_v2_pass"}
    }
    if canonical_hash(id_payload) != lock.get("source_lock_id"):
        raise RuntimeError("training source lock self-hash is invalid")

    repository = lock["repository"]
    if Path(repository["root"]).resolve() != ROOT:
        raise RuntimeError("training source lock belongs to another repository")
    current_head = git_value(ROOT, "rev-parse", "HEAD")
    if not git_is_ancestor(ROOT, repository["head"], current_head):
        raise RuntimeError(
            "V0 training HEAD is not an ancestor of the evaluation checkout"
        )
    current_branch = git_value(ROOT, "branch", "--show-current")

    source_subset = verify_source_manifest(lock["ours_and_upstream_source"])
    preprocessing = verify_named_manifest(lock, "preprocessing")
    postprocessing = verify_named_manifest(lock, "postprocessing")
    verify_git_identity(lock["upstream"], UPSTREAM, "OpenPI upstream")
    verify_git_identity(
        lock["nested_libero"], UPSTREAM / "third_party/libero", "vendored LIBERO"
    )
    if environment_identity() != lock["environment"]:
        raise RuntimeError("runtime environment differs from the V0 training lock")

    checkpoint = Path(checkpoint).resolve()
    norm_stats = Path(norm_stats).resolve()
    if checkpoint != Path(lock["checkpoint"]["directory"]).resolve():
        raise RuntimeError(
            "base checkpoint directory differs from the V0 training lock"
        )
    if (
        sha256_file(checkpoint / "model.safetensors")
        != lock["checkpoint"]["model_sha256"]
    ):
        raise RuntimeError("base checkpoint model hash changed")
    checkpoint_config = Path(lock["checkpoint"]["config_path"]).resolve()
    if not checkpoint_config.is_file():
        raise RuntimeError("source-locked base checkpoint config is missing")
    if sha256_file(checkpoint_config) != lock["checkpoint"]["config_sha256"]:
        raise RuntimeError("base checkpoint config hash changed")
    if norm_stats != Path(lock["normalization"]["path"]).resolve():
        raise RuntimeError("normalization path differs from the V0 training lock")
    if sha256_file(norm_stats) != lock["normalization"]["sha256"]:
        raise RuntimeError("normalization hash changed")

    return lock, {
        "TRAINING_SOURCE_SUBSET_PASS": True,
        "source_lock": str(path),
        "source_lock_id": lock["source_lock_id"],
        "training_repository_head": repository["head"],
        "evaluation_repository_head": current_head,
        "training_repository_branch": repository["branch"],
        "evaluation_repository_branch": current_branch,
        "repository_dirty_identity_policy": (
            "retain the original dirty-tree identity as provenance; allow current checkout drift only when "
            "every source-locked file remains byte-identical"
        ),
        "source_subset": source_subset,
        "preprocessing": preprocessing,
        "postprocessing": postprocessing,
        "base_checkpoint_model_sha256": lock["checkpoint"]["model_sha256"],
        "normalization_sha256": lock["normalization"]["sha256"],
        "environment": lock["environment"],
    }


def require_gate(
    path: str | Path, source_lock_id: str, markers: tuple[str, ...]
) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = load_json(path)
    if payload.get("source_lock_id") != source_lock_id:
        raise RuntimeError(f"gate belongs to another source lock: {path}")
    observed = {str(value) for value in payload.get("markers", [])}
    observed.update(
        key for key, value in payload.items() if key.isupper() and value is True
    )
    missing = sorted(set(markers) - observed)
    if missing:
        raise RuntimeError(f"gate {path} is missing markers {missing}")
    return {"path": str(path), "markers": sorted(observed), "sha256": sha256_file(path)}


def verify_manifest(
    path: str | Path,
    source_lock_id: str,
    suite: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    manifest = load_final_evaluation_manifest(path)
    if manifest.get("source_lock_id") != source_lock_id:
        raise RuntimeError("final evaluation manifest belongs to another source lock")
    required = {
        "action_horizon_h": 10,
        "execution_horizon_r": 5,
        "wait_steps": 10,
        "resize_size": 224,
        "renderer": "egl",
        "trials_per_task": 50,
        "noise_seed_base": 7,
        "policy_noise": "explicit_query_keyed_sha256_v2",
    }
    for key, expected in required.items():
        if manifest["protocol"].get(key) != expected:
            raise RuntimeError(f"final evaluation protocol mismatch for {key}")
    selected = [row for row in manifest["episodes"] if row["suite"] == suite]
    if len(selected) != 500:
        raise RuntimeError(f"suite {suite} does not contain 500 frozen episodes")
    if any(
        row.get("episode_namespace") != "final_scientific_evaluation"
        for row in selected
    ):
        raise RuntimeError("final manifest contains a non-scientific episode namespace")
    return manifest, {
        "FINAL_EVALUATION_MANIFEST_V2_PASS": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest_id": manifest["manifest_id"],
        "suite": suite,
        "selected_episodes": 500,
        "protocol": manifest["protocol"],
    }


def verify_training_run(
    run_summary_path: str | Path,
    adapter_checkpoint_path: str | Path,
    source_lock_id: str,
    manifest: dict[str, Any],
    manifest_sha256: str,
    base_checkpoint: str | Path,
) -> dict[str, Any]:
    run_summary_path = Path(run_summary_path).resolve()
    adapter_checkpoint_path = Path(adapter_checkpoint_path).resolve()
    run_dir = run_summary_path.parent
    summary = load_json(run_summary_path)
    config = load_json(run_dir / "config.json")
    if (
        summary.get("V0_TRAIN_COMPLETE") is not True
        or summary.get("complete") is not True
    ):
        raise RuntimeError("V0 streaming training did not complete")
    if summary.get("source_lock_id") != source_lock_id:
        raise RuntimeError("V0 run summary belongs to another source lock")
    if (
        summary.get("variant") != "v0"
        or summary.get("training_source_mode") != "online_frozen_teacher_rolling_v0"
    ):
        raise RuntimeError("training run is not cacheless rolling V0")
    if int(summary.get("steps", -1)) != 10_000:
        raise RuntimeError("primary V0 screen must be the completed 10,000-step run")
    persistent_bytes = summary.get("streaming_source_statistics", {}).get(
        "persistent_teacher_tensor_bytes"
    )
    if int(persistent_bytes if persistent_bytes is not None else -1) != 0:
        raise RuntimeError("V0 run unexpectedly persisted teacher tensors")
    if not adapter_checkpoint_path.is_relative_to((run_dir / "checkpoints").resolve()):
        raise RuntimeError("adapter checkpoint is not owned by the declared V0 run")
    if adapter_checkpoint_path.name != "best.pt":
        raise RuntimeError(
            "primary evaluation checkpoint must be validation-selected best.pt"
        )

    payload = torch.load(
        adapter_checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint_config = payload.get("config", {})
    provenance = checkpoint_config.get("provenance", {})
    if checkpoint_config.get("adapter_type") != "openpi_variable_time_latentloop":
        raise RuntimeError("adapter checkpoint has an unexpected type")
    if provenance.get("source_lock_id") != source_lock_id:
        raise RuntimeError("adapter checkpoint belongs to another source lock")
    if provenance.get("final_manifest_id") != manifest["manifest_id"]:
        raise RuntimeError("adapter checkpoint final-manifest id mismatch")
    if provenance.get("final_manifest_sha256") != manifest_sha256:
        raise RuntimeError("adapter checkpoint final-manifest hash mismatch")
    if (
        Path(provenance.get("teacher_checkpoint", "")).resolve()
        != Path(base_checkpoint).resolve()
    ):
        raise RuntimeError("adapter teacher/base checkpoint mismatch")
    if provenance.get("training_source_mode") != "online_frozen_teacher_rolling_v0":
        raise RuntimeError("adapter was not produced by cacheless rolling V0 training")

    best_step = int(summary["best_step"])
    if int(payload.get("step", -1)) != best_step:
        raise RuntimeError("best.pt step differs from run_summary best_step")
    best_metric = float(summary["best_validation_first_r_action_mse"])
    checkpoint_metric = float(
        payload.get("validation", {}).get("recursive_first_r", math.nan)
    )
    if not math.isclose(best_metric, checkpoint_metric, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("best.pt validation metric differs from run_summary")
    parameter_count = sum(
        int(tensor.numel()) for tensor in payload.get("adapter", {}).values()
    )
    if parameter_count != int(summary["adapter_trainable_parameters"]):
        raise RuntimeError("adapter parameter count differs from run_summary")
    if config.get("provenance", {}).get("source_lock_id") != source_lock_id:
        raise RuntimeError("training config belongs to another source lock")

    return {
        "V0_STREAMING_CHECKPOINT_PASS": True,
        "run_summary": str(run_summary_path),
        "run_summary_sha256": sha256_file(run_summary_path),
        "training_config_sha256": sha256_file(run_dir / "config.json"),
        "adapter_checkpoint": str(adapter_checkpoint_path),
        "adapter_checkpoint_sha256": sha256_file(adapter_checkpoint_path),
        "checkpoint_selection": "heldout_checkpoint_validation_minimum",
        "checkpoint_step": best_step,
        "training_steps": int(summary["steps"]),
        "validation_first_r_action_mse": best_metric,
        "adapter_trainable_parameters": parameter_count,
        "training_source_id": summary["training_source_id"],
        "training_source_mode": summary["training_source_mode"],
        "persistent_teacher_tensor_bytes": 0,
    }


def evaluation_harness_manifest() -> dict[str, Any]:
    files = {relative: sha256_file(ROOT / relative) for relative in EVALUATION_FILES}
    return {"files": files, "combined_sha256": canonical_hash(files)}


def verify_evaluation_inputs(
    *,
    source_lock: str | Path,
    checkpoint: str | Path,
    norm_stats: str | Path,
    final_manifest: str | Path,
    suite: str,
    k1_tensor_report: str | Path,
    k1_episode_gate: str | Path,
    freeze_gate: str | Path,
    training_run_summary: str | Path,
    adapter_checkpoint: str | Path,
) -> dict[str, Any]:
    if suite not in SUITES:
        raise ValueError(f"unknown LIBERO suite {suite!r}")
    lock, source_result = verify_source_lock(source_lock, checkpoint, norm_stats)
    source_lock_id = lock["source_lock_id"]
    manifest, manifest_result = verify_manifest(final_manifest, source_lock_id, suite)
    gates = {
        "k1_tensor": require_gate(
            k1_tensor_report,
            source_lock_id,
            ("REAL_KV_ROUNDTRIP_PASS", "K1_ACTION_PARITY_PASS"),
        ),
        "k1_episode": require_gate(
            k1_episode_gate,
            source_lock_id,
            (
                "REAL_KV_ROUNDTRIP_PASS",
                "K1_ACTION_PARITY_PASS",
                "K1_EPISODE_PARITY_PASS",
            ),
        ),
        "freeze": require_gate(freeze_gate, source_lock_id, ("BASE_FREEZE_PASS",)),
    }
    training = verify_training_run(
        training_run_summary,
        adapter_checkpoint,
        source_lock_id,
        manifest,
        manifest_result["sha256"],
        checkpoint,
    )
    return {
        "V0_STREAMING_EVALUATION_PREFLIGHT_PASS": True,
        "source_lock_id": source_lock_id,
        "suite": suite,
        "paired_rows": ["original", "v0"],
        "evaluation_order": ["v0", "original"],
        "source_verification": source_result,
        "manifest": manifest_result,
        "gates": gates,
        "training": training,
        "evaluation_harness": evaluation_harness_manifest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--final-manifest", required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--k1-tensor-report", required=True)
    parser.add_argument("--k1-episode-gate", required=True)
    parser.add_argument("--freeze-gate", required=True)
    parser.add_argument("--training-run-summary", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = verify_evaluation_inputs(
        source_lock=args.source_lock,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        final_manifest=args.final_manifest,
        suite=args.suite,
        k1_tensor_report=args.k1_tensor_report,
        k1_episode_gate=args.k1_episode_gate,
        freeze_gate=args.freeze_gate,
        training_run_summary=args.training_run_summary,
        adapter_checkpoint=args.adapter_checkpoint,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
