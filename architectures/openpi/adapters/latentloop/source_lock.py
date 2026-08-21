"""Immutable local pi0.5 baseline and evaluation provenance."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "openpi" / "upstream"
EXPECTED_UPSTREAM_COMMIT = "813c9b19f8a5058f71de8b462ae69d08bf60d8e1"
BASELINE_LABEL = (
    "pi0.5 base -> local OpenPI PR-854 PyTorch LoRA LIBERO baseline, "
    "4x RTX 3090, train seed 42, eval seed 7"
)
EXPECTED_CHECKPOINT_SHA256 = "2c2183dc53a8493e3fcde3ee2e41aa593b027977a456719312ab96d7127e7ff1"

SOURCE_FILES = (
    "src/openpi/models_pytorch/pi0_pytorch.py",
    "src/openpi/models_pytorch/gemma_pytorch.py",
    "src/openpi/models_pytorch/preprocessing_pytorch.py",
    "src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py",
    "src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py",
    "src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py",
    "src/openpi/policies/policy.py",
    "src/openpi/policies/policy_config.py",
    "src/openpi/policies/libero_policy.py",
    "src/openpi/transforms.py",
    "src/openpi/training/config.py",
    "examples/libero/main.py",
    "uv.lock",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_source_manifest(upstream: Path = UPSTREAM) -> dict[str, Any]:
    files: dict[str, str] = {}
    digest = hashlib.sha256()
    for relative in SOURCE_FILES:
        path = upstream / relative
        file_hash = sha256_file(path)
        files[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {"combined_sha256": digest.hexdigest(), "files": files}


def _command(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"ERROR: {exc}"


def _upstream_git_identity() -> dict[str, Any]:
    top_level = _command(["git", "rev-parse", "--show-toplevel"], UPSTREAM)
    dedicated = False
    if not top_level.startswith("ERROR:"):
        dedicated = Path(top_level).resolve() == UPSTREAM.resolve()
    if dedicated:
        return {
            "dedicated_git_repository": True,
            "git_top_level": top_level,
            "observed_commit": _command(["git", "rev-parse", "HEAD"], UPSTREAM),
            "status_short": _command(["git", "status", "--short"], UPSTREAM),
        }
    return {
        "dedicated_git_repository": False,
        "git_top_level": top_level,
        "observed_commit": None,
        "status_short": None,
        "commit_observation_limit": (
            "The vendored upstream .git directory is intentionally excluded; commit basis is provenance, "
            "and the combined per-file SHA256 manifest is the executable source lock."
        ),
    }


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def evaluation_manifest(evaluation_root: str | Path) -> dict[str, Any]:
    root = Path(evaluation_root).resolve()
    suites: dict[str, Any] = {}
    total_rows = 0
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        outcomes = root / suite / "episode_outcomes.csv"
        rows = list(csv.DictReader(outcomes.open(encoding="utf-8")))
        tasks: list[str] = []
        for row in rows:
            if row["task"] not in tasks:
                tasks.append(row["task"])
        suites[suite] = {
            "episode_outcomes_path": str(outcomes),
            "episode_outcomes_sha256": sha256_file(outcomes),
            "rows": len(rows),
            "tasks": tasks,
            "episode_key": ["suite", "task", "trial"],
            "trial_range": [0, 49],
        }
        total_rows += len(rows)
    combined = root / "combined_summary.json"
    return {
        "root": str(root),
        "combined_summary_sha256": sha256_file(combined),
        "total_episode_rows": total_rows,
        "suites": suites,
    }


def collect_source_lock(
    *,
    checkpoint_dir: str | Path,
    norm_stats_path: str | Path,
    evaluation_root: str | Path,
    hash_checkpoint: bool = True,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint = checkpoint_dir / "model.safetensors"
    checkpoint_config = checkpoint_dir / "config.json"
    if not checkpoint_config.exists():
        checkpoint_config = UPSTREAM / "checkpoints" / "pi05_base_pytorch" / "config.json"
    norm_stats = Path(norm_stats_path).resolve()
    source = combined_source_manifest()
    checkpoint_hash = sha256_file(checkpoint) if hash_checkpoint else EXPECTED_CHECKPOINT_SHA256
    checkpoint_hash_matches_expected = checkpoint_hash == EXPECTED_CHECKPOINT_SHA256
    evaluation = evaluation_manifest(evaluation_root)
    source_lock_pass = bool(
        hash_checkpoint
        and checkpoint_hash_matches_expected
        and evaluation["total_episode_rows"] == 2000
        and len(source["files"]) == len(SOURCE_FILES)
    )
    return {
        "schema_version": 1,
        "source_lock_pass": source_lock_pass,
        "baseline_label": BASELINE_LABEL,
        "baseline_result": {
            "libero_spatial": 0.974,
            "libero_object": 0.994,
            "libero_goal": 0.972,
            "libero_10": 0.948,
            "four_suite_average": 0.972,
            "successes": 1944,
            "episodes": 2000,
        },
        "repository": {
            "root": str(ROOT),
            "branch": _command(["git", "branch", "--show-current"], ROOT),
            "head": _command(["git", "rev-parse", "HEAD"], ROOT),
        },
        "openpi_source": {
            "root": str(UPSTREAM),
            "branch_basis": "pr-854",
            "expected_commit": EXPECTED_UPSTREAM_COMMIT,
            **_upstream_git_identity(),
            **source,
        },
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "model_path": str(checkpoint),
            "model_size_bytes": checkpoint.stat().st_size,
            "model_sha256": checkpoint_hash,
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
            "hash_computed_in_this_run": bool(hash_checkpoint),
            "hash_verified": bool(hash_checkpoint and checkpoint_hash_matches_expected),
            "recorded_hash_matches_expected": checkpoint_hash_matches_expected,
            "config_path": str(checkpoint_config),
            "config_sha256": sha256_file(checkpoint_config),
        },
        "normalization": {
            "path": str(norm_stats),
            "sha256": sha256_file(norm_stats),
        },
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": {
                name: _version(name)
                for name in ("torch", "transformers", "numpy", "wandb", "lerobot")
            },
            "baseline_main_environment": {
                "python": "3.11.15",
                "torch": "2.7.1+cu126",
                "transformers": "4.53.2",
                "wandb": "0.19.11",
                "lerobot": "0.1.0",
            },
            "baseline_libero_client_environment": {
                "python": "3.8.20",
                "torch": "1.11.0+cu113",
                "numpy": "1.22.4",
                "mujoco": "3.2.3",
                "robosuite": "1.4.1",
            },
        },
        "native_intervals": {
            "action_prediction_horizon_h": 10,
            "execution_horizon_r": 5,
            "target_k_q": 4,
            "target_k_a": 20,
        },
        "evaluation_manifest": evaluation,
        "policy_noise": {
            "baseline_seed": 7,
            "construction": "torch.normal inside PI0Pytorch.sample_noise from one process-seeded global RNG stream",
            "per_query_keys_persisted": False,
            "pairing_limit": (
                "The completed baseline preserves episode keys but did not persist per-query noise keys. "
                "New method rows use explicit hash-derived noise keys; exact action-noise pairing to the "
                "completed baseline cannot be claimed without rerunning a full control row."
            ),
        },
        "preprocessing": {
            "client": "rotate camera images 180 degrees, resize_with_pad to 224, convert_to_uint8",
            "policy_inputs": "LiberoInputs -> Normalize -> model transforms -> preprocess_observation_pytorch(train=False)",
            "state": "8-D LIBERO state normalized and padded to model action/state dimension",
            "source_files": [
                "examples/libero/main.py",
                "src/openpi/policies/libero_policy.py",
                "src/openpi/policies/policy_config.py",
                "src/openpi/models_pytorch/preprocessing_pytorch.py",
            ],
        },
        "postprocessing": {
            "path": "model transforms outputs -> Unnormalize -> LiberoOutputs",
            "returned_action_dim": 7,
            "native_action_chunk": 10,
            "executed_prefix": 5,
        },
    }


def write_source_lock(output_dir: str | Path, **kwargs: Any) -> tuple[Path, Path]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "openpi_pi05_ours_source_lock.json"
    markdown_path = output / "openpi_pi05_ours_source_lock.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(f"refusing to overwrite source lock under {output}")
    payload = collect_source_lock(**kwargs)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint = payload["checkpoint"]
    native = payload["native_intervals"]
    markdown_path.write_text(
        "\n".join(
            (
                "# OpenPI pi0.5 Ours source lock",
                "",
                f"- Baseline: `{payload['baseline_label']}`",
                f"- Source lock gate: `{payload['source_lock_pass']}`",
                f"- OpenPI commit basis: `{payload['openpi_source']['expected_commit']}`",
                f"- Combined source SHA256: `{payload['openpi_source']['combined_sha256']}`",
                f"- Checkpoint SHA256: `{checkpoint['model_sha256']}`",
                f"- Checkpoint hash computed in this run: `{checkpoint['hash_computed_in_this_run']}`",
                f"- Checkpoint hash verified: `{checkpoint['hash_verified']}`",
                f"- Norm-stats SHA256: `{payload['normalization']['sha256']}`",
                f"- H: `{native['action_prediction_horizon_h']}`",
                f"- R: `{native['execution_horizon_r']}`",
                f"- Target K_q: `{native['target_k_q']}`",
                f"- Target K_a: `{native['target_k_a']}`",
                "",
                "The completed baseline uses a process-seeded sequential policy-noise stream. "
                "Its episode keys are reusable, but exact per-query noise pairing cannot be retroactively claimed.",
                "",
                f"Machine-readable lock: `{json_path}`",
                "",
            )
        ),
        encoding="utf-8",
    )
    return markdown_path, json_path
