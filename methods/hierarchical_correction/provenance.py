"""Source fingerprints for staged hierarchical-correction experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


SOURCE_GLOBS = (
    "methods/hierarchical_correction/*.py",
    "methods/latentloop/modules/*.py",
    "methods/latentloop/eval/*.py",
    "architectures/simvla/adapters/hierarchical_correction/*.py",
    "architectures/simvla/upstream/models/*.py",
    "architectures/simvla/upstream/evaluation/libero/*.py",
    "architectures/simvla/upstream/evaluation/libero/LIBERO/libero/libero/benchmark/**/*.py",
    "architectures/simvla/upstream/evaluation/libero/LIBERO/libero/libero/envs/**/*.py",
    "architectures/simvla/upstream/evaluation/libero/LIBERO/libero/libero/bddl_files/libero_10/*.bddl",
    "architectures/simvla/upstream/evaluation/libero/LIBERO/libero/libero/init_files/libero_10/*.pruned_init",
)

SOURCE_FILES = (
    "architectures/simvla/adapters/latentloop/action_adapter.py",
    "architectures/simvla/adapters/latentloop/checkpoint.py",
    "architectures/simvla/adapters/latentloop/condition_adapter.py",
    "architectures/simvla/adapters/latentloop/query_cache_state.py",
    "architectures/simvla/adapters/latentloop/simvla_policy.py",
    "architectures/simvla/adapters/latentloop/source_lock.py",
    "architectures/simvla/wrappers/dcld_eval/rollout_runner.py",
    "architectures/simvla/wrappers/eval_hierarchical_correction.sh",
    "methods/latentloop/training/query_cache_dataset.py",
    "tools/simvla/analyze_hierarchical_correction.py",
    "tools/simvla/analyze_native_horizon_results.py",
    "tools/simvla/audit_horizon_provenance.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hierarchical_source_manifest(root: str | Path) -> dict[str, Any]:
    """Hash the opt-in implementation and every local source dependency it reuses."""

    root_path = Path(root).expanduser().resolve()
    candidates = {root_path / relative for relative in SOURCE_FILES}
    for pattern in SOURCE_GLOBS:
        candidates.update(root_path.glob(pattern))
    files: dict[str, str] = {}
    missing: list[str] = []
    for path in sorted(candidates):
        relative = path.relative_to(root_path).as_posix()
        if path.is_file():
            files[relative] = _sha256(path)
        else:
            missing.append(relative)
    digest = hashlib.sha256()
    for relative, file_hash in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "schema_version": "hierarchical_correction_source_manifest_v1",
        "combined_sha256": digest.hexdigest(),
        "files": files,
        "missing": missing,
    }


def experiment_source_signature(source_lock: Mapping[str, Any]) -> dict[str, Any]:
    """Select immutable fields that must agree across staged artifacts."""

    checkpoint = source_lock.get("checkpoint", {})
    processor = source_lock.get("processor_checkpoint", {})
    implementation = source_lock.get("hierarchical_implementation", {})
    return {
        "root_commit": source_lock.get("root_commit"),
        "simvla_upstream_commit": source_lock.get("simvla_upstream_commit"),
        "norm_stats_sha256": source_lock.get("norm_stats_sha256"),
        "checkpoint_revision": checkpoint.get("revision"),
        "checkpoint_blob": checkpoint.get("hf_blob_key_sha256"),
        "checkpoint_identifier": checkpoint.get("identifier"),
        "processor_revision": processor.get("revision"),
        "processor_blob": processor.get("hf_blob_key_sha256"),
        "processor_identifier": processor.get("identifier"),
        "hierarchical_checkpoints": source_lock.get("hierarchical_checkpoints"),
        "hierarchical_implementation_sha256": implementation.get("combined_sha256"),
        "packages": source_lock.get("packages"),
        "python": source_lock.get("python"),
        "python_executable": source_lock.get("python_executable"),
        "conda_env": source_lock.get("conda_env"),
        "torch": source_lock.get("torch"),
        "torch_cuda": source_lock.get("torch_cuda"),
    }
