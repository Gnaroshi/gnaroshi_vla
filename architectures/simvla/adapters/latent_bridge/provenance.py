"""Pinned-source validation for the official Latent Bridge implementation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_REPOSITORY = "https://github.com/1999Lyd/Latent-Bridge.git"
OFFICIAL_COMMIT = "ed556014aa96bae8ed85768194f02360389b9365"
OFFICIAL_FILE_SHA256 = {
    "README.md": "42e1268933fa65be5999e08b2968c64bb8b8e55cff640907855952bc96b93c36",
    "qcvla/model/rectified_flow_bridge.py": "df19ff70722347c071fd08244875867d9b8110b76735e72ee50053fbd3351da7",
    "scripts/groot/collect_dagger_bridge_data.py": "28652483f2b4d4f12b3279035229adf81092ce89cee34867f651fb255b725037",
    "scripts/groot/eval_stable_dynamic_bridge.py": "6b1a38a939428a4907a295af0123eef0f35c6480de23f68b853864b99b690640",
    "scripts/groot/run_pipeline.sh": "83528d1125a251d80b224e294d95e554caceeee7248079a926d1e32d43fc0636",
    "scripts/groot/train_single_step_dit.py": "9c4aa67f61b1f94afe4d56c059a24be755157b3050bab0f6eeb774bbf938b5a7",
}
INTEGRATION_FILES = (
    "architectures/simvla/adapters/latent_bridge/checkpoint.py",
    "architectures/simvla/adapters/latent_bridge/condition_hook.py",
    "architectures/simvla/adapters/latent_bridge/collect_sync.py",
    "architectures/simvla/adapters/latent_bridge/dataset.py",
    "architectures/simvla/adapters/latent_bridge/eval.py",
    "architectures/simvla/adapters/latent_bridge/model.py",
    "architectures/simvla/adapters/latent_bridge/policy.py",
    "architectures/simvla/adapters/latent_bridge/prepare_cache.py",
    "architectures/simvla/adapters/latent_bridge/provenance.py",
    "architectures/simvla/adapters/latent_bridge/recipe.py",
    "architectures/simvla/adapters/latent_bridge/summarize.py",
    "architectures/simvla/adapters/latent_bridge/train.py",
    "architectures/simvla/wrappers/simvla_latent_bridge_collect_sync.sh",
    "architectures/simvla/wrappers/simvla_latent_bridge_eval.sh",
    "architectures/simvla/wrappers/simvla_latent_bridge_paper_pipeline.sh",
    "architectures/simvla/wrappers/simvla_latent_bridge_train.sh",
)
SIMVLA_CONTRACT_FILES = (
    "models/modeling_smolvlm_vla.py",
    "models/processing_smolvlm_vla.py",
)


def resolve_upstream_root(path: str | Path | None = None) -> Path:
    configured = path or os.environ.get("LATENT_BRIDGE_UPSTREAM_ROOT")
    root = (
        Path(configured).expanduser()
        if configured
        else ROOT / "architectures" / "latent_bridge" / "upstream"
    ).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"official Latent Bridge clone not found: {root}; "
            "see architectures/latent_bridge/README.md"
        )
    return root


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def latent_bridge_source_manifest(
    path: str | Path | None = None,
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    root = resolve_upstream_root(path)
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--short")
    file_hashes = {
        relative: sha256_file(root / relative) for relative in OFFICIAL_FILE_SHA256
    }
    checks = {
        "commit_matches": commit == OFFICIAL_COMMIT,
        "working_tree_clean": not status,
        "file_hashes_match": file_hashes == OFFICIAL_FILE_SHA256,
    }
    if not checks["commit_matches"]:
        raise RuntimeError(f"Latent Bridge commit mismatch: {commit} != {OFFICIAL_COMMIT}")
    if not checks["file_hashes_match"]:
        mismatches = {
            name: {"expected": OFFICIAL_FILE_SHA256[name], "observed": file_hashes[name]}
            for name in file_hashes
            if file_hashes[name] != OFFICIAL_FILE_SHA256[name]
        }
        raise RuntimeError(f"official Latent Bridge source hash mismatch: {mismatches}")
    if require_clean and not checks["working_tree_clean"]:
        raise RuntimeError(f"official Latent Bridge clone is dirty:\n{status}")
    scientific = {
        "repository": OFFICIAL_REPOSITORY,
        "root": str(root),
        "commit": commit,
        "status_short": status,
        "file_sha256": file_hashes,
        "checks": checks,
        "integration_label": "official-algorithm SimVLA adaptation",
        "official_simvla_implementation": False,
    }
    portable_identity = {
        "repository": OFFICIAL_REPOSITORY,
        "commit": commit,
        "file_sha256": file_hashes,
        "integration_label": scientific["integration_label"],
        "official_simvla_implementation": False,
    }
    canonical = json.dumps(
        portable_identity, sort_keys=True, separators=(",", ":")
    ).encode()
    scientific["combined_sha256"] = hashlib.sha256(canonical).hexdigest()
    return scientific


def simvla_latent_bridge_integration_manifest() -> dict[str, Any]:
    """Hash the adaptation and the frozen SimVLA interfaces it depends on."""

    missing = [relative for relative in INTEGRATION_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Latent Bridge integration files are missing: {missing}")
    upstream = Path(
        os.environ.get("SIMVLA_UPSTREAM_ROOT", ROOT / "architectures/simvla/upstream")
    ).expanduser().resolve()
    missing_upstream = [
        relative for relative in SIMVLA_CONTRACT_FILES if not (upstream / relative).is_file()
    ]
    if missing_upstream:
        raise FileNotFoundError(
            f"SimVLA contract files are missing under {upstream}: {missing_upstream}"
        )
    integration_hashes = {
        relative: sha256_file(ROOT / relative) for relative in INTEGRATION_FILES
    }
    simvla_hashes = {
        relative: sha256_file(upstream / relative) for relative in SIMVLA_CONTRACT_FILES
    }
    identity = {
        "integration_file_sha256": integration_hashes,
        "simvla_upstream_commit": _git(upstream, "rev-parse", "HEAD"),
        "simvla_contract_file_sha256": simvla_hashes,
        "official_latent_bridge_sha256": latent_bridge_source_manifest()[
            "combined_sha256"
        ],
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {
        **identity,
        "root": str(ROOT),
        "simvla_upstream_root": str(upstream),
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
    }


@lru_cache(maxsize=4)
def load_official_bridge_core(path: str | None = None) -> ModuleType:
    """Load only the official DiT core after commit and file-hash validation."""

    root = resolve_upstream_root(path)
    latent_bridge_source_manifest(root)
    source = root / "qcvla" / "model" / "rectified_flow_bridge.py"
    module_name = f"_gnaroshi_latent_bridge_core_{OFFICIAL_COMMIT[:12]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official Latent Bridge core: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for symbol in ("DiTCrossBlock", "DiTFinalLayer"):
        if not hasattr(module, symbol):
            raise ImportError(f"official core is missing {symbol}")
    return module
