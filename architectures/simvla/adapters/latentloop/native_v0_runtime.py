"""Shared source-lock, gate, and frozen-runtime utilities for native SimVLA V0."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
for _path in (ROOT, UPSTREAM):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter  # noqa: E402
from architectures.simvla.adapters.latentloop.native_v0_condition_hook import (  # noqa: E402
    ConditionTokenLayout,
    build_condition_token_layout,
)
from architectures.simvla.adapters.latentloop.source_lock import (  # noqa: E402
    collect_source_lock,
    sha256_file,
)


DEFAULT_CHECKPOINT = "YuankaiLuo/SimVLA-LIBERO"
DEFAULT_SMOLVLM = "HuggingFaceTB/SmolVLM-500M-Instruct"


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def require_new_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    return output


def require_gate(
    path: str | Path,
    *,
    verdicts: Sequence[str],
    source_combined_sha256: str | None = None,
) -> dict[str, Any]:
    gate_path = Path(path).expanduser().resolve()
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if str(payload.get("verdict")) not in set(verdicts):
        raise RuntimeError(
            f"gate {gate_path} verdict={payload.get('verdict')!r}, expected one of {list(verdicts)}"
        )
    if source_combined_sha256 is not None:
        observed = payload.get("source_combined_sha256")
        if observed != source_combined_sha256:
            raise RuntimeError(
                f"gate source hash mismatch: {observed} != {source_combined_sha256}"
            )
    return payload


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def _critical_source_files() -> list[Path]:
    patterns = (
        "methods/latentloop/modules/native_simvla_v0.py",
        "methods/latentloop/training/native_simvla_v0.py",
        "architectures/simvla/adapters/latentloop/native_v0*.py",
        "architectures/simvla/adapters/latentloop/assets/*.json",
        "architectures/simvla/adapters/latentloop/action_adapter.py",
        "architectures/simvla/adapters/latentloop/source_lock.py",
        "architectures/simvla/adapters/dcld/simvla_action_adapter.py",
        "architectures/simvla/adapters/dcld/simvla_condition_adapter.py",
        "architectures/simvla/adapters/dcld/simvla_delta_obs_adapter.py",
        "architectures/simvla/wrappers/dcld_eval/rollout_runner.py",
        "architectures/simvla/wrappers/simvla_native_v0*.sh",
        "architectures/simvla/wrappers/simvla_two_gpu_guard.py",
        "methods/latentloop/training/query_cache_dataset.py",
        "architectures/simvla/upstream/models/modeling_smolvlm_vla.py",
        "architectures/simvla/upstream/models/transformer_smolvlm.py",
        "architectures/simvla/upstream/models/processing_smolvlm_vla.py",
    )
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(files)


def native_v0_source_manifest(
    *,
    checkpoint: str,
    norm_stats: str | Path,
    cache: str | Path | None,
) -> dict[str, Any]:
    norm_path = Path(norm_stats).expanduser().resolve()
    base = collect_source_lock(checkpoint=checkpoint, norm_stats_path=norm_path)
    file_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in _critical_source_files()
    }
    scientific = {
        "checkpoint": base["checkpoint"],
        "norm_stats_path": str(norm_path),
        "norm_stats_sha256": sha256_file(norm_path),
        "cache_manifest_path": None,
        "cache_manifest_sha256": None,
        "cache_generation_norm_stats_sha256": None,
        "cache_norm_matches_runtime_norm": None,
        "cached_teacher_actions_used_in_objective": False,
        "critical_file_sha256": file_hashes,
        "simvla_upstream_commit": _git(["git", "rev-parse", "HEAD"], UPSTREAM),
        "libero_root": str(UPSTREAM / "evaluation" / "libero" / "LIBERO"),
        "libero_commit": _git(
            ["git", "rev-parse", "HEAD"],
            UPSTREAM / "evaluation" / "libero" / "LIBERO",
        ),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "mujoco": _package_version("mujoco"),
            "robosuite": _package_version("robosuite"),
            "transformers": _package_version("transformers"),
            "numpy": _package_version("numpy"),
        },
        "selected_physical_gpu_ids": [
            int(value) for value in os.environ.get("SIMVLA_GPU_IDS", "").split(",") if value
        ],
    }
    if cache is not None:
        manifest = Path(cache).expanduser().resolve() / "manifest.json"
        scientific["cache_manifest_path"] = str(manifest)
        scientific["cache_manifest_sha256"] = sha256_file(manifest)
        cache_payload = json.loads(manifest.read_text(encoding="utf-8"))
        cache_norm_hash = (
            cache_payload.get("metadata", {})
            .get("source_lock", {})
            .get("norm_stats_sha256")
        )
        scientific["cache_generation_norm_stats_sha256"] = cache_norm_hash
        scientific["cache_norm_matches_runtime_norm"] = (
            cache_norm_hash == scientific["norm_stats_sha256"]
        )
    canonical = json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode("utf-8")
    scientific["combined_sha256"] = hashlib.sha256(canonical).hexdigest()
    scientific["complete_source_lock"] = base
    return scientific


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def configure_strict_torch_determinism(seed: int) -> dict[str, Any]:
    """Apply the existing SimVLA deterministic CUDA contract in-process."""

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "seed": int(seed),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_debug_mode": int(torch.get_deterministic_debug_mode()),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def load_frozen_simvla(
    *,
    checkpoint: str,
    norm_stats: str | Path,
    smolvlm_model: str,
    device: torch.device,
) -> tuple[Any, Any, SimVLAActionAdapter]:
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    model = SmolVLMVLA.from_pretrained(checkpoint).to(device).eval()
    model.action_space.load_norm_stats(str(norm_stats))
    freeze_module(model)
    processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_model)
    return model, processor, SimVLAActionAdapter(model)


def cached_batch_token_layout(
    *,
    condition: torch.Tensor,
    language_instructions: Sequence[str],
    processor: Any,
    num_views: int = 2,
) -> ConditionTokenLayout:
    input_ids = processor.encode_language(list(language_instructions))["input_ids"]
    image_mask = torch.zeros(
        (len(language_instructions), int(getattr(processor, "num_views", 3))),
        dtype=torch.bool,
    )
    image_mask[:, :num_views] = True
    tokenizer = processor.tokenizer
    return build_condition_token_layout(
        condition=condition,
        image_mask=image_mask,
        input_ids=input_ids,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        special_token_ids=getattr(tokenizer, "all_special_ids", ()),
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
