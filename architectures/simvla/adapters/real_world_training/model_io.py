"""Exact official-base loading and compact real-world checkpoint I/O."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from architectures.simvla.adapters.latentloop_real_deploy.bootstrap import (
    configure_model_imports,
)

from .io_utils import sha256_file


REAL_ACTION_CHECKPOINT_FORMAT = "simvla_real_action_transformer_v2"
OFFICIAL_SIMVLA_LIBERO_WEIGHTS_SHA256 = (
    "9d3b1767773da86906d771b1eca2c2911087371bf8b3890a7336b6773270f6be"
)


@dataclass(frozen=True)
class OfficialBaseIdentity:
    model_directory: str
    model_weights_sha256: str
    processor_directory: str
    action_mode: str
    action_horizon: int
    transformer_hidden_size: int
    transformer_depth: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _weights_path(model_directory: str | Path) -> Path:
    root = Path(model_directory).expanduser().resolve()
    direct = root / "model.safetensors"
    if direct.is_file():
        return direct
    shards = sorted(root.glob("model-*-of-*.safetensors"))
    if shards and (root / "model.safetensors.index.json").is_file():
        # A stable identity for sharded models is the index plus every shard.
        # The released SimVLA checkpoint is not sharded, but fail explicitly if
        # a future artifact changes this contract.
        raise ValueError("sharded SimVLA checkpoints are not supported by this source lock")
    raise FileNotFoundError(f"model.safetensors was not found in {root}")


def official_base_identity(
    model_directory: str | Path,
    processor_directory: str | Path,
) -> OfficialBaseIdentity:
    configure_model_imports()
    from models.configuration_smolvlm_vla import SmolVLMVLAConfig

    model_root = Path(model_directory).expanduser().resolve()
    processor_root = Path(processor_directory).expanduser().resolve()
    config = SmolVLMVLAConfig.from_pretrained(str(model_root), local_files_only=True)
    weights_sha256 = sha256_file(_weights_path(model_root))
    if weights_sha256 != OFFICIAL_SIMVLA_LIBERO_WEIGHTS_SHA256:
        raise ValueError(
            "real adaptation requires the pinned YuankaiLuo/SimVLA-LIBERO weights: "
            f"observed={weights_sha256} "
            f"expected={OFFICIAL_SIMVLA_LIBERO_WEIGHTS_SHA256}"
        )
    identity = OfficialBaseIdentity(
        model_directory=str(model_root),
        model_weights_sha256=weights_sha256,
        processor_directory=str(processor_root),
        action_mode=str(config.action_mode),
        action_horizon=int(config.num_actions),
        transformer_hidden_size=int(config.hidden_size),
        transformer_depth=int(config.depth),
    )
    expected = {
        "action_mode": "libero_joint",
        "action_horizon": 10,
        "transformer_hidden_size": 1024,
        "transformer_depth": 24,
    }
    observed = {
        key: getattr(identity, key)
        for key in expected
    }
    if observed != expected:
        raise ValueError(
            f"pinned SimVLA-LIBERO architecture contract changed: {observed} != {expected}"
        )
    return identity


def _nonempty_loading_info(info: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    return {key: list(info.get(key) or ()) for key in keys if info.get(key)}


def _load_local_processor(processor_directory: str | Path) -> Any:
    configure_model_imports()
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    root = Path(processor_directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"local processor snapshot was not found: {root}")
    # The upstream factory catches errors and substitutes a Hub model. The
    # constructor preserves preprocessing but propagates local-load failures.
    return SmolVLMVLAProcessor(smolvlm_model_path=str(root))


def load_exact_official_model(
    *,
    model_directory: str | Path,
    processor_directory: str | Path,
    norm_stats: str | Path,
    device: torch.device | str,
    real_action_checkpoint: str | Path | None = None,
    freeze_vlm: bool = True,
    freeze_action_transformer: bool = False,
    expected_dataset_identity_sha256: str | None = None,
    expected_cache_identity_sha256: str | None = None,
    expected_cache_attestation_identity_sha256: str | None = None,
    expected_real_action_optimizer_step: int | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load every released SimVLA tensor, then optionally overlay only its head.

    A non-empty Hugging Face loading report is a hard error.  This prevents an
    accidental action-head reinitialization from being mislabeled as transfer
    from ``YuankaiLuo/SimVLA-LIBERO``.
    """

    configure_model_imports()
    from models.configuration_smolvlm_vla import SmolVLMVLAConfig
    from models.modeling_smolvlm_vla import SmolVLMVLA

    model_root = Path(model_directory).expanduser().resolve()
    processor_root = Path(processor_directory).expanduser().resolve()
    config = SmolVLMVLAConfig.from_pretrained(str(model_root), local_files_only=True)
    config.smolvlm_model_path = str(processor_root)
    loaded = SmolVLMVLA.from_pretrained(
        str(model_root),
        config=config,
        local_files_only=True,
        output_loading_info=True,
    )
    model, loading_info = loaded
    failures = _nonempty_loading_info(loading_info)
    if failures:
        raise RuntimeError(f"official SimVLA checkpoint did not load exactly: {failures}")
    if str(model.action_mode) != "libero_joint" or int(model.num_actions) != 10:
        raise RuntimeError(
            "real adaptation requires the released libero_joint H=10 checkpoint"
        )
    model.action_space.load_norm_stats(str(Path(norm_stats).expanduser().resolve()))
    base = official_base_identity(model_root, processor_root)
    overlay_report: dict[str, Any] | None = None
    if real_action_checkpoint is not None:
        overlay_report = apply_real_action_checkpoint(
            model,
            real_action_checkpoint,
            expected_base_sha256=base.model_weights_sha256,
            expected_norm_sha256=sha256_file(norm_stats),
            expected_dataset_identity_sha256=expected_dataset_identity_sha256,
            expected_cache_identity_sha256=expected_cache_identity_sha256,
            expected_cache_attestation_identity_sha256=(
                expected_cache_attestation_identity_sha256
            ),
            expected_optimizer_step=expected_real_action_optimizer_step,
        )
    for parameter in model.vlm.parameters():
        parameter.requires_grad_(not freeze_vlm)
    for parameter in model.transformer.parameters():
        parameter.requires_grad_(not freeze_action_transformer)
    model.vlm.eval() if freeze_vlm else model.vlm.train()
    model.transformer.eval() if freeze_action_transformer else model.transformer.train()
    model.to(device)
    processor = _load_local_processor(processor_root)
    report = {
        "verdict": "EXACT_OFFICIAL_INITIALIZATION_PASS",
        "base": base.to_dict(),
        "loading_info": {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
        "action_transformer_reinitialized": False,
        "vlm_frozen": bool(freeze_vlm),
        "action_transformer_frozen": bool(freeze_action_transformer),
        "real_action_overlay": overlay_report,
    }
    return model, processor, report


def _atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_real_action_checkpoint(
    path: str | Path,
    *,
    transformer: torch.nn.Module,
    official_base: OfficialBaseIdentity,
    norm_stats_path: str | Path,
    dataset_identity_sha256: str,
    optimizer_step: int,
    training_config: Mapping[str, Any],
    validation: Mapping[str, Any],
    optimizer_state: Mapping[str, Any] | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
) -> Path:
    payload = {
        "checkpoint_format": REAL_ACTION_CHECKPOINT_FORMAT,
        "action_transformer_state_dict": {
            key: value.detach().cpu() for key, value in transformer.state_dict().items()
        },
        "official_base": official_base.to_dict(),
        "norm_stats_sha256": sha256_file(norm_stats_path),
        "dataset_identity_sha256": str(dataset_identity_sha256),
        "optimizer_step": int(optimizer_step),
        "training_config": dict(training_config),
        "validation": dict(validation),
        "optimizer_state_dict": optimizer_state,
        "scheduler_state_dict": scheduler_state,
        "initialization_contract": {
            "source": "complete released SimVLA-LIBERO checkpoint",
            "action_transformer_reinitialized": False,
            "vlm_frozen_during_real_adaptation": True,
        },
    }
    return _atomic_torch_save(payload, path)


def load_real_action_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("checkpoint_format") != REAL_ACTION_CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported real action checkpoint: {payload.get('checkpoint_format')!r}"
        )
    if payload.get("initialization_contract", {}).get("action_transformer_reinitialized") is not False:
        raise ValueError("checkpoint does not prove full official action-head initialization")
    return payload


def apply_real_action_checkpoint(
    model: Any,
    path: str | Path,
    *,
    expected_base_sha256: str,
    expected_norm_sha256: str,
    expected_dataset_identity_sha256: str | None = None,
    expected_cache_identity_sha256: str | None = None,
    expected_cache_attestation_identity_sha256: str | None = None,
    expected_optimizer_step: int | None = None,
) -> dict[str, Any]:
    payload = load_real_action_payload(path)
    observed_base = payload.get("official_base", {}).get("model_weights_sha256")
    if observed_base != expected_base_sha256:
        raise ValueError(
            f"real action checkpoint base mismatch: {observed_base} != {expected_base_sha256}"
        )
    observed_norm = payload.get("norm_stats_sha256")
    if observed_norm != expected_norm_sha256:
        raise ValueError(
            f"real action checkpoint norm mismatch: {observed_norm} != {expected_norm_sha256}"
        )
    observed_dataset = payload.get("dataset_identity_sha256")
    if (
        expected_dataset_identity_sha256 is not None
        and observed_dataset != expected_dataset_identity_sha256
    ):
        raise ValueError(
            "real action checkpoint dataset mismatch: "
            f"{observed_dataset} != {expected_dataset_identity_sha256}"
        )
    observed_cache = payload.get("training_config", {}).get(
        "condition_cache_identity_sha256"
    )
    if (
        expected_cache_identity_sha256 is not None
        and observed_cache != expected_cache_identity_sha256
    ):
        raise ValueError(
            "real action checkpoint Condition cache mismatch: "
            f"{observed_cache} != {expected_cache_identity_sha256}"
        )
    observed_attestation = payload.get("training_config", {}).get(
        "condition_cache_attestation_identity_sha256"
    )
    if (
        expected_cache_attestation_identity_sha256 is not None
        and observed_attestation != expected_cache_attestation_identity_sha256
    ):
        raise ValueError(
            "real action checkpoint Condition cache attestation mismatch: "
            f"{observed_attestation} != {expected_cache_attestation_identity_sha256}"
        )
    observed_step = int(payload.get("optimizer_step", -1))
    if expected_optimizer_step is not None and observed_step != int(
        expected_optimizer_step
    ):
        raise ValueError(
            "real action checkpoint optimizer step mismatch: "
            f"{observed_step} != {expected_optimizer_step}"
        )
    model.transformer.load_state_dict(
        payload["action_transformer_state_dict"], strict=True
    )
    return {
        "path": str(Path(path).expanduser().resolve()),
        "sha256": sha256_file(path),
        "optimizer_step": observed_step,
        "dataset_identity_sha256": str(payload["dataset_identity_sha256"]),
        "strict_state_dict_load": True,
    }
