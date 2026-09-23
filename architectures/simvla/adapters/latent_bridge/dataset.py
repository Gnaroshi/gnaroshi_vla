"""Training-only sidecar dataset for SimVLA Latent Bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from architectures.simvla.adapters.latentloop.native_v0_dataset import (
    stable_episode_partition,
)
from architectures.simvla.adapters.latentloop.native_v0_training_cache import (
    load_compact_sequence,
    load_training_manifest,
)


SIDECAR_SCHEMA = "simvla_latent_bridge_training_sidecar_v1"
DAGGER_SCHEMA = "simvla_latent_bridge_dagger_v2"
SYNC_SCHEMA = "simvla_latent_bridge_sync_v1"


def _validate_fraction(value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError("heldout_fraction must be in (0,1)")


def _format_transition(sample: dict[str, Any], *, context: str) -> dict[str, torch.Tensor]:
    required = {
        "condition_input",
        "condition_target",
        "stable_anchor",
        "state",
        "previous_action",
    }
    missing = required - sample.keys()
    if missing:
        raise ValueError(f"{context} transition is missing {sorted(missing)}")
    expected = {
        "condition_input": (122, 960),
        "condition_target": (122, 960),
        "stable_anchor": (122, 960),
        "state": (8,),
        "previous_action": (7,),
    }
    for name, shape in expected.items():
        value = sample[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            observed = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            raise ValueError(f"{context} {name} must have shape {shape}, got {observed}")
    return {
        "condition_t": sample["condition_input"].float(),
        "condition_t1": sample["condition_target"].float(),
        "stable_t": sample["stable_anchor"].float(),
        "state_t": sample["state"].float(),
        "previous_action_t": sample["previous_action"].float(),
        "transition_age": torch.tensor(int(sample.get("age", 1)), dtype=torch.long),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sidecar_manifest(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != SIDECAR_SCHEMA:
        raise ValueError(f"unsupported Latent Bridge sidecar: {payload.get('schema_version')}")
    return payload


def validate_sidecar(path: str | Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    manifest = load_sidecar_manifest(root)
    errors: list[str] = []
    identities: set[tuple[int, str, int]] = set()
    for entry in manifest["sequences"]:
        identity = (
            int(entry["task_id"]),
            str(entry["episode_id"]),
            int(entry["anchor_query_index"]),
        )
        if identity in identities:
            errors.append(f"duplicate identity: {identity}")
        identities.add(identity)
        item = root / entry["file"]
        if not item.is_file():
            errors.append(f"missing sidecar: {item}")
            continue
        if item.stat().st_size != int(entry["size_bytes"]):
            errors.append(f"size mismatch: {item}")
        if verify_hashes and sha256_file(item) != entry["sha256"]:
            errors.append(f"hash mismatch: {item}")
    source = Path(manifest["source_cache_root"])
    if sha256_file(source / "manifest.json") != manifest["source_cache_manifest_sha256"]:
        errors.append("source training-cache manifest changed")
    return {
        "passed": not errors,
        "errors": errors,
        "sequences": len(identities),
        "source_cache_root": str(source),
        "hashes_verified": bool(verify_hashes),
    }


class SimVLALatentBridgeDataset(Dataset[dict[str, torch.Tensor]]):
    """Flatten each four-query training sequence into three one-step transitions."""

    def __init__(
        self,
        sidecar_root: str | Path,
        *,
        split: str,
        heldout_fraction: float = 0.1,
        split_seed: int = 42,
    ) -> None:
        if split not in {"train", "heldout", "all"}:
            raise ValueError("split must be train, heldout, or all")
        _validate_fraction(heldout_fraction)
        self.sidecar_root = Path(sidecar_root).expanduser().resolve()
        self.sidecar_manifest = load_sidecar_manifest(self.sidecar_root)
        self.source_root = Path(self.sidecar_manifest["source_cache_root"]).resolve()
        self.source_manifest = load_training_manifest(self.source_root)
        source_by_identity = {
            (
                int(entry["task_id"]),
                str(entry["episode_id"]),
                int(entry["anchor_query_index"]),
            ): entry
            for entry in self.source_manifest["sequences"]
        }
        self.samples: list[tuple[dict[str, Any], dict[str, Any], int]] = []
        identities: list[tuple[int, str, int, int]] = []
        for sidecar_entry in self.sidecar_manifest["sequences"]:
            identity = (
                int(sidecar_entry["task_id"]),
                str(sidecar_entry["episode_id"]),
                int(sidecar_entry["anchor_query_index"]),
            )
            source_entry = source_by_identity.get(identity)
            if source_entry is None:
                raise KeyError(f"sidecar identity missing from source cache: {identity}")
            heldout = (
                stable_episode_partition(identity[0], identity[1], split_seed)
                < heldout_fraction
            )
            if split == "train" and heldout:
                continue
            if split == "heldout" and not heldout:
                continue
            for transition in range(3):
                self.samples.append((source_entry, sidecar_entry, transition))
                identities.append((*identity, transition))
        if not self.samples:
            raise ValueError(f"no samples for split={split}")
        self.split = split
        self.heldout_fraction = float(heldout_fraction)
        self.split_seed = int(split_seed)
        encoded = json.dumps(identities, separators=(",", ":")).encode()
        self.split_sha256 = hashlib.sha256(encoded).hexdigest()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_entry, sidecar_entry, transition = self.samples[index]
        source = load_compact_sequence(self.source_root, source_entry)
        sidecar_path = self.sidecar_root / sidecar_entry["file"]
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        conditions = source["condition_sequence"].float()
        stable = sidecar["stable_sequence"].float()
        actions = sidecar["previous_action_sequence"].float()
        proprio = source["proprio_sequence"].float()
        if conditions.shape != stable.shape or conditions.shape != (4, 122, 960):
            raise AssertionError("Latent Bridge source/sidecar condition shapes changed")
        if actions.shape != (4, 7) or proprio.shape != (4, 8):
            raise AssertionError("Latent Bridge state/action sidecar shapes changed")
        return {
            "condition_t": conditions[transition],
            "condition_t1": conditions[transition + 1],
            "stable_t": stable[transition],
            "state_t": proprio[transition],
            "previous_action_t": actions[transition],
            "transition_age": torch.tensor(transition + 1, dtype=torch.long),
        }

    def contract(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "split_seed": self.split_seed,
            "heldout_fraction": self.heldout_fraction,
            "split_sha256": self.split_sha256,
            "transitions": len(self),
            "split_unit": "task_id+episode_id",
            "data_role": self.source_manifest["data_role"],
            "final_eval_episode_overlap": self.source_manifest["metadata"][
                "train_test_episode_overlap"
            ],
        }


class SimVLALatentBridgeDaggerDataset(Dataset[dict[str, torch.Tensor]]):
    """On-policy bridge states paired with full-SimVLA teacher conditions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"complete DAgger manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != DAGGER_SCHEMA:
            raise ValueError("unsupported DAgger dataset schema")
        required_manifest_fields = {
            "base_checkpoint",
            "norm_stats_sha256",
            "bridge_checkpoint_sha256",
            "stable_layer_index",
            "token_mode",
            "latent_bridge_upstream",
            "simvla_latent_bridge_integration",
            "action_horizon",
            "execution_horizon",
            "flow_steps",
            "episodes",
            "shards",
        }
        missing = required_manifest_fields - self.manifest.keys()
        if missing:
            raise ValueError(f"DAgger manifest is missing {sorted(missing)}")
        self.samples: list[dict[str, Any]] = []
        episode_identities: set[tuple[int, int]] = set()
        for entry in self.manifest["shards"]:
            identity = (int(entry["task_id"]), int(entry["trial_id"]))
            if identity in episode_identities:
                raise RuntimeError(f"duplicate DAgger episode identity: {identity}")
            episode_identities.add(identity)
            path = self.root / entry["file"]
            if path.stat().st_size != int(entry["size_bytes"]):
                raise RuntimeError(f"DAgger shard size changed: {path}")
            if sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"DAgger shard hash changed: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema_version") != DAGGER_SCHEMA:
                raise RuntimeError(f"DAgger shard schema changed: {path}")
            if (int(payload["task_id"]), int(payload["trial_id"])) != identity:
                raise RuntimeError(f"DAgger shard identity differs from manifest: {path}")
            transitions = payload.get("transitions", [])
            if len(transitions) != int(entry["transitions"]):
                raise RuntimeError(f"DAgger transition count changed: {path}")
            for transition in transitions:
                self.samples.append(transition)
        if int(self.manifest["episodes"]) != len(episode_identities):
            raise RuntimeError("DAgger manifest episode count does not match its shards")
        if not self.samples:
            raise ValueError(f"no DAgger transitions found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return _format_transition(sample, context="DAgger")


class SimVLALatentBridgeSyncDataset(Dataset[dict[str, torch.Tensor]]):
    """Episode-disjoint transitions collected from full-SimVLA rollouts."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        heldout_fraction: float = 0.1,
        split_seed: int = 42,
    ) -> None:
        if split not in {"train", "heldout", "all"}:
            raise ValueError("split must be train, heldout, or all")
        _validate_fraction(heldout_fraction)
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"complete sync manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SYNC_SCHEMA:
            raise ValueError("unsupported sync rollout schema")
        shards = self.manifest.get("shards", [])
        expected_episodes = len(self.manifest.get("task_ids", [])) * int(
            self.manifest.get("trials_per_task", 0)
        )
        if int(self.manifest.get("episodes", -1)) != len(shards):
            raise RuntimeError("sync manifest episode count does not match its shards")
        if expected_episodes != len(shards):
            raise RuntimeError("sync collection is incomplete for its declared task/trial grid")
        self.samples: list[dict[str, Any]] = []
        identities: list[tuple[int, int, int]] = []
        episode_identities: set[tuple[int, int]] = set()
        for entry in shards:
            task_id = int(entry["task_id"])
            trial_id = int(entry["trial_id"])
            episode_identity = (task_id, trial_id)
            if episode_identity in episode_identities:
                raise RuntimeError(f"duplicate sync episode identity: {episode_identity}")
            episode_identities.add(episode_identity)
            partition = stable_episode_partition(
                task_id, f"sync_trial_{trial_id}", split_seed
            )
            heldout = partition < heldout_fraction
            if split == "train" and heldout:
                continue
            if split == "heldout" and not heldout:
                continue
            path = self.root / entry["file"]
            if path.stat().st_size != int(entry["size_bytes"]):
                raise RuntimeError(f"sync shard size changed: {path}")
            if sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"sync shard hash changed: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema_version") != SYNC_SCHEMA:
                raise RuntimeError(f"sync shard schema changed: {path}")
            if (int(payload["task_id"]), int(payload["trial_id"])) != episode_identity:
                raise RuntimeError(f"sync shard identity differs from manifest: {path}")
            transitions = payload.get("transitions", [])
            if len(transitions) != int(entry["transitions"]):
                raise RuntimeError(f"sync transition count changed: {path}")
            for index, transition in enumerate(transitions):
                self.samples.append(transition)
                identities.append((task_id, trial_id, index))
        if not self.samples:
            raise ValueError(f"no sync transitions found for split={split}")
        self.split = split
        self.heldout_fraction = float(heldout_fraction)
        self.split_seed = int(split_seed)
        self.split_sha256 = hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode()
        ).hexdigest()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return _format_transition(sample, context="sync")

    def contract(self) -> dict[str, Any]:
        return {
            "data_role": self.manifest["data_role"],
            "split": self.split,
            "split_unit": "task_id+trial_id",
            "split_seed": self.split_seed,
            "heldout_fraction": self.heldout_fraction,
            "split_sha256": self.split_sha256,
            "transitions": len(self),
            "trial_offset": self.manifest["trial_offset"],
            "trials_per_task": self.manifest["trials_per_task"],
        }
