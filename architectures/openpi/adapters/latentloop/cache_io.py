"""Episode-sharded pi0.5 teacher-cache contract and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Literal

import torch


REQUIRED_RECORD_KEYS = frozenset(
    {
        "suite",
        "task_id",
        "episode_id",
        "query_index",
        "environment_step",
        "language_instruction",
        "raw_images",
        "preprocessed_images",
        "robot_state_raw",
        "robot_state_normalized",
        "prefix_embeddings",
        "pre_rope_keys",
        "values",
        "prefix_pad_mask",
        "prefix_attention_pattern",
        "prefix_position_ids",
        "action_noise",
        "action_noise_seed",
        "teacher_action_chunk_normalized",
        "teacher_action_chunk_postprocessed",
        "executed_actions",
        "next_query_observation",
        "source_hashes",
        "timing_ms",
    }
)


def episode_split(
    suite: str,
    task_id: int,
    episode_id: int,
    *,
    seed: int,
    validation_fraction: float = 0.15,
    calibration_fraction: float = 0.15,
) -> Literal["train", "validation", "calibration"]:
    if validation_fraction + calibration_fraction >= 1.0:
        raise ValueError("validation and calibration fractions must sum to less than one")
    payload = f"{seed}:{suite}:{task_id}:{episode_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
    if value < validation_fraction:
        return "validation"
    if value < validation_fraction + calibration_fraction:
        return "calibration"
    return "train"


@dataclass(frozen=True)
class EpisodeIndexEntry:
    suite: str
    task_id: int
    benchmark_task_index: int
    episode_id: int
    split: str
    records: int
    path: str
    sha256: str


class EpisodeCacheWriter:
    def __init__(self, output_dir: str | Path, metadata: dict[str, Any]) -> None:
        self.output = Path(output_dir).resolve()
        if self.output.exists():
            raise FileExistsError(f"refusing to overwrite or reuse cache root: {self.output}")
        (self.output / "episodes").mkdir(parents=True, exist_ok=True)
        self.metadata = metadata
        self.entries: list[EpisodeIndexEntry] = []

    def write_episode(
        self,
        records: list[dict[str, Any]],
        *,
        suite: str,
        task_id: int,
        episode_id: int,
        split: str,
    ) -> EpisodeIndexEntry:
        if not records:
            raise ValueError("cannot write an empty episode")
        for record in records:
            missing = REQUIRED_RECORD_KEYS - record.keys()
            if missing:
                raise ValueError(f"cache record is missing keys: {sorted(missing)}")
            if record["suite"] != suite or int(record["task_id"]) != task_id or int(record["episode_id"]) != episode_id:
                raise ValueError("record episode identity does not match shard identity")
        relative = Path("episodes") / f"{suite}_task{task_id:02d}_episode{episode_id:06d}.pt"
        destination = self.output / relative
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save({"records": records}, temporary_path)
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        entry = EpisodeIndexEntry(
            suite=suite,
            task_id=task_id,
            benchmark_task_index=task_id,
            episode_id=episode_id,
            split=split,
            records=len(records),
            path=str(relative),
            sha256=_sha256(destination),
        )
        self.entries.append(entry)
        self._write_manifest(complete=False)
        return entry

    def finalize(self, statistics: dict[str, Any] | None = None) -> Path:
        return self._write_manifest(complete=True, statistics=statistics or {})

    def _write_manifest(self, complete: bool, statistics: dict[str, Any] | None = None) -> Path:
        path = self.output / "pi05_latentloop_cache_manifest.json"
        payload = {
            "schema_version": int(self.metadata.get("schema_version", 1)),
            "complete": bool(complete),
            "metadata": self.metadata,
            "episodes": [asdict(entry) for entry in self.entries],
            "statistics": statistics or {},
        }
        if payload["schema_version"] == 2:
            payload["cache_manifest_id"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        split_path = self.output / "pi05_latentloop_split.json"
        split_payload = {
            split: [
                {"suite": entry.suite, "task_id": entry.task_id, "episode_id": entry.episode_id}
                for entry in self.entries
                if entry.split == split
            ]
            for split in sorted({entry.split for entry in self.entries})
        }
        split_path.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cache(cache_root: str | Path, *, verify_hashes: bool = False) -> dict[str, Any]:
    del cache_root, verify_hashes
    raise RuntimeError(
        "DISABLED_SUPERSEDED_CACHE_VALIDATOR_V1: use tools/openpi/validate_pi05_cache_v2.py; "
        "the legacy validator encoded fixed checkpoint dimensions"
    )
