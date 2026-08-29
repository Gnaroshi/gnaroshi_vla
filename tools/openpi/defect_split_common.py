"""Episode-disjoint defect/scheduler/final split enforcement."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROLES = ("defect_fit", "defect_validity", "scheduler_calibration", "final_scientific_evaluation")


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    task = row.get("benchmark_task_index", row.get("task_id"))
    episode = row.get("episode_id", row.get("trial"))
    if task is None or episode is None or row.get("suite") is None:
        raise ValueError("row lacks suite/task/episode identity")
    namespace = row.get("episode_namespace")
    if namespace is None:
        raise ValueError("row lacks episode_namespace")
    return (str(row["suite"]), int(task), str(namespace), str(episode))


def load_contract(path: str | Path) -> tuple[dict[str, set[tuple[str, int, str, str]]], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2 or payload.get("frozen") is not True:
        raise ValueError("defect split contract must be frozen schema v2")
    identity = payload.get("defect_split_contract_id")
    body = {key: value for key, value in payload.items() if key != "defect_split_contract_id"}
    expected_identity = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if identity != expected_identity:
        raise ValueError("defect split contract self-hash is absent or invalid")
    for path_key, hash_key in (
        ("cache_split_contract", "cache_split_contract_sha256"),
        ("final_evaluation_manifest", "final_evaluation_manifest_sha256"),
    ):
        if payload.get(path_key) in (None, "") or payload.get(hash_key) in (None, ""):
            raise ValueError(f"defect split contract lacks {path_key} provenance")
        if sha256_file(payload[path_key]) != payload[hash_key]:
            raise ValueError(f"defect split contract input changed: {path_key}")
    role_sets: dict[str, set[tuple[str, int, str, str]]] = {}
    for role in ROLES:
        rows = payload.get("roles", {}).get(role)
        if rows is None:
            raise ValueError(f"defect split contract has no {role} role")
        keys = {episode_key(row) for row in rows}
        if len(keys) != len(rows):
            raise ValueError(f"duplicate episode in role {role}")
        role_sets[role] = keys
    for index, left in enumerate(ROLES):
        for right in ROLES[index + 1 :]:
            overlap = role_sets[left] & role_sets[right]
            if overlap:
                raise ValueError(f"episode overlap between {left} and {right}: {sorted(overlap)[:3]}")
    return role_sets, payload


def load_role_rows(metrics_path: str | Path, contract_path: str | Path, role: str) -> list[dict[str, str]]:
    if role not in ROLES[:-1]:
        raise ValueError(f"metrics role is not fit/validity/scheduler: {role}")
    role_sets, _ = load_contract(contract_path)
    rows = list(csv.DictReader(Path(metrics_path).open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"no rows in {metrics_path}")
    observed = {episode_key(row) for row in rows}
    unexpected = observed - role_sets[role]
    if unexpected:
        raise ValueError(f"metrics contain episodes outside role {role}: {sorted(unexpected)[:3]}")
    if observed != role_sets[role]:
        missing = role_sets[role] - observed
        raise ValueError(f"metrics omit frozen role episodes: {sorted(missing)[:3]}")
    return rows


def verify_offline_summary(
    summary_path: str | Path,
    *,
    metrics_path: str | Path,
    contract_path: str | Path,
    role: str,
    source_lock_id: str,
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    metrics_path = Path(metrics_path).resolve()
    contract_path = Path(contract_path).resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    _, contract = load_contract(contract_path)
    if (
        payload.get("complete") is not True
        or payload.get("split") != role
        or payload.get("source_lock_id") != source_lock_id
    ):
        raise RuntimeError(f"offline summary is incomplete, stale, or not role {role}")
    expected = {
        "offline_metrics": str(metrics_path),
        "offline_metrics_sha256": sha256_file(metrics_path),
        "split_contract_sha256": contract["cache_split_contract_sha256"],
        "split_contract_id": contract["cache_split_contract_id"],
        "final_evaluation_manifest_sha256": contract[
            "final_evaluation_manifest_sha256"
        ],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"offline summary provenance mismatch for {key}")
    for key in (
        "adapter_checkpoint",
        "adapter_checkpoint_sha256",
        "base_checkpoint",
        "base_checkpoint_sha256",
        "cache_manifest",
        "cache_manifest_id",
        "cache_manifest_sha256",
        "final_evaluation_manifest",
        "final_evaluation_manifest_sha256",
    ):
        if payload.get(key) in (None, ""):
            raise RuntimeError(f"offline summary lacks required producer identity {key}")
    for path_key, hash_key in (
        ("adapter_checkpoint", "adapter_checkpoint_sha256"),
        ("cache_manifest", "cache_manifest_sha256"),
        ("final_evaluation_manifest", "final_evaluation_manifest_sha256"),
    ):
        if sha256_file(payload[path_key]) != payload[hash_key]:
            raise RuntimeError(f"offline summary producer input changed: {path_key}")
    return payload
