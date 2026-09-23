#!/usr/bin/env python3
"""Select one baseline checkpoint using held-out validation loss only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(inputs: list[Path], output: Path, expected_mode: str) -> dict:
    """Choose the finite minimum validation metric without reading test SR."""

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    candidates = []
    source_lock_sha256: str | None = None
    teacher_sha256: str | None = None
    for path in inputs:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("mode") != expected_mode:
            raise ValueError(f"Mode mismatch in {path}")
        row_lock = row.get("source_lock_sha256")
        row_teacher = row.get("teacher_sha256")
        if not row_lock or not row_teacher:
            raise ValueError(f"Missing source/checkpoint identity in {path}")
        if source_lock_sha256 is None:
            source_lock_sha256 = str(row_lock)
            teacher_sha256 = str(row_teacher)
        elif row_lock != source_lock_sha256 or row_teacher != teacher_sha256:
            raise RuntimeError("Validation candidates came from different source locks")
        if not row.get("gates", {}).get("pass", False):
            continue
        checkpoint = row.get("adapter_checkpoint")
        if not checkpoint:
            continue
        candidates.append(
            {
                "validation_artifact": str(path.resolve()),
                "checkpoint": str(Path(checkpoint).resolve()),
                "metric": float(row["selection_metric_value"]),
                "examples": int(row["examples"]),
                "validation_split": row["validation_split"],
            }
        )
    if not candidates:
        raise RuntimeError("No validation candidate passed all offline gates")
    candidates.sort(key=lambda row: (row["metric"], row["checkpoint"]))
    selected = candidates[0]
    selected_path = Path(selected["checkpoint"])
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    result = {
        "schema_version": 1,
        "protocol": "validation_only_checkpoint_selection_v1",
        "mode": expected_mode,
        "selection_metric": "validation_total_loss",
        "uses_libero_test_success": False,
        "source_lock_sha256": source_lock_sha256,
        "teacher_sha256": teacher_sha256,
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": _sha256(selected_path),
        "selected_metric": selected["metric"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["action_correction", "anchor_bridge"], required=True
    )
    args = parser.parse_args()
    select([path.resolve() for path in args.input], args.output.resolve(), args.mode)


if __name__ == "__main__":
    main()
