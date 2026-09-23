#!/usr/bin/env python3
"""Persist the verified Seer action-token overlap used by action correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(repo_root: Path, output: Path) -> dict:
    """Verify P=3 zero-offset labels and write the one-query shift contract."""

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    sys.path.insert(0, str(repo_root))
    from architectures.seer.adapters.latentloop_comparison.action_token_alignment import (  # noqa: PLC0415
        ACTION_LABEL_OFFSET,
        action_token_time_mapping,
        assert_canonical_seer_alignment,
        verified_overlap_pairs,
    )

    label_source = repo_root / "architectures/seer/upstream/utils/train_utils.py"
    source = label_source.read_text(encoding="utf-8")
    required = (
        "for j in range(args.action_pred_steps)",
        "actions[:, j:args.sequence_length-args.atten_goal+j, :]",
    )
    if not all(fragment in source for fragment in required):
        raise RuntimeError("Seer action-label construction no longer has zero-offset token labels")
    assert ACTION_LABEL_OFFSET == 0
    assert_canonical_seer_alignment(3)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "action_pred_steps": 3,
        "action_label_offset": 0,
        "previous_query": action_token_time_mapping(0, 3),
        "current_query": action_token_time_mapping(1, 3),
        "verified_overlap_previous_to_current": verified_overlap_pairs(3),
        "shift_rule": "previous token 1->current token 0; previous token 2->current token 1; terminal initializer repeats token 2 with valid_mask=false",
        "label_source": str(label_source),
        "label_source_sha256": _sha256(label_source),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.repo_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
