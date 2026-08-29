#!/usr/bin/env python3
"""Revalidate every producer and selected budget in a V2 threshold lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from defect_split_common import load_contract, sha256_file
from source_lock_v2 import verify_lock


def verify_dynamic_threshold_lock(
    path: str | Path,
    *,
    source_lock_path: str | Path,
    adapter_checkpoint: str | Path,
) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    identity_body = {
        key: value for key, value in payload.items() if key != "dynamic_threshold_lock_id"
    }
    expected_identity = hashlib.sha256(
        json.dumps(identity_body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    lock = verify_lock(source_lock_path)
    checkpoint = Path(adapter_checkpoint).resolve()
    if (
        payload.get("schema_version") != 2
        or payload.get("frozen") is not True
        or payload.get("DYNAMIC_BUDGET_LOCK_PASS") is not True
        or payload.get("source_lock_id") != lock["source_lock_id"]
        or payload.get("dynamic_threshold_lock_id") != expected_identity
    ):
        raise RuntimeError("dynamic threshold lock is absent, stale, or not frozen")
    if (
        Path(payload.get("model_checkpoint", "")).resolve() != checkpoint
        or payload.get("model_checkpoint_sha256") != sha256_file(checkpoint)
    ):
        raise RuntimeError("dynamic threshold lock model/checkpoint identity mismatch")

    artifact_pairs = (
        ("scheduler_metrics", "scheduler_metrics_sha256"),
        ("scheduler_summary", "scheduler_summary_sha256"),
        ("split_contract", "scheduler_calibration_manifest_sha256"),
        ("defect_fit", "defect_fit_sha256"),
        ("defect_validity", "defect_validity_sha256"),
    )
    for path_key, hash_key in artifact_pairs:
        if payload.get(path_key) in (None, "") or payload.get(hash_key) in (None, ""):
            raise RuntimeError(f"dynamic threshold lock lacks producer input {path_key}")
        if sha256_file(payload[path_key]) != payload[hash_key]:
            raise RuntimeError(f"dynamic threshold producer input changed: {path_key}")
    _, split = load_contract(payload["split_contract"])
    if (
        split.get("source_lock_id") != lock["source_lock_id"]
        or payload.get("split_contract_id") != split.get("defect_split_contract_id")
    ):
        raise RuntimeError("dynamic threshold lock uses another frozen split contract")

    target = payload.get("target", {})
    if target != {"K_q": 4.0, "minimum": 3.8, "maximum": 4.2}:
        raise RuntimeError("dynamic threshold lock target budget is not the frozen Kq=4 range")
    selected = payload.get("selected", {})
    m_seq = int(selected.get("M_seq", -1))
    m_full = int(selected.get("M_full", -1))
    k_q = float(selected.get("K_q_hat", float("nan")))
    low = float(selected.get("direct_reanchor_threshold", float("nan")))
    high = float(selected.get("full_refresh_threshold", float("nan")))
    if not (1 <= m_seq <= 3 and 2 <= m_full <= 4):
        raise RuntimeError("dynamic threshold lock selected an unsupported age budget")
    if not (math.isfinite(k_q) and 3.8 <= k_q <= 4.2):
        raise RuntimeError("dynamic threshold lock selected Kq outside the frozen range")
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise RuntimeError("dynamic threshold ordering is invalid")
    n_q = int(selected.get("N_q", 0))
    n_f = int(selected.get("N_F", 0))
    if n_q <= 0 or n_f <= 0 or not math.isclose(k_q, n_q / n_f, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("dynamic threshold lock Kq does not match its operation counts")
    calibration = payload.get("calibration", {})
    if (
        float(calibration.get("low_threshold", float("nan"))) != low
        or float(calibration.get("high_threshold", float("nan"))) != high
    ):
        raise RuntimeError("dynamic policy calibration thresholds differ from selected thresholds")
    producer = payload.get("producer", {})
    if (
        producer.get("adapter_checkpoint") != str(checkpoint)
        or producer.get("adapter_checkpoint_sha256") != sha256_file(checkpoint)
    ):
        raise RuntimeError("dynamic threshold producer identity differs from the served adapter")
    return {
        "DYNAMIC_THRESHOLD_LOCK_V2_VERIFIED": True,
        "source_lock_id": lock["source_lock_id"],
        "dynamic_threshold_lock": str(path),
        "dynamic_threshold_lock_sha256": sha256_file(path),
        "adapter_checkpoint": str(checkpoint),
        "adapter_checkpoint_sha256": sha256_file(checkpoint),
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_dynamic_threshold_lock(
                args.lock,
                source_lock_path=args.source_lock,
                adapter_checkpoint=args.adapter_checkpoint,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
