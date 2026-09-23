#!/usr/bin/env python3
"""Fail unless an existing LatentLoop K1 parity artifact is valid."""

from __future__ import annotations

import argparse
from pathlib import Path

from methods.latentloop_segment_grid.serialization import read_json


def verify_parity_artifact(
    path: Path,
    checkpoint_id: int,
    *,
    checkpoint_profile: str | None = None,
    baseline_sha256: str | None = None,
    adapter_sha256: str | None = None,
) -> None:
    """Validate checkpoint identity and the recorded K1 parity verdict."""

    if not path.is_file():
        raise FileNotFoundError(f"Missing K1 parity artifact: {path}")
    payload = read_json(path)
    if int(payload.get("checkpoint_id", -1)) != int(checkpoint_id):
        raise RuntimeError(
            "K1 parity checkpoint mismatch: "
            f"expected={checkpoint_id}, actual={payload.get('checkpoint_id')}"
        )
    if not bool(payload.get("pass", False)):
        raise RuntimeError(f"K1 parity artifact did not pass: {payload}")
    if not bool(payload.get("outcome_parity", False)):
        raise RuntimeError("K1 parity artifact lacks outcome parity")
    if not bool(payload.get("action_keys_identical", False)):
        raise RuntimeError("K1 parity artifact lacks paired step-action keys")
    expected_identity = {
        "checkpoint_profile": checkpoint_profile,
        "baseline_checkpoint_sha256": baseline_sha256,
        "adapter_checkpoint_sha256": adapter_sha256,
    }
    for key, expected in expected_identity.items():
        if expected is None:
            continue
        actual = payload.get(key)
        if actual != expected:
            raise RuntimeError(
                f"K1 parity {key} mismatch: expected={expected}, actual={actual}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint-id", type=int, required=True)
    parser.add_argument("--checkpoint-profile")
    parser.add_argument("--baseline-sha256")
    parser.add_argument("--adapter-sha256")
    args = parser.parse_args()
    verify_parity_artifact(
        args.artifact.resolve(),
        args.checkpoint_id,
        checkpoint_profile=args.checkpoint_profile,
        baseline_sha256=args.baseline_sha256,
        adapter_sha256=args.adapter_sha256,
    )
    print(f"[VERIFY][OK] preflight K1 parity: {args.artifact}")


if __name__ == "__main__":
    main()
