#!/usr/bin/env python3
"""Write the immutable source/checkpoint/norm/evaluation lock without loading the model."""

from __future__ import annotations

import argparse

from _common import DEFAULT_CHECKPOINT, DEFAULT_EVALUATION, DEFAULT_NORM_STATS
from architectures.openpi.adapters.latentloop.source_lock import write_source_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--norm-stats", default=str(DEFAULT_NORM_STATS))
    parser.add_argument("--evaluation-root", default=str(DEFAULT_EVALUATION))
    parser.add_argument("--trust-recorded-checkpoint-hash", action="store_true")
    args = parser.parse_args()
    markdown, payload = write_source_lock(
        args.output,
        checkpoint_dir=args.checkpoint,
        norm_stats_path=args.norm_stats,
        evaluation_root=args.evaluation_root,
        hash_checkpoint=not args.trust_recorded_checkpoint_hash,
    )
    print(markdown)
    print(payload)


if __name__ == "__main__":
    main()
