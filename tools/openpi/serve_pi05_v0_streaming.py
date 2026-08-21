#!/usr/bin/env python3
"""Serve paired original/V0 evaluation for a cacheless streaming checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import random

import numpy as np
import torch

from _common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_NORM_STATS,
    load_local_policy,
    require_run,
)
from architectures.openpi.adapters.latentloop.online_policy import (
    LatentLoopServingPolicy,
)
from architectures.openpi.adapters.latentloop.serialization import (
    load_adapter_checkpoint,
)
from verify_pi05_v0_streaming_eval import verify_evaluation_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--norm-stats", default=str(DEFAULT_NORM_STATS))
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--training-run-summary", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--final-manifest", required=True)
    parser.add_argument("--k1-tensor-report", required=True)
    parser.add_argument("--k1-episode-gate", required=True)
    parser.add_argument("--freeze-gate", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--k-q", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--noise-seed-base", type=int, default=7)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require_run(args.run, "OPENPI_PI05_V0_EVAL_RUN")
    if args.k_q != 4 or args.flow_steps != 10 or args.noise_seed_base != 7:
        raise ValueError(
            "primary V0 evaluation is pinned to K_q=4, flow_steps=10, noise_seed_base=7"
        )

    verification = verify_evaluation_inputs(
        source_lock=args.source_lock,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        final_manifest=args.final_manifest,
        suite=args.suite,
        k1_tensor_report=args.k1_tensor_report,
        k1_episode_gate=args.k1_episode_gate,
        freeze_gate=args.freeze_gate,
        training_run_summary=args.training_run_summary,
        adapter_checkpoint=args.adapter_checkpoint,
    )

    random.seed(args.noise_seed_base)
    np.random.seed(args.noise_seed_base)
    torch.manual_seed(args.noise_seed_base)
    torch.cuda.manual_seed_all(args.noise_seed_base)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    base = load_local_policy(args.checkpoint, args.device, flow_steps=args.flow_steps)
    adapter, adapter_payload = load_adapter_checkpoint(
        args.adapter_checkpoint, args.device
    )
    provenance = adapter_payload["config"]["provenance"]
    if (
        Path(provenance["teacher_checkpoint"]).resolve()
        != Path(args.checkpoint).resolve()
    ):
        raise RuntimeError(
            "adapter teacher checkpoint differs from the requested base checkpoint"
        )
    if provenance["source_lock_id"] != verification["source_lock_id"]:
        raise RuntimeError(
            "adapter source-lock id differs from the verified training source"
        )

    policy = LatentLoopServingPolicy(
        base,
        adapter,
        mode="v0",
        k_q=args.k_q,
        flow_steps=args.flow_steps,
        execution_horizon=5,
        noise_seed_base=args.noise_seed_base,
    )
    policy._metadata.update(  # noqa: SLF001
        {
            "base_checkpoint": str(Path(args.checkpoint).resolve()),
            "source_lock": str(Path(args.source_lock).resolve()),
            "source_lock_id": verification["source_lock_id"],
            "source_verification_mode": "locked_file_subset_exact_additive_files_allowed",
            "final_evaluation_manifest": str(Path(args.final_manifest).resolve()),
            "final_evaluation_manifest_sha256": verification["manifest"]["sha256"],
            "final_evaluation_manifest_id": verification["manifest"]["manifest_id"],
            "suite": args.suite,
            "adapter_checkpoint": str(Path(args.adapter_checkpoint).resolve()),
            "adapter_checkpoint_sha256": verification["training"][
                "adapter_checkpoint_sha256"
            ],
            "adapter_provenance": provenance,
            "adapter_trainable_parameters": verification["training"][
                "adapter_trainable_parameters"
            ],
            "checkpoint_selection": verification["training"]["checkpoint_selection"],
            "checkpoint_step": verification["training"]["checkpoint_step"],
            "training_steps": verification["training"]["training_steps"],
            "evaluation_harness_sha256": verification["evaluation_harness"][
                "combined_sha256"
            ],
            "paired_policy_paths": ["original", "v0"],
            "efficiency_claim_boundary": "PaliGemma prefix-transformer call reduction",
            "configured_k_q": args.k_q,
            "effective_k_q": args.k_q,
        }
    )

    from openpi.serving import websocket_policy_server

    logging.info(
        "Serving paired original/V0 checkpoint_step=%d K_q=%d K_a=%d suite=%s port=%d",
        verification["training"]["checkpoint_step"],
        args.k_q,
        args.k_q * 5,
        args.suite,
        args.port,
    )
    logging.info(
        "Evaluation verification: %s", json.dumps(verification, sort_keys=True)
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
