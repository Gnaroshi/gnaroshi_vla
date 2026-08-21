#!/usr/bin/env python3
"""Serve external pi0.5 LatentLoop through OpenPI's unchanged websocket API."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random

import numpy as np
import torch

from _common import DEFAULT_CHECKPOINT, load_local_policy, require_run
from architectures.openpi.adapters.latentloop.dynamic_policy import BudgetedDynamicPolicy
from architectures.openpi.adapters.latentloop.online_policy import LatentLoopServingPolicy
from architectures.openpi.adapters.latentloop.serialization import load_adapter_checkpoint
from methods.variable_time_latentloop.budget_calibration import BudgetCalibration
from pi05_stage_gate_v2 import STAGE_REQUIREMENTS, verify_stage
from verify_pi05_final_manifest_v2 import verify_manifest
from verify_pi05_dynamic_threshold_lock_v2 import verify_dynamic_threshold_lock


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument(
        "--mode", choices=("original", "k1", "hold", "latent_bridge", "v0", "v1", "v2"), required=True
    )
    parser.add_argument("--k-q", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--noise-seed-base", type=int, default=7)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_REQUIREMENTS), required=True)
    parser.add_argument("--stage-artifact", action="append", default=[])
    parser.add_argument("--final-evaluation-manifest")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--dynamic-threshold-lock")
    parser.add_argument("--k1-audit", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require_run(args.run, "OPENPI_LATENTLOOP_EVAL_RUN")
    stage_gate = verify_stage(args.stage, args.source_lock, args.stage_artifact)
    if args.final_evaluation_manifest:
        manifest_gate = verify_manifest(
            args.final_evaluation_manifest,
            args.source_lock,
            suite=args.suite,
        )
    elif args.mode == "k1" and args.k1_audit and args.stage == "stage1_episode_smoke":
        manifest_gate = {"manifest": None, "manifest_sha256": None}
    else:
        raise RuntimeError("missing evidence: scientific serving requires a frozen final manifest")
    source_lock = json.loads(Path(args.source_lock).read_text(encoding="utf-8"))
    if Path(source_lock["checkpoint"]["directory"]).resolve() != Path(args.checkpoint).resolve():
        raise RuntimeError("checkpoint mismatch: server checkpoint differs from source lock v2")
    if args.mode in {"latent_bridge", "v0", "v1", "v2"}:
        if not args.adapter_checkpoint:
            raise ValueError(f"{args.mode} requires --adapter-checkpoint")
    if args.mode == "latent_bridge":
        raise RuntimeError("Latent Bridge-style KV baseline remains disabled pending official fidelity")
    if args.mode == "v2":
        if not args.dynamic_threshold_lock:
            raise ValueError("v2 requires --dynamic-threshold-lock")
        dynamic_verification = verify_dynamic_threshold_lock(
            args.dynamic_threshold_lock,
            source_lock_path=args.source_lock,
            adapter_checkpoint=args.adapter_checkpoint,
        )
        dynamic_payload = json.loads(Path(args.dynamic_threshold_lock).read_text(encoding="utf-8"))
        selected = dynamic_payload["selected"]
        dynamic = BudgetedDynamicPolicy(
            BudgetCalibration.from_dict(dynamic_payload["calibration"]),
            execution_horizon=5,
            m_seq=int(selected["M_seq"]),
            m_full=int(selected["M_full"]),
        )
    else:
        dynamic = None
        dynamic_verification = None

    random.seed(args.noise_seed_base)
    np.random.seed(args.noise_seed_base)
    torch.manual_seed(args.noise_seed_base)
    torch.cuda.manual_seed_all(args.noise_seed_base)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    base = load_local_policy(args.checkpoint, args.device, flow_steps=args.flow_steps)
    adapter = None
    adapter_payload = None
    if args.adapter_checkpoint:
        adapter, adapter_payload = load_adapter_checkpoint(args.adapter_checkpoint, args.device)
        provenance = adapter_payload["config"]["provenance"]
        if Path(provenance["base_checkpoint"]).resolve() != Path(args.checkpoint).resolve():
            raise ValueError("adapter base checkpoint does not match the requested policy checkpoint")
        if provenance.get("source_lock_id") != stage_gate["source_lock_id"]:
            raise ValueError("adapter source lock does not match the current stage gate")
    effective_k_q = 1 if args.mode == "k1" else args.k_q
    policy = LatentLoopServingPolicy(
        base,
        adapter,
        mode=args.mode,
        k_q=effective_k_q,
        flow_steps=args.flow_steps,
        execution_horizon=5,
        noise_seed_base=args.noise_seed_base,
        dynamic_policy=dynamic,
        k1_audit=args.k1_audit,
    )
    policy._metadata.update(  # noqa: SLF001
        {
            "base_checkpoint": str(Path(args.checkpoint).resolve()),
            "source_lock": str(Path(args.source_lock).resolve()),
            "source_lock_id": stage_gate["source_lock_id"],
            "stage": args.stage,
            "stage_artifacts": stage_gate["artifacts"],
            "final_evaluation_manifest": manifest_gate["manifest"],
            "final_evaluation_manifest_sha256": manifest_gate["manifest_sha256"],
            "final_evaluation_manifest_id": manifest_gate.get("manifest_id"),
            "suite": args.suite,
            "adapter_checkpoint": (
                str(Path(args.adapter_checkpoint).resolve()) if args.adapter_checkpoint else None
            ),
            "adapter_checkpoint_sha256": (
                _sha256(Path(args.adapter_checkpoint)) if args.adapter_checkpoint else None
            ),
            "adapter_provenance": (
                adapter_payload["config"].get("provenance") if adapter_payload else None
            ),
            "adapter_trainable_parameters": (
                int(getattr(adapter, "trainable_parameters", 0)) if adapter is not None else 0
            ),
            "efficiency_claim_boundary": "PaliGemma prefix-transformer call reduction",
            "configured_k_q": args.k_q,
            "effective_k_q": effective_k_q,
            "dynamic_threshold_verification": dynamic_verification,
        }
    )

    from openpi.serving import websocket_policy_server

    logging.info(
        "Serving mode=%s checkpoint=%s adapter=%s K_q=%d K_a=%d port=%d",
        args.mode,
        args.checkpoint,
        args.adapter_checkpoint,
        effective_k_q,
        effective_k_q * 5,
        args.port,
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
