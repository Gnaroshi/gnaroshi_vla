#!/usr/bin/env python3
"""One real four-rank DDP optimizer update for batch-profile selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from architectures.seer.adapters.latent_bridge.bridge import (
    SeerFeatureBridge,
    SeerFeatureBridgeConfig,
)
from methods.latent_bridge import bridge_distillation_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("full", "small"), required=True)
    parser.add_argument("--stable-seq-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed_all(42 + rank)

    config = SeerFeatureBridgeConfig.from_preset(
        args.preset,
        stable_seq_len=args.stable_seq_len,
        stable_layer="block_07",
        stable_token_group="visual",
    )
    model = SeerFeatureBridge(config).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    batch = args.batch_size
    values = (
        torch.randn(batch, 3, 384, device=device),
        torch.randn(batch, args.stable_seq_len, 384, device=device),
        torch.randn(batch, 8, device=device),
        torch.randn(batch, 7, device=device),
    )
    target = torch.randn(batch, 3, 384, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predicted_next = values[0] + model(*values)
        loss, _ = bridge_distillation_loss(predicted_next.float(), target.float())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize(device)

    local = {
        "rank": rank,
        "device": local_rank,
        "loss": float(loss.detach().item()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    rows = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local, rows, dst=0)
    if rank == 0:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        payload = {
            "status": "SEER_LATENT_BRIDGE_DDP_TRAINING_SMOKE_PASS",
            "world_size": world_size,
            "per_rank_batch": batch,
            "global_batch": batch * world_size,
            "ranks": rows,
            "max_peak_gpu_memory_bytes": max(row["peak_gpu_memory_bytes"] for row in rows),
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
