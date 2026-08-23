"""Two-rank hidden-hook parity and offline naive-NFE audit for SimVLA."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from architectures.simvla.adapters.latentloop.action_adapter import (
    ActionNoiseKey,
    explicit_action_noise,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.contracts import (
    atomic_write_json,
    native_nfe_time_grid,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    ExactTeacherStore,
    _drop_unused_vlm,
    validate_exact_cache,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_hidden import (
    hidden_hook_parity_report,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    load_frozen_simvla,
)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _quantiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _noise(query: dict[str, Any]) -> torch.Tensor:
    fields = query["metadata"]["noise_key"]
    key = ActionNoiseKey(
        checkpoint=str(fields["checkpoint"]),
        task_id=int(fields["task_id"]),
        episode_id=str(fields["episode_id"]),
        policy_query_index=int(fields["policy_query_index"]),
        seed_base=int(fields["seed_base"]),
    )
    if key.seed() != int(query["noise_seed"]):
        raise RuntimeError("query noise-key identity changed")
    return explicit_action_noise(
        key,
        batch_size=1,
        action_horizon=10,
        action_dim=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        raise RuntimeError("generation audit requires torchrun WORLD_SIZE=2")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    configure_strict_torch_determinism(args.seed)
    source = _load(args.source_lock)
    source_hash = str(source["combined_sha256"])
    gpu_contract_path = os.environ.get("SIMVLA_GPU_CONTRACT_JSON")
    if not gpu_contract_path:
        raise RuntimeError("SIMVLA_GPU_CONTRACT_JSON is required")
    gpu_contract = _load(gpu_contract_path)
    if gpu_contract.get("verdict") != "TWO_SELECTED_GPUS_IDLE":
        raise RuntimeError("two-GPU launch contract did not pass")
    validation = validate_exact_cache(args.cache, verify_checksums=False)
    if validation["verdict"] != "EXACT_TEACHER_CACHE_VALID":
        raise RuntimeError(f"exact cache is invalid: {validation}")
    store = ExactTeacherStore(args.cache)
    if store.manifest.get("source_combined_sha256") != source_hash:
        raise RuntimeError("generation audit cache/source mismatch")

    output = Path(args.output).expanduser().resolve()
    exists = torch.tensor([int(output.exists())], device=device)
    dist.all_reduce(exists, op=dist.ReduceOp.MAX)
    if int(exists.item()):
        raise FileExistsError(f"refusing existing generation-audit output: {output}")
    if rank == 0:
        output.mkdir(parents=True)
    dist.barrier()

    model, processor, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    del processor
    _drop_unused_vlm(model)
    query_ids = [str(item["query_id"]) for item in store.manifest["query_index"]]
    selected = query_ids[: min(int(args.queries), len(query_ids))]
    local_ids = selected[rank::2]
    if not local_ids:
        raise RuntimeError("generation audit assigned no query to a rank")

    first = store.query(local_ids[0])
    first_noise = _noise(first).to(device)
    first_condition = first["condition"].unsqueeze(0).to(device)
    first_proprio = first["proprio"].unsqueeze(0).to(device)
    hidden = hidden_hook_parity_report(
        model.transformer,
        condition=first_condition,
        noisy_action=first_noise,
        proprio=action_adapter.normalize_proprio(first_proprio),
        tau=torch.ones((1,), device=device, dtype=first_noise.dtype),
        dt=-0.1,
    )
    hidden.update(
        {
            "source_combined_sha256": source_hash,
            "rank": rank,
            "query_id": local_ids[0],
            "same_condition_proprio_time_noise": True,
        }
    )

    rows: list[dict[str, Any]] = []
    for query_id in local_ids:
        query = store.query(query_id)
        condition = query["condition"].unsqueeze(0).to(device)
        proprio = query["proprio"].unsqueeze(0).to(device)
        noise = _noise(query).to(device)
        reference = query["teacher_action"].unsqueeze(0).to(device)
        for nfe in (10, 5, 3, 2, 1):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            action = action_adapter.decode_action_from_condition(
                condition,
                proprio,
                steps=nfe,
                initial_noise=noise,
                requires_grad=False,
            )
            torch.cuda.synchronize(device)
            latency = time.perf_counter() - started
            difference = (action.float() - reference.float()).abs()
            rows.append(
                {
                    "rank": rank,
                    "query_id": query_id,
                    "nfe": nfe,
                    "time_grid": list(native_nfe_time_grid(nfe)),
                    "latency_seconds": latency,
                    "first5_action_l1": float(difference[:, :5].mean().item()),
                    "full_chunk_action_l1": float(difference.mean().item()),
                    "translation_l1": float(difference[:, :5, :3].mean().item()),
                    "rotation_l1": float(difference[:, :5, 3:6].mean().item()),
                    "continuous_gripper_l1": float(difference[:, :5, 6:].mean().item()),
                }
            )
    local = {"rank": rank, "hidden": hidden, "rows": rows}
    gathered: list[dict[str, Any] | None] = [None, None]
    dist.all_gather_object(gathered, local)
    result: dict[str, Any] = {}
    if rank == 0:
        rank_payloads = [item for item in gathered if item is not None]
        hidden_pass = all(item["hidden"]["verdict"] == "GENERATOR_HIDDEN_HOOK_PASS" for item in rank_payloads)
        all_rows = [row for item in rank_payloads for row in item["rows"]]
        summaries: dict[str, Any] = {}
        for nfe in (10, 5, 3, 2, 1):
            subset = [row for row in all_rows if int(row["nfe"]) == nfe]
            summaries[str(nfe)] = {
                metric: _quantiles([float(row[metric]) for row in subset])
                for metric in (
                    "latency_seconds",
                    "first5_action_l1",
                    "full_chunk_action_l1",
                    "translation_l1",
                    "rotation_l1",
                    "continuous_gripper_l1",
                )
            }
        hidden_gate = {
            "verdict": "GENERATOR_HIDDEN_HOOK_PASS" if hidden_pass else "GENERATOR_HIDDEN_HOOK_FAIL",
            "source_combined_sha256": source_hash,
            "rank_reports": [item["hidden"] for item in rank_payloads],
            "frozen_original_decoder_reused": True,
            "gpu_contract": gpu_contract,
        }
        nfe_gate = {
            "verdict": "NAIVE_NFE_AUDIT_COMPLETE" if hidden_pass else "NAIVE_NFE_AUDIT_BLOCKED",
            "source_combined_sha256": source_hash,
            "queries": len(selected),
            "offline_only": True,
            "full_conditions_only": True,
            "online_evaluation_run": False,
            "summaries": summaries,
            "nfe_2_and_1_generation_loop_enabled": False,
            "gpu_contract": gpu_contract,
        }
        atomic_write_json(output / "generator_hidden_hook_gate.json", hidden_gate)
        atomic_write_json(output / "naive_nfe_audit.json", nfe_gate)
        with (output / "naive_nfe_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in all_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        result = {"hidden": hidden_gate, "nfe": nfe_gate}
    dist.barrier()
    dist.destroy_process_group()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    result = run(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
