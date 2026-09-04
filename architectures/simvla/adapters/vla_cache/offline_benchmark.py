"""Deterministic offline comparison of real-world SimVLA efficiency methods."""

from __future__ import annotations

import argparse
import collections
import csv
import gc
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


METHODS = (
    "baseline",
    "condition_loop",
    "latentloop",
    "vla_cache_full",
    "vla_cache",
)
EXECUTION_HORIZON = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare SimVLA baseline, LatentLoop, and VLA-Cache on one fixed "
            "held-out real-demonstration query plan. This does not measure task success."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-queries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ("git", "-C", str(root), *args), text=True
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short").splitlines(),
    }


def _longest_stride_chain(indices: Iterable[int], stride: int) -> list[int]:
    remaining = set(int(index) for index in indices)
    chains: list[list[int]] = []
    while remaining:
        start = min(remaining)
        chain: list[int] = []
        current = start
        while current in remaining:
            chain.append(current)
            remaining.remove(current)
            current += stride
        chains.append(chain)
    return max(chains, key=lambda chain: (len(chain), -chain[0]))


def build_balanced_query_plan(
    samples: Iterable[tuple[str, int]],
    *,
    query_count: int,
    execution_horizon: int = EXECUTION_HORIZON,
) -> list[dict[str, Any]]:
    """Choose sequential, episode-balanced policy queries at the R-step cadence."""

    if query_count < 1:
        raise ValueError("query_count must be positive")
    by_episode: dict[str, list[int]] = collections.defaultdict(list)
    for episode_id, frame_index in samples:
        by_episode[str(episode_id)].append(int(frame_index))
    chains = {
        episode_id: _longest_stride_chain(indices, execution_horizon)
        for episode_id, indices in sorted(by_episode.items())
    }
    if not chains:
        raise ValueError("validation split has no query candidates")

    selected: dict[str, list[int]] = {episode_id: [] for episode_id in chains}
    offsets = {episode_id: 0 for episode_id in chains}
    episode_ids = list(chains)
    while sum(len(values) for values in selected.values()) < query_count:
        progressed = False
        for episode_id in episode_ids:
            offset = offsets[episode_id]
            chain = chains[episode_id]
            if offset >= len(chain):
                continue
            selected[episode_id].append(chain[offset])
            offsets[episode_id] += 1
            progressed = True
            if sum(len(values) for values in selected.values()) == query_count:
                break
        if not progressed:
            available = sum(len(chain) for chain in chains.values())
            raise ValueError(
                f"requested {query_count} sequential queries, but only {available} are available"
            )

    plan: list[dict[str, Any]] = []
    query_id = 0
    for segment_id, episode_id in enumerate(episode_ids):
        for local_query_index, frame_index in enumerate(selected[episode_id]):
            plan.append(
                {
                    "query_id": query_id,
                    "segment_id": segment_id,
                    "episode_id": episode_id,
                    "frame_index": frame_index,
                    "local_query_index": local_query_index,
                }
            )
            query_id += 1
    return plan


def _latency_stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {key: 0.0 for key in ("mean", "std", "p50", "p95", "min", "max")}
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_flat = left.reshape(left.shape[0], -1).astype(np.float64)
    right_flat = right.reshape(right.shape[0], -1).astype(np.float64)
    denominator = np.linalg.norm(left_flat, axis=1) * np.linalg.norm(right_flat, axis=1)
    numerator = np.sum(left_flat * right_flat, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 0,
    )


def _fidelity(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    gripper_candidate = np.sign(candidate[..., -1])
    gripper_reference = np.sign(reference[..., -1])
    return {
        "action_l1_mean": float(np.abs(difference).mean()),
        "action_l2_per_chunk_mean": float(
            np.linalg.norm(difference.reshape(difference.shape[0], -1), axis=1).mean()
        ),
        "action_max_abs": float(np.abs(difference).max()),
        "action_chunk_cosine_mean": float(_cosine_rows(candidate, reference).mean()),
        "gripper_sign_agreement": float(
            np.mean(gripper_candidate == gripper_reference)
        ),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _drain_five_actions(policy: Any, sample: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    _synchronize(policy.device)
    block_started = time.perf_counter()
    policy.act(
        sample["base_rgb"],
        sample["wrist_rgb"],
        sample["proprio"],
        sample["instruction"],
    )
    _synchronize(policy.device)
    query_ms = (time.perf_counter() - block_started) * 1000.0
    if policy.cached_action_chunk is None:
        raise RuntimeError("policy did not produce a fresh H=10 action chunk")
    chunk = policy.cached_action_chunk.detach().cpu().float().numpy()[0].copy()
    for _ in range(EXECUTION_HORIZON - 1):
        policy.act(
            sample["base_rgb"],
            sample["wrist_rgb"],
            sample["proprio"],
            sample["instruction"],
        )
    _synchronize(policy.device)
    block_ms = (time.perf_counter() - block_started) * 1000.0
    if policy.action_queue:
        raise RuntimeError("R=5 action queue was not drained before the next query")
    return chunk, query_ms, block_ms


def _cache_report(policy: Any) -> dict[str, Any]:
    runtime = getattr(policy, "vla_cache", None)
    if runtime is None or runtime.last_report is None:
        return {}
    decoder = runtime.last_report["decoder"]
    return {
        "first_query": bool(decoder["first_query"]),
        "actual_kv_reuse": bool(decoder["actual_kv_reuse"]),
        "computed_token_layers": int(decoder["computed_token_layers"]),
        "skipped_token_layers": int(decoder["skipped_token_layers"]),
    }


def _run_method(
    *,
    method: str,
    contract: Any,
    dataset: Any,
    sample_lookup: dict[tuple[str, int], int],
    plan: list[dict[str, Any]],
    repeats: int,
    warmup_queries: int,
    device: torch.device,
    output: Path,
) -> dict[str, Any]:
    from architectures.simvla.adapters.latentloop_real_deploy.controller import (
        SimVLARealController,
    )

    controller = SimVLARealController.from_contract(
        contract, deployment_method=method, device=device
    )
    policy = controller.policy
    warmup = plan[:warmup_queries]
    if warmup:
        policy.task_id = 0
        policy.trial_id = 999_999
        policy.reset()
        for item in warmup:
            sample = dataset.raw_sample(
                sample_lookup[(item["episode_id"], item["frame_index"])]
            )
            _drain_five_actions(policy, sample)
    policy.reset()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    method_rows: list[dict[str, Any]] = []
    repeat_summaries: list[dict[str, Any]] = []
    prediction_files: list[str] = []
    from tqdm.auto import tqdm

    for repeat in range(1, repeats + 1):
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        counters: collections.Counter[str] = collections.Counter()
        current_segment: int | None = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        progress = tqdm(
            plan,
            desc=f"{method} repeat{repeat}",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for item in progress:
            if current_segment != int(item["segment_id"]):
                if current_segment is not None:
                    counters.update(policy.metrics.counters)
                policy.reset()
                policy.task_id = 0
                policy.trial_id = int(item["segment_id"])
                current_segment = int(item["segment_id"])
            sample = dataset.raw_sample(
                sample_lookup[(item["episode_id"], item["frame_index"])]
            )
            chunk, query_ms, block_ms = _drain_five_actions(policy, sample)
            predictions.append(chunk)
            targets.append(np.asarray(sample["action"], dtype=np.float32))
            cache = _cache_report(policy)
            method_rows.append(
                {
                    "method": method,
                    "repeat": repeat,
                    **item,
                    "query_ms": query_ms,
                    "block_ms": block_ms,
                    "ms_per_executed_action": block_ms / EXECUTION_HORIZON,
                    **cache,
                }
            )
            progress.set_postfix(ms_action=f"{block_ms / EXECUTION_HORIZON:.2f}")
        counters.update(policy.metrics.counters)
        prediction_array = np.stack(predictions)
        target_array = np.stack(targets)
        prediction_path = output / f"predictions_{method}_repeat{repeat}.npz"
        np.savez_compressed(
            prediction_path,
            prediction=prediction_array,
            target=target_array,
        )
        prediction_files.append(prediction_path.name)
        rows = [
            row
            for row in method_rows
            if row["repeat"] == repeat and row["method"] == method
        ]
        repeat_summaries.append(
            {
                "repeat": repeat,
                "queries": len(rows),
                "query_latency_ms": _latency_stats(row["query_ms"] for row in rows),
                "amortized_latency_ms_per_action": _latency_stats(
                    row["ms_per_executed_action"] for row in rows
                ),
                "peak_cuda_allocated_gib": (
                    float(torch.cuda.max_memory_allocated(device) / (1024**3))
                    if device.type == "cuda"
                    else 0.0
                ),
                "peak_cuda_reserved_gib": (
                    float(torch.cuda.max_memory_reserved(device) / (1024**3))
                    if device.type == "cuda"
                    else 0.0
                ),
                "counters": dict(counters),
                "target_action_error": _fidelity(prediction_array, target_array),
            }
        )

    row_path = output / f"queries_{method}.csv"
    fieldnames = list(method_rows[0])
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(method_rows)
    result = {
        "method": method,
        "deployment": controller.deployment_metadata(),
        "repeats": repeat_summaries,
        "query_csv": row_path.name,
        "prediction_files": prediction_files,
    }
    _write_json(output / f"summary_{method}.json", result)
    del policy, controller
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _aggregate(output: Path, method_results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    predictions: dict[tuple[str, int], np.ndarray] = {}
    for result in method_results:
        method = result["method"]
        csv_rows = np.genfromtxt(
            output / result["query_csv"],
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        query_values = np.atleast_1d(csv_rows["query_ms"]).astype(float).tolist()
        action_values = (
            np.atleast_1d(csv_rows["ms_per_executed_action"]).astype(float).tolist()
        )
        peak_allocated: list[float] = []
        peak_reserved: list[float] = []
        for repeat in result["repeats"]:
            repeat_id = int(repeat["repeat"])
            peak_allocated.append(float(repeat["peak_cuda_allocated_gib"]))
            peak_reserved.append(float(repeat["peak_cuda_reserved_gib"]))
            predictions[(method, repeat_id)] = np.load(
                output / f"predictions_{method}_repeat{repeat_id}.npz"
            )["prediction"]
        aggregate[method] = {
            "query_latency_ms": _latency_stats(query_values),
            "amortized_latency_ms_per_action": _latency_stats(action_values),
            "peak_cuda_allocated_gib_max": max(peak_allocated),
            "peak_cuda_reserved_gib_max": max(peak_reserved),
        }

    baseline_ms = aggregate["baseline"]["amortized_latency_ms_per_action"]["mean"]
    for method in METHODS:
        method_ms = aggregate[method]["amortized_latency_ms_per_action"]["mean"]
        aggregate[method]["speedup_vs_sdpa_baseline"] = baseline_ms / method_ms
        comparisons = []
        for repeat in range(1, len(method_results[0]["repeats"]) + 1):
            comparisons.append(
                _fidelity(predictions[(method, repeat)], predictions[("baseline", repeat)])
            )
        aggregate[method]["action_fidelity_vs_sdpa_baseline_per_repeat"] = comparisons

    eager_comparisons = []
    for repeat in range(1, len(method_results[0]["repeats"]) + 1):
        eager_comparisons.append(
            _fidelity(
                predictions[("vla_cache", repeat)],
                predictions[("vla_cache_full", repeat)],
            )
        )
    aggregate["vla_cache"][
        "action_fidelity_vs_eager_no_reuse_per_repeat"
    ] = eager_comparisons
    return aggregate


def main() -> None:
    args = _parse_args()
    if args.queries < 1 or args.repeats < 1 or args.warmup_queries < 0:
        raise ValueError("queries/repeats must be positive and warmup must be non-negative")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the paper comparison requires one CUDA GPU")
    _configure_determinism(args.seed)

    from architectures.simvla.adapters.latentloop_real_deploy.contracts import (
        load_deployment_contract,
        sha256_file,
    )
    from architectures.simvla.adapters.real_world_training.dataset import (
        RealSimVLADataset,
        query_lookup,
    )

    contract = load_deployment_contract(args.manifest, verify_artifacts=True)
    dataset = RealSimVLADataset(
        args.dataset_manifest,
        split="validation",
        training=False,
        action_horizon=10,
    )
    lookup = query_lookup(dataset)
    plan = build_balanced_query_plan(dataset.samples, query_count=args.queries)
    _write_json(output / "query_plan.json", plan)
    root = Path(__file__).resolve().parents[4]
    experiment = {
        "schema_version": 1,
        "experiment": "simvla_real_vla_cache_offline_comparison",
        "scientific_scope": (
            "offline held-out sequential action fidelity and synchronized GPU efficiency; "
            "not a real-robot task-success evaluation"
        ),
        "methods": list(METHODS),
        "queries": args.queries,
        "repeats": args.repeats,
        "warmup_queries": args.warmup_queries,
        "seed": args.seed,
        "action_horizon": 10,
        "execution_horizon": EXECUTION_HORIZON,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "dataset_manifest": str(Path(args.dataset_manifest).expanduser().resolve()),
        "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        "git": _git_metadata(root),
    }
    _write_json(output / "experiment_config.json", experiment)

    results = []
    for method in METHODS:
        print(f"[comparison] method={method} state=start", flush=True)
        result = _run_method(
            method=method,
            contract=contract,
            dataset=dataset,
            sample_lookup=lookup,
            plan=plan,
            repeats=args.repeats,
            warmup_queries=args.warmup_queries,
            device=device,
            output=output,
        )
        results.append(result)
        print(f"[comparison] method={method} state=complete", flush=True)

    aggregate = _aggregate(output, results)
    cache_counters = [
        repeat["counters"]
        for result in results
        if result["method"] == "vla_cache"
        for repeat in result["repeats"]
    ]
    skipped = sum(int(item.get("skipped_text_token_layers", 0)) for item in cache_counters)
    reused = sum(int(item.get("num_actual_kv_reuse_queries", 0)) for item in cache_counters)
    if skipped <= 0 or reused <= 0:
        raise RuntimeError(
            "VLA-Cache completed without measured token-layer skipping and K/V reuse"
        )
    final = {
        "verdict": "VLA_CACHE_OFFLINE_COMPARISON_COMPLETE",
        "scientific_scope": experiment["scientific_scope"],
        "aggregate": aggregate,
        "vla_cache_measured_skipped_text_token_layers": skipped,
        "vla_cache_measured_kv_reuse_queries": reused,
        "real_robot_success_rate_measured": False,
        "next_required_for_task_success": (
            "run the same baseline, condition_loop, latentloop, vla_cache_full, and "
            "vla_cache rows on the inference computer with a paired robot protocol"
        ),
    }
    _write_json(output / "comparison_summary.json", final)
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
