"""Recompute a bounded, episode-covering sample of a real Condition cache."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from architectures.simvla.adapters.latentloop_real_deploy.contracts import (
    sha256_directory,
    sha256_json,
)

from .condition_cache import _AllSplits, validate_real_condition_cache
from .io_utils import atomic_write_json, sha256_file
from .model_io import load_exact_official_model, official_base_identity


ATTESTATION_SCHEMA = "simvla_real_condition_cache_attestation_v2"
ATTESTATION_VERDICT = "REAL_CONDITION_CACHE_ATTESTATION_PASS"
SELECTION_STRATEGY = "lower_median_valid_h10_query_per_episode"


def select_episode_records(
    records: Iterable[Mapping[str, Any]],
    episode_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Select one deterministic, non-boundary cached query per episode."""

    grouped: dict[str, list[Mapping[str, Any]]] = {
        str(episode_id): [] for episode_id in episode_ids
    }
    for row in records:
        episode_id = str(row.get("episode_id", ""))
        if episode_id in grouped:
            grouped[episode_id].append(row)
    missing = sorted(episode_id for episode_id, rows in grouped.items() if not rows)
    if missing:
        raise ValueError(f"condition cache contains no valid query for episodes: {missing}")

    selected: list[dict[str, Any]] = []
    for episode_id in sorted(grouped):
        rows = sorted(
            grouped[episode_id],
            key=lambda row: (int(row["frame_index"]), int(row["cache_index"])),
        )
        row = rows[(len(rows) - 1) // 2]
        selected.append(
            {
                "episode_id": episode_id,
                "split": str(row["split"]),
                "frame_index": int(row["frame_index"]),
                "cache_index": int(row["cache_index"]),
            }
        )
    return selected


def _attestation_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key != "attestation_identity_sha256"
    }


def _cache_file_stats(cache_root: Path, cache: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the attestation to the files that were fully hashed at creation."""

    relative_paths = ["manifest.json", "records.json", *sorted(cache["arrays"])]
    result: dict[str, Any] = {}
    for relative_path in relative_paths:
        stat = (cache_root / relative_path).stat()
        result[relative_path] = {
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return result


def validate_condition_cache_attestation(
    path: str | Path,
    *,
    condition_cache: str | Path,
    checkpoint: str | Path,
    processor: str | Path,
    norm_stats: str | Path,
    verify_cache_array_checksums: bool,
) -> dict[str, Any]:
    """Validate an attestation against the exact artifacts selected now."""

    attestation_path = Path(path).expanduser().resolve()
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    cache_root = Path(condition_cache).expanduser().resolve()
    cache_manifest_path = cache_root / "manifest.json"
    cache = validate_real_condition_cache(
        cache_root, verify_array_checksums=verify_cache_array_checksums
    )
    dataset_manifest_path = Path(cache["dataset_manifest"]).expanduser().resolve()
    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    base = official_base_identity(checkpoint, processor)
    processor_sha = sha256_directory(processor)
    verifier_sha = sha256_file(Path(__file__).resolve())

    expected = {
        "schema_version": ATTESTATION_SCHEMA,
        "verdict": ATTESTATION_VERDICT,
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "condition_array_sha256": cache["arrays"]["condition.npy"]["sha256"],
        "records_sha256": cache["records_sha256"],
        "condition_cache_identity_sha256": cache[
            "condition_cache_identity_sha256"
        ],
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "dataset_identity_sha256": dataset["dataset_identity_sha256"],
        "official_model_weights_sha256": base.model_weights_sha256,
        "processor_directory_sha256": processor_sha,
        "norm_stats_sha256": sha256_file(norm_stats),
        "verifier_source_sha256": verifier_sha,
        "selection_strategy": SELECTION_STRATEGY,
        "cache_file_stats": _cache_file_stats(cache_root, cache),
    }
    mismatches = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    samples = payload.get("samples")
    comparison = payload.get("comparison")
    episode_ids = sorted(str(item["episode_id"]) for item in dataset["episodes"])
    if not isinstance(samples, list):
        mismatches["samples"] = {"observed": type(samples).__name__, "expected": "list"}
    else:
        observed_episodes = sorted(str(item.get("episode_id", "")) for item in samples)
        if observed_episodes != episode_ids:
            mismatches["sample_episode_coverage"] = {
                "observed": observed_episodes,
                "expected": episode_ids,
            }
    if not isinstance(comparison, Mapping):
        mismatches["comparison"] = {
            "observed": type(comparison).__name__,
            "expected": "object",
        }
    else:
        sample_count = len(samples) if isinstance(samples, list) else len(episode_ids)
        required_comparison = {
            "sample_count": sample_count,
            "exact_equal_count": sample_count,
            "all_samples_bitwise_equal": True,
            "max_abs_difference": 0.0,
            "mean_abs_difference": 0.0,
        }
        for key, value in required_comparison.items():
            if comparison.get(key) != value:
                mismatches[f"comparison.{key}"] = {
                    "observed": comparison.get(key),
                    "expected": value,
                }
    observed_identity = payload.get("attestation_identity_sha256")
    expected_identity = sha256_json(_attestation_identity_payload(payload))
    if observed_identity != expected_identity:
        mismatches["attestation_identity_sha256"] = {
            "observed": observed_identity,
            "expected": expected_identity,
        }
    if mismatches:
        raise ValueError(f"real Condition cache attestation mismatch: {mismatches}")
    return payload


def _chunks(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_attestation(args: argparse.Namespace) -> dict[str, Any]:
    cache_root = Path(args.condition_cache).expanduser().resolve()
    cache_manifest_path = cache_root / "manifest.json"
    cache = validate_real_condition_cache(
        cache_root, verify_array_checksums=True
    )
    dataset_manifest_path = Path(cache["dataset_manifest"]).expanduser().resolve()
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    records = json.loads((cache_root / "records.json").read_text(encoding="utf-8"))
    episode_ids = [str(item["episode_id"]) for item in dataset_manifest["episodes"]]
    selected = select_episode_records(records, episode_ids)
    dataset = _AllSplits(dataset_manifest_path)
    for row in selected:
        split, local_index = dataset.lookup[row["cache_index"]]
        episode_id, frame_index = dataset.datasets[split].samples[local_index]
        observed = (str(split), str(episode_id), int(frame_index))
        expected = (row["split"], row["episode_id"], row["frame_index"])
        if observed != expected:
            raise ValueError(
                "condition cache record no longer maps to the same dataset query: "
                f"cache_index={row['cache_index']} observed={observed} expected={expected}"
            )

    device = torch.device(args.device)
    model, processor, loading = load_exact_official_model(
        model_directory=args.checkpoint,
        processor_directory=args.processor,
        norm_stats=args.norm_stats,
        device=device,
        freeze_vlm=True,
        freeze_action_transformer=True,
    )
    model.eval()
    condition_array = np.load(cache_root / "condition.npy", mmap_mode="r")
    comparison_rows: list[dict[str, Any]] = []
    absolute_difference_sum = 0.0
    element_count = 0
    exact_equal_count = 0
    with torch.inference_mode():
        for rows in _chunks(selected, args.batch_size):
            items = [dataset[row["cache_index"]] for row in rows]
            images = torch.stack([item["image_input"] for item in items]).to(device)
            image_mask = torch.stack([item["image_mask"] for item in items]).to(device)
            instructions = [str(item["language_instruction"]) for item in items]
            input_ids = processor.encode_language(instructions)["input_ids"].to(device)
            recomputed_batch = model.forward_vlm_efficient(
                images, image_mask, input_ids
            )["vlm_features"].float().cpu().numpy()
            for row, recomputed in zip(rows, recomputed_batch):
                cached = np.array(condition_array[row["cache_index"]], copy=True)
                exact = bool(np.array_equal(recomputed, cached))
                difference = np.abs(
                    recomputed.astype(np.float64) - cached.astype(np.float64)
                )
                denominator = float(
                    np.linalg.norm(recomputed.astype(np.float64).reshape(-1))
                    * np.linalg.norm(cached.astype(np.float64).reshape(-1))
                )
                cosine = (
                    float(
                        np.dot(
                            recomputed.astype(np.float64).reshape(-1),
                            cached.astype(np.float64).reshape(-1),
                        )
                        / denominator
                    )
                    if denominator > 0.0
                    else float(exact)
                )
                exact_equal_count += int(exact)
                absolute_difference_sum += float(difference.sum())
                element_count += int(difference.size)
                comparison_rows.append(
                    {
                        **row,
                        "bitwise_equal": exact,
                        "max_abs_difference": float(difference.max(initial=0.0)),
                        "mean_abs_difference": float(difference.mean()),
                        "cosine_similarity": cosine,
                    }
                )

    max_abs = max(row["max_abs_difference"] for row in comparison_rows)
    all_exact = exact_equal_count == len(comparison_rows)
    base = official_base_identity(args.checkpoint, args.processor)
    payload: dict[str, Any] = {
        "schema_version": ATTESTATION_SCHEMA,
        "verdict": ATTESTATION_VERDICT if all_exact else "REAL_CONDITION_CACHE_ATTESTATION_FAIL",
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "condition_array_sha256": cache["arrays"]["condition.npy"]["sha256"],
        "records_sha256": cache["records_sha256"],
        "condition_cache_identity_sha256": cache[
            "condition_cache_identity_sha256"
        ],
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "dataset_identity_sha256": dataset_manifest["dataset_identity_sha256"],
        "official_model_weights_sha256": base.model_weights_sha256,
        "processor_directory_sha256": sha256_directory(args.processor),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "selection_strategy": SELECTION_STRATEGY,
        "cache_file_stats": _cache_file_stats(cache_root, cache),
        "samples": comparison_rows,
        "comparison": {
            "sample_count": len(comparison_rows),
            "exact_equal_count": exact_equal_count,
            "all_samples_bitwise_equal": all_exact,
            "max_abs_difference": float(max_abs),
            "mean_abs_difference": float(
                absolute_difference_sum / max(element_count, 1)
            ),
            "minimum_cosine_similarity": float(
                min(row["cosine_similarity"] for row in comparison_rows)
            ),
        },
        "model_loading": {
            "verdict": loading["verdict"],
            "loading_info": loading["loading_info"],
            "action_transformer_reinitialized": loading[
                "action_transformer_reinitialized"
            ],
            "real_action_overlay": loading["real_action_overlay"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_type": device.type,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }
    payload["attestation_identity_sha256"] = sha256_json(
        _attestation_identity_payload(payload)
    )
    atomic_write_json(args.output, payload)
    if not all_exact:
        raise RuntimeError(
            "Condition cache recomputation was not bitwise identical; "
            f"max_abs_difference={max_abs}. The cache is not authorized for reuse."
        )
    validate_condition_cache_attestation(
        args.output,
        condition_cache=cache_root,
        checkpoint=args.checkpoint,
        processor=args.processor,
        norm_stats=args.norm_stats,
        verify_cache_array_checksums=False,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    result = build_attestation(args)
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "sample_count": result["comparison"]["sample_count"],
                "attestation_identity_sha256": result[
                    "attestation_identity_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
