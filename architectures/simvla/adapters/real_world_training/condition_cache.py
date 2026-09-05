"""Exact frozen-VLM condition cache for efficient real SimVLA adaptation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.native_v0_condition_hook import (
    build_condition_token_layout,
)

from .dataset import DATASET_SCHEMA, RealSimVLADataset
from .distributed import initialize_distributed, seed_process
from .io_utils import atomic_write_json, sha256_directory, sha256_file, sha256_text
from .model_io import load_exact_official_model, official_base_identity


CACHE_SCHEMA = "simvla_real_condition_cache_v2"
BUILDING_SCHEMA = "simvla_real_condition_cache_building_v1"
CONDITION_TOKENS = 122
CONDITION_DIM = 960
TOKEN_LAYOUT_FILENAME = "token_layout.json"


def canonical_condition_token_layout(payload: Any) -> dict[str, Any]:
    """Validate and canonicalize the fixed-instruction SimVLA token layout."""

    if not isinstance(payload, dict):
        raise ValueError("real condition cache token layout must be an object")
    valid_rows = payload.get("valid_mask")
    group_rows = payload.get("group_ids")
    if (
        not isinstance(valid_rows, list)
        or not valid_rows
        or not isinstance(group_rows, list)
        or len(group_rows) != len(valid_rows)
    ):
        raise ValueError("real condition cache token layout rows are invalid")
    first_valid = valid_rows[0]
    first_groups = group_rows[0]
    if (
        not isinstance(first_valid, list)
        or len(first_valid) != CONDITION_TOKENS
        or not all(value is True for value in first_valid)
        or not isinstance(first_groups, list)
        or len(first_groups) != CONDITION_TOKENS
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 6
            for value in first_groups
        )
    ):
        raise ValueError("real condition cache token layout contents are invalid")
    if any(row != first_valid for row in valid_rows[1:]) or any(
        row != first_groups for row in group_rows[1:]
    ):
        raise ValueError(
            "fixed-instruction real cache requires one invariant token layout"
        )
    image_tokens = int(payload.get("image_tokens_per_view", -1))
    text_tokens = int(payload.get("text_tokens", -1))
    if image_tokens < 1 or text_tokens < 1 or 2 * image_tokens + text_tokens != CONDITION_TOKENS:
        raise ValueError("real condition cache token partition is invalid")
    ranges = payload.get("sample_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("real condition cache token ranges are missing")
    canonical = dict(payload)
    canonical["valid_mask"] = [list(first_valid)]
    canonical["group_ids"] = [list(first_groups)]
    group_names = payload.get("group_names")
    if not isinstance(group_names, dict):
        raise ValueError("real condition cache token group names are missing")
    canonical["group_names"] = {
        str(key): str(value) for key, value in group_names.items()
    }
    canonical_range = dict(ranges[0])
    canonical_range["sample"] = 0
    canonical["sample_ranges"] = [canonical_range]
    return canonical


def validate_real_condition_cache(
    root: str | Path,
    *,
    verify_array_checksums: bool,
) -> dict[str, Any]:
    cache_root = Path(root).expanduser().resolve()
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("unsupported real condition cache schema")
    if manifest.get("verdict") != "REAL_CONDITION_CACHE_PASS":
        raise ValueError("real condition cache is not complete")
    count = int(manifest.get("count", -1))
    if count < 1 or manifest.get("shape") != [count, CONDITION_TOKENS, CONDITION_DIM]:
        raise ValueError("real condition cache shape contract changed")
    token_layout = canonical_condition_token_layout(manifest.get("token_layout"))
    if token_layout != manifest.get("token_layout"):
        raise ValueError("real condition cache token layout is not canonical")
    records_path = cache_root / "records.json"
    if sha256_file(records_path) != manifest.get("records_sha256"):
        raise ValueError("real condition cache records checksum mismatch")
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if len(records) != count or sorted(int(row["cache_index"]) for row in records) != list(
        range(count)
    ):
        raise ValueError("real condition cache records are not a complete index set")

    expected = {
        "condition.npy": ((count, CONDITION_TOKENS, CONDITION_DIM), np.dtype("float32")),
        "proprio.npy": ((count, 8), np.dtype("float32")),
        "action.npy": ((count, 10, 7), np.dtype("float32")),
        "complete.npy": ((count,), np.dtype("uint8")),
    }
    for name, (shape, dtype) in expected.items():
        path = cache_root / name
        specification = manifest.get("arrays", {}).get(name, {})
        if not path.is_file() or path.stat().st_size != int(specification.get("size_bytes", -1)):
            raise ValueError(f"real condition cache file size mismatch: {name}")
        array = np.load(path, mmap_mode="r")
        if tuple(array.shape) != shape or array.dtype != dtype:
            raise ValueError(f"real condition cache array contract mismatch: {name}")
        if name == "complete.npy" and not bool(np.all(array == 1)):
            raise ValueError("real condition cache completion bitmap is not all one")
        if verify_array_checksums and sha256_file(path) != specification.get("sha256"):
            raise ValueError(f"real condition cache checksum mismatch: {name}")

    dataset_manifest = Path(manifest["dataset_manifest"]).expanduser().resolve()
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if dataset.get("schema_version") != DATASET_SCHEMA:
        raise ValueError("real condition cache references an unsupported dataset schema")
    if sha256_file(dataset_manifest) != manifest.get("dataset_manifest_sha256"):
        raise ValueError("real condition cache dataset manifest checksum mismatch")
    if dataset.get("dataset_identity_sha256") != manifest.get("dataset_identity_sha256"):
        raise ValueError("real condition cache and dataset identities differ")
    identity = {
        "schema": CACHE_SCHEMA,
        "dataset_identity_sha256": manifest.get("dataset_identity_sha256"),
        "official_model_weights_sha256": manifest.get("official_base", {}).get(
            "model_weights_sha256"
        ),
        "records_sha256": manifest.get("records_sha256"),
        "array_sha256": {
            name: manifest.get("arrays", {}).get(name, {}).get("sha256")
            for name in (
                "condition.npy",
                "proprio.npy",
                "action.npy",
                "complete.npy",
            )
        },
        "preprocessing": dataset.get("image_contract", {}).get(
            "model_preprocessing"
        ),
    }
    expected_identity = sha256_text(json.dumps(identity, sort_keys=True))
    if manifest.get("condition_cache_identity_sha256") != expected_identity:
        raise ValueError("real condition cache identity is invalid")
    return manifest


class _AllSplits(Dataset[dict[str, Any]]):
    def __init__(self, manifest: str | Path) -> None:
        self.datasets = {
            split: RealSimVLADataset(manifest, split=split, training=False)
            for split in ("train", "validation")
        }
        self.lookup: list[tuple[str, int]] = []
        for split in ("train", "validation"):
            self.lookup.extend((split, index) for index in range(len(self.datasets[split])))

    def __len__(self) -> int:
        return len(self.lookup)

    def __getitem__(self, index: int) -> dict[str, Any]:
        split, local_index = self.lookup[index]
        result = dict(self.datasets[split][local_index])
        result["split"] = split
        result["cache_index"] = int(index)
        return result

    def records(self) -> list[dict[str, Any]]:
        records = []
        for index, (split, local_index) in enumerate(self.lookup):
            episode_id, frame_index = self.datasets[split].samples[local_index]
            records.append(
                {
                    "cache_index": index,
                    "split": split,
                    "episode_id": episode_id,
                    "frame_index": int(frame_index),
                }
            )
        return records


def validate_condition_cache_building(
    root: str | Path,
    *,
    dataset_manifest: str | Path,
    checkpoint: str | Path,
    processor: str | Path,
    norm_stats: str | Path,
) -> dict[str, Any]:
    """Prove an interrupted cache can be resumed under the same source contract."""

    manifest_path = Path(dataset_manifest).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = _AllSplits(manifest_path)
    records_text = json.dumps(dataset.records(), indent=2, sort_keys=True) + "\n"
    expected = _building_contract(
        dataset_manifest=manifest_path,
        dataset_payload=payload,
        records_sha256=sha256_text(records_text),
        count=len(dataset),
        checkpoint=checkpoint,
        processor=processor,
        norm_stats=norm_stats,
    )
    _validate_building_state(Path(root).expanduser().resolve(), expected)
    return expected


def _create_arrays(root: Path, count: int) -> None:
    specs = {
        "condition.npy": ((count, CONDITION_TOKENS, CONDITION_DIM), np.float32),
        "proprio.npy": ((count, 8), np.float32),
        "action.npy": ((count, 10, 7), np.float32),
        "complete.npy": ((count,), np.uint8),
    }
    root.mkdir(parents=True, exist_ok=False)
    for name, (shape, dtype) in specs.items():
        array = np.lib.format.open_memmap(root / name, mode="w+", dtype=dtype, shape=shape)
        array[...] = 0
        array.flush()


def _building_contract(
    *,
    dataset_manifest: Path,
    dataset_payload: dict[str, Any],
    records_sha256: str,
    count: int,
    checkpoint: str | Path,
    processor: str | Path,
    norm_stats: str | Path,
) -> dict[str, Any]:
    base = official_base_identity(checkpoint, processor)
    return {
        "schema_version": BUILDING_SCHEMA,
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "dataset_identity_sha256": dataset_payload["dataset_identity_sha256"],
        "records_sha256": records_sha256,
        "count": int(count),
        "official_model_weights_sha256": base.model_weights_sha256,
        "processor_directory": str(Path(processor).expanduser().resolve()),
        "processor_directory_sha256": sha256_directory(processor),
        "norm_stats_sha256": sha256_file(norm_stats),
    }


def _validate_building_state(root: Path, expected: dict[str, Any]) -> None:
    contract_path = root / "building_contract.json"
    observed = json.loads(contract_path.read_text(encoding="utf-8"))
    if observed != expected:
        raise ValueError(
            "incomplete Condition cache belongs to a different source contract"
        )
    if sha256_file(root / "records.json") != expected["records_sha256"]:
        raise ValueError("incomplete Condition cache records changed")
    count = int(expected["count"])
    arrays = {
        "condition.npy": ((count, CONDITION_TOKENS, CONDITION_DIM), np.dtype("float32")),
        "proprio.npy": ((count, 8), np.dtype("float32")),
        "action.npy": ((count, 10, 7), np.dtype("float32")),
        "complete.npy": ((count,), np.dtype("uint8")),
    }
    for name, (shape, dtype) in arrays.items():
        array = np.load(root / name, mmap_mode="r")
        if tuple(array.shape) != shape or array.dtype != dtype:
            raise ValueError(f"incomplete Condition cache array changed: {name}")


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = {
        key: torch.stack([item[key] for item in batch])
        for key in ("image_input", "image_mask", "proprio", "action")
    }
    return {
        **tensors,
        "language_instruction": [str(item["language_instruction"]) for item in batch],
        "cache_index": torch.tensor([int(item["cache_index"]) for item in batch]),
    }


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    context = initialize_distributed(args.device)
    seed_process(args.seed, context.rank)
    try:
        dataset_manifest = Path(args.dataset_manifest).expanduser().resolve()
        dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
        if dataset_payload.get("verdict") != "REAL_DATASET_CONTRACT_PASS":
            raise RuntimeError("dataset conversion contract has not passed")
        dataset = _AllSplits(dataset_manifest)
        output = Path(args.output).expanduser().resolve()
        building = output.with_name(f".{output.name}.building")
        if context.primary:
            records = dataset.records()
            records_text = json.dumps(records, indent=2, sort_keys=True) + "\n"
            expected_building = _building_contract(
                dataset_manifest=dataset_manifest,
                dataset_payload=dataset_payload,
                records_sha256=sha256_text(records_text),
                count=len(dataset),
                checkpoint=args.checkpoint,
                processor=args.processor,
                norm_stats=args.norm_stats,
            )
            if output.exists():
                raise FileExistsError(f"condition cache already exists: {output}")
            if building.exists() and not args.resume:
                raise FileExistsError(
                    f"incomplete cache exists; inspect it and pass --resume: {building}"
                )
            if building.exists():
                _validate_building_state(building, expected_building)
            else:
                _create_arrays(building, len(dataset))
                atomic_write_json(building / "records.json", records)
                atomic_write_json(
                    building / "building_contract.json", expected_building
                )
        context.barrier()

        model, processor, loading = load_exact_official_model(
            model_directory=args.checkpoint,
            processor_directory=args.processor,
            norm_stats=args.norm_stats,
            device=context.device,
            freeze_vlm=True,
            freeze_action_transformer=True,
        )
        model.eval()
        layout_path = building / TOKEN_LAYOUT_FILENAME
        if context.primary:
            sample = dataset[0]
            image_mask = sample["image_mask"].unsqueeze(0).to(context.device)
            input_ids = processor.encode_language(
                [str(sample["language_instruction"])]
            )["input_ids"].to(context.device)
            tokenizer = processor.tokenizer
            expected_layout = canonical_condition_token_layout(
                build_condition_token_layout(
                    condition=torch.empty(
                        (1, CONDITION_TOKENS, CONDITION_DIM),
                        device=context.device,
                    ),
                    image_mask=image_mask,
                    input_ids=input_ids,
                    pad_token_id=getattr(tokenizer, "pad_token_id", None),
                    special_token_ids=getattr(tokenizer, "all_special_ids", ()),
                ).serializable()
            )
            if layout_path.exists():
                observed_layout = canonical_condition_token_layout(
                    json.loads(layout_path.read_text(encoding="utf-8"))
                )
                if observed_layout != expected_layout:
                    raise ValueError(
                        "incomplete Condition cache token layout changed under resume"
                    )
            else:
                atomic_write_json(layout_path, expected_layout)
        context.barrier()
        layout_payload = canonical_condition_token_layout(
            json.loads(layout_path.read_text(encoding="utf-8"))
        )
        complete = np.load(building / "complete.npy", mmap_mode="r+")
        assigned = [
            index
            for index in range(context.rank, len(dataset), context.world_size)
            if not int(complete[index])
        ]
        loader = DataLoader(
            Subset(dataset, assigned),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=context.device.type == "cuda",
            collate_fn=_collate,
        )
        conditions = np.load(building / "condition.npy", mmap_mode="r+")
        proprios = np.load(building / "proprio.npy", mmap_mode="r+")
        actions = np.load(building / "action.npy", mmap_mode="r+")
        iterator = tqdm(loader, disable=not context.primary, desc="cache frozen SimVLA conditions")
        with torch.inference_mode():
            for batch in iterator:
                image_input = batch["image_input"].to(context.device, non_blocking=True)
                image_mask = batch["image_mask"].to(context.device, non_blocking=True)
                input_ids = processor.encode_language(batch["language_instruction"])[
                    "input_ids"
                ].to(context.device)
                encoded = model.forward_vlm_efficient(image_input, image_mask, input_ids)
                condition = encoded["vlm_features"].float()
                if tuple(condition.shape[1:]) != (CONDITION_TOKENS, CONDITION_DIM):
                    raise RuntimeError(
                        f"unexpected real condition shape: {tuple(condition.shape)}"
                    )
                indices = batch["cache_index"].numpy().astype(np.int64)
                conditions[indices] = condition.cpu().numpy()
                proprios[indices] = batch["proprio"].numpy()
                actions[indices] = batch["action"].numpy()
                conditions.flush()
                proprios.flush()
                actions.flush()
                complete[indices] = 1
                complete.flush()
        del conditions, proprios, actions, complete, model
        context.barrier()

        result: dict[str, Any] = {}
        if context.primary:
            final_complete = np.load(building / "complete.npy", mmap_mode="r")
            missing = np.flatnonzero(final_complete != 1).tolist()
            if missing:
                raise RuntimeError(f"condition cache is incomplete at indices {missing[:20]}")
            base = official_base_identity(args.checkpoint, args.processor)
            records_sha = sha256_file(building / "records.json")
            array_specs = {
                name: {
                    "sha256": sha256_file(building / name),
                    "size_bytes": (building / name).stat().st_size,
                }
                for name in (
                    "condition.npy",
                    "proprio.npy",
                    "action.npy",
                    "complete.npy",
                )
            }
            identity = {
                "schema": CACHE_SCHEMA,
                "dataset_identity_sha256": dataset_payload["dataset_identity_sha256"],
                "official_model_weights_sha256": base.model_weights_sha256,
                "records_sha256": records_sha,
                "array_sha256": {
                    name: specification["sha256"]
                    for name, specification in array_specs.items()
                },
                "preprocessing": dataset_payload["image_contract"]["model_preprocessing"],
            }
            result = {
                "schema_version": CACHE_SCHEMA,
                "verdict": "REAL_CONDITION_CACHE_PASS",
                "count": len(dataset),
                "shape": [len(dataset), CONDITION_TOKENS, CONDITION_DIM],
                "dtype": "float32",
                "dataset_manifest": str(dataset_manifest),
                "dataset_manifest_sha256": sha256_file(dataset_manifest),
                "dataset_identity_sha256": dataset_payload["dataset_identity_sha256"],
                "official_base": base.to_dict(),
                "exact_loading": loading,
                "records_sha256": records_sha,
                "condition_cache_identity_sha256": sha256_text(
                    json.dumps(identity, sort_keys=True)
                ),
                "arrays": array_specs,
                "token_layout": layout_payload,
            }
            layout_path.unlink(missing_ok=True)
            atomic_write_json(building / "manifest.json", result)
            os.replace(building, output)
        context.barrier()
        if not context.primary:
            result = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        return result
    finally:
        context.close()


class RealConditionCacheDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: str | Path, *, split: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = validate_real_condition_cache(
            self.root, verify_array_checksums=False
        )
        records = json.loads((self.root / "records.json").read_text(encoding="utf-8"))
        self.indices = [int(item["cache_index"]) for item in records if item["split"] == split]
        self.records = {int(item["cache_index"]): item for item in records}
        self.condition = np.load(self.root / "condition.npy", mmap_mode="r")
        self.proprio = np.load(self.root / "proprio.npy", mmap_mode="r")
        self.action = np.load(self.root / "action.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cache_index = self.indices[index]
        return {
            "cache_index": cache_index,
            "condition": torch.from_numpy(np.array(self.condition[cache_index], copy=True)),
            "proprio": torch.from_numpy(np.array(self.proprio[cache_index], copy=True)),
            "action": torch.from_numpy(np.array(self.action[cache_index], copy=True)),
            **self.records[cache_index],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    result = build_cache(build_parser().parse_args())
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps({"verdict": result["verdict"], "count": result["count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
