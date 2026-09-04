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

from .dataset import RealSimVLADataset
from .distributed import initialize_distributed, seed_process
from .io_utils import atomic_write_json, sha256_file, sha256_text
from .model_io import load_exact_official_model, official_base_identity


CACHE_SCHEMA = "simvla_real_condition_cache_v1"
CONDITION_TOKENS = 122
CONDITION_DIM = 960


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
            if output.exists():
                raise FileExistsError(f"condition cache already exists: {output}")
            if building.exists() and not args.resume:
                raise FileExistsError(
                    f"incomplete cache exists; inspect it and pass --resume: {building}"
                )
            if not building.exists():
                _create_arrays(building, len(dataset))
                atomic_write_json(building / "records.json", dataset.records())
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
        layout_payload = None
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
                if layout_payload is None:
                    tokenizer = processor.tokenizer
                    layout_payload = build_condition_token_layout(
                        condition=condition,
                        image_mask=image_mask,
                        input_ids=input_ids,
                        pad_token_id=getattr(tokenizer, "pad_token_id", None),
                        special_token_ids=getattr(tokenizer, "all_special_ids", ()),
                    ).serializable()
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
            identity = {
                "schema": CACHE_SCHEMA,
                "dataset_identity_sha256": dataset_payload["dataset_identity_sha256"],
                "official_model_weights_sha256": base.model_weights_sha256,
                "records_sha256": records_sha,
                "preprocessing": dataset_payload["image_contract"]["model_preprocessing"],
            }
            result = {
                "schema_version": CACHE_SCHEMA,
                "verdict": "REAL_CONDITION_CACHE_PASS",
                "count": len(dataset),
                "shape": [len(dataset), CONDITION_TOKENS, CONDITION_DIM],
                "dtype": "float32",
                "dataset_manifest": str(dataset_manifest),
                "dataset_identity_sha256": dataset_payload["dataset_identity_sha256"],
                "official_base": base.to_dict(),
                "exact_loading": loading,
                "records_sha256": records_sha,
                "condition_cache_identity_sha256": sha256_text(
                    json.dumps(identity, sort_keys=True)
                ),
                "arrays": {
                    name: {"sha256": sha256_file(building / name), "size_bytes": (building / name).stat().st_size}
                    for name in ("condition.npy", "proprio.npy", "action.npy", "complete.npy")
                },
                "token_layout": layout_payload,
            }
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
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("verdict") != "REAL_CONDITION_CACHE_PASS":
            raise ValueError("real condition cache is not complete")
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

