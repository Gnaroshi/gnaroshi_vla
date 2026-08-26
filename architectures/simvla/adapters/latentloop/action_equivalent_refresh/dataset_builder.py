"""Build compact same-noise action-fidelity data from the locked exact cache."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.action_adapter import (
    ActionNoiseKey,
    explicit_action_noise,
)
from architectures.simvla.adapters.latentloop.action_equivalent_refresh.extraction import (
    extract_all_anchor_fidelity_records,
)
from architectures.simvla.adapters.latentloop.action_equivalent_refresh.features import (
    SimVLAActionFidelityFeatureConfig,
)
from architectures.simvla.adapters.latentloop.action_equivalent_refresh.training import (
    DATASET_SCHEMA,
    save_compact_action_fidelity_dataset,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    _drop_unused_vlm,
    collate_exact_teacher_sequences,
    validate_exact_cache,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import (
    load_generation_checkpoint,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_hidden import (
    full_generation_step_with_hidden,
)
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    load_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    configure_strict_torch_determinism,
    freeze_module,
    load_frozen_simvla,
    move_batch,
)
from architectures.simvla.adapters.latentloop.source_lock import sha256_file
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    GENERATION_NG3_FULL_INDICES,
)
from architectures.simvla.adapters.latentloop.stability_alignment.data import (
    StabilityExactTeacherDataset,
)
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop
from methods.latentloop.modules.action_equivalent_refresh import (
    CounterfactualActionTargets,
)


DEFAULT_CACHE = (
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/"
    "simvla/latentloop/simvla_efficient_coupled_multirate_latentloop_sigfix_v1/"
    "03_exact_teacher_cache"
)
DEFAULT_CONDITION = (
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/"
    "simvla/latentloop/correct_native_v0_seed20260815_v1/08_train_150k/"
    "checkpoints/native_v0_step_150000.pt"
)
DEFAULT_GENERATION = (
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/artifacts/"
    "simvla/generation_eval_bundle_20260824_v1/checkpoint/"
    "generation_step_030000.pt"
)
DEFAULT_NORM = (
    "/home/mingyujung/private/gnaroshi_vla/architectures/simvla/upstream/"
    "norm_stats/libero_norm.json"
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _arm_scale_from_norm_stats(path: str | Path) -> tuple[Tensor, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    stats = payload.get("norm_stats", payload)
    values = [float(value) for value in stats["actions"]["std"][:6]]
    scale = torch.tensor(values, dtype=torch.float32)
    if scale.shape != (6,) or bool((scale <= 0).any()):
        raise ValueError("official action std does not define six positive arm scales")
    return scale, {
        "definition": "official 6500-demo LIBERO action std, dimensions 0--5",
        "values": values,
        "norm_stats": str(source),
        "norm_stats_sha256": sha256_file(source),
        "num_demos": int(payload.get("metadata", {}).get("num_demos", -1)),
    }


class FixedNG3SameNoiseDecoder:
    """Exact runtime-equivalent uncoupled N_G=3 action decoder."""

    def __init__(self, model: Any, action_adapter: Any, updater: Any) -> None:
        self.model = model
        self.action_adapter = action_adapter
        self.updater = updater.eval()
        self.loop = SimVLAGenerationLoop(
            self.updater, self.model.transformer.action_decoder
        ).to(next(self.model.parameters()).device).eval()

    def __call__(self, condition: Tensor, proprio: Tensor, noise: Tensor) -> Tensor:
        normalized = self.action_adapter.normalize_proprio(proprio)

        def full_step(noisy_action: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
            output = full_generation_step_with_hidden(
                self.model.transformer,
                condition=condition,
                noisy_action=noisy_action,
                proprio=normalized,
                tau=tau,
                dt=-0.1,
            )
            return output.action_hidden, output.velocity

        with torch.no_grad():
            trace = self.loop(
                noise,
                full_step=full_step,
                full_step_indices=GENERATION_NG3_FULL_INDICES,
                proprio=normalized,
                condition=condition,
                condition_valid_mask=None,
                condition_change_code=condition.new_zeros(
                    (condition.shape[0], self.updater.condition_code_dim)
                ),
            )
            return self.action_adapter.action_space.postprocess(
                trace.final_noisy_action
            )


def _q0_noise(
    dataset: StabilityExactTeacherDataset,
    query_ids: Sequence[Sequence[str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    rows: list[Tensor] = []
    for window in query_ids:
        item = dataset.store.query(str(window[0]))
        metadata = item["metadata"]
        raw = metadata["noise_key"]
        key = ActionNoiseKey(
            checkpoint=str(raw["checkpoint"]),
            task_id=int(raw["task_id"]),
            episode_id=str(raw["episode_id"]),
            policy_query_index=int(raw["policy_query_index"]),
            seed_base=int(raw["seed_base"]),
        )
        if key.seed() != int(item["noise_seed"]):
            raise RuntimeError("q0 exact-cache action-noise key changed")
        rows.append(
            explicit_action_noise(
                key,
                batch_size=1,
                action_horizon=10,
                action_dim=7,
                device=device,
                dtype=dtype,
            )[0]
        )
    return torch.stack(rows)


def _source_metadata(
    args: argparse.Namespace,
    *,
    dataset: StabilityExactTeacherDataset,
    condition_payload: dict[str, Any],
    generation_payload: dict[str, Any],
    scale_metadata: dict[str, Any],
    indices: Sequence[int],
) -> dict[str, Any]:
    cache = Path(args.cache).expanduser().resolve()
    condition = Path(args.condition_checkpoint).expanduser().resolve()
    generation = Path(args.generation_checkpoint).expanduser().resolve()
    return {
        "cache": str(cache),
        "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
        "condition_checkpoint": str(condition),
        "condition_checkpoint_sha256": sha256_file(condition),
        "condition_optimizer_step": int(condition_payload["global_optimizer_step"]),
        "condition_scientific_primary": bool(
            condition_payload.get("scientific_primary_checkpoint")
        ),
        "generation_checkpoint": str(generation),
        "generation_checkpoint_sha256": sha256_file(generation),
        "generation_optimizer_step": int(generation_payload["optimizer_step"]),
        "fixed_generation_n_g": 3,
        "generation_full_step_indices": list(GENERATION_NG3_FULL_INDICES),
        "action_scale": scale_metadata,
        "split_contract": dataset.contract(),
        "split": str(args.split),
        "split_seed": int(args.split_seed),
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "sequence_indices": [int(value) for value in indices],
        "same_noise_candidate_exact": True,
        "all_reachable_exact_anchors": True,
        "uncoupled_condition_change_code_for_action_decode": True,
        "runtime_exact_tensors_persisted": False,
    }


def command_extract(args: argparse.Namespace) -> dict[str, Any]:
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing existing compact shard: {destination}")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("shard-index must be in [0,num-shards)")
    cache_status = validate_exact_cache(args.cache, verify_checksums=False)
    if not cache_status["passed"]:
        raise RuntimeError(f"exact cache failed validation: {cache_status}")
    configure_strict_torch_determinism(int(args.seed))
    device = torch.device(args.device)
    dataset = StabilityExactTeacherDataset(
        args.cache,
        split=args.split,
        split_seed=int(args.split_seed),
        max_age=3,
    )
    indices = list(range(int(args.shard_index), len(dataset), int(args.num_shards)))
    if int(args.max_sequences) > 0:
        indices = indices[: int(args.max_sequences)]
    if not indices:
        raise ValueError("selected shard contains no sequences")
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_exact_teacher_sequences,
        pin_memory=device.type == "cuda",
    )
    model, _, action_adapter = load_frozen_simvla(
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        smolvlm_model=args.smolvlm_model,
        device=device,
    )
    freeze_module(model)
    dropped = _drop_unused_vlm(model)
    condition, condition_payload = load_native_v0_checkpoint(
        args.condition_checkpoint,
        device=device,
        require_final_150k=bool(args.require_primary_condition),
    )
    generation, generation_payload = load_generation_checkpoint(
        args.generation_checkpoint, device=device
    )
    if int(generation_payload.get("optimizer_step", -1)) != 30_000:
        raise RuntimeError("primary Generation checkpoint must be optimizer step 30000")
    for module in (condition, generation):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    decoder = FixedNG3SameNoiseDecoder(model, action_adapter, generation)
    arm_scale_cpu, scale_metadata = _arm_scale_from_norm_stats(args.norm_stats)
    arm_scale = arm_scale_cpu.to(device)
    config = SimVLAActionFidelityFeatureConfig()
    collected: list[Any] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{args.split} shard{args.shard_index}"):
            query_ids = batch["query_ids"]
            moved = move_batch(batch, device)
            q0 = _q0_noise(
                dataset,
                query_ids,
                device=device,
                dtype=moved["anchor_condition"].dtype,
            )
            conditions = torch.cat(
                (moved["anchor_condition"].unsqueeze(1), moved["teacher_conditions"]),
                dim=1,
            )
            noises = torch.cat((q0.unsqueeze(1), moved["explicit_noises"]), dim=1)
            sequence_ids = [
                f"task{int(task):02d}:{episode}:q{int(anchor):06d}"
                for task, episode, anchor in zip(
                    moved["task_id"].detach().cpu().tolist(),
                    moved["episode_id"],
                    moved["anchor_query_index"].detach().cpu().tolist(),
                )
            ]
            episode_ids = [
                f"task{int(task):02d}:{episode}"
                for task, episode in zip(
                    moved["task_id"].detach().cpu().tolist(), moved["episode_id"]
                )
            ]
            collected.append(
                extract_all_anchor_fidelity_records(
                    condition_adapter=condition,
                    decode_same_noise=decoder,
                    exact_conditions=conditions,
                    image_sequence=moved["image_sequence"],
                    proprio_sequence=moved["proprio_sequence"],
                    explicit_noises=noises,
                    valid_mask=moved["valid_mask"],
                    group_ids=moved["group_ids"],
                    episode_ids=episode_ids,
                    sequence_ids=sequence_ids,
                    arm_scale=arm_scale,
                    feature_config=config,
                )
            )
    features = torch.cat([value.features.cpu() for value in collected], dim=0)
    targets = type(collected[0].targets)(
        **{
            name: torch.cat([getattr(value.targets, name).cpu() for value in collected])
            for name in (
                "arm_normalized_l1",
                "direction_cosine_error",
                "direction_valid",
                "gripper_mismatch",
            )
        }
    )
    starts = torch.cat(
        [value.episode_first_candidates.cpu() for value in collected], dim=0
    )
    episode_ids = [item for value in collected for item in value.episode_ids]
    routing_records = [
        item for value in collected for item in value.routing_records
    ]
    metadata = _source_metadata(
        args,
        dataset=dataset,
        condition_payload=condition_payload,
        generation_payload=generation_payload,
        scale_metadata=scale_metadata,
        indices=indices,
    )
    metadata["unused_vlm_drop"] = dropped
    save_compact_action_fidelity_dataset(
        destination,
        split=args.split,
        features=features,
        targets=targets,
        episode_first_candidates=starts,
        episode_ids=episode_ids,
        feature_config=config,
        source_metadata=metadata,
        routing_records=routing_records,
    )
    summary = {
        "verdict": "ACTION_FIDELITY_COMPACT_SHARD_COMPLETE",
        "output": str(destination),
        "split": args.split,
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "sequences": len(indices),
        "rows": int(features.shape[0]),
        "elapsed_seconds": time.perf_counter() - started,
        "cache_validation": cache_status["verdict"],
        "runtime_exact_tensors_persisted": False,
    }
    _atomic_json(destination.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _feature_config(raw: dict[str, Any]) -> SimVLAActionFidelityFeatureConfig:
    return SimVLAActionFidelityFeatureConfig(
        **{
            key: int(raw[key])
            for key in (
                "delta_dim",
                "proprio_dim",
                "action_dim",
                "first_r",
                "num_token_groups",
                "max_age",
            )
        }
    )


def command_merge(args: argparse.Namespace) -> dict[str, Any]:
    destination = Path(args.output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing existing merged data: {destination}")
    inputs = [Path(value).expanduser().resolve() for value in args.inputs]
    payloads = [torch.load(value, map_location="cpu", weights_only=False) for value in inputs]
    if not payloads:
        raise ValueError("merge needs at least one input")
    if any(value.get("schema_version") != DATASET_SCHEMA for value in payloads):
        raise ValueError("merge input schema mismatch")
    if any(value.get("split") != args.split for value in payloads):
        raise ValueError("merge input split mismatch")
    config = _feature_config(payloads[0]["feature_config"])
    if any(_feature_config(value["feature_config"]) != config for value in payloads):
        raise ValueError("merge input feature contracts differ")
    source_keys = (
        "cache_manifest_sha256",
        "condition_checkpoint_sha256",
        "generation_checkpoint_sha256",
        "fixed_generation_n_g",
        "split_seed",
    )
    reference = payloads[0]["source_metadata"]
    for value in payloads[1:]:
        source = value["source_metadata"]
        if any(source.get(key) != reference.get(key) for key in source_keys):
            raise ValueError("merge input source contracts differ")
    sequence_sets = [
        {str(item["sequence_id"]) for item in value.get("routing_records", ())}
        for value in payloads
    ]
    for left in range(len(sequence_sets)):
        for right in range(left + 1, len(sequence_sets)):
            if sequence_sets[left] & sequence_sets[right]:
                raise ValueError("compact shards contain duplicate sequences")
    targets = CounterfactualActionTargets(
        arm_normalized_l1=torch.cat([value["arm_normalized_l1"] for value in payloads]),
        direction_cosine_error=torch.cat([value["direction_cosine_error"] for value in payloads]),
        direction_valid=torch.cat([value["direction_valid"] for value in payloads]),
        gripper_mismatch=torch.cat([value["gripper_mismatch"] for value in payloads]),
    )
    metadata = dict(reference)
    metadata["merged_shards"] = [str(value) for value in inputs]
    metadata["sequence_indices"] = sorted(
        int(index)
        for value in payloads
        for index in value["source_metadata"]["sequence_indices"]
    )
    metadata.pop("shard_index", None)
    metadata["num_shards"] = len(payloads)
    save_compact_action_fidelity_dataset(
        destination,
        split=args.split,
        features=torch.cat([value["features"] for value in payloads]),
        targets=targets,
        episode_first_candidates=torch.cat(
            [value["episode_first_candidates"] for value in payloads]
        ),
        episode_ids=[item for value in payloads for item in value["episode_ids"]],
        feature_config=config,
        source_metadata=metadata,
        routing_records=[
            item for value in payloads for item in value.get("routing_records", ())
        ],
    )
    summary = {
        "verdict": "ACTION_FIDELITY_COMPACT_MERGE_COMPLETE",
        "output": str(destination),
        "split": args.split,
        "inputs": [str(value) for value in inputs],
        "sequences": len(set().union(*sequence_sets)),
        "rows": sum(int(value["features"].shape[0]) for value in payloads),
    }
    _atomic_json(destination.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--cache", default=DEFAULT_CACHE)
    extract.add_argument("--split", choices=("train", "checkpoint_validation", "final_offline"), required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    extract.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
    extract.add_argument("--norm-stats", default=DEFAULT_NORM)
    extract.add_argument("--condition-checkpoint", default=DEFAULT_CONDITION)
    extract.add_argument("--generation-checkpoint", default=DEFAULT_GENERATION)
    extract.add_argument("--require-primary-condition", action=argparse.BooleanOptionalAction, default=True)
    extract.add_argument("--split-seed", type=int, default=20260822)
    extract.add_argument("--seed", type=int, default=20260827)
    extract.add_argument("--shard-index", type=int, default=0)
    extract.add_argument("--num-shards", type=int, default=1)
    extract.add_argument("--max-sequences", type=int, default=0)
    extract.add_argument("--batch-size", type=int, default=1)
    extract.add_argument("--num-workers", type=int, default=0)
    extract.add_argument("--device", default="cuda")
    extract.set_defaults(handler=command_extract)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--split", choices=("train", "checkpoint_validation", "final_offline"), required=True)
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(handler=command_merge)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
