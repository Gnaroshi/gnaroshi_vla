"""Production-oriented SimVLA teacher-cache helpers for DCLD.

The cache stores teacher condition/action targets plus references back to the
original LIBERO HDF5 raw RGB frames. It intentionally avoids duplicating large
image tensors in normal operation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch

ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from methods.dcld.modules import DCLDCore, DeltaObservation, FastVisualDeltaEncoder  # noqa: E402
from methods.dcld.training import TeacherCacheMetadata, TeacherCacheShardWriter  # noqa: E402

from .simvla_action_adapter import SimVLAActionAdapter  # noqa: E402
from .simvla_condition_adapter import SimVLAConditionAdapter  # noqa: E402
from .simvla_delta_obs_adapter import (  # noqa: E402
    LIBERO_DCLD_CAMERA_ALIASES,
    LIBERO_RAW_RGB_CAMERAS,
    LiberoRawSampleRef,
    SimVLADeltaObsAdapter,
    iter_libero_raw_sample_refs,
    load_libero_raw_sample,
    proprio_to_tensor,
    raw_rgb_to_tensor,
)


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def run_cmd(args: list[str], cwd: str | Path = ROOT) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def tensor_stats(x: torch.Tensor) -> dict[str, Any]:
    y = x.detach()
    stats: dict[str, Any] = {
        "shape": list(y.shape),
        "dtype": str(y.dtype),
        "device": str(y.device),
    }
    if y.numel() == 0:
        return stats
    if not torch.is_floating_point(y):
        y = y.float()
    else:
        y = y.float()
    stats.update(
        {
            "min": float(y.min().item()),
            "max": float(y.max().item()),
            "mean": float(y.mean().item()),
            "std": float(y.std(unbiased=False).item()) if y.numel() > 1 else 0.0,
            "norm": float(y.norm().item()),
            "has_nan": bool(torch.isnan(y).any().item()),
        }
    )
    return stats


@contextmanager
def deterministic_rng(seed: int, device: torch.device):
    devices: list[int] = []
    if device.type == "cuda":
        devices = [device.index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield


def resolve_suite_meta(
    metas_path: str | Path,
    *,
    suite: str,
    output_dir: str | Path,
    upstream_root: str | Path = UPSTREAM,
) -> Path:
    """Create an absolute-path meta JSON, optionally filtered to one LIBERO suite."""

    metas_path = Path(metas_path)
    with metas_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    filtered = dict(meta)
    datalist = []
    for item in meta["datalist"]:
        item_dict = dict(item) if isinstance(item, dict) else {"path": item, "task": ""}
        raw_path = Path(item_dict["path"])
        abs_path = raw_path if raw_path.is_absolute() else (Path(upstream_root) / raw_path).resolve()
        if suite not in ("all", "standard", "libero_all"):
            path_text = str(abs_path)
            if f"/{suite}/" not in path_text and suite not in path_text:
                continue
        item_dict["path"] = str(abs_path)
        datalist.append(item_dict)

    if not datalist:
        raise ValueError(f"No metadata entries found for suite={suite!r} in {metas_path}")

    filtered["datalist"] = datalist
    filtered["num_files"] = len(datalist)
    filtered["num_episodes"] = len(datalist)
    filtered["source_metas_path"] = str(metas_path)
    filtered["filtered_suite"] = suite

    out_path = Path(output_dir) / "cache_generation_meta.json"
    write_json(out_path, filtered)
    return out_path


def load_official_batch_iterator(
    *,
    metas_path: str | Path,
    smolvlm_model_path: str,
    num_actions: int,
    image_size: int,
) -> Iterator[dict[str, Any]]:
    from datasets import create_smolvlm_dataloader
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor

    old_cwd = Path.cwd()
    os.chdir(UPSTREAM)
    try:
        loader = create_smolvlm_dataloader(
            batch_size=1,
            metas_path=str(metas_path),
            num_actions=num_actions,
            training=False,
            action_mode="libero_joint",
            num_workers=0,
            image_size=image_size,
        )
        iterator = iter(loader)
        processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_model_path)
        while True:
            batch = next(iterator)
            batch.update(processor.encode_language(batch["language_instruction"]))
            yield batch
    finally:
        os.chdir(old_cwd)


def model_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"].to(device),
        "image_input": batch["image_input"].to(device),
        "image_mask": batch["image_mask"].to(device),
        "proprio": batch["proprio"].to(device),
        "action": batch["action"].to(device),
    }


class SimVLATeacherCacheBuilder:
    """Generate teacher condition/action records from prepared SimVLA batches."""

    def __init__(
        self,
        condition_adapter: SimVLAConditionAdapter,
        writer: TeacherCacheShardWriter,
        *,
        action_adapter: SimVLAActionAdapter | None = None,
    ) -> None:
        self.condition_adapter = condition_adapter
        self.action_adapter = action_adapter or condition_adapter.action_adapter
        self.writer = writer

    def add_batches(
        self,
        batches: Iterable[dict[str, torch.Tensor]],
        *,
        include_action: bool = False,
        steps: int = 10,
    ) -> int:
        count = 0
        for batch in batches:
            out = self.condition_adapter.full_forward_return_latent(
                batch,
                return_action=include_action,
                steps=steps,
                deterministic=True,
            )
            sample = {
                "condition": out.condition.detach().cpu(),
                "aux": out.aux,
            }
            if out.action is not None:
                sample["teacher_action_chunk"] = out.action.detach().cpu()
            self.writer.add(sample)
            count += int(out.condition.shape[0])
        self.writer.flush()
        return count


def generate_teacher_cache(args: argparse.Namespace) -> Path:
    from models.modeling_smolvlm_vla import SmolVLMVLA

    cache_dir = Path(args.output).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else cache_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    generation_meta = resolve_suite_meta(
        args.metas_path,
        suite=args.suite,
        output_dir=cache_dir,
        upstream_root=UPSTREAM,
    )

    model = SmolVLMVLA.from_pretrained(args.checkpoint).to(device)
    model.eval()
    if Path(args.norm_stats).exists():
        model.action_space.load_norm_stats(str(args.norm_stats))
    for param in model.parameters():
        param.requires_grad_(False)

    condition_adapter = SimVLAConditionAdapter(model)
    action_adapter = SimVLAActionAdapter(model)
    metadata = TeacherCacheMetadata(
        architecture="simvla",
        checkpoint=args.checkpoint,
        dataset="LIBERO",
        condition_key="vlm_features",
        norm_stats_path=str(args.norm_stats),
        action_mode="libero_joint",
        notes=[
            "production teacher-cache format",
            "raw RGB stored as HDF5 references, not duplicated image tensors",
        ],
        extra={
            "suite": args.suite,
            "dataset_root": str(UPSTREAM / "datasets"),
            "source_metas_path": str(args.metas_path),
            "generation_meta_path": str(generation_meta),
            "action_horizon": args.steps,
            "raw_rgb_camera_names": list(LIBERO_RAW_RGB_CAMERAS),
            "raw_rgb_camera_aliases": list(LIBERO_DCLD_CAMERA_ALIASES),
            "vlm_input_resolution": args.image_size,
            "simvla_upstream_commit": run_cmd(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"]),
            "gnaroshi_vla_commit": run_cmd(["git", "rev-parse", "HEAD"]),
            "language_prompt_format": "task language_instruction from LIBERO meta",
            "denoising_config": {"steps": args.steps, "seed_base": args.seed},
            "dtype": "torch.float32",
            "cache_generation_command": " ".join(sys.argv),
        },
    )
    writer = TeacherCacheShardWriter(cache_dir, metadata, samples_per_shard=args.samples_per_shard)

    raw_refs = iter_libero_raw_sample_refs(
        generation_meta,
        upstream_root=UPSTREAM,
        num_actions=args.steps,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
    )
    batches = load_official_batch_iterator(
        metas_path=generation_meta,
        smolvlm_model_path=args.smolvlm_model_path,
        num_actions=args.steps,
        image_size=args.image_size,
    )

    count = 0
    for count, (raw_ref, batch_cpu) in enumerate(zip(raw_refs, batches), start=1):
        batch = model_batch_to_device(batch_cpu, device)
        with torch.no_grad():
            condition = condition_adapter.encode_condition(
                input_ids=batch["input_ids"],
                image_input=batch["image_input"],
                image_mask=batch["image_mask"],
            )
        denoising_seed = int(args.seed + count - 1)
        with torch.no_grad(), deterministic_rng(denoising_seed, device):
            teacher_action = action_adapter.decode_action_from_condition(
                condition,
                batch["proprio"],
                steps=args.steps,
                deterministic=False,
            )

        writer.add(
            {
                "episode_id": raw_ref.episode_id,
                "sample_id": raw_ref.sample_id,
                "timestep": raw_ref.timestep,
                "task_name": raw_ref.task_name,
                "language_instruction": raw_ref.language_instruction,
                "hdf5_path": raw_ref.hdf5_path,
                "demo_key": raw_ref.demo_key,
                "raw_rgb_ref": raw_ref.to_dict(),
                "raw_rgb_camera_names": list(raw_ref.camera_names),
                "raw_rgb_camera_aliases": list(raw_ref.camera_aliases),
                "raw_rgb_value_range_on_load": "[0,1]",
                "proprio": batch["proprio"].detach().cpu(),
                "condition": condition.detach().cpu(),
                "teacher_action_chunk": teacher_action.detach().cpu(),
                "dataset_action_chunk": batch["action"].detach().cpu(),
                "action_normalization_metadata": {
                    "norm_stats_path": str(args.norm_stats),
                    "action_mode": "libero_joint",
                    "postprocessed_teacher_action": True,
                },
                "denoising_config": {"steps": args.steps, "seed": denoising_seed},
            }
        )

    writer.close()
    write_cache_reports(cache_dir, report_dir=report_dir)
    return cache_dir


def load_cache_manifest(cache_dir: str | Path) -> dict[str, Any]:
    with (Path(cache_dir) / "manifest.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_cache_samples(cache_dir: str | Path) -> list[dict[str, Any]]:
    cache_dir = Path(cache_dir)
    manifest = load_cache_manifest(cache_dir)
    samples: list[dict[str, Any]] = []
    for shard in manifest.get("shards", []):
        samples.extend(torch.load(cache_dir / shard, map_location="cpu", weights_only=False))
    return samples


def iter_cache_samples_stream(cache_dir: str | Path) -> Iterator[dict[str, Any]]:
    cache_dir = Path(cache_dir)
    manifest = load_cache_manifest(cache_dir)
    for shard in manifest.get("shards", []):
        for sample in torch.load(cache_dir / shard, map_location="cpu", weights_only=False):
            yield sample


def _ensure_batched_condition(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    while x.ndim > 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim == 2:
        return x.unsqueeze(0)
    return x


def _ensure_batched_action(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    while x.ndim > 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim == 2:
        return x.unsqueeze(0)
    return x


def _ensure_batched_proprio(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x.to(device)
    return x.reshape(-1, x.shape[-1])


def build_transition_batch(
    prev_sample: dict[str, Any],
    cur_sample: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    prev_raw = load_libero_raw_sample(prev_sample["raw_rgb_ref"])
    cur_raw = load_libero_raw_sample(cur_sample["raw_rgb_ref"])
    delta_obs = SimVLADeltaObsAdapter().make_delta_observation_from_raw_samples(
        prev_raw,
        cur_raw,
        device=device,
        age=int(cur_sample["timestep"]) - int(prev_sample["timestep"]),
        metadata={
            "prev_sample_id": prev_sample.get("sample_id"),
            "cur_sample_id": cur_sample.get("sample_id"),
        },
    )
    return {
        "c_prev": _ensure_batched_condition(prev_sample["condition"], device),
        "c_teacher": _ensure_batched_condition(cur_sample["condition"], device),
        "action_teacher": _ensure_batched_action(cur_sample["teacher_action_chunk"], device),
        "proprio": _ensure_batched_proprio(cur_sample["proprio"], device),
        "delta_obs": delta_obs,
        "prev_sample_id": prev_sample.get("sample_id"),
        "cur_sample_id": cur_sample.get("sample_id"),
    }


def iter_transition_batches(
    cache_dir: str | Path,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> Iterator[dict[str, Any]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    yielded = 0
    prev: dict[str, Any] | None = None
    for cur in iter_cache_samples_stream(cache_dir):
        if prev is not None:
            is_adjacent = (
                prev.get("episode_id") == cur.get("episode_id")
                and int(cur.get("timestep", -999)) == int(prev.get("timestep", -999)) + 1
            )
            if is_adjacent:
                pair = (prev, cur)
                pairs.append(pair)
                yield build_transition_batch(prev, cur, device=device)
                yielded += 1
                if max_batches is not None and yielded >= max_batches:
                    return
        prev = cur

    if not pairs:
        raise RuntimeError(f"No adjacent transition pairs found in cache: {cache_dir}")

    if max_batches is None or yielded >= max_batches:
        return
    for prev, cur in cycle(pairs):
        if yielded >= max_batches:
            return
        yielded += 1
        yield build_transition_batch(prev, cur, device=device)


def write_cache_reports(cache_dir: str | Path, *, report_dir: str | Path) -> None:
    cache_dir = Path(cache_dir)
    report_dir = Path(report_dir)
    manifest = load_cache_manifest(cache_dir)
    first: dict[str, Any] = {}
    sample_count = 0
    for sample in iter_cache_samples_stream(cache_dir):
        if not first:
            first = sample
        sample_count += 1
    shapes = {
        "num_samples": sample_count,
        "condition": tensor_stats(first["condition"]) if first else None,
        "teacher_action_chunk": tensor_stats(first["teacher_action_chunk"]) if first else None,
        "proprio": tensor_stats(first["proprio"]) if first else None,
        "raw_rgb_ref": first.get("raw_rgb_ref") if first else None,
    }
    reload_check = {
        "cache_dir": str(cache_dir),
        "manifest_exists": (cache_dir / "manifest.json").exists(),
        "num_shards": len(manifest.get("shards", [])),
        "num_samples": sample_count,
        "condition_shape_ok": bool(first and list(first["condition"].shape)[-2:] == [122, 960]),
        "teacher_action_shape_ok": bool(first and list(first["teacher_action_chunk"].shape)[-2:] == [10, 7]),
        "proprio_shape_ok": bool(first and list(first["proprio"].shape)[-1:] == [8]),
        "raw_rgb_ref_valid": False,
        "no_condition_nan": bool(first and not torch.isnan(first["condition"].float()).any().item()),
        "no_action_nan": bool(first and not torch.isnan(first["teacher_action_chunk"].float()).any().item()),
    }
    if first:
        try:
            raw = load_libero_raw_sample(first["raw_rgb_ref"])
            reload_check["raw_rgb_ref_valid"] = True
            shapes["raw_rgb_loaded"] = {
                "rgb_shape": list(raw["rgb"].shape),
                "rgb_dtype": str(raw["rgb"].dtype),
                "proprio_shape": list(raw["proprio"].shape),
            }
        except Exception as exc:
            reload_check["raw_rgb_ref_error"] = str(exc)

    write_json(report_dir / "teacher_cache_smoke_metadata.json", manifest)
    write_json(report_dir / "teacher_cache_smoke_shapes.json", shapes)
    write_json(report_dir / "teacher_cache_reload_check.json", reload_check)
    write_text(
        report_dir / "teacher_cache_full_generator_report.md",
        "\n".join(
            [
                "# Teacher Cache Full Generator Report",
                "",
                f"- cache_dir: `{cache_dir}`",
                f"- num_samples: `{sample_count}`",
                f"- num_shards: `{len(manifest.get('shards', []))}`",
                f"- condition_shape_ok: `{reload_check['condition_shape_ok']}`",
                f"- teacher_action_shape_ok: `{reload_check['teacher_action_shape_ok']}`",
                f"- raw_rgb_ref_valid: `{reload_check['raw_rgb_ref_valid']}`",
                f"- metadata keys: `{sorted(manifest.get('metadata', {}).keys())}`",
                "",
                "The generator stores HDF5 raw RGB references rather than duplicating image arrays.",
            ]
        ),
    )


def run_raw_rgb_delta_smoke(args: argparse.Namespace) -> None:
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    generation_meta = resolve_suite_meta(
        args.metas_path,
        suite=args.suite,
        output_dir=out,
        upstream_root=UPSTREAM,
    )
    refs = list(
        iter_libero_raw_sample_refs(
            generation_meta,
            upstream_root=UPSTREAM,
            num_actions=args.steps,
            max_episodes=1,
            max_samples=2,
        )
    )
    if len(refs) < 2:
        raise RuntimeError("Need at least two consecutive raw LIBERO samples for delta smoke")
    prev = load_libero_raw_sample(refs[0])
    cur = load_libero_raw_sample(refs[1])
    adapter = SimVLADeltaObsAdapter()
    delta_obs = adapter.make_delta_observation_from_raw_samples(prev, cur, device=device, age=1)
    encoder = FastVisualDeltaEncoder(image_size=64, output_dim=512).to(device)
    with torch.no_grad():
        u_delta = encoder(delta_obs)
    key_rgb = raw_rgb_to_tensor(prev["rgb"])
    cur_rgb = raw_rgb_to_tensor(cur["rgb"])
    smoke = {
        "status": "passed",
        "camera_names": list(LIBERO_RAW_RGB_CAMERAS),
        "camera_aliases": list(LIBERO_DCLD_CAMERA_ALIASES),
        "key_rgb_shape": list(key_rgb.shape),
        "current_rgb_shape": list(cur_rgb.shape),
        "key_rgb_range": [float(key_rgb.min().item()), float(key_rgb.max().item())],
        "current_rgb_range": [float(cur_rgb.min().item()), float(cur_rgb.max().item())],
        "key_proprio_shape": list(proprio_to_tensor(prev["proprio"]).shape),
        "current_proprio_shape": list(proprio_to_tensor(cur["proprio"]).shape),
        "dt": 1.0,
        "age": 1,
        "u_delta": tensor_stats(u_delta),
        "u_delta_shape": list(u_delta.shape),
        "u_delta_norm": float(u_delta.float().norm(dim=-1).mean().item()),
        "no_nans": bool(not torch.isnan(u_delta.float()).any().item()),
        "prev_ref": refs[0].to_dict(),
        "cur_ref": refs[1].to_dict(),
    }
    write_json(out / "raw_rgb_sample_shapes.json", {"prev": smoke["prev_ref"], "cur": smoke["cur_ref"], **smoke})
    write_json(out / "raw_rgb_delta_smoke.json", smoke)
    write_text(
        out / "raw_rgb_visual_delta_wiring.md",
        "\n".join(
            [
                "# Raw RGB Visual Delta Wiring",
                "",
                "Production DCLD visual deltas are wired to raw LIBERO HDF5 RGB references.",
                "",
                f"- camera order: `{list(LIBERO_RAW_RGB_CAMERAS)}`",
                f"- camera aliases: `{list(LIBERO_DCLD_CAMERA_ALIASES)}`",
                "- front view maps to `agentview_rgb`.",
                "- wrist view maps to `eye_in_hand_rgb`.",
                "- frames are rotated 180 degrees to match the official SimVLA LIBERO loader.",
                "- tensors fed to `FastVisualDeltaEncoder` are float RGB in `[0,1]`.",
                "- tensor shape before encoder splitting is `[B, V, H, W, C]`.",
                "- teacher-cache training reloads raw frames from `raw_rgb_ref` instead of storing large image arrays.",
                "- real LIBERO eval should pass raw env RGB observations under the `raw_rgb` batch key.",
                "",
                f"- smoke u_delta_shape: `{smoke['u_delta_shape']}`",
                f"- smoke u_delta_norm: `{smoke['u_delta_norm']}`",
                f"- smoke no_nans: `{smoke['no_nans']}`",
            ]
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--metas-path", default=str(UPSTREAM / "datasets" / "metas" / "libero_train.json"))
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--output", default=str(ROOT / "results" / "simvla" / "dcld_cache" / "libero_10" / "simvla_libero_hf"))
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--samples-per-shard", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--raw-rgb-smoke-only", action="store_true")
    args = parser.parse_args()

    if args.report_dir is None:
        args.report_dir = args.output

    if args.raw_rgb_smoke_only:
        run_raw_rgb_delta_smoke(args)
    else:
        generate_teacher_cache(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
