"""EGL and robust-finalization control layer for native V0 diagnostics.

The original diagnostic evaluator is retained byte-for-byte because its hash is
part of existing manifests.  This entry point records its own hash in new
manifests, enforces EGL, and replaces only the final NCCL barrier with a
filesystem rendezvous followed by a short collective.  Ranks can therefore
finish uneven LIBERO task shards without hitting PyTorch's ten-minute
collective timeout.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

import torch.distributed as dist

from architectures.simvla.adapters.latentloop.native_v0_long_eval import (
    _canonical_hash,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import write_json
from architectures.simvla.adapters.latentloop.source_lock import sha256_file
from methods.latentloop.modules.native_simvla_v0 import NativeSimVLAV0
from tools.simvla import native_v0_intermediate_eval as evaluator


CONTROL_SCHEMA = "simvla_native_v0_intermediate_eval_control_v1"
FINAL_RENDEZVOUS_TIMEOUT_SECONDS = 4 * 60 * 60


def _require_egl() -> None:
    required = {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
    }
    mismatches = {
        key: os.environ.get(key)
        for key, expected in required.items()
        if os.environ.get(key) != expected
    }
    forbidden = {
        key: os.environ.get(key)
        for key in ("GALLIUM_DRIVER", "LIBGL_ALWAYS_SOFTWARE")
        if os.environ.get(key)
    }
    if mismatches or forbidden:
        raise RuntimeError(
            f"EGL-only contract failed: mismatches={mismatches}, forbidden={forbidden}"
        )


def _augment_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["distributed_finalization"] = {
        "schema_version": CONTROL_SCHEMA,
        "control_path": str(Path(__file__).resolve()),
        "control_sha256": sha256_file(Path(__file__).resolve()),
        "original_evaluator_path": str(Path(evaluator.__file__).resolve()),
        "original_evaluator_sha256": sha256_file(Path(evaluator.__file__).resolve()),
        "final_rendezvous": "filesystem_markers_then_nccl_barrier",
        "timeout_seconds": FINAL_RENDEZVOUS_TIMEOUT_SECONDS,
    }
    preflight_dir = Path(os.environ["SIMVLA_EGL_PREFLIGHT_DIR"]).expanduser().resolve()
    preflights: dict[str, Any] = {}
    gpu_ids = sorted(
        {
            int(gpu_id)
            for values in payload["row_runtime_gpu_ids"].values()
            for gpu_id in values
        }
    )
    for gpu_id in gpu_ids:
        preflight_path = preflight_dir / f"gpu_{gpu_id}.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight.get("verdict") != "EGL_PREFLIGHT_PASS":
            raise RuntimeError(f"EGL preflight did not pass: {preflight_path}")
        if int(preflight.get("physical_gpu_id", -1)) != gpu_id:
            raise RuntimeError(f"EGL preflight GPU mismatch: {preflight_path}")
        preflights[str(gpu_id)] = {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
            "gl_vendor": preflight.get("gl_vendor"),
            "gl_renderer": preflight.get("gl_renderer"),
            "mujoco_version": preflight.get("mujoco_version"),
        }
    payload["egl_preflights"] = preflights
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = _canonical_hash(payload)
    write_json(manifest_path, payload)
    return payload


def _install_parameter_audit_preservation() -> None:
    original_freeze = evaluator.freeze_module
    original_audit = NativeSimVLAV0.parameter_audit

    def freeze_with_audit(module: Any) -> None:
        if isinstance(module, NativeSimVLAV0):
            module._evaluation_parameter_audit = original_audit(module)
        original_freeze(module)

    def preserved_audit(self: NativeSimVLAV0) -> dict[str, int | bool]:
        cached = getattr(self, "_evaluation_parameter_audit", None)
        return dict(cached) if cached is not None else original_audit(self)

    evaluator.freeze_module = freeze_with_audit
    NativeSimVLAV0.parameter_audit = preserved_audit


def _install_final_rendezvous(output: str | Path) -> None:
    original_barrier: Callable[..., Any] = dist.barrier
    barrier_count = 0
    output_path = Path(output).expanduser().resolve()

    def controlled_barrier(*args: Any, **kwargs: Any) -> Any:
        nonlocal barrier_count
        barrier_count += 1
        if barrier_count == 1:
            return original_barrier(*args, **kwargs)
        if barrier_count != 2:
            raise RuntimeError(f"unexpected distributed barrier #{barrier_count}")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        rendezvous = output_path / ".final_rendezvous"
        rendezvous.mkdir(parents=True, exist_ok=True)
        write_json(
            rendezvous / f"rank_{rank}.json",
            {"rank": rank, "world_size": world_size, "finished_at_unix": time.time()},
        )
        expected = [rendezvous / f"rank_{index}.json" for index in range(world_size)]
        deadline = time.monotonic() + FINAL_RENDEZVOUS_TIMEOUT_SECONDS
        while not all(path.is_file() for path in expected):
            if time.monotonic() >= deadline:
                missing = [path.name for path in expected if not path.is_file()]
                raise TimeoutError(f"final filesystem rendezvous timed out; missing={missing}")
            time.sleep(2.0)
        return original_barrier(*args, **kwargs)

    evaluator.dist.barrier = controlled_barrier


def _prepare_evaluation(args: Any) -> None:
    _require_egl()
    selected = evaluator.parse_selected_gpu_ids(os.environ.get("SIMVLA_GPU_IDS"))
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(selected[local_rank])
    _install_parameter_audit_preservation()
    _install_final_rendezvous(args.output)


def main() -> int:
    args = evaluator.build_parser().parse_args()
    if args.command in {"manifest", "evaluate"}:
        _require_egl()
    if args.command == "evaluate":
        _prepare_evaluation(args)
    result = args.handler(args)
    if args.command == "manifest":
        result = _augment_manifest(args.output)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
