#!/usr/bin/env python3
"""User-run full-checkpoint freeze and optimizer-filter gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from _common import DEFAULT_CHECKPOINT, load_local_policy, require_run, require_source_lock_v2
from architectures.openpi.adapters.latentloop.trainer import freeze_base_model, optimizer_parameter_names
from architectures.openpi.adapters.latentloop.transition_core import OpenPIKVLatentLoop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require_run(args.run, "OPENPI_LATENTLOOP_FREEZE_RUN")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite freeze audit output: {output}")
    lock_verification = require_source_lock_v2(args.source_lock)
    lock = json.loads(Path(args.source_lock).read_text(encoding="utf-8"))
    if Path(lock["checkpoint"]["directory"]).resolve() != Path(args.checkpoint).resolve():
        raise RuntimeError("checkpoint mismatch: freeze audit checkpoint differs from source lock v2")

    policy = load_local_policy(args.checkpoint, args.device, flow_steps=10)
    base_model = policy._model  # noqa: SLF001
    adapter = OpenPIKVLatentLoop().to(args.device)
    freeze_base_model(base_model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapter.parameters() if parameter.requires_grad], lr=1e-4
    )
    base_inventory = [
        {"name": name, "numel": parameter.numel(), "requires_grad": parameter.requires_grad}
        for name, parameter in base_model.named_parameters()
    ]
    adapter_names = optimizer_parameter_names(adapter)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    base_ids = {id(parameter) for parameter in base_model.parameters()}
    adapter_ids = {id(parameter) for parameter in adapter.parameters() if parameter.requires_grad}
    passed = bool(
        all(not row["requires_grad"] for row in base_inventory)
        and not (optimizer_ids & base_ids)
        and optimizer_ids == adapter_ids
        and adapter.trainable_parameters <= 19_000_000
    )
    payload = {
        "schema_version": 2,
        "BASE_FREEZE_PASS": passed,
        "markers": ["BASE_FREEZE_PASS"] if passed else [],
        "source_lock_id": lock_verification["source_lock_id"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "base_total_parameters": sum(row["numel"] for row in base_inventory),
        "base_trainable_parameters": sum(
            row["numel"] for row in base_inventory if row["requires_grad"]
        ),
        "adapter_trainable_parameters": adapter.trainable_parameters,
        "optimizer_parameter_count": sum(parameter.numel() for group in optimizer.param_groups for parameter in group["params"]),
        "optimizer_contains_base_parameter": bool(optimizer_ids & base_ids),
        "optimizer_exactly_matches_trainable_adapter": optimizer_ids == adapter_ids,
        "python_executable": os.path.realpath(os.sys.executable),
    }
    output.mkdir(parents=True)
    (output / "freeze_gate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "base_requires_grad_inventory.json").write_text(
        json.dumps(base_inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "optimizer_param_names.txt").write_text(
        "\n".join(adapter_names) + "\n", encoding="utf-8"
    )
    print(output / "freeze_gate.json")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
