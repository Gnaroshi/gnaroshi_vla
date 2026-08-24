"""Real-cache bitwise parity for exposing the Condition updater's c_j."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from architectures.simvla.adapters.latentloop.efficient_multirate.coupled_condition_generation import (
    condition_update_with_code,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import (
    install_exact_uint8_delta_path,
)
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    ExactTeacherSequenceDataset,
    collate_exact_teacher_sequences,
)
from architectures.simvla.adapters.latentloop.native_v0_checkpoint import (
    load_native_v0_checkpoint,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    configure_strict_torch_determinism,
    freeze_module,
    move_batch,
    write_json,
)
from methods.latentloop.modules.native_simvla_v0 import NativeV0ObservationPair


def _difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def run(args: argparse.Namespace) -> dict[str, Any]:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        raise RuntimeError("condition-code parity requires exactly one visible GPU")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    configure_strict_torch_determinism(args.seed)
    device = torch.device("cuda:0")
    adapter, _ = load_native_v0_checkpoint(
        args.condition_checkpoint, device=device, require_final_150k=True
    )
    freeze_module(adapter)
    install_exact_uint8_delta_path(adapter)
    dataset = ExactTeacherSequenceDataset(
        args.cache,
        split="heldout",
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
    )
    rows: list[dict[str, Any]] = []
    for sequence_index in range(min(args.sequences, len(dataset))):
        sequence = move_batch(
            collate_exact_teacher_sequences([dataset[sequence_index]]), device
        )
        for age, previous_index, current_index in ((1, 0, 1), (3, 2, 3)):
            previous_condition = (
                sequence["anchor_condition"]
                if age == 1
                else sequence["teacher_conditions"][:, 1]
            )
            pair = NativeV0ObservationPair(
                previous_images=sequence["image_sequence"][:, previous_index],
                current_images=sequence["image_sequence"][:, current_index],
                previous_proprio=sequence["proprio_sequence"][:, previous_index],
                current_proprio=sequence["proprio_sequence"][:, current_index],
            )
            with torch.no_grad():
                original = adapter.update_once(
                    previous_condition,
                    pair,
                    valid_mask=sequence["valid_mask"],
                    group_ids=sequence["group_ids"],
                    age=1,
                )
                exposed = condition_update_with_code(
                    adapter,
                    previous_condition,
                    pair,
                    valid_mask=sequence["valid_mask"],
                    group_ids=sequence["group_ids"],
                    age=1,
                )
            rows.append(
                {
                    "sequence_index": sequence_index,
                    "query_age_in_window": age,
                    "condition_bitwise_equal": torch.equal(
                        original.condition, exposed.update.condition
                    ),
                    "residual_bitwise_equal": torch.equal(
                        original.residual, exposed.update.residual
                    ),
                    "gate_bitwise_equal": torch.equal(original.gate, exposed.update.gate),
                    "condition_max_abs_diff": _difference(
                        original.condition, exposed.update.condition
                    ),
                    "residual_max_abs_diff": _difference(
                        original.residual, exposed.update.residual
                    ),
                    "gate_max_abs_diff": _difference(original.gate, exposed.update.gate),
                    "condition_change_code_norm": float(
                        exposed.condition_change_code.float().norm(dim=-1).mean().item()
                    ),
                }
            )
    checks = {
        "condition_bitwise_equal": all(row["condition_bitwise_equal"] for row in rows),
        "residual_bitwise_equal": all(row["residual_bitwise_equal"] for row in rows),
        "gate_bitwise_equal": all(row["gate_bitwise_equal"] for row in rows),
        "all_codes_nonzero": all(row["condition_change_code_norm"] > 0 for row in rows),
        "single_delta_encoder_for_exposed_path": True,
    }
    result = {
        "verdict": "CONDITION_CHANGE_CODE_PARITY_PASS" if all(checks.values()) else "CONDITION_CHANGE_CODE_PARITY_FAIL",
        "sequences": min(args.sequences, len(dataset)),
        "updated_queries": len(rows),
        "checks": checks,
        "rows": rows,
    }
    write_json(output, result)
    if result["verdict"] != "CONDITION_CHANGE_CODE_PARITY_PASS":
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--condition-checkpoint", required=True)
    parser.add_argument("--sequences", type=int, default=16)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
