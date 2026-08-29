#!/usr/bin/env python3
"""Run one bounded CPU contract test under an exact source lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from _common import require_run
from source_lock_v2 import verify_lock


ROOT = Path(__file__).resolve().parents[2]
CHECKS = {
    "v0_recursive": (
        "V0_RECURSIVE_SYNTHETIC_PASS",
        (
            "tests/openpi_latentloop/test_scientific_contract_v2.py::"
            "test_v0_ages_two_and_three_consume_predicted_states_with_full_gradient",
        ),
    ),
    "v1_causal": (
        "V1_CAUSAL_SYNTHETIC_PASS",
        (
            "tests/openpi_latentloop/test_scientific_contract_v2.py::"
            "test_v1_uses_actual_ordered_actions_and_level1_resets_only_recurrent_state",
            "tests/openpi_latentloop/test_scientific_contract_v2.py::"
            "test_v1_composition_loss_remains_connected_to_trainable_values",
        ),
    ),
    "v2_budget": (
        "V2_BUDGET_SYNTHETIC_PASS",
        (
            "tests/openpi_latentloop/test_scientific_contract_v2.py::"
            "test_v2_level_resets_and_maximum_ages",
            "tests/openpi_latentloop/test_scientific_contract_v2.py::"
            "test_episode_ordered_budget_counts_initial_full_and_terminal_partial_chunk",
        ),
    ),
    "operation_counters": (
        "OPERATION_COUNTER_CONTRACT_PASS",
        (
            "tests/openpi_latentloop/test_scientific_contract_v2.py::"
            "test_operation_counter_names_and_level2_costs_are_explicit",
        ),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=tuple(CHECKS), required=True)
    parser.add_argument("--source-lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    require_run(args.run, "OPENPI_LATENTLOOP_SYNTHETIC_RUN")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite or reuse synthetic-check output: {output}")
    lock_result = verify_lock(args.source_lock)
    lock_payload = json.loads(Path(args.source_lock).read_text(encoding="utf-8"))
    marker, tests = CHECKS[args.check]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    python_paths = (
        str(ROOT),
        str(ROOT / "architectures" / "openpi" / "upstream" / "src"),
        str(ROOT / "architectures" / "openpi" / "upstream" / "packages" / "openpi-client" / "src"),
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (*python_paths, environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"bounded synthetic check {args.check} failed without creating output:\n{completed.stdout}"
        )
    payload = {
        "schema_version": 2,
        marker: True,
        "markers": [marker],
        "check": args.check,
        "source_lock": str(Path(args.source_lock).resolve()),
        "source_lock_id": lock_result["source_lock_id"],
        "checkpoint_sha256": lock_payload["checkpoint"]["model_sha256"],
        "config_sha256": lock_payload["checkpoint"]["config_sha256"],
        "norm_stats_sha256": lock_payload["normalization"]["sha256"],
        "tests": list(tests),
        "command": command,
        "cuda_visible_devices": "",
    }
    output.mkdir(parents=True)
    (output / "pytest.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "verification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output / "verification.json")


if __name__ == "__main__":
    main()
