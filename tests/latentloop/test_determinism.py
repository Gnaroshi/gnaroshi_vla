"""CPU tests for the strict SimVLA online determinism contract."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from architectures.simvla.adapters.latentloop.determinism import (
    REQUIRED_PROCESS_ENV,
    compare_manifest_contracts,
    configure_strict_determinism,
    evaluation_episode_seed,
    exact_hash,
    resolve_seed_plan,
    seed_all,
)
from architectures.simvla.adapters.latentloop.online_evaluator import EvalRow, planned_rows
from architectures.simvla.adapters.latentloop.repeat_verifier import verify_repeat


def test_single_experiment_seed_derives_stable_disjoint_namespaces() -> None:
    first = resolve_seed_plan(
        experiment_seed=20260815,
        environment_seed_base=7,
        action_noise_seed_base=17,
        bootstrap_seed=19,
    )
    second = resolve_seed_plan(
        experiment_seed=20260815,
        environment_seed_base=999,
        action_noise_seed_base=999,
        bootstrap_seed=999,
    )
    assert first == second
    assert first.environment_seed_base != first.action_noise_seed_base
    assert evaluation_episode_seed(first.environment_seed_base, "libero_10", 4, 2) == (
        evaluation_episode_seed(first.environment_seed_base, "libero_10", 4, 2)
    )
    assert evaluation_episode_seed(first.environment_seed_base, "libero_10", 4, 2) != (
        evaluation_episode_seed(first.environment_seed_base, "libero_10", 4, 3)
    )


def test_official_compatible_baseline_has_no_adapter_path() -> None:
    rows = planned_rows(SimpleNamespace(matrix="official_compatible_baseline"))
    assert rows == [EvalRow("full_k1", "full", 1, None)]


def test_seed_all_replays_python_numpy_and_torch_rngs() -> None:
    seed_all(123)
    first = (random.random(), np.random.rand(), torch.rand(4))
    seed_all(123)
    second = (random.random(), np.random.rand(), torch.rand(4))
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_exact_hash_is_mapping_order_independent_and_byte_exact() -> None:
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    assert exact_hash({"a": tensor, "b": 3}) == exact_hash({"b": 3, "a": tensor})
    changed = tensor.clone()
    changed[0, 1] = torch.nextafter(changed[0, 1], torch.tensor(float("inf")))
    assert exact_hash(tensor) != exact_hash(changed)


def test_strict_runtime_fails_closed_then_enables_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="before Python starts"):
        configure_strict_determinism(7)
    for name, value in REQUIRED_PROCESS_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    snapshot = configure_strict_determinism(7)
    assert snapshot["backend"]["deterministic_algorithms"] is True
    assert isinstance(snapshot["backend"]["flash_sdp_enabled"], bool)
    assert isinstance(snapshot["backend"]["mem_efficient_sdp_enabled"], bool)
    assert isinstance(snapshot["backend"]["math_sdp_enabled"], bool)


def _write_repeat(path: Path, *, trace_hash: str, runtime_hash: str = "runtime") -> None:
    path.mkdir()
    manifest = {
        "protocol": "simvla_online_determinism_v1",
        "runtime_sha256": runtime_hash,
        "run_contract_sha256": "run",
    }
    results = {
        "protocol": "simvla_online_determinism_v1",
        "run_contract_sha256": "run",
        "trajectory_sha256": trace_hash,
        "episodes": [
            {
                "row": "full_k1",
                "task_id": 0,
                "episode": 0,
                "success": True,
                "episode_trace_hash": trace_hash,
            }
        ],
    }
    (path / "determinism_manifest.json").write_text(json.dumps(manifest))
    (path / "deterministic_results.json").write_text(json.dumps(results))


def test_repeat_verifier_requires_exact_episode_trace(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_repeat(reference, trace_hash="same")
    _write_repeat(candidate, trace_hash="same")
    assert verify_repeat(reference, candidate)["passed"] is True

    (candidate / "deterministic_results.json").unlink()
    results = json.loads((reference / "deterministic_results.json").read_text())
    results["trajectory_sha256"] = "different"
    results["episodes"][0]["episode_trace_hash"] = "different"
    (candidate / "deterministic_results.json").write_text(json.dumps(results))
    failed = verify_repeat(reference, candidate)
    assert failed["passed"] is False
    assert failed["mismatch_count"] == 1


def test_manifest_comparison_rejects_runtime_change() -> None:
    with pytest.raises(RuntimeError, match="contract mismatch"):
        compare_manifest_contracts(
            {
                "protocol": "simvla_online_determinism_v1",
                "runtime_sha256": "gpu-a",
                "run_contract_sha256": "run",
            },
            {
                "protocol": "simvla_online_determinism_v1",
                "runtime_sha256": "gpu-b",
                "run_contract_sha256": "run",
            },
        )
