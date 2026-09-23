from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


aggregate_module = _load(
    "aggregate_unified_policy_latency",
    "tools/seer/aggregate_unified_policy_latency.py",
)


def _fake_method(name: str, mean: float) -> dict:
    if name == "seer_full_k1":
        actions, full, updates, group = 8, 8, 0, "schedule"
    elif name.startswith("latentloop_k"):
        actions = int(name.removeprefix("latentloop_k"))
        full, updates, group = 1, actions - 1, "schedule"
    elif "latent_bridge_large_k4" in name:
        actions, full, updates = 4, 1, 3
        group = "schedule" if name.endswith("compiled") else "schedule_control"
    elif name.startswith("vla_cache"):
        actions, full, updates = 8, 8, 0
        group = "schedule" if name.endswith("reuse") else "schedule_control"
    else:
        actions, full, updates, group = 1, 0, 1, "component"
    samples = [mean * 0.9, mean * 1.1]
    summary = {"count": 2, "mean": mean, "std": mean * 0.1, "p50": mean,
               "p95": mean * 1.09, "p99": mean * 1.098, "min": samples[0], "max": samples[1]}
    return {
        "group": group,
        "actions_per_measured_call": actions,
        "full_forwards_per_measured_call": full,
        "lightweight_updates_per_measured_call": updates,
        "full_forward_fraction": full / actions,
        "wall_ms_per_action": summary,
        "cuda_event_ms_per_action": summary,
        "speedup_vs_same_process_seer_wall": 50.0 / mean,
        "raw_wall_ms_per_action": samples,
        "raw_cuda_event_ms_per_action": samples,
    }


def _fake_payload(replicate_id: str, baseline: float) -> dict:
    all_methods = (
        aggregate_module.PRIMARY_METHODS
        + aggregate_module.CONTROL_METHODS
        + aggregate_module.COMPONENT_METHODS
    )
    means = {name: 20.0 for name in all_methods}
    means["seer_full_k1"] = baseline
    for k in range(2, 9):
        means[f"latentloop_k{k}"] = (baseline + (k - 1) * 8.0) / k
    return {
        "status": "PASS",
        "replicate_id": replicate_id,
        "hardware": {"name": "NVIDIA GeForce RTX 3090"},
        "timing_contract": {"primary_metric": "wall", "included": ["model"], "excluded": ["sim"]},
        "software": {"torch": "test"},
        "assets": {"sha256": {"seer": "same"}},
        "source": {"file_sha256": {"model.py": "same"}},
        "benchmark": {"measured_cycles": 2, "warmup_cycles_excluded": 1},
        "input_audit": {"exact_shifted_window_overlap": True},
        "method_audit": {"adapter": "same"},
        "results": {name: _fake_method(name, means[name]) for name in all_methods},
        "vla_cache_runtime_validation": {"reuse": {"actual_kv_reuse_calls": 1}},
    }


def test_aggregate_uses_per_gpu_paired_seer_denominator(tmp_path: Path):
    paths = []
    baselines = [48.0, 50.0, 52.0, 54.0]
    for index, baseline in enumerate(baselines):
        path = tmp_path / f"gpu{index}.json"
        path.write_text(json.dumps(_fake_payload(f"gpu{index}", baseline)), encoding="utf-8")
        paths.append(path)

    result = aggregate_module.aggregate(paths)
    expected = sum(baseline / 20.0 for baseline in baselines) / len(baselines)
    assert result["aggregate"]["vla_cache_reuse"]["speedup_vs_paired_seer"]["mean"] == pytest.approx(expected)
    assert result["aggregate"]["seer_full_k1"]["speedup_vs_paired_seer"]["mean"] == pytest.approx(1.0)
    assert not result["diagnostics"]["latentloop_monotonic_violations"]


def test_aggregate_accepts_second_physical_gpu_partition(tmp_path: Path):
    paths = []
    for physical_gpu in range(4, 8):
        path = tmp_path / f"gpu{physical_gpu}.json"
        path.write_text(
            json.dumps(_fake_payload(f"gpu{physical_gpu}", 50.0)), encoding="utf-8"
        )
        paths.append(path)
    result = aggregate_module.aggregate(paths)
    assert result["replicate_ids"] == ["gpu4", "gpu5", "gpu6", "gpu7"]


def test_writer_emits_json_csv_and_markdown(tmp_path: Path):
    paths = []
    for index in range(4):
        path = tmp_path / f"gpu{index}.json"
        path.write_text(json.dumps(_fake_payload(f"gpu{index}", 50.0)), encoding="utf-8")
        paths.append(path)
    payload = aggregate_module.aggregate(paths)
    output = tmp_path / "analysis"
    aggregate_module.write_outputs(payload, output)
    assert (output / "unified_policy_latency.json").is_file()
    assert (output / "unified_policy_latency.csv").is_file()
    markdown = (output / "unified_policy_latency.md").read_text(encoding="utf-8")
    assert "LatentLoop (K=4)" in markdown
    assert "Latent Bridge Large (compiled)" in markdown
    assert "VLA-Cache (reuse)" in markdown


def test_vla_cache_reset_is_outside_timed_method_body():
    source = (REPO_ROOT / "tools/seer/unified_policy_latency.py").read_text(encoding="utf-8")
    body = source.split("    def vla_method():", 1)[1].split("    for name, config in", 1)[0]
    assert "reset_vla_cache_state" not in body
    assert "model.vla_cache_config" not in body


def test_launcher_uses_canonical_sd1_seer_assets():
    source = (
        REPO_ROOT
        / "architectures/seer/wrappers/latency/run_seer_unified_policy_latency.sh"
    ).read_text(encoding="utf-8")
    assert "seer_node2" not in source
    assert (
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/"
        "LIBERO_DATASETS/libero_10_converted"
    ) in source
    assert (
        "/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/"
        "vit_mae/mae_pretrain_vit_base.pth"
    ) in source
