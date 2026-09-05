import argparse
import json

import pytest

from architectures.simvla.adapters.vla_cache.eval import implementation_identity
from architectures.simvla.adapters.vla_cache.summarize import summarize


def fixture_files(tmp_path):
    manifests = {f"seed{i:02d}": str(i) for i in (1, 2, 3)}
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"manifest_sha256": manifests, "row_summaries": [
        {"row": "full_nfe10", "episodes": 1500, "seeds": 3,
         "seed_mean_latency_per_action_ms": 20, "seed_mean_success_rate": .95}]}))
    for seed, sha in manifests.items():
        path = tmp_path / "vla_cache" / seed / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"verdict": "SIMVLA_VLA_CACHE_LIBERO_EVAL_COMPLETE",
            "episodes": 500, "successes": 450, "success_rate": .9,
            "implementation_identity": implementation_identity(), "manifest_declared_sha256": sha,
            "latency_per_executed_action_ms": 25, "text_token_layer_reduction": .2,
            "actual_kv_reuse_queries": 100, "peak_cuda_memory_gib": 6}))
    return argparse.Namespace(eval_root=str(tmp_path), reference_summary=str(reference), output=str(tmp_path / "aggregate"))


def test_summary_uses_native_not_inflated_control(tmp_path):
    args = fixture_files(tmp_path)
    result = summarize(args)
    assert result["vla_cache_three_seed"]["historical_speed_ratio_native_over_cache"] == .8
    assert result["comparison_axis"]["baseline_rerun"] is False


@pytest.mark.parametrize("field,value", [("episodes", 499), ("manifest_declared_sha256", "wrong"), ("implementation_identity", {})])
def test_summary_rejects_mixed_or_incomplete_results(tmp_path, field, value):
    args = fixture_files(tmp_path)
    path = tmp_path / "vla_cache/seed02/summary.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        summarize(args)
