import argparse
import json

import pytest

from architectures.simvla.adapters.vla_cache.eval import implementation_identity
from tools.simvla import vla_cache_paper as paper


def rows():
    return [dict(suite=suite, seed=seed, episodes=500, successes=450 + 10 * i,
                 success_rate_percent=90 + 2 * i, latency_ms_per_action=20 + i)
            for suite in paper.SUITES for i, seed in enumerate(paper.SEEDS)]


def test_paper_aggregation_uses_sample_std_and_native_denominator():
    result = paper.aggregate(rows(), {"seed_mean_latency_per_action_ms": 42})
    assert result["four_suite_sr_mean"] == 92
    assert result["suites"]["libero_10"]["sr_sample_std"] == 2
    assert result["suites"]["libero_10"]["latency_sample_std"] == 1
    assert result["historical_long_speedup"] == 2


def test_paper_aggregation_rejects_missing_seed():
    with pytest.raises(ValueError, match="three complete distinct seeds"):
        paper.aggregate(rows()[:-1], {"seed_mean_latency_per_action_ms": 42})


def fixture_cell(tmp_path):
    path = tmp_path / "libero_10/vla_cache/seed01"
    path.mkdir(parents=True)
    data = [dict(suite="libero_10", task_id=t, trial_id=i, init_state_index=i,
                 environment_seed=7, episode_length=10, num_policy_queries=2,
                 num_action_transformer_calls=20, success=True, policy_latency_mean_ms=20)
            for t in range(10) for i in range(50)]
    manifest = dict(selected_episodes=data, manifest_file_sha256="matched")
    summary = dict(verdict="SIMVLA_VLA_CACHE_LIBERO_EVAL_COMPLETE", episodes=500, suite="libero_10",
                   implementation_identity=implementation_identity(), manifest_file_sha256="matched",
                   successes=500, success_rate=1, latency_per_executed_action_ms=20)
    meta = dict(hostname="jbr-TRX50", gpu="NVIDIA GeForce RTX 5090", model_dtype="torch.float32",
                mujoco="2.3.7", renderer={"MUJOCO_GL": "egl"}, implementation_identity=implementation_identity())
    (path / "summary.json").write_text(json.dumps(summary))
    (path / "environment_metadata.json").write_text(json.dumps(meta))
    (path / "progress.jsonl").write_text("".join(json.dumps(x) + "\n" for x in data))
    return path, manifest


def test_paper_cell_checks_actual_episode_records(tmp_path):
    path, manifest = fixture_cell(tmp_path)
    result = paper.validate_cell(tmp_path, "libero_10", "seed01", manifest)
    assert result["successes"] == 500
    lines = (path / "progress.jsonl").read_text().splitlines()
    lines[-1] = lines[0]
    (path / "progress.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="duplicate or missing"):
        paper.validate_cell(tmp_path, "libero_10", "seed01", manifest)


@pytest.mark.parametrize("field,value", [("model_dtype", "torch.bfloat16"), ("gpu", "RTX 3090"), ("mujoco", "3.1.0")])
def test_paper_cell_rejects_wrong_execution_environment(tmp_path, field, value):
    path, manifest = fixture_cell(tmp_path)
    meta = json.loads((path / "environment_metadata.json").read_text())
    meta[field] = value
    (path / "environment_metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        paper.validate_cell(tmp_path, "libero_10", "seed01", manifest)


def test_table_export_without_running_evaluation(tmp_path, monkeypatch):
    native = dict(seed_mean_latency_per_action_ms=42, seed_mean_success_rate=.95)
    expected = {(suite, seed): {"success_rate": .95} for suite in paper.SUITES for seed in paper.SEEDS}
    monkeypatch.setattr(paper, "validate_manifests", lambda args: (expected, expected, native))
    cells = {(row["suite"], row["seed"]): row for row in rows()}
    monkeypatch.setattr(paper, "validate_cell", lambda root, suite, seed, manifest: cells[suite, seed])
    paper.main(argparse.Namespace(command="summary", output=str(tmp_path)))
    tex = (tmp_path / "summary/main_table_rows.tex").read_text()
    lines = [line for line in tex.splitlines() if line.startswith("SimVLA")]
    assert len(lines) == 2 and all(line.endswith("\\\\") for line in lines)
    assert all(line.count("&") == 7 for line in lines)
    assert "$92.0{\\scriptstyle\\pm2.0}$" in tex
    assert "-- & 10" in tex
    result = json.loads((tmp_path / "summary/paper_summary.json").read_text())
    assert result["baseline_rerun"] is False and result["episodes"] == 6000
