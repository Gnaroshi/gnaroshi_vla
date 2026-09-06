import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner(tmp_path):
    path = Path(__file__).resolve().parents[2] / "tools/simvla/run_latent_bridge_large_nonlong.py"
    spec = importlib.util.spec_from_file_location("lb_nonlong", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RESULTS = tmp_path
    module.OUTPUT = tmp_path / "next"
    module.DOLL = tmp_path / "doll"
    for seed in module.SEEDS:
        ref = tmp_path / ("paper_nonlong_seed01_primary_v1" if seed == "seed01"
                         else "paper_completion/three_seed_5090_egl_v1/nonlong")
        for suite in module.SUITES:
            manifest = dict(suite=suite, inference_seed_replica=seed, max_policy_actions=800,
                manifest_sha256=f"{suite}:{seed}", renderer={"PYTHONHASHSEED": seed},
                episodes=[dict(task_id=t, trial_id=i, init_state_index=i, environment_seed=7)
                          for t in range(10) for i in range(50)])
            module.write(ref / "manifests" / suite / seed / "episode_manifest.json", manifest)
            csv_path = ref / f"{suite}_{seed}.csv"
            csv_path.write_text("task_id,trial_id\n" + "".join(f"{t},{i}\n" for t in range(10) for i in range(50)))
            registry = ref / "summary/selected_matrix_summary.json"
            data = module.read(registry) if registry.exists() else {"cell_reports": {}}
            for row in ("full_nfe10", "condition_kc2_ng3"):
                data["cell_reports"][f"{suite}:{seed}:{row}"] = dict(suite=suite, inference_seed=seed,
                    row=row, manifest_sha256=manifest["manifest_sha256"], metrics_path=str(csv_path))
            module.write(registry, data)
    return module


def episodes(runner, cell, row):
    return [dict(**e, row=row, success=True, episode_length=5, num_policy_queries=1,
        num_action_transformer_calls=10, num_full_vlm_calls=1, num_condition_updater_calls=0,
        latency_per_executed_action_ms=20., vlm_latency_total_ms=10.,
        bridge_latency_total_ms=0., action_latency_total_ms=80.)
        for e in runner.read(cell["manifest"])["episodes"]]


def test_exact_scope_and_options(runner):
    plan = runner.build_plan()
    assert len(plan) == 9
    assert len(plan) * len(runner.ROWS) * 500 == 13500
    assert len({(c["suite"], c["seed"]) for c in plan}) == 9
    for cell in plan:
        command = runner.options(cell)
        assert command[command.index("--rows") + 1:command.index("--bridge-precision")] == list(runner.ROWS)
        assert command[command.index("--max-policy-steps") + 1] == "800"
        assert command[command.index("--num-trials") + 1] == "50"
        assert "baseline_k1" not in command and "libero_10" not in command
        assert "--collect-dagger-teacher" not in command
        assert "--resume-output" in command and "--compile-bridge" in command


def test_reference_duplicate_rejected(runner):
    cell = runner.build_plan()[0]
    path = Path(cell["references"]["full_nfe10"]["metrics_path"])
    lines = path.read_text().splitlines()
    lines[-1] = lines[-2]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        runner.build_plan()


@pytest.mark.parametrize("mutation", ("duplicate", "initial_state", "row", "calls", "nan"))
def test_bad_progress_rejected(runner, mutation):
    cell = runner.build_plan()[0]
    data = episodes(runner, cell, runner.ROWS[0])
    if mutation == "duplicate":
        data[-1] = data[0]
    elif mutation == "initial_state":
        data[0]["init_state_index"] = 99
    elif mutation == "row":
        data[0]["row"] = "baseline_k1"
    elif mutation == "calls":
        data[0]["num_action_transformer_calls"] = 3
    else:
        data[0]["latency_per_executed_action_ms"] = float("nan")
    with pytest.raises(ValueError):
        runner.validate_episodes(data, cell, runner.ROWS[0])


def test_cpu_recovery_does_not_rerun_or_change_episode_records(runner, monkeypatch):
    cell = runner.build_plan()[0]
    directory = Path(cell["output"])
    runner.write(directory / "environment_metadata.json", {})
    monkeypatch.setattr(runner, "metadata_ok", lambda *args: None)
    for row in runner.ROWS:
        target = directory / row / "progress.jsonl"
        target.parent.mkdir(parents=True)
        target.write_text("".join(json.dumps(e) + "\n" for e in episodes(runner, cell, row)))
    before = {row: runner.sha(directory / row / "progress.jsonl") for row in runner.ROWS}
    result = runner.recover(cell)
    assert len(result) == 3
    assert runner.read(directory / "comparison_summary.json")["verdict"] == "SIMVLA_LATENT_BRIDGE_EVAL_COMPLETE"
    assert before == {row: runner.sha(directory / row / "progress.jsonl") for row in runner.ROWS}
    assert all(c["summary"]["successes"] == 500 for c in result)
    assert len(runner.recover(cell)) == 3


def test_partial_records_do_not_pass_completion(runner, monkeypatch):
    cell = runner.build_plan()[0]
    directory = Path(cell["output"])
    runner.write(directory / "environment_metadata.json", {})
    monkeypatch.setattr(runner, "metadata_ok", lambda *args: None)
    target = directory / runner.ROWS[0] / "progress.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(episodes(runner, cell, runner.ROWS[0])[0]) + "\n")
    assert runner.recover(cell) == []
    assert not (directory / "comparison_summary.json").exists()


def test_failed_startup_preserved(runner):
    cell = runner.build_plan()[0]
    directory = Path(cell["output"])
    directory.mkdir(parents=True)
    (directory / "startup_note.txt").write_text("interrupted")
    assert runner.recover(cell) == []
    assert not directory.exists()
    assert (directory.with_name(directory.name + "_startup_interrupted_1") / "startup_note.txt").read_text() == "interrupted"


def test_environment_replaces_inherited_renderer(runner, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/wrong/source")
    monkeypatch.setenv("LIBGL_ALWAYS_SOFTWARE", "1")
    cell = runner.build_plan()[0]
    env = runner.environment(cell)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["PYTHONHASHSEED"] == "seed01"
    assert "LIBGL_ALWAYS_SOFTWARE" not in env
    assert "/wrong/source" not in env["PYTHONPATH"]


def test_doll_requires_successful_training(runner):
    assert runner.doll_ready() is False
    path = runner.DOLL / "logs/pipeline.exit_code"
    path.parent.mkdir(parents=True)
    path.write_text("1\n")
    with pytest.raises(RuntimeError, match="unsuccessfully"):
        runner.doll_ready()
    path.write_text("0\n")
    assert runner.doll_ready() is False
    runner.write(runner.DOLL / "deployment_bundle_v4/bundle_inventory.json", dict(verdict="REAL_SIMVLA_DEPLOYMENT_BUNDLE_PASS"))
    assert runner.doll_ready() is True
