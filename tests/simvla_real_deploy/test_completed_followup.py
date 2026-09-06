import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner(tmp_path):
    path = Path(__file__).resolve().parents[2] / "tools/simvla/run_completed_followup.py"
    spec = importlib.util.spec_from_file_location("completed_followup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RESULTS = tmp_path
    module.OUTPUT = tmp_path / "next"
    for suite in module.SUITES:
        for seed in module.SEEDS:
            ref = tmp_path / ("paper_nonlong_seed01_primary_v1" if seed == "seed01"
                             else "paper_completion/three_seed_5090_egl_v1/nonlong")
            manifest = dict(suite=suite, inference_seed_replica=seed, max_policy_actions=800,
                            action_horizon=10, execution_horizon=5, flow_steps=10,
                            manifest_sha256=f"{suite}_{seed}",
                            episodes=[dict(task_id=t, trial_id=i) for t in range(10) for i in range(50)])
            module.write(ref / "manifests" / suite / seed / "episode_manifest.json", manifest)
            module.write(ref / "gates" / suite / seed / "fixed_2x2_parity.json",
                         dict(verdict="FIXED_2X2_PARITY_PASS", manifest_sha256=manifest["manifest_sha256"]))
            csv_path = ref / f"{suite}_{seed}.csv"
            csv_path.write_text("task_id,trial_id,success\n" + "".join(f"{t},{i},True\n" for t in range(10) for i in range(50)))
            registry = ref / "summary/selected_matrix_summary.json"
            data = module.read(registry) if registry.exists() else {"cell_reports": {}}
            data["cell_reports"][f"{suite}:{seed}"] = dict(suite=suite, inference_seed=seed,
                row="full_nfe10", manifest_sha256=manifest["manifest_sha256"], metrics_path=str(csv_path))
            module.write(registry, data)
    return module


def test_only_nine_missing_coupled_cells(runner):
    plan = runner.build_plan()
    assert len(plan) == 9
    assert {c["suite"] for c in plan} == set(runner.SUITES)
    assert all("libero_10" not in c["output"] for c in plan)
    for cell in plan:
        cmd = runner.command(cell)
        assert cmd[cmd.index("--row") + 1] == "condition_kc2_ng3_coupled"
        assert cmd[cmd.index("--episodes-per-task-limit") + 1] == "50"
        assert cmd[cmd.index("--physical-gpu-id") + 1] == "0"
        assert "--coupled-generation-checkpoint" in cmd


def test_duplicate_episode_cannot_pass(runner):
    cell = runner.build_plan()[0]
    path = Path(cell["baseline"])
    lines = path.read_text().splitlines()
    lines[-1] = lines[-2]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="Baseline episode IDs"):
        runner.build_plan()


def test_wrong_horizon_cannot_pass(runner):
    path = Path(runner.build_plan()[0]["manifest"])
    data = runner.read(path)
    data["execution_horizon"] = 10
    runner.write(path, data)
    with pytest.raises(ValueError, match="Reference contract"):
        runner.build_plan()


def test_wrong_parity_cannot_pass(runner):
    path = Path(runner.build_plan()[0]["parity"])
    data = runner.read(path)
    data["manifest_sha256"] = "wrong"
    runner.write(path, data)
    with pytest.raises(ValueError, match="Reference parity"):
        runner.build_plan()
