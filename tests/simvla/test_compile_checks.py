import json

import pytest

from tools.simvla.compile_checks import assess_result, required_components
from tools.simvla.compile_benchmark import verify_recorded_inputs, verify_result_artifacts, sha


def result():
    return {"row": "ours_kc2_ng3", "mode": "compile", "verdict": "RECORDED_INPUT_BENCHMARK_COMPLETE",
        "compiler": {name: {"graphs": 1} for name in required_components("ours_kc2_ng3")},
        "graphs_before_measurement": 5, "graphs_after_measurement": 5,
        "output_comparisons": [{"finite": True}], "gripper_sign_changes": 0,
        "bitwise_equal_on_all_recorded_inputs": False}


def test_completion_is_not_output_or_sr_pass():
    assessment = assess_result(result())
    assert assessment["timing_checks"] == "PASS"
    assert assessment["output_status"] == "NUMERICAL_DIFFERENCES_REQUIRE_REVIEW"
    assert not assessment["success_rate_validated"]
    assert not assessment["paper_latency_validated"]


def test_total_graphs_cannot_hide_component_bypass():
    value = result()
    value["compiler"]["generation_updater"]["graphs"] = 0
    assert "generation_updater" in str(assess_result(value)["issues"])


def test_recompile_limit_cannot_be_timing_pass():
    assert assess_result(result(), log_text="torch._dynamo hit config.recompile_limit (8)")["timing_checks"] != "PASS"


@pytest.mark.parametrize("change", ["nonfinite", "empty", "gripper", "recompile", "failed", "mode", "missing_graph_count"])
def test_incomplete_checks_fail_closed(change):
    value = result()
    if change == "nonfinite": value["output_comparisons"] = [{"finite": False}]
    if change == "empty": value["output_comparisons"] = []
    if change == "gripper": value["gripper_sign_changes"] = 1
    if change == "recompile": value["graphs_after_measurement"] = 6
    if change == "failed": value["verdict"] = "WORKER_FAILED"
    if change == "mode": value.pop("mode")
    if change == "missing_graph_count":
        value.pop("graphs_before_measurement")
        value.pop("graphs_after_measurement")
    assert assess_result(value)["timing_checks"] != "PASS"


def test_existing_input_must_have_verified_contract(tmp_path):
    path = tmp_path / "recorded_inputs.pt"
    path.write_bytes(b"original")
    with pytest.raises(FileNotFoundError): verify_recorded_inputs(tmp_path)
    (tmp_path / "input_contract.json").write_text(json.dumps({"sha256": sha(path), "samples": 10, "queries_per_sample": 4}))
    verify_recorded_inputs(tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="bytes differ"): verify_recorded_inputs(tmp_path)


@pytest.mark.parametrize("change", [None, "input", "run", "mode", "actions"])
def test_result_reuse_rejects_wrong_inputs_or_actions(tmp_path, change):
    actions = tmp_path / "actions.pt"
    actions.write_bytes(b"saved actions")
    value = result()
    value.update(input_sha256="same-input", run_identity_sha256="same-run", actions_sha256=sha(actions))
    if change == "input": value["input_sha256"] = "other-input"
    if change == "run": value["run_identity_sha256"] = "other-run"
    if change == "mode": value["mode"] = "eager"
    if change == "actions": actions.write_bytes(b"different actions")
    (tmp_path / "result.json").write_text(json.dumps(value))
    kwargs = dict(row="ours_kc2_ng3", mode="compile", input_sha256="same-input", run_identity_sha256="same-run")
    if change:
        with pytest.raises(RuntimeError): verify_result_artifacts(tmp_path, **kwargs)
    else:
        assert verify_result_artifacts(tmp_path, **kwargs) == value
