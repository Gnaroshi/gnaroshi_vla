"""Fail-closed interpretation of compile experiments, independent of CUDA."""

from pathlib import Path


def required_components(row):
    names = {"action_transformer", "vlm"}
    if row in {"condition_naive3", "condition_nfe10", "ours_kc2_ng3"}:
        names |= {"observation_encoder", "condition_updater"}
    if row in {"ours_kc2_ng3", "generation_ng3"}:
        names.add("generation_updater")
    if row == "latent_bridge_f2":
        names.add("bridge_predict_next")
    return names


def assess_result(result, *, log_text=""):
    failures = []
    if result.get("mode") not in {"eager", "compile"} or not result.get("row"):
        failures.append("row_or_mode_missing")
    if result.get("verdict") != "RECORDED_INPUT_BENCHMARK_COMPLETE":
        failures.append("execution_not_complete")
    if result.get("mode") == "compile":
        records = result.get("compiler", {})
        missing = sorted(name for name in required_components(result["row"])
                         if records.get(name, {}).get("graphs", 0) == 0)
        if missing:
            failures.append("compile_not_observed:" + ",".join(missing))
        if not result.get("graphs_before_measurement") or result.get("graphs_before_measurement") != result.get("graphs_after_measurement"):
            failures.append("recompiled_during_measurement")
        if any(token in log_text for token in (
            "hit config.recompile_limit", "hit config.cache_size_limit", "WON'T CONVERT",
        )):
            failures.append("compiler_fallback_detected")
        comparisons = result.get("output_comparisons", [])
        if not comparisons or not all(x.get("finite", False) for x in comparisons):
            failures.append("finite_output_comparisons_missing")
        if result.get("gripper_sign_changes", 0):
            failures.append("executed_gripper_changed")
    return {
        "execution_complete": result.get("verdict") == "RECORDED_INPUT_BENCHMARK_COMPLETE",
        "timing_checks": "PASS" if not failures else "REVIEW_REQUIRED",
        "issues": failures,
        "output_status": ("BITWISE_EQUAL_ON_TESTED_INPUTS" if result.get("bitwise_equal_on_all_recorded_inputs")
                          else "NUMERICAL_DIFFERENCES_REQUIRE_REVIEW" if result.get("mode") == "compile"
                          else "EAGER_REFERENCE"),
        "success_rate_validated": False,
        "paper_latency_validated": False,
        "scope": "component compilation only; nonzero graphs do not imply full-graph compilation",
    }


def read_log(path):
    path = Path(path)
    return path.read_text(errors="replace") if path.exists() else ""
