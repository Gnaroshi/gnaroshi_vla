import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/seer/analyze_latentloop_budget_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("latentloop_budget_analysis", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def result(lane, epochs, latent, action, smooth=0.1, passed=True):
    return {
        "protocol": "latentloop_posttrain_budget_validation_v1",
        "lane": lane,
        "budget_epochs": epochs,
        "teacher": "/teacher.pth",
        "teacher_sha256": "teacher",
        "adapter": f"/{lane}_{epochs}.pth",
        "adapter_sha256": f"adapter-{lane}-{epochs}",
        "vit_checkpoint_sha256": "vit",
        "validation_split_sha256": "split",
        "selection_metric_value": 0.05 * latent + 0.1 * action + 0.001 * smooth,
        "gates": {"pass": passed},
        "metrics": {
            "latent_mse": {"mean": latent},
            "action_l1": {"mean": action},
            "smooth_mse": {"mean": smooth},
            "hold_latent_mse": {"mean": 2.0},
            "hold_action_l1": {"mean": 2.0},
        },
        "validation_result_path": f"/{lane}_{epochs}.json",
    }


def test_selects_smallest_budget_within_independent_40_epoch_control():
    rows = [
        result("compressed", 1, 1.20, 1.20),
        result("compressed", 2, 1.04, 1.04),
        result("compressed", 4, 0.95, 0.95),
        result("standard_40", 40, 1.0, 1.0),
    ]
    selected = MODULE.select(rows, tolerance=0.05)
    assert selected["status"] == "MINIMUM_BUDGET_WITHIN_TOLERANCE"
    assert selected["selected_budget_epochs"] == 2


def test_falls_back_to_best_passing_compressed_budget():
    rows = [
        result("compressed", 1, 1.30, 1.30),
        result("compressed", 2, 1.20, 1.20),
        result("compressed", 4, 1.10, 1.10),
        result("standard_40", 40, 1.0, 1.0),
    ]
    selected = MODULE.select(rows, tolerance=0.05)
    assert selected["status"] == "FALLBACK_BEST_COMPRESSED_NO_TOLERANCE_MATCH"
    assert selected["selected_budget_epochs"] == 4


def test_loader_rejects_mixed_validation_splits(tmp_path):
    first = result("compressed", 1, 1.0, 1.0)
    second = result("standard_40", 40, 1.0, 1.0)
    second["validation_split_sha256"] = "different"
    paths = []
    for index, row in enumerate((first, second)):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(row))
        paths.append(path)
    try:
        MODULE.load_results(paths)
    except RuntimeError as error:
        assert "one teacher, validation split" in str(error)
    else:
        raise AssertionError("mixed validation splits must fail")
