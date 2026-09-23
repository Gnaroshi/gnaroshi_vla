"""Synthetic artifact tests for the offline latent-filter aggregator."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "architectures" / "seer" / "upstream"))
sys.path.insert(0, str(ROOT / "tools" / "seer"))

from analyze_every_step_latent_filter import (  # noqa: E402
    _paired_present_tensors,
    aggregate,
    discover_rows,
    endpoint_parity,
)
from utils.lrnode_mechanism_utils import save_trace_episode  # noqa: E402


def test_paired_present_tensors_uses_shared_timestep_mask() -> None:
    left = np.arange(4 * 3 * 7).reshape(4, 3, 7)
    right = left + 1
    tensors = {
        "left": left,
        "left__present": np.array([0, 1, 1, 0], dtype=np.uint8),
        "right": right,
        "right__present": np.array([1, 1, 0, 1], dtype=np.uint8),
    }
    paired_left, paired_right = _paired_present_tensors(
        tensors,
        "left",
        "right",
    )
    assert np.array_equal(paired_left, left[[1]])
    assert np.array_equal(paired_right, right[[1]])


def _make_row(
    root: Path,
    checkpoint: int,
    label: str,
    mode: str,
    success: int,
    action_value: float,
    latent_value: float,
    scope: str,
    updater_calls: int = 0,
) -> None:
    row_root = root / label
    analysis = row_root / "run" / "analysis"
    trace_dir = analysis / "mechanism_trace"
    analysis.mkdir(parents=True)
    (row_root / "latent_filter_row.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": scope,
                "row_label": label,
                "checkpoint_id": checkpoint,
                "adapter_id": 39,
                "mode": mode,
                "alpha": 0.5,
                "beta": 0.5,
                "diagnostics": int("diagnostics_on" in label),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "success_rate": float(success),
        "num_filter_action_head_calls": 1,
        "query_reduction": {
            "num_env_steps": 2,
            "num_full_forward_calls": 2,
            "num_lrnode_update_calls": updater_calls,
            "full_forward_calls_per_policy_step": 1.0,
            "query_reduction_claim_allowed": False,
            "full_query_reduction_ratio": 0.0,
        },
        "lrnode": {
            "avg_full_forward_latency_sec": 0.04,
            "avg_every_step_filter_prior_latency_sec": 0.002,
            "avg_every_step_filter_fusion_latency_sec": 0.0001,
            "avg_every_step_filter_action_head_latency_sec": 0.001,
            "avg_policy_step_latency_sec": 0.043,
            "avg_every_step_filter_diagnostic_latency_sec": 0.0005,
            "every_step_filter_diagnostic_action_head_calls": 1,
            "every_step_filter_rng_failures": 0,
        },
    }
    (analysis / "eval_summary.json").write_text(
        json.dumps(summary) + "\n",
        encoding="utf-8",
    )
    episode = {
        "task_id": 0,
        "task_name": "task0",
        "episode_id": 0,
        "success": success,
        "num_steps": 2,
    }
    with (analysis / "eval_episode_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode))
        writer.writeheader()
        writer.writerow(episode)

    actions = torch.full((1, 2, 7), action_value)
    final = np.full(7, action_value, dtype=np.float32)
    latent = torch.full((1, 2, 4), latent_value)
    save_trace_episode(
        trace_dir,
        f"{label}_task00_episode000_rank0",
        [
            {"timestep": 0, "filter_diagnostics_rng_preserved": 1},
            {"timestep": 1, "filter_diagnostics_rng_preserved": 1},
        ],
        [
            {
                "a_executed_final": final,
                "a_executed_raw": actions,
                "a_full_raw": actions,
                "a_filter_raw": actions,
                "z_executed": latent,
            },
            {
                "a_executed_final": final,
                "a_executed_raw": actions,
                "a_full_raw": actions,
                "a_filter_raw": actions,
                "z_executed": latent,
            },
        ],
        {
            "task_id": 0,
            "episode_id": 0,
            "success": success,
            "steps": 2,
        },
    )


def test_endpoint_smoke_exactness(tmp_path: Path) -> None:
    definitions = (
        ("canonical_k1", "canonical_k1", 0.25, 0.5, 0),
        ("raw_full", "raw_full", 0.25, 0.5, 0),
        ("raw_full_diagnostics_on", "raw_full", 0.25, 0.5, 0),
        ("fixed_filter_alpha1", "fixed_filter", 0.25, 0.5, 0),
        ("recurrent_prior", "recurrent_prior", 0.75, 1.5, 1),
        ("fixed_filter_alpha0", "fixed_filter", 0.75, 1.5, 1),
    )
    for label, mode, action, latent, updater_calls in definitions:
        _make_row(
            tmp_path,
            36,
            label,
            mode,
            success=1,
            action_value=action,
            latent_value=latent,
            scope="endpoint_smoke",
            updater_calls=updater_calls,
        )
    result = endpoint_parity(discover_rows([tmp_path]))
    assert result["all_pass"]
    assert result["alpha_1_exact"]
    assert result["alpha_0_exact"]
    assert result["diagnostics_invariant"]


def test_aggregator_writes_all_required_outputs(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    for checkpoint in (36, 38):
        checkpoint_root = result_root / f"checkpoint_{checkpoint}"
        for mode in (
            "raw_full",
            "recurrent_prior",
            "fixed_filter",
            "full_latent_ema",
        ):
            _make_row(
                checkpoint_root,
                checkpoint,
                mode,
                mode,
                success=int(mode == "fixed_filter"),
                action_value=0.1,
                latent_value=0.2,
                scope="primary_heldout",
                updater_calls=int(mode in {"recurrent_prior", "fixed_filter"}),
            )
    rows = discover_rows([result_root])
    output = tmp_path / "aggregate"
    rule = {
        "heldout_checkpoints": [36, 38],
        "fixed_vs_raw": {
            "minimum_mean_sr_gain_pp": 2.0,
            "maximum_checkpoint_drop_pp": 1.0,
        },
        "ema_exclusion": {
            "minimum_sr_advantage_pp": 1.0,
            "sr_equivalence_margin_pp": 1.0,
            "continuity_reduction_fraction": 0.2,
        },
    }
    aggregate(
        rows,
        output,
        endpoint={"alpha_1_exact": True},
        decision_rule=rule,
        require_primary_complete=True,
    )
    expected = {
        "latent_filter_main_table.csv",
        "latent_filter_per_task.csv",
        "latent_filter_paired_flips.csv",
        "latent_filter_continuity.csv",
        "latent_filter_latency.csv",
        "latent_filter_decision.json",
        "final_latent_filter_result_report.md",
    }
    assert expected == {path.name for path in output.iterdir()}
    decision = json.loads(
        (output / "latent_filter_decision.json").read_text(encoding="utf-8")
    )
    assert decision["verdict"] == "LATENT_FILTER_CONFIRMED"
