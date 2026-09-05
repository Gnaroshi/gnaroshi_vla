"""Contracts for the Seer VLA-Cache campaign analyzer."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "seer" / "analyze_seer_vla_cache.py"
SPEC = importlib.util.spec_from_file_location("analyze_seer_vla_cache", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def _row(mode: str, *, reuse_calls: int, reduction: float) -> dict[str, object]:
    return {
        "mode": mode,
        "summary": f"/{mode}/eval_summary.json",
        "episodes": 500,
        "success_rate_pct": 80.0,
        "num_env_steps": 1000,
        "avg_policy_step_latency_ms": 50.0,
        "avg_full_forward_latency_ms": 45.0,
        "renderer": "hardware_egl",
        "cache_calls": 1000 if mode != "off" else 0,
        "cache_first_queries": 500 if mode != "off" else 0,
        "cache_nonfirst_queries": 500 if mode != "off" else 0,
        "actual_kv_reuse_calls": reuse_calls,
        "actual_kv_reuse_rate_nonfirst": reuse_calls / 500,
        "avg_reusable_candidates": 10.0 if mode == "reuse" else 0.0,
        "avg_removed_final": 5.0 if mode == "reuse" else 0.0,
        "full_token_layers": 10000 if mode != "off" else 0,
        "computed_token_layers": 9000 if mode == "reuse" else 10000 if mode != "off" else 0,
        "token_layer_reduction_pct": reduction,
    }


def test_complete_three_mode_campaign_passes_and_renders_markdown(tmp_path: Path) -> None:
    rows = [
        _row("off", reuse_calls=0, reduction=0.0),
        _row("matched_full", reuse_calls=0, reduction=0.0),
        _row("reuse", reuse_calls=400, reduction=10.0),
    ]
    ANALYZER._validate(rows, expected_episodes=500)
    output = tmp_path / "summary.md"
    ANALYZER._write_markdown(output, rows)
    text = output.read_text(encoding="utf-8")
    assert "matched_full" in text
    assert "Token-layer reduction" in text


def test_reuse_without_actual_kv_reuse_is_rejected() -> None:
    rows = [
        _row("off", reuse_calls=0, reduction=0.0),
        _row("matched_full", reuse_calls=0, reduction=0.0),
        _row("reuse", reuse_calls=0, reduction=0.0),
    ]
    try:
        ANALYZER._validate(rows, expected_episodes=500)
    except RuntimeError as error:
        assert "without any actual K/V reuse" in str(error)
    else:
        raise AssertionError("invalid no-reuse campaign was accepted")
