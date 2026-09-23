"""CPU-only source/artifact audit for SimVLA action-horizon provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CACHE_ROOT = Path(
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/"
    "simvla/latentloop/20260804_chunkaware_v3/cache"
)
DEFAULT_RESULT_ROOT = ROOT / "results/simvla/latentloop/20260804_chunkaware_v3"

from methods.hierarchical_correction.horizon_provenance import (  # noqa: E402
    derive_provenance_schedule,
    original_tokens_remaining,
    r1_hybrid_readiness_verdict,
    simulate_token_provenance,
)
from methods.latentloop.modules.action_chunk_correction import shift_action_chunk  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _line(path: Path, text: str) -> int:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if text in line:
            return number
    raise RuntimeError(f"source evidence not found in {path}: {text}")


def _source_ref(path: Path, text: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "line": _line(path, text),
        "needle": text,
        "sha256": _sha256(path),
    }


def _protocol_rows(
    name: str,
    *,
    action_horizon: int,
    execution_horizon: int,
    levels: tuple[int, ...],
) -> list[dict[str, Any]]:
    records = simulate_token_provenance(
        action_horizon=action_horizon,
        execution_horizon=execution_horizon,
        levels=levels,
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        for prefix_slot, token in enumerate(record["executed_tokens"]):
            rows.append(
                {
                    "protocol": name,
                    "action_horizon_H": action_horizon,
                    "execution_horizon_R": execution_horizon,
                    "query_index": record["query_index"],
                    "level": record["level"],
                    "level_source": record["level_source"],
                    "executed_prefix_slot": prefix_slot,
                    "chunk_slot": token["slot"],
                    "generator_query": token["generator_query"],
                    "generator_token": token["generator_token"],
                    "generator_backed": token["generator_backed"],
                    "original_q0_generator_lineage": token["generator_query"] == 0,
                    "correction_depth": token["correction_depth"],
                    "synthesized_at_query": token["synthesized_at_query"],
                    "token_source": token["source"],
                    "generator_backed_before_query": record[
                        "generator_backed_before_query"
                    ],
                    "generator_backed_after_query": record[
                        "generator_backed_after_query"
                    ],
                    "level1_provenance_reset": record["level1_provenance_reset"],
                }
            )
    return rows


def _project_r1_hybrid_latency(protocol_summary: dict[str, Any]) -> dict[str, float | str]:
    rows = protocol_summary["rows"]
    full = float(rows["full_k1"]["latency_ms"]["policy_query_total_ms"]["mean"])
    condition_average = float(
        rows["chunk_aware_latentloop_k4"]["latency_ms"]["policy_query_total_ms"]["mean"]
    )
    action_average = float(
        rows["action_chunk_correction_k4"]["latency_ms"]["policy_query_total_ms"]["mean"]
    )
    condition_light = (4.0 * condition_average - full) / 3.0
    action_light = (4.0 * action_average - full) / 3.0
    condition_components = rows["chunk_aware_latentloop_k4"]["latency_ms"]
    condition_update_only = sum(
        float(condition_components[key]["mean"])
        for key in (
            "FastEncoder_ms",
            "executed_action_encoder_ms",
            "condition_updater_ms",
        )
    )
    combined_level0 = action_light + condition_update_only
    projected_query_ms = (full + combined_level0 + condition_light + combined_level0) / 4.0
    return {
        "projection_type": "additive component estimate; not a measured hybrid runtime",
        "full_query_ms": full,
        "condition_light_query_ms_inferred": condition_light,
        "action_light_query_ms_inferred": action_light,
        "condition_sidecar_ms_component_sum": condition_update_only,
        "combined_level0_ms_estimate": combined_level0,
        "hybrid_policy_query_ms_estimate": projected_query_ms,
        "pure_action_policy_query_ms_measured": action_average,
        "pure_condition_policy_query_ms_measured": condition_average,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.checkpoint_config).expanduser().resolve()
    r1_manifest_path = Path(args.r1_cache).expanduser().resolve() / "manifest.json"
    r5_manifest_path = Path(args.r5_cache).expanduser().resolve() / "manifest.json"
    protocol_summary_path = Path(args.protocol_a_summary).expanduser().resolve()
    condition_run_path = Path(args.r1_condition_run_summary).expanduser().resolve()
    action_run_path = Path(args.r1_action_run_summary).expanduser().resolve()

    config = _load_json(config_path)
    r1_manifest = _load_json(r1_manifest_path)
    r5_manifest = _load_json(r5_manifest_path)
    protocol_summary = _load_json(protocol_summary_path)
    condition_run = _load_json(condition_run_path)
    action_run = _load_json(action_run_path)

    action_horizon = int(config["num_actions"])
    r1_h = int(r1_manifest["metadata"]["protocol"]["action_horizon_H_a"])
    r5_h = int(r5_manifest["metadata"]["protocol"]["action_horizon_H_a"])
    r1_execution = int(r1_manifest["execution_horizon"])
    r5_execution = int(r5_manifest["execution_horizon"])
    if action_horizon != r1_h or action_horizon != r5_h:
        raise RuntimeError("checkpoint and cache action horizons disagree")
    if (r1_execution, r5_execution) != (1, 5):
        raise RuntimeError("reviewed caches do not provide the expected R=1 and R=5 axes")

    client = ROOT / "architectures/simvla/upstream/evaluation/libero/libero_client.py"
    latent_policy = ROOT / "architectures/simvla/adapters/latentloop/simvla_policy.py"
    correction = ROOT / "methods/latentloop/modules/action_chunk_correction.py"
    hybrid = ROOT / "architectures/simvla/adapters/hierarchical_correction/simvla_hybrid_policy.py"
    source_evidence = {
        "native_default_replan": _source_ref(client, "replan_steps: int = 5"),
        "official_queue_refill": _source_ref(client, "action_chunk[:self.replan_steps]"),
        "official_queue_pop": _source_ref(client, "return self.action_plan.popleft()"),
        "adapter_queue_refill": _source_ref(
            latent_policy, "action_chunk[0, : self.execution_horizon]"
        ),
        "adapter_queue_pop": _source_ref(latent_policy, "self.action_queue.popleft()"),
        "exact_left_shift": _source_ref(correction, "previous_chunk[index, start:]"),
        "zero_tail_initialization": _source_ref(
            correction, "new_zeros(previous_chunk.shape)"
        ),
        "learned_all_token_residual": _source_ref(
            correction, "shifted.actions[..., :6] + arm_residual"
        ),
        "corrector_previous_chunk_input": _source_ref(
            hybrid, 'inputs["previous_action_chunk"]'
        ),
        "corrected_chunk_cache_replace": _source_ref(
            hybrid, "self.hybrid_cache.commit_lightweight"
        ),
    }

    synthetic = torch.arange(action_horizon * 7, dtype=torch.float32).reshape(
        1, action_horizon, 7
    )
    shifted = shift_action_chunk(synthetic, r5_execution)
    shift_check = bool(
        torch.equal(shifted.actions[:, :5], synthetic[:, 5:])
        and torch.equal(shifted.actions[:, 5:], torch.zeros_like(synthetic[:, 5:]))
        and shifted.validity_mask.tolist() == [[True] * 5 + [False] * 5]
    )
    if not shift_check:
        raise RuntimeError("functional left-shift check failed")

    r1_schedule = derive_provenance_schedule(
        action_horizon=action_horizon,
        execution_horizon=r1_execution,
        full_refresh_interval=4,
    )
    r5_schedule = derive_provenance_schedule(
        action_horizon=action_horizon,
        execution_horizon=r5_execution,
        full_refresh_interval=4,
    )
    r1_hybrid_levels = (2, 0, 1, 0, 2)
    r1_pure_action_levels = (2, 0, 0, 0, 2)
    r5_native_levels = tuple(r5_schedule.levels_through_next_full_refresh)
    if r5_native_levels != (2, 0, 1, 0, 2):
        raise RuntimeError(f"unexpected native schedule: {r5_native_levels}")

    protocol_rows: list[dict[str, Any]] = []
    for name, execution, levels in (
        ("protocol_a_pure_action_k4", r1_execution, r1_pure_action_levels),
        ("protocol_a_completed_hybrid_kf4_kg2", r1_execution, r1_hybrid_levels),
        ("native_pure_action_k4", r5_execution, r1_pure_action_levels),
        ("native_horizon_hybrid_kf4_kg2", r5_execution, r5_native_levels),
    ):
        protocol_rows.extend(
            _protocol_rows(
                name,
                action_horizon=action_horizon,
                execution_horizon=execution,
                levels=levels,
            )
        )

    r1_results = {
        name: {
            "success_rate": float(protocol_summary["rows"][name]["success_rate"]),
            "episodes": int(protocol_summary["rows"][name]["episodes"]),
            "amortized_policy_ms_per_environment_action": float(
                protocol_summary["rows"][name][
                    "amortized_policy_ms_per_environment_action"
                ]
            ),
        }
        for name in (
            "full_k1",
            "chunk_aware_latentloop_k4",
            "nonrecurrent_condition_k4",
            "action_chunk_correction_k4",
        )
    }
    r1_verdict = r1_hybrid_readiness_verdict(
        implementation_valid=True,
        lineage_exhausted_before_regeneration=False,
        measured_regeneration_need=False,
        pure_action_dominates=(
            r1_results["action_chunk_correction_k4"]["success_rate"]
            > r1_results["chunk_aware_latentloop_k4"]["success_rate"]
        ),
        regeneration_adds_action_transformer_compute=True,
    )
    parameter_audit = {
        "condition_adapter_parameters": int(condition_run["adapter_trainable_parameters"]),
        "action_correction_adapter_parameters": int(action_run["adapter_trainable_parameters"]),
        "combined_hybrid_parameters": int(condition_run["adapter_trainable_parameters"])
        + int(action_run["adapter_trainable_parameters"]),
        "shared_parameters": 0,
    }

    rules = {
        "schema_version": "simvla_action_horizon_provenance_v1",
        "verified": True,
        "checkpoint_config": {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "action_horizon_H": action_horizon,
        },
        "protocols": {
            "protocol_a": {
                "execution_horizon_R": r1_execution,
                "cache_manifest": str(r1_manifest_path),
                "record_count": sum(int(item["records"]) for item in r1_manifest["shards"]),
            },
            "native": {
                "execution_horizon_R": r5_execution,
                "cache_manifest": str(r5_manifest_path),
                "record_count": sum(int(item["records"]) for item in r5_manifest["shards"]),
            },
        },
        "queue_rule": "refill with the first R actions, then popleft one action per environment step",
        "shift_rule": "left shift cached chunk by R; initialize the final R slots to zero",
        "tail_rule": "the learned residual corrector writes every slot, including zero-initialized tail slots",
        "recursive_input_rule": "each correction-only query consumes the immediately previous corrected chunk",
        "original_token_formula": "N_remain(m)=max(H-mR,0)",
        "formula_values": {
            "protocol_a_R1": [original_tokens_remaining(action_horizon, 1, m) for m in range(5)],
            "native_R5": [original_tokens_remaining(action_horizon, 5, m) for m in range(4)],
        },
        "source_evidence": source_evidence,
        "functional_shift_check": shift_check,
        "derived_schedules": {
            "protocol_a": r1_schedule.to_dict(),
            "native": r5_schedule.to_dict(),
        },
        "r1_hybrid": {
            "levels": list(r1_hybrid_levels),
            "verdict": r1_verdict,
            "results": r1_results,
            "parameter_audit": parameter_audit,
            "latency_projection": _project_r1_hybrid_latency(protocol_summary),
        },
    }
    (output / "simvla_provenance_rules.json").write_text(
        json.dumps(rules, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "protocol_a_hybrid_readiness.json").write_text(
        json.dumps(rules["r1_hybrid"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (output / "simvla_executed_token_provenance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(protocol_rows[0]))
        writer.writeheader()
        writer.writerows(protocol_rows)

    r1_calls = {
        "completed_hybrid_first_4_queries": {
            "levels": [2, 0, 1, 0],
            "full_vlm_calls": 1,
            "condition_updates": 3,
            "action_transformer_decodes": 2,
            "action_corrections": 2,
        },
        "pure_action_first_4_queries": {
            "levels": [2, 0, 0, 0],
            "full_vlm_calls": 1,
            "condition_updates": 0,
            "action_transformer_decodes": 1,
            "action_corrections": 3,
        },
    }
    markdown = f"""# SimVLA action-horizon provenance audit

## Verified implementation facts

- Official checkpoint action horizon: `H={action_horizon}` from `{config_path}`.
- Protocol A cache: `R={r1_execution}`, {rules['protocols']['protocol_a']['record_count']:,} records.
- Native cache and official client default: `R={r5_execution}`, {rules['protocols']['native']['record_count']:,} records.
- Queue semantics: refill from the first `R` actions and execute them in FIFO order.
- Correction semantics: shift left by `R`, zero-initialize the tail, predict residuals for every slot, and cache the corrected output as the next correction input.
- Functional synthetic shift check: `PASS`.

The implementation therefore exactly supports:

```text
N_remain(m) = max(H - mR, 0)
R=1: 10, 9, 8, 7, 6 original q0 tokens remain for m=0..4
R=5: 10, 5, 0, 0 original q0 tokens remain for m=0..3
```

`simvla_executed_token_provenance.csv` records every executed token for the pure-action and hybrid schedules. A learned tail token has no original generator token even though the corrector may assign it a useful value.

## Protocol A, R=1

The completed hybrid schedule is `[2,0,1,0,2]`. At query 2, before Level-1 regeneration, nine q0-generator-backed tokens still remain in the cached chunk; pure action correction would execute an original-lineage token at every lightweight query before query 4. The Level-1 decode therefore does not repair provenance exhaustion.

Fixed 100-episode diagnostic results:

| Row | Success | Policy ms/action |
|---|---:|---:|
| Full K1 | {100*r1_results['full_k1']['success_rate']:.1f}% | {r1_results['full_k1']['amortized_policy_ms_per_environment_action']:.3f} |
| Recurrent condition K4 | {100*r1_results['chunk_aware_latentloop_k4']['success_rate']:.1f}% | {r1_results['chunk_aware_latentloop_k4']['amortized_policy_ms_per_environment_action']:.3f} |
| Nonrecurrent condition K4 | {100*r1_results['nonrecurrent_condition_k4']['success_rate']:.1f}% | {r1_results['nonrecurrent_condition_k4']['amortized_policy_ms_per_environment_action']:.3f} |
| Action correction K4 | {100*r1_results['action_chunk_correction_k4']['success_rate']:.1f}% | {r1_results['action_chunk_correction_k4']['amortized_policy_ms_per_environment_action']:.3f} |

The completed hybrid has {parameter_audit['combined_hybrid_parameters']:,} adapter parameters with no sharing. Over four queries it performs two action-transformer decodes versus one for pure action correction and also advances the separate recurrent condition adapter. An additive component projection gives approximately {rules['r1_hybrid']['latency_projection']['hybrid_policy_query_ms_estimate']:.1f} ms/query; this is not an empirical runtime and must not be reported as a result.

Verdict: **`{r1_verdict}`**. Keep the implementation and its endpoint/schedule tests, but leave it default-off and do not create a paper result from it.

## Native R=5

The first q0-lineage exhaustion age is `ceil(10/5)=2`. With `K_F=4`, the provenance-derived schedule is exactly:

```text
query 0: Level 2, full condition plus full action chunk
query 1: Level 0, shift/correct; execute original q0 tokens 5..9
query 2: Level 1, regenerate a complete action chunk
query 3: Level 0, shift/correct the q2-generated chunk
query 4: Level 2, full refresh
```

Without Level 1, the query-2 executed prefix is derived only from synthetic tail positions produced at query 1. With Level 1, all ten chunk slots receive fresh query-2 generator lineage.

## Source locations

See `simvla_provenance_rules.json` for source hashes, exact line numbers, cache manifests, and the machine-readable derivation.
"""
    (output / "simvla_action_horizon_provenance.md").write_text(markdown, encoding="utf-8")
    return rules


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint-config",
        default=str(
            ROOT
            / ".cache/huggingface/hub/models--YuankaiLuo--SimVLA-LIBERO/snapshots/"
            "93dc4d90b0596c652ad2840ad743c62b9c4473fb/config.json"
        ),
    )
    parser.add_argument("--r1-cache", default=str(DEFAULT_CACHE_ROOT / "query_v3_r1_full_10x20"))
    parser.add_argument("--r5-cache", default=str(DEFAULT_CACHE_ROOT / "query_v3_r5_full_10x20"))
    parser.add_argument(
        "--protocol-a-summary",
        default=str(
            DEFAULT_RESULT_ROOT
            / "online/protocol_a_r1_screening_10x10_parallel/merged/online_summary.json"
        ),
    )
    parser.add_argument(
        "--r1-condition-run-summary",
        default=str(DEFAULT_RESULT_ROOT / "train/t1_r1/chunk_aware_latentloop/run_summary.json"),
    )
    parser.add_argument(
        "--r1-action-run-summary",
        default=str(DEFAULT_RESULT_ROOT / "train/t1_r1/action_chunk_correction/run_summary.json"),
    )
    result = run(parser.parse_args())
    print(json.dumps({"verified": result["verified"], "r1": result["r1_hybrid"]["verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
