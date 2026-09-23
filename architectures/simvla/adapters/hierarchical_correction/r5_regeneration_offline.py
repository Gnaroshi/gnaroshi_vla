"""Exact-age R5 regeneration gate on an existing SimVLA query cache.

This module never steps LIBERO and never trains. It evaluates the first query
where a recursively corrected H=10 chunk has exhausted its original generator
lineage under native R=5 execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
for path in (ROOT, UPSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter  # noqa: E402
from architectures.simvla.adapters.hierarchical_correction.source_locked_loading import (  # noqa: E402
    load_source_locked_simvla,
)
from architectures.simvla.adapters.latentloop.checkpoint import (  # noqa: E402
    freeze_module,
    load_adapter_checkpoint,
)
from architectures.simvla.adapters.latentloop.source_lock import (  # noqa: E402
    collect_source_lock,
    require_empty_output,
    resolve_huggingface_checkpoint,
    sha256_file,
)
from methods.hierarchical_correction.decisions import (  # noqa: E402
    RegenerationCandidateGateInputs,
    evaluate_r5_regeneration_gate,
)
from methods.hierarchical_correction.horizon_provenance import (  # noqa: E402
    derive_provenance_schedule,
    simulate_token_provenance,
)
from methods.hierarchical_correction.provenance import (  # noqa: E402
    experiment_source_signature,
    hierarchical_source_manifest,
)
from methods.latentloop.eval import distribution_summary  # noqa: E402
from methods.latentloop.training import normalized_condition_mse  # noqa: E402
from methods.latentloop.training.query_cache_dataset import (  # noqa: E402
    iter_query_records,
    load_manifest,
)


SCHEMA_VERSION = "simvla_r5_age2_regeneration_gate_v1"
CANDIDATES = ("recurrent_age2", "nonrecurrent_anchor_age2")
REFERENCE_ROWS = ("full_teacher_reference", "hold_stale_condition", "old_observation_only_age2")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _batch(value: Tensor, device: torch.device) -> Tensor:
    return value.unsqueeze(0).to(device=device, non_blocking=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _heldout_episode(task_id: int, episode_id: str, *, fraction: float, seed: int) -> bool:
    threshold = int(float(fraction) * 10_000)
    digest = hashlib.sha256(f"{task_id}|{episode_id}|{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10_000 < threshold


def _boundary_errors(previous: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if int(previous["next_query_index"]) != int(current["query_index"]):
        errors.append("query_index")
    for left, right in (
        ("next_raw_rgb", "raw_rgb"),
        ("next_proprio", "proprio"),
        ("next_full_condition", "full_condition"),
        ("next_teacher_action_chunk", "teacher_action_chunk"),
        ("next_initial_noise", "initial_noise"),
    ):
        if not torch.equal(previous[left], current[right]):
            errors.append(f"{left}->{right}")
    return errors


def iter_age2_pairs(
    cache: str | Path,
    *,
    full_refresh_interval: int,
    heldout_fraction: float,
    split_seed: int,
    max_pairs: int = 0,
) -> Iterator[tuple[dict[str, Any], dict[str, Any], list[str]]]:
    """Yield q0->q1 and q1->q2 records for held-out full-refresh anchors."""

    previous_by_episode: dict[tuple[int, str], dict[str, Any]] = {}
    emitted = 0
    for current in iter_query_records(cache):
        key = (int(current["task_id"]), str(current["episode_id"]))
        previous = previous_by_episode.get(key)
        if (
            previous is not None
            and int(previous["query_index"]) % int(full_refresh_interval) == 0
            and int(current["query_index"]) == int(previous["next_query_index"])
            and _heldout_episode(
                key[0],
                key[1],
                fraction=heldout_fraction,
                seed=split_seed,
            )
        ):
            yield previous, current, _boundary_errors(previous, current)
            emitted += 1
            if max_pairs > 0 and emitted >= max_pairs:
                return
        previous_by_episode[key] = current


def _features(
    adapter: Any,
    *,
    previous_rgb: Tensor,
    current_rgb: Tensor,
    previous_proprio: Tensor,
    current_proprio: Tensor,
    executed: Tensor,
    execution_horizon: int,
    elapsed_time: float,
) -> tuple[Tensor, Tensor]:
    observation = adapter.encode_observation(
        previous_rgb,
        current_rgb,
        previous_proprio,
        current_proprio,
    )
    action = adapter.encode_executed_actions(
        executed,
        execution_horizon,
        elapsed_time,
        reference_feature=observation,
    )
    return observation, action


def _recurrent_step(
    adapter: Any,
    previous_condition: Tensor,
    *,
    previous_rgb: Tensor,
    current_rgb: Tensor,
    previous_proprio: Tensor,
    current_proprio: Tensor,
    executed: Tensor,
    execution_horizon: int,
    elapsed_time: float,
    query_age: int,
) -> Tensor:
    observation, action = _features(
        adapter,
        previous_rgb=previous_rgb,
        current_rgb=current_rgb,
        previous_proprio=previous_proprio,
        current_proprio=current_proprio,
        executed=executed,
        execution_horizon=execution_horizon,
        elapsed_time=elapsed_time,
    )
    return adapter.update_recurrent_condition(
        previous_condition,
        observation,
        action,
        execution_horizon=execution_horizon,
        elapsed_time=elapsed_time,
        query_age=query_age,
    )


def _nonrecurrent_age2(
    adapter: Any,
    anchor_condition: Tensor,
    *,
    anchor_rgb: Tensor,
    current_rgb: Tensor,
    anchor_proprio: Tensor,
    current_proprio: Tensor,
    executed_history: tuple[Tensor, Tensor],
    execution_horizon: int,
    elapsed_time: float,
) -> Tensor:
    observation = adapter.encode_observation(
        anchor_rgb,
        current_rgb,
        anchor_proprio,
        current_proprio,
    )
    action_features = [
        adapter.encode_executed_actions(
            executed,
            execution_horizon,
            elapsed_time,
            reference_feature=observation,
        )
        for executed in executed_history
    ]
    action_history = torch.stack(action_features, dim=0).mean(dim=0)
    return adapter.predict_nonrecurrent_condition(
        anchor_condition,
        observation,
        action_history,
        execution_horizon=execution_horizon,
        elapsed_time=elapsed_time,
        query_age=2,
    )


def _condition_metrics(prediction: Tensor, teacher: Tensor) -> dict[str, float]:
    return {
        "condition_cosine": float(
            F.cosine_similarity(prediction.flatten(1), teacher.flatten(1), dim=1)
            .mean()
            .item()
        ),
        "condition_normalized_mse": float(normalized_condition_mse(prediction, teacher).item()),
        "condition_mse": float(F.mse_loss(prediction, teacher).item()),
    }


def _action_metrics(prediction: Tensor, teacher: Tensor, execution_horizon: int) -> dict[str, float]:
    difference = prediction - teacher
    prefix_difference = difference[:, :execution_horizon]
    predicted_prefix = prediction[:, :execution_horizon]
    teacher_prefix = teacher[:, :execution_horizon]
    predicted_sign = predicted_prefix[..., 6] >= 0
    teacher_sign = teacher_prefix[..., 6] >= 0
    if execution_horizon > 1:
        predicted_switch = predicted_sign[:, 1:] != predicted_sign[:, :-1]
        teacher_switch = teacher_sign[:, 1:] != teacher_sign[:, :-1]
        switch_agreement = (predicted_switch == teacher_switch).float().mean()
    else:
        switch_agreement = (predicted_sign == teacher_sign).float().mean()
    return {
        "chunk_l1": float(difference.abs().mean().item()),
        "chunk_l2": float(torch.linalg.vector_norm(difference.flatten(1), dim=1).mean().item()),
        "prefix_l1": float(prefix_difference.abs().mean().item()),
        "prefix_l2": float(
            torch.linalg.vector_norm(prefix_difference.flatten(1), dim=1).mean().item()
        ),
        "translation_prefix_l1": float(prefix_difference[..., :3].abs().mean().item()),
        "translation_prefix_l2": float(
            torch.linalg.vector_norm(prefix_difference[..., :3].flatten(1), dim=1)
            .mean()
            .item()
        ),
        "rotation_prefix_l1": float(prefix_difference[..., 3:6].abs().mean().item()),
        "rotation_prefix_l2": float(
            torch.linalg.vector_norm(prefix_difference[..., 3:6].flatten(1), dim=1)
            .mean()
            .item()
        ),
        "gripper_command_l1": float(prefix_difference[..., 6].abs().mean().item()),
        "gripper_command_l2": float(
            torch.linalg.vector_norm(prefix_difference[..., 6], dim=1).mean().item()
        ),
        "gripper_sign_agreement": float((predicted_sign == teacher_sign).float().mean().item()),
        "gripper_switch_agreement": float(switch_agreement.item()),
        "finite": float(torch.isfinite(prediction).all().item()),
    }


def _paired_ci(differences: Iterable[float], *, seed: int, samples: int) -> tuple[float, float]:
    values = torch.tensor(list(differences), dtype=torch.float64)
    if values.numel() == 0:
        return (float("nan"), float("nan"))
    generator = torch.Generator().manual_seed(int(seed))
    means: list[Tensor] = []
    remaining = int(samples)
    while remaining:
        count = min(remaining, 512)
        indices = torch.randint(
            0,
            values.numel(),
            (count, values.numel()),
            generator=generator,
        )
        means.append(values[indices].mean(dim=1))
        remaining -= count
    bootstrapped = torch.cat(means)
    return (
        float(torch.quantile(bootstrapped, 0.025).item()),
        float(torch.quantile(bootstrapped, 0.975).item()),
    )


def _validate_checkpoint(
    path: Path,
    *,
    expected_variant: str,
    execution_horizon: int,
    action_horizon: int,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    adapter, payload = load_adapter_checkpoint(path, device=device)
    if adapter.variant != expected_variant:
        raise ValueError(f"{path} contains {adapter.variant}, expected {expected_variant}")
    if int(payload.get("step", -1)) != 150000:
        raise ValueError(f"{path} is not the selected step-150000 checkpoint")
    if int(adapter.config.maximum_execution_horizon) != int(execution_horizon):
        raise ValueError(f"{path} has the wrong maximum execution horizon")
    if int(adapter.config.action_horizon) != int(action_horizon):
        raise ValueError(f"{path} has the wrong action horizon")
    freeze_module(adapter)
    return adapter, payload


def _summary_row(
    name: str,
    role: str,
    values: Mapping[str, list[float]],
    *,
    trainable_parameters: int,
    gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row": name,
        "role": role,
        "trainable_parameters": trainable_parameters,
        "gate_pass": gate.get("pass") if gate is not None else "",
    }
    for metric, samples in sorted(values.items()):
        summary = distribution_summary(samples)
        for statistic, value in summary.items():
            row[f"{metric}_{statistic}"] = value
    return row


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    preferred = ["row", "role", "trainable_parameters", "gate_pass"]
    fields = preferred + [field for field in fields if field not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: Path,
    *,
    gate: Mapping[str, Any],
    pairs: int,
    episodes: int,
    cache: str,
    old_gate_path: str,
) -> None:
    lines = [
        "# R5 provenance-exhaustion regeneration gate",
        "",
        f"Schema: `{SCHEMA_VERSION}`.",
        "",
        "## Scope",
        "",
        f"- Existing cache: `{cache}`.",
        f"- Held-out exact q0->q1->q2 pairs: {pairs} across {episodes} episodes.",
        "- No LIBERO environment was stepped and no model was trained.",
        "- Candidate A is recurrent q0->q1->q2 condition propagation.",
        "- Candidate B is direct nonrecurrent anchor-q0 to current-q2 prediction with both executed R=5 subchunks.",
        "- Candidate C is the logging-only full-condition/same-noise teacher.",
        "- Hold and the prior R5 old-observation-only model are paired references.",
        "",
        "## Representation note",
        "",
        "SimVLA LIBERO emits a continuous postprocessed gripper command. It does not expose a gripper logit or probability at this boundary. Therefore this gate reports command L1/L2, sign agreement, and switch agreement; logit/probability errors are explicitly not applicable.",
        "",
        "## Frozen gate",
        "",
        f"- `R5_REGENERATION_GATE_PASS`: `{gate['R5_REGENERATION_GATE_PASS']}`",
        f"- `ONLINE_R5_GATE_PASS`: `{gate['ONLINE_R5_GATE_PASS']}`",
        f"- selected candidate: `{gate.get('selected_candidate')}`",
        "- candidate must beat hold by paired 95% CI, be no worse than old-observation-only by paired 95% CI, retain noncollapsed gripper output, have prefix-L1 p99 no worse than old-observation-only, remain finite, and reset complete action-token provenance.",
        "",
        "## Candidate checks",
        "",
    ]
    for name, result in gate["candidates"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"Pass: `{result['pass']}`")
        lines.append("")
        for check, passed in result["checks"].items():
            lines.append(f"- `{check}`: `{passed}`")
        lines.append("")
    lines.extend(
        [
            "## Prior gate",
            "",
            f"The prior adjacent age-1 gate remains at `{old_gate_path}`. It is contextual evidence only and is not substituted for this exact age-2 gate.",
            "",
            "## Next action",
            "",
            (
                "The guarded native R5 online matrix may be reviewed. It still requires explicit opt-in and a matching gate/source signature."
                if gate["ONLINE_R5_GATE_PASS"]
                else "Do not run R5 online K>1. Use the gap-matched regeneration training specification before another gate attempt."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the source-locked, teacher-forced exact-age gate."""

    from models.modeling_smolvlm_vla import SmolVLMVLA

    output = require_empty_output(args.output)
    device = torch.device(args.device)
    manifest = load_manifest(args.cache)
    protocol = manifest.get("metadata", {}).get("protocol", {})
    action_horizon = int(protocol.get("action_horizon_H_a", 0))
    execution_horizon = int(manifest["execution_horizon"])
    if (action_horizon, execution_horizon) != (10, 5):
        raise ValueError(
            f"native gate requires cache-confirmed H=10,R=5; got H={action_horizon},R={execution_horizon}"
        )
    schedule = derive_provenance_schedule(
        action_horizon=action_horizon,
        execution_horizon=execution_horizon,
        full_refresh_interval=args.full_refresh_interval,
    )
    if schedule.regeneration_interval != 2 or schedule.levels_through_next_full_refresh != (
        2,
        0,
        1,
        0,
        2,
    ):
        raise RuntimeError("local H/R do not derive the reviewed 2,0,1,0,2 native schedule")

    checkpoints = {
        "condition": Path(args.recurrent_checkpoint).expanduser().resolve(),
        "nonrecurrent": Path(args.nonrecurrent_checkpoint).expanduser().resolve(),
        "old_observation_only": Path(args.old_observation_checkpoint).expanduser().resolve(),
        "action_correction": Path(args.action_correction_checkpoint).expanduser().resolve(),
    }
    source_lock = collect_source_lock(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    source_lock["processor_checkpoint"] = resolve_huggingface_checkpoint(args.smolvlm_model_path)
    source_lock["hierarchical_checkpoints"] = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in checkpoints.items()
    }
    source_lock["hierarchical_implementation"] = hierarchical_source_manifest(ROOT)
    source_signature = experiment_source_signature(source_lock)
    _write_json(output / "source_lock.json", source_lock)

    cache_lock = manifest.get("metadata", {}).get("source_lock", {})
    current_revision = source_lock["checkpoint"].get("revision")
    current_norm_hash = source_lock.get("norm_stats_sha256")
    if cache_lock.get("checkpoint", {}).get("revision") != current_revision:
        raise RuntimeError("R5 cache was generated from a different SimVLA checkpoint")
    if cache_lock.get("norm_stats_sha256") != current_norm_hash:
        raise RuntimeError("R5 cache was generated with different norm stats")

    previous_gate_path = Path(args.previous_r5_gate).expanduser().resolve()
    if not previous_gate_path.is_file():
        raise FileNotFoundError(previous_gate_path)
    previous_gate = json.loads(previous_gate_path.read_text(encoding="utf-8"))
    if int(previous_gate.get("execution_horizon", -1)) != execution_horizon:
        raise RuntimeError("previous R5 reference gate has the wrong execution horizon")

    k1_path = Path(args.k1_summary).expanduser().resolve()
    if not k1_path.is_file():
        raise FileNotFoundError(k1_path)
    k1 = json.loads(k1_path.read_text(encoding="utf-8"))

    model = load_source_locked_simvla(SmolVLMVLA, source_lock, device=device)
    model.action_space.load_norm_stats(args.norm_stats)
    freeze_module(model)
    action_adapter = SimVLAActionAdapter(model)
    if action_adapter.num_actions != action_horizon:
        raise RuntimeError("loaded checkpoint action horizon disagrees with cache")
    recurrent, recurrent_payload = _validate_checkpoint(
        checkpoints["condition"],
        expected_variant="chunk_aware_latentloop",
        execution_horizon=execution_horizon,
        action_horizon=action_horizon,
        device=device,
    )
    nonrecurrent, nonrecurrent_payload = _validate_checkpoint(
        checkpoints["nonrecurrent"],
        expected_variant="nonrecurrent_condition",
        execution_horizon=execution_horizon,
        action_horizon=action_horizon,
        device=device,
    )
    old_observation, old_payload = _validate_checkpoint(
        checkpoints["old_observation_only"],
        expected_variant="old_observation_only",
        execution_horizon=execution_horizon,
        action_horizon=action_horizon,
        device=device,
    )
    _, correction_payload = _validate_checkpoint(
        checkpoints["action_correction"],
        expected_variant="action_chunk_correction",
        execution_horizon=execution_horizon,
        action_horizon=action_horizon,
        device=torch.device("cpu"),
    )
    checkpoint_payloads = {
        "recurrent_age2": recurrent_payload,
        "nonrecurrent_anchor_age2": nonrecurrent_payload,
        "old_observation_only_age2": old_payload,
        "action_correction_online_component": correction_payload,
    }
    for name, payload in checkpoint_payloads.items():
        trained_lock = payload.get("metadata", {}).get("source_lock", {})
        if trained_lock.get("checkpoint", {}).get("revision") != current_revision:
            raise RuntimeError(f"{name} was trained against a different SimVLA checkpoint")
        if trained_lock.get("norm_stats_sha256") != current_norm_hash:
            raise RuntimeError(f"{name} was trained against different norm stats")

    k1_pass = bool(k1.get("K1_PARITY_PASS", False))
    values: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in (*CANDIDATES, *REFERENCE_ROWS)
    }
    gripper_values: dict[str, list[float]] = defaultdict(list)
    pair_rows: list[dict[str, Any]] = []
    continuity_errors: list[str] = []
    teacher_reload_max = 0.0
    episodes: set[tuple[int, str]] = set()
    provenance_trace = simulate_token_provenance(
        action_horizon=action_horizon,
        execution_horizon=execution_horizon,
        levels=(2, 0, 1),
    )
    level1_reset = bool(provenance_trace[-1]["level1_provenance_reset"])
    elapsed_time = execution_horizon / float(args.control_hz)
    pair_iterator = iter_age2_pairs(
        args.cache,
        full_refresh_interval=args.full_refresh_interval,
        heldout_fraction=args.heldout_fraction,
        split_seed=args.split_seed,
        max_pairs=args.max_pairs,
    )
    progress = tqdm(pair_iterator, desc="R5 exact-age2 regeneration", mininterval=args.tqdm_mininterval)
    run_started = time.perf_counter()
    with torch.no_grad():
        for pair_index, (q0, q1, errors) in enumerate(progress):
            key = (int(q0["task_id"]), str(q0["episode_id"]))
            episodes.add(key)
            continuity_errors.extend(
                f"task={key[0]} episode={key[1]} q={q0['query_index']}: {error}"
                for error in errors
            )
            tensors = {
                "c0": _batch(q0["full_condition"], device),
                "rgb0": _batch(q0["raw_rgb"], device),
                "p0": _batch(q0["proprio"], device),
                "e0": _batch(q0["executed_subchunk"], device),
                "rgb1": _batch(q0["next_raw_rgb"], device),
                "p1": _batch(q0["next_proprio"], device),
                "e1": _batch(q1["executed_subchunk"], device),
                "rgb2": _batch(q1["next_raw_rgb"], device),
                "p2": _batch(q1["next_proprio"], device),
                "c2": _batch(q1["next_full_condition"], device),
                "a2": _batch(q1["next_teacher_action_chunk"], device),
                "noise2": _batch(q1["next_initial_noise"], device),
            }

            recurrent_c1 = _recurrent_step(
                recurrent,
                tensors["c0"],
                previous_rgb=tensors["rgb0"],
                current_rgb=tensors["rgb1"],
                previous_proprio=tensors["p0"],
                current_proprio=tensors["p1"],
                executed=tensors["e0"],
                execution_horizon=execution_horizon,
                elapsed_time=elapsed_time,
                query_age=1,
            )
            old_c1 = _recurrent_step(
                old_observation,
                tensors["c0"],
                previous_rgb=tensors["rgb0"],
                current_rgb=tensors["rgb1"],
                previous_proprio=tensors["p0"],
                current_proprio=tensors["p1"],
                executed=tensors["e0"],
                execution_horizon=execution_horizon,
                elapsed_time=elapsed_time,
                query_age=1,
            )

            predictions: dict[str, tuple[Tensor, Tensor, float]] = {}
            _sync(device)
            started = time.perf_counter()
            recurrent_c2 = _recurrent_step(
                recurrent,
                recurrent_c1,
                previous_rgb=tensors["rgb1"],
                current_rgb=tensors["rgb2"],
                previous_proprio=tensors["p1"],
                current_proprio=tensors["p2"],
                executed=tensors["e1"],
                execution_horizon=execution_horizon,
                elapsed_time=elapsed_time,
                query_age=2,
            )
            recurrent_a2 = action_adapter.decode_action_from_condition(
                recurrent_c2,
                tensors["p2"],
                steps=args.flow_steps,
                initial_noise=tensors["noise2"],
            )
            _sync(device)
            predictions["recurrent_age2"] = (
                recurrent_c2,
                recurrent_a2,
                (time.perf_counter() - started) * 1000.0,
            )

            _sync(device)
            started = time.perf_counter()
            nonrecurrent_c2 = _nonrecurrent_age2(
                nonrecurrent,
                tensors["c0"],
                anchor_rgb=tensors["rgb0"],
                current_rgb=tensors["rgb2"],
                anchor_proprio=tensors["p0"],
                current_proprio=tensors["p2"],
                executed_history=(tensors["e0"], tensors["e1"]),
                execution_horizon=execution_horizon,
                elapsed_time=elapsed_time,
            )
            nonrecurrent_a2 = action_adapter.decode_action_from_condition(
                nonrecurrent_c2,
                tensors["p2"],
                steps=args.flow_steps,
                initial_noise=tensors["noise2"],
            )
            _sync(device)
            predictions["nonrecurrent_anchor_age2"] = (
                nonrecurrent_c2,
                nonrecurrent_a2,
                (time.perf_counter() - started) * 1000.0,
            )

            _sync(device)
            started = time.perf_counter()
            old_c2 = _recurrent_step(
                old_observation,
                old_c1,
                previous_rgb=tensors["rgb1"],
                current_rgb=tensors["rgb2"],
                previous_proprio=tensors["p1"],
                current_proprio=tensors["p2"],
                executed=tensors["e1"],
                execution_horizon=execution_horizon,
                elapsed_time=elapsed_time,
                query_age=2,
            )
            old_a2 = action_adapter.decode_action_from_condition(
                old_c2,
                tensors["p2"],
                steps=args.flow_steps,
                initial_noise=tensors["noise2"],
            )
            _sync(device)
            predictions["old_observation_only_age2"] = (
                old_c2,
                old_a2,
                (time.perf_counter() - started) * 1000.0,
            )

            _sync(device)
            started = time.perf_counter()
            hold_a2 = action_adapter.decode_action_from_condition(
                tensors["c0"],
                tensors["p2"],
                steps=args.flow_steps,
                initial_noise=tensors["noise2"],
            )
            _sync(device)
            predictions["hold_stale_condition"] = (
                tensors["c0"],
                hold_a2,
                (time.perf_counter() - started) * 1000.0,
            )

            _sync(device)
            started = time.perf_counter()
            teacher_reload = action_adapter.decode_action_from_condition(
                tensors["c2"],
                tensors["p2"],
                steps=args.flow_steps,
                initial_noise=tensors["noise2"],
            )
            _sync(device)
            teacher_latency = (time.perf_counter() - started) * 1000.0
            teacher_reload_max = max(
                teacher_reload_max,
                float((teacher_reload - tensors["a2"]).abs().max().item()),
            )
            predictions["full_teacher_reference"] = (
                tensors["c2"],
                tensors["a2"],
                teacher_latency,
            )

            for row_name, (condition, action, latency_ms) in predictions.items():
                metrics = {
                    **_condition_metrics(condition, tensors["c2"]),
                    **_action_metrics(action, tensors["a2"], execution_horizon),
                    "latency_ms": latency_ms,
                }
                for metric, value in metrics.items():
                    values[row_name][metric].append(float(value))
                gripper_values[row_name].extend(
                    action[0, :execution_horizon, 6].detach().cpu().tolist()
                )
                pair_rows.append(
                    {
                        "row": row_name,
                        "task_id": key[0],
                        "episode_id": key[1],
                        "anchor_query_index": int(q0["query_index"]),
                        "target_query_index": int(q1["next_query_index"]),
                        **metrics,
                    }
                )
            progress.set_postfix(pairs=pair_index + 1, episodes=len(episodes))
    progress.close()

    if not pair_rows:
        raise RuntimeError("no held-out exact-age2 pairs were selected")
    pair_count = len(pair_rows) // len(values)
    cache_continuity_pass = not continuity_errors
    reload_pass = teacher_reload_max <= args.teacher_reload_tolerance
    old_prefix = values["old_observation_only_age2"]["prefix_l1"]
    gate_inputs: list[RegenerationCandidateGateInputs] = []
    for offset, name in enumerate(CANDIDATES):
        candidate_prefix = values[name]["prefix_l1"]
        hold_prefix = values["hold_stale_condition"]["prefix_l1"]
        candidate_gripper = gripper_values[name]
        signs = {value >= 0.0 for value in candidate_gripper}
        gate_inputs.append(
            RegenerationCandidateGateInputs(
                name=name,
                finite=all(value == 1.0 for value in values[name]["finite"]),
                mean_prefix_l1=float(distribution_summary(candidate_prefix)["mean"]),
                candidate_minus_hold_prefix_l1_ci95=_paired_ci(
                    (candidate - hold for candidate, hold in zip(candidate_prefix, hold_prefix)),
                    seed=args.bootstrap_seed + 2 * offset,
                    samples=args.bootstrap_samples,
                ),
                candidate_minus_old_observation_prefix_l1_ci95=_paired_ci(
                    (candidate - old for candidate, old in zip(candidate_prefix, old_prefix)),
                    seed=args.bootstrap_seed + 2 * offset + 1,
                    samples=args.bootstrap_samples,
                ),
                gripper_noncollapsed=len(signs) == 2 and len(set(candidate_gripper)) > 1,
                prefix_l1_p99=float(distribution_summary(candidate_prefix)["p99"]),
                old_observation_prefix_l1_p99=float(distribution_summary(old_prefix)["p99"]),
                level1_provenance_reset=level1_reset,
                latency_ms_mean=float(distribution_summary(values[name]["latency_ms"])["mean"]),
            )
        )
    gate = evaluate_r5_regeneration_gate(
        gate_inputs,
        k1_parity_pass=k1_pass,
        exact_age2_pairs_present=pair_count > 0,
        cache_continuity_pass=cache_continuity_pass,
        same_noise_teacher_reload_pass=reload_pass,
    )
    gate.update(
        {
            "schema_version": SCHEMA_VERSION,
            "source_signature": source_signature,
            "action_horizon": action_horizon,
            "execution_horizon": execution_horizon,
            "full_refresh_interval": args.full_refresh_interval,
            "first_provenance_exhaustion_query": schedule.first_exhaustion_query,
            "native_level_sequence": list(schedule.levels_through_next_full_refresh),
            "pairs": pair_count,
            "episodes": len(episodes),
            "cache_continuity_error_count": len(continuity_errors),
            "cache_continuity_errors_first20": continuity_errors[:20],
            "teacher_same_noise_reload_max_abs_diff": teacher_reload_max,
            "teacher_reload_tolerance": args.teacher_reload_tolerance,
            "gripper_representation": {
                "type": "continuous_postprocessed_command",
                "logit_error_applicable": False,
                "probability_error_applicable": False,
                "reported_instead": [
                    "gripper_command_l1",
                    "gripper_command_l2",
                    "gripper_sign_agreement",
                    "gripper_switch_agreement",
                ],
            },
            "nonrecurrent_elapsed_time_semantics": (
                "matches current online policy: per-query R/control_hz with query_age=2; "
                "both executed subchunks are encoded and mean-pooled"
            ),
            "prior_adjacent_age1_gate": args.previous_r5_gate,
            "elapsed_seconds": time.perf_counter() - run_started,
        }
    )
    _write_json(output / "r5_regeneration_gate.json", gate)
    _write_json(
        output / "r5_regeneration_metrics.json",
        {
            "schema_version": SCHEMA_VERSION,
            "rows": {
                name: {metric: distribution_summary(samples) for metric, samples in metrics.items()}
                for name, metrics in values.items()
            },
            "parameter_counts": {
                "recurrent_age2": sum(parameter.numel() for parameter in recurrent.parameters()),
                "nonrecurrent_anchor_age2": sum(
                    parameter.numel() for parameter in nonrecurrent.parameters()
                ),
                "old_observation_only_age2": sum(
                    parameter.numel() for parameter in old_observation.parameters()
                ),
                "full_teacher_reference": 0,
                "hold_stale_condition": 0,
            },
            "gate": gate,
        },
    )
    gate_by_name = gate["candidates"]
    parameter_counts = {
        "recurrent_age2": sum(parameter.numel() for parameter in recurrent.parameters()),
        "nonrecurrent_anchor_age2": sum(parameter.numel() for parameter in nonrecurrent.parameters()),
        "old_observation_only_age2": sum(parameter.numel() for parameter in old_observation.parameters()),
        "full_teacher_reference": 0,
        "hold_stale_condition": 0,
    }
    summary_rows = [
        _summary_row(
            name,
            "candidate" if name in CANDIDATES else "reference",
            values[name],
            trainable_parameters=parameter_counts[name],
            gate=gate_by_name.get(name),
        )
        for name in (*CANDIDATES, *REFERENCE_ROWS)
    ]
    _write_summary_csv(output / "r5_regeneration_offline_summary.csv", summary_rows)
    with (output / "r5_regeneration_pair_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)
    _write_report(
        output / "r5_regeneration_gate_report.md",
        gate=gate,
        pairs=pair_count,
        episodes=len(episodes),
        cache=str(Path(args.cache).resolve()),
        old_gate_path=args.previous_r5_gate,
    )
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--smolvlm-model-path", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--k1-summary", required=True)
    parser.add_argument("--recurrent-checkpoint", required=True)
    parser.add_argument("--nonrecurrent-checkpoint", required=True)
    parser.add_argument("--old-observation-checkpoint", required=True)
    parser.add_argument("--action-correction-checkpoint", required=True)
    parser.add_argument("--previous-r5-gate", required=True)
    parser.add_argument("--full-refresh-interval", type=int, default=4)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260804)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--teacher-reload-tolerance", type=float, default=1e-6)
    parser.add_argument("--tqdm-mininterval", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.control_hz <= 0 or not 0 < args.heldout_fraction < 1:
        raise ValueError("control-hz must be positive and heldout-fraction must be in (0,1)")
    gate = run(args)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
