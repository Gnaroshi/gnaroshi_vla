"""Two-rank recurrence-focused V3 trainer and corrected offline gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import (
    _drop_unused_vlm,
)
from architectures.simvla.adapters.latentloop.native_v0_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SMOLVLM,
    freeze_module,
    load_frozen_simvla,
    move_batch,
)
from architectures.simvla.adapters.latentloop.stability_alignment.checkpoint import (
    GroupWarmupCosine,
)
from architectures.simvla.adapters.latentloop.stability_alignment.contracts import (
    GRAD_CLIP_NORM,
    atomic_write_json,
    canonical_sha256,
    load_json,
    sha256_file,
)
from architectures.simvla.adapters.latentloop.stability_alignment.model import (
    StabilityAlignedModules,
    configure_condition_only_stage,
    load_warm_start,
    optimizer_parameter_groups,
    zero_code_parity,
)
from architectures.simvla.adapters.latentloop.stability_alignment.trainer import (
    _allreduce_gradients,
    _assert_parent_contract,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_checkpoint import (
    load_v3_checkpoint,
    save_v3_checkpoint,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_contracts import (
    V3_HARD_POOL_SCHEMA,
    V3_LOSS_NAMES,
    V3_LOSS_SCHEMA,
    evaluate_v3_moving_window,
    evaluate_v3_scientific_gate,
    evaluate_v3_stage_gate,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_data import (
    V3PoolSampler,
    V3StabilityExactTeacherDataset,
    collate_v3_sequences,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_diagnostics import (
    _comparison,
    _evaluate_adapter,
    _merge_rank_rows,
    _write_rows,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_prepare import (
    _v3_forward,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_runtime import (
    DEFAULT_CACHE,
    DEFAULT_GENERATION_30K,
    DEFAULT_NORM,
    cleanup_distributed,
    condition_updater_parameters,
    configure_determinism,
    distributed_pair,
    flatten_gradients,
    v3_source_lock,
)
from architectures.simvla.adapters.latentloop.stability_alignment.v3_objectives import (
    weighted_v3_total,
)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _write_latest(path: Path, checkpoint: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(str(checkpoint) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_contracts(
    args: argparse.Namespace, source: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float], float]:
    pool = load_json(args.hard_pool)
    weights_payload = load_json(args.loss_weights)
    if pool.get("schema_version") != V3_HARD_POOL_SCHEMA:
        raise ValueError("V3 hard-pool schema changed")
    if weights_payload.get("schema_version") != V3_LOSS_SCHEMA:
        raise ValueError("V3 loss-weight schema changed")
    approved_verdicts = {
        "STABILITY_V3_WEIGHTS_APPROVED",
        "STABILITY_V3_BOUNDED_PILOT_APPROVED_WITH_CONFLICT_WARNING",
        "STABILITY_V3_BOUNDED_PILOT_APPROVED_WITH_ALIGNMENT_WARNING",
    }
    if not bool(weights_payload.get("approved_for_bounded_pilot")) or weights_payload.get(
        "verdict"
    ) not in approved_verdicts:
        raise RuntimeError("V3 gradient-calibrated loss weights are not approved")
    expected = str(source["combined_sha256"])
    if pool["source_lock"]["combined_sha256"] != expected:
        raise RuntimeError("V3 hard pool uses another source lock")
    if weights_payload["source_contract_sha256"] != expected:
        raise RuntimeError("V3 weights use another source lock")
    if weights_payload["hard_pool_combined_sha256"] != pool["combined_sha256"]:
        raise RuntimeError("V3 weights use another hard-pool contract")
    weights = {
        name: float(weights_payload["weights"][name]) for name in V3_LOSS_NAMES
    }
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("V3 weights must be finite and non-negative")
    gamma = float(weights_payload["gamma"])
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("V3 frozen gamma must be finite and non-negative")
    return pool, weights_payload, weights, gamma


def _wandb(args: argparse.Namespace, rank: int, config: Mapping[str, Any]) -> Any | None:
    if rank != 0 or not args.wandb_project:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or Path(args.output).name,
        dir=args.output,
        config=dict(config),
    )


def command_train(args: argparse.Namespace) -> dict[str, Any]:
    rank, _, _, device = distributed_pair()
    run = None
    try:
        configure_determinism(args.seed)
        output = Path(args.output).expanduser().resolve()
        resume = Path(args.resume).expanduser().resolve() if args.resume else None
        if rank == 0:
            if resume is None and output.exists():
                raise FileExistsError(f"refusing existing V3 training output: {output}")
            if resume is not None and not output.is_dir():
                raise FileNotFoundError("V3 resume output directory is missing")
            output.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        source = v3_source_lock(
            repository=args.repository,
            cache=args.cache,
            condition_parent=args.condition_parent,
            generation_parent=args.generation_parent,
            norm_stats=args.norm_stats,
            checkpoint=args.checkpoint,
            smolvlm_model=args.smolvlm_model,
            split_seed=args.split_seed,
            training_seed=args.seed,
        )
        pool, weights_payload, weights, gamma = _load_contracts(args, source)
        modules, parent, parent_generation, payloads = load_warm_start(
            condition_checkpoint=args.condition_parent,
            generation_checkpoint=args.generation_parent,
            device=device,
        )
        parent_identity = _assert_parent_contract(payloads)
        stage_audit = configure_condition_only_stage(modules)
        freeze_module(parent)
        freeze_module(parent_generation)
        parity = zero_code_parity(parent_generation, modules.generation, device=device)
        if parity["verdict"] != "ZERO_CODE_PARENT_PARITY_PASS":
            raise RuntimeError(json.dumps(parity, indent=2, sort_keys=True))
        groups = optimizer_parameter_groups(
            modules,
            base_lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        optimizer = torch.optim.AdamW(groups)
        scheduler = GroupWarmupCosine(optimizer)
        start_step = 0
        checkpoint_payload: dict[str, Any] | None = None
        if resume is not None:
            checkpoint_payload = load_v3_checkpoint(
                resume,
                modules=modules,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            start_step = int(checkpoint_payload["optimizer_step"])
            if checkpoint_payload["source_lock"]["combined_sha256"] != source[
                "combined_sha256"
            ]:
                raise RuntimeError("V3 resume source lock changed")
            if checkpoint_payload["loss_weight_contract"]["combined_sha256"] != weights_payload[
                "combined_sha256"
            ]:
                raise RuntimeError("V3 resume loss weights changed")
        if int(args.stop_step) not in {500, 2_000, 5_000, 10_000}:
            raise ValueError("V3 training may stop only at 500, 2K, 5K, or 10K")
        if int(args.stop_step) <= start_step:
            raise ValueError("V3 stop step must be after the resume step")
        scheduler.set_step(start_step)
        frozen_model, processor, action_adapter = load_frozen_simvla(
            checkpoint=args.checkpoint,
            norm_stats=args.norm_stats,
            smolvlm_model=args.smolvlm_model,
            device=device,
        )
        del processor
        _drop_unused_vlm(frozen_model)
        dataset = V3StabilityExactTeacherDataset(
            args.cache, split="train", split_seed=args.split_seed
        )
        if dataset.split_sha256 != pool["dataset_contract"]["split_sha256"]:
            raise RuntimeError("V3 training split changed after pool construction")
        sampler = V3PoolSampler(
            pool,
            seed=args.seed,
            start_step=start_step,
            stop_step=args.stop_step,
        )
        training_contract = {
            "schema_version": "simvla_stability_v3_training_contract_v2",
            "branch": str(args.branch),
            "source_combined_sha256": source["combined_sha256"],
            "condition_parent_step": parent_identity["condition_optimizer_step"],
            "generation_parent_step": 30_000,
            "start_step": start_step,
            "stop_step": int(args.stop_step),
            "scheduler_horizon": 30_000,
            "warmup_steps": 1_500,
            "effective_unique_global_batch": 1,
            "physical_replicas": 2,
            "sampling_ratio": pool["sampling_ratio"],
            "hard_pool_combined_sha256": pool["combined_sha256"],
            "gamma": gamma,
            "loss_weights_combined_sha256": weights_payload["combined_sha256"],
            "parent_preservation_contribution": 0.0,
            "student_path": "recursive_only",
            "teacher_path": "frozen_parent_teacher_forced",
            "generation_change_code": "zero_128d",
            "condition_age_weights": {"2": 1.0, "3": 2.0},
            "audit_interval": int(args.audit_interval),
            "audit_window": int(args.audit_window),
            "clipping_window": int(args.clipping_window),
            "calibration_approval_scope": weights_payload["approval_scope"],
            "moving_window_policy": {
                "gradient_share_targets": "diagnostic_warning",
                "clipping_fraction": "hard_numerical_safety_check",
                "mid_segment_abort": "nonfinite_loss_only",
                "continuation": "offline_multi_metric_gate_and_numerical_safety",
            },
            "stage_audit": stage_audit,
            "zero_code_parity": parity,
        }
        training_contract["combined_sha256"] = canonical_sha256(training_contract)
        if rank == 0:
            atomic_write_json(
                output / f"training_contract_step_{int(args.stop_step):06d}.json",
                training_contract,
            )
            atomic_write_json(output / "training_contract.json", training_contract)
            if start_step == 0:
                atomic_write_json(output / "source_lock.json", source)
                atomic_write_json(output / "parameter_audit.json", modules.parameter_audit())
        run = _wandb(
            args,
            rank,
            {
                **training_contract,
                "loss_weights": weights,
            },
        )
        audit_parameters = condition_updater_parameters(modules)
        trainable = tuple(
            parameter for parameter in modules.parameters() if parameter.requires_grad
        )
        gradient_window: deque[dict[str, float]] = deque(maxlen=args.audit_window)
        clipping_window: deque[bool] = deque(maxlen=args.clipping_window)
        last_audit: dict[str, Any] = (
            dict(checkpoint_payload.get("moving_window_audit", {}))
            if checkpoint_payload
            else {}
        )
        audit_history: list[dict[str, Any]] = []
        started = time.perf_counter()
        progress = tqdm(
            range(start_step, int(args.stop_step)),
            disable=rank != 0,
            desc=f"stability V3 {args.branch}",
            dynamic_ncols=True,
        )
        for optimizer_step in progress:
            dataset_index, stream = sampler.index(optimizer_step)
            batch = move_batch(collate_v3_sequences([dataset[dataset_index]]), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                raw, diagnostics = _v3_forward(
                    modules=modules,
                    parent=parent,
                    frozen_model=frozen_model,
                    action_adapter=action_adapter,
                    batch=batch,
                    optimizer_step=optimizer_step,
                    gamma=gamma,
                    hard_sample=stream == "recurrence_action_tail",
                )
                total, weighted = weighted_v3_total(raw, weights)
            if not bool(torch.isfinite(total).item()):
                raise FloatingPointError(f"non-finite V3 loss at step {optimizer_step}")
            audit_now = optimizer_step % int(args.audit_interval) == 0
            audit_row: dict[str, float] | None = None
            if audit_now:
                audit_row = {}
                for name in V3_LOSS_NAMES:
                    vector = flatten_gradients(
                        raw[name], audit_parameters, retain_graph=True
                    )
                    audit_row[name] = weights[name] * float(vector.norm().item())
                gradient_window.append(audit_row)
            total.backward()
            _allreduce_gradients(modules)
            preclip = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
            clipping_window.append(float(preclip.item()) > float(GRAD_CLIP_NORM))
            optimizer.step()
            scheduler.set_step(optimizer_step + 1)
            if audit_now and len(gradient_window) >= int(args.audit_window):
                last_audit = evaluate_v3_moving_window(
                    tuple(gradient_window), clipping_flags=tuple(clipping_window)
                )
                audit_record = {
                    "optimizer_step": optimizer_step + 1,
                    **last_audit,
                }
                audit_history.append(audit_record)
                if rank == 0:
                    atomic_write_json(
                        output / f"moving_window_audit_{optimizer_step + 1:06d}.json",
                        audit_record,
                    )
            if rank == 0 and (
                optimizer_step == start_step
                or (optimizer_step + 1) % int(args.log_interval) == 0
            ):
                row = {
                    "step": optimizer_step + 1,
                    "dataset_index": int(dataset_index),
                    "sampling_stream": str(stream),
                    "loss": float(total.detach().item()),
                    "preclip_gradient_norm": float(preclip.item()),
                    "gradient_clipped": float(preclip.item()) > GRAD_CLIP_NORM,
                    **{
                        f"raw/{name}": float(raw[name].detach().item())
                        for name in V3_LOSS_NAMES
                    },
                    **{
                        f"weighted/{name}": float(weighted[name].detach().item())
                        for name in V3_LOSS_NAMES
                    },
                    **{
                        f"diagnostic/{name}": float(value.detach().mean().item())
                        for name, value in diagnostics.items()
                    },
                    **{
                        f"lr/{group['name']}": float(group["lr"])
                        for group in optimizer.param_groups
                    },
                }
                if last_audit:
                    row.update(
                        {
                            "audit/recurrence_share": last_audit.get(
                                "weighted_gradient_shares", {}
                            ).get("recurrence_gain"),
                            "audit/clipping_fraction": last_audit.get(
                                "clipping_fraction"
                            ),
                        }
                    )
                _append_jsonl(output / "train_metrics.jsonl", row)
                progress.set_postfix(
                    loss=f"{row['loss']:.4g}",
                    gain=f"{row['raw/recurrence_gain']:.4g}",
                    stream=stream,
                )
                if run is not None:
                    run.log(row, step=optimizer_step + 1)
        if len(gradient_window) < int(args.audit_window):
            raise RuntimeError("V3 segment ended before a complete gradient audit window")
        last_audit = evaluate_v3_moving_window(
            tuple(gradient_window), clipping_flags=tuple(clipping_window)
        )
        result: dict[str, Any] = {}
        if rank == 0:
            atomic_write_json(
                output / f"moving_window_audits_step_{int(args.stop_step):06d}.json",
                {
                    "schema_version": "simvla_stability_v3_moving_window_history_v1",
                    "start_step": start_step,
                    "stop_step": int(args.stop_step),
                    "audits": audit_history,
                    "final_audit": last_audit,
                },
            )
            checkpoint = output / "checkpoints" / (
                f"stability_v3_step_{int(args.stop_step):06d}.pt"
            )
            save_v3_checkpoint(
                checkpoint,
                modules=modules,
                optimizer=optimizer,
                scheduler=scheduler,
                optimizer_step=args.stop_step,
                sampler_state=sampler.state_dict(args.stop_step),
                source_lock=source,
                training_contract=training_contract,
                parent_identity=parent_identity,
                loss_weight_contract=weights_payload,
                moving_window_audit=last_audit,
            )
            _write_latest(output / "latest_checkpoint.txt", checkpoint)
            result = {
                "schema_version": "simvla_stability_v3_train_summary_v2",
                "verdict": (
                    "STABILITY_V3_TRAINING_SEGMENT_COMPLETE"
                    if last_audit["passed"]
                    else "STABILITY_V3_TRAINING_SEGMENT_COMPLETE_WITH_AUDIT_WARNING"
                ),
                "branch": str(args.branch),
                "start_step": start_step,
                "optimizer_step": int(args.stop_step),
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "moving_window_audit": last_audit,
                "moving_window_warning_count": sum(
                    not bool(row["passed"]) for row in audit_history
                ),
                "source_combined_sha256": source["combined_sha256"],
            }
            atomic_write_json(
                output / f"run_summary_step_{int(args.stop_step):06d}.json", result
            )
            if run is not None:
                run.finish()
        dist.barrier()
        return result
    finally:
        cleanup_distributed()


def command_offline(args: argparse.Namespace) -> dict[str, Any]:
    rank, _, world, device = distributed_pair()
    try:
        configure_determinism(args.seed)
        output = Path(args.output).expanduser().resolve()
        if rank == 0:
            if output.exists():
                raise FileExistsError(f"refusing existing V3 offline output: {output}")
            (output / "shards").mkdir(parents=True)
        dist.barrier()
        modules, parent, parent_generation, payloads = load_warm_start(
            condition_checkpoint=args.condition_parent,
            generation_checkpoint=args.generation_parent,
            device=device,
        )
        _assert_parent_contract(payloads)
        parent_modules = StabilityAlignedModules(parent, parent_generation).to(device)
        freeze_module(parent_modules)
        candidate_payload = load_v3_checkpoint(args.candidate, modules=modules)
        freeze_module(modules)
        source = v3_source_lock(
            repository=args.repository,
            cache=args.cache,
            condition_parent=args.condition_parent,
            generation_parent=args.generation_parent,
            norm_stats=args.norm_stats,
            checkpoint=args.checkpoint,
            smolvlm_model=args.smolvlm_model,
            split_seed=args.split_seed,
            training_seed=args.seed,
        )
        if candidate_payload["source_lock"]["combined_sha256"] != source[
            "combined_sha256"
        ]:
            raise RuntimeError("V3 offline candidate source lock changed")
        frozen_model, processor, action_adapter = load_frozen_simvla(
            checkpoint=args.checkpoint,
            norm_stats=args.norm_stats,
            smolvlm_model=args.smolvlm_model,
            device=device,
        )
        del processor
        _drop_unused_vlm(frozen_model)
        dataset = V3StabilityExactTeacherDataset(
            args.cache, split=args.split, split_seed=args.split_seed
        )
        parent_rows = _evaluate_adapter(
            modules=parent_modules,
            frozen_model=frozen_model,
            action_adapter=action_adapter,
            dataset=dataset,
            rank=rank,
            world=world,
            device=device,
            description=f"V3 {args.branch} parent {args.split}",
        )
        candidate_rows = _evaluate_adapter(
            modules=modules,
            frozen_model=frozen_model,
            action_adapter=action_adapter,
            dataset=dataset,
            rank=rank,
            world=world,
            device=device,
            description=f"V3 {args.branch} candidate {args.split}",
        )
        atomic_write_json(
            output / "shards" / f"rank_{rank}_parent.json", parent_rows
        )
        atomic_write_json(
            output / "shards" / f"rank_{rank}_candidate.json", candidate_rows
        )
        dist.barrier()
        result: dict[str, Any] = {}
        if rank == 0:
            parent_merged = _merge_rank_rows(output, "parent", world)
            candidate_merged = _merge_rank_rows(output, "candidate", world)
            comparison = _comparison(
                parent_merged,
                candidate_merged,
                branch=args.branch,
                split=args.split,
                optimizer_step=int(candidate_payload["optimizer_step"]),
                checkpoint=args.candidate,
            )
            step = int(candidate_payload["optimizer_step"])
            previous = None
            if args.previous_gate:
                previous = load_json(args.previous_gate)["comparison"]["metrics"]
            gate = (
                evaluate_v3_scientific_gate(comparison["metrics"])
                if args.split == "final_offline"
                else evaluate_v3_stage_gate(
                    comparison["metrics"],
                    optimizer_step=step,
                    previous_metrics=previous,
                )
            )
            result = {
                "schema_version": "simvla_stability_v3_offline_gate_v1",
                "verdict": gate.verdict,
                "passed": gate.passed,
                "branch": str(args.branch),
                "split": str(args.split),
                "optimizer_step": step,
                "candidate": str(Path(args.candidate).expanduser().resolve()),
                "candidate_sha256": sha256_file(args.candidate),
                "dataset_contract": dataset.contract(),
                "comparison": comparison,
                "gate": gate.to_dict(),
                "source_combined_sha256": source["combined_sha256"],
                "evaluation_generation_change_code": "zero_128d",
            }
            atomic_write_json(output / "offline_gate.json", result)
            _write_rows(output / "parent_rows.csv", parent_merged)
            _write_rows(output / "candidate_rows.csv", candidate_merged)
        dist.barrier()
        return result
    finally:
        cleanup_distributed()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(value: argparse.ArgumentParser) -> None:
        value.add_argument("--repository", required=True)
        value.add_argument("--output", required=True)
        value.add_argument("--branch", choices=("R50", "R150"), required=True)
        value.add_argument("--cache", default=DEFAULT_CACHE)
        value.add_argument("--condition-parent", required=True)
        value.add_argument("--generation-parent", default=DEFAULT_GENERATION_30K)
        value.add_argument("--norm-stats", default=DEFAULT_NORM)
        value.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
        value.add_argument("--smolvlm-model", default=DEFAULT_SMOLVLM)
        value.add_argument("--split-seed", type=int, default=20260822)
        value.add_argument("--seed", type=int, default=20260825)
        value.add_argument(
            "--bf16", action=argparse.BooleanOptionalAction, default=True
        )

    train = subparsers.add_parser("train")
    common(train)
    train.add_argument("--hard-pool", required=True)
    train.add_argument("--loss-weights", required=True)
    train.add_argument("--resume")
    train.add_argument("--stop-step", type=int, required=True)
    train.add_argument("--learning-rate", type=float, default=3e-5)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--log-interval", type=int, default=20)
    # 97 is coprime with the sampler's 20-step cycle, so moving-window audits
    # observe base, gripper, and tail streams instead of aliasing to base only.
    train.add_argument("--audit-interval", type=int, default=97)
    train.add_argument("--audit-window", type=int, default=16)
    train.add_argument("--clipping-window", type=int, default=200)
    train.add_argument("--wandb-project", default="gnaroshi-simvla-stability-v3")
    train.add_argument("--wandb-name")

    offline = subparsers.add_parser("offline")
    common(offline)
    offline.add_argument("--candidate", required=True)
    offline.add_argument(
        "--split",
        choices=("checkpoint_validation", "final_offline"),
        required=True,
    )
    offline.add_argument("--previous-gate")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = command_train(args) if args.command == "train" else command_offline(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
