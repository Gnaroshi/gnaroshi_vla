"""Guarded cache-backed trainer for the two native-R5 exact-q2 candidates."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "architectures" / "simvla" / "upstream"
for path in (ROOT, UPSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from architectures.simvla.adapters.dcld import SimVLAActionAdapter  # noqa: E402
from architectures.simvla.adapters.exact_q2.simvla_exact_q2_adapter import (  # noqa: E402
    build_exact_q2_adapter,
    freeze_module,
    load_exact_q2_checkpoint,
    parameter_budget_audit,
    save_exact_q2_checkpoint,
    trainable_parameter_names,
)
from architectures.simvla.adapters.hierarchical_correction.source_locked_loading import (  # noqa: E402
    load_source_locked_simvla,
)
from architectures.simvla.adapters.latentloop.source_lock import (  # noqa: E402
    collect_source_lock,
    require_empty_output,
    resolve_huggingface_checkpoint,
    sha256_file,
)
from methods.latentloop.training import DeterministicStepBatchSampler  # noqa: E402
from methods.simvla_exact_q2.dataset import (  # noqa: E402
    ExactQ2TupleDataset,
    collate_exact_q2,
)
from methods.simvla_exact_q2.losses import (  # noqa: E402
    RawLossScaleAccumulator,
    compute_exact_q2_losses,
    load_approved_loss_contract,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
    }
    if device.type == "cuda":
        state["torch_cuda_all"] = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # map_location="cuda" also moves the CPU generator state into CUDA memory.
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda_all"]])


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _source_signature(source_lock: dict[str, Any]) -> dict[str, Any]:
    checkpoint = source_lock["checkpoint"]
    return {
        "simvla_upstream_commit": source_lock["simvla_upstream_commit"],
        "checkpoint_identifier": checkpoint["identifier"],
        "checkpoint_revision": checkpoint["revision"],
        "checkpoint_blob_sha256": checkpoint["hf_blob_key_sha256"],
        "norm_stats_sha256": source_lock["norm_stats_sha256"],
    }


def _require_cache_source_match(dataset_manifest: dict[str, Any], source_lock: dict[str, Any]) -> None:
    expected = dataset_manifest["source_signature"]
    actual = _source_signature(source_lock)
    fields = (
        "simvla_upstream_commit",
        "checkpoint_identifier",
        "checkpoint_revision",
        "checkpoint_blob_sha256",
        "norm_stats_sha256",
    )
    mismatch = {name: (expected.get(name), actual.get(name)) for name in fields if expected.get(name) != actual.get(name)}
    if mismatch:
        raise RuntimeError(f"training runtime differs from cache source lock: {mismatch}")


def _same_noise_reload_check(
    action_adapter: SimVLAActionAdapter,
    batch: dict[str, Any],
    *,
    flow_steps: int,
) -> dict[str, Any]:
    with torch.no_grad():
        a1 = action_adapter.decode_action_from_condition(
            batch["c1_full"], batch["q1_proprio"], steps=flow_steps, initial_noise=batch["epsilon1"]
        )
        a2 = action_adapter.decode_action_from_condition(
            batch["c2_full"], batch["q2_proprio"], steps=flow_steps, initial_noise=batch["epsilon2"]
        )
    q1_max = float((a1 - batch["a1_full"]).abs().max().item())
    q2_max = float((a2 - batch["a2_full"]).abs().max().item())
    passed = q1_max <= 1e-5 and q2_max <= 1e-5
    if not passed:
        raise RuntimeError(f"same-noise teacher reload failed: q1={q1_max}, q2={q2_max}")
    return {
        "passed": passed,
        "q1_max_action_difference": q1_max,
        "q2_max_action_difference": q2_max,
        "epsilon1_sha256": list(batch["epsilon1_sha256"]),
        "epsilon2_sha256": list(batch["epsilon2_sha256"]),
    }


def _forward_losses(
    adapter: Any,
    action_adapter: SimVLAActionAdapter,
    batch: dict[str, Any],
    *,
    flow_steps: int,
    weights: Any,
) -> tuple[Any, dict[str, torch.Tensor]]:
    prediction = adapter(batch)
    a1_pred = action_adapter.decode_action_from_condition(
        prediction.c1,
        batch["q1_proprio"],
        steps=flow_steps,
        initial_noise=batch["epsilon1"],
        requires_grad=True,
    )
    a2_pred = action_adapter.decode_action_from_condition(
        prediction.c2,
        batch["q2_proprio"],
        steps=flow_steps,
        initial_noise=batch["epsilon2"],
        requires_grad=True,
    )
    losses = compute_exact_q2_losses(
        c0_full=batch["c0_full"],
        c1_pred=prediction.c1,
        c2_pred=prediction.c2,
        c1_full=batch["c1_full"],
        c2_full=batch["c2_full"],
        a1_pred=a1_pred,
        a2_pred=a2_pred,
        a1_full=batch["a1_full"],
        a2_full=batch["a2_full"],
        execution_horizon=adapter.config.execution_horizon,
        weights=weights,
    )
    return prediction, losses


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Calibrate raw scales or train only one exact-q2 lightweight candidate."""

    from models.modeling_smolvlm_vla import SmolVLMVLA

    if args.smoke and args.max_steps > 2:
        raise ValueError("smoke is capped at two optimizer steps")
    if args.calibrate_losses and args.resume_from:
        raise ValueError("calibration cannot be resumed")
    device = torch.device(args.device)
    _seed_everything(args.seed)
    resume_path = Path(args.resume_from).resolve() if args.resume_from else None
    output = require_empty_output(args.output)
    if resume_path and not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")

    dataset_manifest = json.loads(Path(args.dataset_manifest).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    source_lock = collect_source_lock(
        checkpoint=args.checkpoint,
        norm_stats_path=args.norm_stats,
    )
    source_lock["processor_checkpoint"] = resolve_huggingface_checkpoint(
        "HuggingFaceTB/SmolVLM-500M-Instruct"
    )
    _require_cache_source_match(dataset_manifest, source_lock)
    if resume_path:
        adapter, resume_payload = load_exact_q2_checkpoint(resume_path, device=device)
        if adapter.candidate != args.candidate:
            raise ValueError("resume candidate differs from --candidate")
        start_step = int(resume_payload["step"])
        previous_source = resume_payload["metadata"]["source_lock"]
        if _source_signature(previous_source) != _source_signature(source_lock):
            raise RuntimeError("resume runtime/source signature changed")
    else:
        adapter = build_exact_q2_adapter(args.candidate).to(device)
        resume_payload = None
        start_step = 0
    _write_json(output / "source_lock.json", source_lock)

    if args.calibrate_losses:
        weights = None
        loss_contract = None
        optimizer = None
        optimizer_steps = int(args.calibration_batches)
        gradient_accumulation = 1
    else:
        if not args.loss_contract:
            raise ValueError("training requires --loss-contract approved after raw calibration")
        weights, loss_contract = load_approved_loss_contract(args.loss_contract)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in adapter.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        optimizer_steps = int(args.max_steps)
        gradient_accumulation = int(args.gradient_accumulation_steps)
    if optimizer_steps < 1 or gradient_accumulation < 1:
        raise ValueError("step counts must be positive")
    if start_step > optimizer_steps:
        raise ValueError("resume step exceeds requested max steps")

    dataset = ExactQ2TupleDataset(
        dataset_manifest,
        split,
        partition="train",
    )
    sampler = DeterministicStepBatchSampler(
        dataset_size=len(dataset),
        batch_size=args.batch_size,
        seed=args.seed,
        start_step=start_step * gradient_accumulation,
        max_steps=optimizer_steps * gradient_accumulation,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_exact_q2,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_source_locked_simvla(SmolVLMVLA, source_lock, device=device)
    model.action_space.load_norm_stats(args.norm_stats)
    freeze_module(model)
    action_adapter = SimVLAActionAdapter(model)
    adapter.train()
    optimizer_names = trainable_parameter_names(adapter)
    if not optimizer_names:
        raise RuntimeError("candidate has no trainable parameters")
    teacher_trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if teacher_trainable:
        raise RuntimeError(f"frozen SimVLA leaked into optimizer: {teacher_trainable[:5]}")
    audit = parameter_budget_audit()
    if not audit["parameter_match_pass"]:
        raise RuntimeError("candidate parameter counts violate the frozen fairness criterion")
    freeze_snapshot = {
        "teacher_trainable_parameters": 0,
        "candidate": args.candidate,
        "candidate_trainable_parameters": sum(
            parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
        ),
        "optimizer_parameter_count": len(optimizer_names),
    }
    _write_json(output / "freeze_status_snapshot.json", freeze_snapshot)
    _write_json(output / "parameter_budget_audit.json", audit)
    (output / "optimizer_param_names.txt").write_text("\n".join(optimizer_names) + "\n", encoding="utf-8")
    _write_json(
        output / "dataset_contract.json",
        {
            "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
            "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
            "split": str(Path(args.split).resolve()),
            "split_sha256": sha256_file(args.split),
            "partition": "train",
            "tuples": len(dataset),
        },
    )
    if loss_contract is not None:
        _write_json(output / "loss_contract.json", loss_contract)

    if optimizer is not None and resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        _restore_rng_state(resume_payload["training_state"]["rng_state"], device)

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or output.name,
            dir=str(output),
            config=vars(args),
        )
    progress_path = output / "train_progress.jsonl"
    raw_scales = RawLossScaleAccumulator()
    same_noise_check: dict[str, Any] | None = None
    step = start_step
    micro_step = start_step * gradient_accumulation
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    stop_signal: dict[str, int | None] = {"value": None}

    def request_stop(signum: int, _frame: Any) -> None:
        stop_signal["value"] = int(signum)

    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    for signum in handlers:
        signal.signal(signum, request_stop)
    bar = tqdm(
        total=optimizer_steps,
        initial=start_step,
        desc=f"exact-q2/{args.candidate}",
        dynamic_ncols=True,
        mininterval=args.tqdm_mininterval,
    )
    started = time.time()
    last_losses: dict[str, float] = {}
    latest_checkpoint: Path | None = None

    def save_checkpoint() -> Path:
        assert optimizer is not None
        path = output / "checkpoints" / f"exact_q2_step_{step:06d}.pt"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
        save_exact_q2_checkpoint(
            path,
            adapter=adapter,
            step=step,
            metadata={
                "source_lock": source_lock,
                "args": vars(args),
                "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
                "split_sha256": sha256_file(args.split),
                "loss_contract": loss_contract,
                "selection_status": "UNSELECTED_TRAINING_CHECKPOINT",
            },
            optimizer_state_dict=optimizer.state_dict(),
            training_state={
                "rng_state": _rng_state(device),
                "micro_step": micro_step,
                "last_losses": last_losses,
            },
        )
        _atomic_text(output / "latest_checkpoint.txt", str(path) + "\n")
        return path

    try:
        for batch in loader:
            batch = _to_device(batch, device)
            if same_noise_check is None:
                same_noise_check = _same_noise_reload_check(
                    action_adapter, batch, flow_steps=args.flow_steps
                )
                _write_json(output / "same_noise_sanity.json", same_noise_check)
            _, losses = _forward_losses(
                adapter,
                action_adapter,
                batch,
                flow_steps=args.flow_steps,
                weights=weights,
            )
            raw_scales.update(losses)
            micro_step += 1
            if optimizer is not None:
                (losses["total"] / gradient_accumulation).backward()
            update_due = optimizer is None or micro_step % gradient_accumulation == 0
            if not update_due:
                continue
            if optimizer is not None:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            bar.update(1)
            last_losses = {
                name: float(value.detach().item())
                for name, value in losses.items()
            }
            if step == 1 or step == optimizer_steps or step % args.log_interval == 0:
                event = {
                    "candidate": args.candidate,
                    "step": step,
                    "max_steps": optimizer_steps,
                    "elapsed_seconds": time.time() - started,
                    "losses": last_losses,
                }
                _append_jsonl(progress_path, event)
                bar.set_postfix(
                    q2_prefix=f"{last_losses['q2_prefix_l1']:.5g}",
                    q1_prefix=f"{last_losses['q1_prefix_l1']:.5g}",
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {f"loss/{name}": value for name, value in last_losses.items()},
                        step=step,
                    )
            if optimizer is not None and args.save_interval > 0 and step % args.save_interval == 0:
                latest_checkpoint = save_checkpoint()
            if stop_signal["value"] is not None:
                break
    finally:
        bar.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)

    completed = step == optimizer_steps
    if optimizer is not None and (
        latest_checkpoint is None
        or latest_checkpoint.name != f"exact_q2_step_{step:06d}.pt"
    ):
        latest_checkpoint = save_checkpoint()
    scale_summary = raw_scales.summary()
    _write_json(output / "raw_loss_scales.json", scale_summary)
    if args.calibrate_losses:
        template = {
            "schema_version": "simvla_exact_q2_loss_contract_v1",
            "experiment_identifier": "simvla_r5_exact_q2_regeneration",
            "approval_status": "REQUIRES_RAW_SCALE_REVIEW",
            "candidate": args.candidate,
            "raw_loss_scales_path": str(output / "raw_loss_scales.json"),
            "weights": None,
            "intended_weighted_contributions": None,
            "constraint": "q2_prefix must be the largest intended contribution",
        }
        _write_json(output / "loss_contract_template.json", template)
    serialization = None
    if latest_checkpoint is not None:
        reloaded, _ = load_exact_q2_checkpoint(latest_checkpoint, device="cpu")
        serialization = {
            "passed": all(
                torch.equal(left.detach().cpu(), right.detach().cpu())
                for left, right in zip(adapter.parameters(), reloaded.parameters())
            ),
            "checkpoint": str(latest_checkpoint),
        }
        _write_json(output / "serialization_sanity.json", serialization)
        if not serialization["passed"]:
            raise RuntimeError("exact-q2 checkpoint serialization changed parameters")
    result = {
        "schema_version": "simvla_exact_q2_training_run_v1",
        "mode": "raw_loss_calibration" if args.calibrate_losses else "training",
        "candidate": args.candidate,
        "smoke": bool(args.smoke),
        "scientific_decision_allowed": not args.smoke and not args.calibrate_losses,
        "start_step": start_step,
        "steps": step,
        "max_steps": optimizer_steps,
        "completed": completed,
        "interrupted": stop_signal["value"] is not None and not completed,
        "last_losses": last_losses,
        "raw_loss_scales": scale_summary,
        "same_noise_sanity": same_noise_check,
        "serialization_sanity": serialization,
        "teacher_trainable_parameters": 0,
        "candidate_trainable_parameters": freeze_snapshot["candidate_trainable_parameters"],
        "final_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output / "run_summary.json", result)
    if wandb_run is not None:
        wandb_run.summary.update(result)
        wandb_run.finish(exit_code=130 if result["interrupted"] else 0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", required=True, choices=("recurrent_exact_q2", "direct_exact_q2"))
    parser.add_argument("--checkpoint", default="YuankaiLuo/SimVLA-LIBERO")
    parser.add_argument("--norm-stats", default=str(UPSTREAM / "norm_stats" / "libero_norm.json"))
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--calibrate-losses", action="store_true")
    parser.add_argument("--calibration-batches", type=int, default=100)
    parser.add_argument("--loss-contract", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=12_500)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--tqdm-mininterval", type=float, default=1.0)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.log_interval < 1:
        parser.error("--log-interval must be at least 1")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 130 if result["interrupted"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
