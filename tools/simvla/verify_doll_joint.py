"""One-batch joint backward/save/reload check. Never connects to robot hardware."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from architectures.simvla.adapters.real_world_training.dataset import RealSimVLADataset
from architectures.simvla.adapters.real_world_training.io_utils import atomic_write_json, sha256_file
from architectures.simvla.adapters.real_world_training.model_io import (
    load_exact_official_model, official_base_identity, apply_real_action_checkpoint,
    save_real_joint_checkpoint,
)
from architectures.simvla.adapters.real_world_training.train_joint_baseline import (
    enable_checkpointing, gradient_audit, inputs_for,
)


def verify(args):
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(20260904)
    device = torch.device(args.device)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = RealSimVLADataset(Path(args.dataset) / "manifest.json", split="train", training=False)
    norm = Path(args.dataset) / data.manifest["norm_stats"]["path"]
    batch = next(iter(DataLoader(data, batch_size=1, num_workers=0)))
    started = time.perf_counter()
    model, processor, initialization = load_exact_official_model(
        model_directory=args.checkpoint, processor_directory=args.processor,
        norm_stats=norm, device=device, freeze_vlm=False, freeze_action_transformer=False)
    enable_checkpointing(model)
    model.train()
    inputs = inputs_for(batch, processor, device)
    probe_parameters = {name: next(module.parameters()) for name, module in
                        (("vlm", model.vlm), ("action", model.transformer))}
    before = {name: value.detach().clone() for name, value in probe_parameters.items()}
    optimizer = torch.optim.AdamW([
        {"params": model.vlm.parameters(), "lr": 1e-5},
        {"params": model.transformer.parameters(), "lr": 1e-4},
    ], betas=(0.9, 0.95), weight_decay=0)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        loss = model(**inputs)["velocity_loss"]
    loss.backward()
    gradients = gradient_audit(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    changed = {name: not torch.equal(value, before[name]) for name, value in probe_parameters.items()}
    if not all(changed.values()):
        raise RuntimeError(f"one-step optimizer did not change both modules: {changed}")
    del optimizer, before
    gc.collect()
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        torch.manual_seed(17)
        predicted = model.generate_actions(**{k: v for k, v in inputs.items() if k != "action"}, steps=10)
    temporary = output / "verification_joint_checkpoint.pt"
    if temporary.exists():
        raise FileExistsError(f"previous verification artifact exists: {temporary}")
    try:
        save_real_joint_checkpoint(temporary, model=model,
                                  official_base=official_base_identity(args.checkpoint, args.processor),
                                  norm_stats_path=norm, dataset_identity_sha256=data.manifest["dataset_identity_sha256"],
                                  optimizer_step=1, training_config={"protocol": "bounded_verification"},
                                  validation={"robot_success_measured": False})
        expected = {name: value.detach().clone() for name, value in probe_parameters.items()}
        with torch.no_grad():
            for value in probe_parameters.values():
                value.zero_()
        loaded = apply_real_action_checkpoint(
            model, temporary, expected_base_sha256=initialization["base"]["model_weights_sha256"],
            expected_norm_sha256=sha256_file(norm),
            expected_dataset_identity_sha256=data.manifest["dataset_identity_sha256"])
        restored = {name: torch.equal(value, expected[name]) for name, value in probe_parameters.items()}
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            torch.manual_seed(17)
            reloaded = model.generate_actions(**{k: v for k, v in inputs.items() if k != "action"}, steps=10)
        error = float((predicted - reloaded).abs().max())
        if not all(restored.values()) or error != 0.0 or not torch.isfinite(reloaded).all():
            raise RuntimeError(f"joint reload changed policy actions: restored={restored}, max_abs={error}")
    finally:
        temporary.unlink(missing_ok=True)
    result = {"verdict": "JOINT_BACKWARD_RELOAD_PASS", "device": str(device),
              "elapsed_s": time.perf_counter() - started, "loss": float(loss.detach()),
              "gradients": gradients, "both_modules_updated": changed,
              "strict_joint_reload": loaded, "same_noise_reload_action_max_abs": error,
              "action_shape": list(reloaded.shape), "robot_connected": False,
              "training_checkpoint_produced": False}
    atomic_write_json(output / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "checkpoint", "processor", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    print(json.dumps(verify(parser.parse_args()), indent=2))

