"""CPU-testable utilities for LR-NODE K=4 mechanism diagnostics."""

from __future__ import annotations

import csv
import json
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


TRACE_SCHEMA_VERSION = 1
ARM_SLICE = slice(0, 6)
TRANSLATION_SLICE = slice(0, 3)
ROTATION_SLICE = slice(3, 6)
GRIPPER_INDEX = 6


def temporal_ensemble_probability(
    action_sequence: torch.Tensor,
    timestep: int,
    buffer: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    """Apply Seer's legacy temporal ensemble to one branch-local buffer."""
    if action_sequence.ndim != 3 or action_sequence.shape[0] != 1:
        raise ValueError(f"Expected [1, H, A], got {tuple(action_sequence.shape)}")
    horizon = action_sequence.shape[1]
    buffer[timestep:timestep + 1, timestep:timestep + horizon] = action_sequence
    candidates = buffer[:, timestep]
    # Preserve the original Seer evaluation protocol exactly. Its action
    # buffer treats a row as populated only when every action axis is nonzero.
    candidates = candidates[torch.all(candidates != 0, dim=1)]
    weights = np.exp(-float(temperature) * np.arange(len(candidates)))
    weights /= weights.sum()
    # The original implementation keeps NumPy's float64 weights. This makes
    # the temporally ensembled environment action float64 as well.
    weight_tensor = torch.from_numpy(weights).to(candidates.device).unsqueeze(1)
    return (candidates * weight_tensor).sum(dim=0, keepdim=True), int(len(candidates))


def capture_rng_state(include_cuda: bool = True) -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": None,
    }
    if include_cuda and torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def preserve_rng_state(include_cuda: bool = True):
    state = capture_rng_state(include_cuda=include_cuda)
    try:
        yield
    finally:
        restore_rng_state(state)


def rng_states_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left["python"] != right["python"]:
        return False
    left_np, right_np = left["numpy"], right["numpy"]
    if left_np[0] != right_np[0] or not np.array_equal(left_np[1], right_np[1]):
        return False
    if left_np[2:] != right_np[2:]:
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    left_cuda, right_cuda = left.get("torch_cuda"), right.get("torch_cuda")
    if (left_cuda is None) != (right_cuda is None):
        return False
    if left_cuda is not None:
        return len(left_cuda) == len(right_cuda) and all(
            torch.equal(a, b) for a, b in zip(left_cuda, right_cuda)
        )
    return True


def classify_transition(previous_was_full: Optional[bool], current_is_full: bool) -> str:
    if previous_was_full is None:
        return "episode_start_full" if current_is_full else "episode_start_skip"
    if previous_was_full and current_is_full:
        return "full_to_full"
    if previous_was_full and not current_is_full:
        return "full_to_skip"
    if not previous_was_full and current_is_full:
        return "skip_to_full"
    return "skip_to_skip"


def counterfactual_requires_skip_shadow(
    mode: str,
    latent_fusion_mode: str,
) -> bool:
    """Whether the executed skip action depends on a same-step full forward."""
    if mode in {"standard", "lr_arm_lr_gripper"}:
        return False
    if mode in {
        "full_arm_full_gripper",
        "lr_arm_full_gripper",
        "full_arm_lr_gripper",
        "matched_random",
    }:
        return True
    if mode == "latent_fusion":
        return latent_fusion_mode == "every_step"
    raise ValueError(f"Unsupported counterfactual mode: {mode}")


def mix_action_tokens(
    lr_action: torch.Tensor,
    full_action: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, str, str]:
    if lr_action.shape != full_action.shape or lr_action.shape[-1] != 7:
        raise ValueError(
            f"Expected matching [..., 7] action tensors, got "
            f"{tuple(lr_action.shape)} and {tuple(full_action.shape)}"
        )
    if mode == "lr_arm_lr_gripper":
        return lr_action, "lr", "lr"
    if mode == "full_arm_full_gripper":
        return full_action, "full", "full"
    if mode == "lr_arm_full_gripper":
        return torch.cat([lr_action[..., :6], full_action[..., 6:7]], dim=-1), "lr", "full"
    if mode == "full_arm_lr_gripper":
        return torch.cat([full_action[..., :6], lr_action[..., 6:7]], dim=-1), "full", "lr"
    raise ValueError(f"Unsupported arm/gripper counterfactual mode: {mode}")


def fuse_latents(
    z_lr: torch.Tensor,
    z_full: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if z_lr.shape != z_full.shape:
        raise ValueError(f"Latent shape mismatch: {tuple(z_lr.shape)} vs {tuple(z_full.shape)}")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return (1.0 - float(alpha)) * z_lr + float(alpha) * z_full


def matched_random_latent(
    z_lr: torch.Tensor,
    z_full: torch.Tensor,
    seed: int,
    norm_mode: str = "per_token",
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if z_lr.shape != z_full.shape:
        raise ValueError(f"Latent shape mismatch: {tuple(z_lr.shape)} vs {tuple(z_full.shape)}")
    if norm_mode not in {"per_token", "global"}:
        raise ValueError(f"norm_mode must be per_token or global, got {norm_mode}")

    learned = z_lr - z_full
    generator = torch.Generator(device=z_full.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(
        z_full.shape,
        dtype=z_full.dtype,
        device=z_full.device,
        generator=generator,
    )
    if norm_mode == "per_token":
        learned_norm = torch.linalg.vector_norm(learned, dim=-1, keepdim=True)
        noise_norm = torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
    else:
        dims = tuple(range(1, learned.dim()))
        learned_norm = torch.linalg.vector_norm(learned, dim=dims, keepdim=True)
        noise_norm = torch.linalg.vector_norm(noise, dim=dims, keepdim=True)
    random_delta = learned_norm * noise / noise_norm.clamp_min(eps)
    return z_full + random_delta, learned, random_delta


def action_second_differences(actions: np.ndarray) -> Dict[str, np.ndarray]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions [T, 7], got {actions.shape}")
    second = np.zeros_like(actions)
    if len(actions) >= 3:
        second[2:] = actions[2:] - 2.0 * actions[1:-1] + actions[:-2]
    return {
        "translation": np.linalg.norm(second[:, TRANSLATION_SLICE], axis=-1),
        "rotation": np.linalg.norm(second[:, ROTATION_SLICE], axis=-1),
        "arm": np.linalg.norm(second[:, ARM_SLICE], axis=-1),
        "gripper_discrete_second_difference": second[:, GRIPPER_INDEX],
    }


def gripper_event_rows(
    actions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    logits: Optional[np.ndarray] = None,
    contacts: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> list[Dict[str, Any]]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions [T, 7], got {actions.shape}")
    probs = None if probabilities is None else np.asarray(probabilities, dtype=np.float64).reshape(-1)
    logit_values = None if logits is None else np.asarray(logits, dtype=np.float64).reshape(-1)
    contact_values = None if contacts is None else np.asarray(contacts).reshape(-1)
    if probs is not None and len(probs) != len(actions):
        raise ValueError("probabilities length does not match actions")
    if logit_values is not None and len(logit_values) != len(actions):
        raise ValueError("logits length does not match actions")

    states = actions[:, GRIPPER_INDEX] > 0.0
    switch_indices = np.flatnonzero(states[1:] != states[:-1]) + 1
    rows = []
    for index in switch_indices:
        is_close = bool(states[index])  # robosuite convention: +1 closes, -1 opens.
        reverse_distances = [
            int(other - index)
            for other in switch_indices
            if other > index and states[other] != states[index]
        ]
        next_reverse = min(reverse_distances) if reverse_distances else None
        probability = float(probs[index]) if probs is not None else None
        logit = float(logit_values[index]) if logit_values is not None else None
        rows.append(
            {
                "step": int(index),
                "event": "close" if is_close else "open",
                "reverse_within_1": int(next_reverse is not None and next_reverse <= 1),
                "reverse_within_2": int(next_reverse is not None and next_reverse <= 2),
                "reverse_within_5": int(next_reverse is not None and next_reverse <= 5),
                "next_reverse_steps": "" if next_reverse is None else int(next_reverse),
                "gripper_probability": "" if probability is None else probability,
                "gripper_logit": "" if logit is None else logit,
                "probability_margin": "" if probability is None else abs(probability - threshold),
                "contact": (
                    ""
                    if contact_values is None or index >= len(contact_values)
                    else int(bool(contact_values[index]))
                ),
            }
        )
    return rows


def gripper_summary(actions: np.ndarray) -> Dict[str, float]:
    actions = np.asarray(actions, dtype=np.float64)
    rows = gripper_event_rows(actions)
    steps = max(1, len(actions))
    close_count = sum(row["event"] == "close" for row in rows)
    open_count = sum(row["event"] == "open" for row in rows)
    reverse_1 = float(sum(row["reverse_within_1"] for row in rows))
    reverse_2 = float(sum(row["reverse_within_2"] for row in rows))
    reverse_5 = float(sum(row["reverse_within_5"] for row in rows))
    return {
        "gripper_switch_count": float(len(rows)),
        "gripper_switches_per_100_steps": 100.0 * len(rows) / steps,
        "gripper_close_count": float(close_count),
        "gripper_open_count": float(open_count),
        "gripper_reverse_within_1_count": reverse_1,
        "gripper_reverse_within_2_count": reverse_2,
        "gripper_reverse_within_5_count": reverse_5,
        "gripper_reverse_within_1_per_100_steps": 100.0 * reverse_1 / steps,
        "gripper_reverse_within_2_per_100_steps": 100.0 * reverse_2 / steps,
        "gripper_reverse_within_5_per_100_steps": 100.0 * reverse_5 / steps,
    }


def _small_signal(value: Any) -> Optional[Any]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0 or array.size > 16 or array.dtype.kind not in "biuf":
        return None
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return [float(item) for item in array.reshape(-1)]


def extract_simulator_signals(
    observation: Optional[Mapping[str, Any]],
    info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    signals: Dict[str, Any] = {}
    for source_name, source in (("obs", observation), ("info", info)):
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            lowered = str(key).lower()
            if not any(token in lowered for token in ("contact", "grasp", "height", "success")):
                continue
            encoded = _small_signal(value)
            if encoded is not None:
                signals[f"{source_name}_{key}"] = encoded
    return signals


def deterministic_step_seed(base_seed: int, task_id: int, episode_id: int, step: int) -> int:
    value = (
        int(base_seed) * 1_000_003
        + int(task_id) * 10_007
        + int(episode_id) * 1_009
        + int(step)
    )
    return int(value % (2**63 - 1))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def save_trace_episode(
    output_dir: Path,
    episode_key: str,
    scalar_rows: Sequence[Mapping[str, Any]],
    tensor_rows: Sequence[Mapping[str, Any]],
    episode_metadata: Mapping[str, Any],
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / episode_key

    scalar_keys = sorted(set().union(*(row.keys() for row in scalar_rows))) if scalar_rows else []
    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in scalar_rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in scalar_keys})

    arrays: Dict[str, np.ndarray] = {}
    tensor_keys = sorted(set().union(*(row.keys() for row in tensor_rows))) if tensor_rows else []
    for key in tensor_keys:
        values = []
        template = None
        for row in tensor_rows:
            value = row.get(key)
            if value is not None:
                array = np.asarray(
                    value.detach().cpu().float().numpy() if isinstance(value, torch.Tensor) else value,
                    dtype=np.float32,
                )
                template = array if template is None else template
                values.append(array)
            else:
                values.append(None)
        if template is None:
            continue
        filled = [
            np.full(template.shape, np.nan, dtype=np.float32) if value is None else value
            for value in values
        ]
        arrays[key] = np.stack(filled)
        arrays[f"{key}__present"] = np.asarray([value is not None for value in values], dtype=np.uint8)

    npz_path = stem.with_suffix(".npz")
    np.savez_compressed(npz_path, **arrays)
    metadata_path = stem.with_suffix(".json")
    metadata_payload = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "episode_key": episode_key,
        "num_steps": len(scalar_rows),
        "scalar_csv": csv_path.name,
        "tensor_npz": npz_path.name,
        "tensor_keys": tensor_keys,
        "episode": _json_safe(dict(episode_metadata)),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2) + "\n", encoding="utf-8")
    return {
        "csv": str(csv_path),
        "npz": str(npz_path),
        "json": str(metadata_path),
    }


def load_trace_shard(metadata_path: Path) -> tuple[Dict[str, Any], list[Dict[str, str]], Dict[str, np.ndarray]]:
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with (metadata_path.parent / metadata["scalar_csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with np.load(metadata_path.parent / metadata["tensor_npz"], allow_pickle=False) as archive:
        tensors = {key: archive[key] for key in archive.files}
    return metadata, rows, tensors


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
