"""Shared CLI helpers for external OpenPI LatentLoop tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "architectures" / "openpi" / "upstream"
for path in (ROOT, UPSTREAM, UPSTREAM / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_CHECKPOINT = Path(
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/openpi/checkpoints/"
    "pi05_libero_lora_pytorch/pi05_base_lora_r16_b16_4gpu_seed42_30k/30000"
)
DEFAULT_NORM_STATS = Path(
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/openpi/assets/"
    "pi05_libero_lora_pytorch/physical-intelligence/libero/norm_stats.json"
)
DEFAULT_EVALUATION = Path(
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/openpi/eval/"
    "pi05_base_lora_r16_b16_4gpu_seed42_30k/seed7_official_50"
)
DEFAULT_RESULT_ROOT = Path(
    "/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/openpi/latentloop"
)


def refuse_nonempty_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite or reuse output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def require_run(enabled: bool, environment_variable: str) -> None:
    if not enabled or os.environ.get(environment_variable) != "1":
        raise RuntimeError(
            f"long/mutating execution is disabled; pass --run and set {environment_variable}=1"
        )


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_gate(path: str | Path, key: str) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get(key) is not True:
        raise RuntimeError(f"required gate {key}=true did not pass in {path}")
    return payload


def require_source_lock_v2(path: str | Path) -> dict[str, Any]:
    """Recompute and verify every source-lock-v2 field before side effects."""

    from source_lock_v2 import verify_lock

    return verify_lock(path)


def load_local_policy(checkpoint: str | Path, device: str, *, flow_steps: int = 10):
    from openpi.policies import policy_config
    from openpi.training import config

    train_config = config.get_config("pi05_libero_lora_pytorch")
    policy = policy_config.create_trained_policy(
        train_config,
        Path(checkpoint),
        pytorch_device=device,
        sample_kwargs={"num_steps": flow_steps},
    )
    return policy
