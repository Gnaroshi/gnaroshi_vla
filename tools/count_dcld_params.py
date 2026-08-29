"""Count SimVLA DCLD parameter variants without running training or eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.dcld.modules import DCLDCore, DeltaObservation  # noqa: E402

SEER_OURS_PARAMS = 470_146


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_dummy_obs(device: torch.device) -> tuple[torch.Tensor, DeltaObservation]:
    latent = torch.randn(1, 122, 960, device=device)
    obs = DeltaObservation(
        key_images=torch.rand(1, 2, 128, 128, 3, device=device),
        cur_images=torch.rand(1, 2, 128, 128, 3, device=device),
        key_proprio=torch.randn(1, 8, device=device),
        cur_proprio=torch.randn(1, 8, device=device),
    )
    return latent, obs


def count_variant(name: str, kwargs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    core = DCLDCore(**kwargs).to(device)
    latent, obs = make_dummy_obs(device)
    with torch.no_grad():
        update = core.update_latent(latent, obs)
    total = 0
    trainable = 0
    groups: dict[str, int] = {
        "delta_encoder": 0,
        "dynamics_vector_field": 0,
        "dynamics_gate": 0,
        "dynamics_projection": 0,
        "dynamics_other": 0,
    }
    parameters = []
    for param_name, param in core.named_parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
        if param_name.startswith("delta_encoder."):
            group = "delta_encoder"
        elif param_name.startswith("dynamics.vector_field."):
            group = "dynamics_vector_field"
        elif param_name.startswith("dynamics.gate."):
            group = "dynamics_gate"
        elif param_name.startswith(("dynamics.down_proj.", "dynamics.up_proj.")):
            group = "dynamics_projection"
        else:
            group = "dynamics_other"
        groups[group] += count
        parameters.append({"name": param_name, "shape": list(param.shape), "numel": count})
    return {
        "variant": name,
        "config": kwargs,
        "condition_shape": list(latent.shape),
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "param_groups": groups,
        "ratio_to_seer_470146": trainable / SEER_OURS_PARAMS,
        "forward_smoke": {
            "latent_shape": list(update.latent.shape),
            "same_shape": list(update.latent.shape) == list(latent.shape),
            "no_nan": bool(torch.isfinite(update.latent).all().item()),
            "gate_mean": float(update.dynamics.gate.detach().float().mean().item()),
            "gate_min": float(update.dynamics.gate.detach().float().min().item()),
            "gate_max": float(update.dynamics.gate.detach().float().max().item()),
            "update_norm": float(update.dynamics.update.detach().flatten(start_dim=1).norm(dim=-1).mean().item()),
        },
        "parameters": parameters,
    }


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# SimVLA DCLD Parameter Count Report",
        "",
        f"Seer Ours reference params: `{SEER_OURS_PARAMS}`",
        "",
        "| Variant | Dynamics | Trainable Params | Ratio To Seer | Delta Encoder | Vector Field | Gate | Projection | Other |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["variants"]:
        groups = item["param_groups"]
        lines.append(
            "| {variant} | {dynamics} | `{trainable}` | `{ratio:.4f}` | `{delta}` | `{vf}` | `{gate}` | `{proj}` | `{other}` |".format(
                variant=item["variant"],
                dynamics=item["config"]["dynamics_type"],
                trainable=item["trainable_params"],
                ratio=item["ratio_to_seer_470146"],
                delta=groups["delta_encoder"],
                vf=groups["dynamics_vector_field"],
                gate=groups["dynamics_gate"],
                proj=groups["dynamics_projection"],
                other=groups["dynamics_other"],
            )
        )
    pm = next(item for item in result["variants"] if item["variant"] == "simvla_dcld_pm047m")
    lines.extend(
        [
            "",
            "## Pass Condition",
            "",
            f"- pm047m trainable params: `{pm['trainable_params']}`",
            f"- target range: `350000` to `600000`",
            f"- passed: `{350000 <= pm['trainable_params'] <= 600000}`",
            "",
            "## Notes",
            "",
            "- Counts include only DCLD parameters, not the frozen SimVLA teacher.",
            "- The low-rank variant uses fixed Euler dynamics with no adaptive solver dependency.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    variants = [
        (
            "simvla_dcld_adapted_5m",
            {
                "latent_dim": 960,
                "delta_dim": 512,
                "hidden_dim": 1024,
                "dynamics_type": "dense",
                "rank_dim": 64,
                "gate_mode": "dense",
                "gate_bias": -4.0,
            },
        ),
        (
            "simvla_dcld_pm047m",
            {
                "latent_dim": 960,
                "delta_dim": 128,
                "hidden_dim": 128,
                "dynamics_type": "low_rank",
                "rank_dim": 64,
                "gate_mode": "scalar",
                "gate_bias": -4.0,
            },
        ),
    ]
    result = {
        "seer_ours_reference_params": SEER_OURS_PARAMS,
        "variants": [count_variant(name, kwargs, device) for name, kwargs in variants],
    }
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "param_matched_count.json", result)
        (args.output_dir / "param_matched_count_report.md").write_text(markdown_report(result), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
