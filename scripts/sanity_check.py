#!/usr/bin/env python3
"""Lightweight gnaroshi_vla sanity checks.

This script intentionally avoids importing heavy Seer modules or launching GPU
jobs. It checks workspace structure and reports whether optional packages such as
Hydra and PyTorch are importable in the selected environment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def parse_key_values(argv: list[str]) -> dict[str, str]:
    values = {
        "architecture": "seer",
        "method": "lrnode",
        "env": "seer_libero",
        "node": "lrnode",
        "experiment": "seer_lrnode_debug",
    }
    for item in argv:
        if "=" in item:
            key, value = item.split("=", 1)
            values[key] = value
    return values


def require_path(path: Path, label: str) -> bool:
    if path.exists():
        print(f"[OK] {label}: {path}")
        return True
    print(f"[FAIL] {label} missing: {path}")
    return False


def architecture_upstream_path(root: Path, architecture: str) -> Path:
    return root / "architectures" / architecture / "upstream"


def main() -> int:
    args = parse_key_values(sys.argv[1:])
    root = Path(__file__).resolve().parents[1]
    print(f"[CONTEXT] root={root}")
    print(f"[CONTEXT] architecture={args['architecture']} method={args['method']} env={args['env']} node={args['node']}")
    print(f"[CONTEXT] python={sys.executable}")
    print(f"[CONTEXT] python_version={sys.version.split()[0]}")

    upstream = architecture_upstream_path(root, args["architecture"])
    checks = [
        require_path(upstream, "architecture upstream"),
        require_path(root / "architectures" / args["architecture"] / "adapters", "architecture adapters"),
        require_path(root / "architectures" / args["architecture"] / "wrappers", "architecture wrappers"),
        require_path(root / "configs" / "env" / f"{args['env']}.yaml", "env config"),
        require_path(root / "configs" / "method" / f"{args['method']}.yaml", "method config"),
        require_path(root / "results" / "README.md", "results schema"),
    ]

    for heavy_name in ["runs_lrnode_protocol_20260616", "archived_experiment_results_20260616", "checkpoints", "wandb"]:
        heavy_path = upstream / heavy_name
        if heavy_path.exists():
            print(f"[WARN] excluded heavy path is present in upstream copy: {heavy_path}")
        else:
            print(f"[OK] excluded heavy path absent: {heavy_name}")

    print(f"[OPTIONAL] hydra importable={importlib.util.find_spec('hydra') is not None}")
    print(f"[OPTIONAL] omegaconf importable={importlib.util.find_spec('omegaconf') is not None}")
    print(f"[OPTIONAL] torch importable={importlib.util.find_spec('torch') is not None}")

    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
