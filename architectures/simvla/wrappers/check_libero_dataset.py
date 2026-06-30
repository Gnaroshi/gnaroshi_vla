#!/usr/bin/env python3
"""Validate the SimVLA LIBERO dataset links and metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py


SUBSETS = ("libero_10", "libero_goal", "libero_object", "libero_spatial", "libero_90")
EXPECTED_COUNTS = {
    "libero_10": 10,
    "libero_goal": 10,
    "libero_object": 10,
    "libero_spatial": 10,
    "libero_90": 90,
}
REQUIRED_DEMO_KEYS = (
    "actions",
    "obs/agentview_rgb",
    "obs/eye_in_hand_rgb",
    "obs/ee_pos",
    "obs/ee_ori",
    "obs/gripper_states",
)


def resolve_data_root(libero_root: Path) -> Path:
    if (libero_root / "datasets" / "libero_10").is_dir():
        return libero_root / "datasets"
    if (libero_root / "libero_10").is_dir():
        return libero_root
    raise FileNotFoundError(
        f"Could not find LIBERO subsets under {libero_root}; expected datasets/libero_10 or libero_10"
    )


def check_demo_keys(h5_path: Path) -> list[str]:
    errors: list[str] = []
    with h5py.File(h5_path, "r") as handle:
        if "data" not in handle:
            return [f"{h5_path}: missing top-level data group"]
        demo_keys = sorted(k for k in handle["data"].keys() if k.startswith("demo"))
        if not demo_keys:
            return [f"{h5_path}: no demo_* groups under data"]
        demo = handle["data"][demo_keys[0]]
        for key in REQUIRED_DEMO_KEYS:
            if key not in demo:
                errors.append(f"{h5_path}/{demo_keys[0]}: missing {key}")
    return errors


def resolve_meta_path(simvla_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return simvla_dir / path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    arch_dir = script_dir.parent
    default_simvla = arch_dir / "upstream"
    default_libero = Path("/home/mingyujung/shared/nvme1/mingyujung/datasets/robotics/LIBERO")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simvla-dir", type=Path, default=default_simvla)
    parser.add_argument("--libero-root", type=Path, default=default_libero)
    parser.add_argument("--metadata", type=str, default="./datasets/metas/libero_train.json")
    parser.add_argument("--norm-stats", type=str, default="./norm_stats/libero_norm.json")
    args = parser.parse_args()

    simvla_dir = args.simvla_dir.resolve()
    data_root = resolve_data_root(args.libero_root.expanduser().resolve())
    metas_dir = simvla_dir / "datasets" / "metas"
    errors: list[str] = []

    print(f"[CONTEXT] simvla_dir={simvla_dir}")
    print(f"[CONTEXT] libero_data_root={data_root}")
    print(f"[CONTEXT] metas_dir={metas_dir}")

    total_files = 0
    for subset in SUBSETS:
        target_dir = data_root / subset
        linked_dir = metas_dir / subset
        files = sorted(target_dir.glob("*.hdf5"))
        total_files += len(files)
        expected = EXPECTED_COUNTS[subset]

        if len(files) != expected:
            errors.append(f"{subset}: expected {expected} hdf5 files, found {len(files)}")

        if not linked_dir.exists():
            errors.append(f"{subset}: missing SimVLA metadata link/path {linked_dir}")
        elif linked_dir.resolve() != target_dir.resolve():
            errors.append(f"{subset}: {linked_dir} resolves to {linked_dir.resolve()}, expected {target_dir}")

        if files:
            errors.extend(check_demo_keys(files[0]))

        print(f"[CHECK] {subset}: {len(files)} hdf5 files, link={linked_dir}")

    metadata_path = resolve_meta_path(simvla_dir, args.metadata)
    norm_path = resolve_meta_path(simvla_dir, args.norm_stats)

    if not metadata_path.is_file():
        errors.append(f"missing metadata file: {metadata_path}")
    else:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        datalist = metadata.get("datalist", [])
        print(f"[CHECK] metadata datalist entries={len(datalist)} path={metadata_path}")
        if len(datalist) != total_files:
            errors.append(f"metadata datalist has {len(datalist)} entries, expected {total_files}")
        for item in datalist[:10]:
            raw_path = item["path"] if isinstance(item, dict) else item
            resolved = resolve_meta_path(simvla_dir, raw_path)
            if not resolved.exists():
                errors.append(f"metadata path does not exist from SimVLA cwd: {raw_path}")
                break

    if not norm_path.is_file():
        errors.append(f"missing norm stats file: {norm_path}")
    else:
        print(f"[CHECK] norm stats path={norm_path}")

    if errors:
        print("[FAIL] SimVLA LIBERO dataset check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("[OK] SimVLA LIBERO dataset links and metadata are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
