#!/usr/bin/env python3
"""Fail-closed identity gate for the s18 teacher33 canonical tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("status") != "LOCKED_CONTRACT":
        fail("canonical source contract is not locked")
    if socket.gethostname() != contract["expected_host"]:
        fail(f"host mismatch: {socket.gethostname()}")
    if root != Path(contract["target_source_tree"]):
        fail(f"target source tree mismatch: {root}")
    forbidden = Path(contract["forbidden_scientific_source"]).resolve()
    if root == forbidden or forbidden in root.parents:
        fail("s5 budget-locked tree is historical evidence only")

    lock_path = root / contract["source_lock_relative_path"]
    if not lock_path.is_file() or sha256_file(lock_path) != contract["source_lock_sha256"]:
        fail("canonical source-lock file is absent or mismatched")
    source_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if source_lock.get("status") != "PASS":
        fail("canonical source lock does not carry PASS status")
    if source_lock["git"]["branch"] != contract["git_branch"]:
        fail("source-lock branch mismatch")
    if source_lock["git"]["commit"] != contract["git_commit"]:
        fail("source-lock commit mismatch")

    current_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if current_commit != contract["git_commit"]:
        fail(f"git commit mismatch: {current_commit}")
    current_branch = subprocess.check_output(
        ["git", "-C", str(root), "branch", "--show-current"], text=True
    ).strip()
    if current_branch != contract["git_branch"]:
        fail(f"git branch mismatch: {current_branch}")
    tracked_diff = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--binary", "--no-ext-diff", "HEAD"]
    )
    diff_sha = hashlib.sha256(tracked_diff).hexdigest()
    if diff_sha != contract["tracked_dirty_diff_sha256"]:
        fail(f"dirty source fingerprint mismatch: {diff_sha}")

    mismatches: list[str] = []
    for relative, expected in contract["source_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(relative)
    if mismatches:
        fail(f"canonical source file mismatch: {mismatches}")

    libero_contract = contract["libero_source"]
    libero_root = Path(
        os.environ.get("CANONICAL_LIBERO_PATH", root / libero_contract["relative_path"])
    ).resolve()
    if not (libero_root / ".git").exists():
        fail(f"canonical LIBERO git source is missing: {libero_root}")
    libero_commit = subprocess.check_output(
        ["git", "-C", str(libero_root), "rev-parse", "HEAD"], text=True
    ).strip()
    libero_branch = subprocess.check_output(
        ["git", "-C", str(libero_root), "branch", "--show-current"], text=True
    ).strip()
    libero_diff = subprocess.check_output(
        ["git", "-C", str(libero_root), "diff", "--binary", "--no-ext-diff", "HEAD"]
    )
    if libero_commit != libero_contract["commit"] or libero_branch != libero_contract["branch"]:
        fail("canonical LIBERO revision mismatch")
    libero_diff_sha = hashlib.sha256(libero_diff).hexdigest()
    if libero_diff_sha != libero_contract["tracked_dirty_diff_sha256"]:
        fail("canonical LIBERO dirty fingerprint mismatch")
    for relative, expected in libero_contract["locked_files"].items():
        path = libero_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            fail(f"canonical LIBERO source file mismatch: {relative}")

    for name, item in contract["artifacts"].items():
        path = root / item["relative_path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            fail(f"artifact mismatch: {name}")
    episode_manifest = root / contract["canonical_episode_manifest"]["relative_path"]
    if not episode_manifest.is_file() or sha256_file(episode_manifest) != contract["canonical_episode_manifest"]["sha256"]:
        fail("canonical 200-episode manifest mismatch")
    for name, item in contract["canonical_reference"].items():
        path = root / item["relative_path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            fail(f"canonical reference mismatch: {name}")
    dataset_meta = Path(os.environ.get("CANONICAL_DATASET_META", contract["dataset_meta_path"]))
    if not dataset_meta.is_file() or sha256_file(dataset_meta) != contract["dataset_meta_sha256"]:
        fail("libero_10_converted meta_info.h5 mismatch")
    if args.output:
        if args.output.exists():
            fail(f"refusing to overwrite gate artifact: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "status": "SOURCE_GATE_PASS",
            "host": socket.gethostname(),
            "repo_root": str(root),
            "git_branch": current_branch,
            "git_commit": current_commit,
            "tracked_dirty_diff_sha256": diff_sha,
            "source_lock_sha256": sha256_file(lock_path),
            "libero_source": {
                "path": str(libero_root),
                "branch": libero_branch,
                "commit": libero_commit,
                "tracked_dirty_diff_sha256": libero_diff_sha,
            },
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("SOURCE_GATE_PASS")


if __name__ == "__main__":
    main()
