"""Post-training model checks; never authorize or initialize live hardware."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .contracts import load_deployment_contract, hardware_configuration_issues, sha256_file
from .environment import inspect_environment

METHODS = ("baseline", "condition_loop", "latentloop")


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(*, manifest: str, output: str, device: str, require_gui: bool) -> dict[str, Any]:
    directory = Path(output).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "verdict": "DEPLOYMENT_PREPARATION_FAILED", "stage": "environment",
        "robot_commands_issued": 0, "live_authorized_by_preparation": False,
        "task_success_verified": False, "methods": {},
    }
    try:
        environment = inspect_environment(
            require_cuda=device.startswith("cuda"), require_gui=require_gui
        )
        _write(directory / "environment.json", environment)
        if environment["verdict"] != "REAL_ENVIRONMENT_PASS":
            raise RuntimeError("environment failed: " + "; ".join(environment["failures"]))
        report["stage"] = "artifact_lineage"
        contract = load_deployment_contract(manifest, verify_artifacts=True)
        report.update({
            "manifest": str(contract.path), "manifest_sha256": sha256_file(contract.path),
            "deployment_id": contract.deployment_id,
            "runtime_source_identity_sha256": contract.payload["runtime_source_identity_sha256"],
            "hardware_configuration_issues": hardware_configuration_issues(contract),
            "gui_checked": require_gui,
        })
        for method in METHODS:
            report["stage"] = method
            destination = directory / method
            command = [sys.executable, "-m", "architectures.simvla.adapters.latentloop_real_deploy.cli",
                       "artifact-preflight", "--manifest", str(contract.path),
                       "--method", method, "--device", device, "--output", str(destination)]
            print(f"[SimVLA prepare] {method}: model loading and H10/R5 checks", flush=True)
            with (directory / f"{method}.log").open("w", encoding="utf-8") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                           env={**os.environ, "HF_HUB_OFFLINE": "1",
                                                "TRANSFORMERS_OFFLINE": "1"}, timeout=600)
            if completed.returncode != 0:
                raise RuntimeError(f"{method} failed; inspect {directory / (method + '.log')}")
            result = json.loads((destination / "artifact_preflight.json").read_text())
            if (result.get("verdict") != "ARTIFACT_PREFLIGHT_PASS"
                    or result.get("actions_finite") is not True
                    or result.get("robot_command_issued") is not False):
                raise RuntimeError(f"{method} completion report is invalid")
            report["methods"][method] = {"verdict": result["verdict"],
                "report": str(destination / "artifact_preflight.json"),
                "report_sha256": sha256_file(destination / "artifact_preflight.json")}
        report.update({"stage": "complete", "verdict": "MODELS_READY_FOR_SITE_REVIEW",
                       "next": "Review hardware configuration, run read-only profiles, then supervised baseline canary."})
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        _write(directory / "preparation_summary.json", report)
    return report
