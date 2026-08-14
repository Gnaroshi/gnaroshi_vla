"""Seer baseline and LatentLoop deployment GUI on the preserved real GUI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tkinter as tk
from pathlib import Path

from architectures.seer.adapters.latentloop_real_deploy.controller import (
    LatentLoopSeerController,
)
from architectures.seer.adapters.latentloop_real_deploy.hardware import (
    DeployConfig,
    UR5eDeployEnv,
    _camera_serials_file,
    _save_camera_serial_cache,
    load_legacy_gui,
)


legacy_gui = load_legacy_gui()


class TimedUR5eDeployEnv(UR5eDeployEnv):
    """Record actual robot-command cadence without changing preserved sources."""

    def __init__(self, cfg, command_callback):
        super().__init__(cfg)
        self._command_callback = command_callback

    def step(self, target_pose, target_gripper):
        result = super().step(target_pose, target_gripper)
        self._command_callback(time.perf_counter())
        return result


def parse_latentloop_gui_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--camera-c",
        choices=("ask", "yes", "no"),
        default="no",
    )
    parser.add_argument("--gui-refresh-ms", type=int, default=150)
    parser.add_argument(
        "--deployment-method", choices=("baseline", "latentloop"), required=True
    )
    parser.add_argument("--deployment-control-freq", type=float, required=True)
    parser.add_argument("--latentloop-adapter-checkpoint")
    parser.add_argument("--latentloop-artifact-manifest", required=True)
    parser.add_argument("--latentloop-teacher-id", type=int, required=True)
    parser.add_argument("--latentloop-adapter-id", type=int)
    parser.add_argument("--latentloop-deployment-profile", required=True)
    parser.add_argument("--latentloop-preflight-only", action="store_true")
    gui_args, seer_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *seer_args]
    return gui_args


class LatentLoopDeployGuiApp(legacy_gui.DeployGuiApp):
    def __init__(self, root, cfg, controller, env, gui_args):
        cfg.results_dir = os.path.join(cfg.results_dir, controller.deployment_profile)
        super().__init__(root, cfg, controller, env, gui_args)
        controller.attach_session_dir(self.session_dir)
        self.write_session_files()
        self.update_metrics()

    def write_session_files(self):
        super().write_session_files()
        manifest_path = Path(self.session_manifest_file)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        metadata = self.controller.deployment_metadata()
        manifest["seer_deployment"] = metadata
        if metadata["deployment_method"] == "latentloop":
            manifest["latentloop"] = metadata
        manifest["runtime_summary_file_pattern"] = "deployment_runtime_rollout_*.json"
        manifest["policy_step_log_pattern"] = "policy_steps_rollout_*.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        self.controller.write_runtime_summary()

    def update_metrics(self):
        super().update_metrics()
        metadata = self.controller.deployment_metadata()
        self.metrics_var.set(
            self.metrics_var.get()
            + "\n\n"
            + "Deployment profile\n"
            + metadata["deployment_profile"]
            + "\n\n"
            + f"Method / target control rate\n{metadata['method']} / "
            + f"{metadata['target_control_hz']:g} Hz"
            + "\n\n"
            + f"Teacher / adapter / K\n{metadata['teacher_id']} / "
            + f"{metadata['adapter_id'] if metadata['adapter_id'] is not None else 'none'} / "
            + f"{metadata['query_interval']}"
            + "\n\n"
            + f"Adapter checkpoint\n{metadata['adapter_checkpoint'] or 'not loaded'}"
        )

    def save_results(self):
        super().save_results()
        result_count = len(self.deploy_results)
        previous_count = getattr(self, "_latentloop_saved_result_count", 0)
        if result_count > previous_count:
            self.controller.mark_rollout_complete()
        else:
            self.controller.write_runtime_summary()
        self._latentloop_saved_result_count = result_count

    def on_close(self):
        self.controller.write_runtime_summary()
        super().on_close()


def main():
    gui_args = parse_latentloop_gui_args()
    controller = LatentLoopSeerController(
        deployment_method=gui_args.deployment_method,
        deployment_control_freq=gui_args.deployment_control_freq,
        adapter_checkpoint=gui_args.latentloop_adapter_checkpoint,
        artifact_manifest=gui_args.latentloop_artifact_manifest,
        teacher_id=gui_args.latentloop_teacher_id,
        adapter_id=gui_args.latentloop_adapter_id,
        deployment_profile=gui_args.latentloop_deployment_profile,
    )
    if gui_args.latentloop_preflight_only:
        synthetic = controller.run_synthetic_preflight(
            os.environ["SEER_LANGUAGE_INSTRUCTION"]
        )
        print(
            "[Seer deploy preflight][OK] "
            + json.dumps(
                {
                    "deployment": controller.deployment_metadata(),
                    "synthetic_k_cycle": synthetic,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    cfg = DeployConfig()
    if abs(float(cfg.control_freq) - controller.target_control_hz) > 1e-9:
        raise ValueError(
            "SEER_CONTROL_FREQ and --deployment-control-freq disagree: "
            f"{cfg.control_freq} vs {controller.target_control_hz}"
        )
    legacy_gui.configure_camera_c(cfg, gui_args.camera_c)
    env = TimedUR5eDeployEnv(cfg, controller.record_control_command)
    _save_camera_serial_cache(
        _camera_serials_file(cfg),
        env.deploy_camera_serials,
        [cfg.exterior_camera_name, cfg.wrist_camera_name],
    )
    root = tk.Tk()
    LatentLoopDeployGuiApp(root, cfg, controller, env, gui_args)
    root.mainloop()


if __name__ == "__main__":
    main()
