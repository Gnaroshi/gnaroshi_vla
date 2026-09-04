"""V2 Seer/LatentLoop live GUI and read-only real-hardware profiler."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from architectures.seer.adapters.latentloop_real_deploy.controller_v2 import (
    LatentLoopSeerControllerV2,
)
from architectures.seer.adapters.latentloop_real_deploy.hardware import (
    DeployConfig,
    UR5eDeployEnv,
    _camera_serials_file,
    _save_camera_serial_cache,
    load_legacy_gui,
)
from architectures.seer.adapters.latentloop_real_deploy.runtime_v2 import (
    ReadOnlyDeployEnvV2,
    run_read_only_profile,
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
    parser.add_argument(
        "--rollout-policy",
        choices=("full", "latentloop", "hold_action", "hold_latent"),
        required=True,
    )
    parser.add_argument("--latentloop-adapter-checkpoint")
    parser.add_argument("--latentloop-artifact-manifest", required=True)
    parser.add_argument("--latentloop-teacher-id", type=int, required=True)
    parser.add_argument("--latentloop-adapter-id", type=int)
    parser.add_argument("--latentloop-deployment-profile", required=True)
    parser.add_argument("--latentloop-preflight-only", action="store_true")
    parser.add_argument(
        "--v2-camera-mode",
        choices=("sync", "async_latest"),
        default="sync",
    )
    parser.add_argument("--v2-profile-steps", type=int, default=300)
    parser.add_argument("--v2-profile-warmup-steps", type=int, default=8)
    parser.add_argument("--v2-profile-output-dir")
    parser.add_argument(
        "--v2-optimized-fast-path",
        type=int,
        choices=(0, 1),
        default=1,
    )
    gui_args, seer_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *seer_args]
    return gui_args


def execution_mode_from_environment() -> str:
    value = os.environ.get("SEER_EXECUTION_MODE", "").strip().lower()
    if value not in {"live", "read_only_profile"}:
        raise ValueError(
            "SEER_EXECUTION_MODE must be configured in deploy_ll_gui_v2.sh as "
            f"'live' or 'read_only_profile', got {value!r}"
        )
    return value


class LatentLoopDeployGuiAppV2(legacy_gui.DeployGuiApp):
    def __init__(self, root, cfg, controller, env, gui_args):
        self.controller = controller
        self.runtime_control_freq_var = tk.StringVar(
            master=root, value=f"{controller.control_freq:g}"
        )
        self.runtime_query_interval_var = tk.StringVar(
            master=root, value=str(controller.query_interval)
        )
        self.runtime_rollout_policy_var = tk.StringVar(
            master=root, value=controller.rollout_policy
        )
        self.runtime_settings_status_var = tk.StringVar(
            master=root,
            value=self._runtime_settings_label(
                controller.control_freq,
                controller.query_interval,
                controller.rollout_policy,
            ),
        )
        self.runtime_settings_by_rollout = {}
        cfg.results_dir = os.path.join(cfg.results_dir, controller.deployment_profile)
        super().__init__(root, cfg, controller, env, gui_args)
        controller.attach_session_dir(self.session_dir)
        self.write_session_files()
        self.update_metrics()

    def _runtime_settings_label(self, control_freq, query_interval, rollout_policy):
        return (
            f"Current: {float(control_freq):g} Hz; "
            f"{rollout_policy}; K={int(query_interval)}"
        )

    def _build_controls(self, parent):
        settings = self._panel(parent, "Runtime settings")
        settings.pack(fill=tk.X, pady=(0, 12))

        rate_row = tk.Frame(settings, bg="#ffffff")
        rate_row.pack(fill=tk.X, pady=(2, 7))
        tk.Label(
            rate_row,
            text="Control frequency (Hz)",
            bg="#ffffff",
            fg="#18212f",
            font=self.font_body,
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.runtime_control_freq_input = tk.Spinbox(
            rate_row,
            from_=1,
            to=60,
            increment=1,
            textvariable=self.runtime_control_freq_var,
            width=8,
            font=self.font_body,
        )
        self.runtime_control_freq_input.pack(side=tk.RIGHT)

        policy_row = tk.Frame(settings, bg="#ffffff")
        policy_row.pack(fill=tk.X, pady=(2, 7))
        tk.Label(
            policy_row,
            text="Rollout policy",
            bg="#ffffff",
            fg="#18212f",
            font=self.font_body,
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        policy_values = (
            ("full", "hold_action", "hold_latent")
            if self.controller.deployment_method == "baseline"
            else ("latentloop",)
        )
        self.runtime_rollout_policy_input = ttk.Combobox(
            policy_row,
            textvariable=self.runtime_rollout_policy_var,
            values=policy_values,
            width=13,
            state=(
                "readonly"
                if self.controller.deployment_method == "baseline"
                else "disabled"
            ),
        )
        self.runtime_rollout_policy_input.pack(side=tk.RIGHT)
        self.runtime_rollout_policy_input.bind(
            "<<ComboboxSelected>>", self._on_rollout_policy_changed
        )

        query_row = tk.Frame(settings, bg="#ffffff")
        query_row.pack(fill=tk.X, pady=(2, 7))
        tk.Label(
            query_row,
            text="Full-refresh interval (K)",
            bg="#ffffff",
            fg="#18212f",
            font=self.font_body,
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.runtime_query_interval_input = tk.Spinbox(
            query_row,
            from_=1,
            to=100,
            increment=1,
            textvariable=self.runtime_query_interval_var,
            width=8,
            font=self.font_body,
            disabledforeground="#64748b",
        )
        self.runtime_query_interval_input.pack(side=tk.RIGHT)
        self._sync_query_interval_state()

        tk.Button(
            settings,
            text="Apply for next rollout",
            command=self.apply_runtime_settings,
            font=self.font_body,
        ).pack(fill=tk.X, pady=(3, 5))
        tk.Label(
            settings,
            textvariable=self.runtime_settings_status_var,
            bg="#ffffff",
            fg="#475569",
            font=self.font_small,
            justify=tk.LEFT,
            anchor="w",
            wraplength=285,
        ).pack(fill=tk.X)
        super()._build_controls(parent)

    def _on_rollout_policy_changed(self, _event=None):
        self._sync_query_interval_state()

    def _sync_query_interval_state(self):
        if self.runtime_rollout_policy_var.get() == "full":
            self.runtime_query_interval_var.set("1")
            self.runtime_query_interval_input.configure(state=tk.DISABLED)
        else:
            self.runtime_query_interval_input.configure(state=tk.NORMAL)

    def _rollout_is_active(self):
        return self.rollout_thread is not None and self.rollout_thread.is_alive()

    def apply_runtime_settings(self, show_error=True):
        if self._rollout_is_active():
            message = "Runtime settings cannot change while a rollout is active."
            self.set_status(message)
            if show_error:
                messagebox.showwarning("Rollout running", message)
            return False

        requested_policy = self.runtime_rollout_policy_var.get()
        requested_interval = (
            1 if requested_policy == "full" else self.runtime_query_interval_var.get()
        )
        try:
            control_freq, query_interval, rollout_policy = (
                self.controller.configure_runtime_settings(
                    control_freq=self.runtime_control_freq_var.get(),
                    query_interval=requested_interval,
                    rollout_policy=requested_policy,
                )
            )
        except ValueError as exc:
            self.set_status(str(exc))
            if show_error:
                messagebox.showerror("Invalid runtime settings", str(exc))
            return False

        # The preserved 3DFlow GUI loop reads this legacy compatibility field.
        self.cfg.control_freq = control_freq
        self.runtime_control_freq_var.set(f"{control_freq:g}")
        self.runtime_query_interval_var.set(str(query_interval))
        self.runtime_rollout_policy_var.set(rollout_policy)
        self._sync_query_interval_state()
        self.runtime_settings_status_var.set(
            self._runtime_settings_label(control_freq, query_interval, rollout_policy)
        )
        self.write_session_files()
        self.update_metrics()
        self.set_status(self.runtime_settings_status_var.get())
        return True

    def start_rollout(self):
        if self._rollout_is_active():
            super().start_rollout()
            return
        if not self.apply_runtime_settings(show_error=True):
            return
        rollout_index = len(self.deploy_results) + 1
        self.runtime_settings_by_rollout[str(rollout_index)] = {
            "control_freq": self.controller.control_freq,
            "query_interval": self.controller.query_interval,
            "rollout_policy": self.controller.rollout_policy,
        }
        super().start_rollout()

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
            + f"Method / control frequency\n{metadata['method']} / "
            + f"{metadata['control_freq']:g} Hz"
            + "\n\n"
            + f"Teacher / adapter / policy / K\n{metadata['teacher_id']} / "
            + f"{metadata['adapter_id'] if metadata['adapter_id'] is not None else 'none'} / "
            + f"{metadata['rollout_policy']} / {metadata['query_interval']}"
            + "\n\n"
            + f"Adapter checkpoint\n{metadata['adapter_checkpoint'] or 'not loaded'}"
        )

    def save_results(self):
        super().save_results()
        self._normalize_v2_results_metadata()
        result_count = len(self.deploy_results)
        previous_count = getattr(self, "_latentloop_saved_result_count", 0)
        if result_count > previous_count:
            self.controller.mark_rollout_complete()
        else:
            self.controller.write_runtime_summary()
        self._latentloop_saved_result_count = result_count

    def _normalize_v2_results_metadata(self):
        results_path = Path(self.results_file)
        if not results_path.is_file():
            return
        with results_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
            return
        metadata = payload["metadata"]
        metadata["control_freq"] = float(
            metadata.get("control_freq", self.controller.control_freq)
        )
        metadata["query_interval"] = int(self.controller.query_interval)
        metadata["rollout_policy"] = self.controller.rollout_policy
        metadata["runtime_settings_by_rollout"] = dict(
            self.runtime_settings_by_rollout
        )
        temporary_path = results_path.with_suffix(results_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        os.replace(temporary_path, results_path)

    def on_close(self):
        self.controller.write_runtime_summary()
        super().on_close()


def main():
    gui_args = parse_latentloop_gui_args()
    execution_mode = execution_mode_from_environment()
    controller = LatentLoopSeerControllerV2(
        deployment_method=gui_args.deployment_method,
        deployment_control_freq=gui_args.deployment_control_freq,
        rollout_policy=gui_args.rollout_policy,
        adapter_checkpoint=gui_args.latentloop_adapter_checkpoint,
        artifact_manifest=gui_args.latentloop_artifact_manifest,
        teacher_id=gui_args.latentloop_teacher_id,
        adapter_id=gui_args.latentloop_adapter_id,
        deployment_profile=gui_args.latentloop_deployment_profile,
        optimized_fast_path=bool(gui_args.v2_optimized_fast_path),
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
    configured_control_freq = os.environ.get("SEER_CONTROL_FREQ")
    if configured_control_freq is None:
        raise ValueError("SEER_CONTROL_FREQ must be configured by deploy_ll_gui_v2.sh")
    if abs(float(configured_control_freq) - controller.control_freq) > 1e-9:
        raise ValueError(
            "SEER_CONTROL_FREQ and --deployment-control-freq disagree: "
            f"{configured_control_freq} vs {controller.control_freq}"
        )
    cfg.control_freq = controller.control_freq
    legacy_gui.configure_camera_c(cfg, gui_args.camera_c)
    if execution_mode == "read_only_profile":
        if not gui_args.v2_profile_output_dir:
            raise ValueError(
                "--v2-profile-output-dir is required for read_only_profile"
            )
        cfg.enable_rollout_media = False
        cfg.enable_observer_media = False
        env = ReadOnlyDeployEnvV2(cfg, camera_mode=gui_args.v2_camera_mode)
        try:
            _save_camera_serial_cache(
                _camera_serials_file(cfg),
                env.deploy_camera_serials,
                [cfg.exterior_camera_name, cfg.wrist_camera_name],
            )
            summary = run_read_only_profile(
                controller=controller,
                env=env,
                instruction=os.environ["SEER_LANGUAGE_INSTRUCTION"],
                steps=gui_args.v2_profile_steps,
                warmup_steps=gui_args.v2_profile_warmup_steps,
                control_freq=controller.control_freq,
                output_dir=gui_args.v2_profile_output_dir,
            )
            print(
                "[Seer deploy v2 read-only profile][OK] "
                + json.dumps(summary, indent=2, sort_keys=True)
            )
        finally:
            env.close()
        return

    env = TimedUR5eDeployEnv(cfg, controller.record_control_command)
    _save_camera_serial_cache(
        _camera_serials_file(cfg),
        env.deploy_camera_serials,
        [cfg.exterior_camera_name, cfg.wrist_camera_name],
    )
    root = tk.Tk()
    LatentLoopDeployGuiAppV2(root, cfg, controller, env, gui_args)
    root.mainloop()


if __name__ == "__main__":
    main()
