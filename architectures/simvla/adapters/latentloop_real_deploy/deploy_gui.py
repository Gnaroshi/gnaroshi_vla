"""Safety-gated SimVLA GUI built on the untouched 3DFlow deployment snapshot."""

from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

from .hardware import (
    POLICY_STEP_CANCELLED,
    SafeUR5eDeployEnv,
    build_deploy_config,
    load_legacy_gui,
)


legacy_gui = load_legacy_gui()


class TimedSafeUR5eDeployEnv(SafeUR5eDeployEnv):
    def __init__(
        self,
        cfg,
        *,
        workspace_min,
        workspace_max,
        tracking_error_guard,
        command_callback,
    ):
        super().__init__(
            cfg,
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            tracking_error_guard=tracking_error_guard,
        )
        self._command_callback = command_callback

    def step(self, target_pose6d, target_gripper):
        result = super().step(target_pose6d, target_gripper)
        if result is POLICY_STEP_CANCELLED:
            return result
        self._command_callback(time.perf_counter(), self.last_tracking_error)
        return result


class SimVLADeployGuiApp(legacy_gui.DeployGuiApp):
    """Preserve the validated GUI while attaching SimVLA provenance and metrics."""

    def __init__(self, root, cfg, controller, env, gui_args):
        self.simvla_controller = controller
        self.emergency_stop_reports = []
        cfg.results_dir = str(
            Path(cfg.results_dir)
            / controller.contract.deployment_id
            / controller.deployment_method
        )
        super().__init__(root, cfg, controller, env, gui_args)
        controller.attach_session_dir(self.session_dir)
        self.write_session_files()
        self.update_metrics()
        root.title(
            f"SimVLA real deployment: {controller.deployment_method} "
            f"[{controller.contract.deployment_id}]"
        )

    def _request_arm_stop(self, reason: str) -> dict:
        # emergency_stop raises the abort latch before waiting for the command
        # lock. Calling disarm first would delay that latch behind an active step.
        report = {
            **self.env.emergency_stop(),
            "reason": str(reason),
            "wall_time": time.time(),
        }
        self.emergency_stop_reports.append(report)
        print(
            "[simvla-deploy] arm stop: "
            + json.dumps(report, sort_keys=True),
            flush=True,
        )
        return report

    def start_rollout(self):
        rollout = getattr(self, "rollout_thread", None)
        if rollout is not None and rollout.is_alive():
            return super().start_rollout()
        self.env.arm_policy_commands()
        try:
            return super().start_rollout()
        except BaseException:
            self.env.disarm_policy_commands()
            raise

    def set_run_state(self, message, color="#16a34a"):
        if message == "ERROR" and hasattr(self, "env"):
            self._request_arm_stop("rollout_exception")
        elif message in {"READY TO START", "STOPPED", "RETRY STOP"} and hasattr(
            self, "env"
        ):
            self.env.disarm_policy_commands()
        return super().set_run_state(message, color)

    def signal_current(self, action):
        if action not in {"stop", "retry", "success", "failure"}:
            raise ValueError(f"Unsupported rollout signal: {action}")
        if self.current_events is None:
            self.set_status("No active rollout.")
            return

        # The copied GUI checks its event only at loop boundaries and its
        # ``stop`` branch automatically moves home. Halt the active servo now,
        # then use the motion-free discard branch. A later Start action performs
        # the reviewed home move.
        report = self._request_arm_stop(f"operator_{action}")
        self.env.cancel_pending_policy_step()
        if action in {"stop", "retry"}:
            self.current_events["retry"].set()
            state = "STOPPED" if action == "stop" else "RETRY STOP"
        else:
            self.current_events[action].set()
            state = f"STOPPED: {action.upper()}"
        self.set_run_state(state, "#dc2626")
        if report.get("stopped"):
            disposition = "saved" if action in {"success", "failure"} else "discarded"
            self.set_status(
                f"Arm stopped; rollout will be {disposition} without an automatic home move."
            )
        else:
            self.set_status(
                "Arm stop was requested but not confirmed; use the physical emergency stop."
            )

    def _build_controls(self, parent):
        sections = [
            (
                "Rollout",
                [
                    (
                        "Start New Rollout  [N]",
                        self.start_rollout,
                        "Move home, capture media if enabled, then begin policy rollout.",
                    ),
                    (
                        "Stop and Discard  [X]",
                        lambda: self.signal_current("stop"),
                        "Request an arm stop at the current command boundary, discard the rollout, and remain stationary.",
                    ),
                    (
                        "Stop for Retry  [R]",
                        self.restart_rollout,
                        "Stop the arm and discard the rollout. Start again only after inspection.",
                    ),
                ],
            ),
            (
                "Outcome",
                [
                    (
                        "Success  [S]",
                        lambda: self.signal_current("success"),
                        "Mark the active rollout as successful and save it.",
                    ),
                    (
                        "Failure  [F]",
                        lambda: self.signal_current("failure"),
                        "Mark the active rollout as failed and save it.",
                    ),
                ],
            ),
            (
                "Records",
                [
                    (
                        "Save Results Now  [W]",
                        self.save_results,
                        "Write the current JSON summary immediately.",
                    ),
                    (
                        "Delete Previous Rollout  [D]",
                        self.delete_previous_rollout,
                        "Remove the most recently saved rollout and its media folder.",
                    ),
                ],
            ),
            (
                "Session",
                [
                    (
                        "Exit GUI  [Esc]",
                        self.on_close,
                        "Stop the arm, save results, close connections, and exit.",
                    ),
                ],
            ),
        ]
        for title, items in sections:
            section = self._panel(parent, title)
            section.pack(fill=tk.X, pady=(0, 12))
            for label, command, help_text in items:
                row = tk.Frame(section, bg="#ffffff")
                row.pack(fill=tk.X, pady=6)
                tk.Button(
                    row,
                    text=label,
                    command=command,
                    font=self.font_body,
                    height=2,
                    relief=tk.RAISED,
                    bd=1,
                ).pack(fill=tk.X)
                tk.Label(
                    row,
                    text=help_text,
                    bg="#ffffff",
                    fg="#64748b",
                    font=self.font_small,
                    justify=tk.LEFT,
                    wraplength=285,
                    anchor="w",
                ).pack(fill=tk.X, pady=(3, 0))

    def write_session_files(self):
        super().write_session_files()
        if not hasattr(self, "simvla_controller"):
            return
        path = Path(self.session_manifest_file)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["simvla_deployment"] = self.simvla_controller.deployment_metadata()
        payload["runtime_summary_file_pattern"] = "deployment_runtime_rollout_*.json"
        payload["policy_step_log_pattern"] = "policy_steps_rollout_*.jsonl"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def update_metrics(self):
        super().update_metrics()
        if not hasattr(self, "simvla_controller"):
            return
        metadata = self.simvla_controller.deployment_metadata()
        self.metrics_var.set(
            self.metrics_var.get()
            + "\n\nSimVLA deployment\n"
            + f"{metadata['deployment_id']} / {metadata['deployment_method']}\n"
            + "Protocol: fresh H=10, execute R=5\n"
            + {
                "baseline": "Compute: K_C=1, N_G=10",
                "condition_loop": "Compute: K_C=2, N_G=10",
                "latentloop": "Compute: K_C=2, N_G=3",
                "vla_cache_full": "Compute: VLA-Cache eager reference, no reuse, N_G=10",
                "vla_cache": "Compute: actual visual-token pruning/KV reuse, N_G=10",
            }[metadata["deployment_method"]]
        )

    def save_results(self):
        super().save_results()
        if not hasattr(self, "simvla_controller"):
            return
        results_path = Path(self.results_file)
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            metadata = payload.setdefault("metadata", {})
            metadata["simvla_deployment"] = (
                self.simvla_controller.deployment_metadata()
            )
            metadata["emergency_stop_reports"] = list(self.emergency_stop_reports)
            temporary = results_path.with_suffix(results_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(results_path)
        self.simvla_controller.write_runtime_summary()

    def on_close(self):
        if self.current_events is not None:
            self._request_arm_stop("gui_close")
            # The copied close path sets the event that triggers an automatic
            # home move. Exit through the motion-free discard branch instead.
            self.current_events["retry"].set()
        rollout = self.rollout_thread
        if rollout is not None and rollout.is_alive():
            rollout.join(timeout=5.0)
        self.simvla_controller.write_runtime_summary()
        self.preview.stop()
        self.save_results()
        try:
            self.env.close()
        finally:
            for _name, camera, _serial in self.preview_only_cameras:
                try:
                    camera.close()
                except Exception:
                    pass
            self.root.destroy()


def run_live_gui(*, controller) -> None:
    contract = controller.contract
    cfg = build_deploy_config(contract)
    workspace = contract.hardware["robot"]["workspace_m"]
    tracking = contract.hardware["robot"]["control"]["tracking_error_guard"]
    gui_args = SimpleNamespace(camera_c="no", gui_refresh_ms=150)
    # Establish the operator UI before any robot control connection exists.
    root = tk.Tk()
    env = None
    try:
        root.withdraw()
        env = TimedSafeUR5eDeployEnv(
            cfg,
            workspace_min=workspace["min"],
            workspace_max=workspace["max"],
            tracking_error_guard=dict(tracking),
            command_callback=controller.record_control_command,
        )
        SimVLADeployGuiApp(root, cfg, controller, env, gui_args)
        root.deiconify()
        root.mainloop()
    except BaseException:
        if env is not None:
            try:
                report = env.emergency_stop()
                print("[simvla-deploy] emergency stop after GUI failure: "
                      + json.dumps(report, sort_keys=True), flush=True)
            finally:
                env.close()
        raise
    finally:
        if env is not None:
            env.close()
        try:
            root.destroy()
        except tk.TclError:
            pass
