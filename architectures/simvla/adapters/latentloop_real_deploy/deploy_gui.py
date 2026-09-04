"""Safety-gated SimVLA GUI built on the untouched 3DFlow deployment snapshot."""

from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace

from .hardware import SafeUR5eDeployEnv, build_deploy_config, load_legacy_gui


legacy_gui = load_legacy_gui()


class TimedSafeUR5eDeployEnv(SafeUR5eDeployEnv):
    def __init__(self, cfg, *, workspace_min, workspace_max, command_callback):
        super().__init__(
            cfg, workspace_min=workspace_min, workspace_max=workspace_max
        )
        self._command_callback = command_callback

    def step(self, target_pose6d, target_gripper):
        result = super().step(target_pose6d, target_gripper)
        self._command_callback(time.perf_counter())
        return result


class SimVLADeployGuiApp(legacy_gui.DeployGuiApp):
    """Preserve the validated GUI while attaching SimVLA provenance and metrics."""

    def __init__(self, root, cfg, controller, env, gui_args):
        self.simvla_controller = controller
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
            payload.setdefault("metadata", {})["simvla_deployment"] = (
                self.simvla_controller.deployment_metadata()
            )
            temporary = results_path.with_suffix(results_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(results_path)
        self.simvla_controller.write_runtime_summary()

    def on_close(self):
        self.simvla_controller.write_runtime_summary()
        super().on_close()


def run_live_gui(*, controller) -> None:
    contract = controller.contract
    cfg = build_deploy_config(contract)
    workspace = contract.hardware["robot"]["workspace_m"]
    env = TimedSafeUR5eDeployEnv(
        cfg,
        workspace_min=workspace["min"],
        workspace_max=workspace["max"],
        command_callback=controller.record_control_command,
    )
    gui_args = SimpleNamespace(camera_c="no", gui_refresh_ms=150)
    root = tk.Tk()
    try:
        SimVLADeployGuiApp(root, cfg, controller, env, gui_args)
        root.mainloop()
    except BaseException:
        env.close()
        raise
