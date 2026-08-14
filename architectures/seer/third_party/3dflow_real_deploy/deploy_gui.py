import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

import numpy as np

from deploy import (
    DeployConfig,
    DeployResult,
    RealSenseCamera,
    RolloutMediaCapture,
    UR5eDeployEnv,
    _6d_to_pose,
    _camera_serials_file,
    _checkpoint_results_dir,
    _discover_realsense_devices,
    _known_deploy_camera_serials,
    _load_camera_serial_cache,
    _maybe_cuda_sync,
    _prompt_enter,
    _prompt_yes_no,
    _resolve_realsense_serials,
    _save_camera_serial_cache,
    _write_instruction_markers,
    pose_to_6d,
    save_deploy_results,
)
from real_controller.controller import SeerController


class LockedCamera:
    def __init__(self, camera):
        self.camera = camera
        self.lock = threading.Lock()
        self.serial_number = getattr(camera, "serial_number", None)
        self.index = getattr(camera, "index", None)

    def read(self):
        with self.lock:
            return self.camera.read()

    def close(self):
        with self.lock:
            return self.camera.close()


def parse_gui_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--camera-c",
        choices=("ask", "yes", "no"),
        default=os.getenv("SEER_GUI_CAMERA_C", "ask"),
        help="Whether to use camera C for additional observer logging.",
    )
    parser.add_argument(
        "--gui-refresh-ms",
        type=int,
        default=int(os.getenv("SEER_GUI_REFRESH_MS", "150")),
        help="Camera preview refresh period in milliseconds.",
    )
    gui_args, seer_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + seer_args
    return gui_args


def configure_camera_c(cfg, choice):
    cfg.camera_serial_cache = _load_camera_serial_cache(_camera_serials_file(cfg))
    if not cfg.enable_rollout_media:
        print("[deploy-gui] rollout media capture is disabled by SEER_ENABLE_ROLLOUT_MEDIA=0.")
        return
    if choice == "ask":
        choice = "yes" if _prompt_yes_no(
            "[deploy-gui] Use camera C for additional observer logging? Front/wrist rollout videos are saved when media capture is enabled.",
            False,
        ) else "no"

    if choice == "no":
        cfg.enable_observer_media = False
        print("[deploy-gui] Camera C logging disabled. Front/wrist rollout videos will still be saved.")
    else:
        cfg.enable_observer_media = True

    if choice == "no":
        camera_names = [cfg.exterior_camera_name, cfg.wrist_camera_name]
        serials_file = _camera_serials_file(cfg)
        serial_cache = _load_camera_serial_cache(serials_file)
        cfg.camera_serial_cache = serial_cache
        while not _known_deploy_camera_serials(camera_names, serial_cache):
            devices = _discover_realsense_devices()
            if len(devices) <= len(camera_names):
                break
            print(
                f"[deploy-gui] Detected {len(devices)} RealSense cameras. "
                "A/B serials are not cached, so unplug camera C or any unused camera first."
            )
            _prompt_enter("[deploy-gui] After only A/B are connected, press Enter to identify and cache them...")
            serial_cache = _load_camera_serial_cache(serials_file)
        return

    camera_names = [cfg.exterior_camera_name, cfg.wrist_camera_name]
    serials_file = _camera_serials_file(cfg)
    serial_cache = _load_camera_serial_cache(serials_file)
    cfg.camera_serial_cache = serial_cache

    if not _known_deploy_camera_serials(camera_names, serial_cache):
        print(
            "[deploy-gui] A/B deploy camera serials are not cached yet. "
            "Leave camera C unplugged so only A/B are connected."
        )
        while True:
            _prompt_enter("[deploy-gui] After confirming C is unplugged, press Enter to identify A/B...")
            devices = _discover_realsense_devices()
            if len(devices) == len(camera_names):
                serial_map = _resolve_realsense_serials(
                    camera_names,
                    require_explicit_when_extra=False,
                    serial_cache=serial_cache,
                )
                _save_camera_serial_cache(serials_file, serial_map, camera_names)
                cfg.camera_serial_cache = _load_camera_serial_cache(serials_file)
                break
            if len(devices) > len(camera_names):
                print(
                    f"[deploy-gui] Detected {len(devices)} RealSense cameras. "
                    "Unplug C or any unused camera first."
                )
            else:
                print(
                    f"[deploy-gui] Detected only {len(devices)} RealSense cameras. "
                    "Make sure A/B are connected."
                )

    _prompt_enter("[deploy-gui] Plug in camera C now, wait a few seconds, then press Enter...")
    cfg.camera_serial_cache = _load_camera_serial_cache(serials_file)


class CameraPreview:
    def __init__(self, cameras, refresh_s):
        self.cameras = cameras
        self.refresh_s = refresh_s
        self.frames = {}
        self.errors = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _loop(self):
        while not self.stop_event.is_set():
            for name, camera, _serial in self.cameras:
                try:
                    frame = camera.read()
                    with self.lock:
                        self.frames[name] = frame
                        self.errors.pop(name, None)
                except Exception as exc:
                    with self.lock:
                        self.errors[name] = str(exc)
            self.stop_event.wait(self.refresh_s)

    def snapshot(self):
        with self.lock:
            return dict(self.frames), dict(self.errors)


class CompatibleRolloutMediaCapture:
    def __init__(self, media_root, task_index, rollout_index, exterior_camera, observer_camera):
        self.media_root = os.path.abspath(media_root)
        self.task_index = task_index
        self.rollout_index = rollout_index
        self.exterior_camera = exterior_camera
        self.observer_camera = observer_camera
        self.final_dir = os.path.join(
            self.media_root,
            f"task_{task_index:02d}",
            f"rollout_{rollout_index:03d}",
        )
        tmp_name = f"task_{task_index:02d}_rollout_{rollout_index:03d}_{int(time.time() * 1000)}"
        self.tmp_dir = os.path.join(self.media_root, "_tmp", tmp_name)
        prefix = f"task{task_index:02d}_rollout{rollout_index:03d}"
        self.before_image = os.path.join(self.tmp_dir, f"{prefix}_B_exterior_before_deploy.jpg")
        self.observer_start_image = os.path.join(self.tmp_dir, f"{prefix}_C_observer_start.jpg")
        self.observer_video = os.path.join(self.tmp_dir, f"{prefix}_C_observer_rollout.mp4")
        self.raw_video = os.path.join(self.tmp_dir, f"{prefix}_C_observer_rollout_raw.avi")
        self._stop_event = threading.Event()
        self._thread = None
        self._video_error = None

    def _write_rgb_image(self, path, image):
        import cv2

        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def capture_before_image(self):
        if self.exterior_camera is None:
            return None
        frame = self.exterior_camera.read()
        self._write_rgb_image(self.before_image, frame)
        return self.before_image

    def start_observer_recording(self):
        if self.observer_camera is None:
            return None, None

        import cv2

        os.makedirs(self.tmp_dir, exist_ok=True)
        start_frame = self.observer_camera.read()
        self._write_rgb_image(self.observer_start_image, start_frame)
        height, width = start_frame.shape[:2]
        width -= width % 2
        height -= height % 2
        fps = int(os.getenv("SEER_OBSERVER_VIDEO_FPS", os.getenv("SEER_CAMERA_FPS", "30")))

        def fit_even(frame):
            return frame[:height, :width]

        def record_loop():
            writer = None
            try:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(self.raw_video, fourcc, float(fps), (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open raw video writer: {self.raw_video}")
                writer.write(cv2.cvtColor(fit_even(start_frame), cv2.COLOR_RGB2BGR))
                frame_period = 1.0 / max(fps, 1)
                while not self._stop_event.is_set():
                    loop_start = time.perf_counter()
                    frame = fit_even(self.observer_camera.read())
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    sleep_left = frame_period - (time.perf_counter() - loop_start)
                    if sleep_left > 0:
                        self._stop_event.wait(sleep_left)
            except Exception as exc:
                self._video_error = exc
                print(f"[deploy-gui] observer video recording error: {exc}")
            finally:
                if writer is not None:
                    writer.release()

        self._stop_event.clear()
        self._thread = threading.Thread(target=record_loop, daemon=True)
        self._thread.start()
        return self.observer_start_image, self.observer_video

    def _encode_mp4(self):
        if not os.path.exists(self.raw_video):
            return
        ffmpeg = self._find_ffmpeg()
        if ffmpeg:
            ffmpeg_codecs = [
                ("libx264", ["-preset", "veryfast", "-crf", "23"]),
                ("mpeg4", ["-q:v", "4"]),
            ]
            for codec, codec_args in ffmpeg_codecs:
                cmd = [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    self.raw_video,
                    "-an",
                    "-c:v",
                    codec,
                    *codec_args,
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    self.observer_video,
                ]
                try:
                    subprocess.run(cmd, check=True)
                    if os.path.exists(self.observer_video) and os.path.getsize(self.observer_video) > 0:
                        os.remove(self.raw_video)
                        self._video_error = None
                        return
                except Exception as exc:
                    self._video_error = exc
                    print(f"[deploy-gui] ffmpeg {codec} encode failed: {exc}")

        import cv2

        cap = cv2.VideoCapture(self.raw_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or float(os.getenv("SEER_OBSERVER_VIDEO_FPS", "30"))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            self.observer_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if writer is None:
            cap.release()
            raise RuntimeError(f"Failed to create playable mp4: {self.observer_video}")
        if not writer.isOpened():
            writer.release()
            cap.release()
            raise RuntimeError(f"Failed to open mp4v writer: {self.observer_video}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        cap.release()
        if os.path.exists(self.observer_video) and os.path.getsize(self.observer_video) > 0:
            os.remove(self.raw_video)
            self._video_error = None

    def _find_ffmpeg(self):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            print(f"[deploy-gui] imageio-ffmpeg is not available: {exc}")
            return None

    def stop_recording(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        try:
            self._encode_mp4()
        except Exception as exc:
            self._video_error = exc
            print(f"[deploy-gui] observer mp4 finalization error: {exc}")

    def commit(self):
        self.stop_recording()
        if os.path.exists(self.final_dir):
            shutil.rmtree(self.final_dir)
        os.makedirs(os.path.dirname(self.final_dir), exist_ok=True)
        if os.path.exists(self.tmp_dir):
            shutil.move(self.tmp_dir, self.final_dir)
        return self.to_result_media()

    def discard(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def to_result_media(self):
        def final_path(path):
            candidate = os.path.join(self.final_dir, os.path.basename(path))
            return candidate if os.path.exists(candidate) else None

        media = {
            "directory": self.final_dir,
            "before_deploy_image": final_path(self.before_image),
            "observer_start_image": final_path(self.observer_start_image),
            "observer_rollout_video": final_path(self.observer_video),
        }
        if self._video_error is not None:
            media["observer_video_error"] = str(self._video_error)
        return media


class DeployGuiApp:
    def __init__(self, root, cfg, controller, env, gui_args):
        self.root = root
        self.cfg = cfg
        self.controller = controller
        self.env = env
        self.gui_args = gui_args
        self.listener = None
        self.deploy_results = []
        self.rollout_thread = None
        self.rollout_lock = threading.Lock()
        self.current_events = None
        self.status_text = tk.StringVar(value="Ready")
        self.state_text = tk.StringVar(value="READY TO START")
        self.ui_thread_id = threading.get_ident()
        self.panel_order = ["state", "camera", "controls"]
        self.panel_widgets = {}
        self.panel_bodies = {}
        self.panel_headers = {}
        self.dragging_panel = None
        self.metadata_expanded = False
        self.metadata_var = tk.StringVar(value="No saved rollout yet.")
        self.metadata_button_var = tk.StringVar(value="Show latest media metadata")
        self.rollouts_count_var = tk.StringVar(value="0")
        self.success_count_var = tk.StringVar(value="0")
        self.failure_count_var = tk.StringVar(value="0")
        self.success_rate_var = tk.StringVar(value="0.0%")
        self.notes_save_after = None

        self.task_instructions = cfg.language_instructions or [cfg.language_instruction]
        self.task_index = 0
        self.session_stamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.checkpoint_dir = _checkpoint_results_dir(cfg.results_dir, controller)
        self.session_dir = os.path.join(self.checkpoint_dir, f"deploy_{self.session_stamp}")
        self.results_file = os.path.join(self.session_dir, "deploy_results.json")
        self.media_dir = os.path.join(self.session_dir, "rollouts")
        self.notes_file = os.path.join(self.session_dir, "NOTES.txt")
        self.session_manifest_file = os.path.join(self.session_dir, "SESSION_MANIFEST.json")
        self.session_camera_serials_file = os.path.join(self.session_dir, "CAMERA_SERIALS.json")
        os.makedirs(self.session_dir, exist_ok=True)
        _write_instruction_markers(self.session_dir, self.task_instructions)
        self._ensure_notes_file()
        self.write_session_files()
        if cfg.enable_rollout_media:
            os.makedirs(self.media_dir, exist_ok=True)

        self.rollout_steps = int(getattr(controller.args, "real_eval_max_steps", 600))
        self.frame_stride = max(int(getattr(controller.args, "eval_frame_stride", 1)), 1)
        self.frame_offset = int(getattr(controller.args, "eval_frame_offset", 0))
        self.skip_blend_ratio = float(getattr(controller.args, "skip_action_blend_ratio", 1.0))
        self.skip_blend_offset = int(getattr(controller.args, "skip_action_blend_offset", 0))
        self.skip_action_direct = bool(getattr(controller.args, "skip_action_direct", False))

        self._wrap_env_cameras()
        self.preview_only_cameras = self._open_preview_only_cameras()
        self.cameras = self._camera_entries()
        self.preview = CameraPreview(self.cameras, max(gui_args.gui_refresh_ms, 50) / 1000.0)

        self.camera_labels = {}
        self.camera_images = {}
        self.metrics_var = tk.StringVar()
        self.camera_canvas = None
        self.camera_grid = None
        self._build_ui()
        self.save_results()
        self.preview.start()
        self.root.after(gui_args.gui_refresh_ms, self.refresh_ui)

    def write_session_files(self):
        os.makedirs(self.session_dir, exist_ok=True)
        camera_payload = {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "deploy_cameras": dict(getattr(self.env, "deploy_camera_serials", {})),
            "all_session_cameras": dict(getattr(self.env, "camera_serials", {})),
            "notes": (
                "This file is a per-deploy-session copy. The global camera serial "
                "cache may also exist at real_deploy_results/deploy_camera_serials.json."
            ),
        }
        manifest = {
            "session_started_at": self.session_started_at,
            "session_stamp": self.session_stamp,
            "checkpoint_dir": self.checkpoint_dir,
            "session_dir": self.session_dir,
            "results_file": self.results_file,
            "media_dir": self.media_dir if self.cfg.enable_rollout_media else None,
            "notes_file": self.notes_file,
            "camera_serials_file": self.session_camera_serials_file,
            "instructions": self.task_instructions,
            "checkpoint": str(getattr(self.controller.args, "resume_from_checkpoint", "")),
            "media_capture_enabled": self.cfg.enable_rollout_media,
        }
        with open(self.session_camera_serials_file, "w", encoding="utf-8") as file:
            json.dump(camera_payload, file, indent=2, ensure_ascii=False)
        with open(self.session_manifest_file, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)

    def _ensure_notes_file(self):
        if not os.path.exists(self.notes_file):
            with open(self.notes_file, "w", encoding="utf-8") as file:
                file.write("")

    def set_status(self, message):
        if threading.get_ident() == self.ui_thread_id:
            self.status_text.set(message)
        else:
            self.root.after(0, self.status_text.set, message)

    def set_run_state(self, message, color="#16a34a"):
        def apply():
            self.state_text.set(message)
            if hasattr(self, "state_badge"):
                self.state_badge.configure(bg=color)

        if threading.get_ident() == self.ui_thread_id:
            apply()
        else:
            self.root.after(0, apply)

    def schedule_notes_save(self):
        if self.notes_save_after is not None:
            self.root.after_cancel(self.notes_save_after)
        self.notes_save_after = self.root.after(750, self.save_notes)

    def save_notes(self):
        if not hasattr(self, "notes_text"):
            return
        self.notes_save_after = None
        try:
            with open(self.notes_file, "w", encoding="utf-8") as file:
                file.write(self.notes_text.get("1.0", tk.END).rstrip() + "\n")
        except OSError as exc:
            self.set_status(f"Failed to save notes: {exc}")

    def _wrap_env_cameras(self):
        self.env.exterior_camera = LockedCamera(self.env.exterior_camera)
        self.env.wrist_camera = LockedCamera(self.env.wrist_camera)
        if self.env.observer_camera is not None:
            self.env.observer_camera = LockedCamera(self.env.observer_camera)

    def _camera_entries(self):
        entries = [
            ("B exterior", self.env.exterior_camera, self.env.camera_serials.get("exterior")),
            ("A wrist", self.env.wrist_camera, self.env.camera_serials.get("wrist")),
        ]
        if self.env.observer_camera is not None:
            entries.append(("C observer", self.env.observer_camera, self.env.camera_serials.get("observer")))
        entries.extend(self.preview_only_cameras)
        return entries

    def _open_preview_only_cameras(self):
        entries = []
        used_serials = {serial for serial in self.env.camera_serials.values() if serial}
        for device in _discover_realsense_devices():
            serial = device["serial_number"]
            if serial in used_serials:
                continue
            try:
                camera = RealSenseCamera(
                    serial,
                    width=int(os.getenv("SEER_CAMERA_WIDTH", "640")),
                    height=int(os.getenv("SEER_CAMERA_HEIGHT", "480")),
                    fps=int(os.getenv("SEER_CAMERA_FPS", "30")),
                )
            except Exception as exc:
                print(f"[deploy-gui] failed to open preview-only camera {serial}: {exc}")
                continue
            entries.append((f"extra RealSense {len(entries) + 1}", LockedCamera(camera), serial))
        return entries

    def _build_ui(self):
        self.root.title("3DFlow-Seer Real Deploy GUI")
        self.root.geometry("1440x980")
        self.root.minsize(1120, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.configure(bg="#f4f6f8")

        self.font_title = tkfont.Font(family="TkDefaultFont", size=16, weight="bold")
        self.font_section = tkfont.Font(family="TkDefaultFont", size=12, weight="bold")
        self.font_body = tkfont.Font(family="TkDefaultFont", size=11)
        self.font_small = tkfont.Font(family="TkDefaultFont", size=10)
        self.font_metric = tkfont.Font(family="TkDefaultFont", size=12)

        header = tk.Frame(self.root, bg="#18212f", padx=14, pady=10)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text="3DFlow-Seer Real Deploy",
            fg="#ffffff",
            bg="#18212f",
            font=self.font_title,
            anchor="w",
        ).pack(side=tk.LEFT)
        tk.Label(
            header,
            textvariable=self.status_text,
            fg="#d7e2f0",
            bg="#18212f",
            font=self.font_body,
            anchor="e",
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.state_badge = tk.Label(
            header,
            textvariable=self.state_text,
            fg="#ffffff",
            bg="#16a34a",
            font=self.font_section,
            padx=14,
            pady=6,
        )
        self.state_badge.pack(side=tk.RIGHT, padx=12)

        main = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            bg="#d8dee8",
            sashwidth=8,
            sashrelief=tk.RAISED,
            showhandle=True,
        )
        self.main = main
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        info_panel, info_body = self._draggable_panel(main, "state", "Live deploy state")
        info_body.grid_columnconfigure(0, weight=1)
        info_body.grid_rowconfigure(3, weight=1)
        self._build_stats_cards(info_body)
        tk.Label(
            info_body,
            textvariable=self.metrics_var,
            justify=tk.LEFT,
            anchor="nw",
            bg="#ffffff",
            fg="#1f2937",
            font=self.font_metric,
            wraplength=380,
        ).grid(row=1, column=0, sticky="ew", padx=12, pady=(10, 8))
        self._build_latest_media_section(info_body)
        self._build_notes_section(info_body)

        camera_panel, camera_body = self._draggable_panel(main, "camera", "Camera previews")
        tk.Label(
            camera_body,
            text="All connected cameras are shown with serial numbers. Scroll if the window is small.",
            bg="#ffffff",
            fg="#516070",
            font=self.font_small,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(0, 8))
        camera_body.grid_rowconfigure(1, weight=1)
        camera_body.grid_columnconfigure(0, weight=1)
        self._build_camera_board(camera_body)

        controls_panel, controls_body = self._draggable_panel(main, "controls", "Controls")
        self._build_controls(controls_body)
        self._place_main_panels()
        self._bind_shortcuts()
        self.update_metrics()

    def _panel(self, parent, title):
        frame = tk.LabelFrame(
            parent,
            text=title,
            bg="#ffffff",
            fg="#18212f",
            font=self.font_section,
            padx=10,
            pady=10,
            labelanchor="n",
        )
        return frame

    def _build_stats_cards(self, parent):
        stats = tk.Frame(parent, bg="#ffffff")
        stats.grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 8))
        for col in range(4):
            stats.grid_columnconfigure(col, weight=1)
        cards = [
            ("Rollouts", self.rollouts_count_var, "#1d4ed8"),
            ("Success", self.success_count_var, "#15803d"),
            ("Failure", self.failure_count_var, "#b91c1c"),
            ("Rate", self.success_rate_var, "#7c3aed"),
        ]
        big_font = tkfont.Font(family="TkDefaultFont", size=18, weight="bold")
        for col, (label, var, color) in enumerate(cards):
            card = tk.Frame(stats, bg="#f8fafc", highlightthickness=1, highlightbackground="#d8dee8")
            card.grid(row=0, column=col, sticky="nsew", padx=4)
            tk.Label(card, text=label, bg="#f8fafc", fg="#475569", font=self.font_small).pack(pady=(7, 0))
            tk.Label(card, textvariable=var, bg="#f8fafc", fg=color, font=big_font).pack(pady=(0, 7))

    def _build_latest_media_section(self, parent):
        section = self._panel(parent, "Latest media metadata")
        section.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 10))
        tk.Button(
            section,
            textvariable=self.metadata_button_var,
            command=self.toggle_latest_metadata,
            font=self.font_body,
        ).pack(fill=tk.X)
        self.metadata_details = tk.Label(
            section,
            textvariable=self.metadata_var,
            justify=tk.LEFT,
            anchor="nw",
            bg="#ffffff",
            fg="#334155",
            font=self.font_small,
            wraplength=380,
        )

    def _build_notes_section(self, parent):
        section = self._panel(parent, "Notes")
        section.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
        section.grid_rowconfigure(0, weight=1)
        section.grid_columnconfigure(0, weight=1)
        self.notes_text = tk.Text(
            section,
            height=9,
            wrap=tk.WORD,
            font=self.font_body,
            padx=8,
            pady=8,
            relief=tk.SOLID,
            bd=1,
        )
        self.notes_text.grid(row=0, column=0, sticky="nsew")
        notes_scroll = tk.Scrollbar(section, orient=tk.VERTICAL, command=self.notes_text.yview)
        notes_scroll.grid(row=0, column=1, sticky="ns")
        self.notes_text.configure(yscrollcommand=notes_scroll.set)
        notes_footer = tk.Frame(section, bg="#ffffff")
        notes_footer.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        tk.Button(
            notes_footer,
            text="Done Notes",
            command=self.release_notes_focus,
            font=self.font_small,
        ).pack(side=tk.LEFT)
        tk.Label(
            notes_footer,
            text="Ctrl+Enter also returns focus to deploy controls.",
            bg="#ffffff",
            fg="#64748b",
            font=self.font_small,
        ).pack(side=tk.LEFT, padx=8)
        try:
            with open(self.notes_file, "r", encoding="utf-8") as file:
                self.notes_text.insert("1.0", file.read())
        except OSError:
            pass
        self.notes_text.bind("<KeyRelease>", lambda _event: self.schedule_notes_save())
        self.notes_text.bind("<Control-Return>", lambda _event: self.release_notes_focus())

    def toggle_latest_metadata(self):
        self.metadata_expanded = not self.metadata_expanded
        if self.metadata_expanded:
            self.metadata_details.pack(fill=tk.X, padx=4, pady=(8, 2))
            self.metadata_button_var.set("Hide latest media metadata")
        else:
            self.metadata_details.pack_forget()
            self.metadata_button_var.set("Show latest media metadata")

    def release_notes_focus(self):
        self.save_notes()
        self.root.focus_set()
        self.set_status("Notes saved. Keyboard shortcuts are active again.")
        return "break"

    def _draggable_panel(self, parent, key, title):
        outer = tk.Frame(parent, bg="#ffffff", highlightthickness=1, highlightbackground="#d8dee8")
        header = tk.Label(
            outer,
            text=f"{title}    drag to move",
            bg="#e8eef6",
            fg="#18212f",
            font=self.font_section,
            padx=10,
            pady=8,
            anchor="w",
            cursor="fleur",
        )
        header.pack(fill=tk.X)
        body = tk.Frame(outer, bg="#ffffff", padx=10, pady=10)
        body.pack(fill=tk.BOTH, expand=True)
        header.bind("<ButtonPress-1>", lambda event, panel_key=key: self._begin_panel_drag(panel_key))
        header.bind("<ButtonRelease-1>", lambda event, panel_key=key: self._finish_panel_drag(panel_key, event))
        self.panel_widgets[key] = outer
        self.panel_bodies[key] = body
        self.panel_headers[key] = header
        return outer, body

    def _begin_panel_drag(self, key):
        self.dragging_panel = key
        for panel_key, header in self.panel_headers.items():
            if panel_key == key:
                header.configure(bg="#f59e0b", text=f"{header.cget('text')}    dragging...")
            else:
                header.configure(bg="#e8eef6")

    def _finish_panel_drag(self, key, event):
        if self.dragging_panel != key:
            return
        self.dragging_panel = None
        for panel_key, header in self.panel_headers.items():
            title = header.cget("text").split("    drag to move")[0].split("    dragging")[0]
            header.configure(bg="#e8eef6", text=f"{title}    drag to move")
        target_key = None
        for candidate, widget in self.panel_widgets.items():
            left = widget.winfo_rootx()
            right = left + widget.winfo_width()
            if left <= event.x_root <= right:
                target_key = candidate
                break
        if target_key is None or target_key == key:
            return
        source_index = self.panel_order.index(key)
        target_index = self.panel_order.index(target_key)
        self.panel_order[source_index], self.panel_order[target_index] = (
            self.panel_order[target_index],
            self.panel_order[source_index],
        )
        self._place_main_panels()

    def _place_main_panels(self):
        for pane in self.main.panes():
            self.main.forget(pane)
        for key in self.panel_order:
            minsize = 360 if key == "camera" else 300
            stretch = "always" if key == "camera" else "never"
            self.main.add(self.panel_widgets[key], minsize=minsize, stretch=stretch)

    def _bind_shortcuts(self):
        self.root.bind_all("<KeyPress>", self._handle_keypress)

    def _handle_keypress(self, event):
        if self.root.focus_displayof() is None:
            return
        if isinstance(event.widget, (tk.Text, tk.Entry)):
            return
        key = event.keysym.lower()
        shortcuts = {
            "n": self.start_rollout,
            "s": lambda: self.signal_current("success"),
            "f": lambda: self.signal_current("failure"),
            "r": self.restart_rollout,
            "x": lambda: self.signal_current("stop"),
            "w": self.save_results,
            "d": self.delete_previous_rollout,
            "escape": self.on_close,
        }
        command = shortcuts.get(key)
        if command is not None:
            command()

    def _build_controls(self, parent):
        sections = [
            (
                "Rollout",
                [
                    ("Start New Rollout  [N]", self.start_rollout, "Move home, capture media if enabled, then begin policy rollout."),
                    ("Restart / Retry  [R]", self.restart_rollout, "Discard the current rollout and start again. If idle, starts a new rollout."),
                    ("Stop Current  [X]", lambda: self.signal_current("stop"), "Stop, discard the active rollout, and return the robot home."),
                    ("Discard Current Deploy", self.discard_current_deploy, "Discard this deploy session folder and reset counts. No keyboard shortcut."),
                ],
            ),
            (
                "Outcome",
                [
                    ("Success  [S]", lambda: self.signal_current("success"), "Mark the active rollout as successful and save it."),
                    ("Failure  [F]", lambda: self.signal_current("failure"), "Mark the active rollout as failed and save it."),
                ],
            ),
            (
                "Records",
                [
                    ("Save Results Now  [W]", self.save_results, "Write the current JSON summary immediately."),
                    ("Delete Previous Rollout  [D]", self.delete_previous_rollout, "Remove the most recently saved rollout and its media folder."),
                ],
            ),
            (
                "Session",
                [
                    ("Exit GUI  [Esc]", self.on_close, "Save current results, close cameras/robot connections, and exit the GUI."),
                ],
            ),
        ]
        for idx, (title, items) in enumerate(sections):
            section = self._panel(parent, title)
            section.pack(fill=tk.X, pady=(0, 12))
            for label, command, help_text in items:
                row = tk.Frame(section, bg="#ffffff")
                row.pack(fill=tk.X, pady=6)
                btn = tk.Button(
                    row,
                    text=label,
                    command=command,
                    font=self.font_body,
                    height=2,
                    relief=tk.RAISED,
                    bd=1,
                )
                btn.pack(fill=tk.X)
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

    def _build_camera_board(self, parent):
        container = tk.Frame(parent, bg="#ffffff")
        container.grid(row=1, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.camera_canvas = tk.Canvas(container, bg="#ffffff", highlightthickness=0)
        y_scroll = tk.Scrollbar(container, orient=tk.VERTICAL, command=self.camera_canvas.yview)
        x_scroll = tk.Scrollbar(container, orient=tk.HORIZONTAL, command=self.camera_canvas.xview)
        self.camera_canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.camera_canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        self.camera_grid = tk.Frame(self.camera_canvas, bg="#ffffff")
        window_id = self.camera_canvas.create_window((0, 0), window=self.camera_grid, anchor="nw")
        self.camera_grid.bind(
            "<Configure>",
            lambda _event: self.camera_canvas.configure(scrollregion=self.camera_canvas.bbox("all")),
        )
        self.camera_canvas.bind(
            "<Configure>",
            lambda event: self._layout_camera_cards(event.width, window_id),
        )
        self._create_camera_cards()

    def _create_camera_cards(self):
        self.camera_cards = []
        for name, _camera, serial in self.cameras:
            frame = tk.LabelFrame(
                self.camera_grid,
                text=f"{name}  |  serial: {serial or 'unknown'}",
                bg="#ffffff",
                fg="#1f2937",
                font=self.font_body,
                padx=8,
                pady=8,
                labelanchor="n",
            )
            label = tk.Label(
                frame,
                text="waiting for frame",
                bg="#111827",
                fg="#e5e7eb",
                font=self.font_body,
                compound=tk.CENTER,
            )
            label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
            self.camera_cards.append((frame, name))
            self.camera_labels[name] = label

    def _layout_camera_cards(self, canvas_width, window_id):
        columns = 1 if canvas_width < 760 else 2
        for index, (frame, _name) in enumerate(self.camera_cards):
            row = index // columns
            col = index % columns
            frame.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)
            self.camera_grid.grid_rowconfigure(row, weight=1, minsize=390)
        for col in range(columns):
            self.camera_grid.grid_columnconfigure(col, weight=1, minsize=500)
        self.camera_canvas.itemconfigure(window_id, width=max(canvas_width, 720))

    def refresh_ui(self):
        frames, errors = self.preview.snapshot()
        for name, label in self.camera_labels.items():
            if name in frames:
                photo = self._frame_to_photo(frames[name], label=label)
                label.configure(image=photo, text="")
                label.image = photo
            elif name in errors:
                label.configure(text=errors[name], image="")
                label.image = None
        self.update_metrics()
        self.root.after(self.gui_args.gui_refresh_ms, self.refresh_ui)

    def _frame_to_photo(self, frame, label=None):
        from PIL import Image, ImageTk

        max_w, max_h = 460, 345
        widget_w = label.winfo_width() if label is not None else 0
        widget_h = label.winfo_height() if label is not None else 0
        if widget_w > 40 and widget_h > 40:
            max_w, max_h = widget_w - 8, widget_h - 8
        h, w = frame.shape[:2]
        scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
        if scale <= 0:
            scale = 1.0
        image = Image.fromarray(frame)
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        return ImageTk.PhotoImage(image=image)

    def update_metrics(self):
        total = len(self.deploy_results)
        success = sum(1 for result in self.deploy_results if result.success)
        failure = total - success
        rate = (success / total * 100.0) if total else 0.0
        self.rollouts_count_var.set(str(total))
        self.success_count_var.set(str(success))
        self.failure_count_var.set(str(failure))
        self.success_rate_var.set(f"{rate:.1f}%")
        self.metadata_var.set(self._latest_media_metadata_text())
        checkpoint = getattr(self.controller.args, "resume_from_checkpoint", "unknown")
        instruction = self.task_instructions[self.task_index]
        self.metrics_var.set(
            "\n".join(
                [
                    f"Instruction\n{instruction}",
                    "",
                    f"Checkpoint\n{checkpoint}",
                    "",
                    f"Results JSON\n{self.results_file}",
                    "",
                    f"Media directory\n{self.media_dir if self.cfg.enable_rollout_media else 'disabled'}",
                    "",
                    f"Camera serials\n{self.env.camera_serials}",
                ]
            )
        )

    def _latest_media_metadata_text(self):
        if not self.deploy_results:
            return "No saved rollout yet."
        result = self.deploy_results[-1]
        lines = [
            f"Rollout: task {result.task_index}, rollout {result.rollout_index}",
            f"Outcome: {'success' if result.success else 'failure'}",
            f"Steps: {result.steps_completed}/{result.total_steps}",
            f"Duration: {result.duration:.1f}s",
        ]
        media = result.media or {}
        for label, key in (
            ("Directory", "directory"),
            ("Front before image", "before_deploy_image"),
            ("Front rollout video", "front_rollout_video"),
            ("Wrist rollout video", "wrist_rollout_video"),
            ("C start image", "observer_start_image"),
            ("C rollout video", "observer_rollout_video"),
        ):
            path = media.get(key)
            if not path:
                lines.append(f"{label}: not saved")
                continue
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
                lines.append(f"{label}: {path}")
                lines.append(f"  size={size_mb:.2f} MB, modified={modified}")
            else:
                lines.append(f"{label}: missing on disk ({path})")
        if media.get("observer_video_error"):
            lines.append(f"Video warning: {media['observer_video_error']}")
        for role, error in (media.get("rollout_video_errors") or {}).items():
            if role == "observer" and media.get("observer_video_error"):
                continue
            lines.append(f"{role} video warning: {error}")
        return "\n".join(lines)

    def save_results(self):
        self.save_notes()
        self.write_session_files()
        save_deploy_results(
            self.results_file,
            self.deploy_results,
            cfg=self.cfg,
            controller=self.controller,
            session_started_at=self.session_started_at,
            media_dir=self.media_dir if self.cfg.enable_rollout_media else None,
            camera_serials=self.env.camera_serials,
        )
        self.set_status(f"Saved results: {self.results_file}")

    def start_rollout(self):
        with self.rollout_lock:
            if self.rollout_thread is not None and self.rollout_thread.is_alive():
                self.set_status("A rollout is already running.")
                return
            rollout_index = len(self.deploy_results) + 1
            self.current_events = {
                "stop": threading.Event(),
                "success": threading.Event(),
                "failure": threading.Event(),
                "retry": threading.Event(),
            }
            self.rollout_thread = threading.Thread(
                target=self._run_rollout,
                args=(self.task_index, rollout_index, self.current_events),
                daemon=True,
            )
            self.rollout_thread.start()
            self.set_status(f"Starting rollout {rollout_index}...")
            self.set_run_state("RUNNING", "#2563eb")

    def signal_current(self, action):
        if self.current_events is None:
            self.set_status("No active rollout.")
            return
        if action in self.current_events:
            self.current_events[action].set()
            self.set_status(f"Requested {action}.")
            if action == "stop":
                self.set_run_state("STOPPING", "#dc2626")

    def restart_rollout(self):
        if self.rollout_thread is not None and self.rollout_thread.is_alive():
            self.signal_current("retry")
        else:
            self.start_rollout()

    def discard_current_deploy(self):
        if self.rollout_thread is not None and self.rollout_thread.is_alive():
            self.signal_current("stop")
            return
        session_dir = self.session_dir
        self.deploy_results.clear()
        if os.path.isdir(session_dir):
            shutil.rmtree(session_dir)
        os.makedirs(self.session_dir, exist_ok=True)
        _write_instruction_markers(self.session_dir, self.task_instructions)
        self._ensure_notes_file()
        if hasattr(self, "notes_text"):
            self.notes_text.delete("1.0", tk.END)
        if self.cfg.enable_rollout_media:
            os.makedirs(self.media_dir, exist_ok=True)
        self.write_session_files()
        self.save_results()
        self.update_metrics()
        self.set_status(f"Discarded current deploy session and reset folder: {session_dir}")

    def delete_previous_rollout(self):
        if self.rollout_thread is not None and self.rollout_thread.is_alive():
            messagebox.showwarning("Rollout running", "Stop or finish the current rollout first.")
            return
        if not self.deploy_results:
            self.set_status("No previous rollout to delete.")
            return
        result = self.deploy_results.pop()
        media_dir = result.media.get("directory") if result.media else None
        if media_dir and os.path.isdir(media_dir):
            shutil.rmtree(media_dir)
        self.save_results()
        self.set_status(f"Deleted rollout {result.rollout_index}.")

    def _run_rollout(self, task_index, rollout_index, events):
        instruction = self.task_instructions[task_index]
        media_capture = None
        timestep = 0
        rollout_start_wall = time.time()
        try:
            self.set_run_state("MOVING HOME", "#7c3aed")
            self.set_status("Moving robot home. Reset scene if needed.")
            self.env.move_to_home()
            self.set_run_state("RUNNING", "#2563eb")

            if self.cfg.enable_rollout_media:
                media_capture = RolloutMediaCapture(
                    self.media_dir,
                    task_index + 1,
                    rollout_index,
                    self.env.exterior_camera,
                    self.env.observer_camera,
                    wrist_camera=self.env.wrist_camera,
                    camera_serials=self.env.camera_serials,
                )
                try:
                    media_capture.capture_before_image()
                    media_capture.start_observer_recording()
                except Exception as exc:
                    self.set_status(f"Media capture warning: {exc}")

            for _ in range(self.cfg.warmup_steps):
                if events["stop"].is_set() or events["retry"].is_set():
                    break
                obs = {
                    "robot_state": self.env.get_robot_state(),
                    "color_image": self.env.get_color_images(),
                    "language_instruction": instruction,
                }
                self.controller.forward(obs, include_info=True, timestep=0)

            self.controller.reset()
            robot_state = self.env.get_robot_state()
            last2robot_pose = robot_state["pose"]
            last_raw_action = None
            rollout_success = None

            while timestep < self.rollout_steps:
                if events["stop"].is_set():
                    break
                if events["retry"].is_set():
                    break
                if events["success"].is_set():
                    rollout_success = True
                    break
                if events["failure"].is_set():
                    rollout_success = False
                    break

                _maybe_cuda_sync()
                start_time = time.perf_counter()
                obs = {
                    "robot_state": self.env.get_robot_state(),
                    "color_image": self.env.get_color_images(),
                    "language_instruction": instruction,
                }
                do_infer = (
                    timestep == 0
                    or self.frame_stride == 1
                    or last_raw_action is None
                    or (timestep % self.frame_stride) == self.frame_offset
                )
                if do_infer:
                    target_pos, target_euler, target_gripper, _ = self.controller.forward(
                        obs, include_info=True, timestep=timestep
                    )
                    last_raw_action = np.array(
                        [
                            target_pos[0],
                            target_pos[1],
                            target_pos[2],
                            target_euler[0],
                            target_euler[1],
                            target_euler[2],
                            float(target_gripper),
                        ],
                        dtype=np.float64,
                    )
                else:
                    action_vec = last_raw_action
                    if (
                        self.skip_action_direct
                        and self.controller.use_ensembling
                        and hasattr(self.controller, "get_skip_action")
                    ):
                        a2 = self.controller.get_skip_action(timestep)
                        if a2 is not None:
                            action_vec = a2
                    elif (
                        self.skip_blend_ratio < 1.0
                        and self.controller.use_ensembling
                        and hasattr(self.controller, "get_skip_action")
                    ):
                        a2 = self.controller.get_skip_action(timestep + self.skip_blend_offset)
                        if a2 is not None:
                            action_vec = self.skip_blend_ratio * last_raw_action + (1.0 - self.skip_blend_ratio) * a2
                    target_pos = action_vec[:3]
                    target_euler = action_vec[3:6]
                    target_gripper = float(action_vec[6])

                target_pos = np.asarray(target_pos, dtype=np.float64) * self.cfg.max_rel_pos
                target_euler = np.asarray(target_euler, dtype=np.float64) * self.cfg.max_rel_orn
                cur2last_pose = _6d_to_pose(np.concatenate([target_pos, target_euler]))
                last2robot_pose = last2robot_pose @ cur2last_pose
                target_pose = pose_to_6d(last2robot_pose)

                self.env.step(target_pose, float(target_gripper))
                timestep += 1
                _maybe_cuda_sync()
                sleep_left = (1.0 / self.cfg.control_freq) - (time.perf_counter() - start_time)
                if sleep_left > 0:
                    time.sleep(sleep_left)

            if events["stop"].is_set() or events["retry"].is_set():
                if media_capture is not None:
                    media_capture.discard()
                if events["stop"].is_set():
                    self.set_run_state("MOVING HOME", "#7c3aed")
                    self.set_status("Stopped. Returning robot home.")
                    self.env.move_to_home()
                self.set_status("Rollout discarded.")
                self.set_run_state("READY TO START", "#16a34a")
                return

            if rollout_success is None:
                self.set_run_state("WAITING FOR OUTCOME", "#d97706")
                self.set_status("Rollout reached max steps. Click Success, Failure, or Restart.")
                while True:
                    if events["stop"].is_set() or events["retry"].is_set():
                        if media_capture is not None:
                            media_capture.discard()
                        if events["stop"].is_set():
                            self.set_run_state("MOVING HOME", "#7c3aed")
                            self.set_status("Stopped. Returning robot home.")
                            self.env.move_to_home()
                        self.set_status("Rollout discarded.")
                        self.set_run_state("READY TO START", "#16a34a")
                        return
                    if events["success"].is_set():
                        rollout_success = True
                        break
                    if events["failure"].is_set():
                        rollout_success = False
                        break
                    time.sleep(0.05)

            media = media_capture.commit() if media_capture is not None else {}
            result = DeployResult(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                task_index=task_index + 1,
                rollout_index=rollout_index,
                instruction=instruction,
                success=bool(rollout_success),
                duration=time.time() - rollout_start_wall,
                steps_completed=timestep,
                total_steps=self.rollout_steps,
                media=media,
            )
            self.deploy_results.append(result)
            self.save_results()
            self.set_status(f"Saved rollout {rollout_index}: success={rollout_success}, steps={timestep}")
            self.set_run_state("MOVING HOME", "#7c3aed")
            self.env.move_to_home()
            self.set_run_state("READY TO START", "#16a34a")
        except Exception as exc:
            if media_capture is not None:
                media_capture.discard()
            self.set_status(f"Rollout error: {exc}")
            self.set_run_state("ERROR", "#dc2626")
            print(f"[deploy-gui] rollout error: {exc}")

    def on_close(self):
        if self.current_events is not None:
            self.current_events["stop"].set()
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


def main():
    gui_args = parse_gui_args()
    cfg = DeployConfig()
    configure_camera_c(cfg, gui_args.camera_c)
    controller = SeerController()
    env = UR5eDeployEnv(cfg)
    _save_camera_serial_cache(
        _camera_serials_file(cfg),
        env.deploy_camera_serials,
        [cfg.exterior_camera_name, cfg.wrist_camera_name],
    )
    root = tk.Tk()
    app = DeployGuiApp(root, cfg, controller, env, gui_args)
    root.mainloop()


if __name__ == "__main__":
    main()
