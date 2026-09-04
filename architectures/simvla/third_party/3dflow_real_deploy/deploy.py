import os
import json
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from real_controller.controller import SeerController
from real_controller.robotiq_gripper import RobotiqGripper

try:
    from pynput import keyboard
except ImportError:
    keyboard = None


def _6d_to_pose(pose6d, degrees=False):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = pose6d[:3]
    pose[:3, :3] = R.from_euler("xyz", pose6d[3:6], degrees=degrees).as_matrix()
    return pose


def pose_to_6d(pose, degrees=False):
    pose6d = np.zeros(6, dtype=np.float64)
    pose6d[:3] = pose[:3, 3]
    pose6d[3:6] = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=degrees)
    return pose6d


def _pose6d_to_ur_tcp(pose6d):
    tcp = np.zeros(6, dtype=np.float64)
    tcp[:3] = pose6d[:3]
    tcp[3:] = R.from_euler("xyz", pose6d[3:6], degrees=False).as_rotvec()
    return tcp


def _ur_tcp_to_pose6d(tcp_pose):
    pose6d = np.zeros(6, dtype=np.float64)
    pose6d[:3] = tcp_pose[:3]
    pose6d[3:] = R.from_rotvec(np.asarray(tcp_pose[3:], dtype=np.float64)).as_euler(
        "xyz", degrees=False
    )
    return pose6d


def _maybe_cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _parse_language_instructions():
    raw = os.getenv("SEER_LANGUAGE_INSTRUCTIONS", "")
    if raw.strip():
        instructions = [part.strip() for part in raw.split("||") if part.strip()]
        if instructions:
            return instructions
    default_instruction = os.getenv(
        "SEER_LANGUAGE_INSTRUCTION",
        os.getenv("LANGUAGE_INSTRUCTION", "put the green apple, put it in the drawer, and close the drawer."),
    )
    return [default_instruction]

@dataclass
class DeployResult:
    timestamp: str
    task_index: int
    rollout_index: int
    instruction: str
    success: bool
    duration: float
    steps_completed: int
    total_steps: int
    media: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _make_control_events():
    return {
        "stop_deployment": False,
        "next_task": False,
        "retry_task": False,
        "success_task": False,
        "failure_task": False,
    }


def _reset_control_events(events):
    for key in events:
        events[key] = False


def _clear_non_stop_events(events):
    for key in ("next_task", "retry_task", "success_task", "failure_task"):
        events[key] = False


def _prompt_yes_no(message, default=False):
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"{message} {suffix}: ").strip().lower()
    except EOFError:
        return default
    if not answer:
        return default
    return answer in {"y", "yes"}


def _prompt_enter(message):
    try:
        input(message)
    except EOFError:
        pass


def init_keyboard_listener():
    if keyboard is None:
        raise RuntimeError(
            "pynput is required for keyboard control in deploy.py. Install it to use manual task labeling."
        )

    events = _make_control_events()

    def on_press(key):
        try:
            if key == keyboard.Key.esc:
                print("[deploy] ESC pressed. Stopping deployment.")
                events["stop_deployment"] = True
            elif key == keyboard.Key.right:
                events["next_task"] = True
            elif key == keyboard.Key.left:
                events["retry_task"] = True
            elif key == keyboard.Key.up:
                events["success_task"] = True
            elif key == keyboard.Key.down:
                events["failure_task"] = True
        except Exception as exc:
            print(f"[deploy] keyboard handler error: {exc}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def _make_deploy_metadata(cfg, controller, deploy_results, session_started_at, media_dir, camera_serials):
    total_rollouts = len(deploy_results)
    success_count = sum(1 for result in deploy_results if result.success)
    failure_count = total_rollouts - success_count
    success_rate = (success_count / total_rollouts) if total_rollouts else 0.0
    args = getattr(controller, "args", None)
    checkpoint = getattr(args, "resume_from_checkpoint", None)
    checkpoint = str(checkpoint) if checkpoint is not None else None
    vit_checkpoint = getattr(args, "vit_checkpoint_path", None)
    vit_checkpoint = str(vit_checkpoint) if vit_checkpoint is not None else None
    return {
        "session_started_at": session_started_at,
        "last_updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": checkpoint,
        "checkpoint_name": os.path.basename(str(checkpoint)) if checkpoint else None,
        "checkpoint_run": os.path.basename(os.path.dirname(str(checkpoint))) if checkpoint else None,
        "vit_checkpoint_path": vit_checkpoint,
        "tasks": cfg.language_instructions or [cfg.language_instruction],
        "requested_rollouts_per_task": cfg.num_rollouts,
        "requested_total_rollouts": cfg.num_rollouts
        * len(cfg.language_instructions or [cfg.language_instruction]),
        "total_rollouts": total_rollouts,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_rate,
        "success_rate_percent": success_rate * 100.0,
        "rollout_steps": int(getattr(args, "real_eval_max_steps", 600)) if args is not None else None,
        "control_freq": cfg.control_freq,
        "camera_names": {
            "exterior": cfg.exterior_camera_name,
            "wrist": cfg.wrist_camera_name,
            "observer": cfg.observer_camera_name,
        },
        "camera_serials": camera_serials,
        "media_capture_enabled": cfg.enable_rollout_media,
        "observer_media_enabled": cfg.enable_observer_media,
        "media_dir": media_dir,
    }


def _safe_filename(text, max_len=160):
    safe = "".join(ch if ch.isalnum() or ch in (" ", "-", "_", ".") else "_" for ch in text)
    safe = "_".join(safe.strip().split())
    safe = safe.strip("._")
    if not safe:
        safe = "instruction"
    return safe[:max_len]


def _safe_marker_filename(text, max_len=180):
    safe = "".join(ch if ch not in ('/', '\\', ':', '*', '?', '"', '<', '>', '|') else "_" for ch in text)
    safe = " ".join(safe.strip().split())
    safe = safe.strip(" .")
    if not safe:
        safe = "instruction"
    return safe[:max_len]


def _checkpoint_results_dir(results_root, controller):
    args = getattr(controller, "args", None)
    checkpoint = getattr(args, "resume_from_checkpoint", None)
    if checkpoint:
        checkpoint = str(checkpoint)
        run_name = os.path.basename(os.path.dirname(checkpoint)) or "checkpoint"
        ckpt_name = os.path.splitext(os.path.basename(checkpoint))[0] or "unknown"
        folder_name = f"{_safe_filename(run_name)}_ckpt_{_safe_filename(ckpt_name)}"
    else:
        folder_name = "checkpoint_unknown"
    return os.path.join(results_root, folder_name)


def _write_instruction_markers(session_dir, instructions):
    os.makedirs(session_dir, exist_ok=True)
    multiple = len(instructions) > 1
    for index, instruction in enumerate(instructions, start=1):
        stem = _safe_marker_filename(instruction)
        filename = f"{index:02d}_{stem}.txt" if multiple else f"{stem}.txt"
        path = os.path.join(session_dir, filename)
        Path(path).touch(exist_ok=True)


def save_deploy_results(
    results_file,
    deploy_results,
    cfg=None,
    controller=None,
    session_started_at=None,
    media_dir=None,
    camera_serials=None,
):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    if cfg is not None and controller is not None:
        payload = {
            "metadata": _make_deploy_metadata(
                cfg,
                controller,
                deploy_results,
                session_started_at,
                media_dir,
                camera_serials or {},
            ),
            "results": [result.to_dict() for result in deploy_results],
        }
    else:
        payload = [result.to_dict() for result in deploy_results]
    with open(results_file, "w") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


class RealSenseCamera:
    def __init__(self, serial_number: str, width: int = 640, height: int = 480, fps: int = 30):
        import pyrealsense2 as rs

        self._rs = rs
        self.serial_number = serial_number
        self._read_lock = threading.Lock()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial_number)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.pipeline.start(self.config)

    def read(self):
        with self._read_lock:
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
            color = frames.get_color_frame()
            if color is None:
                raise RuntimeError(f"Failed to read RGB frame from RealSense {self.serial_number}")
            return np.asanyarray(color.get_data())

    def close(self):
        with self._read_lock:
            try:
                self.pipeline.stop()
            except Exception:
                pass


class OpenCVCamera:
    def __init__(self, index: int, width: int = 640, height: int = 480, fps: int = 30):
        import cv2

        self._cv2 = cv2
        self.index = index
        self._read_lock = threading.Lock()
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        with self._read_lock:
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame from camera index {self.index}")
            return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def close(self):
        with self._read_lock:
            try:
                self.cap.release()
            except Exception:
                pass


def _safe_camera_token(value, fallback="unknown"):
    text = str(value) if value is not None else fallback
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    safe = safe.strip("._-")
    return safe or fallback


class RolloutMediaCapture:
    def __init__(
        self,
        media_root,
        task_index,
        rollout_index,
        exterior_camera,
        observer_camera,
        wrist_camera=None,
        camera_serials=None,
    ):
        self.media_root = Path(media_root)
        self.task_index = task_index
        self.rollout_index = rollout_index
        self.camera_serials = camera_serials or {}
        self.exterior_camera = exterior_camera
        self.wrist_camera = wrist_camera
        self.observer_camera = observer_camera
        self.final_dir = (
            self.media_root
            / f"task_{task_index:02d}"
            / f"rollout_{rollout_index:03d}"
        )
        self.tmp_dir = (
            self.media_root
            / "_tmp"
            / f"task_{task_index:02d}_rollout_{rollout_index:03d}_{int(time.time() * 1000)}"
        )
        prefix = f"task{task_index:02d}_rollout{rollout_index:03d}"
        front_serial = self._camera_serial("front", exterior_camera)
        observer_serial = self._camera_serial("observer", observer_camera)
        self.before_image = self.tmp_dir / (
            f"{prefix}_front_exterior_serial-{_safe_camera_token(front_serial)}_before_deploy.jpg"
        )
        self.observer_start_image = self.tmp_dir / (
            f"{prefix}_observer_serial-{_safe_camera_token(observer_serial)}_start.jpg"
        )
        self._recording_cameras = {
            "front": {
                "label": "front_exterior",
                "camera": exterior_camera,
                "serial": front_serial,
            },
            "wrist": {
                "label": "wrist",
                "camera": wrist_camera,
                "serial": self._camera_serial("wrist", wrist_camera),
            },
            "observer": {
                "label": "observer",
                "camera": observer_camera,
                "serial": observer_serial,
            },
        }
        self.video_paths = {}
        for role, info in self._recording_cameras.items():
            if info["camera"] is None:
                continue
            label = info["label"]
            serial = _safe_camera_token(info["serial"])
            self.video_paths[role] = self.tmp_dir / (
                f"{prefix}_{label}_serial-{serial}_rollout.mp4"
            )
        self.observer_video = self.video_paths.get("observer")
        self._stop_event = threading.Event()
        self._threads = {}
        self._video_error = None
        self._video_errors = {}
        self._default_video_backend = os.getenv("SEER_ROLLOUT_VIDEO_BACKEND", "ffmpeg").strip().lower()
        self._video_backends = {}
        self._video_validation = {}

    def _camera_serial(self, role, camera):
        serial_key = "exterior" if role == "front" else role
        serial = self.camera_serials.get(serial_key)
        if serial:
            return serial
        serial = getattr(camera, "serial_number", None)
        if serial:
            return serial
        index = getattr(camera, "index", None)
        if index is not None:
            return f"opencv{index}"
        return "unknown"

    def _write_rgb_image(self, path, image):
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def capture_before_image(self):
        if self.exterior_camera is None:
            return None
        frame = self.exterior_camera.read()
        self._write_rgb_image(self.before_image, frame)
        return self.before_image

    def start_observer_recording(self):
        return self.start_rollout_recordings()

    def start_rollout_recordings(self):
        if not self.video_paths:
            return None, None

        fps = int(
            os.getenv(
                "SEER_ROLLOUT_VIDEO_FPS",
                os.getenv("SEER_OBSERVER_VIDEO_FPS", os.getenv("SEER_CAMERA_FPS", "30")),
            )
        )

        def record_loop(role, camera, video_path, start_frame):
            import cv2

            writer = None
            ffmpeg_proc = None
            try:
                height, width = start_frame.shape[:2]
                backend = self._default_video_backend
                use_ffmpeg = backend != "opencv" and shutil.which("ffmpeg") is not None
                if use_ffmpeg:
                    ffmpeg_proc = self._start_ffmpeg_writer(video_path, width, height, fps, role)
                    self._write_ffmpeg_frame(ffmpeg_proc, start_frame)
                else:
                    self._video_backends[role] = "opencv"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(video_path),
                        fourcc,
                        float(fps),
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open video writer: {video_path}")
                    writer.write(cv2.cvtColor(start_frame, cv2.COLOR_RGB2BGR))
                frame_period = 1.0 / max(fps, 1)
                while not self._stop_event.is_set():
                    loop_start = time.perf_counter()
                    frame = camera.read()
                    if ffmpeg_proc is not None:
                        self._write_ffmpeg_frame(ffmpeg_proc, frame)
                    else:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    sleep_left = frame_period - (time.perf_counter() - loop_start)
                    if sleep_left > 0:
                        self._stop_event.wait(sleep_left)
            except Exception as exc:
                self._video_errors[role] = exc
                if role == "observer":
                    self._video_error = exc
                print(f"[deploy] {role} video recording error: {exc}")
            finally:
                if ffmpeg_proc is not None:
                    self._close_ffmpeg_writer(ffmpeg_proc, role)
                if writer is not None:
                    writer.release()

        self._stop_event.clear()
        for role, info in self._recording_cameras.items():
            camera = info["camera"]
            video_path = self.video_paths.get(role)
            if camera is None or video_path is None:
                continue
            start_frame = camera.read()
            if role == "observer":
                self._write_rgb_image(self.observer_start_image, start_frame)
            thread = threading.Thread(
                target=record_loop,
                args=(role, camera, video_path, start_frame),
                daemon=True,
            )
            self._threads[role] = thread
            thread.start()
        return self.observer_start_image, self.observer_video

    def _start_ffmpeg_writer(self, video_path, width, height, fps, role):
        video_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            os.getenv(
                "SEER_ROLLOUT_VIDEO_X264_PRESET",
                os.getenv("SEER_OBSERVER_VIDEO_X264_PRESET", "ultrafast"),
            ),
            "-crf",
            os.getenv("SEER_ROLLOUT_VIDEO_CRF", os.getenv("SEER_OBSERVER_VIDEO_CRF", "23")),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._video_backends[role] = "ffmpeg"
        return proc

    @staticmethod
    def _as_rgb24_frame(frame):
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise RuntimeError(f"Expected RGB frame with shape HxWx3, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    def _write_ffmpeg_frame(self, proc, frame):
        if proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        proc.stdin.write(self._as_rgb24_frame(frame).tobytes())

    def _close_ffmpeg_writer(self, proc, role):
        stderr = b""
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            stderr = proc.stderr.read() if proc.stderr is not None else b""
            return_code = proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            stderr = proc.stderr.read() if proc.stderr is not None else b""
            return_code = proc.wait(timeout=2.0)
        if return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            error = RuntimeError(f"ffmpeg exited with code {return_code}: {message}")
            self._video_errors[role] = error
            if role == "observer":
                self._video_error = error
            print(f"[deploy] {role} video recording error: {error}")

    def stop_recording(self):
        self._stop_event.set()
        stop_timeout = float(
            os.getenv(
                "SEER_ROLLOUT_VIDEO_STOP_TIMEOUT_SEC",
                os.getenv("SEER_OBSERVER_VIDEO_STOP_TIMEOUT_SEC", "15.0"),
            )
        )
        for role, thread in list(self._threads.items()):
            thread.join(timeout=stop_timeout)
            if thread.is_alive():
                error = RuntimeError(
                    f"{role} video thread did not stop within {stop_timeout:.1f}s"
                )
                self._video_errors[role] = error
                if role == "observer":
                    self._video_error = error
                print(f"[deploy] {role} video recording error: {error}")
            else:
                self._threads.pop(role, None)
        self._video_validation = {
            role: self._validate_video_file(path)
            for role, path in self.video_paths.items()
        }

    def commit(self):
        self.stop_recording()
        if self.final_dir.exists():
            shutil.rmtree(self.final_dir)
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.tmp_dir.exists():
            shutil.move(str(self.tmp_dir), str(self.final_dir))
        self._video_validation = {
            role: self._validate_video_file(self.final_dir / path.name)
            for role, path in self.video_paths.items()
        }
        return self.to_result_media()

    def discard(self):
        self.stop_recording()
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)

    def to_result_media(self):
        def final_path(tmp_path):
            path = self.final_dir / tmp_path.name
            return str(path) if path.exists() else None

        media = {
            "directory": str(self.final_dir),
            "before_deploy_image": final_path(self.before_image),
            "observer_start_image": final_path(self.observer_start_image),
            "front_rollout_video": final_path(self.video_paths["front"])
            if "front" in self.video_paths
            else None,
            "wrist_rollout_video": final_path(self.video_paths["wrist"])
            if "wrist" in self.video_paths
            else None,
            "observer_rollout_video": final_path(self.observer_video)
            if self.observer_video is not None
            else None,
            "rollout_videos": {
                role: final_path(path)
                for role, path in self.video_paths.items()
                if final_path(path) is not None
            },
            "rollout_video_camera_serials": {
                role: info["serial"]
                for role, info in self._recording_cameras.items()
                if role in self.video_paths
            },
            "rollout_video_backends": dict(self._video_backends),
            "observer_video_backend": self._video_backends.get("observer"),
        }
        validation = self._video_validation
        if not validation:
            validation = {
                role: self._validate_video_file(Path(path))
                for role, path in media["rollout_videos"].items()
            }
        if validation:
            media["rollout_video_validation"] = validation
            if "observer" in validation:
                media["observer_video_validation"] = validation["observer"]
        if self._video_errors:
            media["rollout_video_errors"] = {
                role: str(error) for role, error in self._video_errors.items()
            }
        if self._video_error is not None:
            media["observer_video_error"] = str(self._video_error)
        return media

    @staticmethod
    def _validate_video_file(path):
        path = Path(path)
        validation = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "ffprobe_ok": False,
        }
        if not path.exists() or path.stat().st_size <= 0:
            validation["error"] = "missing_or_empty_video"
            return validation
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            validation["error"] = "ffprobe_not_found"
            return validation
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames,duration",
                "-show_entries",
                "format=duration,size",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        validation["ffprobe_returncode"] = result.returncode
        if result.returncode != 0:
            validation["error"] = result.stderr.strip()
            return validation
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            validation["error"] = f"ffprobe_json_decode_failed:{exc}"
            return validation
        validation["ffprobe_ok"] = True
        validation["ffprobe"] = payload
        return validation


def _get_camera_serial_from_env(name: str):
    candidates = [
        f"SEER_{name.upper()}_CAMERA_SERIAL",
        f"UR5E_{name.upper()}_CAMERA_SERIAL",
    ]
    if name.lower() in {"observer", "record", "recording", "side", "c"}:
        candidates.extend(
            [
                "SEER_C_CAMERA_SERIAL",
                "SEER_RECORD_CAMERA_SERIAL",
                "SEER_RECORDING_CAMERA_SERIAL",
                "UR5E_C_CAMERA_SERIAL",
            ]
        )
    for key in candidates:
        value = os.getenv(key)
        if value:
            return value
    return None


def _discover_realsense_devices():
    try:
        import pyrealsense2 as rs
    except ImportError:
        return []

    context = rs.context()
    devices = []
    for device in context.query_devices():
        devices.append(
            {
                "name": device.get_info(rs.camera_info.name),
                "serial_number": device.get_info(rs.camera_info.serial_number),
            }
        )
    return devices


def _camera_serials_file(cfg):


    if cfg.camera_serials_file:
        return cfg.camera_serials_file
    return os.path.join(cfg.results_dir, "deploy_camera_serials.json")


def _load_camera_serial_cache(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[deploy] failed to load camera serial cache {path}: {exc}")
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("deploy_cameras"), dict):
        return {
            key: value
            for key, value in payload["deploy_cameras"].items()
            if isinstance(value, str) and value
        }
    return {
        key: value
        for key, value in payload.items()
        if isinstance(key, str) and isinstance(value, str) and value
    }


def _save_camera_serial_cache(path, serial_map, camera_names):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing_cache = _load_camera_serial_cache(path)
    deploy_cameras = {
        key: value
        for key, value in existing_cache.items()
        if isinstance(value, str) and value
    }
    deploy_cameras.update({
        name: serial_map[name]
        for name in camera_names
        if name in serial_map and serial_map[name]
    })
    missing_names = [name for name in camera_names if not deploy_cameras.get(name)]
    if missing_names:
        print(
            "[deploy] deploy camera serial cache was not updated because not all "
            f"deploy views have RealSense serials. Missing: {missing_names}, "
            f"known: {deploy_cameras}"
        )
        return
    payload = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "deploy_cameras": deploy_cameras,
        "notes": (
            "These are the RealSense serials used by deploy inference views. "
            "Any connected RealSense not listed here can be used as the observer camera."
        ),
    }
    with open(path, "w") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    print(f"[deploy] saved deploy camera serial cache to {path}: {deploy_cameras}")


def _known_deploy_camera_serials(camera_names, serial_cache):
    return all(_get_camera_serial_from_env(name) or serial_cache.get(name) for name in camera_names)


def _resolve_realsense_serials(camera_names, require_explicit_when_extra=False, serial_cache=None):
    devices = _discover_realsense_devices()
    if devices:
        print("[deploy] detected RealSense devices:")
        for device in devices:
            print(
                f"[deploy]   name={device['name']} serial={device['serial_number']}"
            )

    available_serials = [device["serial_number"] for device in devices]
    assigned = {}
    used_serials = set()

    for name in camera_names:
        explicit_serial = _get_camera_serial_from_env(name)
        if explicit_serial is None and serial_cache:
            explicit_serial = serial_cache.get(name)
            if explicit_serial is not None:
                print(
                    f"[deploy] camera view '{name}' -> cached RealSense serial "
                    f"{explicit_serial}"
                )
        if explicit_serial is None:
            continue
        if available_serials and explicit_serial not in available_serials:
            raise RuntimeError(
                f"Requested {name} RealSense serial {explicit_serial} was not detected. "
                f"Available serials: {available_serials}"
            )
        assigned[name] = explicit_serial
        used_serials.add(explicit_serial)

    missing_names = [name for name in camera_names if name not in assigned]
    remaining_serials = sorted(
        [serial for serial in available_serials if serial not in used_serials]
    )

    if (
        require_explicit_when_extra
        and missing_names
        and len(available_serials) > len(camera_names)
    ):
        raise RuntimeError(
            "More RealSense cameras were detected than deploy input views, so A/B camera "
            "serials must be fixed before deployment starts. "
            f"Missing deploy camera serials for views: {missing_names}. "
            "Set SEER_EXTERIOR_CAMERA_SERIAL and SEER_WRIST_CAMERA_SERIAL, then the "
            "remaining unused RealSense serial can be treated as camera C."
        )

    if missing_names and remaining_serials:
        if len(remaining_serials) < len(missing_names):
            raise RuntimeError(
                f"Not enough RealSense cameras detected for automatic assignment. "
                f"Missing views: {missing_names}, available serials: {remaining_serials}"
            )
        if len(missing_names) > 1:
            print(
                "[deploy] auto-assigning RealSense serials by sorted serial order. "
                "Override with SEER_*_CAMERA_SERIAL if the mapping is wrong."
            )
        for name, serial in zip(missing_names, remaining_serials):
            assigned[name] = serial

    for name in camera_names:
        if name in assigned:
            print(f"[deploy] camera view '{name}' -> RealSense serial {assigned[name]}")

    return assigned


def _env_flag(name: str, default: str = "0"):
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_observer_serial(observer_name, serial_map, serial_cache=None):
    explicit_serial = _get_camera_serial_from_env(observer_name)
    if explicit_serial is None and serial_cache:
        explicit_serial = serial_cache.get(observer_name)
        if explicit_serial is not None:
            print(
                f"[deploy] camera view '{observer_name}' -> cached RealSense serial "
                f"{explicit_serial}"
            )
    devices = _discover_realsense_devices()
    available_serials = [device["serial_number"] for device in devices]
    if explicit_serial:
        if available_serials and explicit_serial not in available_serials:
            raise RuntimeError(
                f"Requested observer RealSense serial {explicit_serial} was not detected. "
                f"Available serials: {available_serials}"
            )
        if explicit_serial in set(serial_map.values()):
            raise RuntimeError(
                f"Requested observer RealSense serial {explicit_serial} is already used "
                f"by a deploy input camera: {serial_map}"
            )
        print(f"[deploy] camera view '{observer_name}' -> RealSense serial {explicit_serial}")
        return explicit_serial

    used_serials = set(serial_map.values())
    remaining_serials = sorted(
        [serial for serial in available_serials if serial not in used_serials]
    )
    if len(remaining_serials) == 1:
        print(
            f"[deploy] camera view '{observer_name}' -> RealSense serial "
            f"{remaining_serials[0]} (auto-detected as unused camera)"
        )
        return remaining_serials[0]
    if len(remaining_serials) > 1:
        print(
            "[deploy] observer camera was not opened because multiple unused RealSense "
            f"serials are available: {remaining_serials}. Set SEER_OBSERVER_CAMERA_SERIAL."
        )
    else:
        print(
            "[deploy] observer camera was not opened because no unused RealSense camera "
            "was detected. Set SEER_OBSERVER_CAMERA_SERIAL after plugging in camera C."
        )
    return None


def _prepare_observer_media_choice(cfg):
    if not cfg.enable_rollout_media:
        print("[deploy] rollout media capture is disabled by SEER_ENABLE_ROLLOUT_MEDIA=0.")
        return

    use_observer = _prompt_yes_no(
        "[deploy] Use camera C for additional observer logging? Front/wrist rollout videos are saved when media capture is enabled.",
        default=False,
    )
    if not use_observer:
        cfg.enable_observer_media = False
        print("[deploy] Camera C logging disabled. Front/wrist rollout videos will still be saved.")
        camera_names = [cfg.exterior_camera_name, cfg.wrist_camera_name]
        serials_file = _camera_serials_file(cfg)
        serial_cache = _load_camera_serial_cache(serials_file)
        cfg.camera_serial_cache = serial_cache
        while not _known_deploy_camera_serials(camera_names, serial_cache):
            devices = _discover_realsense_devices()
            if len(devices) <= len(camera_names):
                break
            print(
                f"[deploy] Detected {len(devices)} RealSense cameras. "
                "A/B serials are not cached, so unplug camera C or any unused cameras first."
            )
            _prompt_enter("[deploy] After only A/B are connected, press Enter to identify and cache them...")
            serial_cache = _load_camera_serial_cache(serials_file)
        return

    cfg.enable_observer_media = True
    camera_names = [cfg.exterior_camera_name, cfg.wrist_camera_name]
    serials_file = _camera_serials_file(cfg)
    serial_cache = _load_camera_serial_cache(serials_file)
    cfg.camera_serial_cache = serial_cache

    if not _known_deploy_camera_serials(camera_names, serial_cache):
        print(
            "[deploy] A/B deploy camera serials are not fixed yet. "
            "Leave camera C unplugged now so only the deploy cameras A/B are connected."
        )
        while True:
            _prompt_enter("[deploy] After confirming C is unplugged, press Enter to identify A/B...")
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
                    f"[deploy] Detected {len(devices)} RealSense cameras. "
                    "Unplug camera C first; A/B must be identified without C."
                )
            else:
                print(
                    f"[deploy] Detected only {len(devices)} RealSense cameras. "
                    "Make sure A/B are connected."
                )
    else:
        print(
            f"[deploy] using fixed A/B camera serials from env/cache: "
            f"{ {name: _get_camera_serial_from_env(name) or cfg.camera_serial_cache.get(name) for name in camera_names} }"
        )

    _prompt_enter("[deploy] Plug in camera C now, wait a few seconds, then press Enter to continue...")
    cfg.camera_serial_cache = _load_camera_serial_cache(serials_file)


def _build_camera(name: str, serial_map=None):
    index_env = f"SEER_{name.upper()}_CAMERA_INDEX"
    serial = None if serial_map is None else serial_map.get(name)
    if serial is None:
        serial = _get_camera_serial_from_env(name)
    width = int(os.getenv("SEER_CAMERA_WIDTH", "640"))
    height = int(os.getenv("SEER_CAMERA_HEIGHT", "480"))
    fps = int(os.getenv("SEER_CAMERA_FPS", "30"))
    if serial:
        return RealSenseCamera(serial, width=width, height=height, fps=fps)
    index = int(os.getenv(index_env, "0" if name == "exterior" else "1"))
    print(f"[deploy] camera view '{name}' -> OpenCV index {index}")
    return OpenCVCamera(index, width=width, height=height, fps=fps)


@dataclass
class DeployConfig:
    robot_ip: str = os.getenv("SEER_ROBOT_IP", "192.168.56.101")
    language_instruction: str = os.getenv(
        "SEER_LANGUAGE_INSTRUCTION",
        os.getenv("LANGUAGE_INSTRUCTION", "put the green apple, put it in the drawer, and close the drawer."),
    )
    language_instructions: list[str] = field(default_factory=_parse_language_instructions)
    home_pose: list[float] = field(
        default_factory=lambda: [3.14, -1.57, 1.57, -1.57, -1.57, -1.57, -1.57]
    )

    # 3dflow hang cup pose
    # home_pose: list[float] = field(
    #     default_factory=lambda: [2.58, -1.61, 1.88, -1.85, -1.57, -1.57]
    # ) 
    # 147.93, -92, 107.49, -105.89, -89.95, -89.96
    
    # 3dflow stack cups pose
    # home_pose: list[float] = field(
    # default_factory=lambda: [3.0663, -1.5523, 1.9526, -1.9541, -1.5631, -1.5320]
    # )
    # 175.69, -88.92, 111.91, -111.97, -89.56, -87.80
    
    # 3dflow rings pose
    # home_pose: list[float] = field(
    # default_factory=lambda: [2.9933, -1.5375, 1.7312, -1.7524, -1.5400, -1.6705]
    # )
    # 171.50, -88.09, 99.19, -100.40, -88.23, -95.71

    # 3dflow wipe pose
    # home_pose: list[float] = field(
    # default_factory=lambda: [2.9765, -1.6107, 1.9039, -1.8115, -1.5169, -1.6909]
    # )
    # 170.54, -92.28, 109.09, -103.79, -86.91, -96.88

    # 3dflow cabinet pose
    # home_pose: list[float] = field(
    # default_factory=lambda: [2.9622, -1.6058, 1.9031, -1.8445, -1.5429, -1.7120]
    # )
    # 169.72, -92.01, 109.04, -105.68, -88.40, -98.09

    home_move_duration: float = float(os.getenv("SEER_HOME_MOVE_DURATION", "3.0"))
    home_move_fps: int = int(os.getenv("SEER_HOME_MOVE_FPS", "60"))
    control_freq: float = float(os.getenv("SEER_CONTROL_FREQ", "10"))
    max_rel_pos: float = float(os.getenv("SEER_MAX_REL_POS", "0.02"))
    max_rel_orn: float = float(os.getenv("SEER_MAX_REL_ORN", "0.05"))
    exterior_camera_name: str = os.getenv("SEER_EXTERIOR_CAMERA_NAME", "exterior")
    wrist_camera_name: str = os.getenv("SEER_WRIST_CAMERA_NAME", "wrist")
    observer_camera_name: str = os.getenv("SEER_OBSERVER_CAMERA_NAME", "observer")
    gripper_open_threshold: float = float(os.getenv("SEER_GRIPPER_OPEN_THRESHOLD", "0.1"))
    gripper_speed: int = int(os.getenv("SEER_GRIPPER_SPEED", "255"))
    gripper_force: int = int(os.getenv("SEER_GRIPPER_FORCE", "10"))
    gripper_min_delta: int = int(os.getenv("SEER_GRIPPER_MIN_DELTA", "3"))
    gripper_min_period_s: float = float(os.getenv("SEER_GRIPPER_MIN_PERIOD", "0.05"))
    arm_acceleration: float = float(os.getenv("SEER_ARM_ACCELERATION", "0.5"))
    arm_velocity: float = float(os.getenv("SEER_ARM_VELOCITY", "0.5"))
    servoj_time: float = float(os.getenv("SEER_SERVOJ_TIME", str(1.0 / 500.0)))
    servoj_lookahead: float = float(os.getenv("SEER_SERVOJ_LOOKAHEAD", "0.2"))
    servoj_gain: float = float(os.getenv("SEER_SERVOJ_GAIN", "100"))
    warmup_steps: int = int(os.getenv("SEER_WARMUP_STEPS", "3"))
    num_rollouts: int = int(os.getenv("SEER_NUM_ROLLOUTS", "15"))
    results_dir: str = os.getenv("SEER_RESULTS_DIR", "real_deploy_results")
    enable_rollout_media: bool = _env_flag("SEER_ENABLE_ROLLOUT_MEDIA", "1")
    enable_observer_media: bool = _env_flag("SEER_ENABLE_OBSERVER_MEDIA", "1")
    camera_serials_file: str = os.getenv("SEER_CAMERA_SERIALS_FILE", "")
    camera_serial_cache: dict = field(default_factory=dict)


class UR5eDeployEnv:
    def __init__(self, cfg: DeployConfig):
        import rtde_control
        import rtde_receive

        self.cfg = cfg
        self.rtde_ctrl = rtde_control.RTDEControlInterface(cfg.robot_ip)
        self.rtde_rec = rtde_receive.RTDEReceiveInterface(cfg.robot_ip)

        self.gripper = RobotiqGripper()
        self.gripper.connect(cfg.robot_ip, 63352)
        self.gripper.activate(auto_calibrate=True)

        camera_names = [cfg.exterior_camera_name, cfg.wrist_camera_name]
        serial_map = _resolve_realsense_serials(
            camera_names,
            require_explicit_when_extra=True,
            serial_cache=cfg.camera_serial_cache,
        )
        self.exterior_camera = _build_camera(cfg.exterior_camera_name, serial_map)
        self.wrist_camera = _build_camera(cfg.wrist_camera_name, serial_map)
        self.deploy_camera_serials = {
            name: serial_map.get(name)
            for name in camera_names
            if serial_map.get(name)
        }
        self.observer_camera = None
        observer_serial = None
        if cfg.enable_rollout_media and cfg.enable_observer_media:
            observer_serial = _resolve_observer_serial(
                cfg.observer_camera_name,
                serial_map,
                serial_cache=cfg.camera_serial_cache,
            )
            if observer_serial is not None:
                self.observer_camera = RealSenseCamera(
                    observer_serial,
                    width=int(os.getenv("SEER_CAMERA_WIDTH", "640")),
                    height=int(os.getenv("SEER_CAMERA_HEIGHT", "480")),
                    fps=int(os.getenv("SEER_CAMERA_FPS", "30")),
                )
        self.camera_serials = {
            "exterior": serial_map.get(cfg.exterior_camera_name),
            "wrist": serial_map.get(cfg.wrist_camera_name),
            "observer": observer_serial,
        }

        self._last_gripper_pos = None
        self._last_gripper_cmd_time = 0.0
        self._gripper_status = None  # 'open' or 'close'
        self._gripper_lock_counter = 0
        self._step_count = 0

    def close(self):
        for camera in (self.exterior_camera, self.wrist_camera, self.observer_camera):
            if camera is None:
                continue
            try:
                camera.close()
            except Exception:
                pass
        try:
            self.rtde_ctrl.servoStop()
        except Exception:
            pass
        try:
            self.rtde_ctrl.disconnect()
        except Exception:
            pass
        try:
            self.rtde_rec.disconnect()
        except Exception:
            pass

    def get_robot_state(self):
        tcp_pose = np.asarray(self.rtde_rec.getActualTCPPose(), dtype=np.float64)
        pose6d = _ur_tcp_to_pose6d(tcp_pose)
        pose = _6d_to_pose(pose6d)
        joints = np.asarray(self.rtde_rec.getActualQ(), dtype=np.float64)
        gripper_position = float(self.gripper.get_current_position()) / 255.0
        gripper_open_state = (
            1.0 if gripper_position <= self.cfg.gripper_open_threshold else -1.0
        )
        return {
            "pose": pose,
            "pose6d": pose6d.astype(np.float32),
            "gripper_open_state": np.array([gripper_open_state], dtype=np.float32),
            "gripper_position": np.array([gripper_position], dtype=np.float32),
            "joint_positions": joints.astype(np.float32),
        }

    def get_color_images(self):
        return [self.exterior_camera.read(), self.wrist_camera.read()]

    def _solve_ik(self, target_tcp):
        current_q = np.asarray(self.rtde_rec.getActualQ(), dtype=np.float64)
        if hasattr(self.rtde_ctrl, "getInverseKinematicsHasSolution"):
            try:
                if not self.rtde_ctrl.getInverseKinematicsHasSolution(target_tcp.tolist()):
                    return None
            except TypeError:
                if not self.rtde_ctrl.getInverseKinematicsHasSolution(target_tcp.tolist(), current_q.tolist()):
                    return None
        if not hasattr(self.rtde_ctrl, "getInverseKinematics"):
            raise RuntimeError(
                "RTDEControlInterface does not expose getInverseKinematics; "
                "pose deployment requires IK support or a custom Cartesian controller."
            )

        ik_attempts = (
            (target_tcp.tolist(), current_q.tolist()),
            (target_tcp.tolist(),),
        )
        for args in ik_attempts:
            try:
                result = self.rtde_ctrl.getInverseKinematics(*args)
            except TypeError:
                continue
            if result is not None:
                return np.asarray(result, dtype=np.float64)
        return None

    def _command_gripper(self, target_gripper):
        self._step_count += 1
        desired_status = 'open' if target_gripper > 0 else 'close'

        # Force gripper open during the first 10 steps
        if self._step_count <= 50:
            desired_status = 'open'

        # Decrement lock counter each step
        if self._gripper_lock_counter > 0:
            self._gripper_lock_counter -= 1

        # If status would change, only allow it when lock has expired
        if self._gripper_status is not None and desired_status != self._gripper_status:
            if self._gripper_lock_counter > 0:
                desired_status = self._gripper_status  # keep current status
            else:
                self._gripper_lock_counter = 10
                self._gripper_status = desired_status
        elif self._gripper_status is None:
            self._gripper_status = desired_status

        target_normalized = 0.0 if desired_status == 'open' else 1.0
        target_pos = int(np.clip(round(target_normalized * 255.0), 0, 255))
        now = time.time()
        if self._last_gripper_pos is not None:
            if abs(target_pos - self._last_gripper_pos) < self.cfg.gripper_min_delta:
                return
            if now - self._last_gripper_cmd_time < self.cfg.gripper_min_period_s:
                return
        self.gripper.move(target_pos, self.cfg.gripper_speed, self.cfg.gripper_force)
        self._last_gripper_pos = target_pos
        self._last_gripper_cmd_time = now

    def _command_gripper_normalized(self, normalized_gripper):
        target_normalized = float(np.clip(normalized_gripper, 0.0, 1.0))
        target_pos = int(np.clip(round(target_normalized * 255.0), 0, 255))
        now = time.time()
        if self._last_gripper_pos is not None:
            if abs(target_pos - self._last_gripper_pos) < self.cfg.gripper_min_delta:
                return
            if now - self._last_gripper_cmd_time < self.cfg.gripper_min_period_s:
                return
        self.gripper.move(target_pos, self.cfg.gripper_speed, self.cfg.gripper_force)
        self._last_gripper_pos = target_pos
        self._last_gripper_cmd_time = now
        threshold_pos = self.cfg.gripper_open_threshold * 255.0
        self._gripper_status = "open" if target_pos <= threshold_pos else "close"

    def move_to_home(self):
        if len(self.cfg.home_pose) < 6:
            raise ValueError("home_pose must contain at least 6 joint values")

        print("[deploy] moving robot to home pose before rollout")
        current_joints = np.asarray(self.rtde_rec.getActualQ(), dtype=np.float64)
        target_joints = np.asarray(self.cfg.home_pose[:6], dtype=np.float64)
        gripper_target = (
            float(np.clip(self.cfg.home_pose[6], 0.0, 1.0))
            if len(self.cfg.home_pose) > 6
            else float(self.gripper.get_current_position()) / 255.0
        )
        steps = max(1, int(self.cfg.home_move_duration * self.cfg.home_move_fps))

        for i in range(steps):
            loop_start = time.perf_counter()
            alpha = (i + 1) / steps
            interpolated_joints = current_joints + alpha * (target_joints - current_joints)

            t_start = self.rtde_ctrl.initPeriod()
            self.rtde_ctrl.servoJ(
                interpolated_joints.tolist(),
                self.cfg.arm_velocity,
                self.cfg.arm_acceleration,
                self.cfg.servoj_time,
                self.cfg.servoj_lookahead,
                self.cfg.servoj_gain,
            )
            self._command_gripper_normalized(gripper_target)
            self.rtde_ctrl.waitPeriod(t_start)

            sleep_left = (1.0 / self.cfg.home_move_fps) - (time.perf_counter() - loop_start)
            if sleep_left > 0:
                time.sleep(sleep_left)

        time.sleep(0.5)
        self._step_count = 0
        self._gripper_lock_counter = 0

    def step(self, target_pose6d, target_gripper):
        target_tcp = _pose6d_to_ur_tcp(target_pose6d)
        target_joints = self._solve_ik(target_tcp)
        if target_joints is None:
            raise RuntimeError(f"IK failed for target pose {target_pose6d.tolist()}")

        t_start = self.rtde_ctrl.initPeriod()
        self.rtde_ctrl.servoJ(
            target_joints.tolist(),
            self.cfg.arm_velocity,
            self.cfg.arm_acceleration,
            self.cfg.servoj_time,
            self.cfg.servoj_lookahead,
            self.cfg.servoj_gain,
        )
        self._command_gripper(float(target_gripper))
        self.rtde_ctrl.waitPeriod(t_start)


def main():
    cfg = DeployConfig()
    cfg.camera_serial_cache = _load_camera_serial_cache(_camera_serials_file(cfg))
    _prepare_observer_media_choice(cfg)
    controller = SeerController()
    env = UR5eDeployEnv(cfg)
    _save_camera_serial_cache(
        _camera_serials_file(cfg),
        env.deploy_camera_serials,
        [cfg.exterior_camera_name, cfg.wrist_camera_name],
    )
    listener = None
    deploy_results = []
    session_stamp = time.strftime("%Y%m%d_%H%M%S")
    session_started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    session_dir = _checkpoint_results_dir(cfg.results_dir, controller)
    results_file = os.path.join(session_dir, f"deploy_results_{session_stamp}.json")
    media_dir = os.path.join(session_dir, f"rollouts_{session_stamp}")

    try:
        listener, events = init_keyboard_listener()
        task_instructions = cfg.language_instructions or [cfg.language_instruction]
        rollout_steps = int(getattr(controller.args, "real_eval_max_steps", 600))
        frame_stride = max(int(getattr(controller.args, "eval_frame_stride", 1)), 1)
        frame_offset = int(getattr(controller.args, "eval_frame_offset", 0))
        skip_blend_ratio = float(getattr(controller.args, "skip_action_blend_ratio", 1.0))
        skip_blend_offset = int(getattr(controller.args, "skip_action_blend_offset", 0))
        skip_action_direct = bool(getattr(controller.args, "skip_action_direct", False))
        os.makedirs(session_dir, exist_ok=True)
        _write_instruction_markers(session_dir, task_instructions)
        if cfg.enable_rollout_media:
            os.makedirs(media_dir, exist_ok=True)

        print(f"[deploy] saving this checkpoint session under {session_dir}")
        print(f"[deploy] saving deploy results to {results_file}")
        if cfg.enable_rollout_media:
            print(f"[deploy] saving rollout media to {media_dir}")
        save_deploy_results(
            results_file,
            deploy_results,
            cfg=cfg,
            controller=controller,
            session_started_at=session_started_at,
            media_dir=media_dir if cfg.enable_rollout_media else None,
            camera_serials=env.camera_serials,
        )

        task_index = 0
        while task_index < len(task_instructions):
            instruction = task_instructions[task_index]
            print(f"[deploy] === Starting Task {task_index + 1}/{len(task_instructions)} ===")
            print(f"[deploy] instruction: {instruction}")

            rollout_index = 0
            while rollout_index < cfg.num_rollouts:
                _reset_control_events(events)
                print(
                    f"[deploy] --- Rollout {rollout_index + 1}/{cfg.num_rollouts} "
                    f"for Task {task_index + 1}/{len(task_instructions)} ---"
                )
                print(
                    "[deploy] Moving home. Reset the scene, then press RIGHT to start "
                    "this rollout. Press ESC to stop."
                )

                env.move_to_home()

                media_capture = None
                if cfg.enable_rollout_media:
                    media_capture = RolloutMediaCapture(
                        media_dir,
                        task_index + 1,
                        rollout_index + 1,
                        env.exterior_camera,
                        env.observer_camera,
                        wrist_camera=env.wrist_camera,
                        camera_serials=env.camera_serials,
                    )

                _clear_non_stop_events(events)
                while True:
                    if events["stop_deployment"]:
                        break
                    if events["next_task"]:
                        break
                    time.sleep(0.05)

                if events["stop_deployment"]:
                    if media_capture is not None:
                        media_capture.discard()
                    break

                if media_capture is not None:
                    try:
                        before_path = media_capture.capture_before_image()
                        if before_path is not None:
                            print(f"[deploy] saved B before-deploy image: {before_path}")
                    except Exception as exc:
                        print(f"[deploy] failed to save B before-deploy image: {exc}")
                    try:
                        start_path, video_path = media_capture.start_observer_recording()
                        if start_path is not None:
                            print(f"[deploy] saved C start image: {start_path}")
                        if video_path is not None:
                            print(f"[deploy] recording C rollout video: {video_path}")
                        for role, path in media_capture.video_paths.items():
                            if role == "observer":
                                continue
                            print(f"[deploy] recording {role} rollout video: {path}")
                    except Exception as exc:
                        print(f"[deploy] failed to start rollout video recording: {exc}")

                print(
                    "[deploy] Rollout started. Press UP for Success, DOWN for Failure, "
                    "LEFT to Retry, ESC to Stop."
                )

                for _ in range(cfg.warmup_steps):
                    obs = {
                        "robot_state": env.get_robot_state(),
                        "color_image": env.get_color_images(),
                        "language_instruction": instruction,
                    }
                    controller.forward(obs, include_info=True, timestep=0)

                controller.reset()
                timestep = 0
                rollout_start_wall = time.time()
                rollout_success = None
                retry_rollout = False
                robot_state = env.get_robot_state()
                last2robot_pose = robot_state["pose"]
                last_raw_action = None  # 7-dim normalized action from last inference

                while timestep < rollout_steps:
                    if events["stop_deployment"]:
                        break
                    if events["success_task"]:
                        rollout_success = True
                        print("[deploy] User marked SUCCESS.")
                        break
                    if events["failure_task"]:
                        rollout_success = False
                        print("[deploy] User marked FAILURE.")
                        break
                    if events["retry_task"]:
                        retry_rollout = True
                        print("[deploy] User requested RETRY for current rollout.")
                        break

                    _maybe_cuda_sync()
                    start_time = time.perf_counter()

                    obs = {
                        "robot_state": env.get_robot_state(),
                        "color_image": env.get_color_images(),
                        "language_instruction": instruction,
                    }
                    do_infer = (
                        timestep == 0
                        or frame_stride == 1
                        or last_raw_action is None
                        or (timestep % frame_stride) == frame_offset
                    )
                    if do_infer:
                        target_pos, target_euler, target_gripper, _ = controller.forward(
                            obs, include_info=True, timestep=timestep
                        )
                        last_raw_action = np.array(
                            [
                                target_pos[0], target_pos[1], target_pos[2],
                                target_euler[0], target_euler[1], target_euler[2],
                                float(target_gripper),
                            ],
                            dtype=np.float64,
                        )
                    else:
                        # frame-skip path: avoid running the model and either reuse
                        # the previous action, blend it with the ensembled action,
                        # or use the ensembled action directly.
                        action_vec = last_raw_action
                        if skip_action_direct and controller.use_ensembling:
                            a2 = controller.get_skip_action(timestep)
                            if a2 is not None:
                                action_vec = a2
                        elif skip_blend_ratio < 1.0 and controller.use_ensembling:
                            a2 = controller.get_skip_action(timestep + skip_blend_offset)
                            if a2 is not None:
                                action_vec = (
                                    skip_blend_ratio * last_raw_action
                                    + (1.0 - skip_blend_ratio) * a2
                                )
                        target_pos = action_vec[:3]
                        target_euler = action_vec[3:6]
                        target_gripper = float(action_vec[6])

                    target_pos = np.asarray(target_pos, dtype=np.float64) * cfg.max_rel_pos
                    target_euler = np.asarray(target_euler, dtype=np.float64) * cfg.max_rel_orn
                    cur2last_pose = _6d_to_pose(np.concatenate([target_pos, target_euler]))
                    last2robot_pose = last2robot_pose @ cur2last_pose
                    target_pose = pose_to_6d(last2robot_pose)

                    env.step(target_pose, float(target_gripper))
                    timestep += 1

                    _maybe_cuda_sync()
                    elapsed = time.perf_counter() - start_time
                    sleep_left = (1.0 / cfg.control_freq) - elapsed
                    if sleep_left > 0:
                        time.sleep(sleep_left)

                if events["stop_deployment"]:
                    if media_capture is not None:
                        media_capture.discard()
                    break

                if retry_rollout:
                    if media_capture is not None:
                        media_capture.discard()
                    continue

                if rollout_success is None:
                    print(
                        "[deploy] Rollout finished. Press UP for Success, DOWN for Failure, "
                        "LEFT to Retry, ESC to Stop."
                    )
                    _clear_non_stop_events(events)
                    while True:
                        if events["stop_deployment"]:
                            break
                        if events["success_task"]:
                            rollout_success = True
                            print("[deploy] User marked SUCCESS.")
                            break
                        if events["failure_task"]:
                            rollout_success = False
                            print("[deploy] User marked FAILURE.")
                            break
                        if events["retry_task"]:
                            retry_rollout = True
                            print("[deploy] User requested RETRY for current rollout.")
                            break
                        time.sleep(0.05)

                if events["stop_deployment"]:
                    if media_capture is not None:
                        media_capture.discard()
                    break

                if retry_rollout:
                    if media_capture is not None:
                        media_capture.discard()
                    continue

                media = {}
                if media_capture is not None:
                    media = media_capture.commit()

                result = DeployResult(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    task_index=task_index + 1,
                    rollout_index=rollout_index + 1,
                    instruction=instruction,
                    success=bool(rollout_success),
                    duration=time.time() - rollout_start_wall,
                    steps_completed=timestep,
                    total_steps=rollout_steps,
                    media=media,
                )
                deploy_results.append(result)
                save_deploy_results(
                    results_file,
                    deploy_results,
                    cfg=cfg,
                    controller=controller,
                    session_started_at=session_started_at,
                    media_dir=media_dir if cfg.enable_rollout_media else None,
                    camera_serials=env.camera_serials,
                )
                print(
                    f"[deploy] Task {task_index + 1} rollout {rollout_index + 1} "
                    f"saved: success={rollout_success}, steps={timestep}/{rollout_steps}"
                )

                env.move_to_home()
                rollout_index += 1

            if events["stop_deployment"]:
                break

            task_index += 1

        print(f"[deploy] Deployment session ended. Results saved to {results_file}")
    finally:
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        env.close()


if __name__ == "__main__":
    main()
