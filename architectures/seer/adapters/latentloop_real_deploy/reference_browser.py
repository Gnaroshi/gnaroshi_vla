"""Read-only training-start gallery; independent of policy and robot state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


def _button_icon(button, name, light=False):
    """Keep local Lucide PNGs alive and use a disabled tone without new dependencies."""
    folder = Path(__file__).with_name('assets') / 'icons'
    normal = tk.PhotoImage(master=button, file=str(folder / f'{name}_{"white" if light else "ink"}.png'))
    disabled = tk.PhotoImage(master=button, file=str(folder / f'{name}_muted.png'))
    button._icon_images = (normal, disabled)
    button.configure(image=(normal, 'disabled', disabled), compound='left')


class ReferenceCatalog:
    def __init__(self, artifact_manifest, instruction=None):
        artifact_manifest = Path(artifact_manifest)
        checkpoint = json.loads(artifact_manifest.read_text())
        self.root = artifact_manifest.parent / "reference_start_frames_all"
        self.manifest_path = self.root / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("task") != checkpoint["task"]:
            raise ValueError("Reference task does not match the selected checkpoint")
        if instruction and manifest.get("instruction") != instruction:
            raise ValueError("Reference instruction does not match the selected task")
        self.samples = manifest["samples"]
        if not self.samples or len(self.samples) != manifest["episode_count"]:
            raise ValueError("Incomplete reference episode collection")
        identities = [sample["dataset_episode"] for sample in self.samples]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate reference episodes")
        for sample in self.samples:
            if sample["step"] != "0000":
                raise ValueError("Reference is not the start of an episode")
            for camera in ("primary", "wrist"):
                path = self.image_path(sample, camera)
                if hashlib.sha256(path.read_bytes()).hexdigest() != sample[camera]["sha256"]:
                    raise ValueError(f"Reference checksum mismatch: {path.name}")
        self.task = artifact_manifest.parent.name.replace("_", " ").title()
        self.index = 0
        # Never consume Python/NumPy/Torch RNG used by the policy.
        self._random = random.SystemRandom()

    def image_path(self, sample, camera):
        filename = sample[camera]["filename"]
        path = (self.root / filename).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Reference image must be inside its task directory")
        return path

    @property
    def current(self):
        return self.samples[self.index]

    def select(self, index):
        if not 0 <= index < len(self.samples):
            raise IndexError(index)
        self.index = index

    def move(self, offset):
        self.index = (self.index + offset) % len(self.samples)

    def randomize(self):
        if len(self.samples) > 1:
            self.move(self._random.randrange(1, len(self.samples)))

    def load_pair(self):
        images = []
        for camera in ("primary", "wrist"):
            with Image.open(self.image_path(self.current, camera)) as image:
                images.append(image.convert("RGB"))
        return images


class ReferenceImagePanel(tk.Frame):
    def __init__(self, parent, artifact_manifest, instruction=None):
        super().__init__(parent, bg="#ffffff")
        self.catalog = None
        self.images = []
        self.zoom_window = None
        self._resize_after = None
        self._zoom_after = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        try:
            self.catalog = ReferenceCatalog(artifact_manifest, instruction)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.error_label = tk.Label(
                self, text=f"Reference images unavailable\n{exc}", justify="left",
                anchor="w", bg="#ffffff", fg="#a33b32", wraplength=340,
            )
            self.error_label.grid(sticky="ew")
            return

        heading = tk.Frame(self, bg="#ffffff")
        heading.grid(row=0, sticky="ew", pady=(0, 8))
        tk.Frame(heading, bg="#dde0e3", height=1).pack(fill="x", pady=(0, 8))
        toolbar = tk.Frame(heading, bg="#ffffff")
        toolbar.pack(fill="x")
        tk.Label(toolbar, text="Training start", bg="#ffffff", fg="#24272b", anchor="w").pack(side="left", padx=(0, 16))
        controls = tk.Frame(toolbar, bg="#ffffff")
        controls.pack(side="left")
        controls.columnconfigure(5, weight=1)
        self.previous_button = ttk.Button(controls, text="<", width=3,
                                          command=lambda: self.move(-1))
        self.previous_button.grid(row=0, column=0, padx=(0, 4))
        values = [f"{i+1:02d} / {len(self.catalog.samples)}"
                  for i in range(len(self.catalog.samples))]
        self.selection_var = tk.StringVar(master=self)
        self.episode_selector = ttk.Combobox(controls, values=values, state="readonly", width=10,
                                           textvariable=self.selection_var)
        self.episode_selector.grid(row=0, column=1, sticky="ew")
        self.episode_selector.bind("<<ComboboxSelected>>", self._selected)
        self.next_button = ttk.Button(controls, text=">", width=3,
                                      command=lambda: self.move(1))
        self.next_button.grid(row=0, column=2, padx=4)
        self.random_button = ttk.Button(controls, text="Random", width=7, command=self.randomize)
        self.random_button.grid(row=0, column=3)
        self.zoom_button = ttk.Button(controls, text="Zoom", width=6, command=self.open_zoom)
        self.zoom_button.grid(row=0, column=4, padx=(4, 0))
        for button, icon in ((self.previous_button, 'chevron-left'),
                             (self.next_button, 'chevron-right'),
                             (self.random_button, 'shuffle'), (self.zoom_button, 'expand')):
            _button_icon(button, icon)
            button.configure(text='', width=0, padding=(8, 6))
        self._tooltips = [
            _Tooltip(self.previous_button, "Previous episode"),
            _Tooltip(self.next_button, "Next episode"),
            _Tooltip(self.random_button, "Random episode"),
            _Tooltip(self.zoom_button, "Enlarge both camera views"),
        ]

        self.image_board = tk.Frame(self, bg="#ffffff", height=174)
        self.image_board.grid(row=2, sticky="nsew")
        self.image_board.grid_propagate(False)
        self.image_board.rowconfigure(1, weight=1)
        self.image_labels = []
        for column, name in enumerate(("Primary", "Wrist")):
            self.image_board.columnconfigure(column, weight=1, uniform="reference")
            tk.Label(self.image_board, text=name, bg="#ffffff", fg="#59665f", anchor="w").grid(
                row=0, column=column, sticky="w", pady=(0, 5))
            label = tk.Label(self.image_board, bg="#edf0f2", cursor="hand2", width=1, height=1)
            label.grid(row=1, column=column, sticky="nsew", padx=(0, 4) if column == 0 else (4, 0))
            label.bind("<Button-1>", lambda _event: self.open_zoom())
            self.image_labels.append(label)
        self.details_var = tk.StringVar(master=self)
        self.details_label = tk.Label(self, textvariable=self.details_var, bg="#ffffff",
                                     fg="#59665f", font=("DejaVu Sans", 10), anchor="w", justify="left")
        self.details_label.grid(row=3, sticky="ew", pady=(5, 0))
        self.image_board.bind("<Configure>", self._resize)
        self.bind("<Destroy>", self._destroyed, add="+")
        self._show()

    def _selected(self, _event=None):
        self.catalog.select(self.episode_selector.current())
        self._show()

    def move(self, offset):
        self.catalog.move(offset)
        self._show()

    def randomize(self):
        self.catalog.randomize()
        self._show()

    def _show(self):
        self.episode_selector.current(self.catalog.index)
        sample = self.catalog.current
        self.details_var.set(f"Episode {sample['original_episode']}  /  Frame {sample['step']}")
        try:
            self.images = self.catalog.load_pair()
        except (OSError, ValueError) as exc:
            self.images = []
            self.details_var.set(f"Image unavailable: {exc}")
        self._paint()
        self._paint_zoom()

    @staticmethod
    def _paint_pair(labels, images):
        for index, label in enumerate(labels):
            if not images:
                label.configure(image="", text="Image unavailable")
                label.image = None
                continue
            image = images[index].copy()
            image.thumbnail((max(1, label.winfo_width()-4), max(1, label.winfo_height()-4)),
                            Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image, master=label)
            label.configure(image=photo, text="")
            label.image = photo

    def _paint(self):
        if self._resize_after:
            self.after_cancel(self._resize_after)
        self._resize_after = None
        self._paint_pair(self.image_labels, self.images)
        self.details_label.configure(wraplength=max(180, self.winfo_width()-24))

    def _resize(self, _event=None):
        if self._resize_after:
            self.after_cancel(self._resize_after)
        self._resize_after = self.after(60, self._paint)

    def open_zoom(self):
        if self.zoom_window is not None and self.zoom_window.winfo_exists():
            self.zoom_window.lift()
            return
        window = self.zoom_window = tk.Toplevel(self)
        window.title(f"{self.catalog.task} | Training start reference")
        window.geometry("1100x540")
        window.minsize(620, 380)
        window.configure(bg="#ffffff")
        window.rowconfigure(1, weight=1)
        self.zoom_labels = []
        for column, name in enumerate(("Primary", "Wrist")):
            window.columnconfigure(column, weight=1, uniform="zoom")
            tk.Label(window, text=name, bg="#ffffff").grid(row=0, column=column, pady=6)
            label = tk.Label(window, bg="#eef1f2")
            label.grid(row=1, column=column, sticky="nsew", padx=6)
            self.zoom_labels.append(label)
        controls = tk.Frame(window, bg="#ffffff")
        controls.grid(row=2, column=0, columnspan=2, pady=8)
        ttk.Button(controls, text="<", command=lambda: self.move(-1)).pack(side="left")
        ttk.Label(controls, textvariable=self.selection_var).pack(side="left", padx=15)
        ttk.Button(controls, text=">", command=lambda: self.move(1)).pack(side="left")
        ttk.Button(controls, text="Random", command=self.randomize).pack(side="left", padx=8)
        window.bind("<Configure>", self._resize_zoom)
        window.bind("<Escape>", lambda _event: (window.destroy(), "break")[1])
        self._resize_zoom()

    def _resize_zoom(self, _event=None):
        if self._zoom_after:
            self.after_cancel(self._zoom_after)
        self._zoom_after = self.after(60, self._paint_zoom)

    def _paint_zoom(self):
        if self._zoom_after:
            self.after_cancel(self._zoom_after)
        self._zoom_after = None
        if self.zoom_window is not None and self.zoom_window.winfo_exists():
            self.zoom_window.title(
                f"{self.catalog.task} | Episode {self.catalog.index+1}/{len(self.catalog.samples)}"
            )
            self._paint_pair(self.zoom_labels, self.images)

    def _destroyed(self, event):
        if event.widget is self:
            for timer in (self._resize_after, self._zoom_after):
                if timer:
                    self.after_cancel(timer)


class _Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text, self.window, self.timer = widget, text, None, None
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")
        widget.bind("<Destroy>", self.hide, add="+")

    def schedule(self, _event=None):
        self.timer = self.widget.after(500, self.show)

    def show(self):
        self.timer = None
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.geometry(f"+{self.widget.winfo_rootx()}+{self.widget.winfo_rooty()+30}")
        tk.Label(self.window, text=self.text() if callable(self.text) else self.text,
                 bg="#202924", fg="white", padx=9, pady=6, wraplength=360, justify="left").pack()

    def hide(self, _event=None):
        if self.timer:
            self.widget.after_cancel(self.timer)
            self.timer = None
        if self.window is not None:
            self.window.destroy()
            self.window = None
