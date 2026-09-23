"""Presentation-only layout for the existing Seer deployment controller."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont, messagebox, ttk

from architectures.seer.adapters.latentloop_real_deploy.reference_browser import (
    ReferenceImagePanel, _Tooltip, _button_icon,
)

FIELD = "#f4f5f6"
WHITE = "#ffffff"
INK = "#24272b"
MUTED = "#626970"
LINE = "#dde0e3"
GREEN = "#23664e"
RED = "#b33d46"
SPACE = 4
GAP = 2 * SPACE
SECTION = 4 * SPACE
PAGE = 5 * SPACE


class DeploymentViewMixin:
    """Build widgets only; rollout, model, camera and home routines are inherited."""

    def _label(self, parent, text=None, variable=None, muted=False, **kwargs):
        return tk.Label(parent, text=text, textvariable=variable,
                        bg=parent.cget("bg"), fg=MUTED if muted else INK,
                        font=self.font_body, anchor="w", **kwargs)

    def _button(self, parent, text, command, style="Deploy.TButton", tooltip=None):
        button = ttk.Button(parent, text=text, command=command, style=style)
        icon = {"Start": "play", "Stop": "square", "Success": "check", "Failure": "x",
                "Retry": "rotate-ccw", "Save": "save", "Delete": "trash-2", "Exit": "log-out"}.get(text.split()[0])
        if icon:
            _button_icon(button, icon, light=style in ("Start.Deploy.TButton", "Stop.Deploy.TButton"))
        if tooltip:
            self._view_tooltips.append(_Tooltip(button, tooltip))
        return button

    def _build_ui(self):
        root = self.root
        root.title("Seer | Real-world deployment")
        root.geometry("1440x960")
        root.minsize(1100, 760)
        root.configure(bg=FIELD)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        family = "DejaVu Sans"
        self.font_title = tkfont.Font(root=root, family=family, size=17, weight="bold")
        self.font_section = tkfont.Font(root=root, family=family, size=11, weight="bold")
        self.font_body = tkfont.Font(root=root, family=family, size=11)
        self.font_small = tkfont.Font(root=root, family=family, size=10)
        self.font_metric = tkfont.Font(root=root, family=family, size=24, weight="bold")
        root.option_add("*Font", self.font_body)
        root.option_add("*TCombobox*Listbox.font", self.font_body)
        style = ttk.Style(root)
        style.theme_use("clam")
        # Remove inherited 3-D borders, while retaining native focus/keyboard behavior.
        style.layout("TButton", [("Button.border", {"sticky": "nswe", "children": [
            ("Button.focus", {"sticky": "nswe", "children": [
                ("Button.padding", {"sticky": "nswe", "children": [
                    ("Button.label", {"sticky": "nswe"})]})]})]})])
        for name in ("TButton", "Deploy.TButton"):
            style.configure(name, font=self.font_body, padding=(12, 8),
                            background=WHITE, foreground=INK, bordercolor=LINE,
                            lightcolor=WHITE, darkcolor=WHITE, borderwidth=1,
                            relief="flat", focusthickness=1, focuscolor=GREEN)
            style.map(name, background=[("disabled", FIELD), ("pressed", "#e2e5e8"), ("active", "#eceef0")],
                      foreground=[("disabled", "#8a9096")], bordercolor=[("focus", GREEN)])
        for name, color, active in (("Start", GREEN, "#125139"), ("Stop", RED, "#922733")):
            style.configure(f"{name}.Deploy.TButton", background=color, foreground=WHITE,
                            bordercolor=color, padding=(16, 10), font=self.font_section)
            style.map(f"{name}.Deploy.TButton",
                      background=[("disabled", FIELD), ("active", active)],
                      bordercolor=[("disabled", LINE), ("!disabled", color)],
                      foreground=[("disabled", "#8a9096"), ("!disabled", WHITE)])
        style.configure("Quiet.Deploy.TButton", borderwidth=0, background=WHITE)
        style.configure("Record.Deploy.TButton", font=self.font_small, padding=(6, 7), width=0)
        style.configure("TCombobox", padding=6, font=self.font_body, arrowsize=16,
                        fieldbackground=WHITE, foreground=INK, bordercolor=LINE, background=WHITE, arrowcolor=MUTED,
                        lightcolor=WHITE, darkcolor=WHITE, borderwidth=1)
        style.map("TCombobox", fieldbackground=[("readonly", WHITE), ("disabled", FIELD)],
                  foreground=[("disabled", MUTED)])
        style.configure("TSpinbox", padding=6, font=self.font_body, arrowsize=16,
                        fieldbackground=WHITE, foreground=INK, bordercolor=LINE, background=WHITE, arrowcolor=MUTED,
                        lightcolor=WHITE, darkcolor=WHITE, borderwidth=1)
        style.map("TSpinbox", bordercolor=[("focus", GREEN)],
                  fieldbackground=[("disabled", FIELD)])
        style.configure("Vertical.TScrollbar", background=LINE, troughcolor=WHITE,
                        bordercolor=WHITE, lightcolor=LINE, darkcolor=LINE,
                        arrowsize=12, relief="flat", borderwidth=0)
        style.configure("TNotebook", background=WHITE, borderwidth=0)
        style.configure("Flat.TNotebook", background=WHITE, bordercolor=WHITE,
                        lightcolor=WHITE, darkcolor=WHITE, borderwidth=0)
        style.layout("Flat.TNotebook.Tab", [])
        style.layout("Flat.TRadiobutton", [("Radiobutton.padding", {"sticky": "nswe", "children": [
            ("Radiobutton.focus", {"sticky": "nswe", "children": [
                ("Radiobutton.label", {"sticky": "nswe"})]})]})])
        style.configure("Flat.TRadiobutton", padding=(10, 8), font=self.font_small,
                        background=WHITE, foreground=MUTED, focuscolor=GREEN)
        style.map("Flat.TRadiobutton", background=[("selected", "#e8efeb"), ("active", FIELD)],
                  foreground=[("selected", GREEN)])
        style.layout("TNotebook.Tab", [("Notebook.tab", {"sticky": "nswe", "children": [
            ("Notebook.padding", {"sticky": "nswe", "children": [
                ("Notebook.label", {"sticky": "nswe"})]})]})])
        style.configure("TNotebook.Tab", padding=(12, 8), font=self.font_small,
                        borderwidth=0, lightcolor=WHITE, darkcolor=WHITE)
        style.map("TNotebook.Tab", background=[("selected", "#e8efeb"), ("!selected", WHITE)],
                  foreground=[("selected", GREEN), ("!selected", MUTED)])
        self._view_tooltips = []
        meta = self.controller.deployment_metadata()
        task = Path(self.gui_args.latentloop_artifact_manifest).parent.name.replace("_", " ").title()
        header = tk.Frame(root, bg=WHITE, padx=PAGE, pady=GAP)
        header.pack(fill="x")
        heading = self._label(header, task)
        heading.configure(font=self.font_title)
        heading.pack(side="left")
        tk.Frame(header, bg=LINE, width=1, height=24).pack(side="left", padx=SECTION)
        self._label(header, meta['method']).pack(side="left")
        self.state_badge = tk.Label(header, textvariable=self.state_text, bg=GREEN, fg=WHITE,
                                    font=self.font_section, padx=14, pady=7)
        self.exit_button = self._button(header, "Exit [Esc]", self.on_close, "Quiet.Deploy.TButton")
        self.exit_button.pack(side="right", padx=(SECTION, 0))
        self.state_badge.pack(side="right")

        instruction = tk.Frame(root, bg=WHITE, padx=PAGE)
        instruction.pack(fill="x")
        self.instruction_label = self._label(instruction,
            self.task_instructions[self.task_index], justify="left", wraplength=1000)
        self.instruction_label.configure(fg=MUTED)
        self.instruction_label.pack(fill="x", pady=(0, GAP))
        instruction.bind("<Configure>", lambda event: self.instruction_label.configure(
            wraplength=max(200, event.width - 40)))
        tk.Frame(root, bg=LINE, height=1).pack(fill="x")
        self._build_run_controls(root)

        footer = tk.Frame(root, bg=WHITE, height=28, padx=PAGE)
        footer.pack(side="bottom", fill="x")
        footer.pack_propagate(False)
        self._view_status = tk.StringVar(root, "Ready")
        status_label = self._label(footer, variable=self._view_status, muted=True)
        status_label.pack(side="left", fill="x", expand=True)
        self._view_tooltips.append(_Tooltip(status_label, lambda: self.status_text.get()))

        body = tk.Frame(root, bg=FIELD)
        body.pack(fill="both", expand=True, padx=PAGE, pady=GAP)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, minsize=360)
        body.rowconfigure(0, weight=1)
        visual = tk.Frame(body, bg=WHITE, padx=SECTION, pady=SECTION)
        visual.grid(row=0, column=0, sticky="nsew", padx=(0, SECTION))
        visual.columnconfigure(0, weight=1)
        visual.rowconfigure(1, weight=1)
        visual.rowconfigure(2, weight=1)
        self._label(visual, "Live cameras").grid(row=0, column=0, sticky="w", pady=(0, GAP))
        self._build_live_views(visual)
        self.reference_panel = ReferenceImagePanel(visual, self.gui_args.latentloop_artifact_manifest,
                                                   self.cfg.language_instruction)
        self.reference_panel.grid(row=2, column=0, sticky="nsew", pady=(SECTION, 0))
        sidebar = tk.Frame(body, bg=WHITE, width=360, padx=SECTION, pady=SECTION)
        sidebar.grid(row=0, column=1, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(2, weight=1)
        self._build_live_metrics(sidebar)
        self._build_runtime_fields(sidebar)
        self._build_session_tabs(sidebar)
        self._bind_shortcuts()
        self.update_metrics()

    def _build_live_views(self, parent):
        board = tk.Frame(parent, bg=WHITE)
        board.grid(row=1, column=0, sticky="nsew")
        self.camera_cards = []
        for index, (name, _camera, serial) in enumerate(self.cameras):
            row, col = divmod(index, 2)
            board.columnconfigure(col, weight=1, uniform="live")
            board.rowconfigure(row, weight=1)
            card = tk.Frame(board, bg=WHITE)
            card.grid(row=row, column=col, sticky="nsew", padx=(0, SPACE) if col == 0 else (SPACE, 0))
            card.columnconfigure(0, weight=1)
            card.rowconfigure(1, weight=1)
            role = "Primary" if name == self.cfg.exterior_camera_name else (
                "Wrist" if name == self.cfg.wrist_camera_name else name.title())
            caption = self._label(card, role, muted=True)
            caption.grid(row=0, column=0, sticky="w", pady=(0, SPACE))
            self._view_tooltips.append(_Tooltip(caption, f"Camera serial: {serial or 'unknown'}"))
            viewport = tk.Frame(card, bg="#1f2327", height=174, width=200)
            viewport.grid(row=1, column=0, sticky="nsew")
            viewport.grid_propagate(False)
            viewport.rowconfigure(0, weight=1)
            viewport.columnconfigure(0, weight=1)
            label = tk.Label(viewport, text="Waiting for frame", bg="#1f2327", fg="#bdc3c8",
                             font=self.font_small, width=1, height=1, wraplength=240)
            label.grid(sticky="nsew")
            self.camera_labels[name] = label
            self.camera_cards.append((card, name))

    def _build_run_controls(self, parent):
        section = tk.Frame(parent, bg=WHITE, padx=PAGE, pady=GAP)
        section.pack(fill="x")
        # Related commands stay together even when the window gets wider.
        self.start_button = self._button(section, "Start rollout  [N]", self.start_rollout,
            "Start.Deploy.TButton", "Start rollout [N]; moves to task home first")
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, GAP))
        self.stop_button = self._button(section, "Stop & home  [X]", lambda: self.signal_current("stop"),
            "Stop.Deploy.TButton", "Stop and discard active rollout, then return home [X]. Not a hardware emergency stop.")
        self.stop_button.grid(row=0, column=1, sticky="ew")
        tk.Frame(section, bg=LINE, width=1).grid(row=0, column=2, sticky="ns", padx=SECTION, pady=SPACE)
        outcomes = tk.Frame(section, bg=WHITE)
        outcomes.grid(row=0, column=3, sticky="ew")
        for col in range(3): outcomes.columnconfigure(col, weight=1, uniform="outcome")
        self.success_button = self._button(outcomes, "Success [S]", lambda: self.signal_current("success"),
                                          tooltip="Record success [S]")
        self.success_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.failure_button = self._button(outcomes, "Failure [F]", lambda: self.signal_current("failure"),
                                          tooltip="Record failure [F]")
        self.failure_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.retry_button = self._button(outcomes, "Retry [R]", self.restart_rollout,
                                         tooltip="Discard active rollout and retry [R]")
        self.retry_button.grid(row=0, column=2, sticky="ew", padx=(4, 0))

    def _build_runtime_fields(self, parent):
        settings = tk.Frame(parent, bg=WHITE)
        settings.grid(row=1, sticky="ew", pady=GAP)
        for col in (0, 1): settings.columnconfigure(col, weight=1, uniform="runtime")
        tk.Frame(settings, bg=LINE, height=1).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, GAP))
        settings_header = tk.Frame(settings, bg=WHITE)
        settings_header.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, GAP))
        self._label(settings_header, "Next rollout").pack(side="left")
        self._settings_feedback_var = tk.StringVar(self.root, "Applied")
        self.settings_feedback = self._label(settings_header, variable=self._settings_feedback_var, muted=True)
        self.settings_feedback.configure(font=self.font_small)
        self.settings_feedback.pack(side="left", padx=GAP)
        self.apply_button = self._button(settings_header, "Apply", self.apply_runtime_settings, "Record.Deploy.TButton")
        self.apply_button.pack(side="right")
        self._label(settings, "Target rate (Hz)").grid(row=2, column=0, sticky="w")
        self._label(settings, "Refresh interval K").grid(row=2, column=1, sticky="w", padx=(GAP, 0))
        self.runtime_control_freq_input = ttk.Spinbox(settings, from_=1, to=60,
            textvariable=self.runtime_control_freq_var, width=8)
        self.runtime_control_freq_input.grid(row=3, column=0, sticky="ew", padx=(0, SPACE), pady=(SPACE, GAP))
        self.runtime_query_interval_input = ttk.Spinbox(settings, from_=1, to=100,
            textvariable=self.runtime_query_interval_var, width=8)
        self.runtime_query_interval_input.grid(row=3, column=1, sticky="ew", padx=(SPACE, 0), pady=(SPACE, GAP))
        self.runtime_rollout_policy_input = ttk.Combobox(settings, textvariable=self.runtime_rollout_policy_var,
            values=("full", "hold_action", "hold_latent") if self.controller.deployment_method == "baseline"
            else ("latentloop",), state="readonly", width=12)
        if self.controller.deployment_method == "baseline":
            self.runtime_rollout_policy_input.grid(row=4, column=0, sticky="ew", padx=(0, SPACE))
        self.runtime_rollout_policy_input.bind("<<ComboboxSelected>>", self._on_rollout_policy_changed)
        self._view_tooltips.append(_Tooltip(self.runtime_control_freq_input,
            "Requested command frequency for the next rollout. Achieved control is measured separately."))
        self._view_tooltips.append(_Tooltip(self.runtime_query_interval_input,
            "Full policy once every K control steps. Intermediate steps use the selected policy."))
        self._sync_query_interval_state()
        for variable in (self.runtime_control_freq_var, self.runtime_query_interval_var, self.runtime_rollout_policy_var):
            variable.trace_add("write", self._update_settings_feedback)
        self._update_settings_feedback()

    def _update_settings_feedback(self, *_args):
        try:
            requested = (float(self.runtime_control_freq_var.get()), int(self.runtime_query_interval_var.get()),
                         self.runtime_rollout_policy_var.get())
            applied = (self.controller.control_freq, self.controller.query_interval, self.controller.rollout_policy)
            self._settings_feedback_var.set("Applied" if requested == applied else "Not applied")
        except ValueError:
            self._settings_feedback_var.set("Check values")

    def _build_live_metrics(self, parent):
        section = tk.Frame(parent, bg=WHITE)
        section.grid(row=0, sticky="ew")
        self.metrics_scope_var = tk.StringVar(self.root, "Rollout / No measurements")
        self._label(section, variable=self.metrics_scope_var).pack(anchor="w", pady=(0, GAP))
        self._live_values = {}
        self.metric_labels = []
        headline = tk.Frame(section, bg=WHITE)
        headline.pack(fill="x")
        for col, (key, title, unit) in enumerate((("hz", "Achieved rate", "Hz"), ("ms", "Policy latency", "ms"))):
            headline.columnconfigure(col, weight=1, uniform="metrics")
            cell = tk.Frame(headline, bg=WHITE)
            cell.grid(row=0, column=col, sticky="ew")
            caption = self._label(cell, title)
            caption.pack(anchor="w")
            self.metric_labels.append(caption)
            var = tk.StringVar(self.root, "--")
            self._live_values[key] = var
            number = tk.Frame(cell, bg=WHITE)
            number.pack(anchor="w", pady=(SPACE, 0))
            label = self._label(number, variable=var)
            label.configure(font=self.font_metric)
            label.pack(side="left")
            self._label(number, unit, muted=True).pack(side="left", anchor="s", padx=(SPACE, 0), pady=(0, GAP))
        breakdown = tk.Frame(section, bg=WHITE)
        breakdown.pack(fill="x", pady=(GAP, 0))
        for col, (key, title) in enumerate((("span", "Active (s)"), ("steps", "Steps"), ("calls", "Full / Fast"))):
            breakdown.columnconfigure(col, weight=1, uniform="breakdown")
            cell = tk.Frame(breakdown, bg=WHITE)
            cell.grid(row=0, column=col, sticky="ew")
            caption = self._label(cell, title, muted=True)
            caption.configure(font=self.font_small)
            caption.pack(anchor="w")
            self.metric_labels.append(caption)
            self._live_values[key] = tk.StringVar(self.root, "--")
            value = self._label(cell, variable=self._live_values[key])
            value.pack(anchor="w", pady=(SPACE, 0))
        meanings = (
            "All command intervals in this rollout: (commands - 1) / elapsed time. Not the robot's internal servo frequency.",
            "Mean controller latency over this rollout, including image preprocessing and action postprocessing; excludes camera capture and robot I/O.",
            "Elapsed time between the first and latest completed command. Excludes initial home movement and warmup.",
            "Policy steps recorded for this rollout, excluding warmup.",
            "Full policy calls / intermediate calls in this rollout. Baseline Full has no intermediate calls.",
        )
        for label, text in zip(self.metric_labels, meanings):
            self._view_tooltips.append(_Tooltip(label, text))

    def _build_session_tabs(self, parent):
        panel = tk.Frame(parent, bg=WHITE, height=1)
        panel.grid(row=2, sticky="nsew")
        panel.grid_propagate(False)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        selectors = tk.Frame(panel, bg=WHITE)
        selectors.grid(row=0, sticky="ew")
        tk.Frame(selectors, bg=LINE, height=1).pack(fill="x", pady=(0, GAP))
        tab_row = tk.Frame(selectors, bg=WHITE)
        tab_row.pack(fill="x")
        self.tabs = ttk.Notebook(panel, style="Flat.TNotebook", height=1)
        self.tabs.grid(row=1, sticky="nsew")
        session = tk.Frame(self.tabs, bg=WHITE, pady=SPACE)
        notes = tk.Frame(self.tabs, bg=WHITE, pady=GAP)
        details = tk.Frame(self.tabs, bg=WHITE, pady=8)
        self.tabs.add(session, text="Rollouts")
        self.tabs.add(notes, text="Notes")
        self.tabs.add(details, text="Details")
        self._tab_selection = tk.IntVar(self.root, 0)
        self.tab_buttons = []
        for index, label in enumerate(("Rollouts", "Notes", "Details")):
            button = ttk.Radiobutton(tab_row, text=label, value=index,
                variable=self._tab_selection, style="Flat.TRadiobutton",
                command=lambda i=index: self.tabs.select(i))
            button.pack(side="left", padx=(0, SPACE))
            self.tab_buttons.append(button)
        self.tabs.bind('<<NotebookTabChanged>>', lambda _event: self._tab_selection.set(self.tabs.index('current')))
        session.columnconfigure(0, weight=1)
        session.rowconfigure(0, weight=1)
        columns = ("run", "k", "result", "steps", "duration")
        self.results_table = ttk.Treeview(session, columns=columns, show="headings", height=3)
        ttk.Style(self.root).configure("Treeview", font=self.font_small, rowheight=28,
                                       background=WHITE, fieldbackground=WHITE, foreground=INK,
                                       borderwidth=0, lightcolor=WHITE, darkcolor=WHITE)
        ttk.Style(self.root).layout("Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
        ttk.Style(self.root).map("Treeview", background=[("selected", "#e8efeb")],
                                foreground=[("selected", INK)])
        ttk.Style(self.root).configure("Treeview.Heading", font=self.font_small,
            background=FIELD, foreground=INK, relief="flat", borderwidth=0,
            lightcolor=FIELD, darkcolor=FIELD)
        for key, title, width in zip(columns, ("Run", "K", "Result", "Steps", "Total (s)"),
                                      (32, 26, 68, 60, 72)):
            self.results_table.heading(key, text=title)
            self.results_table.column(key, width=width, minwidth=width, anchor="center", stretch=True)
        self.results_table.tag_configure("alternate", background="#f6f7f8")
        self.results_table.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(session, command=self.results_table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.results_table.configure(yscrollcommand=scroll.set)
        self._view_tooltips.append(_Tooltip(self.results_table,
            "Saved outcomes only. Total (s) includes initial home, warmup and media finalization; Active control above measures command execution."))
        self._last_results = None
        notes.columnconfigure(0, weight=1)
        notes.rowconfigure(1, weight=1)
        self._label(notes, "Session notes / Autosaved").grid(row=0, sticky="w", pady=(0, GAP))
        self.notes_text = tk.Text(notes, height=3, width=20, wrap="word", relief="flat",
            highlightthickness=1, highlightbackground=LINE, highlightcolor=GREEN,
            bg=FIELD, fg=INK, insertbackground=INK, padx=9, pady=8, font=self.font_body)
        self.notes_text.grid(row=1, sticky="nsew")
        self.notes_text.insert("1.0", Path(self.notes_file).read_text() if Path(self.notes_file).is_file() else "")
        self.notes_text.bind("<KeyRelease>", lambda _event: self.schedule_notes_save())
        self.notes_text.bind("<Control-Return>", lambda _event: self.release_notes_focus())
        details.columnconfigure(0, weight=1)
        details.rowconfigure(0, weight=1)
        self.details_text = tk.Text(details, height=5, width=20, wrap="word", relief="flat",
            bg=WHITE, fg=MUTED, font=self.font_small, padx=4, pady=4)
        self.details_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(details, command=self.details_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.details_text.configure(yscrollcommand=scroll.set, state="disabled")
        self._last_details = None
        self._build_record_controls(panel)

    def _build_record_controls(self, parent):
        records = tk.Frame(parent, bg=WHITE)
        records.grid(row=2, column=0, sticky="ew", pady=(GAP, 0))
        session = tk.Frame(records, bg=WHITE)
        session.pack(fill="x", pady=(0, GAP))
        self._label(session, "Saved successes", muted=True).pack(side="left")
        self.saved_success_var = tk.StringVar(self.root, "--")
        self._label(session, variable=self.saved_success_var).pack(side="right")
        actions = tk.Frame(records, bg=WHITE)
        actions.pack(anchor="w")
        self.save_button = self._button(actions, "Save [W]", self.save_results, "Record.Deploy.TButton",
            tooltip="Save this session's results [W]")
        self.delete_button = self._button(actions, "Delete [D]", self.delete_previous_rollout,
            "Record.Deploy.TButton",
            tooltip="Delete the last saved result and its media; confirmation required.")
        self.discard_button = self._button(actions, "Discard all", self.discard_current_deploy,
            "Record.Deploy.TButton",
            tooltip="Delete this session's results, notes and media; confirmation required.")
        self.save_button.pack(side="left")
        tk.Frame(actions, bg=LINE, width=1, height=24).pack(side="left", padx=SPACE)
        self.delete_button.pack(side="left")
        self.discard_button.pack(side="left", padx=(SPACE, 0))

    def update_metrics(self):
        total = len(self.deploy_results)
        success = sum(bool(result.success) for result in self.deploy_results)
        for var, value in ((self.rollouts_count_var, total), (self.success_count_var, success),
                           (self.failure_count_var, total-success)):
            var.set(str(value))
        self.success_rate_var.set(f"{100 * success / total:.1f}%" if total else "--")
        if not hasattr(self, "_live_values"):
            return
        # The displayed measurement scope is one complete rollout, not a rolling window.
        records = list(getattr(self.controller, "step_records", []))
        times = list(getattr(self.controller, "control_command_monotonic_s", []))
        hz = (len(times)-1) / (times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else None
        self._live_values["steps"].set(str(records[-1]["timestep"]+1) if records else "--")
        self._live_values["hz"].set(f"{hz:.1f}" if hz is not None else "--")
        self._live_values["ms"].set(f"{sum(r['policy_ms'] for r in records)/len(records):.1f}" if records else "--")
        self._live_values["span"].set(f"{times[-1]-times[0]:.1f}" if len(times)>1 else "--")
        full = sum(r.get('mode') == 'full' for r in records)
        fast = sum(r.get('mode') in ('latentloop', 'hold_latent', 'hold_action') for r in records)
        self._live_values["calls"].set(f"{full} / {fast}" if records else "--")
        self.saved_success_var.set(f"{success} / {total}" if total else "--")
        self._live_values["sr"] = self.saved_success_var
        if records:
            phase = "Running" if self.state_text.get() == 'RUNNING' else "Last rollout"
            self.metrics_scope_var.set(f"{phase}  /  K={records[-1].get('query_interval', self.controller.query_interval)}")
        else:
            self.metrics_scope_var.set("Rollout / No measurements")
        active = self._rollout_is_active()
        self.start_button.configure(state="disabled" if active else "normal")
        self.stop_button.configure(state="normal" if active else "disabled")
        for button in (self.success_button, self.failure_button):
            button.configure(state="normal" if active and self.state_text.get() in ("RUNNING", "WAITING FOR OUTCOME") else "disabled")
        self.apply_button.configure(state="disabled" if active else "normal")
        self.runtime_control_freq_input.configure(state="disabled" if active else "normal")
        self.runtime_query_interval_input.configure(state="disabled" if active or self.runtime_rollout_policy_var.get()=="full" else "normal")
        self.runtime_rollout_policy_input.configure(state="readonly" if not active and self.controller.deployment_method=="baseline" else "disabled")
        self.delete_button.configure(state="disabled" if active or not total else "normal")
        self.discard_button.configure(state="disabled" if active else "normal")
        self._update_settings_feedback()
        settings = getattr(self, 'runtime_settings_by_rollout', {})
        rows = [(i, settings.get(str(i), {}).get('query_interval', '--'),
                 'Success' if r.success else 'Failure', r.steps_completed, f'{r.duration:.1f}')
                for i, r in enumerate(self.deploy_results, 1)]
        if rows != self._last_results:
            self.results_table.delete(*self.results_table.get_children())
            for index, row in enumerate(rows):
                self.results_table.insert('', 'end', values=row, tags=('alternate',) if index % 2 == 0 else ())
            self._last_results = rows
        message = self.status_text.get()
        if message.startswith("Saved results:"):
            message = "Results saved"
        elif message.startswith("Applied target:"):
            message = "Settings applied"
        self._view_status.set(message[:110])
        meta = self.controller.deployment_metadata()
        latest = self.deploy_results[-1] if self.deploy_results else None
        latest_summary = ({"success": bool(latest.success), "steps": latest.steps_completed,
                           "duration_s": latest.duration, "media": latest.media} if latest else None)
        details = json.dumps({"latest_rollout": latest_summary, "deployment": meta,
                              "home": self.cfg.home_configuration, "results_file": self.results_file,
                              "camera_serials": self.env.camera_serials}, indent=2)
        if details != self._last_details:
            self.details_text.configure(state="normal")
            self.details_text.delete("1.0", "end")
            self.details_text.insert("1.0", details)
            self.details_text.configure(state="disabled")
            self._last_details = details

    def set_run_state(self, message, color=GREEN):
        colors = {"READY TO START": GREEN, "RUNNING": GREEN, "MOVING HOME": "#8a6518",
                  "WAITING FOR OUTCOME": "#8a6518", "STOPPING": RED, "ERROR": RED}
        return super().set_run_state(message, colors.get(message, color))

    def delete_previous_rollout(self):
        if self._rollout_is_active():
            return super().delete_previous_rollout()
        if self.deploy_results and messagebox.askyesno("Delete previous rollout?",
                "The saved result and its media will be deleted.", parent=self.root):
            return super().delete_previous_rollout()

    def discard_current_deploy(self):
        if self._rollout_is_active():
            self.set_status("Finish or stop the active rollout before discarding the session.")
            return
        if messagebox.askyesno("Discard session?", "All results, notes and media in this session will be deleted.", parent=self.root):
            return super().discard_current_deploy()

    def _handle_keypress(self, event):
        panel = getattr(self, "reference_panel", None)
        if panel is not None and str(event.widget).startswith(str(panel)+".") and event.keysym.lower()=="x":
            return super()._handle_keypress(event)
        if isinstance(event.widget, (tk.Entry, tk.Text, tk.Spinbox, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return
        return super()._handle_keypress(event)
