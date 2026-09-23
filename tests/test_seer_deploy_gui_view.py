import os
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools.seer.preview_real_deploy_ui import build_preview, paint_preview
from architectures.seer.adapters.latentloop_real_deploy.deploy_ll_gui_v2 import legacy_gui
from architectures.seer.adapters.latentloop_real_deploy.gui_view import DeploymentViewMixin

pytestmark = pytest.mark.skipif(not os.environ.get('DISPLAY'), reason='Requires virtual display')


@pytest.fixture
def view():
    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    app = build_preview(root)
    yield app
    root.destroy()
    app.preview_storage.cleanup()
    assert not errors


@pytest.mark.parametrize('size', ['1100x760', '1280x800', '1440x960', '1920x1080'])
def test_layout_and_images_fit_without_scroll(view, size):
    view.root.geometry(size)
    paint_preview(view)
    widgets = [view.start_button, view.stop_button,
               view.success_button, view.failure_button, view.retry_button,
               view.save_button, view.exit_button, view.delete_button, view.discard_button,
               view.reference_panel.details_label, view.results_table,
               *view.camera_labels.values(), *view.reference_panel.image_labels]
    width, height = view.root.winfo_width(), view.root.winfo_height()
    for widget in widgets:
        x = widget.winfo_rootx() - view.root.winfo_rootx()
        y = widget.winfo_rooty() - view.root.winfo_rooty()
        assert x >= 0 and y >= 0, str(widget)
        assert x + widget.winfo_width() <= width and y + widget.winfo_height() <= height, str(widget)
        # A widget may be inside the root yet clipped by an undersized parent.
        assert widget.winfo_y() + widget.winfo_height() <= widget.master.winfo_height(), str(widget)
        assert widget.winfo_x() + widget.winfo_width() <= widget.master.winfo_width(), str(widget)
        assert widget.winfo_height() >= 20, str(widget)
    for label in (*view.camera_labels.values(), *view.reference_panel.image_labels):
        assert label.image is not None and label.image.width() > 100


def test_settings_are_visible_without_changing_tabs_on_small_display(view):
    view.root.geometry('1100x760')
    view.root.update()
    for widget in (view.runtime_control_freq_input, view.runtime_query_interval_input,
                   view.apply_button):
        assert widget.winfo_ismapped()
        assert widget.winfo_height() >= 24
        assert widget.winfo_y() + widget.winfo_height() <= widget.master.winfo_height()
    view.apply_button.invoke()
    view.apply_runtime_settings.assert_called_once()
    assert not view.runtime_rollout_policy_input.winfo_ismapped()


def test_controls_keep_existing_commands(view):
    view.start_button.invoke()
    view.start_rollout.assert_called_once()
    view.rollout_thread = SimpleNamespace(is_alive=lambda: True)
    view.state_text.set('RUNNING')
    view.update_metrics()
    assert view.start_button.instate(['disabled'])
    assert view.apply_button.instate(['disabled'])
    assert view.runtime_query_interval_input.instate(['disabled'])
    view.stop_button.invoke()
    view.signal_current.assert_called_with('stop')
    view.success_button.invoke()
    view.signal_current.assert_called_with('success')
    view.failure_button.invoke()
    view.signal_current.assert_called_with('failure')
    view.retry_button.invoke()
    view.restart_rollout.assert_called_once()


def test_metrics_read_existing_records_without_changing_them(view):
    view.controller.step_records = [{'timestep': i, 'policy_ms': float(i)} for i in range(25)]
    view.controller.control_command_monotonic_s = [i / 10 for i in range(26)]
    before = list(view.controller.step_records)
    view.update_metrics()
    assert view._live_values['steps'].get() == '25'
    assert view._live_values['hz'].get() == '10.0'
    assert view._live_values['ms'].get() == '12.0'
    assert view._live_values['span'].get() == '2.5'
    assert view.controller.step_records == before


def test_form_input_does_not_start_rollout_from_typing(view):
    with patch.object(legacy_gui.DeployGuiApp, '_handle_keypress') as legacy:
        view._handle_keypress(SimpleNamespace(widget=view.runtime_rollout_policy_input, keysym='r'))
        legacy.assert_not_called()
        view._handle_keypress(SimpleNamespace(widget=view.reference_panel.episode_selector, keysym='x'))
        legacy.assert_called_once()


def test_destructive_actions_require_confirmation(view):
    view.deploy_results = [SimpleNamespace(success=True)]
    with patch('architectures.seer.adapters.latentloop_real_deploy.gui_view.messagebox.askyesno',return_value=False), \
            patch.object(legacy_gui.DeployGuiApp,'delete_previous_rollout') as delete, \
            patch.object(legacy_gui.DeployGuiApp,'discard_current_deploy') as discard:
        DeploymentViewMixin.delete_previous_rollout(view)
        DeploymentViewMixin.discard_current_deploy(view)
        delete.assert_not_called()
        discard.assert_not_called()


def test_every_original_command_is_visible_with_its_shortcut(view):
    for button, shortcut in ((view.start_button, '[N]'), (view.stop_button, '[X]'),
                            (view.success_button, '[S]'), (view.failure_button, '[F]'),
                            (view.retry_button, '[R]'), (view.save_button, '[W]'),
                            (view.delete_button, '[D]'), (view.exit_button, '[Esc]')):
        assert shortcut in button.cget('text')
        assert button.winfo_ismapped()
    assert view.discard_button.winfo_ismapped()


def test_all_metrics_share_rollout_scope_and_preserve_inputs(view):
    view.controller.step_records = [dict(timestep=i, policy_ms=40.0,
        mode='full' if i % 4 == 0 else 'latentloop', query_interval=4) for i in range(40)]
    view.controller.control_command_monotonic_s = [i / 10 for i in range(40)]
    view.update_metrics()
    assert view._live_values['calls'].get() == '10 / 30'
    assert view._live_values['ms'].get() == '40.0'
    assert view._live_values['span'].get() == '3.9'
    view.controller.step_records = []
    view.controller.control_command_monotonic_s = []
    view.update_metrics()
    for key in ('calls', 'ms', 'hz', 'span', 'steps'):
        assert view._live_values[key].get() == '--'


def test_saved_result_table_shows_runtime_k_and_does_not_invent_latency(view):
    view.deploy_results = [SimpleNamespace(success=True, steps_completed=400, duration=42.0, media={})]
    view.runtime_settings_by_rollout = {'1': {'query_interval': 8}}
    view.update_metrics()
    row = view.results_table.item(view.results_table.get_children()[0])['values']
    assert row == [1, 8, 'Success', 400, '42.0']
    assert view._live_values['sr'].get() == '1 / 1'


def test_metric_captions_fit_and_do_not_overlap(view):
    view.root.geometry('1100x760')
    view.root.update()
    for label in view.metric_labels:
        font = tk.font.Font(font=label.cget('font'))
        assert font.measure(label.cget('text')) <= label.winfo_width()
    for index, left in enumerate(view.metric_labels):
        for right in view.metric_labels[index+1:]:
            assert (left.winfo_rootx() + left.winfo_width() <= right.winfo_rootx()
                    or right.winfo_rootx() + right.winfo_width() <= left.winfo_rootx()
                    or left.winfo_rooty() + left.winfo_height() <= right.winfo_rooty()
                    or right.winfo_rooty() + right.winfo_height() <= left.winfo_rooty())


def test_tab_selectors_keep_native_notebook_pages_and_keyboard(view):
    for index, button in enumerate(view.tab_buttons):
        button.invoke()
        view.root.update()
        assert view.tabs.index('current') == index
        assert view._tab_selection.get() == index
    view.tabs.select(0)
    view.root.update()
    assert view._tab_selection.get() == 0


def test_lucide_assets_are_loaded_and_reference_navigation_is_functional(view):
    for button in (view.start_button, view.stop_button, view.save_button,
                   view.reference_panel.previous_button, view.reference_panel.next_button,
                   view.reference_panel.random_button, view.reference_panel.zoom_button):
        assert len(button._icon_images) == 2
        assert button._icon_images[0].width() == 18
    original = view.reference_panel.catalog.index
    view.reference_panel.next_button.invoke()
    assert view.reference_panel.catalog.index == (original+1) % len(view.reference_panel.catalog.samples)
    view.reference_panel.previous_button.invoke()
    assert view.reference_panel.catalog.index == original


@pytest.mark.parametrize('size', ['1100x760', '1440x960', '1920x1080'])
def test_related_buttons_stay_close_at_every_window_size(view, size):
    view.root.geometry(size)
    view.root.update()
    for left, right in zip((view.start_button, view.stop_button, view.success_button, view.failure_button),
                           (view.stop_button, view.success_button, view.failure_button, view.retry_button)):
        gap = right.winfo_rootx() - (left.winfo_rootx() + left.winfo_width())
        assert 0 <= gap <= 33
        assert abs(left.winfo_rooty() - right.winfo_rooty()) <= 4
    for left, right in ((view.save_button, view.delete_button), (view.delete_button, view.discard_button)):
        gap = right.winfo_rootx() - (left.winfo_rootx() + left.winfo_width())
        assert 0 <= gap <= 12
    table_bottom = view.results_table.winfo_rooty() + view.results_table.winfo_height()
    assert 0 <= view.save_button.winfo_rooty() - table_bottom <= 64
    assert view.results_table.winfo_height() >= 78
    for button in (view.save_button, view.delete_button, view.discard_button):
        style = tk.ttk.Style(view.root)
        font = tk.font.Font(font=style.lookup(button.cget('style'), 'font'))
        content_width = font.measure(button.cget('text')) + (18 if getattr(button, '_icon_images', None) else 0)
        assert button.winfo_width() >= content_width + 12


def test_settings_feedback_tracks_edits_without_changing_controller(view):
    assert view._settings_feedback_var.get() == 'Applied'
    view.runtime_control_freq_var.set('30')
    assert view._settings_feedback_var.get() == 'Not applied'
    assert view.controller.control_freq == 60
    view.runtime_control_freq_var.set('')
    assert view._settings_feedback_var.get() == 'Check values'
    view.runtime_control_freq_var.set('60')
    assert view._settings_feedback_var.get() == 'Applied'
    view.runtime_query_interval_var.set('8')
    assert view._settings_feedback_var.get() == 'Not applied'
    assert view.controller.query_interval == 4


def test_baseline_keeps_its_ablation_policy_selector():
    root = tk.Tk()
    app = build_preview(root, method='baseline')
    try:
        assert app.runtime_rollout_policy_input.winfo_ismapped()
        assert tuple(app.runtime_rollout_policy_input.cget('values')) == ('full', 'hold_action', 'hold_latent')
        assert app.runtime_query_interval_input.instate(['disabled'])
        app.runtime_rollout_policy_var.set('hold_latent')
        app._on_rollout_policy_changed()
        assert not app.runtime_query_interval_input.instate(['disabled'])
    finally:
        root.destroy()
        app.preview_storage.cleanup()


@pytest.mark.parametrize('key,method', [('n','start_rollout'),('r','restart_rollout'),('w','save_results')])
def test_actual_keyboard_shortcuts_dispatch_to_existing_actions(view, key, method):
    view.root.focus_force()
    view.root.update()
    view.root.event_generate('<KeyPress>', keysym=key)
    view.root.update()
    getattr(view, method).assert_called_once()
