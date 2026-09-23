import hashlib
import json
import os
from pathlib import Path
import random
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import pytest

from architectures.seer.adapters.latentloop_real_deploy.reference_browser import (
    ReferenceCatalog, ReferenceImagePanel,
)


@pytest.fixture
def artifact(tmp_path):
    root = tmp_path / "doll"
    folder = root / "reference_start_frames_all"
    folder.mkdir(parents=True)
    path = root / "checkpoint_manifest.json"
    path.write_text(json.dumps({"task": "doll_filtered_40p"}))
    samples = []
    for index in range(3):
        sample = {"dataset_episode": f"0000/{index:06d}", "step": "0000",
                  "original_episode": f"recording_{index}"}
        for camera in ("primary", "wrist"):
            image = folder / f"{index}_{camera}.jpg"
            Image.new("RGB", (640, 480), (index*60, 60, 150)).save(image)
            sample[camera] = {"filename": image.name,
                              "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
        samples.append(sample)
    (folder / "manifest.json").write_text(json.dumps({
        "task": "doll_filtered_40p", "instruction": "Doll instruction",
        "episode_count": len(samples), "samples": samples,
    }))
    return path


def test_catalog_all_starts_navigation_and_pair(artifact):
    catalog = ReferenceCatalog(artifact, "Doll instruction")
    assert len(catalog.samples) == 3
    catalog.move(-1)
    assert catalog.index == 2
    catalog.move(1)
    assert catalog.index == 0
    catalog.select(1)
    assert catalog.current["dataset_episode"] == "0000/000001"
    assert [im.size for im in catalog.load_pair()] == [(640, 480)]*2
    with pytest.raises(IndexError):
        catalog.select(3)


def test_random_changes_episode_without_changing_policy_rng(artifact):
    catalog = ReferenceCatalog(artifact)
    state = random.getstate()
    for _ in range(50):
        previous = catalog.index
        catalog.randomize()
        assert catalog.index != previous
    assert random.getstate() == state


@pytest.mark.parametrize("mutation", ["wrong_task", "duplicate", "count", "nonstart", "escape", "checksum"])
def test_corrupt_or_wrong_collection_is_rejected(artifact, mutation):
    path = artifact.parent / "reference_start_frames_all/manifest.json"
    content = json.loads(path.read_text())
    if mutation == "wrong_task": content["task"] = "cabinet_filtered_40p"
    if mutation == "duplicate": content["samples"][1]["dataset_episode"] = content["samples"][0]["dataset_episode"]
    if mutation == "count": content["episode_count"] = 40
    if mutation == "nonstart": content["samples"][0]["step"] = "0010"
    if mutation == "escape": content["samples"][0]["primary"]["filename"] = "../checkpoint_manifest.json"
    if mutation == "checksum": content["samples"][0]["wrist"]["sha256"] = "wrong"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        ReferenceCatalog(artifact)


def test_wrong_instruction_is_rejected(artifact):
    with pytest.raises(ValueError, match="instruction"):
        ReferenceCatalog(artifact, "Cabinet instruction")


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Tk requires a virtual display")
def test_gui_navigation_zoom_resize_and_unavailable(artifact):
    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *args: errors.append(args)
    try:
        root.geometry("460x340")
        panel = ReferenceImagePanel(root, artifact, "Doll instruction")
        panel.pack(fill="both", expand=True)
        root.update()
        panel._paint()
        assert panel.episode_selector.current() == 0
        assert all(label.image is not None for label in panel.image_labels)
        panel.previous_button.invoke()
        assert panel.catalog.index == 2
        panel.next_button.invoke()
        assert panel.catalog.index == 0
        panel.episode_selector.current(1)
        panel._selected()
        assert panel.catalog.index == 1
        panel.random_button.invoke()
        assert panel.catalog.index != 1
        panel.zoom_button.invoke()
        root.update()
        panel._paint_zoom()
        assert all(label.image is not None for label in panel.zoom_labels)
        panel.move(1)
        root.update()
        assert f"{panel.catalog.index+1}/3" in panel.zoom_window.title()
        panel.zoom_window.destroy()
        panel.destroy()
        missing = ReferenceImagePanel(root, artifact.parent / "missing.json")
        missing.pack()
        root.update()
        assert missing.catalog is None and "unavailable" in missing.error_label.cget("text")
        assert not errors
    finally:
        root.destroy()


def test_reference_focus_does_not_trigger_rollout_but_keeps_stop():
    from architectures.seer.adapters.latentloop_real_deploy.deploy_ll_gui_v2 import (
        LatentLoopDeployGuiAppV2, legacy_gui,
    )
    app = object.__new__(LatentLoopDeployGuiAppV2)
    app.reference_panel = ".reference"
    with patch.object(legacy_gui.DeployGuiApp, "_handle_keypress") as handler:
        app._handle_keypress(SimpleNamespace(widget=".reference.selector", keysym="r"))
        handler.assert_not_called()
        app._handle_keypress(SimpleNamespace(widget=".reference.selector", keysym="x"))
        assert handler.call_count == 1
        app._handle_keypress(SimpleNamespace(widget=".controls", keysym="n"))
        assert handler.call_count == 2
