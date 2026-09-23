"""Render the Seer GUI without connecting hardware or loading model weights."""

import argparse
import json
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import ImageGrab

from architectures.seer.adapters.latentloop_real_deploy.deploy_ll_gui_v2 import LatentLoopDeployGuiAppV2
from architectures.seer.adapters.latentloop_real_deploy.reference_browser import ReferenceCatalog

REPO = Path(__file__).resolve().parents[2]


def build_preview(root, task="doll", method="latentloop"):
    """Return the actual view with an inert controller and real reference frames."""
    artifact = REPO / 'artifacts/seer/real_world' / task
    catalog = ReferenceCatalog(artifact / 'checkpoint_manifest.json')
    instruction = json.loads(catalog.manifest_path.read_text())['instruction']
    app = object.__new__(LatentLoopDeployGuiAppV2)
    app.root = root
    app.ui_thread_id = threading.get_ident()
    app.controller = SimpleNamespace(
        deployment_method=method, control_freq=60, query_interval=4 if method=='latentloop' else 1,
        rollout_policy='latentloop' if method=='latentloop' else 'full',
        step_records=[], control_command_monotonic_s=[],
    )
    meta = {'method':'LatentLoop' if method=='latentloop' else 'Seer baseline',
            'deployment_method':method, 'teacher_id':37 if task=='basketball' else 38,
            'adapter_id':39 if method=='latentloop' else None,
            'teacher_checkpoint':str(artifact/'baseline/teacher_38.pth'),
            'adapter_checkpoint':str(artifact/'latentloop/teacher_38/teacher_38_adapter_39.pth'),
            'deployment_profile':'hardware_free_preview'}
    app.controller.deployment_metadata = lambda: dict(meta)
    app.cfg = SimpleNamespace(language_instruction=instruction, home_configuration={},
                              exterior_camera_name='primary', wrist_camera_name='wrist')
    app.gui_args = SimpleNamespace(latentloop_artifact_manifest=str(artifact/'checkpoint_manifest.json'))
    app.task_instructions = [instruction]
    app.task_index = 0
    app.rollout_thread = None
    app.deploy_results = []
    app.cameras = [('primary',None,'preview'),('wrist',None,'preview')]
    app.camera_labels = {}
    app.camera_images = {}
    app.env = SimpleNamespace(camera_serials={'primary':'preview','wrist':'preview'})
    app.preview_storage = tempfile.TemporaryDirectory(prefix='seer_gui_preview_')
    app.notes_file = str(Path(app.preview_storage.name)/'NOTES.txt')
    app.results_file = str(Path(app.preview_storage.name)/'deploy_results.json')
    values = {'status_text':'Preview only: images are training references; hardware is disconnected.',
              'state_text':'PREVIEW ONLY','rollouts_count_var':'0','success_count_var':'0',
              'failure_count_var':'0','success_rate_var':'--','metrics_var':'',
              'runtime_control_freq_var':'60', 'runtime_query_interval_var':str(app.controller.query_interval),
              'runtime_rollout_policy_var':app.controller.rollout_policy,
              'runtime_settings_status_var':f'Applied: 60 Hz / K={app.controller.query_interval}'}
    for key,value in values.items(): setattr(app,key,tk.StringVar(root,value))
    app.notes_save_after = None
    app.on_close = root.destroy
    # No command in the preview can reach a hardware or deployment lifecycle method.
    for name in ('start_rollout','restart_rollout','signal_current','apply_runtime_settings',
                 'save_results','delete_previous_rollout','discard_current_deploy'):
        setattr(app,name,Mock(name=name))
    app._build_ui()
    app.state_badge.configure(bg='#59665f')
    root.update()
    app.preview_frames = dict(zip(('primary','wrist'),[np.asarray(im) for im in catalog.load_pair()]))
    return app


def paint_preview(app):
    app.root.update()
    for name,label in app.camera_labels.items():
        photo = app._frame_to_photo(app.preview_frames[name],label)
        label.configure(image=photo,text='')
        label.image=photo
    app.reference_panel._paint()
    app.root.update()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',default='1440x960')
    parser.add_argument('--task',default='doll',choices=['doll','cabinet','basketball'])
    parser.add_argument('--method',default='latentloop',choices=['latentloop','baseline'])
    args=parser.parse_args()
    root=tk.Tk()
    app=build_preview(root,args.task,args.method)
    root.geometry(args.size+'+0+0')
    root.after(350,lambda: capture(root,app,args.output))
    root.mainloop()
    app.preview_storage.cleanup()


def capture(root,app,output):
    paint_preview(app)
    output.parent.mkdir(parents=True,exist_ok=True)
    ImageGrab.grab(bbox=(root.winfo_rootx(),root.winfo_rooty(),
                        root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())).save(output)
    print(output)
    root.destroy()


if __name__=='__main__': main()
