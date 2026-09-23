"""Exercise shell configuration and argv without importing robot/camera APIs."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from types import SimpleNamespace

from architectures.seer.adapters.latentloop_real_deploy.home_pose import configure_home_pose

REPO = Path(__file__).resolve().parents[1]
REAL = Path('architectures/seer/upstream/scripts/REAL')
SOURCE = REPO / REAL / 'deploy_ll_gui_unified.sh'
PRESETS = [f'{task}_{method}' for task in ('basketball','doll','cabinet')
           for method in ('baseline','latentloop')]
HOME_POSES = {
    'basketball': [3.14, -1.57, 1.57, -1.57, -1.57, -1.57, 0.0],
    'doll': [3.0502887, -1.6030570, 1.8191951, -1.8019783, -1.5417574, -1.6144441, 0.0],
    'cabinet': [2.9891653, -1.5753395, 1.8866094, -1.8454653, -1.5462163, -1.6641129, 0.0],
}
ENV_KEYS = ('SEER_LANGUAGE_INSTRUCTION','SEER_ROBOT_IP','SEER_EXTERIOR_CAMERA_SERIAL',
            'SEER_WRIST_CAMERA_SERIAL','SEER_CAMERA_WIDTH','SEER_CAMERA_HEIGHT','SEER_CAMERA_FPS',
            'SEER_CONTROL_FREQ','SEER_EXECUTION_MODE','SEER_MAX_REL_POS','SEER_MAX_REL_ORN',
            'SEER_NUM_ROLLOUTS','SEER_WARMUP_STEPS','SEER_ENABLE_ROLLOUT_MEDIA')


def selected(source, presets):
    value, count = re.subn(r'deploy_presets=\(.*?\n\)',
                          'deploy_presets=(\n'+'\n'.join(f'    "{p}"' for p in presets)+'\n)',
                          source, count=1, flags=re.S)
    assert count == 1
    return value


@pytest.fixture
def workspace(tmp_path):
    root=tmp_path/'repo'; folder=root/REAL; folder.mkdir(parents=True)
    bin_dir=tmp_path/'bin'; bin_dir.mkdir()
    (bin_dir/'git').write_text('#!/bin/sh\nprintf "test-git-state\\n"\n')
    (bin_dir/'git').chmod(0o755)
    (bin_dir/'torchrun').write_text(
        f'#!{sys.executable}\nimport json,os,sys\n'
        'from pathlib import Path\n'
        'Path(os.environ["TEST_CAPTURE"]).write_text(json.dumps({"args":sys.argv[1:],'
        '"env":{k:v for k,v in os.environ.items() if k.startswith("SEER_")}}))\n')
    (bin_dir/'torchrun').chmod(0o755)
    for task in ('basketball','doll','cabinet'):
        artifact=root/'artifacts/seer/real_world'/task
        artifact.mkdir(parents=True)
        manifest=json.loads((REPO/'artifacts/seer/real_world'/task/'checkpoint_manifest.json').read_text())
        (artifact/'checkpoint_manifest.json').write_text(json.dumps(manifest))
        (artifact/'shared').mkdir(); (artifact/'shared/mae_pretrain_vit_base.pth').write_bytes(b'test')
        (artifact/'baseline').mkdir()
        for teacher, spec in manifest['teachers'].items():
            (artifact/'baseline'/f'teacher_{teacher}.pth').write_bytes(b'test')
            target=artifact/'latentloop'/f'teacher_{teacher}'; target.mkdir(parents=True)
            for adapter in spec['adapters']:
                (target/f'teacher_{teacher}_adapter_{adapter}.pth').write_bytes(b'test')
    env=dict(os.environ,PATH=f'{bin_dir}:{Path(sys.executable).parent}:{os.environ["PATH"]}',
             CONDA_DEFAULT_ENV='seer',TEST_CAPTURE=str(tmp_path/'capture.json'))
    return root,folder,env


def run(workspace,source,args=('--print-config',)):
    _,folder,env=workspace
    path=folder/'test_launcher.sh'; path.write_text(source)
    return subprocess.run(['bash',str(path),*args],env=env,text=True,capture_output=True)


@pytest.mark.parametrize('preset',PRESETS)
def test_all_presets_resolve_without_hardware(workspace,preset):
    result=run(workspace,selected(SOURCE.read_text(),[preset]))
    assert result.returncode==0,result.stderr
    config=dict(line.split('=',1) for line in result.stdout.splitlines())
    task,method=preset.rsplit('_',1)
    assert config['task_name']==task and config['deployment_method']==method
    assert config['teacher_id']==('37' if task=='basketball' else '38')
    assert config['query_interval']==('1' if method=='baseline' else '4')
    assert json.loads(config['home_pose_json']) == HOME_POSES[task]
    assert config['results_root'].endswith(f'/real_deploy_results_v2/{method}/{task}')
    if method=='baseline': assert config['adapter_checkpoint']=='not_loaded'
    assert not Path(workspace[2]['TEST_CAPTURE']).exists()


@pytest.mark.parametrize('preset',PRESETS)
def test_argv_and_robot_settings_preserve_original_seer_v2(workspace,preset):
    task,method=preset.rsplit('_',1)
    legacy_name='deploy_ll_gui_v2.sh' if task=='basketball' else f'deploy_ll_gui_{task}_v2.sh'
    legacy=(REPO/'tests/fixtures/seer/deploy'/f'{legacy_name}.txt').read_text()
    legacy=re.sub(r'(?m)^deployment_method="[^"]+"$',f'deployment_method="{method}"',legacy)
    original=run(workspace,legacy,('--preflight',))
    assert original.returncode==0,original.stderr
    before=json.loads(Path(workspace[2]['TEST_CAPTURE']).read_text())
    updated=run(workspace,selected(SOURCE.read_text(),[preset]),('--preflight',))
    assert updated.returncode==0,updated.stderr
    after=json.loads(Path(workspace[2]['TEST_CAPTURE']).read_text())
    cfg = SimpleNamespace(home_move_duration=3.0, home_move_fps=60)
    manifest = workspace[0]/'artifacts/seer/real_world'/task/'checkpoint_manifest.json'
    home = configure_home_pose(cfg, manifest, after['env'])
    assert cfg.home_pose == HOME_POSES[task]
    assert home['source'] == after['env']['SEER_HOME_POSE_SOURCE']
    def normalize(args):
        result=list(args)
        index=result.index('--v2-profile-output-dir')+1
        result[index]='<unique-launch-log-dir>'
        return result
    assert normalize(before['args'])==normalize(after['args'])
    assert {k:before['env'][k] for k in ENV_KEYS}=={k:after['env'][k] for k in ENV_KEYS}
    assert '--latentloop-preflight-only' in after['args']
    assert ('--latentloop-adapter-checkpoint' in after['args'])==(method=='latentloop')


@pytest.mark.parametrize('presets',[[],['doll_baseline','cabinet_latentloop'],['fruit_latentloop']])
def test_ambiguous_or_unavailable_presets_fail_before_launch(workspace,presets):
    result=run(workspace,selected(SOURCE.read_text(),presets))
    assert result.returncode==2
    assert not Path(workspace[2]['TEST_CAPTURE']).exists()


def test_wrong_task_manifest_rejected_before_model_load(workspace):
    path=workspace[0]/'artifacts/seer/real_world/doll/checkpoint_manifest.json'
    manifest=json.loads(path.read_text()); manifest['task']='cabinet_filtered_40p'
    path.write_text(json.dumps(manifest))
    result=run(workspace,selected(SOURCE.read_text(),['doll_latentloop']),('--preflight',))
    assert result.returncode!=0 and 'Wrong task manifest' in result.stderr
    assert not Path(workspace[2]['TEST_CAPTURE']).exists()


def test_baseline_does_not_require_adapter_files(workspace):
    artifact=workspace[0]/'artifacts/seer/real_world/doll'
    for path in (artifact/'latentloop').rglob('*.pth'): path.unlink()
    result=run(workspace,selected(SOURCE.read_text(),['doll_baseline']),('--preflight',))
    assert result.returncode==0,result.stderr
    assert '--latentloop-adapter-checkpoint' not in json.loads(Path(workspace[2]['TEST_CAPTURE']).read_text())['args']


@pytest.mark.parametrize('policy',['hold_action','hold_latent'])
def test_baseline_ablation_keeps_selected_query_interval(workspace,policy):
    source=selected(SOURCE.read_text(),['doll_baseline']).replace('baseline_rollout_policy="full"',f'baseline_rollout_policy="{policy}"')
    result=run(workspace,source)
    assert result.returncode==0
    assert 'query_interval=4' in result.stdout and f'rollout_policy={policy}' in result.stdout


def test_execution_and_preflight_are_shell_owned(workspace):
    source=selected(SOURCE.read_text(),['doll_latentloop']).replace('preflight_only=0','preflight_only=1')
    result=run(workspace,source,args=())
    assert result.returncode==0,result.stderr
    assert 'robot_motion_commands_enabled=0' in result.stdout
    assert '--latentloop-preflight-only' in json.loads(Path(workspace[2]['TEST_CAPTURE']).read_text())['args']
    assert '--execution-mode' not in source and 'control_hz' not in source
