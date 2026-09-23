"""CPU-only task home tests: never construct a robot or camera interface."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from architectures.seer.adapters.latentloop_real_deploy.home_pose import configure_home_pose

REPO = Path(__file__).resolve().parents[1]
POSE = [3.0502887, -1.6030570, 1.8191951, -1.8019783, -1.5417574, -1.6144441, 0.0]


@pytest.fixture
def settings(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'task': 'doll_filtered_40p'}))
    env = {
        'SEER_HOME_POSE': json.dumps(POSE),
        'SEER_HOME_TASK': 'doll_filtered_40p',
        'SEER_HOME_POSE_SOURCE': 'reference:TASK_SPECS[3]',
    }
    cfg = SimpleNamespace(home_pose=[0] * 7, home_move_duration=3, home_move_fps=60)
    return cfg, path, env


@pytest.mark.parametrize('key', ['SEER_HOME_POSE', 'SEER_HOME_TASK', 'SEER_HOME_POSE_SOURCE'])
def test_missing_home_setting_rejected(settings, key):
    cfg, path, env = settings
    del env[key]
    with pytest.raises(ValueError, match='unified'):
        configure_home_pose(cfg, path, env)
    assert cfg.home_pose == [0] * 7


@pytest.mark.parametrize('value', [
    'not json', 'null', '{}', '[0,0,0,0,0,0]', '[0,0,0,0,0,0,0,0]',
    '[true,0,0,0,0,0,0]', '["3.1",0,0,0,0,0,0]',
    '[NaN,0,0,0,0,0,0]', '[0,0,Infinity,0,0,0,0]',
    '[0,0,0,0,0,0,-0.1]', '[0,0,0,0,0,0,1.1]',
])
def test_invalid_home_pose_rejected(settings, value):
    cfg, path, env = settings
    env['SEER_HOME_POSE'] = value
    with pytest.raises(ValueError):
        configure_home_pose(cfg, path, env)
    assert cfg.home_pose == [0] * 7


def test_task_mismatch_rejected(settings):
    cfg, path, env = settings
    env['SEER_HOME_TASK'] = 'cabinet_filtered_40p'
    with pytest.raises(ValueError, match='differs'):
        configure_home_pose(cfg, path, env)


def test_actual_home_function_sends_configured_joints_and_open_gripper(settings):
    cfg, path, env = settings
    configure_home_pose(cfg, path, env)
    cfg.arm_velocity = 0.5
    cfg.arm_acceleration = 0.5
    cfg.servoj_time = 1 / 500
    cfg.servoj_lookahead = 0.2
    cfg.servoj_gain = 100
    source = REPO / 'architectures/seer/third_party/3dflow_real_deploy/deploy.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'UR5eDeployEnv')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'move_to_home')
    # Execute only the unchanged home routine against mocked RTDE and clock.
    # Do not import the hardware module or run the environment constructor.
    namespace = {'np': np, 'time': SimpleNamespace(perf_counter=lambda: 0, sleep=Mock())}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    robot = SimpleNamespace(cfg=cfg, rtde_rec=Mock(), rtde_ctrl=Mock(),
                            _command_gripper_normalized=Mock())
    robot.rtde_rec.getActualQ.return_value = [0] * 6
    namespace['move_to_home'](robot)
    assert robot.rtde_ctrl.servoJ.call_count == 180
    np.testing.assert_array_equal(robot.rtde_ctrl.servoJ.call_args.args[0], POSE[:6])
    assert robot.rtde_ctrl.servoJ.call_args.args[1:] == (0.5, 0.5, 0.002, 0.2, 100)
    assert all(call.args == (0.0,) for call in robot._command_gripper_normalized.call_args_list)
    assert robot._step_count == 0 and robot._gripper_lock_counter == 0


def test_home_configuration_precedes_model_and_hardware_construction():
    path = REPO / 'architectures/seer/adapters/latentloop_real_deploy/deploy_ll_gui_v2.py'
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = {n.func.id: n.lineno for n in ast.walk(main)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for name in ('LatentLoopSeerControllerV2', 'TimedUR5eDeployEnv', 'ReadOnlyDeployEnvV2'):
        assert calls['configure_home_pose'] < calls[name]
