"""Check shell initialization without converting data or starting GPU work."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "filename,stop_before",
    [
        ("prepare_real_doll_data_sd1.sh", '[[ -x "${py}"'),
        ("train_real_stackcupanddoll.sh", '[[ -x "${python_bin}"'),
    ],
)
def test_launch_context_ignores_foreign_worktree(tmp_path, filename, stop_before):
    repo = tmp_path / "ours"
    wrappers = repo / "architectures" / "simvla" / "wrappers"
    wrappers.mkdir(parents=True)
    (repo / "architectures" / "__init__.py").write_text('SOURCE = "ours"\n')
    foreign = tmp_path / "foreign"
    (foreign / "architectures").mkdir(parents=True)
    (foreign / "architectures" / "__init__.py").write_text('SOURCE = "foreign"\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    hostname = bin_dir / "hostname"
    hostname.write_text('#!/bin/sh\nprintf "jbrserver1\\n"\n')
    hostname.chmod(0o755)
    source = (ROOT / "architectures" / "simvla" / "wrappers" / filename).read_text()
    # Run the real initialization block, stopping before executable/data checks.
    prefix = source[:source.index(stop_before)]
    probe = (
        'import architectures,json,os; '
        'print(json.dumps({"cwd":os.getcwd(),"pythonpath":os.environ["PYTHONPATH"],'
        '"source":architectures.SOURCE}))'
    )
    script = wrappers / filename
    script.write_text(prefix + f'\n{shlex.quote(sys.executable)} -c {shlex.quote(probe)}\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith("SIMVLA_REAL_")}
    env.update(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        PYTHONPATH=str(foreign),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    result = subprocess.run(
        ["bash", str(script), "--preflight"], cwd=foreign, env=env,
        capture_output=True, text=True, check=True, timeout=30,
    )
    assert json.loads(result.stdout) == {
        "cwd": str(repo), "pythonpath": str(repo), "source": "ours",
    }


@pytest.mark.parametrize("filename,args,stop_before", [
    ("setup_real_deploy_env.sh", ["--check"], '"${prefix}/bin/python" -c'),
    ("deploy_latentloop_real.sh", ["source-preflight"], "command=("),
])
def test_deploy_context_excludes_user_packages(tmp_path, filename, args, stop_before):
    repo = tmp_path / "ours"
    wrappers = repo / "architectures/simvla/wrappers"
    wrappers.mkdir(parents=True)
    (repo / "architectures/__init__.py").write_text('SOURCE = "ours"\n')
    foreign = tmp_path / "foreign"
    (foreign / "architectures").mkdir(parents=True)
    (foreign / "architectures/__init__.py").write_text('SOURCE = "foreign"\n')
    source = (ROOT / "architectures/simvla/wrappers" / filename).read_text()
    probe = ('import architectures,json,os,site; '
             'print(json.dumps({"cwd":os.getcwd(),"path":os.environ["PYTHONPATH"],'
             '"user_site":site.ENABLE_USER_SITE,"source":architectures.SOURCE}))')
    script = wrappers / filename
    script.write_text(source[:source.index(stop_before)] +
                      f'\n{shlex.quote(sys.executable)} -c {shlex.quote(probe)}\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith("SIMVLA_REAL_")}
    env.update(PYTHONPATH=str(foreign), PYTHONHOME="/nonexistent/foreign-python",
               PYTHONNOUSERSITE="0", SIMVLA_REAL_PYTHON=sys.executable,
               SIMVLA_REAL_ENV_PREFIX=str(Path(sys.executable).parent.parent),
               SIMVLA_REAL_LOG_ROOT=str(tmp_path / "logs"))
    result = subprocess.run(["bash", str(script), *args], cwd=foreign, env=env,
                            capture_output=True, text=True, check=True, timeout=30)
    assert json.loads(result.stdout) == {
        "cwd": str(repo), "path": str(repo), "user_site": False, "source": "ours",
    }
