from pathlib import Path

import pytest

from tools.simvla.nvidia_user_runtime import runtime_environment


def fixtures(tmp_path, kernel='580.173.02', system='580.178.04', private='580.173.02'):
    k = tmp_path/'kernel'
    k.write_text(kernel)
    s = tmp_path/('libnvidia-ml.so.'+system)
    s.touch()
    link = tmp_path/'libnvidia-ml.so.1'
    link.symlink_to(s)
    root = tmp_path/'private'
    lib = root/'usr/lib/x86_64-linux-gnu'
    lib.mkdir(parents=True)
    for alias, name in [('libcuda.so.1','libcuda.so.'),('libnvidia-ml.so.1','libnvidia-ml.so.'),
                        ('libEGL_nvidia.so.0','libEGL_nvidia.so.')]:
        target = lib/(name+private)
        target.touch()
        (lib/alias).symlink_to(target)
    vendor = root/'usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    vendor.parent.mkdir(parents=True)
    vendor.write_text('{}')
    return dict(kernel_file=k,system_nvml=link,runtime_root=root),lib,vendor


def test_mismatch_selects_complete_matching_runtime(tmp_path):
    args,lib,vendor = fixtures(tmp_path)
    original = dict(LD_LIBRARY_PATH='/other',CUDA_VISIBLE_DEVICES='0')
    env,status = runtime_environment(original,**args)
    assert status['mode']=='private_matched'
    assert env['LD_LIBRARY_PATH']==str(lib)+':/other'
    assert env['__EGL_VENDOR_LIBRARY_FILENAMES']==str(vendor)
    assert env['CUDA_VISIBLE_DEVICES']=='0' and original['LD_LIBRARY_PATH']=='/other'


def test_reboot_returns_to_system_and_removes_stale_private_path(tmp_path):
    args,lib,vendor = fixtures(tmp_path,kernel='580.178.04')
    env,status = runtime_environment(dict(LD_LIBRARY_PATH=str(lib)+':/other',
        __EGL_VENDOR_LIBRARY_FILENAMES=str(vendor)),**args)
    assert status['mode']=='system' and env['LD_LIBRARY_PATH']=='/other'
    assert '__EGL_VENDOR_LIBRARY_FILENAMES' not in env


def test_refuses_mismatched_or_incomplete_private_runtime(tmp_path):
    args,lib,_ = fixtures(tmp_path,private='570.0')
    with pytest.raises(RuntimeError,match='does not match'):
        runtime_environment({},**args)


def test_missing_kernel_is_explicit(tmp_path):
    with pytest.raises(RuntimeError,match='No loaded'):
        runtime_environment({},kernel_file=tmp_path/'missing')


def test_repeated_activation_does_not_duplicate_path(tmp_path):
    args,lib,_ = fixtures(tmp_path)
    env,_=runtime_environment(dict(LD_LIBRARY_PATH=str(lib)+':/other:'+str(lib)),**args)
    assert env['LD_LIBRARY_PATH']==str(lib)+':/other'
