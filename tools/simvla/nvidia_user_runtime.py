"""Run a command with user-space NVIDIA libraries matching the loaded kernel.

No packages, kernel modules or system library links are changed. The default
private runtime is the already verified rb2 package extraction.
"""
import json
import os
from pathlib import Path
import sys

DEFAULT_ROOT = Path.home() / 'private/gnaroshi_vla_storage/runtime/nvidia_580_173/root'


def runtime_environment(environment=None, *, kernel_file=Path('/sys/module/nvidia/version'),
                        system_nvml=Path('/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1'),
                        runtime_root=DEFAULT_ROOT):
    env = dict(os.environ if environment is None else environment)
    if not kernel_file.is_file():
        raise RuntimeError('No loaded NVIDIA kernel module')
    version = kernel_file.read_text().strip()
    lib = runtime_root / 'usr/lib/x86_64-linux-gnu'
    vendor = runtime_root / 'usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    paths = [p for p in env.get('LD_LIBRARY_PATH', '').split(':') if p and p != str(lib)]
    system_version = system_nvml.resolve().name.removeprefix('libnvidia-ml.so.')
    if system_nvml.is_file() and system_version == version:
        if paths:
            env['LD_LIBRARY_PATH'] = ':'.join(paths)
        else:
            env.pop('LD_LIBRARY_PATH', None)
        if env.get('__EGL_VENDOR_LIBRARY_FILENAMES') == str(vendor):
            env.pop('__EGL_VENDOR_LIBRARY_FILENAMES', None)
        mode = 'system'
    else:
        for name in ('libnvidia-ml.so.1', 'libcuda.so.1', 'libEGL_nvidia.so.0'):
            path = lib/name
            if not path.is_file() or not path.resolve().name.endswith('.'+version):
                raise RuntimeError('Private library does not match loaded kernel '+version+': '+str(path))
        if not vendor.is_file():
            raise RuntimeError('Missing EGL vendor manifest: '+str(vendor))
        env['LD_LIBRARY_PATH'] = ':'.join([str(lib)]+paths)
        env['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(vendor)
        mode = 'private_matched'
    env['GNAROSHI_NVIDIA_RUNTIME_MODE'] = mode
    env['GNAROSHI_NVIDIA_KERNEL_VERSION'] = version
    return env, dict(mode=mode, kernel_version=version, system_library_version=system_version,
                     private_runtime_root=str(runtime_root))


def main():
    env, status = runtime_environment()
    args = sys.argv[1:]
    if args == ['--status']:
        print(json.dumps(status, indent=2))
        return
    if not args:
        raise SystemExit('Usage: nvidia_user_runtime.py COMMAND [ARGS...]')
    os.execvpe(args[0], args, env)


if __name__ == '__main__':
    main()
