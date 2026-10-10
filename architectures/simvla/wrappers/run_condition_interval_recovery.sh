#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
case "$(hostname)" in
  jbrserver1)
    PY=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
    MODULE=tools.simvla.condition_interval_recovery
    OUT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/condition_output_split/interval_recovery_nfe1_seed01_v1
    ;;
  jbr-TRX50)
    PY=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
    MODULE=tools.simvla.condition_interval_recovery_rb2
    OUT=/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/condition_output_split/interval_recovery_nfe1_compiled_seed01_v1
    ;;
  *) printf 'Unsupported host\n'; exit 0 ;;
esac
mkdir -p "$OUT/logs" || exit 0
if [[ -n "${SIMVLA_NVIDIA_RUNTIME_ROOT:-}" ]]; then
  LIB="$SIMVLA_NVIDIA_RUNTIME_ROOT/usr/lib/x86_64-linux-gnu"
  VERSION=$(readlink "$LIB/libcuda.so.1")
  VERSION=${VERSION#libcuda.so.}
  if [[ ! -f "$LIB/libcuda.so.$VERSION" ]] || ! grep -Fq " $VERSION " /proc/driver/nvidia/version; then
    printf 'NVIDIA_RUNTIME_FAIL: private libraries must match the loaded kernel driver\n'
    printf '1\n' > "$OUT/logs/launcher.status"
    exit 0
  fi
  export LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$SIMVLA_NVIDIA_RUNTIME_ROOT/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
  if ! nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; then
    printf '1\n' > "$OUT/logs/launcher.status"
    exit 0
  fi
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" -u -m "$MODULE" "$@" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
printf 'INTERVAL_RECOVERY_EXIT=%s status=%s\n' "$rc" "$OUT/logs/launcher.status"
exit 0
