#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
case "$(hostname)" in
  jbrserver1)
    PY=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
    MODULE=tools.simvla.bounded_history_pipeline
    OUT=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/bounded_history/fresh_nfe1_seed01_v1
    ;;
  jbr-TRX50)
    PY=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
    MODULE=tools.simvla.bounded_history_rb2
    OUT=/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/bounded_history/fresh_nfe1_compiled_seed01_v1
    export SIMVLA_NVIDIA_RUNTIME_ROOT=${SIMVLA_NVIDIA_RUNTIME_ROOT:-/home/mingyujung/private/gnaroshi_vla_storage/runtime/nvidia_580_173/root}
    ;;
  *) printf 'Unsupported host\n'; exit 0 ;;
esac
mkdir -p "$OUT/logs" || exit 0
if [[ -n "${SIMVLA_NVIDIA_RUNTIME_ROOT:-}" ]]; then
  LIB="$SIMVLA_NVIDIA_RUNTIME_ROOT/usr/lib/x86_64-linux-gnu"
  VERSION=$(readlink "$LIB/libcuda.so.1")
  VERSION=${VERSION#libcuda.so.}
  if [[ ! -f "$LIB/libcuda.so.$VERSION" ]] || ! grep -Fq " $VERSION " /proc/driver/nvidia/version; then
    printf 'NVIDIA_RUNTIME_FAIL: libraries do not match kernel driver\n'
    printf '1\n' > "$OUT/logs/launcher.status"
    exit 0
  fi
  export LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$SIMVLA_NVIDIA_RUNTIME_ROOT/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" -u -m "$MODULE" "$@" 2>&1 | tee -a "$OUT/logs/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/logs/launcher.status"
printf 'BOUNDED_HISTORY_EXIT=%s status=%s\n' "$rc" "$OUT/logs/launcher.status"
exit 0
