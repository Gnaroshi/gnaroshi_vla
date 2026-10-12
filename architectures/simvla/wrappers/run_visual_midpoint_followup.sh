#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
case "$(hostname)" in
  jbrserver1)
    PY=/home/mingyujung/miniconda3/envs/simvla_libero/bin/python
    STORAGE=/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla
    ;;
  jbr-TRX50)
    PY=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
    STORAGE=/home/mingyujung/private/gnaroshi_vla_storage
    RUNTIME="$STORAGE/runtime/nvidia_580_173/root"
    export LD_LIBRARY_PATH="$RUNTIME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$RUNTIME/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    ;;
  *) printf 'Unsupported host\n'; exit 0 ;;
esac
OUT="$STORAGE/results/simvla/visual_midpoint/current_vision_nfe1_v1/followup"
mkdir -p "$OUT" || exit 0
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
"$PY" -u -m tools.simvla.visual_midpoint_followup all 2>&1 | tee -a "$OUT/launcher.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "$rc" > "$OUT/launcher.status"
printf 'VISUAL_FOLLOWUP_EXIT=%s status=%s\n' "$rc" "$OUT/launcher.status"
exit 0
