#!/usr/bin/env bash
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 0
if [[ "$(hostname)" != jbr-TRX50 ]]; then printf 'rb2 only\n'; exit 0; fi
STORAGE=/home/mingyujung/private/gnaroshi_vla_storage
PY="$STORAGE/envs/simvla/libero_mujoco237/bin/python"
RUNTIME="$STORAGE/runtime/nvidia_580_173/root"
export LD_LIBRARY_PATH="$RUNTIME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export __EGL_VENDOR_LIBRARY_FILENAMES="$RUNTIME/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" USE_TF=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
OUT="$STORAGE/results/simvla/visual_readout_distillation/preserved_reader_nfe1_v1"
mkdir -p "$OUT/logs" || exit 0
"$PY" -u -m tools.simvla.visual_readout_remote "$@" 2>&1 | tee -a "$OUT/logs/remote.log"
statuses=("${PIPESTATUS[@]}")
rc=${statuses[0]}
if [[ "$rc" -eq 0 && "${statuses[1]}" -ne 0 ]]; then rc=${statuses[1]}; fi
printf '%s\n' "$rc" > "$OUT/logs/remote.status"
printf 'VISUAL_READOUT_REMOTE_EXIT=%s status=%s\n' "$rc" "$OUT/logs/remote.status"
exit 0
