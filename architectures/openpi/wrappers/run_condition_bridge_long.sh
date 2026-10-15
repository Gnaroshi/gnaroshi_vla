#!/usr/bin/env bash
main() (
  set -euo pipefail
  [[ $(hostname) == jbr-TRX50 ]] || { echo 'rb2 only'; return 2; }
  ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
  export PI05_STORAGE=/home/mingyujung/private/gnaroshi_vla_storage
  export OPENPI_UPSTREAM_ROOT=/home/mingyujung/private/gnaroshi_vla/architectures/openpi/upstream
  export LATENT_BRIDGE_UPSTREAM_ROOT=$PI05_STORAGE/upstream/latent_bridge
  export PI05_CLIENT_PY=$OPENPI_UPSTREAM_ROOT/examples/libero/.venv/bin/python
  PY=$OPENPI_UPSTREAM_ROOT/.venv/bin/python
  RUNTIME=$PI05_STORAGE/runtime/nvidia_580_173/root
  export LD_LIBRARY_PATH=$RUNTIME/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  export __EGL_VENDOR_LIBRARY_FILENAMES=$RUNTIME/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export HF_HOME=$PI05_STORAGE/cache/huggingface HF_LEROBOT_HOME=$PI05_STORAGE/datasets/lerobot
  export XDG_CACHE_HOME=$PI05_STORAGE/cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 USE_TF=0
  export LIBERO_CONFIG_PATH=$PI05_STORAGE/cache/openpi/dual_loop/libero_config
  export NUMBA_CACHE_DIR=$PI05_STORAGE/cache/openpi/dual_loop/numba MPLCONFIGDIR=$PI05_STORAGE/cache/openpi/dual_loop/matplotlib
  export PYTHONPATH=$ROOT:$OPENPI_UPSTREAM_ROOT/src:$OPENPI_UPSTREAM_ROOT/packages/openpi-client/src:$OPENPI_UPSTREAM_ROOT/third_party/libero
  OUT=$PI05_STORAGE/results/openpi/condition_bridge/recurrent_nfe1_long_seed7
  mkdir -p "$OUT/logs"
  cd "$ROOT"
  "$PY" -u tools/openpi/run_condition_bridge.py "${1:-all}" 2>&1 | tee -a "$OUT/logs/launcher.log"
)
set +e
main "$@"
rc=$?
echo "PI05_CONDITION_BRIDGE_EXIT=$rc (pane retained)"
exit 0
