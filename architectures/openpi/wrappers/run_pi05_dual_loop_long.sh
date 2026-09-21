#!/usr/bin/env bash
# Run in its own shell: failures never close the caller's tmux pane.
main() (
  set -euo pipefail
  ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
  export PI05_STORAGE=${PI05_STORAGE:-/home/mingyujung/private/gnaroshi_vla_storage}
  export OPENPI_UPSTREAM_ROOT=${OPENPI_UPSTREAM_ROOT:-/home/mingyujung/private/gnaroshi_vla/architectures/openpi/upstream}
  export PI05_CLIENT_PY=${PI05_CLIENT_PY:-${OPENPI_UPSTREAM_ROOT}/examples/libero/.venv/bin/python}
  PY=${PI05_PYTHON:-${OPENPI_UPSTREAM_ROOT}/.venv/bin/python}
  export CUDA_VISIBLE_DEVICES=${PI05_GPU:-0}
  case "$(hostname)" in
    jbrserver1) case "$CUDA_VISIBLE_DEVICES" in 4|5|6|7) ;; *) echo 'sd1 permits physical GPUs 4-7 only'; return 2;; esac ;;
  esac
  [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || { echo 'This pipeline uses one GPU'; return 2; }
  export HF_HOME=${PI05_STORAGE}/cache/huggingface
  export HF_LEROBOT_HOME=${PI05_STORAGE}/datasets/lerobot
  export HF_DATASETS_CACHE=${HF_HOME}/datasets
  export XDG_CACHE_HOME=${PI05_STORAGE}/cache
  export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTHONHASHSEED=42 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONDONTWRITEBYTECODE=1
  export PI05_CONDITION_STEPS=${PI05_CONDITION_STEPS:-10000}
  export PI05_GENERATION_STEPS=${PI05_GENERATION_STEPS:-10000}
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
  export NUMBA_CACHE_DIR=${PI05_STORAGE}/cache/openpi/dual_loop/numba
  export MPLCONFIGDIR=${PI05_STORAGE}/cache/openpi/dual_loop/matplotlib
  export WANDB_MODE=${WANDB_MODE:-online}
  LIBERO=${PI05_LIBERO_ROOT:-${OPENPI_UPSTREAM_ROOT}/third_party/libero}
  export LIBERO_CONFIG_PATH=${PI05_STORAGE}/cache/openpi/dual_loop/libero_config
  mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$LIBERO_CONFIG_PATH"
  export PYTHONPATH="$ROOT:$OPENPI_UPSTREAM_ROOT/src:$OPENPI_UPSTREAM_ROOT/packages/openpi-client/src:$LIBERO${PYTHONPATH:+:$PYTHONPATH}"
  export PI05_LIBERO_ROOT=$LIBERO
  "$PY" - <<'PY'
import os,pathlib,yaml
p=pathlib.Path(os.environ['PI05_LIBERO_ROOT'])/'libero'
assert (p/'libero/bddl_files').is_dir(), p
values={'benchmark_root':str(p),'bddl_files':str(p/'libero/bddl_files'),
        'init_states':str(p/'libero/init_files'),'assets':str(p/'libero/assets'),
        'datasets':str(pathlib.Path(os.environ['PI05_STORAGE'])/'datasets/LIBERO_demonstrations')}
pathlib.Path(os.environ['LIBERO_CONFIG_PATH'],'config.yaml').write_text(yaml.safe_dump(values))
PY
  cd "$ROOT"
  mode=${1:---all}
  case "$mode" in --all) mode=all;; --verify) mode=verify;; --preflight) mode=prepare;; --summarize) mode=summarize;; *) echo 'Use --all / --verify / --preflight / --summarize'; return 2;; esac
  "$PY" tools/openpi/run_pi05_dual_loop.py "$mode"
)
set +e
main "$@"
rc=$?
if [[ $rc == 0 ]]; then
  echo 'PI05_DUAL_LOOP_COMMAND_COMPLETE'
else
  echo "PI05_DUAL_LOOP_FAILED rc=$rc; inspect the output status.json and logs. Re-running resumes completed work."
  if [[ ${PI05_STRICT_EXIT:-0} == 1 ]]; then exit "$rc"; fi
fi
exit 0
