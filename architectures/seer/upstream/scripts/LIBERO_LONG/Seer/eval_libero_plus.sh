#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:-libero_10}"

case "${SUITE}" in
  libero_10|libero_spatial|libero_object|libero_goal)
    ;;
  *)
    echo "[LIBERO-PLUS EVAL] ERROR: unsupported suite '${SUITE}'"
    echo "[LIBERO-PLUS EVAL] Use one of: libero_10 libero_spatial libero_object libero_goal"
    exit 1
    ;;
esac

ckpt_ids_str="${CKPT_IDS:-36 35}"
read -r -a pthlist <<< "${ckpt_ids_str}"

LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/home/mingyujung/private/LIBERO-plus}"
LIBERO_PLUS_CONFIG_FILE="${LIBERO_PLUS_CONFIG_FILE:-${HOME}/.libero_plus/config.yaml}"
protocol_root="${LRNODE_PROTOCOL_ROOT:-/home/mingyujung/private/seer/seer_node3/runs_lrnode_protocol_20260616}"
baseline_env="${BASELINE_ENV:-${protocol_root}/train/_latest/scratch.env}"

if [[ -z "${RUN_NAME:-}" || -z "${CKPT_ROOT:-}" ]]; then
    if [[ ! -f "${baseline_env}" ]]; then
        echo "[LIBERO-PLUS EVAL] ERROR: baseline env not found: ${baseline_env}" >&2
        echo "[LIBERO-PLUS EVAL] Set RUN_NAME and CKPT_ROOT explicitly, or run scratch.sh first." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${baseline_env}"
    RUN_NAME="${RUN_NAME:-${LRNODE_RUN_NAME}}"
    CKPT_ROOT="${CKPT_ROOT:-${LRNODE_SAVE_CHECKPOINT_PATH}}"
    echo "[LIBERO-PLUS EVAL] loaded latest baseline run from ${baseline_env}"
fi

run_name="${RUN_NAME}"
resume_from_checkpoint_root="${CKPT_ROOT}/${run_name}"

vit_checkpoint_path="checkpoints/vit_mae/mae_pretrain_vit_base.pth"
save_checkpoint_path="checkpoints/"

LOG_ROOT="${LOG_ROOT:-${protocol_root}/eval_libero_plus}"
mkdir -p "${LOG_ROOT}"

RUNTIME_CONFIG_DIR="${HOME}/.libero"
RUNTIME_CONFIG_FILE="${RUNTIME_CONFIG_DIR}/config.yaml"
BACKUP_CONFIG_FILE="${RUNTIME_CONFIG_FILE}.bak.$$"
HAD_ORIGINAL_CONFIG=0

cleanup() {
    if [ "${HAD_ORIGINAL_CONFIG}" -eq 1 ] && [ -f "${BACKUP_CONFIG_FILE}" ]; then
        mv -f "${BACKUP_CONFIG_FILE}" "${RUNTIME_CONFIG_FILE}"
        echo "[LIBERO-PLUS EVAL] restored original runtime config: ${RUNTIME_CONFIG_FILE}"
    else
        rm -f "${RUNTIME_CONFIG_FILE}"
        rm -f "${BACKUP_CONFIG_FILE}"
        echo "[LIBERO-PLUS EVAL] removed temporary runtime config: ${RUNTIME_CONFIG_FILE}"
    fi
}
trap cleanup EXIT

mkdir -p "${RUNTIME_CONFIG_DIR}"

if [ ! -f "${LIBERO_PLUS_CONFIG_FILE}" ]; then
    echo "[LIBERO-PLUS EVAL] ERROR: LIBERO-plus config file not found: ${LIBERO_PLUS_CONFIG_FILE}"
    exit 1
fi

if [ -f "${RUNTIME_CONFIG_FILE}" ]; then
    cp "${RUNTIME_CONFIG_FILE}" "${BACKUP_CONFIG_FILE}"
    HAD_ORIGINAL_CONFIG=1
fi

cp "${LIBERO_PLUS_CONFIG_FILE}" "${RUNTIME_CONFIG_FILE}"

export PYTHONPATH="${LIBERO_PLUS_ROOT}:${PYTHONPATH:-}"
export LIBERO_PLUS_ROOT

LIBERO_GL_BACKEND="${LIBERO_GL_BACKEND:-egl}"
case "${LIBERO_GL_BACKEND}" in
  osmesa|egl)
    ;;
  *)
    echo "[LIBERO-PLUS EVAL] ERROR: LIBERO_GL_BACKEND must be 'osmesa' or 'egl', got '${LIBERO_GL_BACKEND}'"
    exit 1
    ;;
esac

export MUJOCO_GL="${LIBERO_GL_BACKEND}"
export PYOPENGL_PLATFORM="${LIBERO_GL_BACKEND}"

echo "[LIBERO-PLUS EVAL] SUITE=${SUITE}"
echo "[LIBERO-PLUS EVAL] LIBERO_PLUS_ROOT=${LIBERO_PLUS_ROOT}"
echo "[LIBERO-PLUS EVAL] checkpoint_root=${resume_from_checkpoint_root}"
echo "[LIBERO-PLUS EVAL] log_root=${LOG_ROOT}"
echo "[LIBERO-PLUS EVAL] LIBERO_PLUS_CONFIG_FILE=${LIBERO_PLUS_CONFIG_FILE}"
echo "[LIBERO-PLUS EVAL] RUNTIME_CONFIG_FILE=${RUNTIME_CONFIG_FILE}"
echo "[LIBERO-PLUS EVAL] PYTHONPATH=${PYTHONPATH}"
echo "[LIBERO-PLUS EVAL] MUJOCO_GL=${MUJOCO_GL}"
echo "[LIBERO-PLUS EVAL] PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"

if [ ! -d "${LIBERO_PLUS_ROOT}/libero/libero/assets" ]; then
    echo "[LIBERO-PLUS EVAL] ERROR: assets directory not found: ${LIBERO_PLUS_ROOT}/libero/libero/assets"
    exit 1
fi

if [ ! -d "${LIBERO_PLUS_ROOT}/libero/libero/benchmark" ]; then
    echo "[LIBERO-PLUS EVAL] ERROR: benchmark directory not found: ${LIBERO_PLUS_ROOT}/libero/libero/benchmark"
    exit 1
fi

if [ ! -d "${LIBERO_PLUS_ROOT}/libero/libero/bddl_files/${SUITE}" ]; then
    echo "[LIBERO-PLUS EVAL] ERROR: suite BDDL directory not found: ${LIBERO_PLUS_ROOT}/libero/libero/bddl_files/${SUITE}"
    echo "[LIBERO-PLUS EVAL] Available suite directories:"
    ls -1 "${LIBERO_PLUS_ROOT}/libero/libero/bddl_files"
    exit 1
fi

python - <<'PY'
import importlib
import numpy as np

print("[LIBERO-PLUS EVAL] numpy version:", np.__version__)
if not np.__version__.startswith("1."):
    raise SystemExit("[LIBERO-PLUS EVAL] ERROR: NumPy 1.x is required for this Seer environment.")

import torch
print("[LIBERO-PLUS EVAL] torch version:", torch.__version__)

import cv2
print("[LIBERO-PLUS EVAL] cv2 version:", cv2.__version__)

from wand.api import library as wandlibrary  # noqa: F401
print("[LIBERO-PLUS EVAL] wand ok")

from skimage.filters import gaussian  # noqa: F401
print("[LIBERO-PLUS EVAL] skimage ok")

libero_pkg = importlib.import_module("libero.libero")
print("[LIBERO-PLUS EVAL] libero.libero package:", libero_pkg.__file__)
PY

echo "[LIBERO-PLUS EVAL] active runtime config:"
cat "${RUNTIME_CONFIG_FILE}"

for ckpt_id in "${pthlist[@]}"; do
    this_resume_from_checkpoint="${resume_from_checkpoint_root}/${ckpt_id}.pth"
    dirname=$(basename "${resume_from_checkpoint_root}")
    LOG_DIR="${LOG_ROOT}/${dirname}/${SUITE}/ckpt-${ckpt_id}"
    mkdir -p "${LOG_DIR}"
    logfile="${LOG_DIR}/${ckpt_id}.log"
    export LIBERO_PLUS_EVAL_OUTDIR="${LOG_DIR}"
    export LIBERO_PLUS_EVAL_CKPT_ID="${ckpt_id}"

    if [ ! -f "${this_resume_from_checkpoint}" ]; then
        echo "[LIBERO-PLUS EVAL] ERROR: checkpoint not found: ${this_resume_from_checkpoint}"
        exit 1
    fi

    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#    export CUDA_VISIBLE_DEVICES=0,1,2,3
#    export CUDA_VISIBLE_DEVICES=4,5,6,7

    node=1
    node_num=8
#    node_num=4
#    master_port=10911
#    master_port=10912
    master_port=10913
#    master_port=10914

    echo "============================================================"
    echo "[LIBERO-PLUS EVAL] suite               : ${SUITE}"
    echo "[LIBERO-PLUS EVAL] ckpt_id             : ${ckpt_id}"
    echo "[LIBERO-PLUS EVAL] checkpoint path     : ${this_resume_from_checkpoint}"
    echo "[LIBERO-PLUS EVAL] logfile             : ${logfile}"
    echo "============================================================"

    python -m torch.distributed.run \
        --nnodes=${node} \
        --nproc_per_node=${node_num} \
        --master_port=${master_port} \
        eval_libero_plus.py \
        --traj_cons \
        --rgb_pad 10 \
        --gripper_pad 4 \
        --gradient_accumulation_steps 1 \
        --bf16_module "vision_encoder" \
        --vit_checkpoint_path "${vit_checkpoint_path}" \
        --calvin_dataset "" \
        --workers 16 \
        --lr_scheduler cosine \
        --save_every_iter 50000 \
        --num_epochs 20 \
        --seed 42 \
        --batch_size 64 \
        --precision fp32 \
        --weight_decay 1e-4 \
        --num_resampler_query 6 \
        --run_name "${run_name}_${SUITE}_libero_plus_eval" \
        --transformer_layers 24 \
        --phase "evaluate" \
        --finetune_type "${SUITE}" \
        --libero_path "${LIBERO_PLUS_ROOT}" \
        --save_checkpoint_path "${save_checkpoint_path}" \
        --action_pred_steps 3 \
        --future_steps 3 \
        --sequence_length 7 \
        --obs_pred \
        --gripper_width \
        --eval_libero_ensembling \
        --resume_from_checkpoint ${this_resume_from_checkpoint} | tee ${logfile}

done
