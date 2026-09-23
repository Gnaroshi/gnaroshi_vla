#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"
VERIFY_SCRIPT="${REPO_ROOT}/tools/seer/verify_realworld_latentloop_assets.py"

TASK_ID="${1:-}"
if [[ "${TASK_ID}" != "doll" && "${TASK_ID}" != "cabinet" && \
      "${TASK_ID}" != "rings" && "${TASK_ID}" != "stacking_cups" ]]; then
    echo "Usage: $0 {doll|cabinet|rings|stacking_cups}" >&2
    exit 2
fi

# Heavy artifacts stay in server-local shared storage, outside the source checkout.
STORAGE_ROOT="${SEER_REALWORLD_STORAGE_ROOT:-/var/shared/hdd_ext/ssd8000/mingyujung/gnaroshi_vla}"
ENV_PREFIX="${SEER_ENV_PREFIX:-${STORAGE_ROOT}/envs/seer_libero}"
CHECKPOINT_ROOT="${STORAGE_ROOT}/artifacts/checkpoints/seer"
DATASET_BASE="${STORAGE_ROOT}/artifacts/datasets/seer/realworld"
RESULT_BASE="${STORAGE_ROOT}/results/seer/latentloop/realworld/droid38"

VIT_CHECKPOINT="${CHECKPOINT_ROOT}/vit_mae/mae_pretrain_vit_base.pth"
VIT_SHA256="aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d"
CLIP_CHECKPOINT="${CHECKPOINT_ROOT}/clip/ViT-B-32.pt"
CLIP_SHA256="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"

case "${TASK_ID}" in
    doll)
        DATASET_NAME="doll_filtered_40p"
        TEACHER_SHA256="481a8dff0b0425b30fb9a589bbc20f89944e9e27240e1384bce0ccebfb5610b9"
        DATA_INFO_SHA256="dae8aaf341d8ad18b6f6e5d077f6ae7ea2ee399a9afba55ff125f15ed5ea2bce"
        EXPECTED_TOTAL_FRAMES=16635
        EXPECTED_TRAIN_WINDOWS=16235
        EXPECTED_INSTRUCTION="Pick up the white cup and place it on top of the upside-down pink cup, then pick up the blue penguin plush toy and put it in the white cup"
        DEFAULT_MASTER_PORT=18100
        ;;
    cabinet)
        DATASET_NAME="cabinet_filtered_40p"
        TEACHER_SHA256="fe642965f46a35ad6810f155bf7f62c1219e0a78fbe32fd2c3e1b5b95e563996"
        DATA_INFO_SHA256="be8c119b7fd04e7eb0abb9ce1d5dfb604ea7f2a57b0586cb75c931fcd51359fe"
        EXPECTED_TOTAL_FRAMES=43041
        EXPECTED_TRAIN_WINDOWS=42641
        EXPECTED_INSTRUCTION="Open the drawer, take the orange cup out and put it on the table, then put the blue cup in the drawer and close the drawer"
        DEFAULT_MASTER_PORT=18110
        ;;
    rings)
        DATASET_NAME="rings_filtered_40p"
        TEACHER_SHA256="1755e91255b405d56e81223634fdc373e1502a92d5cda4be40cc5d56ce2cb219"
        DATA_INFO_SHA256="22f7a64cb2de12181db7542a4fd35e18040fe9005627a93ba057cb711aff8fc9"
        EXPECTED_TOTAL_FRAMES=17371
        EXPECTED_TRAIN_WINDOWS=16971
        EXPECTED_INSTRUCTION="Put the blue ring on the wooden stand, then put the pink ring on the wooden stand"
        DEFAULT_MASTER_PORT=18120
        ;;
    stacking_cups)
        DATASET_NAME="stacking_cups_filtered_40p"
        TEACHER_SHA256="53cd9d1647dd1be42d7761c38952f9c4c43ecd336807dff25e0ac9738aa0c98a"
        DATA_INFO_SHA256="5cca7a31471ee64801fdae0d899e2d7367573d897c2019356095467970f104ca"
        EXPECTED_TOTAL_FRAMES=18439
        EXPECTED_TRAIN_WINDOWS=18039
        EXPECTED_INSTRUCTION="Stack the orange cup on top of the blue cup, then stack the yellow cup on top of the orange cup"
        DEFAULT_MASTER_PORT=18130
        ;;
esac

TEACHER_CHECKPOINT="${CHECKPOINT_ROOT}/realworld/droid38/${TASK_ID}/38.pth"
DATASET_ROOT="${DATASET_BASE}/${DATASET_NAME}"
DATA_INFO="${UPSTREAM_DIR}/data_info/${DATASET_NAME}.json"
RUN_ID="${RUN_ID:-r1}"
RUN_NAME="real_${TASK_ID}_latentloop_droid38_adapter_${RUN_ID}"
SAVE_CHECKPOINT_PATH="${RESULT_BASE}/${TASK_ID}"
RUN_DIR="${SAVE_CHECKPOINT_PATH}/${RUN_NAME}"
MASTER_PORT="${MASTER_PORT:-${DEFAULT_MASTER_PORT}}"

require_file() {
    local path="$1"
    if [[ ! -s "${path}" ]]; then
        echo "[ERROR] missing or empty file: ${path}" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1"
    if [[ ! -d "${path}" ]]; then
        echo "[ERROR] missing directory: ${path}" >&2
        exit 1
    fi
}

require_file "${ENV_PREFIX}/bin/python"
require_file "${ENV_PREFIX}/bin/torchrun"
require_file "${TEACHER_CHECKPOINT}"
require_file "${VIT_CHECKPOINT}"
require_file "${CLIP_CHECKPOINT}"
require_file "${DATA_INFO}"
require_file "${VERIFY_SCRIPT}"
require_dir "${DATASET_ROOT}/${DATASET_NAME}/0000"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    echo "[ERROR] set CUDA_VISIBLE_DEVICES to exactly four physical GPU IDs" >&2
    exit 1
fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_IDS[@]}" -ne 4 ]]; then
    echo "[ERROR] exactly four GPUs are required; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi
declare -A SEEN_GPU=()
for gpu in "${GPU_IDS[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] invalid physical GPU ID: ${gpu}" >&2
        exit 1
    fi
    if [[ -n "${SEEN_GPU[${gpu}]:-}" ]]; then
        echo "[ERROR] duplicate physical GPU ID: ${gpu}" >&2
        exit 1
    fi
    SEEN_GPU[${gpu}]=1
done

if ! [[ "${MASTER_PORT}" =~ ^[0-9]+$ ]] || (( MASTER_PORT < 1024 || MASTER_PORT > 65535 )); then
    echo "[ERROR] MASTER_PORT must be an integer in [1024, 65535]: ${MASTER_PORT}" >&2
    exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${STORAGE_ROOT}/cache/huggingface"
export TORCH_HOME="${STORAGE_ROOT}/cache/torch"
export XDG_CACHE_HOME="${STORAGE_ROOT}/cache/xdg"
export NUMBA_CACHE_DIR="${STORAGE_ROOT}/cache/numba"
export MPLCONFIGDIR="${STORAGE_ROOT}/cache/matplotlib"
export WANDB_DIR="${SAVE_CHECKPOINT_PATH}/wandb"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${REPO_ROOT}:${UPSTREAM_DIR}:${PYTHONPATH:-}"
export PATH="${ENV_PREFIX}/bin:${PATH}"

if [[ "$(command -v python)" != "${ENV_PREFIX}/bin/python" ]]; then
    echo "[ERROR] relocated environment Python is not first on PATH" >&2
    exit 1
fi
"${ENV_PREFIX}/bin/torchrun" --help >/dev/null
echo "[VERIFY][OK] relocated Python and torchrun entrypoints"

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" \
    "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${WANDB_DIR}"

# Seer checks this repo-relative location before asking CLIP to download.
mkdir -p "${UPSTREAM_DIR}/checkpoints/clip"
ln -sfn "${CLIP_CHECKPOINT}" "${UPSTREAM_DIR}/checkpoints/clip/ViT-B-32.pt"

"${ENV_PREFIX}/bin/python" "${VERIFY_SCRIPT}" \
    --task "${TASK_ID}" \
    --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
    --teacher-sha256 "${TEACHER_SHA256}" \
    --vit-checkpoint "${VIT_CHECKPOINT}" \
    --vit-sha256 "${VIT_SHA256}" \
    --clip-checkpoint "${CLIP_CHECKPOINT}" \
    --clip-sha256 "${CLIP_SHA256}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --data-info "${DATA_INFO}" \
    --data-info-sha256 "${DATA_INFO_SHA256}" \
    --expected-instruction "${EXPECTED_INSTRUCTION}" \
    --expected-episodes 40 \
    --expected-total-frames "${EXPECTED_TOTAL_FRAMES}" \
    --expected-train-windows "${EXPECTED_TRAIN_WINDOWS}" \
    --window-size 10

"${ENV_PREFIX}/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
print("[VERIFY][OK] CUDA devices:", [torch.cuda.get_device_name(i) for i in range(4)])
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[PREFLIGHT][PASS] task=${TASK_ID}"
    exit 0
fi

if [[ -e "${RUN_DIR}" ]]; then
    echo "[ERROR] refusing to overwrite existing run: ${RUN_DIR}" >&2
    echo "[ERROR] choose a new semantic RUN_ID, or inspect/remove the partial run explicitly" >&2
    exit 1
fi
mkdir -p "${RUN_DIR}"

WANDB_ARGS=()
if [[ "${REPORT_TO_WANDB:-1}" == "1" ]]; then
    WANDB_ARGS+=(--report_to_wandb)
fi

TRAIN_COMMAND=(
    "${ENV_PREFIX}/bin/torchrun"
    --nnodes=1
    --nproc_per_node=4
    --master_port="${MASTER_PORT}"
    train.py
    --traj_cons
    --rgb_pad 10
    --gripper_pad 4
    --use_aug_data
    --batch_size 32
    --gradient_accumulation_steps 4
    --workers 8
    --num_epochs 40
    --seed 42
    --precision fp32
    --bf16_module vision_encoder
    --learning_rate 1e-3
    --lr_scheduler cosine
    --warmup_epochs 2
    --weight_decay 1e-4
    --vit_checkpoint_path "${VIT_CHECKPOINT}"
    --finetune_from_pretrained_ckpt "${TEACHER_CHECKPOINT}"
    --finetune_type real
    --root_dir "${DATASET_ROOT}"
    --real_dataset_names "${DATASET_NAME}"
    --image_primary_size 200
    --image_wrist_size 84
    --calvin_input_image_size 224
    --transformer_layers 24
    --hidden_dim 384
    --transformer_heads 12
    --num_resampler_query 6
    --phase finetune
    --obs_pred
    --sequence_length 7
    --future_steps 3
    --window_size 10
    --action_pred_steps 3
    --multi_step_action 1
    --save_checkpoint
    --save_checkpoint_path "${SAVE_CHECKPOINT_PATH}"
    --run_name "${RUN_NAME}"
    --save_checkpoint_seq 1
    --start_save_checkpoint 25
    --wandb_project seer
    --use_lrnode_latent_update 1
    --lrnode_train_latent_distill 1
    --lrnode_teacher_target_mode shifted_context
    --lrnode_context_selected_step -1
    --lrnode_train_protocol adapter
    --lrnode_freeze_seer_for_adapter 1
    --lrnode_assert_only_lrnode_trainable 1
    --lrnode_latent_weight 0.05
    --lrnode_action_distill_weight 0.1
    --lrnode_bc_weight 0.0
    --lrnode_smooth_weight 0.001
    --lrnode_hidden_dim 256
    --lrnode_motion_dim 128
    --lrnode_fast_encoder_type diffcnn
    --lrnode_detach_input_latent 1
    --lrnode_detach_teacher_latent 1
    --lrnode_freeze_action_head_for_lrnode 1
    --lrnode_use_post_layernorm 0
    --lrnode_multistep_train 0
    --lrnode_train_max_horizon 2
    --lrnode_gate_init_bias -4.0
    --lrnode_log_sanity 1
    "${WANDB_ARGS[@]}"
)

{
    printf 'task=%q\n' "${TASK_ID}"
    printf 'instruction=%q\n' "${EXPECTED_INSTRUCTION}"
    printf 'hostname=%q\n' "$(hostname)"
    printf 'cuda_visible_devices=%q\n' "${CUDA_VISIBLE_DEVICES}"
    printf 'teacher_checkpoint=%q\n' "${TEACHER_CHECKPOINT}"
    printf 'teacher_sha256=%q\n' "${TEACHER_SHA256}"
    printf 'dataset_root=%q\n' "${DATASET_ROOT}"
    printf 'data_info_sha256=%q\n' "${DATA_INFO_SHA256}"
    printf 'vit_checkpoint=%q\n' "${VIT_CHECKPOINT}"
    printf 'vit_sha256=%q\n' "${VIT_SHA256}"
    printf 'clip_checkpoint=%q\n' "${CLIP_CHECKPOINT}"
    printf 'clip_sha256=%q\n' "${CLIP_SHA256}"
    printf 'global_effective_batch=%q\n' 512
    printf 'wandb_mode=%q\n' "${WANDB_MODE}"
    printf 'command='; printf '%q ' "${TRAIN_COMMAND[@]}"; printf '\n'
} > "${RUN_DIR}/launch_manifest.env"

echo "[TRAIN] task=${TASK_ID}"
echo "[TRAIN] GPUs=${CUDA_VISIBLE_DEVICES}; global effective batch=32*4*4=512"
echo "[TRAIN] frozen teacher=${TEACHER_CHECKPOINT}"
echo "[TRAIN] dataset=${DATASET_ROOT}/${DATASET_NAME}"
echo "[TRAIN] output=${RUN_DIR}"
echo "[TRAIN] checkpoints=26.pth..39.pth"

cd "${UPSTREAM_DIR}"
set +e
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/console.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e
if [[ "${TRAIN_STATUS}" -ne 0 ]]; then
    printf '%s\n' "${TRAIN_STATUS}" > "${RUN_DIR}/exit_code.txt"
    echo "[ERROR] training failed with exit code ${TRAIN_STATUS}; artifacts preserved at ${RUN_DIR}" >&2
    exit "${TRAIN_STATUS}"
fi

for epoch in $(seq 26 39); do
    require_file "${RUN_DIR}/${epoch}.pth"
done
date --iso-8601=seconds > "${RUN_DIR}/training_complete.txt"
printf '0\n' > "${RUN_DIR}/exit_code.txt"
echo "[DONE] task=${TASK_ID}; output=${RUN_DIR}"
