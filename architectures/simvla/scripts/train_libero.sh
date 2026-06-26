#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${ARCH_DIR}/../.." && pwd)"
SIMVLA_DIR="${SIMVLA_DIR:-${ARCH_DIR}/SimVLA}"

model_size="${SIMVLA_MODEL_SIZE:-small}"
batch_size="${SIMVLA_BATCH_SIZE:-64}"
learning_coef="${SIMVLA_LEARNING_COEF:-0.1}"
result_dir="${SIMVLA_RESULT_DIR:-}"
output_dir="${SIMVLA_OUTPUT_DIR:-}"
resume_ckpt="${SIMVLA_RESUME_CKPT:-}"
dry_run="${SIMVLA_DRY_RUN:-0}"

usage() {
    cat <<'EOF'
Usage:
  train_libero.sh [--model-size small|large] [--batch-size N] [--learning-coef X]
                  [--result-dir PATH] [--output-dir PATH] [--resume-ckpt PATH]
                  [--dry-run]

Environment overrides:
  LIBERO_ROOT                 Official LIBERO root. Default:
                              /home/mingyujung/shared/nvme1/mingyujung/datasets/robotics/LIBERO
  CUDA_VISIBLE_DEVICES        GPU ids. Defaults to 0,1,2,3 for small and 4,5,6,7 for large.
  SIMVLA_NUM_PROCESSES        accelerate process count. Default: 4
  SIMVLA_MAIN_PROCESS_PORT    accelerate main process port. Default: 29504
  SIMVLA_ITERS                training iterations. Default: 200000
  SIMVLA_SAVE_INTERVAL        checkpoint interval. Default: 10000
  SIMVLA_LOG_INTERVAL         log interval. Default: 20
  SIMVLA_NUM_WORKERS          dataloader workers. Default: 4
  SIMVLA_SMOLVLM_MODEL        HF repo or local backbone path. Default:
                              HuggingFaceTB/SmolVLM-500M-Instruct
  SIMVLA_DRY_RUN              If 1, print/write launch command and exit before training.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-size)
            model_size="$2"
            shift 2
            ;;
        --batch-size)
            batch_size="$2"
            shift 2
            ;;
        --learning-coef)
            learning_coef="$2"
            shift 2
            ;;
        --result-dir)
            result_dir="$2"
            shift 2
            ;;
        --output-dir)
            output_dir="$2"
            shift 2
            ;;
        --resume-ckpt)
            resume_ckpt="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        small|large)
            model_size="$1"
            shift
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${model_size}" in
    small)
        experiment_name="simvla_libero_small"
        default_cuda_visible_devices="0,1,2,3"
        learning_rate="${SIMVLA_LEARNING_RATE:-1e-4}"
        hidden_size="${SIMVLA_HIDDEN_SIZE:-768}"
        depth="${SIMVLA_DEPTH:-12}"
        num_heads="${SIMVLA_NUM_HEADS:-12}"
        ;;
    large)
        experiment_name="simvla_libero_large"
        default_cuda_visible_devices="4,5,6,7"
        learning_rate="${SIMVLA_LEARNING_RATE:-2e-4}"
        hidden_size="${SIMVLA_HIDDEN_SIZE:-1024}"
        depth="${SIMVLA_DEPTH:-24}"
        num_heads="${SIMVLA_NUM_HEADS:-16}"
        ;;
    *)
        echo "[ERROR] model-size must be small or large: ${model_size}" >&2
        exit 2
        ;;
esac

if [[ "${CONDA_DEFAULT_ENV:-}" != "simvla_libero" && "${SIMVLA_SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        eval "$(conda shell.bash hook)"
        conda activate simvla_libero
    else
        echo "[WARN] conda not found; continuing with current shell environment" >&2
    fi
fi

bash "${SCRIPT_DIR}/prepare_libero_links.sh"

if [[ "${SIMVLA_SKIP_DATA_CHECK:-0}" != "1" ]]; then
    python "${SCRIPT_DIR}/check_libero_dataset.py"
fi

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
run_id="${RUN_ID:-$(date +%s)}"
if [[ -z "${result_dir}" ]]; then
    result_dir="${ROOT_DIR}/results/simvla/original/${experiment_name}/${timestamp}_${run_id}"
fi
if [[ -z "${output_dir}" ]]; then
    output_dir="${result_dir}/checkpoints"
fi

mkdir -p "${result_dir}/logs" "${result_dir}/metrics" "${output_dir}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${default_cuda_visible_devices}}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

num_processes="${SIMVLA_NUM_PROCESSES:-4}"
main_process_port="${SIMVLA_MAIN_PROCESS_PORT:-29504}"
mixed_precision="${SIMVLA_MIXED_PRECISION:-bf16}"
smolvlm_model="${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}"
train_metas_path="${SIMVLA_TRAIN_METAS_PATH:-./datasets/metas/libero_train.json}"
norm_stats_path="${SIMVLA_NORM_STATS_PATH:-./norm_stats/libero_norm.json}"
iters="${SIMVLA_ITERS:-200000}"
warmup_steps="${SIMVLA_WARMUP_STEPS:-0}"
freeze_steps="${SIMVLA_FREEZE_STEPS:-1000}"
save_interval="${SIMVLA_SAVE_INTERVAL:-10000}"
log_interval="${SIMVLA_LOG_INTERVAL:-20}"
num_workers="${SIMVLA_NUM_WORKERS:-4}"
max_grad_norm="${SIMVLA_MAX_GRAD_NORM:-1.0}"
num_actions="${SIMVLA_NUM_ACTIONS:-10}"
image_size="${SIMVLA_IMAGE_SIZE:-384}"
use_adaln="${SIMVLA_USE_ADALN:-false}"

train_args=(
    --output_dir "${output_dir}"
    --train_metas_path "${train_metas_path}"
    --smolvlm_model_path "${smolvlm_model}"
    --action_mode libero_joint
    --batch_size "${batch_size}"
    --learning_rate "${learning_rate}"
    --learning_coef "${learning_coef}"
    --num_actions "${num_actions}"
    --iters "${iters}"
    --warmup_steps "${warmup_steps}"
    --freeze_steps "${freeze_steps}"
    --hidden_size "${hidden_size}"
    --depth "${depth}"
    --num_heads "${num_heads}"
    --num_workers "${num_workers}"
    --save_interval "${save_interval}"
    --log_interval "${log_interval}"
    --image_size "${image_size}"
    --norm_stats_path "${norm_stats_path}"
    --max_grad_norm "${max_grad_norm}"
)

if [[ "${use_adaln}" == "true" ]]; then
    train_args+=(--use_adaln)
fi

if [[ -n "${resume_ckpt}" ]]; then
    train_args+=(--models "${resume_ckpt}" --resume)
fi

launch_cmd=(
    accelerate launch
    --num_processes="${num_processes}"
    --main_process_port "${main_process_port}"
    --mixed_precision "${mixed_precision}"
    train_smolvlm.py
    "${train_args[@]}"
)

cat > "${result_dir}/simvla_launch.env" <<EOF
model_size=${model_size}
batch_size=${batch_size}
learning_rate=${learning_rate}
learning_coef=${learning_coef}
hidden_size=${hidden_size}
depth=${depth}
num_heads=${num_heads}
num_processes=${num_processes}
main_process_port=${main_process_port}
mixed_precision=${mixed_precision}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
simvla_dir=${SIMVLA_DIR}
result_dir=${result_dir}
output_dir=${output_dir}
hf_home=${HF_HOME}
EOF

printf '%q ' "${launch_cmd[@]}" > "${result_dir}/simvla_command.sh"
printf '\n' >> "${result_dir}/simvla_command.sh"

echo "[SIMVLA] model_size=${model_size}"
echo "[SIMVLA] simvla_dir=${SIMVLA_DIR}"
echo "[SIMVLA] result_dir=${result_dir}"
echo "[SIMVLA] output_dir=${output_dir}"
echo "[SIMVLA] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[SIMVLA] num_processes=${num_processes}"
echo "[SIMVLA] command file=${result_dir}/simvla_command.sh"

if [[ "${dry_run}" == "1" ]]; then
    echo "[SIMVLA] dry_run=1; launch command was written but training was not started."
    exit 0
fi

cd "${SIMVLA_DIR}"
"${launch_cmd[@]}"
