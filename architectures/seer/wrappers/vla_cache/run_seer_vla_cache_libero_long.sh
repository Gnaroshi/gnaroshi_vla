#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
UPSTREAM=${REPO_ROOT}/architectures/seer/upstream
ANALYZER=${REPO_ROOT}/tools/seer/analyze_seer_vla_cache.py
PYTHON_BIN=${PYTHON_BIN:-/home/mingyujung/miniconda3/envs/seer_libero/bin/python}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
NODE_NUM=${NODE_NUM:-4}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-18100}
EVAL_SEED=${EVAL_SEED:-42}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-50}
NUM_TASKS=${NUM_TASKS:-10}
MODES_STR=${MODES_STR:-"off matched_full reuse"}
CAMPAIGN_TAG=${CAMPAIGN_TAG:-public33_long_seed${EVAL_SEED}_r1}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
SAVE_VIDEO=${SAVE_VIDEO:-0}
LIBERO_GL_BACKEND=${LIBERO_GL_BACKEND:-egl}

PUBLIC33=${PUBLIC33:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/artifacts/checkpoints/seer/paper/libero_long/teacher_public33.pth}
PUBLIC33_SHA256=${PUBLIC33_SHA256:-a74f200bb91618a27cbb8e25bc6e1008647056ebe4155348095d63b658936646}
VIT_CHECKPOINT_PATH=${VIT_CHECKPOINT_PATH:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/vit_mae/mae_pretrain_vit_base.pth}
VIT_SHA256=${VIT_SHA256:-aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d}
CLIP_CHECKPOINT_PATH=${CLIP_CHECKPOINT_PATH:-/home/mingyujung/.cache/clip/ViT-B-32.pt}
LIBERO_PATH=${LIBERO_PATH:-/home/mingyujung/private/LIBERO}
RESULT_ROOT=${RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/seer/vla_cache/${CAMPAIGN_TAG}}

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "missing or empty file: $1"
}

verify_sha256() {
    local path=$1 expected=$2 actual
    actual=$(sha256sum "${path}" | awk '{print $1}')
    [[ "${actual}" == "${expected}" ]] || fail "sha256 mismatch: ${path}; expected=${expected}; actual=${actual}"
    echo "[VERIFY][OK] $(basename "${path}") sha256=${actual}"
}

[[ -x "${PYTHON_BIN}" ]] || fail "Python executable not found: ${PYTHON_BIN}"
require_file "${PUBLIC33}"
require_file "${VIT_CHECKPOINT_PATH}"
require_file "${CLIP_CHECKPOINT_PATH}"
require_file "${ANALYZER}"
[[ -d "${LIBERO_PATH}/libero" ]] || fail "invalid LIBERO root: ${LIBERO_PATH}"
verify_sha256 "${PUBLIC33}" "${PUBLIC33_SHA256}"
verify_sha256 "${VIT_CHECKPOINT_PATH}" "${VIT_SHA256}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#GPU_IDS[@]}" -eq "${NODE_NUM}" ]] || fail "NODE_NUM=${NODE_NUM} but CUDA_VISIBLE_DEVICES exposes ${#GPU_IDS[@]} devices"
case "${CUDA_VISIBLE_DEVICES}" in
    0,1,2,3|4,5,6,7) ;;
    *) fail "sd1 Seer jobs require one complete four-GPU partition: 0,1,2,3 or 4,5,6,7; got ${CUDA_VISIBLE_DEVICES}" ;;
esac
(( EPISODES_PER_TASK > 0 )) || fail "EPISODES_PER_TASK must be positive"
(( NUM_TASKS > 0 && NUM_TASKS <= 10 )) || fail "NUM_TASKS must be in [1,10]"
(( EPISODES_PER_TASK * NUM_TASKS >= NODE_NUM )) || fail "total episodes must be at least NODE_NUM"

read -r -a MODES <<< "${MODES_STR}"
(( ${#MODES[@]} > 0 )) || fail "MODES_STR is empty"
for mode in "${MODES[@]}"; do
    case "${mode}" in
        off|matched_full|reuse) ;;
        *) fail "unsupported VLA-Cache mode: ${mode}" ;;
    esac
done

export CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LIBERO_GL_BACKEND
export MUJOCO_GL=${LIBERO_GL_BACKEND}
export PYOPENGL_PLATFORM=${LIBERO_GL_BACKEND}
export PYTHONPATH="${REPO_ROOT}:${UPSTREAM}:${LIBERO_PATH}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-/tmp/numba_cache_${USER}}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib_${USER}}
export EVAL_NUM_EPISODES_PER_TASK=${EPISODES_PER_TASK}
export EVAL_NUM_TASKS=${NUM_TASKS}
export EVAL_CONTROL_HZ=${EVAL_CONTROL_HZ:-20}
export SAVE_VIDEO
export SAVE_VIDEO_SUCC=${SAVE_VIDEO_SUCC:-1}
export SAVE_VIDEO_FAIL=${SAVE_VIDEO_FAIL:-1}
export SAVE_VIDEO_ALL_RANKS=${SAVE_VIDEO_ALL_RANKS:-0}

"${PYTHON_BIN}" - <<PY
import torch

expected = int(${NODE_NUM})
actual = torch.cuda.device_count()
if actual != expected:
    raise SystemExit(
        f"[ERROR] CUDA runtime exposes {actual} devices, expected {expected}; "
        "check CUDA_VISIBLE_DEVICES"
    )
print("[PREFLIGHT][PASS] CUDA devices:", [torch.cuda.get_device_name(i) for i in range(actual)])
PY

"${PYTHON_BIN}" -m pytest -q -p no:cacheprovider "${REPO_ROOT}/tests/seer_vla_cache"
"${PYTHON_BIN}" - <<'PY'
from models.vla_cache import build_seer_vla_cache_config

config = build_seer_vla_cache_config(
    mode="reuse",
    pruning_layers="2,6,9,11",
    reference_attention_layer=15,
    similarity_threshold=0.996,
    positive_growth_factor=0.55,
    transformer_layers=24,
    sequence_length=7,
    num_resampler_query=6,
    num_obs_token_per_image=9,
    obs_pred=True,
    action_pred_steps=3,
)
assert config["total_tokens"] == 259
assert config["visual_tokens_per_view"] == 7
assert config["stable_top_k"] == 4
assert config["task_relevant_top_k"] == 3
print("[PREFLIGHT][PASS] Seer public33 token layout and VLA-Cache contract")
PY
(
    cd "${UPSTREAM}"
    "${PYTHON_BIN}" eval_libero.py --help >/dev/null
)
echo "[PREFLIGHT][PASS] evaluator imports and VLA-Cache CLI"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
    echo "[PREFLIGHT][PASS] no evaluation launched"
    exit 0
fi

mkdir -p "${RESULT_ROOT}"
[[ ! -e "${RESULT_ROOT}/campaign_complete.txt" ]] || fail "campaign is already complete: ${RESULT_ROOT}"
{
    echo "architecture=seer"
    echo "method=vla_cache"
    echo "campaign_tag=${CAMPAIGN_TAG}"
    echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "git_status_porcelain_begin"
    git -C "${REPO_ROOT}" status --short
    echo "git_status_porcelain_end"
    echo "public33=${PUBLIC33}"
    echo "public33_sha256=${PUBLIC33_SHA256}"
    echo "vit_checkpoint=${VIT_CHECKPOINT_PATH}"
    echo "vit_sha256=${VIT_SHA256}"
    echo "clip_checkpoint=${CLIP_CHECKPOINT_PATH}"
    echo "libero_path=${LIBERO_PATH}"
    echo "renderer=${LIBERO_GL_BACKEND}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "node_num=${NODE_NUM}"
    echo "eval_seed=${EVAL_SEED}"
    echo "episodes_per_task=${EPISODES_PER_TASK}"
    echo "num_tasks=${NUM_TASKS}"
    echo "modes=${MODES_STR}"
    echo "vla_cache_official_commit=a4909880573868dee2769343d52e793c0341678b"
    echo "vla_cache_transformers_commit=2302fce58afa3a4f8461625b1394f9e9c8a7f1ea"
} > "${RESULT_ROOT}/campaign_manifest.txt"
git -C "${REPO_ROOT}" diff --binary > "${RESULT_ROOT}/source_worktree.patch"
sha256sum \
    "${UPSTREAM}/models/vla_cache.py" \
    "${UPSTREAM}/models/gpt2.py" \
    "${UPSTREAM}/models/seer_model.py" \
    "${UPSTREAM}/eval_libero.py" \
    "${UPSTREAM}/utils/arguments_utils.py" \
    "${UPSTREAM}/utils/eval_utils_libero.py" \
    "${BASH_SOURCE[0]}" \
    "${ANALYZER}" \
    > "${RESULT_ROOT}/source_files.sha256"

COMMON_ARGS=(
    --traj_cons
    --rgb_pad 10
    --gripper_pad 4
    --gradient_accumulation_steps 1
    --bf16_module vision_encoder
    --vit_checkpoint_path "${VIT_CHECKPOINT_PATH}"
    --libero_path "${LIBERO_PATH}"
    --calvin_dataset ""
    --workers 16
    --lr_scheduler cosine
    --save_every_iter 50000
    --num_epochs 20
    --seed "${EVAL_SEED}"
    --batch_size 64
    --precision fp32
    --weight_decay 1e-4
    --num_resampler_query 6
    --transformer_layers 24
    --phase evaluate
    --finetune_type libero_10
    --save_checkpoint_path "${RESULT_ROOT}/unused_checkpoints"
    --action_pred_steps 3
    --future_steps 3
    --sequence_length 7
    --obs_pred
    --gripper_width
    --eval_libero_ensembling
    --multi_step_action 1
    --lrnode_eval_step_log 1
    --use_lrnode_latent_update 0
    --lrnode_eval_skip_full_forward 0
)

run_mode() {
    local mode=$1 index=$2
    local row_root=${RESULT_ROOT}/${mode}
    local summary=${row_root}/analysis/eval_summary.json
    if [[ -s "${summary}" ]]; then
        "${PYTHON_BIN}" - "${summary}" "${mode}" "$((EPISODES_PER_TASK * NUM_TASKS))" <<'PY'
import json
import sys

path, expected_mode, expected_episodes = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = json.load(open(path, encoding="utf-8"))
actual_mode = data.get("vla_cache", {}).get("mode", "off")
episodes = sum(int(row.get("num_episodes", 0)) for row in data.get("task_results", []))
if actual_mode != expected_mode or episodes != expected_episodes:
    raise SystemExit(
        f"existing row mismatch: mode={actual_mode}, episodes={episodes}; "
        f"expected mode={expected_mode}, episodes={expected_episodes}"
    )
print(f"[RESUME][SKIP] validated complete row: {path}")
PY
        return
    fi
    [[ ! -e "${row_root}" ]] || fail "partial row exists; inspect before rerun: ${row_root}"
    mkdir -p "${row_root}"
    export LOG_DIR=${row_root}
    export RUN_NAME=seer_public33_vla_cache_${mode}_seed${EVAL_SEED}
    export CKPT_TAG=public33_vla_cache_${mode}
    echo "[RUN] mode=${mode} seed=${EVAL_SEED} episodes=$((EPISODES_PER_TASK * NUM_TASKS)) output=${row_root}"
    (
        cd "${UPSTREAM}"
        "${PYTHON_BIN}" -m torch.distributed.run \
            --nnodes=1 \
            --nproc_per_node="${NODE_NUM}" \
            --master_port="$((MASTER_PORT_BASE + index))" \
            eval_libero.py \
            "${COMMON_ARGS[@]}" \
            --run_name "${RUN_NAME}" \
            --vla_cache_mode "${mode}" \
            --vla_cache_pruning_layers 2,6,9,11 \
            --vla_cache_reference_attention_layer 15 \
            --vla_cache_similarity_threshold 0.996 \
            --vla_cache_growth_factor 0.55 \
            --resume_from_checkpoint "${PUBLIC33}"
    ) 2>&1 | tee "${row_root}/eval.log"
    require_file "${summary}"
}

for index in "${!MODES[@]}"; do
    run_mode "${MODES[$index]}" "${index}"
done

if [[ " ${MODES_STR} " == *" off "* && " ${MODES_STR} " == *" matched_full "* && " ${MODES_STR} " == *" reuse "* ]]; then
    "${PYTHON_BIN}" "${ANALYZER}" \
        --campaign-root "${RESULT_ROOT}" \
        --expected-episodes "$((EPISODES_PER_TASK * NUM_TASKS))"
    date --iso-8601=seconds > "${RESULT_ROOT}/campaign_complete.txt"
else
    echo "[DONE] requested subset complete; all-mode campaign marker was not written"
fi
echo "[DONE] ${RESULT_ROOT}"
