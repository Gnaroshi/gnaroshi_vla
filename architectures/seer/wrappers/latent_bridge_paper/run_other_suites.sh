#!/usr/bin/env bash

set -euo pipefail

wrapper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${wrapper_dir}/../../../.." && pwd)"
expected_repo="${EXPECTED_REPO:-/home/mingyujung/private/gnaroshi_vla_latent_bridge_paper}"
shared_root=/home/mingyujung/shared/nvme1/mingyujung/robotics/seer
suite_study_root="${shared_root}/libero_suite_study"
paper_root="${LATENT_BRIDGE_PAPER_ROOT:-${shared_root}/paper_latent_bridge}"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ "$(hostname)" == "${EXPECTED_HOST:-jbrserver1}" ]] || fail "this launcher is locked to sd1"
[[ "$(readlink -f "${repo_root}")" == "$(readlink -f "${expected_repo}")" ]] || \
    fail "unexpected source: ${repo_root}"
[[ "${CUDA_VISIBLE_DEVICES:-}" == "4,5,6,7" ]] || \
    fail "set CUDA_VISIBLE_DEVICES=4,5,6,7"

source /home/mingyujung/miniconda3/etc/profile.d/conda.sh

run_suite() {
    local suite="$1"
    local checkpoint="$2"
    local checkpoint_sha256="$3"
    local conda_env="$4"
    local port="$5"
    local dataset_name="${suite}_converted"

    echo "============================================================"
    echo "[SUITE START] ${suite} env=${conda_env} GPUs=4,5,6,7"
    echo "[SUITE START] teacher=${checkpoint}"
    echo "============================================================"
    conda activate "${conda_env}"
    if [[ "${suite}" == "libero_object" ]]; then
        python - <<'PY'
import mujoco
if mujoco.__version__ != "3.3.2":
    raise RuntimeError(f"LIBERO-Object requires MuJoCo 3.3.2, got {mujoco.__version__}")
print("[VERIFY][OK] LIBERO-Object MuJoCo 3.3.2")
PY
    fi

    LATENT_BRIDGE_EXPECTED_CONDA_ENV="${conda_env}" \
    LATENT_BRIDGE_EXPECTED_CUDA_VISIBLE_DEVICES=4,5,6,7 \
    LATENT_BRIDGE_BASE_CHECKPOINT="${checkpoint}" \
    LATENT_BRIDGE_BASE_CHECKPOINT_SHA256="${checkpoint_sha256}" \
    LATENT_BRIDGE_VIT="${shared_root}/vit_mae/mae_pretrain_vit_base.pth" \
    LATENT_BRIDGE_DATASET_ROOT="${suite_study_root}/datasets" \
    LATENT_BRIDGE_DATASET_NAME="${dataset_name}" \
    LATENT_BRIDGE_DATASET_INFO="${suite_study_root}/datasets/${dataset_name}/data_info.json" \
    LATENT_BRIDGE_LIBERO_PATH=/home/mingyujung/private/LIBERO \
    LATENT_BRIDGE_RESULT_ROOT="${paper_root}/${suite}_feature_bridge_v1" \
    LATENT_BRIDGE_RENDERER=egl \
    LATENT_BRIDGE_SUITE="${suite}" \
    LATENT_BRIDGE_RUN_LABEL="seer_${suite}_latent_bridge" \
    SEER_LATENT_BRIDGE_DETERMINISTIC=0 \
    EVAL_SEEDS="${EVAL_SEEDS:-42 43 44}" \
    REFRESH_PERIODS="${REFRESH_PERIODS:-3 4}" \
    EVAL_EPISODES_PER_TASK=50 \
    EVAL_NUM_TASKS=10 \
    RUN_BASELINE=1 \
    RUN_COMPONENT_LATENCY=0 \
    R0_OPTIMIZER_STEPS="${R0_OPTIMIZER_STEPS:-50200}" \
    R1_OPTIMIZER_STEPS="${R1_OPTIMIZER_STEPS:-44000}" \
    MASTER_PORT_BASE="${port}" NODE_NUM=4 \
    bash "${wrapper_dir}/run_suite_pipeline.sh"
    echo "[SUITE DONE] ${suite}"
}

run_suite \
    libero_spatial \
    "${suite_study_root}/campaigns/spatial_object_goal_v1/train/libero_spatial/baseline/seer_libero_spatial_scratch_seed42/33.pth" \
    05965065657b3a8797df034292baed648d73e4c3c1f2466617781242a945925c \
    seer_libero "$(( ${MASTER_PORT_BASE:-19200} + 0 ))"

run_suite \
    libero_object \
    "${suite_study_root}/campaigns/spatial_object_goal_v1/train/libero_object/baseline/seer_libero_object_scratch_seed42/39.pth" \
    d9b8833ff8fc2819736f6f13862b488bce90c132973391552bf6c6f60dbbc417 \
    seer_libero_mojoco332 "$(( ${MASTER_PORT_BASE:-19200} + 300 ))"

run_suite \
    libero_goal \
    "${suite_study_root}/campaigns/spatial_object_goal_v1/train/libero_goal/baseline/seer_libero_goal_scratch_seed42/39.pth" \
    76da3ba380cf0fc730c9dc29e5dcfddd3b5db7082f1e873967b8031eb000afec \
    seer_libero "$(( ${MASTER_PORT_BASE:-19200} + 600 ))"

printf 'SEER_LATENT_BRIDGE_OTHER_SUITES_COMPLETE\n' > \
    "${paper_root}/OTHER_SUITES_COMPLETE"
echo "[DONE] Spatial, Object, and Goal pipelines are complete"
