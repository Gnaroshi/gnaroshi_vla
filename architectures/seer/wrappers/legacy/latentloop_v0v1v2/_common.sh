#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -L)"
EXPECTED_REPO="/home/mingyujung/private/gnaroshi_vla_latentloop_canonical"
EXPECTED_REPO_REAL="/home/mingyujung/private/gnaroshi_vla_sd1_canonical_v0v1v2_20260821"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/home/mingyujung/shared/hdd_ext/nvme8000/mingyujung/gnaroshi_vla/results/seer/latentloop/teacher33_v0v1v2}"
S18_RUNTIME="${S18_RUNTIME:-/home/mingyujung/shared/hdd_ext/nvme8000/mingyujung/gnaroshi_vla/envs/seer_libero_canonical}"
S18_PYTHON="${S18_PYTHON:-${S18_RUNTIME}/bin/python}"
LIBERO_PATH="${LIBERO_PATH:-${REPO_ROOT}/.canonical/libero_source}"
DATASET_OUTER_ROOT="${S18_DATASET_ROOT:-/home/mingyujung/shared/hdd_ext/nvme8000/mingyujung/gnaroshi_vla/datasets/LIBERO_DATASETS/libero_10_converted}"
SOURCE_CONTRACT="${REPO_ROOT}/s18_canonical_source_lock.json"
GPU_CONTRACT="${REPO_ROOT}/four_gpu_execution_contract.json"
SOURCE_GATE_ARTIFACT="${REPO_ROOT}/.canonical/source_gate_pass.json"
PREFLIGHT_ARTIFACT="${CAMPAIGN_ROOT}/gates/s18_canonical_preflight.json"
K1_GATE="${CAMPAIGN_ROOT}/gates/k1_parity.json"
K4_GATE="${CAMPAIGN_ROOT}/gates/k4_call_path.json"
REPRO_DECISION="${CAMPAIGN_ROOT}/gates/canonical_reproduction_decision.json"
TEACHER="${REPO_ROOT}/.canonical/artifacts/teacher33.pth"
ADAPTER="${REPO_ROOT}/.canonical/artifacts/adapter39.pth"
VIT="${REPO_ROOT}/.canonical/artifacts/mae_pretrain_vit_base.pth"

export PATH="${S18_RUNTIME}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/architectures/seer/upstream:${LIBERO_PATH}:${PYTHONPATH:-}"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_base_identity() {
  [[ "$(hostname)" == "jbrserver18" ]] || fail "expected host jbrserver18"
  [[ "${REPO_ROOT}" == "${EXPECTED_REPO}" ]] || fail "wrong source tree: ${REPO_ROOT}"
  [[ "$(readlink -f "${REPO_ROOT}")" == "${EXPECTED_REPO_REAL}" ]] \
    || fail "stable source alias points to the wrong immutable tree"
  [[ -x "${S18_PYTHON}" ]] || fail "missing locked runtime: ${S18_PYTHON}"
  [[ -d "${LIBERO_PATH}" ]] || fail "missing LIBERO source: ${LIBERO_PATH}"
  [[ -f "${SOURCE_GATE_ARTIFACT}" ]] || fail "canonical import/source gate has not completed"
  CANONICAL_DATASET_META="${DATASET_OUTER_ROOT}/libero_10_converted/meta_info.h5" \
  CANONICAL_LIBERO_PATH="${LIBERO_PATH}" \
    "${S18_PYTHON}" "${REPO_ROOT}/source_gate.py" \
      --repo-root "${REPO_ROOT}" --contract "${SOURCE_CONTRACT}"
  "${S18_PYTHON}" "${REPO_ROOT}/tools/seer/check_process_visibility.py" \
    "${REPO_ROOT}" "${CAMPAIGN_ROOT}" "${S18_PYTHON}" "${LIBERO_PATH}" \
    "${TEACHER}" "${ADAPTER}" "${VIT}"
}

require_four_gpu_lane() {
  "${S18_PYTHON}" "${REPO_ROOT}/four_gpu_launcher_guard.py" \
    --contract "${GPU_CONTRACT}" --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-}" \
    --nproc-per-node 4
}

require_gate() {
  local path="$1" expected="$2"
  [[ -f "${path}" ]] || fail "required gate missing: ${path}"
  "${S18_PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("status",p.get("verdict"))==sys.argv[2], p' "${path}" "${expected}" \
    || fail "required gate did not pass: ${path}"
}

require_reproduction_pass() {
  [[ -f "${REPRO_DECISION}" ]] || fail "canonical reproduction decision is missing"
  "${S18_PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("v1_training_enabled") is True, p' "${REPRO_DECISION}" \
    || fail "canonical reproduction did not enable V1"
}

refuse_existing() {
  [[ ! -e "$1" ]] || fail "refusing to overwrite existing path: $1"
}

run_v0_eval() {
  local result_root="$1" episodes="$2" tasks="$3" run_baseline="$4" run_ours_full="$5" intervals="$6" port="$7"
  refuse_existing "${result_root}"
  (
    export LIBERO_GL_BACKEND=osmesa MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
    export LIBERO_GL_REQUIRE_ACTUAL=1
    export SAVE_VIDEO=0 SAVE_VIDEO_SUCC=0 SAVE_VIDEO_FAIL=0 SAVE_VIDEO_ALL_RANKS=0
    export EVAL_NUM_EPISODES_PER_TASK="${episodes}" EVAL_NUM_TASKS="${tasks}"
    export EVAL_CONTROL_HZ=20 EVAL_LIBERO_ENSEMBLING=1 LIBERO_EVAL_MAX_STEPS=600
    export BASELINE_CKPT="${TEACHER}" BASELINE_CKPT_ID=33 BASELINE_NAME=sd1_teacher33
    export OURS_CKPT="${ADAPTER}" OURS_CKPT_ID=39 OURS_NAME=latentloop_v0_adapter39
    export LRNODE_EVAL_BASE_CKPT="${TEACHER}" LRNODE_TRAIN_PROTOCOL=adapter
    export LRNODE_FREEZE_SEER_FOR_ADAPTER=1 LRNODE_ASSERT_ONLY_LRNODE_TRAINABLE=1
    export LRNODE_EVAL_STEP_LOG=1 LRNODE_EVAL_SHADOW_FULL_FORWARD=0
    export LRNODE_EVAL_REFRESH_POLICY=periodic LRNODE_EVAL_ABLATION_MODE=stepwise
    export RUN_BASELINE="${run_baseline}" RUN_OURS_FULL="${run_ours_full}"
    export LRNODE_QUERY_INTERVALS_STR="${intervals}" NODE_NUM=4 MASTER_PORT="${port}"
    export RESULT_ROOT="${result_root}" LIBERO_PATH="${LIBERO_PATH}"
    export VIT_CHECKPOINT_PATH="${VIT}"
    cd "${REPO_ROOT}"
    bash architectures/seer/upstream/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh
  )
}
