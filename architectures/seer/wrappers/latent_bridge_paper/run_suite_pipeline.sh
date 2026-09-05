#!/usr/bin/env bash

set -euo pipefail

wrapper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${wrapper_dir}/../latent_bridge/common.sh"
latent_bridge_require_runtime

export LATENT_BRIDGE_EFFICIENT_SYNC_STAGE="${LATENT_BRIDGE_RESULT_ROOT}/continuity_and_sync"
export LATENT_BRIDGE_EFFICIENT_PREFLIGHT_ROOT="${LATENT_BRIDGE_RESULT_ROOT}/training_preflight"
export SEER_LATENT_BRIDGE_DETERMINISTIC="${SEER_LATENT_BRIDGE_DETERMINISTIC:-0}"
export EVAL_SEEDS="${EVAL_SEEDS:-42 43 44}"
export REFRESH_PERIODS="${REFRESH_PERIODS:-3 4}"
export EVAL_EPISODES_PER_TASK="${EVAL_EPISODES_PER_TASK:-50}"
export EVAL_NUM_TASKS="${EVAL_NUM_TASKS:-10}"
export RUN_COMPONENT_LATENCY="${RUN_COMPONENT_LATENCY:-0}"

mkdir -p "${LATENT_BRIDGE_RESULT_ROOT}"
exec 9>"${LATENT_BRIDGE_RESULT_ROOT}/pipeline.lock"
flock -n 9 || latent_bridge_fail "another pipeline owns ${LATENT_BRIDGE_RESULT_ROOT}"

current_head="$(git -C "${LATENT_BRIDGE_REPO_ROOT}" rev-parse HEAD)"
[[ -z "$(git -C "${LATENT_BRIDGE_REPO_ROOT}" status --porcelain --untracked-files=no)" ]] || \
    latent_bridge_fail "source worktree has tracked modifications"
if [[ -s "${LATENT_BRIDGE_RESULT_ROOT}/SOURCE_LOCK" ]]; then
    [[ "$(<"${LATENT_BRIDGE_RESULT_ROOT}/SOURCE_LOCK")" == "${current_head}" ]] || \
        latent_bridge_fail "result root is locked to another source commit"
else
    printf '%s\n' "${current_head}" > "${LATENT_BRIDGE_RESULT_ROOT}/SOURCE_LOCK"
fi

exec > >(tee -a "${LATENT_BRIDGE_RESULT_ROOT}/pipeline.log") 2>&1
rm -f "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_FAILED"
printf '%s\n' "$$" > "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_RUNNING"
pipeline_complete=0
pipeline_exit() {
    local rc=$?
    rm -f "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_RUNNING"
    if [[ "${pipeline_complete}" -ne 1 ]]; then
        printf 'exit_code=%s\nfinished_at=%s\n' \
            "${rc}" "$(date --iso-8601=seconds)" > "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_FAILED"
    fi
}
trap pipeline_exit EXIT

stages=(
    "${wrapper_dir}/../latent_bridge/00_preflight.sh"
    "${wrapper_dir}/../latent_bridge/01_collect_continuity_sync.sh"
    "${wrapper_dir}/../latent_bridge_efficient/00_preflight.sh"
    "${wrapper_dir}/../latent_bridge_efficient/01_train_r0.sh"
    "${wrapper_dir}/../latent_bridge_efficient/02_collect_dagger.sh"
    "${wrapper_dir}/../latent_bridge_efficient/03_train_r1.sh"
    "${wrapper_dir}/evaluate.sh"
)

for stage in "${stages[@]}"; do
    echo "[PIPELINE] starting $(basename "${stage}") suite=${LATENT_BRIDGE_SUITE}"
    if bash "${stage}"; then
        echo "[PIPELINE] completed $(basename "${stage}")"
        continue
    fi
    echo "[PIPELINE][WARN] first attempt failed; retrying once: $(basename "${stage}")" >&2
    if ! LATENT_BRIDGE_RETRY_PARTIAL=1 bash "${stage}"; then
        latent_bridge_fail "bounded retry failed: $(basename "${stage}")"
    fi
    echo "[PIPELINE] completed after bounded retry: $(basename "${stage}")"
done

printf 'SEER_LATENT_BRIDGE_SUITE_PIPELINE_COMPLETE\n' > \
    "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_COMPLETE"
pipeline_complete=1
rm -f "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_FAILED"
echo "[DONE] suite pipeline complete: ${LATENT_BRIDGE_RESULT_ROOT}"
