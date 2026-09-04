#!/usr/bin/env bash

set -euo pipefail
wrapper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${wrapper_dir}/common.sh"
latent_bridge_require_runtime
mkdir -p "${LATENT_BRIDGE_RESULT_ROOT}"
exec 9>"${LATENT_BRIDGE_RESULT_ROOT}/pipeline.lock"
flock -n 9 || latent_bridge_fail "another efficient Latent Bridge pipeline owns the lock"
latent_bridge_efficient_initialize_source_lock
exec > >(tee -a "${LATENT_BRIDGE_RESULT_ROOT}/pipeline.log") 2>&1

rm -f "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_FAILED"
printf '%s\n' "$$" > "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_RUNNING"
pipeline_complete=0
pipeline_exit() {
    local rc=$?
    rm -f "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_RUNNING"
    if [[ "${pipeline_complete}" -ne 1 ]]; then
        printf 'exit_code=%s\nfinished_at=%s\n' \
            "${rc}" "$(date --iso-8601=seconds)" > \
            "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_FAILED"
    fi
}
trap pipeline_exit EXIT

stages=(
    00_preflight.sh
    01_train_r0.sh
    02_collect_dagger.sh
    03_train_r1.sh
    04_evaluate.sh
)
for stage in "${stages[@]}"; do
    echo "[PIPELINE] starting ${stage}"
    if bash "${wrapper_dir}/${stage}"; then
        echo "[PIPELINE] completed ${stage}"
        continue
    fi
    echo "[PIPELINE][FAIL] ${stage}; invoking one bounded resume/retry" >&2
    if ! bash "${wrapper_dir}/${stage}"; then
        echo "[PIPELINE][FAIL] bounded retry failed for ${stage}" >&2
        exit 1
    fi
    echo "[PIPELINE] completed ${stage} after one bounded retry"
done
printf 'SEER_LATENT_BRIDGE_EFFICIENT_PIPELINE_COMPLETE\n' > \
    "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_COMPLETE"
pipeline_complete=1
rm -f "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_FAILED"
echo "[PIPELINE] all efficient stages complete"
