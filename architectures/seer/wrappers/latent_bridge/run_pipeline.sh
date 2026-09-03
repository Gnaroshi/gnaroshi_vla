#!/usr/bin/env bash

set -euo pipefail
wrapper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${wrapper_dir}/common.sh"
latent_bridge_require_runtime
mkdir -p "${LATENT_BRIDGE_RESULT_ROOT}"
exec 9>"${LATENT_BRIDGE_RESULT_ROOT}/pipeline.lock"
flock -n 9 || latent_bridge_fail "another Seer Latent Bridge pipeline owns the lock"

stages=(
    00_preflight.sh
    01_collect_continuity_sync.sh
    02_train_r0.sh
    03_collect_dagger.sh
    04_train_r1.sh
    05_evaluate.sh
)
for stage in "${stages[@]}"; do
    echo "[PIPELINE] starting ${stage}"
    if LATENT_BRIDGE_RETRY_PARTIAL="${LATENT_BRIDGE_RETRY_PARTIAL:-0}" \
        bash "${wrapper_dir}/${stage}"; then
        echo "[PIPELINE] completed ${stage}"
        continue
    fi
    echo "[PIPELINE][FAIL] ${stage}; partial output was preserved" >&2
    if [[ "${LATENT_BRIDGE_AUTO_RETRY_ONCE:-0}" != "1" ]]; then
        exit 1
    fi
    echo "[PIPELINE] retrying ${stage} once with the first partial attempt preserved"
    if ! LATENT_BRIDGE_RETRY_PARTIAL=1 bash "${wrapper_dir}/${stage}"; then
        echo "[PIPELINE][FAIL] bounded retry failed for ${stage}" >&2
        exit 1
    fi
    echo "[PIPELINE] completed ${stage} after one bounded retry"
done
printf 'SEER_LATENT_BRIDGE_PIPELINE_COMPLETE\n' > "${LATENT_BRIDGE_RESULT_ROOT}/PIPELINE_COMPLETE"
echo "[PIPELINE] all stages complete"
