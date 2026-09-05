#!/usr/bin/env bash
# Keep the interactive pane alive; the inner launcher/status retains real errors.
set +e
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SIMVLA_VLA_CACHE_LIBERO_RUN=1
bash "${HERE}/run_vla_cache_libero_long_rb2.sh" "${1:---all}"
rc=$?
if [[ ${rc} -ne 0 ]]; then
  printf '\nVLA_CACHE_STOPPED rc=%s. Inspect the preceding error and pipeline.status.\n' "${rc}"
fi
exit 0
