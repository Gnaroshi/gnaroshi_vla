#!/usr/bin/env bash
set -euo pipefail

ROOT=${SIMVLA_ROOT:-/home/mingyujung/private/gnaroshi_vla}
GENERATION_RESULT=${SIMVLA_GENERATION_RESULT_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/generation_control/20260824_v1/sd1_seed01_three_row_long500}
GENERATION_SUMMARY=${GENERATION_RESULT}/comparison/sd1_generation_control_summary.json
GENERATION_PROCESS_PATTERN=${GENERATION_RESULT}
POLL_SECONDS=${SIMVLA_WAIT_POLL_SECONDS:-60}
RESULT_BASE=${SIMVLA_RESULT_BASE:-/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla/results/simvla/latentloop/correct_native_v0_seed20260815_v1}
STATUS=${SIMVLA_KC2_EGL_WAIT_STATUS:-${RESULT_BASE}/logs/kc2_egl_wait_and_run.status}

if [[ "${SIMVLA_KC2_EGL_WAIT_RUN:-0}" != "1" ]]; then
  echo "SIMVLA_KC2_EGL_WAIT_RUN=1 is required." >&2
  exit 2
fi
mkdir -p "$(dirname "${STATUS}")"
printf 'WAITING generation_result=%s started_at=%s\n' \
  "${GENERATION_RESULT}" "$(date --iso-8601=seconds)" >"${STATUS}"

while [[ ! -f "${GENERATION_SUMMARY}" ]]; do
  if ! pgrep -af "${GENERATION_PROCESS_PATTERN}" >/dev/null; then
    printf 'BLOCKED generation process ended without summary at=%s\n' \
      "$(date --iso-8601=seconds)" >"${STATUS}"
    echo "Generation-control ended without its required summary: ${GENERATION_SUMMARY}" >&2
    exit 1
  fi
  echo "WAITING_FOR_GENERATION_CONTROL $(date --iso-8601=seconds)"
  sleep "${POLL_SECONDS}"
done

python - "${GENERATION_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
rows = payload.get("rows")
required_rows = ("full_nfe10", "naive_nfe3", "generation_ng3")
if not isinstance(rows, dict) or set(rows) != set(required_rows):
    raise SystemExit(f"generation summary does not contain the exact three rows: {path}")
for name in required_rows:
    row = rows[name]
    if row.get("verdict") != "GENERATION_CONTROL_ROW_PASS":
        raise SystemExit(f"generation row did not pass: {name}: {path}")
    if int(row.get("episodes", 0)) != 500:
        raise SystemExit(f"generation row is not complete at 500 episodes: {name}: {path}")
PY

if pgrep -af "${GENERATION_PROCESS_PATTERN}" >/dev/null; then
  echo "Summary exists; waiting for Generation-control processes to release host resources."
  while pgrep -af "${GENERATION_PROCESS_PATTERN}" >/dev/null; do
    sleep 10
  done
fi
sleep 30

printf 'RUNNING generation_summary=%s started_at=%s\n' \
  "${GENERATION_SUMMARY}" "$(date --iso-8601=seconds)" >"${STATUS}"
set +e
SIMVLA_KC2_EGL_RUN=1 \
  bash "${ROOT}/architectures/simvla/wrappers/run_condition_v0_kc2_egl_diagnostic.sh"
rc=$?
set -e
if (( rc == 0 )); then
  printf 'COMPLETE finished_at=%s\n' "$(date --iso-8601=seconds)" >"${STATUS}"
else
  printf 'FAILED exit_code=%s finished_at=%s\n' \
    "${rc}" "$(date --iso-8601=seconds)" >"${STATUS}"
fi
exit "${rc}"
