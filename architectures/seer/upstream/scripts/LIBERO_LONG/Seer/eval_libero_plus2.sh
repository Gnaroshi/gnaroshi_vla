#!/bin/bash

set -euo pipefail

echo "[DEPRECATED] eval_libero_plus2.sh had stale seer_main paths and invalid run_name syntax."
echo "[DEPRECATED] Delegating to eval_libero_plus.sh. Pass the suite as the first argument if needed."

exec bash scripts/LIBERO_LONG/Seer/eval_libero_plus.sh "$@"
