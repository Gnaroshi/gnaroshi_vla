#!/bin/bash

set -euo pipefail

echo "[DEPRECATED] eval_lrnode_adapter_compare.sh has been renamed to eval_lrnode_distill_compare.sh." >&2
echo "[DEPRECATED] Forwarding to scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh" >&2

exec bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh "$@"
