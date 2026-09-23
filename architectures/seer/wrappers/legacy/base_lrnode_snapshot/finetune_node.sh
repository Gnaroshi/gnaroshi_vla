#!/bin/bash

set -euo pipefail

echo "[DEPRECATED] scripts/LIBERO_LONG/Seer/finetune_node.sh has been renamed to distill_node.sh." >&2
echo "[DEPRECATED] This is frozen-baseline LR-NODE distillation, not Seer fine-tuning." >&2
echo "[DEPRECATED] Forwarding to scripts/LIBERO_LONG/Seer/distill_node.sh" >&2

exec bash scripts/LIBERO_LONG/Seer/distill_node.sh "$@"
