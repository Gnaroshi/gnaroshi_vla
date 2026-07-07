#!/usr/bin/env bash
set -euo pipefail

ROOT="${GNAROSHI_VLA_ROOT:-/home/mingyujung/private/gnaroshi_vla}"
UPSTREAM="${ROOT}/architectures/simvla/upstream"

cat <<EOF
SimVLA DCLD condition-hook equivalence check

Preconditions:
  - conda activate simvla_libero
  - export HF_HOME=${ROOT}/.cache/huggingface
  - use a real SimVLA-preprocessed batch
  - load the same checkpoint used for baseline, e.g. YuankaiLuo/SimVLA-LIBERO

Adapter target:
  condition = model.forward_vlm_efficient(...)[\"vlm_features\"]
  action = SimVLAActionAdapter.decode_action_from_condition(...)

This wrapper is dry-run by default. Set SIMVLA_DCLD_RUN=1 only after providing
a concrete batch loader for the equivalence smoke.
EOF

if [[ "${SIMVLA_DCLD_RUN:-0}" != "1" ]]; then
  exit 0
fi

echo "ERROR: executable batch-loader wiring is intentionally not guessed here." >&2
echo "Use SimVLAConditionAdapter.compare_hook_equivalence(batch, steps=10)." >&2
echo "Upstream path: ${UPSTREAM}" >&2
exit 2
