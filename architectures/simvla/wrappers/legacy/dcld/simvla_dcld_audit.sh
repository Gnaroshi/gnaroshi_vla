#!/usr/bin/env bash
set -euo pipefail

ROOT="${GNAROSHI_VLA_ROOT:-/home/mingyujung/private/gnaroshi_vla}"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/gnaroshi_vla_pycache}"

python -m compileall -q methods/dcld architectures/simvla/adapters/dcld
python - <<'PY'
from methods.dcld.modules import DCLDCore, DeltaObservation
from architectures.simvla.adapters.dcld import SimVLAActionAdapter, SimVLAConditionAdapter

print("DCLD imports OK")
print(DCLDCore.__name__, DeltaObservation.__name__)
print(SimVLAActionAdapter.__name__, SimVLAConditionAdapter.__name__)
PY
