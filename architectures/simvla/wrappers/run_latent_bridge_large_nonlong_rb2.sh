#!/usr/bin/env bash
# Keep tmux interactive on failures; preserve the real status separately.
set -uo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
storage=/home/mingyujung/private/gnaroshi_vla_storage
output="$storage/results/simvla/latent_bridge/large_nonlong_three_seed_v1"
python_bin="$storage/envs/simvla/libero_mujoco237/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repo"
cd -- "$repo" || exit 0
"$python_bin" tools/simvla/run_latent_bridge_large_nonlong.py "${1:-preflight}"
rc=$?
printf 'LARGE_NONLONG_EXIT_CODE=%s\n' "$rc"
if [[ -d "$output" ]]; then
    printf '%s\n' "$rc" > "$output/launcher.exit_code"
fi
printf 'Results and real exit status: %s\n' "$output"
exit 0
