#!/usr/bin/env bash
# Run only post-training model checks and missing coupled non-Long evaluations.
set -uo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
python_bin=/home/mingyujung/private/gnaroshi_vla_storage/envs/simvla/libero_mujoco237/bin/python
cd -- "$repo" || exit 1
export PYTHONPATH="$repo"
export PYTHONDONTWRITEBYTECODE=1
"$python_bin" tools/simvla/run_completed_followup.py "${1:-preflight}"
rc=$?
printf 'FOLLOWUP_EXIT_CODE=%s\n' "$rc"
status_root=/home/mingyujung/private/gnaroshi_vla_storage/results/simvla/paper_completion/coupled_nonlong_three_seed_v1
if [[ -d "$status_root" ]]; then
    printf '%s\n' "$rc" > "$status_root/launcher.exit_code"
fi
if (( rc != 0 )); then
    printf 'Inspect results/simvla/paper_completion/coupled_nonlong_three_seed_v1 under rb2 storage.\n'
fi
# The actual exit code is printed; an interactive tmux shell must remain open.
exit 0
