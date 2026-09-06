#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Default excludes coupled generation, as requested. --method latentloop selects it explicitly.
exec bash "${script_dir}/deploy_doll_baseline.sh" --method condition_loop "$@"
