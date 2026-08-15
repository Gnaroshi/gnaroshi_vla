#!/usr/bin/env bash
set -euo pipefail

if [[ "${SIMVLA_LATENTLOOP_DETERMINISTIC_PAIR_RUN:-0}" != "1" ]]; then
  echo "Refusing paired repeat: set SIMVLA_LATENTLOOP_DETERMINISTIC_PAIR_RUN=1." >&2
  exit 2
fi

ROOT=$(git rev-parse --show-toplevel)
EVAL_WRAPPER="$ROOT/architectures/simvla/wrappers/simvla_latentloop_eval.sh"
output_root=""
has_experiment_seed=0
forwarded=()
arguments=("$@")

for ((index = 0; index < ${#arguments[@]}; index++)); do
  argument="${arguments[$index]}"
  case "$argument" in
    --output)
      ((index + 1 < ${#arguments[@]})) || {
        echo "--output requires a value" >&2
        exit 2
      }
      output_root="${arguments[$((index + 1))]}"
      ((index += 1))
      ;;
    --output=*)
      output_root="${argument#--output=}"
      ;;
    --repeat-reference-output|--determinism-reference-manifest)
      echo "$argument is managed by this paired-repeat wrapper" >&2
      exit 2
      ;;
    --repeat-reference-output=*|--determinism-reference-manifest=*)
      echo "${argument%%=*} is managed by this paired-repeat wrapper" >&2
      exit 2
      ;;
    --experiment-seed|--experiment-seed=*)
      has_experiment_seed=1
      forwarded+=("$argument")
      ;;
    *)
      forwarded+=("$argument")
      ;;
  esac
done

[[ -n "$output_root" ]] || {
  echo "--output OUTPUT_ROOT is required" >&2
  exit 2
}
((has_experiment_seed == 1)) || {
  echo "--experiment-seed is required so one seed controls every RNG namespace" >&2
  exit 2
}
output_root=$(realpath -m "$output_root")
if [[ -e "$output_root" ]]; then
  echo "Refusing to reuse deterministic pair output: $output_root" >&2
  exit 2
fi

repeat01="$output_root/repeat01"
repeat02="$output_root/repeat02"
export SIMVLA_LATENTLOOP_EVAL_RUN=1

echo "[determinism] repeat01 -> $repeat01"
bash "$EVAL_WRAPPER" online --output "$repeat01" "${forwarded[@]}"

echo "[determinism] repeat02 -> $repeat02"
bash "$EVAL_WRAPPER" online \
  --output "$repeat02" \
  --repeat-reference-output "$repeat01" \
  "${forwarded[@]}"

bash "$EVAL_WRAPPER" verify-repeat \
  --reference "$repeat01" \
  --candidate "$repeat02" \
  --output "$output_root/deterministic_pair_summary.json"

echo "[pass] exact same-seed trajectory replay verified: $output_root"
