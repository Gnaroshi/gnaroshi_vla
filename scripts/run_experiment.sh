#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

architecture="seer"
method="lrnode"
env_id="seer_libero"
node="lrnode"
experiment="seer_lrnode_debug"
action="sanity"

for arg in "$@"; do
    if [[ "${arg}" == *=* ]]; then
        key="${arg%%=*}"
        value="${arg#*=}"
        case "${key}" in
            architecture) architecture="${value}" ;;
            method) method="${value}" ;;
            env) env_id="${value}" ;;
            node) node="${value}" ;;
            experiment) experiment="${value}" ;;
            action) action="${value}" ;;
            *) echo "[WARN] ignoring unknown argument: ${arg}" >&2 ;;
        esac
    else
        echo "[WARN] ignoring non key=value argument: ${arg}" >&2
    fi
done

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
run_id="${RUN_ID:-$(date +%s)}"
result_dir="${ROOT_DIR}/results/${architecture}/${method}/${experiment}/${timestamp}_${run_id}"

mkdir -p \
    "${result_dir}/env_snapshot" \
    "${result_dir}/git_snapshot" \
    "${result_dir}/logs" \
    "${result_dir}/metrics" \
    "${result_dir}/checkpoints"

cat > "${result_dir}/checkpoints/README.md" <<'EOF'
Large checkpoints are not copied into run directories by default.
Record checkpoint paths or symlinks in run_manifest.yaml and notes.md.
EOF

echo "[RUN CONTEXT] architecture=${architecture}"
echo "[RUN CONTEXT] method=${method}"
echo "[RUN CONTEXT] env=${env_id}"
echo "[RUN CONTEXT] node=${node}"
echo "[RUN CONTEXT] experiment=${experiment}"
echo "[RUN CONTEXT] action=${action}"
echo "[RUN CONTEXT] result_dir=${result_dir}"

case "${env_id}" in
    seer_libero|simvla_libero)
        if command -v conda >/dev/null 2>&1; then
            # shellcheck disable=SC1091
            eval "$(conda shell.bash hook)"
            conda activate "${env_id}"
        else
            echo "[WARN] conda not found; continuing with current shell environment" >&2
        fi
        ;;
    *)
        echo "[WARN] no activation rule implemented for env=${env_id}; continuing with current shell environment" >&2
        ;;
esac

echo "[PYTHON] $(command -v python)"
python --version
python "${ROOT_DIR}/scripts/print_run_context.py" \
    architecture="${architecture}" method="${method}" env="${env_id}" node="${node}" experiment="${experiment}" \
    > "${result_dir}/run_context.json"

python "${ROOT_DIR}/scripts/collect_env_snapshot.py" \
    --env-id "${env_id}" \
    --output-dir "${result_dir}/env_snapshot" \
    --include-conda \
    --include-pip

if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    {
        git -C "${ROOT_DIR}" rev-parse HEAD
        git -C "${ROOT_DIR}" status --short
        git -C "${ROOT_DIR}" remote -v
    } > "${result_dir}/git_snapshot/gnaroshi_vla.txt" 2>&1
else
    echo "not a git repository" > "${result_dir}/git_snapshot/gnaroshi_vla.txt"
fi

upstream_dir="${ROOT_DIR}/architectures/${architecture}/upstream"
if git -C "${upstream_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    {
        git -C "${upstream_dir}" rev-parse HEAD
        git -C "${upstream_dir}" status --short
        git -C "${upstream_dir}" remote -v
    } > "${result_dir}/git_snapshot/architectures_${architecture}_upstream.txt" 2>&1
else
    echo "not a git repository" > "${result_dir}/git_snapshot/architectures_${architecture}_upstream.txt"
fi

cat > "${result_dir}/composed_config.yaml" <<EOF
architecture: ${architecture}
method: ${method}
env: ${env_id}
node: ${node}
experiment: ${experiment}
action: ${action}
config_files:
  - configs/config.yaml
  - configs/architecture/${architecture}.yaml
  - configs/method/${method}.yaml
  - configs/env/${env_id}.yaml
  - configs/node/${node}.yaml
  - configs/experiment/${experiment}.yaml
EOF

case "${action}" in
    sanity)
        run_cmd=(
            python "${ROOT_DIR}/scripts/sanity_check.py"
            "architecture=${architecture}"
            "method=${method}"
            "env=${env_id}"
            "node=${node}"
            "experiment=${experiment}"
        )
        ;;
    seer_script)
        if [[ "${architecture}" != "seer" ]]; then
            echo "[ERROR] action=seer_script is only valid for architecture=seer" >&2
            exit 2
        fi
        if [[ -z "${SEER_SCRIPT:-}" ]]; then
            echo "[ERROR] Set SEER_SCRIPT to a path relative to architectures/seer/upstream" >&2
            exit 2
        fi
        run_cmd=(bash "${upstream_dir}/${SEER_SCRIPT}")
        ;;
    simvla_prepare_data)
        if [[ "${architecture}" != "simvla" ]]; then
            echo "[ERROR] action=simvla_prepare_data is only valid for architecture=simvla" >&2
            exit 2
        fi
        run_cmd=(bash "${ROOT_DIR}/architectures/simvla/wrappers/prepare_libero_links.sh")
        ;;
    simvla_check_data)
        if [[ "${architecture}" != "simvla" ]]; then
            echo "[ERROR] action=simvla_check_data is only valid for architecture=simvla" >&2
            exit 2
        fi
        run_cmd=(python "${ROOT_DIR}/architectures/simvla/wrappers/check_libero_dataset.py")
        ;;
    simvla_train_small|simvla_train_large)
        if [[ "${architecture}" != "simvla" ]]; then
            echo "[ERROR] action=${action} is only valid for architecture=simvla" >&2
            exit 2
        fi
        model_size="${action#simvla_train_}"
        run_cmd=(
            bash "${ROOT_DIR}/architectures/simvla/wrappers/train_libero.sh"
            --model-size "${model_size}"
            --result-dir "${result_dir}"
        )
        ;;
    *)
        echo "[ERROR] Unknown action: ${action}" >&2
        exit 2
        ;;
esac

printf '%q ' "${run_cmd[@]}" > "${result_dir}/command.sh"
printf '\n' >> "${result_dir}/command.sh"

cat > "${result_dir}/run_manifest.yaml" <<EOF
run_id: ${run_id}
timestamp: ${timestamp}
architecture: ${architecture}
method: ${method}
env_id: ${env_id}
node: ${node}
experiment: ${experiment}
action: ${action}
working_directory: ${ROOT_DIR}
result_dir: ${result_dir}
python_executable: $(command -v python)
python_version: "$(python --version 2>&1)"
command_file: command.sh
config_files:
  - configs/config.yaml
  - configs/architecture/${architecture}.yaml
  - configs/method/${method}.yaml
  - configs/env/${env_id}.yaml
  - configs/node/${node}.yaml
  - configs/experiment/${experiment}.yaml
notes: ""
EOF

touch "${result_dir}/metrics/metrics.jsonl"
printf '{\n  "status": "created"\n}\n' > "${result_dir}/metrics/summary.json"
cat > "${result_dir}/notes.md" <<EOF
# Run Notes

- Created by scripts/run_experiment.sh.
- Default action is lightweight sanity checking.
EOF

"${run_cmd[@]}" > >(tee "${result_dir}/logs/stdout.log") 2> >(tee "${result_dir}/logs/stderr.log" >&2)

echo "[RUN DONE] result_dir=${result_dir}"
