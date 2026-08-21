#!/usr/bin/env bash
set -euo pipefail

# Basketball deployment protocol. Select exactly one method by commenting one
# assignment and uncommenting the other. Baseline and LatentLoop use the same
# teacher; only LatentLoop loads the teacher-specific adapter.
deployment_method="baseline"
# deployment_method="latentloop"

teacher_id=37                    # Allowed paired teachers: 37, 34, 35.
adapter_id=39                    # Used only when deployment_method=latentloop.
query_interval=4                 # Full Seer refresh interval for non-full policies.
baseline_rollout_policy="full"  # full | hold_action | hold_latent
master_port=10123
cuda_device=0

# Configure execution here; this is intentionally not a command-line option.
# live opens the GUI and enables robot commands only after a rollout is started.
# read_only_profile opens real sensors/state but never sends robot commands.
execution_mode="live"               # live | read_only_profile
camera_mode="sync"                  # sync | async_latest
profile_steps=300
profile_warmup_steps=8
optimized_fast_path=1
step_console_log=0

language_instruction="Pick up the red ball and place it in the basketball hoop"
robot_ip="192.168.56.101"
exterior_camera_serial="243722072499"
wrist_camera_serial="342222070922"
camera_c="no"
camera_width=640
camera_height=480
camera_fps=60

# The conservative single-GPU live candidate is 15 Hz based on read-only
# profiling. Live command-path rate is not validated yet. Higher targets remain
# explicit profiling/experimental settings selected with --control-freq.
control_freq=60
max_rel_pos=0.02
max_rel_orn=0.05
num_rollouts=15
real_eval_max_steps=5000
warmup_steps=3

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../../../.." && pwd)
upstream_root="${repo_root}/architectures/seer/upstream"
artifact_root="${repo_root}/artifacts/seer/real_world/basketball"
artifact_manifest="${artifact_root}/checkpoint_manifest.json"
teacher_checkpoint="${artifact_root}/baseline/teacher_${teacher_id}.pth"
adapter_checkpoint="${artifact_root}/latentloop/teacher_${teacher_id}/teacher_${teacher_id}_adapter_${adapter_id}.pth"
vit_checkpoint="${artifact_root}/shared/mae_pretrain_vit_base.pth"

case "${deployment_method}" in
    baseline)
        rollout_policy="${baseline_rollout_policy}"
        case "${rollout_policy}" in
            full) initial_query_interval=1 ;;
            hold_action|hold_latent) initial_query_interval="${query_interval}" ;;
            *)
                echo "[ERROR] baseline_rollout_policy must be full, hold_action, or hold_latent: ${rollout_policy}" >&2
                exit 2
                ;;
        esac
        selected_adapter_id="none"
        selected_adapter_checkpoint="not_loaded"
        ;;
    latentloop)
        rollout_policy="latentloop"
        initial_query_interval="${query_interval}"
        selected_adapter_id="${adapter_id}"
        selected_adapter_checkpoint="${adapter_checkpoint}"
        ;;
    *)
        echo "[ERROR] deployment_method must be baseline or latentloop: ${deployment_method}" >&2
        exit 2
        ;;
esac

preflight_only=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preflight)
            preflight_only=1
            shift
            ;;
        --camera-mode)
            [[ $# -ge 2 ]] || { echo "[ERROR] --camera-mode requires a value" >&2; exit 2; }
            camera_mode="$2"
            shift 2
            ;;
        --profile-steps)
            [[ $# -ge 2 ]] || { echo "[ERROR] --profile-steps requires a value" >&2; exit 2; }
            profile_steps="$2"
            shift 2
            ;;
        --control-freq)
            [[ $# -ge 2 ]] || { echo "[ERROR] --control-freq requires a value" >&2; exit 2; }
            control_freq="$2"
            shift 2
            ;;
        --optimized-fast-path)
            [[ $# -ge 2 ]] || { echo "[ERROR] --optimized-fast-path requires a value" >&2; exit 2; }
            optimized_fast_path="$2"
            shift 2
            ;;
        *)
            echo "Usage: bash scripts/REAL/deploy_ll_gui_v2.sh [--preflight] [--camera-mode sync|async_latest] [--profile-steps N] [--control-freq HZ] [--optimized-fast-path 0|1]" >&2
            exit 2
            ;;
    esac
done

case "${execution_mode}" in
    read_only_profile|live) ;;
    *)
        echo "[ERROR] execution_mode must be read_only_profile or live: ${execution_mode}" >&2
        exit 2
        ;;
esac
case "${camera_mode}" in
    sync|async_latest) ;;
    *)
        echo "[ERROR] camera_mode must be sync or async_latest: ${camera_mode}" >&2
        exit 2
        ;;
esac
if [[ "${execution_mode}" == "live" && "${camera_mode}" != "sync" ]]; then
    echo "[ERROR] async_latest is read-only-profile-only until camera parity is validated" >&2
    exit 2
fi
if ! [[ "${profile_steps}" =~ ^[0-9]+$ ]] || (( profile_steps < 2 )); then
    echo "[ERROR] profile_steps must be an integer >= 2: ${profile_steps}" >&2
    exit 2
fi
if ! [[ "${control_freq}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || ! awk -v hz="${control_freq}" 'BEGIN { exit !(hz > 0) }'; then
    echo "[ERROR] control_freq must be a positive number in Hz: ${control_freq}" >&2
    exit 2
fi
if [[ "${optimized_fast_path}" != "0" && "${optimized_fast_path}" != "1" ]]; then
    echo "[ERROR] optimized_fast_path must be 0 or 1: ${optimized_fast_path}" >&2
    exit 2
fi
case "${deployment_method}" in
    baseline)
        deployment_profile="basketball_teacher${teacher_id}_${rollout_policy}_initial_k${initial_query_interval}_${control_freq}hz"
        ;;
    latentloop)
        deployment_profile="basketball_teacher${teacher_id}_adapter${adapter_id}_initial_k${initial_query_interval}_${control_freq}hz"
        ;;
esac
results_root="${repo_root}/real_deploy_results_v2/${deployment_method}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "seer" ]]; then
    echo "[ERROR] Activate the inference computer's existing conda environment first:" >&2
    echo "        conda activate seer" >&2
    exit 2
fi

required_artifacts=(
    "${artifact_manifest}"
    "${teacher_checkpoint}"
    "${vit_checkpoint}"
)
if [[ "${deployment_method}" == "latentloop" ]]; then
    required_artifacts+=("${adapter_checkpoint}")
fi
for required in "${required_artifacts[@]}"; do
    if [[ ! -f "${required}" ]]; then
        echo "[ERROR] Missing deployment artifact: ${required}" >&2
        exit 2
    fi
done

export CUDA_VISIBLE_DEVICES="${cuda_device}"
export PYTHONPATH="${repo_root}:${upstream_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export SEER_LANGUAGE_INSTRUCTION="${language_instruction}"
unset SEER_LANGUAGE_INSTRUCTIONS || true
export SEER_ROBOT_IP="${robot_ip}"
export SEER_EXTERIOR_CAMERA_SERIAL="${exterior_camera_serial}"
export SEER_WRIST_CAMERA_SERIAL="${wrist_camera_serial}"
export SEER_CAMERA_WIDTH="${camera_width}"
export SEER_CAMERA_HEIGHT="${camera_height}"
export SEER_CAMERA_FPS="${camera_fps}"
export SEER_CONTROL_FREQ="${control_freq}"
export SEER_EXECUTION_MODE="${execution_mode}"
export SEER_MAX_REL_POS="${max_rel_pos}"
export SEER_MAX_REL_ORN="${max_rel_orn}"
export SEER_NUM_ROLLOUTS="${num_rollouts}"
export SEER_WARMUP_STEPS="${warmup_steps}"
if [[ "${execution_mode}" == "live" ]]; then
    export SEER_ENABLE_ROLLOUT_MEDIA=1
else
    export SEER_ENABLE_ROLLOUT_MEDIA=0
fi
export SEER_ENABLE_OBSERVER_MEDIA=0
export SEER_OBSERVER_CAMERA_NAME=observer
export SEER_RESULTS_DIR="${results_root}"

run_stamp=$(date +%Y%m%d_%H%M%S)
deployment_profile="${deployment_profile}_v2_${execution_mode}_${camera_mode}"
launch_dir="${results_root}/launch_logs/${deployment_profile}/${run_stamp}"
mkdir -p "${launch_dir}"
cp "${BASH_SOURCE[0]}" "${launch_dir}/deploy_ll_gui.sh.snapshot"
profile_output_dir="${launch_dir}/read_only_profile"

{
    echo "timestamp=${run_stamp}"
    echo "hostname=$(hostname)"
    echo "conda_env=${CONDA_DEFAULT_ENV}"
    echo "repo_root=${repo_root}"
    echo "deployment_method=${deployment_method}"
    echo "deployment_profile=${deployment_profile}"
    echo "teacher_id=${teacher_id}"
    echo "adapter_id=${selected_adapter_id}"
    echo "rollout_policy=${rollout_policy}"
    echo "initial_query_interval=${initial_query_interval}"
    echo "teacher_checkpoint=${teacher_checkpoint}"
    echo "adapter_checkpoint=${selected_adapter_checkpoint}"
    echo "vit_checkpoint=${vit_checkpoint}"
    echo "artifact_manifest=${artifact_manifest}"
    echo "language_instruction=${language_instruction}"
    echo "robot_ip=${robot_ip}"
    echo "exterior_camera_serial=${exterior_camera_serial}"
    echo "wrist_camera_serial=${wrist_camera_serial}"
    echo "camera_width=${camera_width}"
    echo "camera_height=${camera_height}"
    echo "camera_fps=${camera_fps}"
    echo "execution_mode=${execution_mode}"
    echo "camera_mode=${camera_mode}"
    echo "profile_steps=${profile_steps}"
    echo "profile_warmup_steps=${profile_warmup_steps}"
    echo "optimized_fast_path=${optimized_fast_path}"
    echo "robot_motion_commands_enabled=$([[ "${execution_mode}" == "live" ]] && echo 1 || echo 0)"
    echo "control_freq=${control_freq}"
    echo "control_period_ms=$(awk -v hz="${control_freq}" 'BEGIN { printf "%.6f", 1000.0 / hz }')"
    echo "max_rel_pos=${max_rel_pos}"
    echo "max_rel_orn=${max_rel_orn}"
    echo "num_rollouts=${num_rollouts}"
    echo "real_eval_max_steps=${real_eval_max_steps}"
    echo "preflight_only=${preflight_only}"
} > "${launch_dir}/launch_config.txt"

git -C "${repo_root}" rev-parse HEAD > "${launch_dir}/git_commit.txt"
git -C "${repo_root}" status --short > "${launch_dir}/git_status.txt"
sha256sum "${required_artifacts[@]}" > "${launch_dir}/artifact_sha256.txt"

command=(
    torchrun
    --nnodes=1
    --nproc_per_node=1
    --master_port="${master_port}"
    "${repo_root}/architectures/seer/adapters/latentloop_real_deploy/deploy_ll_gui_v2.py"
    --camera-c "${camera_c}"
    --deployment-method "${deployment_method}"
    --deployment-control-freq "${control_freq}"
    --rollout-policy "${rollout_policy}"
    --latentloop-artifact-manifest "${artifact_manifest}"
    --latentloop-teacher-id "${teacher_id}"
    --latentloop-deployment-profile "${deployment_profile}"
    --v2-camera-mode "${camera_mode}"
    --v2-profile-steps "${profile_steps}"
    --v2-profile-warmup-steps "${profile_warmup_steps}"
    --v2-profile-output-dir "${profile_output_dir}"
    --v2-optimized-fast-path "${optimized_fast_path}"
    --traj_cons
    --rgb_pad 10
    --gripper_pad 4
    --gradient_accumulation_steps 1
    --bf16_module vision_encoder
    --vit_checkpoint_path "${vit_checkpoint}"
    --workers 16
    --calvin_dataset ""
    --lr_scheduler cosine
    --save_every_iter 50000
    --num_epochs 20
    --seed 42
    --batch_size 64
    --precision fp32
    --weight_decay 1e-4
    --num_resampler_query 6
    --num_obs_token_per_image 9
    --calvin_input_image_size 224
    --patch_size 16
    --run_name "${deployment_profile}"
    --transformer_layers 24
    --hidden_dim 384
    --transformer_heads 12
    --save_checkpoint_path checkpoint
    --phase evaluate
    --finetune_type real
    --action_pred_steps 3
    --future_steps 3
    --sequence_length 7
    --obs_pred
    --resume_from_checkpoint "${teacher_checkpoint}"
    --real_eval_max_steps "${real_eval_max_steps}"
    --eval_libero_ensembling
    --ensembling_temp 0.01
    --lrnode_hidden_dim 256
    --lrnode_motion_dim 128
    --lrnode_fast_encoder_type diffcnn
    --lrnode_detach_input_latent 1
    --lrnode_detach_teacher_latent 1
    --lrnode_freeze_action_head_for_lrnode 1
    --lrnode_use_post_layernorm 0
    --lrnode_multistep_train 0
    --lrnode_train_max_horizon 2
    --lrnode_log_sanity 1
    --lrnode_gate_init_bias -4.0
    --lrnode_trace 0
    --lrnode_eval_step_log "${step_console_log}"
    --lrnode_eval_profile_full_action_head 1
)
if [[ "${deployment_method}" == "latentloop" ]]; then
    command+=(
        --latentloop-adapter-checkpoint "${adapter_checkpoint}"
        --latentloop-adapter-id "${adapter_id}"
        --use_lrnode_latent_update 1
        --lrnode_eval_skip_full_forward 1
        --lrnode_query_interval "${initial_query_interval}"
    )
else
    command+=(
        --use_lrnode_latent_update 0
        --lrnode_eval_skip_full_forward 0
        --lrnode_query_interval "${initial_query_interval}"
    )
fi
if [[ "${preflight_only}" -eq 1 ]]; then
    command+=(--latentloop-preflight-only)
fi

printf '%q ' "${command[@]}" > "${launch_dir}/command.txt"
printf '\n' >> "${launch_dir}/command.txt"

echo "[Seer deploy] method=${deployment_method} profile=${deployment_profile}"
echo "[Seer deploy] v2 execution=${execution_mode} camera_mode=${camera_mode}"
echo "[Seer deploy] robot_motion_commands_enabled=$([[ "${execution_mode}" == "live" ]] && echo 1 || echo 0)"
echo "[Seer deploy] rollout_policy=${rollout_policy} initial_K=${initial_query_interval}"
echo "[Seer deploy] control_freq=${control_freq}Hz (editable in live GUI)"
echo "[Seer deploy] log=${launch_dir}/console.log"
echo "[Seer deploy] teacher=${teacher_checkpoint}"
if [[ "${deployment_method}" == "latentloop" ]]; then
    echo "[Seer deploy] adapter=${adapter_checkpoint}"
else
    echo "[Seer deploy] adapter=not loaded"
fi

set +e
"${command[@]}" 2>&1 | tee "${launch_dir}/console.log"
exit_code=${PIPESTATUS[0]}
set -e
echo "${exit_code}" > "${launch_dir}/exit_code.txt"
exit "${exit_code}"
