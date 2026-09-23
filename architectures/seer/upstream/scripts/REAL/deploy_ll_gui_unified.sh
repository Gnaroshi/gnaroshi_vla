#!/usr/bin/env bash
set -euo pipefail

# Seer-only unified launcher. Leave exactly ONE preset uncommented.
# All settings live here; no persistent shell-variable overrides are needed.
# Existing per-task launchers are retained unchanged as historical references.
deploy_presets=(
    # "basketball_baseline"
    # "basketball_latentloop"
    "doll_baseline"
    # "doll_latentloop"
    # "cabinet_baseline"
    # "cabinet_latentloop"
)
basketball_teacher_id=37         # Existing scratch pairs: 37, 34, 35.
adapter_id=39                    # Used only when deployment_method=latentloop.
query_interval=4                 # Full Seer refresh interval for non-full policies.
baseline_rollout_policy="full"  # full | hold_action | hold_latent
master_port=10123
cuda_device=0

# Configure execution here; this is intentionally not a command-line option.
# live preserves the existing robot/environment initialization and GUI behavior.
# read_only_profile opens real sensors/state but never sends robot commands.
execution_mode="live"               # live | read_only_profile
preflight_only=0                    # 1: synthetic inputs, no cameras/robot.
camera_mode="sync"                  # sync | async_latest
profile_steps=300
profile_warmup_steps=8
optimized_fast_path=1
step_console_log=0
gui_font_backend="system"           # system: antialiased Tk; conda: original Tk.

robot_ip="192.168.56.101"
exterior_camera_serial="243722072499"
wrist_camera_serial="342222070922"
camera_c="no"
camera_width=640
camera_height=480
camera_fps=60

# Task home targets follow the reference PI0.5 deploy task table.
# Values are joint radians (six) + normalized gripper (0=open), NOT TCP poses.
home_pose_reference="/home/jbr/efficient_Seer-main/pi05_relative_deploy_vtp.py"

# Requested frequency, not guaranteed achieved frequency. Preserve Seer v2
# action scaling and synchronous camera behavior; do not import PI0 settings.
control_freq=60
max_rel_pos=0.02
max_rel_orn=0.05
num_rollouts=15
real_eval_max_steps=5000
warmup_steps=3

# No edits are normally needed below this line.
if (( ${#deploy_presets[@]} != 1 )); then
    echo "[ERROR] Uncomment exactly one entry in deploy_presets (task + method)." >&2
    exit 2
fi
deployment_preset="${deploy_presets[0]}"
case "${deployment_preset}" in
    basketball_baseline|basketball_latentloop|doll_baseline|doll_latentloop|cabinet_baseline|cabinet_latentloop) ;;
    *) echo "[ERROR] Unsupported Seer preset: ${deployment_preset}" >&2; exit 2 ;;
esac
deployment_method="${deployment_preset##*_}"
task_name="${deployment_preset%_*}"
case "${task_name}" in
    basketball)
        teacher_id="${basketball_teacher_id}"
        case "${teacher_id}" in
            34|35|37) ;;
            *) echo "[ERROR] Basketball teacher must be 34, 35, or 37." >&2; exit 2 ;;
        esac
        manifest_task="pick_up_the_red_ball_and_place_it_in_the_basketball_hoop_filtered_40p"
        language_instruction="Pick up the red ball and place it in the basketball hoop"
        home_pose_json='[3.14, -1.57, 1.57, -1.57, -1.57, -1.57, 0.0]'
        home_pose_source="${home_pose_reference}:TASK_SPECS[1]; original base pose; home commands gripper open"
        ;;
    doll)
        teacher_id=38
        manifest_task="doll_filtered_40p"
        language_instruction="Pick up the white cup and place it on top of the upside-down pink cup, then pick up the blue penguin plush toy and put it in the white cup"
        home_pose_json='[3.0502887, -1.6030570, 1.8191951, -1.8019783, -1.5417574, -1.6144441, 0.0]'
        home_pose_source="${home_pose_reference}:TASK_SPECS[3]; episode 0511_172010; home commands gripper open"
        ;;
    cabinet)
        teacher_id=38
        manifest_task="cabinet_filtered_40p"
        language_instruction="Open the drawer, take the orange cup out and put it on the table, then put the blue cup in the drawer and close the drawer"
        home_pose_json='[2.9891653, -1.5753395, 1.8866094, -1.8454653, -1.5462163, -1.6641129, 0.0]'
        home_pose_source="${home_pose_reference}:TASK_SPECS[5]; episode 0507_203729; home commands gripper open"
        ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../../../../.." && pwd)
upstream_root="${repo_root}/architectures/seer/upstream"
artifact_root="${repo_root}/artifacts/seer/real_world/${task_name}"
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

print_config_only=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preflight)
            preflight_only=1
            shift
            ;;
        --print-config)
            print_config_only=1
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
            echo "Usage: bash deploy_ll_gui_unified.sh [--print-config] [--preflight] [--camera-mode sync|async_latest] [--profile-steps N] [--control-freq HZ] [--optimized-fast-path 0|1]" >&2
            exit 2
            ;;
    esac
done

if [[ "${preflight_only}" != "0" && "${preflight_only}" != "1" ]]; then
    echo "[ERROR] preflight_only must be 0 or 1." >&2
    exit 2
fi
if ! [[ "${query_interval}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] query_interval must be a positive integer." >&2
    exit 2
fi
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
        deployment_profile="${task_name}_teacher${teacher_id}_${rollout_policy}_initial_k${initial_query_interval}_${control_freq}hz"
        ;;
    latentloop)
        deployment_profile="${task_name}_teacher${teacher_id}_adapter${adapter_id}_initial_k${initial_query_interval}_${control_freq}hz"
        ;;
esac
results_root="${repo_root}/real_deploy_results_v2/${deployment_method}/${task_name}"

if [[ "${print_config_only}" -eq 1 ]]; then
    printf '%s\n' "deployment_preset=${deployment_preset}" "task_name=${task_name}" \
        "deployment_method=${deployment_method}" "teacher_id=${teacher_id}" \
        "adapter_id=${selected_adapter_id}" "teacher_checkpoint=${teacher_checkpoint}" \
        "adapter_checkpoint=${selected_adapter_checkpoint}" "artifact_manifest=${artifact_manifest}" \
        "vit_checkpoint=${vit_checkpoint}" "language_instruction=${language_instruction}" \
        "home_pose_json=${home_pose_json}" "home_pose_source=${home_pose_source}" \
        "control_freq=${control_freq}" "camera_fps=${camera_fps}" "query_interval=${initial_query_interval}" \
        "rollout_policy=${rollout_policy}" "execution_mode=${execution_mode}" \
        "preflight_only=${preflight_only}" "results_root=${results_root}"
    printf '%s\n' "gui_font_backend=${gui_font_backend}"
    exit 0
fi

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
    if [[ ! -s "${required}" ]]; then
        echo "[ERROR] Missing deployment artifact: ${required}" >&2
        exit 2
    fi
done

# Reject a manifest from another task before loading a model or any hardware.
# The existing Python controller separately verifies actual checkpoint hashes.
python - "${artifact_manifest}" "${manifest_task}" "${teacher_id}" \
    "${deployment_method}" "${adapter_id}" <<'PY'
import json
import sys
from pathlib import Path

path, task, teacher_id, method, adapter_id = sys.argv[1:]
manifest = json.loads(Path(path).read_text())
if manifest.get("task") != task:
    raise SystemExit(f"[ERROR] Wrong task manifest: expected {task}, got {manifest.get('task')}")
teacher = manifest.get("teachers", {}).get(teacher_id)
if teacher is None:
    raise SystemExit(f"[ERROR] Teacher {teacher_id} is absent from {path}")
if method == "latentloop" and adapter_id not in teacher.get("adapters", {}):
    raise SystemExit(f"[ERROR] Adapter {adapter_id} does not belong to teacher {teacher_id}")
print(f"[TASK PASS] {task}: teacher={teacher_id}, method={method}")
PY

export CUDA_VISIBLE_DEVICES="${cuda_device}"
export PYTHONPATH="${repo_root}:${upstream_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export SEER_LANGUAGE_INSTRUCTION="${language_instruction}"
unset SEER_LANGUAGE_INSTRUCTIONS || true
export SEER_ROBOT_IP="${robot_ip}"
export SEER_HOME_POSE="${home_pose_json}"
export SEER_HOME_TASK="${manifest_task}"
export SEER_HOME_POSE_SOURCE="${home_pose_source}"
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

run_stamp="$(date +%Y%m%d_%H%M%S)_$$"
deployment_profile="${deployment_profile}_v2_${execution_mode}_${camera_mode}"
launch_dir="${results_root}/launch_logs/${deployment_profile}/${run_stamp}"
mkdir -p "${launch_dir}"
cp "${BASH_SOURCE[0]}" "${launch_dir}/deploy_ll_gui.sh.snapshot"
profile_output_dir="${results_root}/read_only_profile/${deployment_profile}/launch_$$"

{
    echo "timestamp=${run_stamp}"
    echo "hostname=$(hostname)"
    echo "conda_env=${CONDA_DEFAULT_ENV}"
    echo "repo_root=${repo_root}"
    echo "deployment_method=${deployment_method}"
    echo "deployment_preset=${deployment_preset}"
    echo "task_name=${task_name}"
    echo "manifest_task=${manifest_task}"
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
    echo "home_pose_json=${home_pose_json}"
    echo "home_pose_source=${home_pose_source}"
    echo "exterior_camera_serial=${exterior_camera_serial}"
    echo "wrist_camera_serial=${wrist_camera_serial}"
    echo "camera_width=${camera_width}"
    echo "camera_height=${camera_height}"
    echo "camera_fps=${camera_fps}"
    echo "execution_mode=${execution_mode}"
    echo "camera_mode=${camera_mode}"
    echo "gui_font_backend=${gui_font_backend}"
    echo "profile_steps=${profile_steps}"
    echo "profile_warmup_steps=${profile_warmup_steps}"
    echo "optimized_fast_path=${optimized_fast_path}"
    echo "robot_motion_commands_enabled=$([[ "${execution_mode}" == "live" && "${preflight_only}" -eq 0 ]] && echo 1 || echo 0)"
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

# Only this GUI process uses system Tk; conda packages and the caller's shell
# environment are untouched. Model-only preflight and profiling keep their runtime.
if [[ "${execution_mode}" == "live" && "${preflight_only}" -eq 0 ]]; then
    case "${gui_font_backend}" in
        system)
            for required_tk in /usr/lib/x86_64-linux-gnu/libtcl8.6.so /usr/lib/x86_64-linux-gnu/libtk8.6.so /usr/share/tcltk/tcl8.6/init.tcl /usr/share/tcltk/tk8.6/tk.tcl; do
                [[ -f "${required_tk}" ]] || { echo "[ERROR] Missing ${required_tk}; set gui_font_backend=conda to keep the original Tk." >&2; exit 2; }
            done
            command=(env
                "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcl8.6.so:/usr/lib/x86_64-linux-gnu/libtk8.6.so${LD_PRELOAD:+:${LD_PRELOAD}}"
                "TCL_LIBRARY=/usr/share/tcltk/tcl8.6"
                "TK_LIBRARY=/usr/share/tcltk/tk8.6"
                "${command[@]}")
            ;;
        conda) ;;
        *) echo "[ERROR] gui_font_backend must be system or conda" >&2; exit 2 ;;
    esac
fi

printf '%q ' "${command[@]}" > "${launch_dir}/command.txt"
printf '\n' >> "${launch_dir}/command.txt"

echo "[Seer deploy] preset=${deployment_preset} method=${deployment_method} profile=${deployment_profile}"
echo "[Seer deploy] v2 execution=${execution_mode} camera_mode=${camera_mode}"
echo "[Seer deploy] robot_motion_commands_enabled=$([[ "${execution_mode}" == "live" && "${preflight_only}" -eq 0 ]] && echo 1 || echo 0)"
echo "[Seer deploy] rollout_policy=${rollout_policy} initial_K=${initial_query_interval}"
echo "[Seer deploy] control_freq=${control_freq}Hz (editable in live GUI)"
echo "[Seer deploy] log=${launch_dir}/console.log"
echo "[Seer deploy] teacher=${teacher_checkpoint}"
echo "[Seer deploy] home_pose(rad joints + normalized gripper)=${home_pose_json}"
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
