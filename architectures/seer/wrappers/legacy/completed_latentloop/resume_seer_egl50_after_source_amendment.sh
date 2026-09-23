#!/usr/bin/env bash

set -Eeuo pipefail

# Resume the original EGL-50 paper queue across reviewed source amendments.
# Phase 1/2 locks remain immutable. Phase 3 adds only a null-safe diagnostic
# read for cached-horizon replay; the original queue remains the evaluator.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd -L)"
UPSTREAM_DIR="${REPO_ROOT}/architectures/seer/upstream"
QUEUE_REL="architectures/seer/wrappers/lrnode/run_seer_egl50_main_sequential.sh"
QUEUE="${REPO_ROOT}/${QUEUE_REL}"
EQUIVALENCE_CHECK="${REPO_ROOT}/tools/seer/check_egl50_resume_equivalence.py"

EXPECTED_HOST="${EXPECTED_HOST:-jbrserver1}"
EXPECTED_REPO="${EXPECTED_REPO:-/home/mingyujung/private/gnaroshi_vla_latentloop_canonical}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
PAPER_ROOT="${PAPER_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/paper_egl50/seer_public33_egl50_main_v1}"
PHASE1_LOCK="${PAPER_ROOT}/source_sha256.phase1.lock"
ACTIVE_LOCK="${PAPER_ROOT}/source_sha256.lock"
PHASE2_LOCK="${PAPER_ROOT}/source_sha256.phase2.lock"
PHASE3_LOCK="${PAPER_ROOT}/source_sha256.phase3.lock"
PHASE1_LOCK_SHA256="bf025237a1579fb3a6ac60d4c61c1966114d57c00054871b827a407cf8325363"
PHASE2_LOCK_SHA256="52dda93e44adb233e31e1314591aeecb7662b00164dc111475675bb1cd7e42a5"
AMENDMENT_PREFLIGHT_ONLY="${AMENDMENT_PREFLIGHT_ONLY:-0}"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || fail "missing or empty file: $1"
}

sha256_of() {
    sha256sum "$1" | awk '{print $1}'
}

require_hash() {
    local path="$1" expected="$2" actual
    require_file "${path}"
    actual="$(sha256_of "${path}")"
    [[ "${actual}" == "${expected}" ]] \
        || fail "source hash mismatch: path=${path}, expected=${expected}, actual=${actual}"
}

verify_current_source_allowlist() {
    require_hash "${UPSTREAM_DIR}/eval_libero.py" \
        7b4b68a3b630b95cdfa3831c795bb140252543bc7e2022d0308b79eb949a1029
    require_hash "${UPSTREAM_DIR}/models/seer_model.py" \
        ab67b9dad4d79e533d81bf582cf870ceaf4f51809fd1e937cb278718cd620dfd
    require_hash "${UPSTREAM_DIR}/models/lrnode_modules.py" \
        60e92009d8004fca950441d4f5f779c38383ccf716d48e7d7e2956d568847db0
    require_hash "${UPSTREAM_DIR}/utils/arguments_utils.py" \
        a32bf2cc4e2d5044cb30c4313c6e8ca7e91dc6e6442a24ad1a011ea4bc330477
    require_hash "${UPSTREAM_DIR}/utils/eval_utils_libero.py" \
        d840aefe932f441c986e3812bccf9220a413c7fa417e67bc7049d9da909c5d55
    require_hash "${UPSTREAM_DIR}/utils/train_utils.py" \
        3333b5ac4fddaa64f545cdcfb6268a563716a14b4749f8d6d3baf72b0b059370
    require_hash "${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/scratch.sh" \
        0e45936b163275c9a373bed244c997cdc5b1bfb7bc605d2914618b5d91a0004d
    require_hash "${SCRIPT_DIR}/distill_node.sh" \
        055a3acad20d382df9af4a57f4ff42415456bd90e551fc34872e0c07284d719b
    require_hash "${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh" \
        06d5abe80b99648d9fce0e6617de73c0f414015e890a6d39a6aa2280480c2214
    require_hash "${QUEUE}" \
        0ea8423cda3683254caf2d4e0cd5e1d80d5ab64e3ef6f24ac9e880564afe3e5b
    require_hash "${EQUIVALENCE_CHECK}" \
        d8903457b0458926734a115c6dd14f2731d7516c2bee8879a9545af45c010907
    require_hash "${REPO_ROOT}/tools/seer/fixtures/egl50_phase1_seer_model.py" \
        3f72676e3f64ace7396eefaf9324dc70ad9867ec8b3cfb206966d110e0735a7c
    echo "[VERIFY][OK] current source matches the reviewed phase-3 allowlist"
}

write_current_lock() {
    local output="$1"
    (
        cd "${REPO_ROOT}"
        sha256sum \
            "${UPSTREAM_DIR}/eval_libero.py" \
            "${UPSTREAM_DIR}/models/seer_model.py" \
            "${UPSTREAM_DIR}/models/lrnode_modules.py" \
            "${UPSTREAM_DIR}/utils/arguments_utils.py" \
            "${UPSTREAM_DIR}/utils/eval_utils_libero.py" \
            "${UPSTREAM_DIR}/utils/train_utils.py" \
            "${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/scratch.sh" \
            "${SCRIPT_DIR}/distill_node.sh" \
            "${UPSTREAM_DIR}/scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh" \
            "${QUEUE_REL}"
    ) > "${output}"
}

verify_completed_rows() {
    python - "${PAPER_ROOT}" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = {
    "long_seed42_baseline_k1",
    "long_seed42_latentloop_k4",
    "long_seed42_latentloop_k8",
    "libero_spatial_seed42_baseline_k1",
    "libero_spatial_seed42_latentloop_k4",
    "libero_object_seed42_baseline_k1",
    "libero_object_seed42_latentloop_k4",
    "libero_goal_seed42_baseline_k1",
    "libero_goal_seed42_latentloop_k4",
    "long_seed43_baseline_k1",
    "long_seed43_latentloop_k4",
    "long_seed43_latentloop_k8",
}
planned = set()
for seed in (42, 43, 44):
    planned.update(
        {
            f"long_seed{seed}_baseline_k1",
            f"long_seed{seed}_latentloop_k4",
            f"long_seed{seed}_latentloop_k8",
        }
    )
    for suite in ("libero_spatial", "libero_object", "libero_goal"):
        planned.update(
            {
                f"{suite}_seed{seed}_baseline_k1",
                f"{suite}_seed{seed}_latentloop_k4",
            }
        )
    planned.update(
        {
            f"long_seed{seed}_no_observation_k4",
            f"long_seed{seed}_predicted_horizon_replay_k4",
        }
    )
planned.update({"long_seed42_hold_latent_k4", "long_seed42_hold_action_k4"})
for k in (2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16):
    planned.add(f"long_seed42_latentloop_k{k}")
for seed in (43, 44):
    for k in (2, 12, 16):
        planned.add(f"long_seed{seed}_latentloop_k{k}")
found = set()
for row_root in sorted((root / "eval").iterdir()):
    if not row_root.is_dir() or ".incomplete." in row_root.name:
        continue
    summaries = sorted(row_root.glob("*/analysis/eval_summary.json"))
    if len(summaries) != 1:
        continue
    summary = json.loads(summaries[0].read_text())
    metrics = summaries[0].parent / "eval_episode_metrics.csv"
    with metrics.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 500:
        raise RuntimeError(f"{row_root.name}: expected 500 episodes, found {len(rows)}")
    renderer = summary.get("renderer_backend", {})
    if renderer.get("effective_backend") != "egl" or not renderer.get("all_ranks_actual_context_verified"):
        raise RuntimeError(f"{row_root.name}: invalid EGL provenance")
    found.add(row_root.name)
if not expected.issubset(found):
    raise RuntimeError(
        f"phase-1 completed-row set mismatch: missing={sorted(expected-found)}"
    )
if not found.issubset(planned):
    raise RuntimeError(f"unexpected completed rows: {sorted(found-planned)}")
print(
    f"[VERIFY][OK] completed rows={len(found)}/{len(planned)}; "
    f"phase-1 subset={len(expected)}"
)
PY
}

preserve_prior_artifacts() {
    require_hash "${PHASE1_LOCK}" "${PHASE1_LOCK_SHA256}"
    require_hash "${PHASE2_LOCK}" "${PHASE2_LOCK_SHA256}"
    if [[ -s "${PAPER_ROOT}/queue_failed.env" && ! -e "${PAPER_ROOT}/queue_failed.phase2.env" ]]; then
        cp -p "${PAPER_ROOT}/queue_failed.env" "${PAPER_ROOT}/queue_failed.phase2.env"
    fi
}

verify_logging_fix() {
    local output="$1"
    python - "${UPSTREAM_DIR}/utils/eval_utils_libero.py" "${output}" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

source = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
phase2_sha256 = "4058f5f6043d9559057967f7b82c08c4ac012d6e9190c68e10323a3e9b5a3eec"
old = b'elif "action_pred_gripper_logit" in lrnode_debug:'
new = b'elif lrnode_debug.get("action_pred_gripper_logit") is not None:'
current = source.read_bytes()
if current.count(new) != 1:
    raise RuntimeError("expected exactly one null-safe gripper-logit condition")
reconstructed = current.replace(new, old, 1)
reconstructed_sha256 = hashlib.sha256(reconstructed).hexdigest()
if reconstructed_sha256 != phase2_sha256:
    raise RuntimeError(
        "diagnostic patch is not a one-expression change from phase 2: "
        f"expected={phase2_sha256}, reconstructed={reconstructed_sha256}"
    )
for debug, expected in (
    ({}, False),
    ({"action_pred_gripper_logit": None}, False),
    ({"action_pred_gripper_logit": object()}, True),
):
    actual = debug.get("action_pred_gripper_logit") is not None
    if actual != expected:
        raise RuntimeError("null-safe diagnostic behavior check failed")
payload = {
    "status": "PASS",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "scope": "diagnostic CSV scalar extraction only",
    "phase2_eval_utils_reconstructed_sha256": reconstructed_sha256,
    "phase3_eval_utils_sha256": hashlib.sha256(current).hexdigest(),
    "changed_condition_before": old.decode(),
    "changed_condition_after": new.decode(),
    "none_value_result": "leave executed_gripper_logit blank",
    "tensor_value_result": "extract the existing scalar exactly as before",
    "policy_action_path_changed": False,
    "cache_update_path_changed": False,
    "model_forward_path_changed": False,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2) + "\n")
print(f"[VERIFY][OK] phase-2 -> phase-3 diagnostic-only patch: {output}")
PY
}

write_phase3_amendment() {
    local target="${PAPER_ROOT}/source_transition_phase2_to_phase3.json"
    if [[ -s "${target}" ]]; then
        echo "[AMENDMENT][KEEP] ${target}"
        return
    fi
    python - "${PAPER_ROOT}" "${PHASE2_LOCK}" "${PHASE3_LOCK}" <<'PY'
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

root, phase2, phase3 = map(pathlib.Path, sys.argv[1:])

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

phase2_rows = sorted(
    path.parents[2].name
    for path in (root / "eval").glob("*/*/analysis/eval_summary.json")
    if ".incomplete." not in path.parents[2].name
)
payload = {
    "status": "APPROVED_DIAGNOSTIC_ONLY_SOURCE_TRANSITION",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "reason": "cached-horizon replay may not have a gripper logit; step logging now treats None as unavailable",
    "phase2_lock": str(phase2),
    "phase2_lock_sha256": digest(phase2),
    "phase3_lock": str(phase3),
    "phase3_lock_sha256": digest(phase3),
    "phase2_completed_rows": phase2_rows,
    "phase2_completed_row_count": len(phase2_rows),
    "phase3_scope": "remaining evaluation rows only; no training or policy-path change",
    "diagnostic_fix_evidence": str(root / "parity/source_transition_logging_fix.json"),
    "model_equivalence_evidence": str(root / "parity/source_transition_phase3_model_equivalence.json"),
}
(root / "source_transition_phase2_to_phase3.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
print(f"[AMENDMENT] {root / 'source_transition_phase2_to_phase3.json'}")
PY
}

[[ "$(hostname)" == "${EXPECTED_HOST}" ]] || fail "expected ${EXPECTED_HOST}, got $(hostname)"
[[ "$(readlink -f "${REPO_ROOT}")" == "$(readlink -f "${EXPECTED_REPO}")" ]] \
    || fail "unexpected source tree: ${REPO_ROOT}"
[[ "${CONDA_DEFAULT_ENV:-}" == seer_libero ]] || fail "activate conda environment seer_libero"
[[ "${GPU_LIST}" == 4,5,6,7 ]] || fail "set CUDA_VISIBLE_DEVICES=4,5,6,7"
[[ "${AMENDMENT_PREFLIGHT_ONLY}" == 0 || "${AMENDMENT_PREFLIGHT_ONLY}" == 1 ]] \
    || fail "AMENDMENT_PREFLIGHT_ONLY must be 0 or 1"
require_file "${ACTIVE_LOCK}"
require_hash "${PHASE1_LOCK}" "${PHASE1_LOCK_SHA256}"
require_hash "${PHASE2_LOCK}" "${PHASE2_LOCK_SHA256}"
require_file "${QUEUE}"
require_file "${EQUIVALENCE_CHECK}"
verify_current_source_allowlist
verify_completed_rows

equivalence_output="${PAPER_ROOT}/parity/source_transition_phase3_model_equivalence.json"
logging_fix_output="${PAPER_ROOT}/parity/source_transition_logging_fix.json"
if [[ "${AMENDMENT_PREFLIGHT_ONLY}" == 1 ]]; then
    equivalence_output="/tmp/seer_egl50_source_transition_phase3_model_equivalence.json"
    logging_fix_output="/tmp/seer_egl50_source_transition_logging_fix.json"
fi
python "${EQUIVALENCE_CHECK}" | tee "${equivalence_output}"
verify_logging_fix "${logging_fix_output}"

if [[ "${AMENDMENT_PREFLIGHT_ONLY}" == 1 ]]; then
    echo "[PREFLIGHT][PASS] completed rows, model equivalence, and diagnostic-only fix verified"
    echo "[PREFLIGHT] no campaign lock or evaluation artifact was modified"
    exit 0
fi

preserve_prior_artifacts
phase3_tmp="$(mktemp /tmp/seer_egl50_phase3_lock.XXXXXX)"
write_current_lock "${phase3_tmp}"
if [[ -s "${PHASE3_LOCK}" ]]; then
    cmp -s "${PHASE3_LOCK}" "${phase3_tmp}" \
        || fail "existing phase-3 lock differs from current reviewed source"
else
    cp -p "${phase3_tmp}" "${PHASE3_LOCK}"
fi
rm -f "${phase3_tmp}"
write_phase3_amendment
cp -p "${PHASE3_LOCK}" "${ACTIVE_LOCK}"

completed_rows="$(find "${PAPER_ROOT}/eval" -path '*/analysis/eval_summary.json' -type f | wc -l)"
echo "[RESUME] completed rows=${completed_rows}/54; original queue will validate/skip them"
echo "[RESUME] remaining original rows=$((54 - completed_rows))"
echo "[RESUME] Spatial/Object/Goal continue with teacher39 + adapter39"
echo "[RESUME] Long continues with public33 + adapter39"
cd "${REPO_ROOT}"
env \
    CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
    PAPER_ROOT="${PAPER_ROOT}" \
    RUN_DIRECT_MECHANISMS=1 \
    RUN_K_CURVE=1 \
    PREFLIGHT_ONLY=0 \
    bash "${QUEUE_REL}"
