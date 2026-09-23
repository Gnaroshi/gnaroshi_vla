#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname)" == "jbrserver18" ]] || { echo "[ERROR] expected jbrserver18" >&2; exit 1; }
TARGET="${TARGET_SOURCE_TREE:-/home/mingyujung/private/gnaroshi_vla_sd1_canonical_v0v1v2_20260821}"
STABLE_TARGET="${STABLE_SOURCE_TREE:-/home/mingyujung/private/gnaroshi_vla_latentloop_canonical}"
ARTIFACT_TARGET="${TARGET_ARTIFACT_ROOT:-/home/mingyujung/shared/hdd_ext/nvme8000/mingyujung/gnaroshi_vla/results/seer/latentloop/sd1_teacher33_s18_v0v1v2_20260821}"
STABLE_ARTIFACT="${STABLE_ARTIFACT_ROOT:-/home/mingyujung/shared/hdd_ext/nvme8000/mingyujung/gnaroshi_vla/results/seer/latentloop/teacher33_v0v1v2}"
RUNTIME_TARGET="${TARGET_RUNTIME:-/home/mingyujung/shared/hdd_ext/nvme8000/mingyujung/gnaroshi_vla/envs/seer_libero_sd1_20260819}"
STABLE_RUNTIME="${STABLE_RUNTIME:-/home/mingyujung/shared/hdd_ext/nvme8000/mingyujung/gnaroshi_vla/envs/seer_libero_canonical}"
BUNDLE="${CANONICAL_BUNDLE:?Set CANONICAL_BUNDLE to the transferred tar.gz}"
DATASET_ROOT="${S18_DATASET_ROOT:-/home/mingyujung/shared/hdd_ext/nvme8000/mingyujung/gnaroshi_vla/datasets/LIBERO_DATASETS/libero_10_converted}"
DATASET_META="${DATASET_ROOT}/libero_10_converted/meta_info.h5"
EXPECTED_BRANCH="exp/simvla-latentloop-cache-pipeline-20260804"
EXPECTED_COMMIT="eb28a80ed899ffd15ad079bf2cb5fbb81eb8a972"
EXPECTED_DATASET_META_SHA="08f765dbc4695e9618517762a4e1b297d3062a485a06f15ff52f6e52000e3352"
[[ -f "${BUNDLE}" && -f "${BUNDLE}.sha256" ]] || { echo "[ERROR] bundle or SHA file missing" >&2; exit 1; }
[[ -f "${TARGET}/.campaign_scaffold" ]] || { echo "[ERROR] target is not the prepared scaffold: ${TARGET}" >&2; exit 1; }
[[ ! -e "${TARGET}.preimport_scaffold" ]] || { echo "[ERROR] scaffold backup already exists" >&2; exit 1; }
[[ ! -e "${TARGET}.failed_import" ]] || { echo "[ERROR] failed import evidence already exists" >&2; exit 1; }
[[ ! -e "${STABLE_TARGET}" && ! -L "${STABLE_TARGET}" ]] || { echo "[ERROR] stable source alias already exists" >&2; exit 1; }
[[ ! -e "${ARTIFACT_TARGET}" && ! -e "${STABLE_ARTIFACT}" && ! -L "${STABLE_ARTIFACT}" ]] \
  || { echo "[ERROR] canonical artifact path or alias already exists" >&2; exit 1; }
[[ -x "${RUNTIME_TARGET}/bin/python" ]] || { echo "[ERROR] locked runtime is missing" >&2; exit 1; }
[[ ! -e "${STABLE_RUNTIME}" && ! -L "${STABLE_RUNTIME}" ]] || { echo "[ERROR] stable runtime alias already exists" >&2; exit 1; }
EXPECTED_BUNDLE_SHA="$(awk 'NR == 1 { print $1 }' "${BUNDLE}.sha256")"
[[ "${EXPECTED_BUNDLE_SHA}" =~ ^[[:xdigit:]]{64}$ ]] \
  || { echo "[ERROR] invalid bundle SHA file: ${BUNDLE}.sha256" >&2; exit 1; }
ACTUAL_BUNDLE_SHA="$(sha256sum "${BUNDLE}" | awk '{ print $1 }')"
[[ "${ACTUAL_BUNDLE_SHA}" == "${EXPECTED_BUNDLE_SHA}" ]] \
  || { echo "[ERROR] canonical bundle SHA mismatch" >&2; exit 1; }
echo "$(basename "${BUNDLE}"): OK"
[[ -f "${DATASET_META}" ]] || { echo "[ERROR] canonical dataset metadata missing: ${DATASET_META}" >&2; exit 1; }
[[ "$(sha256sum "${DATASET_META}" | awk '{ print $1 }')" == "${EXPECTED_DATASET_META_SHA}" ]] \
  || { echo "[ERROR] canonical dataset metadata SHA mismatch" >&2; exit 1; }

case "${CANONICAL_IMPORT_VERIFY_ONLY:-0}" in
  0) ;;
  1)
    echo "CANONICAL_IMPORT_PREFLIGHT_PASS"
    exit 0
    ;;
  *)
    echo "[ERROR] CANONICAL_IMPORT_VERIFY_ONLY must be 0 or 1" >&2
    exit 1
    ;;
esac

TMP="$(mktemp -d "$(dirname "${TARGET}")/.canonical_import.XXXXXX")"
ARTIFACT_STAGING=""
ARTIFACT_CREATED=0
SOURCE_SWAPPED=0
IMPORT_COMMITTED=0
cleanup() {
  if [[ "${IMPORT_COMMITTED}" != "1" ]]; then
    if [[ -L "${STABLE_TARGET}" && "$(readlink "${STABLE_TARGET}")" == "${TARGET}" ]]; then
      rm "${STABLE_TARGET}"
    fi
    if [[ -L "${STABLE_ARTIFACT}" && "$(readlink "${STABLE_ARTIFACT}")" == "${ARTIFACT_TARGET}" ]]; then
      rm "${STABLE_ARTIFACT}"
    fi
    if [[ -L "${STABLE_RUNTIME}" && "$(readlink "${STABLE_RUNTIME}")" == "${RUNTIME_TARGET}" ]]; then
      rm "${STABLE_RUNTIME}"
    fi
    if [[ "${SOURCE_SWAPPED}" == "1" && -e "${TARGET}.preimport_scaffold" ]]; then
      mv "${TARGET}" "${TARGET}.failed_import"
      mv "${TARGET}.preimport_scaffold" "${TARGET}"
      SOURCE_SWAPPED=0
    fi
  fi
  rm -rf "${TMP}"
  if [[ -n "${ARTIFACT_STAGING}" && -e "${ARTIFACT_STAGING}" ]]; then
    rm -rf "${ARTIFACT_STAGING}"
  fi
  if [[ "${ARTIFACT_CREATED}" == "1" && -e "${ARTIFACT_TARGET}" ]]; then
    rm -rf "${ARTIFACT_TARGET}"
  fi
}
trap cleanup EXIT
mkdir -p "${TMP}/payload"
tar -C "${TMP}/payload" -xzf "${BUNDLE}"
(cd "${TMP}/payload" && sha256sum -c members.sha256)
git clone --quiet "${TMP}/payload/canonical_base.bundle" "${TMP}/repo"
git -C "${TMP}/repo" checkout --quiet -B "${EXPECTED_BRANCH}" "${EXPECTED_COMMIT}"
git -C "${TMP}/repo" apply --binary "${TMP}/payload/canonical_tracked_dirty.patch"
cp -a "${TMP}/payload/source_overlay/." "${TMP}/repo/"
cp -a "${TMP}/payload/.canonical" "${TMP}/repo/.canonical"
git clone --quiet "${TMP}/payload/libero_base.bundle" "${TMP}/repo/.canonical/libero_source"
git -C "${TMP}/repo/.canonical/libero_source" checkout --quiet -B master 8f1084e3132a39270c3a13ebe37270a43ece2a01
git -C "${TMP}/repo/.canonical/libero_source" apply --binary "${TMP}/payload/libero_tracked_dirty.patch"
cp -a "${TARGET}/." "${TMP}/repo/"

python3 - "${TMP}/repo" "${DATASET_ROOT}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
root, dataset = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
contract = json.loads((root / "s18_canonical_source_lock.json").read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
for relative, expected in contract["source_sha256"].items():
    if sha(root / relative) != expected: raise RuntimeError(f"source mismatch after import: {relative}")
diff = subprocess.check_output(["git", "-C", str(root), "diff", "--binary", "--no-ext-diff", "HEAD"])
if hashlib.sha256(diff).hexdigest() != contract["tracked_dirty_diff_sha256"]: raise RuntimeError("dirty fingerprint mismatch after import")
manifest = root / ".canonical/manifests/dataset_files.sha256"
if not manifest.is_file(): raise RuntimeError("dataset manifest missing")
meta = dataset / "libero_10_converted/meta_info.h5"
if not meta.is_file() or sha(meta) != contract["dataset_meta_sha256"]: raise RuntimeError("s18 dataset identity mismatch")
for item in contract["canonical_reference"].values():
    path = root / item["relative_path"]
    if not path.is_file() or sha(path) != item["sha256"]: raise RuntimeError("canonical reference mismatch")
libero = root / contract["libero_source"]["relative_path"]
libero_commit = subprocess.check_output(["git", "-C", str(libero), "rev-parse", "HEAD"], text=True).strip()
libero_branch = subprocess.check_output(["git", "-C", str(libero), "branch", "--show-current"], text=True).strip()
libero_diff = subprocess.check_output(["git", "-C", str(libero), "diff", "--binary", "--no-ext-diff", "HEAD"])
if libero_commit != contract["libero_source"]["commit"] or libero_branch != contract["libero_source"]["branch"]:
    raise RuntimeError("canonical LIBERO revision mismatch")
if hashlib.sha256(libero_diff).hexdigest() != contract["libero_source"]["tracked_dirty_diff_sha256"]:
    raise RuntimeError("canonical LIBERO dirty fingerprint mismatch")
PY

(cd "${DATASET_ROOT}" && sha256sum -c "${TMP}/repo/.canonical/manifests/dataset_files.sha256")

# Keep large model artifacts on shared storage. The source tree retains the
# canonical relative path through a symlink, so source/runtime call sites do
# not expose the physical dated artifact directory in process arguments.
mkdir -p "$(dirname "${ARTIFACT_TARGET}")"
ARTIFACT_STAGING="$(mktemp -d "$(dirname "${ARTIFACT_TARGET}")/.canonical_artifacts.XXXXXX")"
mv "${TMP}/repo/.canonical/artifacts" "${ARTIFACT_STAGING}/payload"
mv "${ARTIFACT_STAGING}/payload" "${ARTIFACT_TARGET}"
rmdir "${ARTIFACT_STAGING}"
ARTIFACT_STAGING=""
ARTIFACT_CREATED=1
ln -s "${ARTIFACT_TARGET}" "${TMP}/repo/.canonical/artifacts"

mv "${TARGET}" "${TARGET}.preimport_scaffold"
mv "${TMP}/repo" "${TARGET}"
SOURCE_SWAPPED=1
if ! CANONICAL_DATASET_META="${DATASET_META}" \
  python3 "${TARGET}/source_gate.py" --repo-root "${TARGET}" \
    --contract "${TARGET}/s18_canonical_source_lock.json" \
    --output "${TARGET}/.canonical/source_gate_pass.json"; then
  mv "${TARGET}" "${TARGET}.failed_import"
  mv "${TARGET}.preimport_scaffold" "${TARGET}"
  SOURCE_SWAPPED=0
  echo "[ERROR] import rolled back after source gate failure" >&2
  exit 1
fi
date --iso-8601=seconds > "${TARGET}/.canonical/import_complete.txt"
ln -s "${TARGET}" "${STABLE_TARGET}"
ln -s "${ARTIFACT_TARGET}" "${STABLE_ARTIFACT}"
ln -s "${RUNTIME_TARGET}" "${STABLE_RUNTIME}"
SOURCE_SWAPPED=0
ARTIFACT_CREATED=0
IMPORT_COMMITTED=1
echo "[DONE] canonical tree imported: ${TARGET}"
echo "[DONE] stable source alias: ${STABLE_TARGET}"
echo "[DONE] stable artifact alias: ${STABLE_ARTIFACT}"
echo "[DONE] stable runtime alias: ${STABLE_RUNTIME}"
