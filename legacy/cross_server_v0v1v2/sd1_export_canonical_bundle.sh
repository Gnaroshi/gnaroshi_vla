#!/usr/bin/env bash
set -euo pipefail

# Run on sd1 only. This script deliberately refuses the later modified tree.
EXPECTED_HOST="${EXPECTED_HOST:-jbrserver1}"
REPO="${SD1_REPO_ROOT:-/home/mingyujung/private/gnaroshi_vla_latentloop_canonical}"
LOCK_DIR="${SOURCE_LOCK_DIR:-${REPO}/.canonical/source_lock}"
SOURCE_CONTRACT="${SOURCE_CONTRACT:-${REPO}/s18_canonical_source_lock.json}"
TEACHER="${TEACHER33_PATH:-${REPO}/.canonical/artifacts/teacher33.pth}"
ADAPTER="${ADAPTER39_PATH:-${REPO}/.canonical/artifacts/adapter39.pth}"
VIT="${VIT_MAE_PATH:-${REPO}/.canonical/artifacts/mae_pretrain_vit_base.pth}"
DATASET_ROOT="${LIBERO_DATASET_ROOT:-/home/mingyujung/shared/nvme1/mingyujung/robotics/seer/seer_node2/LIBERO_DATASETS/libero_10_converted}"
LIBERO_ROOT="${LIBERO_PATH:-/home/mingyujung/private/LIBERO}"
REFERENCE_ROOT="${CANONICAL_REFERENCE_ROOT:-${REPO}/.canonical/reference}"
REFERENCE_FULL="${REFERENCE_ROOT}/full_k1_eval_episode_metrics.csv"
REFERENCE_V0="${REFERENCE_ROOT}/v0_k4_eval_episode_metrics.csv"
OUT_DIR="${CANONICAL_EXPORT_DIR:-/home/mingyujung/private/seer_canonical_transfer}"
ARCHIVE="${OUT_DIR}/sd1_teacher33_canonical_bundle.tar.gz"
EXPECTED_LOCK_SHA="4bff3c564792748e93c194610db6b0e99933b353520d3c4600250e9d73023fe2"
EXPECTED_BRANCH="exp/simvla-latentloop-cache-pipeline-20260804"
EXPECTED_COMMIT="eb28a80ed899ffd15ad079bf2cb5fbb81eb8a972"
EXPECTED_DIFF_SHA="6156dd8a28e0134cfaacdb4673a1f97e8d2a82ad5bd489172cf61e09c4124d4a"

[[ "$(hostname)" == "${EXPECTED_HOST}" ]] || { echo "[ERROR] expected ${EXPECTED_HOST}" >&2; exit 1; }
[[ ! -e "${OUT_DIR}" ]] || { echo "[ERROR] refusing existing export: ${OUT_DIR}" >&2; exit 1; }
for path in "${REPO}" "${LOCK_DIR}/source_lock_manifest.json" "${LOCK_DIR}/source_hashes.json" "${SOURCE_CONTRACT}" \
  "${LOCK_DIR}/canonical_episode_manifest.csv" "${TEACHER}" "${ADAPTER}" "${VIT}" \
  "${DATASET_ROOT}/libero_10_converted/meta_info.h5" "${LIBERO_ROOT}" \
  "${REFERENCE_FULL}" "${REFERENCE_V0}"; do
  [[ -e "${path}" ]] || { echo "[ERROR] missing: ${path}" >&2; exit 1; }
done
[[ "$(sha256sum "${LOCK_DIR}/source_lock_manifest.json" | awk '{print $1}')" == "${EXPECTED_LOCK_SHA}" ]] \
  || { echo "[ERROR] source-lock SHA mismatch" >&2; exit 1; }

# Verify all identities before creating the requested output directory.
python3 - "${REPO}" "${LOCK_DIR}" "${TEACHER}" "${ADAPTER}" "${VIT}" \
  "${DATASET_ROOT}/libero_10_converted/meta_info.h5" "${REFERENCE_FULL}" "${REFERENCE_V0}" "${SOURCE_CONTRACT}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
repo, lock_dir = map(pathlib.Path, sys.argv[1:3])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
expected = {
    pathlib.Path(sys.argv[3]): "a999bf839acfb6f77beb8b86576933254f1981d2bacd1f0d269da093d7205cc5",
    pathlib.Path(sys.argv[4]): "badc74e135003fee91ccc69c76fe4f225aece856f487236ca7f626424504f132",
    pathlib.Path(sys.argv[5]): "aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d",
    pathlib.Path(sys.argv[6]): "08f765dbc4695e9618517762a4e1b297d3062a485a06f15ff52f6e52000e3352",
    pathlib.Path(sys.argv[7]): "75deb57678bd36860253c98a80850388593208de8f14095d717d7994e8ecfe33",
    pathlib.Path(sys.argv[8]): "a7739ea954dd7c26da091cfd6e224e2bb7b0c5cc965002356135a28a84855d6a",
}
for path, value in expected.items():
    got = sha(path)
    if got != value: raise RuntimeError(f"identity mismatch: {path}: {got}")
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
if head != "eb28a80ed899ffd15ad079bf2cb5fbb81eb8a972":
    raise RuntimeError(f"canonical HEAD mismatch: {head}")
branch = subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip()
if branch != "exp/simvla-latentloop-cache-pipeline-20260804":
    raise RuntimeError(f"canonical branch mismatch: {branch}")
diff = subprocess.check_output(["git", "-C", str(repo), "diff", "--binary", "--no-ext-diff", "HEAD"])
if hashlib.sha256(diff).hexdigest() != "6156dd8a28e0134cfaacdb4673a1f97e8d2a82ad5bd489172cf61e09c4124d4a":
    raise RuntimeError("complete tracked dirty fingerprint mismatch")
source = json.loads((lock_dir / "source_hashes.json").read_text())
mismatch = [rel for rel, value in source.items() if not (repo / rel).is_file() or sha(repo / rel) != value]
if mismatch:
    raise RuntimeError(
        "exact canonical dirty snapshot is unavailable; locked source mismatches: "
        + ", ".join(mismatch)
    )
contract_source = json.loads(pathlib.Path(sys.argv[9]).read_text())["source_sha256"]
contract_mismatch = [
    rel for rel, value in contract_source.items()
    if not (repo / rel).is_file() or sha(repo / rel) != value
]
if contract_mismatch:
    raise RuntimeError(
        "canonical runtime source closure is incomplete: "
        + ", ".join(contract_mismatch)
    )
commit = subprocess.check_output(["git", "-C", str(repo), "cat-file", "-t", "eb28a80ed899ffd15ad079bf2cb5fbb81eb8a972"], text=True).strip()
if commit != "commit": raise RuntimeError("canonical base commit is unavailable")
PY

python3 - "${LIBERO_ROOT}" <<'PY'
import hashlib, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
def run(*args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
if run("branch", "--show-current") != "master":
    raise RuntimeError("canonical LIBERO branch mismatch")
if run("rev-parse", "HEAD") != "8f1084e3132a39270c3a13ebe37270a43ece2a01":
    raise RuntimeError("canonical LIBERO commit mismatch")
diff = subprocess.check_output(["git", "-C", str(root), "diff", "--binary", "--no-ext-diff", "HEAD"])
if hashlib.sha256(diff).hexdigest() != "0d6f27c5832243b085b82176538b2cfea2624492fa7ff408fa964f5f058c1b72":
    raise RuntimeError("canonical LIBERO dirty fingerprint mismatch")
status = subprocess.check_output(
    ["git", "-C", str(root), "status", "--short", "--untracked-files=all"],
    text=True,
).splitlines()
if status != [" M libero/libero/__init__.py"]:
    raise RuntimeError("canonical LIBERO status differs from the locked source")
if sha(root / "libero/libero/__init__.py") != "94d9c14dbadce46f4057761422e02bd540e1a125a3489f0e72b1124bb2381604":
    raise RuntimeError("canonical LIBERO config source mismatch")
PY

case "${CANONICAL_VERIFY_ONLY:-0}" in
  0) ;;
  1)
    echo "CANONICAL_EXPORT_PREFLIGHT_PASS"
    exit 0
    ;;
  *)
    echo "[ERROR] CANONICAL_VERIFY_ONLY must be 0 or 1" >&2
    exit 1
    ;;
esac

TMP="$(mktemp -d /tmp/sd1_canonical_export.XXXXXX)"
trap 'rm -rf "${TMP}"' EXIT
PAYLOAD="${TMP}/payload"
mkdir -p "${PAYLOAD}/source_overlay" "${PAYLOAD}/.canonical/source_lock" \
  "${PAYLOAD}/.canonical/artifacts" "${PAYLOAD}/.canonical/manifests" \
  "${PAYLOAD}/.canonical/reference"

git -C "${REPO}" bundle create "${PAYLOAD}/canonical_base.bundle" --all
git -C "${REPO}" diff --binary --no-ext-diff HEAD > "${PAYLOAD}/canonical_tracked_dirty.patch"
git -C "${LIBERO_ROOT}" bundle create "${PAYLOAD}/libero_base.bundle" --all
git -C "${LIBERO_ROOT}" diff --binary --no-ext-diff HEAD > "${PAYLOAD}/libero_tracked_dirty.patch"
python3 - "${REPO}" "${SOURCE_CONTRACT}" "${PAYLOAD}/source_overlay" <<'PY'
import json, pathlib, shutil, sys
repo, contract_path, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
for relative in json.loads(contract_path.read_text())["source_sha256"]:
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / relative, target)
PY
cp "${LOCK_DIR}/source_lock_manifest.json" "${LOCK_DIR}/source_hashes.json" \
  "${LOCK_DIR}/source_lock_report.md" "${LOCK_DIR}/canonical_episode_manifest.csv" \
  "${PAYLOAD}/.canonical/source_lock/"
cp "${TEACHER}" "${PAYLOAD}/.canonical/artifacts/teacher33.pth"
cp "${ADAPTER}" "${PAYLOAD}/.canonical/artifacts/adapter39.pth"
cp "${VIT}" "${PAYLOAD}/.canonical/artifacts/mae_pretrain_vit_base.pth"
cp "${REFERENCE_FULL}" "${PAYLOAD}/.canonical/reference/full_k1_eval_episode_metrics.csv"
cp "${REFERENCE_V0}" "${PAYLOAD}/.canonical/reference/v0_k4_eval_episode_metrics.csv"

python3 - "${DATASET_ROOT}" "${PAYLOAD}/.canonical/manifests/dataset_files.sha256" <<'PY'
import hashlib, pathlib, sys
root, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
with output.open("w") as handle:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""): digest.update(chunk)
        handle.write(f"{digest.hexdigest()}  {path.relative_to(root)}\n")
PY
python3 - "${LIBERO_ROOT}" "${PAYLOAD}/.canonical/manifests/libero_source_identity.json" <<'PY'
import hashlib, json, pathlib, subprocess, sys
root, output = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def run(*args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
diff = subprocess.check_output(["git", "-C", str(root), "diff", "--binary", "--no-ext-diff", "HEAD"])
status = subprocess.check_output(
    ["git", "-C", str(root), "status", "--short", "--untracked-files=all"],
    text=True,
).splitlines()
payload = {"root": str(root), "commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(), "status_short": status}
output.write_text(json.dumps(payload, indent=2) + "\n")
PY

CONDA_SH="${CONDA_SH:-/home/mingyujung/miniconda3/etc/profile.d/conda.sh}"
[[ -f "${CONDA_SH}" ]] || { echo "[ERROR] missing conda activation script: ${CONDA_SH}" >&2; exit 1; }
source "${CONDA_SH}"
conda activate seer_libero
conda list --explicit > "${PAYLOAD}/.canonical/manifests/conda_explicit.txt"
python -m pip freeze --all > "${PAYLOAD}/.canonical/manifests/pip_freeze.txt"
python -VV > "${PAYLOAD}/.canonical/manifests/python_version.txt" 2>&1

python3 - "${PAYLOAD}" "${DATASET_ROOT}" "${LIBERO_ROOT}" <<'PY'
import hashlib, json, pathlib, sys
payload, dataset, libero = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
lock = payload / ".canonical/source_lock"
manifest = {
  "schema_version": 1,
  "source_identity": {"branch": "exp/simvla-latentloop-cache-pipeline-20260804", "commit": "eb28a80ed899ffd15ad079bf2cb5fbb81eb8a972", "tracked_dirty_diff_sha256": "6156dd8a28e0134cfaacdb4673a1f97e8d2a82ad5bd489172cf61e09c4124d4a", "source_lock_sha256": sha(lock / "source_lock_manifest.json")},
  "artifacts": {name: {"sha256": sha(payload / path), "path": path} for name, path in {"teacher33": ".canonical/artifacts/teacher33.pth", "adapter39": ".canonical/artifacts/adapter39.pth", "vit_mae": ".canonical/artifacts/mae_pretrain_vit_base.pth"}.items()},
  "dataset": {"name": "libero_10_converted", "source_root": str(dataset), "meta_sha256": "08f765dbc4695e9618517762a4e1b297d3062a485a06f15ff52f6e52000e3352", "file_manifest": ".canonical/manifests/dataset_files.sha256"},
  "runtime": {"environment_name": "seer_libero", "conda_explicit": ".canonical/manifests/conda_explicit.txt", "pip_freeze": ".canonical/manifests/pip_freeze.txt"},
  "libero": {"source_root": str(libero), "identity": ".canonical/manifests/libero_source_identity.json", "branch": "master", "commit": "8f1084e3132a39270c3a13ebe37270a43ece2a01", "tracked_dirty_diff_sha256": "0d6f27c5832243b085b82176538b2cfea2624492fa7ff408fa964f5f058c1b72", "bundle": "libero_base.bundle", "tracked_patch": "libero_tracked_dirty.patch"},
  "evaluation": {"episode_manifest_sha256": sha(lock / "canonical_episode_manifest.csv"), "tasks": 10, "episodes_per_task": 20, "seed": 42, "max_steps": 600, "control_hz": 20.0, "precision": "fp32", "temporal_ensembling": True, "temporal_ensemble_temperature": 0.01, "action_prediction_horizon": 3},
  "canonical_reference": {name: {"sha256": sha(payload / path), "path": path} for name, path in {"full_k1": ".canonical/reference/full_k1_eval_episode_metrics.csv", "v0_k4": ".canonical/reference/v0_k4_eval_episode_metrics.csv"}.items()},
  "member_hash_manifest": {"path": "members.sha256", "coverage": "all payload members except members.sha256 and canonical_transfer_manifest.json, which are integrity roots"}
}
(payload / "canonical_transfer_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

(cd "${PAYLOAD}" && find . -type f ! -name members.sha256 ! -name canonical_transfer_manifest.json -print0 | sort -z | xargs -0 sha256sum > members.sha256)
mkdir -p "${OUT_DIR}"
tar -C "${PAYLOAD}" -czf "${ARCHIVE}" .
(cd "${OUT_DIR}" && sha256sum "$(basename "${ARCHIVE}")" > "$(basename "${ARCHIVE}").sha256")
cp "${PAYLOAD}/canonical_transfer_manifest.json" "${PAYLOAD}/members.sha256" "${OUT_DIR}/"
echo "[DONE] ${ARCHIVE}"
