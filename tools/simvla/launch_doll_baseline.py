"""Reuse completed checks, collect operator review, then open the baseline GUI."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import getpass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from architectures.simvla.adapters.latentloop_real_deploy.contracts import (
    DeploymentContract,
    load_deployment_contract,
    require_live_authorization,
    sha256_file,
)
from architectures.simvla.adapters.latentloop_real_deploy.source_lock import verify_source_snapshots

ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_REVIEW_FIELDS = (
    "hardware_configuration_reviewed", "camera_role_mapping_verified",
    "task_home_pose_verified", "workspace_bounds_verified",
    "control_limits_reviewed", "gripper_startup_behavior_reviewed",
    "gripper_no_software_stop_acknowledged", "physical_emergency_stop_verified",
    "runtime_timing_reviewed",
)


def numbers(text: str, count: int, *, positive: bool = False) -> list[float]:
    values = [float(x) for x in text.replace(",", " ").split()]
    if len(values) != count or not all(math.isfinite(x) for x in values):
        raise ValueError(f"유한한 숫자 {count}개를 공백으로 구분해 입력해야 합니다.")
    if positive and not all(x > 0 for x in values):
        raise ValueError("추종오차 한계는 0보다 커야 합니다.")
    return values


def completed_report(log_root: Path, stem: str, filename: str) -> tuple[Path, dict]:
    candidates = []
    for run in log_root.glob(stem + "*"):
        suffix = run.name[len(stem):]
        if suffix and not (suffix.startswith("_r") and suffix[2:].isdigit()):
            continue
        report = run / "output" / filename
        status = run / "exit_code.txt"
        if report.is_file() and status.is_file() and status.read_text().strip() == "0":
            candidates.append(report)
    if not candidates:
        raise ValueError(f"완료된 점검 결과가 없습니다: {stem}/{filename}")
    path = max(candidates, key=lambda x: x.stat().st_mtime_ns)
    return path, json.loads(path.read_text())


def validate_evidence(contract: DeploymentContract, artifact: dict, profile: dict) -> None:
    if artifact.get("verdict") != "ARTIFACT_PREFLIGHT_PASS" or artifact.get("actions_finite") is not True:
        raise ValueError("모델 점검이 통과하지 않았습니다.")
    if profile.get("verdict") != "READ_ONLY_PROFILE_PASS" or profile.get("policy_schedule_validated") is not True:
        raise ValueError("실제 입력 점검이 통과하지 않았습니다.")
    if profile.get("sensor_contract_validated") is not True or profile.get("robot_command_issued") is not False:
        raise ValueError("실제 입력 점검의 비구동/센서 기록이 일치하지 않습니다.")
    expected_hashes = {name: item.sha256 for name, item in contract.artifacts.items()}
    for metadata in (artifact["deployment"], profile["controller"]):
        if metadata.get("deployment_method") != "baseline":
            raise ValueError("baseline 점검 결과가 아닙니다.")
        if metadata.get("deployment_id") != contract.deployment_id:
            raise ValueError("점검한 배포 모델이 다릅니다.")
        if metadata.get("runtime_source_identity_sha256") != contract.payload["runtime_source_identity_sha256"]:
            raise ValueError("점검 이후 추론 코드가 변경됐습니다.")
        if metadata.get("artifact_sha256") != expected_hashes:
            raise ValueError("점검 이후 모델 또는 정규화 파일이 변경됐습니다.")
        for name, expected in (("policy_contract", contract.policy), ("state_contract", contract.state), ("action_contract", contract.action)):
            if metadata.get(name) != expected:
                raise ValueError(f"점검 이후 {name} 설정이 변경됐습니다.")
    if profile.get("deployment_target_hz") != contract.runtime["control_frequency_hz"]:
        raise ValueError("실제 동작 목표 주기가 점검 당시와 다릅니다.")


def reviewed_payload(contract: DeploymentContract, profile: dict, minimum: list[float], maximum: list[float], tracking: list[float], max_steps: int, approval: str) -> dict:
    if approval != "DEPLOY":
        raise PermissionError("승인하지 않았습니다. 하드웨어를 초기화하지 않습니다.")
    if not (0 < max_steps <= int(contract.runtime["max_steps"])):
        raise ValueError("실행 길이는 1 이상, 기존 최대 step 이하이어야 합니다.")
    if len(minimum) != 3 or len(maximum) != 3 or len(tracking) != 2:
        raise ValueError("작업범위 또는 추종오차 차원이 잘못됐습니다.")
    if not all(math.isfinite(v) for v in minimum + maximum + tracking) or not all(v > 0 for v in tracking):
        raise ValueError("작업범위는 유한해야 하며 추종오차는 양수이어야 합니다.")
    observed = profile["observed_tcp_xyz_m"]
    for i in range(3):
        if not minimum[i] < maximum[i]:
            raise ValueError("각 축의 최소값은 최대값보다 작아야 합니다.")
        if not minimum[i] <= observed["min"][i] <= observed["max"][i] <= maximum[i]:
            raise ValueError("입력한 작업범위가 방금 관측한 TCP 위치를 포함하지 않습니다.")
    payload = copy.deepcopy(contract.payload)
    robot = payload["hardware"]["robot"]
    robot["workspace_m"] = {"min": minimum, "max": maximum}
    robot["workspace_source"] = "Operator-entered and reviewed at baseline GUI launch; not inferred from stationary TCP."
    robot["control"]["tracking_error_guard"] = {
        "enabled": True,
        "max_translation_error_m": tracking[0],
        "max_rotation_error_rad": tracking[1],
    }
    payload["runtime"]["max_steps"] = max_steps
    payload["runtime"]["num_rollouts_per_instruction"] = 1
    review = payload["safety_review"]
    review.update({name: True for name in PHYSICAL_REVIEW_FIELDS})
    review.update(model_preflight_passed=True, read_only_profile_passed=True,
                  live_authorized=True, baseline_bounded_canary_passed=False,
                  approved_by=getpass.getuser(), approved_at=datetime.now(timezone.utc).isoformat())
    return payload


def main() -> int:
    runtime = Path.home() / "gnaroshi_vla_runtime"
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=runtime / "artifacts/stackcupanddoll/deployment_manifest.site.json")
    parser.add_argument("--log-root", type=Path, default=Path(os.environ.get("SIMVLA_REAL_LOG_ROOT", runtime / "results/simvla/real_deploy")))
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--check", action="store_true", help="기존 결과만 확인. 하드웨어/GUI 실행 없음.")
    args = parser.parse_args()
    verify_source_snapshots()
    contract = load_deployment_contract(args.manifest, verify_artifacts=True)
    model_path, model = completed_report(args.log_root, "artifact-preflight_baseline", "artifact_preflight.json")
    profile_path, profile = completed_report(args.log_root, "read-only-profile_baseline", "read_only_summary.json")
    validate_evidence(contract, model, profile)
    with (profile_path.parent / "read_only_steps.jsonl").open() as stream:
        first = json.loads(next(stream))
    for role in ("exterior", "wrist"):
        if first[f"{role}_camera"]["serial"] != contract.hardware["cameras"][role]["serial"]:
            raise ValueError("점검 이후 카메라 역할/serial이 변경됐습니다.")
    print("기존 모델·실제 입력 점검 확인 완료. 재학습/재추론 점검은 하지 않습니다.", flush=True)
    if args.check:
        print("CHECK_PASS: 하드웨어 초기화 및 로봇 명령 없음. 실제 실행에는 현장 입력/승인이 필요합니다.")
        return 0
    if not sys.stdin.isatty() or not os.environ.get("DISPLAY"):
        raise RuntimeError("inference computer의 모니터가 연결된 터미널에서 실행하세요.")
    robot = contract.hardware["robot"]
    print(f"\nDoll baseline / H=10, R=5, flow=10 / 목표 {contract.runtime['control_frequency_hz']} Hz")
    print(f"한 번의 rollout, 최대 {args.max_steps} step. GUI에서 Start New Rollout을 눌러 시작합니다.")
    print(f"로봇 IP: {robot['ip']} / 시작 관절각(rad): {robot['home_pose']}")
    print("로봇 속도/servo 및 그리퍼 초기화 설정:")
    print(json.dumps({"control": robot["control"], "gripper": robot["gripper"]}, indent=2))
    print(f"직전 비구동 결과: 목표 {profile['profile_target_hz']} Hz, 처리 {profile['read_only_tick_hz']:.2f} Hz.")
    print("이는 실제 control Hz가 아닙니다. 현재 live 목표 15 Hz 및 학습 action/state 변환은 유지합니다.")
    print("\n안전한 TCP 범위를 로봇 base 좌표계(m)로 입력하세요. 예시값을 자동 승인하지 않습니다.")
    print("범위는 현재 위치뿐 아니라 home 자세와 작업 중 이동을 포함해야 합니다.")
    minimum = numbers(input("최소 x y z (m): "), 3)
    maximum = numbers(input("최대 x y z (m): "), 3)
    tracking = numbers(input("허용 위치 추종오차(m), 회전 추종오차(rad): "), 2, positive=True)
    print(f"\n입력 범위: {minimum} ~ {maximum}; 추종오차: {tracking} (m, rad)")
    print("승인 전 확인: 카메라 배치/로봇/home가 점검 때와 동일하고 home 이동 경로도 비어 있습니다.")
    print("위 속도/그리퍼 설정, 작업범위, 타이밍을 검토했고 물리 비상정지를 확인했습니다.")
    print("GUI 초기화 중 그리퍼가 활성화될 수 있습니다. 소프트웨어 Stop은 그리퍼 정지를 보장하지 않습니다.")
    approval = input("직접 확인했고 실제 구동을 승인하면 DEPLOY 입력 (그 외 취소): ").strip()
    payload = reviewed_payload(contract, profile, minimum, maximum, tracking, args.max_steps, approval)
    payload["operator_review_evidence"] = {
        "artifact_preflight": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "read_only_profile": {"path": str(profile_path), "sha256": sha256_file(profile_path)},
        "launcher_sha256": sha256_file(__file__),
        "original_site_manifest_sha256": sha256_file(contract.path),
        "confirmation": approval,
    }
    # Keep relative artifact paths valid, and never overwrite the site template.
    fd, filename = tempfile.mkstemp(prefix="deployment_manifest.live-baseline-", suffix=".json", dir=contract.path.parent)
    path = Path(filename)
    env = dict(os.environ, SIMVLA_REAL_LIVE_RUN="1", SIMVLA_REAL_DEPLOYMENT_ID=contract.deployment_id)
    with os.fdopen(fd, "w") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    candidate = load_deployment_contract(path, verify_artifacts=False)
    os.environ.update({key: env[key] for key in ("SIMVLA_REAL_LIVE_RUN", "SIMVLA_REAL_DEPLOYMENT_ID")})
    require_live_authorization(candidate, deployment_method="baseline")
    print(f"현장 승인 설정: {path}", flush=True)
    return subprocess.call(["bash", str(ROOT / "architectures/simvla/wrappers/deploy_latentloop_real.sh"),
                            "live", "--manifest", str(path), "--method", "baseline"], env=env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EOFError, KeyboardInterrupt):
        print("\n취소했습니다. 실행 중인 GUI가 있었다면 현장 정지 상태를 확인하세요.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nBASELINE_DEPLOY_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
