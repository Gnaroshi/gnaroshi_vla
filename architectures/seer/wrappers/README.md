# Seer 실행 진입점

sd1 main `gnaroshi_vla`에서 유지하는 Seer 실행 경로다. `lrnode`는 checkpoint/명령
호환성을 위한 내부 이름이며 논문명은 LatentLoop다. 실행 중 실험의 source lock을
보존하기 위해 활성 파일의 경로는 바꾸지 않는다.

## 학습

| 목적 | 진입점 |
| --- | --- |
| 공통 adapter 학습 | `lrnode/train_realworld_latentloop_adapter.sh` |
| Doll / Cabinet | `lrnode/train_realworld_{doll,cabinet}_latentloop.sh` |
| Stacking Cups / Rings | `lrnode/train_realworld_{stacking_cups,rings}_latentloop.sh` |
| Stacking Cups 다음 Rings 순차 학습 | `lrnode/run_realworld_stacking_cups_rings_sequential.sh` |
| 공통 distillation primitive | `lrnode/distill_node.sh` |

## 평가와 분석

| 목적 | 진입점 |
| --- | --- |
| 공통 Seer/LatentLoop 평가 | `lrnode/eval_lrnode_compare.sh` |
| Freshness 최초 진단 | `lrnode/run_latentloop_freshness.sh` |
| Cached-feature / gate 대조군 | `lrnode/run_latentloop_freshness_controls.sh` |
| 3-seed 확인 실험 | `lrnode/run_latentloop_freshness_confirmation.sh` |

Freshness 세 실행 파일은 서로의 결과와 source lock을 참조하므로 모두 유지한다.
논문용 simulation 재현은 각각의 검증된 별도 worktree에서 수행한다.
위 공통 평가 wrapper만으로 최신 논문 전체 프로토콜이 자동 적용된다고 가정하면 안 된다.

## Real-world 배포

하나의 진입점만 사용한다:
`architectures/seer/upstream/scripts/REAL/deploy_ll_gui_unified.sh`.
파일 위쪽의 task/method preset을 하나 선택한다. Basketball/Doll/Cabinet의
baseline과 LatentLoop를 모두 지원한다. 이번 정리에서는 로봇/GPU를 실행하지 않았다.
상세 사용법: [배포 문서](../../../docs/seer/latentloop-real-world-deploy.md).

## 정리 경계

- 완료·폐기된 campaign launcher와 upstream 안의 중복 LR-NODE launcher는 제거했다.
- 기존 Seer `scratch.sh`, `finetune.sh`, `pretrain.sh`, `eval.sh` 및 원본 CALVIN/REAL script는 유지한다.
- 과거 protocol을 source-text로 검사하는 테스트는 `tests/fixtures/seer/`를 읽는다.
- 결과/설정/삭제 경로/해시는 `codex_outputs/seer/README.md`에서 찾는다.
- `methods/`와 adapter의 import 의존 모듈은 폐기된 실험명이 붙어 있어도 지우지 않는다.
- SimVLA/OpenPI 및 별도 worktree는 이번 정리에서 수정하지 않았다.

무거운 결과는 shared의 `gnaroshi_vla/results/seer`에, 작은 분석 문서는
sd1의 `codex_outputs/seer`에 둔다.
