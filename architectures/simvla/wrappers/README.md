# SimVLA 실행 파일 안내

이 폴더는 원본 SimVLA를 수정하지 않고 데이터 준비, 학습, 평가를 연결합니다.
방법 구현은 `methods/`, SimVLA 연결 코드는 `architectures/simvla/adapters/`에
있습니다. 실행 파일 이름이 비슷하더라도 서로 다른 실험 설정을 뜻할 수 있습니다.

## 원본 모델

| 파일 | 용도 |
| --- | --- |
| `prepare_libero_links.sh` | 원본 LIBERO 데이터 연결 |
| `check_libero_dataset.py` | 데이터 수, 경로와 HDF5 항목 검사 |
| `train_libero.sh` | 원본 SimVLA 학습 실행 |

## 이전 실험

| 폴더 | 내용 |
| --- | --- |
| `legacy/dcld/` | 초기 DCLD 학습, cache, 평가 스크립트 |
| `legacy/latentloop/` | 초기 chunk-aware LatentLoop 및 r1/r5 비교 스크립트 |

이 두 폴더는 실험 이력 확인용입니다. 현재 논문의 native Condition/Generation
모델이나 실제 로봇 배포 명령으로 대신 사용하지 않습니다. 폐기된 checkpoint는
존재하지 않을 수 있습니다. 이전 명령에서 해당 shell script 앞에
`legacy/dcld/` 또는 `legacy/latentloop/`를 넣어야 합니다.

`dcld_eval/`과 최상위 Python 파일은 기존 import 경로를 유지합니다.
`dcld_eval/rollout_runner.py`는 이름과 달리 후속 native Condition 평가에서도
사용하므로 이전 DCLD 실행 스크립트와 함께 삭제해서는 안 됩니다.

## 실험별 실행 경로

논문 평가와 로봇 배포는 해당 실험을 완료한 Git worktree의 실행 파일을 사용합니다.
이 체크아웃에 없는 파일을 다른 worktree에서 임의로 섞어 복사하지 않습니다.
공개 모델, 정규화 통계, 선택된 checkpoint와 평가 설정이 함께 맞아야 합니다.

현재 서버별 실행 경로와 결과 위치는 무시된 `codex_outputs/`의 SimVLA 정리
기록에서 관리합니다. 이 README는 체크아웃에 포함된 코드의 역할만 설명합니다.

## 변경 범위

이번 분류는 실행 파일의 위치와 내부 호출 경로만 변경합니다. 모델 수식,
학습 loss, optimizer, seed, renderer, action 실행 주기는 바꾸지 않습니다.
완료 실험의 별도 worktree와 upstream 소스도 수정하지 않습니다.
