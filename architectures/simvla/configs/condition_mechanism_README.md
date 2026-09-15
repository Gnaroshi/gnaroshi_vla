# SimVLA Condition 원인 분석

## 실행

rb2의 tmux pane에서 다음 한 줄을 실행한다. sd1에서는 실행하지 않는다.

```bash
bash /home/mingyujung/private/gnaroshi_vla_worktrees/simvla_condition_mechanism/architectures/simvla/wrappers/run_condition_mechanism_rb2.sh
```

설정은 옆의 `condition_mechanism_rb2.json`에서 확인한다. 바꾸면 기존 결과와
섞이지 않도록 `output`도 새 경로로 지정한다. 실행 중에는 코드를 수정하지 않는다.

## 자동 실행 순서

1. 고정 모델/정규화/cache, 실행 환경, Condition 학습의 episode 분할을 검증한다.
2. GPU0가 비고 메모리가 확보될 때까지 기다린다. 다른 프로세스를 종료하지 않는다.
3. 학습 episode 64개 window에서 zero-feature 갱신의 평균을 계산한다. optimizer는 없다.
4. 학습에서 제외된 200개 window에서 아래 12개 조건을 비교한다.
5. LIBERO-Long 10 task x 2 trial에서 물체 위치 개입 및 원래 환경의 분기 평가를 수행한다.
6. 전체 CSV, 한국어 보고서, 원자료를 포함한 ZIP을 생성한다.

## Offline 비교

모든 action은 공식 모델의 10-step action transformer로 새로 생성한다. H=10,
R=5를 유지하며 Generation updater와 coupling을 사용하지 않는다.

- 이전 condition 유지
- 융합 관측 feature를 0으로 만들고 갱신
- 정상 갱신
- 이미지만 이전 값 사용
- encoder proprioception만 이전 값 사용
- 이미지와 encoder proprioception 모두 이전 값 사용
- 정상 갱신 + action head proprioception만 이전 값 사용
- zero-feature 갱신 + action head proprioception만 이전 값 사용
- 정상 gate + zero-feature residual
- zero-feature gate + 정상 residual
- 학습 구간에서 집계한 고정 갱신
- 학습 구간에서 집계한 age별 갱신

같은 이전 teacher condition에서의 one-step 비교와, 방법별 이전 예측을 사용하는
3회 재귀 비교를 모두 저장한다. 모든 대조군의 현재 입력과 action noise를 대응시킨다.
현재 teacher action은 같은 runtime/정규화/noise로 재계산한다.
표준화 condition 오차는 진단 지표이며 실제 action decoder 계산이라고 주장하지 않는다.

## 환경 개입

고정된 기존 manifest에서 task별 trial 0, 1을 선택한다. 기본 baseline이 생성한
첫 5 action을 모든 분기에서 똑같이 실행한다. 같은 simulator state에서 task의
첫 이동 가능 관심 물체를 x축으로 2 cm 옮긴다. trial 0은 +x, trial 1은 -x다.
물체는 로봇과 접촉하기 전이어야 한다. 옮길 수 없는 사례도 이유와 함께 보존한다.

원래 환경과 개입 환경 각각에서 baseline, 유지, zero-feature, 정상 갱신을 실행한다.
최대 20 x 2 x 4 = 160개 분기이며, 각각 원래 최대 900 action까지 진행한다.
첫 query 반응과 최종 성공을 모두 기록한다. 실패 결과로 후속 단계를 중단하지 않는다.

이 결과는 개입 상황의 탐색 분석이다. 기존 논문용 500 episodes/3-seed SR과 혼합하지 않는다.
Baseline이 반응하지 않거나 Ours가 나쁜 사례도 제외하지 않는다. 향상이나 논문 기여를
자동으로 판정하지 않는다. 실로봇/배포 코드에는 접근하지 않는다.

## 재개와 오류

Window와 평가 분기마다 원자적으로 JSON을 저장한다. 같은 명령을 재실행하면 완료
단위를 건너뛴다. 실행 오류는 한 번 재시도하며, 독립된 나머지 단계와 부분 보고서는
가능한 범위에서 계속 만든다. 성능이 낮다는 이유로 중단하지 않는다.
최종 `COMPLETE`는 구성된 분석이 완료되었다는 뜻이지 Ours 성능 통과를 뜻하지 않는다.

소스/설정/모델이 바뀐 결과는 자동 혼합하지 않는다. 기존 디렉터리를 삭제하지 말고
새 output을 사용한다. 재개 시 stage summary뿐 아니라 예상 window/분기 파일도 검증한다.
완료 분기는 재사용하지만 중간에 끊긴 episode는 해당 분기 처음부터 다시 실행한다.

`Ctrl+C`는 이 launcher가 만든 자식 프로세스 그룹만 종료한다. 다른 사용자의 작업은
건드리지 않는다. 기본 wrapper는 실패해도 상위 tmux shell을 종료시키지 않도록 0으로
반환하되 오류와 실제 상태를 명시한다. 자동화에서 exit code가 필요하면
`SIMVLA_STRICT_EXIT=1`을 설정한다.

```bash
# CPU 경로 검증만
bash architectures/simvla/wrappers/run_condition_mechanism_rb2.sh --preflight
# 별도 output에서 window 1개 + 15-action 분기 검증
bash architectures/simvla/wrappers/run_condition_mechanism_rb2.sh --smoke
# 학습/평가 없이 완료 원자료의 집계만 복구
bash architectures/simvla/wrappers/run_condition_mechanism_rb2.sh --aggregate-only
```

기본 결과 위치는 storage의 `results/simvla/analysis/condition_mechanism_v1`이다.
`pipeline_status.json`, `logs/`, `report/analysis_report_ko.md`,
`simvla_condition_mechanism_results.zip`을 확인한다. 대용량 cache를 새로 생성하거나
기존 checkpoint를 복사하지 않는다.
