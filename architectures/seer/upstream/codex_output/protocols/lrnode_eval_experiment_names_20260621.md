# LR-NODE 평가 실험 이름

작성일: 2026-06-21 KST

## 공통 기준

모든 실험은 `scratch_node.sh`로 학습된 같은 checkpoint 내부 비교를 기본으로 한다.

```text
K=1:
  full-query Seer reference
  매 env step full Seer forward
  LR-NODE update call 없음

K>1:
  full Seer는 K env step마다 1회
  나머지 step은 LR-NODE latent update + 기존 action head
```

`EVAL_CONTROL_HZ`는 `OffScreenRenderEnv(control_freq=...)`로 직접 들어간다. 즉 Hz는 단순 label이 아니라 실제 LIBERO control rate다.

20 Hz 기준 episode duration을 유지하기 위해 기본적으로:

```text
actual_eval_max_steps = round(600 * control_hz / 20)
```

따라서 기본 max step은 다음과 같다.

```text
20 Hz -> 600
40 Hz -> 1200
60 Hz -> 1800
80 Hz -> 2400
```

## 실험 QRED20

정식 명칭:

```text
QRED20: 20Hz query 감소 / Seer 대체
```

목적:

```text
기존 LIBERO 20 Hz control setting에서도 ControlledLatentODE가 full Seer forward를 얼마나 대체할 수 있는지 측정한다.
```

실험 grid:

```text
20:1 20:2 20:3 20:4 20:5 20:6 20:8
```

해석:

```text
20:1 = 같은 checkpoint의 full-query Seer baseline
20:K = action/control은 20 Hz 그대로 유지하되 full Seer query는 20/K Hz로 감소
```

핵심 질문:

```text
20 Hz 환경에서도 LR-NODE가 full Seer 호출을 줄이면서 SR을 얼마나 보존하는가?
```

실행:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_20hz_query_reduction.sh
```

## 실험 HZUP20Q

정식 명칭:

```text
HZUP20Q: 20Hz full-query budget을 둔 high-Hz control
```

목적:

```text
LIBERO control_freq를 40/60/80 Hz로 실제 증가시키면서, expensive full Seer query rate는 기존 20 Hz 수준으로 유지할 수 있는지 측정한다.
```

실험 grid:

```text
20:1 40:1 40:2 60:1 60:3 80:1 80:4
```

핵심 row:

```text
40:2 = 40 Hz control, full Seer 20 Hz, LR-NODE 20 Hz
60:3 = 60 Hz control, full Seer 20 Hz, LR-NODE 40 Hz
80:4 = 80 Hz control, full Seer 20 Hz, LR-NODE 60 Hz
```

해석:

```text
H:1 = 해당 Hz에서 매 step full Seer를 호출하는 expensive upper-bound reference
H:K = 해당 Hz에서 full Seer query budget을 20 Hz 수준으로 제한하고 LR-NODE가 중간 step을 담당
```

핵심 질문:

```text
제어 주기를 높여도 full VLA query frequency를 20 Hz 수준으로 유지하면서 성능과 latency budget을 만족하는가?
```

실행:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_highhz_20hz_query_budget.sh
```

## 실험 GRID

정식 명칭:

```text
GRID: LR-NODE control-Hz / query-interval 진단 grid
```

목적:

```text
QRED20과 HZUP20Q 이후, failure boundary와 K sensitivity를 더 넓게 확인한다.
```

실험 grid:

```text
20:1 20:2 20:3 20:4 20:5 20:6 20:8
40:1 40:2 40:4
60:1 60:2 60:3 60:6
80:1 80:2 80:4 80:8
```

해석:

```text
논문 주장을 위한 primary result는 QRED20/HZUP20Q다.
GRID는 보조 분석과 ablation 성격으로 사용한다.
```

실행:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_full_grid.sh
```

## 추천 실행 순서

1. `QRED20`

20 Hz에서도 LR-NODE가 Seer full forward를 대체할 수 있는지 먼저 본다.

2. `HZUP20Q`

논문 핵심 주장인 high-Hz control + fixed full-query budget을 본다.

3. `GRID`

시간과 리소스가 남으면 추가로 돌린다.

## 결과 폴더명

공통 runner는 `EXPERIMENT_NAME`을 결과 폴더에 넣는다.

예:

```text
runs_lrnode_protocol_20260616/eval/lrnode_qred20_<run_name>_<timestamp>/
runs_lrnode_protocol_20260616/eval/lrnode_hzup20q_<run_name>_<timestamp>/
runs_lrnode_protocol_20260616/eval/lrnode_grid_<run_name>_<timestamp>/
```

각 결과 root에는 다음 파일이 저장된다.

```text
experiment_config.env
hz_sweep_summary.csv
hz_sweep_summary.md
_launch_logs/
```

각 condition에는 다음 artifact가 저장된다.

```text
analysis/eval_summary.json
analysis/eval_episode_metrics.csv
analysis/eval_latency_profile.json
eval_videos/
```
