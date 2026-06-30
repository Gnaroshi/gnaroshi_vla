# LR-NODE scratch_node Hz sweep 계획

작성일: 2026-06-21 KST

## 목적

현재 `scratch_node.sh`로 학습된 checkpoint는 plain `scratch.sh` baseline과 공통 weight가 다르다. 따라서 plain baseline과 직접 비교하지 않고, 같은 `scratch_node` checkpoint 내부에서 다음 비교를 수행한다.

```text
K=1:
  같은 checkpoint의 full-query Seer reference
  매 env step full Seer forward
  LR-NODE update calls = 0

K>1:
  같은 checkpoint의 LR-NODE skip mode
  full Seer는 K step마다 1회
  skip step은 LR-NODE latent update + 기존 action head
```

이 비교는 “LR-NODE 학습 checkpoint에서 실제 LIBERO control rate를 높였을 때 full-query 대비 query를 줄이면서 성능과 latency budget을 유지할 수 있는가”를 보기 위한 것이다.

## 실험 명칭

세부 실험명은 [lrnode_eval_experiment_names_20260621.md](lrnode_eval_experiment_names_20260621.md)에 고정했다.

```text
QRED20:
  20Hz Query-Reduction / Seer Replacement
  20Hz에서도 LR-NODE가 full Seer forward를 대체할 수 있는지 측정

HZUP20Q:
  High-Hz Control with 20Hz Full-Query Budget
  control_freq를 40/60/80Hz로 올리되 full Seer query rate는 20Hz 수준으로 유지

GRID:
  Control-Hz / Query-Interval Diagnostic Grid
  QRED20/HZUP20Q 이후 보조 ablation
```

## 추가된 script

```bash
scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_hz_sweep.sh
```

이 script는 공통 runner다. 직접 실행해도 되지만, 혼동을 줄이기 위해 아래 wrapper를 우선 사용한다.

```bash
scripts/LIBERO_LONG/Seer/eval_lrnode_20hz_query_reduction.sh
scripts/LIBERO_LONG/Seer/eval_lrnode_highhz_20hz_query_budget.sh
scripts/LIBERO_LONG/Seer/eval_lrnode_full_grid.sh
```

기본 동작:

1. `runs_lrnode_protocol_20260616/train/_latest/scratch_node.env`에서 최신 scratch_node run을 읽는다.
2. `CKPT_IDS`가 지정되지 않으면, 가장 최근 `eval_node.sh` K=1 결과에서 best checkpoint를 자동 선택한다.
3. 같은 checkpoint로 여러 `(control Hz, K)` pair를 순차 평가한다.
4. 각 `control Hz`는 `OffScreenRenderEnv(control_freq=control_hz)`로 실제 LIBERO env 생성에 들어간다.
5. 20 Hz 기준 episode duration을 유지하기 위해 `libero_eval_max_steps`, env `horizon`, settling step을 기본적으로 `control_hz / 20` 비율로 늘린다.
6. 각 run은 video, eval JSON, latency profile, per-episode CSV를 저장한다.
7. 마지막에 `hz_sweep_summary.csv`와 `hz_sweep_summary.md`를 생성한다.

## 기본 grid

```text
20:1  -> full Seer 20 Hz reference
20:2  -> action 20 Hz, full Seer 10 Hz
20:3  -> action 20 Hz, full Seer 6.67 Hz
20:4  -> action 20 Hz, full Seer 5 Hz
40:1  -> full Seer 40 Hz upper-bound reference
40:2  -> action 40 Hz, full Seer 20 Hz
60:1  -> full Seer 60 Hz upper-bound reference
60:3  -> action 60 Hz, full Seer 20 Hz
80:1  -> full Seer 80 Hz upper-bound reference
80:4  -> action 80 Hz, full Seer 20 Hz
```

핵심 비교는 `40:2`, `60:3`, `80:4`다. 모두 action/control Hz는 높이면서 full Seer query Hz는 20 Hz로 유지한다.

## 실행 command

현재 돌고 있는 `eval_node.sh`, `distill_node.sh`가 끝난 뒤:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_hz_sweep.sh
```

특정 checkpoint만 강제로 쓰려면:

```bash
CKPT_IDS=30 bash scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_hz_sweep.sh
```

grid를 줄여 먼저 smoke/full experiment를 하고 싶으면:

```bash
CKPT_IDS=30 \
HZ_K_PAIRS_STR="20:1 40:1 40:2 60:1 60:3 80:1 80:4" \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_hz_sweep.sh
```

## 저장 위치

기본 result root:

```text
runs_lrnode_protocol_20260616/eval/lrnode_scratch_hz_sweep_${scratch_node_run_name}_${experiment_tag}
```

주요 파일:

```text
experiment_config.env
hz_sweep_summary.csv
hz_sweep_summary.md
_launch_logs/*.log
hz_${H}_K${K}/.../analysis/eval_summary.json
hz_${H}_K${K}/.../analysis/eval_latency_profile.json
hz_${H}_K${K}/.../analysis/eval_episode_metrics.csv
hz_${H}_K${K}/.../eval_videos/
```

각 `eval_summary.json`에는 다음 값이 저장된다.

```text
environment.control_freq
environment.control_hz
environment.eval_max_steps
environment.env_horizon
environment.settle_steps
environment.scale_max_steps_with_hz
```

## 해석 지표

같은 checkpoint와 같은 `control_hz`에서:

```text
performance preservation (%) = SR(H, K) / SR(H, K=1) * 100
```

query 감소율:

```text
full_query_reduction (%) = 1 - full_forward_calls(H, K) / env_steps(H, K)
```

명목 real-time budget:

```text
policy_budget_ms = 1000 / control_hz
policy_latency_over_budget = avg_policy_step_ms / policy_budget_ms
lrnode_over_budget = avg_lrnode_ms / policy_budget_ms
```

episode duration 유지:

```text
actual_eval_max_steps = round(base_eval_max_steps * control_hz / 20)
```

기본값은 `base_eval_max_steps=600`이므로:

```text
20 Hz -> 600 steps
40 Hz -> 1200 steps
60 Hz -> 1800 steps
80 Hz -> 2400 steps
```

즉 이 실험은 단순 Hz 환산이 아니라, LIBERO env의 `control_freq`를 직접 바꾸는 high-control-rate evaluation이다.
