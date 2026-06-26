# LR-NODE 결과 분석 - 2026-06-23

## 결론 요약

현재 성능 claim에 사용할 수 있는 결과는 `scratch_node ckpt36` 기반의 QRED20/HZUP20Q 결과다.

반대로 `distill_node ckpt39` sanity eval 결과는 성능 결과로 쓰면 안 된다. 원인은 확인됐다. `distill_node` checkpoint는 frozen Seer/action head를 저장하지 않는 adapter-only checkpoint인데, 현재 distill eval은 baseline checkpoint를 먼저 로드하지 않고 adapter checkpoint만 `--resume_from_checkpoint`로 로드했다. 따라서 K=1 full-forward도 baseline Seer가 아니라 랜덤 초기화된 Seer 본체 + LR-NODE adapter 상태로 평가되었다.

학습시간 overhead와 self-KD 성격/weakness는 `codex_output/methodology/lrnode_training_time_weakness_analysis_20260623.md`에 별도로 정리했다.

## 1. QRED20 결과

QRED20은 LIBERO `control_freq=20`을 유지하고, action은 매 env step마다 내보내되 full Seer query만 `K` step마다 호출하는 실험이다.

실험 root:

```text
runs_lrnode_protocol_20260616/eval/lrnode_qred20_sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053_qred20_ckpt36_gpu0123
```

사용 checkpoint:

```text
runs_lrnode_protocol_20260616/train/scratch_node/sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053/36.pth
```

| Setting | SR | Full Seer Hz | LR-NODE Hz | Full-query reduction | Policy latency | LR-NODE latency | Jerk p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20Hz K=1 full | 82.5% | 20.00 | 0.00 | 0.00% | 70.388 ms | 0.000 ms | 0.079720 |
| 20Hz K=2 skip | 79.0% | 10.00 | 10.00 | 49.94% | 47.900 ms | 7.205 ms | 0.216848 |
| 20Hz K=3 skip | 82.5% | 6.67 | 13.33 | 66.58% | 36.843 ms | 6.965 ms | 0.259359 |
| 20Hz K=4 skip | 86.5% | 5.00 | 15.00 | 74.89% | 33.639 ms | 7.237 ms | 0.307808 |

### QRED20 해석

가장 중요한 행은 `K=3`이다.

- K=1과 같은 SR 82.5%를 유지했다.
- full Seer query를 66.58% 줄였다.
- 평균 policy latency를 70.388 ms에서 36.843 ms로 줄였다.
- 20Hz budget은 `1000 / 20 = 50 ms`이므로, K=1 full은 budget을 넘고 K=3 skip은 budget 안에 들어온다.

`K=4`는 SR만 보면 가장 좋지만, jerk p95가 K=1 대비 약 3.86배다.

```text
0.307808 / 0.079720 = 3.86
```

따라서 발표/논문 main row는 `K=3`, aggressive row는 `K=4`로 두는 것이 맞다. `K=2`는 더 보수적인 ablation으로 사용하면 된다.

## 2. HZUP20Q 결과

HZUP20Q는 LIBERO `control_freq` 자체를 올리면서, expensive full Seer query rate를 약 20Hz로 유지하는 실험이다.

실험 root:

```text
runs_lrnode_protocol_20260616/eval/lrnode_hzup20q_sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053_hzup20q_ckpt36_gpu0123
```

현재 완료된 행:

| Setting | SR | Full Seer Hz | LR-NODE Hz | Full-query reduction | Policy latency | LR-NODE latency | Jerk p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20Hz K=1 full | 82.5% | 20.00 | 0.00 | 0.00% | 82.818 ms | 0.000 ms | 0.079720 |
| 40Hz K=1 full | 83.0% | 40.00 | 0.00 | 0.00% | 75.570 ms | 0.000 ms | 0.037257 |
| 40Hz K=2 skip | 84.5% | 20.00 | 20.00 | 49.96% | 45.061 ms | 6.594 ms | 0.166171 |
| 60Hz K=1 full | 73.5% | 60.00 | 0.00 | 0.00% | 76.322 ms | 0.000 ms | 0.030780 |
| 60Hz K=3 skip | 78.5% | 20.00 | 40.00 | 66.64% | 36.392 ms | 6.815 ms | 0.310004 |
| 80Hz K=1 full | 77.5% | 80.00 | 0.00 | 0.00% | 75.659 ms | 0.000 ms | 0.026549 |
| 80Hz K=4 skip | 78.0% | 20.00 | 60.00 | 74.98% | 32.855 ms | 7.512 ms | 0.616898 |

### HZUP20Q 해석

`40Hz K=2`가 현재 가장 좋은 high-Hz 결과다.

- full Seer query rate를 20Hz로 유지했다.
- LR-NODE가 나머지 20Hz action update를 담당했다.
- SR은 84.5%로 20Hz K=1의 82.5%보다 높다.
- 같은 40Hz control에서 full Seer를 매 step 호출한 `40Hz K=1 full`의 83.0%보다도 높다.
- 하지만 평균 policy latency 45.061 ms는 strict real-time 40Hz budget인 25 ms를 넘는다.

여기서 budget은 다음처럼 계산된다.

```text
budget_ms = 1000 / control_hz
40Hz budget = 1000 / 40 = 25 ms
60Hz budget = 1000 / 60 = 16.667 ms
80Hz budget = 1000 / 80 = 12.5 ms
```

즉 현재 HZUP20Q의 올바른 claim은 다음이다.

```text
LR-NODE는 control_freq를 40Hz로 올려도 full Seer query를 20Hz로 제한한 채 성공률을 유지하거나 개선한다.
```

하지만 다음 claim은 아직 하면 안 된다.

```text
현재 구현이 40Hz real-time latency budget을 만족한다.
```

`60Hz K=3`, `80Hz K=4`는 full Seer query rate를 20Hz로 유지한다는 구조적 목적은 달성했지만, SR과 smoothness가 악화된다.

- 60Hz K=3: SR 78.5%, jerk p95 0.310004
- 80Hz K=4: SR 78.0%, jerk p95 0.616898

다만 같은 high-Hz control의 expensive full-Seer upper bound와 비교하면 LR-NODE skip이 오히려 더 높거나 비슷하다.

- 60Hz: K=1 full 73.5% vs K=3 skip 78.5%
- 80Hz: K=1 full 77.5% vs K=4 skip 78.0%

따라서 high-Hz 실험은 두 가지를 동시에 보여준다.

1. Full Seer를 high-Hz로 무작정 더 자주 호출한다고 성공률이 단조 증가하지 않는다.
2. LR-NODE는 full Seer query를 20Hz로 제한하면서 high-Hz control을 구성할 수 있다.

특히 80Hz K=4의 jerk p95는 20Hz K=1 대비 약 7.74배다.

```text
0.616898 / 0.079720 = 7.74
```

따라서 high-Hz 실험의 현재 결론은 `40Hz K=2는 긍정적`, `60/80Hz는 query-efficiency 구조는 유효하지만 smoothness 안정성 개선 필요`다.

## 3. Distill sanity 결과는 무효

실험 root:

```text
runs_lrnode_protocol_20260616/eval/lrnode_distill_compare_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_ckpt39_vs_seer_scratch_baseline_ckpt33_distill_ckpt39_sanity_20260622
```

현재 나온 수치:

| Setting | SR | Full-query reduction | Policy latency | Jerk p95 |
|---|---:|---:|---:|---:|
| distill ckpt39 K=1 full | 0.0% | 0.00% | 69.006 ms | 0.000791 |
| distill ckpt39 K=2 skip | 0.0% | 50.00% | 58.116 ms | 0.418282 |
| distill ckpt39 K=3 skip | 0.0% | 66.67% | 39.167 ms | 0.233674 |

이 수치는 방법론 실패가 아니라 eval 구성 오류다.

확인된 근거:

- `utils/train_utils.py`의 `get_checkpoint()`는 `requires_grad=False`인 파라미터를 checkpoint에서 제거한다.
- distill protocol은 Seer/action head를 freeze하고 LR-NODE만 trainable로 둔다.
- 따라서 distill checkpoint 크기는 약 5.736 MB이고, key 수는 30개뿐이다.
- baseline ckpt33은 약 809.419 MB이고, key 수는 400개다.
- distill ckpt39에는 `module.transformer_backbone_position_embedding`, `module.action_pred_token`, `module.action_decoder.*`가 없다.
- distill eval의 `args_snapshot.json`에는 `finetune_from_pretrained_ckpt=None`이고, `resume_from_checkpoint`만 distill ckpt39로 지정되어 있다.

따라서 distill eval은 다음 순서로 다시 구현/실행해야 한다.

```text
1. baseline ckpt33을 full Seer weight로 로드
2. distill adapter ckpt39의 LR-NODE weight를 추가 로드
3. K=1 full-forward가 baseline ckpt33 SR 83.0%와 일치하는지 먼저 검증
4. 그 다음 K=2/3/4 skip 평가
```

현재 distill 결과는 성능 표, 발표, 논문 claim에 넣으면 안 된다.

## 3.1 Distill 문제의 정확한 성격

현재 문제는 distill 학습 자체가 실패했는지 여부를 아직 판단할 수 없는 상태라는 점이다. 지금 확인된 것은 eval loader가 adapter-only checkpoint를 단독 모델 checkpoint처럼 사용했다는 것이다.

현재 checkpoint 구조:

| Checkpoint | Size | State dict keys | 포함 내용 |
|---|---:|---:|---|
| baseline ckpt33 | 809.419 MB | 400 | Seer backbone, action head, decoder, optimizer 등 full trainable checkpoint |
| distill ckpt39 | 5.736 MB | 30 | `lrnode_delta_encoder`, `lrnode_dynamics` adapter-only |

따라서 distill 평가의 필수 조건은 다음 parity check다.

```text
baseline ckpt33 full-forward SR == baseline ckpt33 + distill adapter ckpt39 K=1 full-forward SR
```

기대값은 baseline ckpt33의 83.0%다. 이 parity가 맞기 전에는 distill K=2/3/4 결과를 해석하면 안 된다.

## 6. 지금 돌려야 하는 실험 우선순위

2026-06-23 수정 사항:

- `eval_libero.py`가 eval 시에도 `--finetune_from_pretrained_ckpt`를 base checkpoint로 먼저 로드한다.
- 그 다음 `--resume_from_checkpoint`를 로드한다.
- adapter-only LR-NODE checkpoint를 base checkpoint 없이 평가하려고 하면 즉시 에러를 낸다.
- `eval_lrnode_distill_compare.sh`는 distill adapter 평가에서 자동으로 `LRNODE_EVAL_BASE_CKPT=${BASELINE_CKPT}`를 넘긴다.
- 전용 실행 스크립트를 추가했다.
  - `scripts/LIBERO_LONG/Seer/eval_lrnode_distill_loadparity.sh`
  - `scripts/LIBERO_LONG/Seer/eval_lrnode_distill_qred20.sh`
  - `scripts/LIBERO_LONG/Seer/eval_lrnode_distill_hzup20q.sh`

검증:

```text
python -m py_compile eval_libero.py
bash -n scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh \
        scripts/LIBERO_LONG/Seer/eval_lrnode_distill_compare.sh \
        scripts/LIBERO_LONG/Seer/eval_lrnode_distill_loadparity.sh \
        scripts/LIBERO_LONG/Seer/eval_lrnode_distill_qred20.sh \
        scripts/LIBERO_LONG/Seer/eval_lrnode_distill_hzup20q.sh
```

둘 다 통과했다.

### 1순위: fixed distill sanity

목적:

```text
baseline ckpt33 + LR-NODE adapter ckpt39를 올바르게 합성 로드했을 때 K=1 full-forward가 baseline ckpt33과 동일한지 확인
```

필수 조건:

- eval에서 baseline ckpt33을 먼저 로드한다.
- 그 다음 distill adapter ckpt39의 LR-NODE key만 추가 로드한다.
- `K=1`, `lrnode_eval_skip_full_forward=0`으로 평가한다.
- SR이 baseline ckpt33 83.0%와 같거나 통계 오차 수준으로 가까워야 한다.

이 실험이 실패하면 distill adapter 결과는 모두 보류한다.

실행 명령:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_loadparity.sh
```

완료 후 확인할 것:

```text
baseline full K=1 SR ~= 83.0%
adapter-composed full K=1 SR ~= 83.0%
```

로그에 다음 형태가 보여야 한다.

```text
[CKPT LOAD:base] path=.../scratch/.../33.pth
[CKPT LOAD:base] adapter_only=False
[CKPT LOAD:resume_or_adapter] path=.../distill_node/.../39.pth
[CKPT LOAD:resume_or_adapter] adapter_only=True
```

이 로그가 없으면 합성 로드가 안 된 것이다.

### 2순위: fixed distill QRED20

1순위 parity가 통과하면 같은 합성 모델로 QRED20을 돌린다.

권장 rows:

```text
20Hz K=1 full
20Hz K=2 skip
20Hz K=3 skip
20Hz K=4 skip
```

목적:

```text
기존 Seer baseline weight를 그대로 둔 상태에서 LR-NODE adapter만 붙여 full Seer query를 줄일 수 있는지 확인
```

이 실험이 현재 연구 claim에서 가장 공정한 adapter/distill 결과다.

실행 명령:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_qred20.sh
```

이 스크립트는 다음을 한 번에 실행한다.

```text
baseline full K=1
adapter-composed full K=1
adapter-composed skip K=2
adapter-composed skip K=3
adapter-composed skip K=4
```

### 3순위: fixed distill HZUP20Q

fixed distill QRED20이 통과하면 high-Hz 조건을 돌린다.

권장 rows:

```text
20Hz K=1 full
40Hz K=1 full
40Hz K=2 skip
60Hz K=1 full
60Hz K=3 skip
80Hz K=1 full
80Hz K=4 skip
```

목적:

```text
baseline Seer를 20Hz query budget으로 묶은 채 control_freq만 40/60/80Hz로 올릴 수 있는지 확인
```

실행 명령:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_hzup20q.sh
```

기본 rows:

```text
20Hz K=1
40Hz K=1
40Hz K=2
60Hz K=1
60Hz K=3
80Hz K=1
80Hz K=4
```

결과 root 아래에 `hzup20q_summary.csv`가 생성된다.

### 4순위: scratch_node shadow-full diagnostic

이미 유효한 scratch_node ckpt36에 대해 `LRNODE_EVAL_SHADOW_FULL_FORWARD=1` diagnostic을 별도로 돌린다.

권장 rows:

```text
20Hz K=2
20Hz K=3
20Hz K=4
40Hz K=2
60Hz K=3
80Hz K=4
```

목적:

```text
LR-NODE skipped action이 같은 시점 full Seer action과 얼마나 다른지 직접 측정
```

이건 latency claim용이 아니라 error analysis용이다. shadow full-forward는 추가 full forward를 실행하므로 policy latency 표에는 사용하지 않는다.

실행 예시:

```bash
LRNODE_EVAL_SHADOW_FULL_FORWARD=1 \
HZ_K_PAIRS_STR="20:2 20:3 20:4 40:2 60:3 80:4" \
EXPERIMENT_NAME="scratch_shadow_diag" \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_scratch_hz_sweep.sh
```

## 4. 기존 Seer 병목과 연결

`codex_output/methodology/seer_bottleneck_analysis_20260622.md` 기준으로 기존 Seer의 실측 병목은 policy inference다.

plain baseline ckpt30-39 평균:

| Metric | Value |
|---|---:|
| Full forward mean | 68.827 ms |
| Policy total mean | 88.527 ms |
| 20Hz control budget | 50.000 ms |
| Full forward / budget | 1.377x |
| Policy total / budget | 1.771x |

이 병목 때문에 LR-NODE의 실험 설계가 성립한다.

- full Seer는 매 step 호출하기에 비싸다.
- LR-NODE update는 약 6.6-7.5 ms 수준이다.
- 따라서 full Seer를 매 `K` step마다만 호출하고 skipped step은 LR-NODE로 latent/action을 갱신하면 expensive query를 줄일 수 있다.

현재 결과는 이 방향을 지지한다. 다만 high-Hz real-time latency budget을 만족하려면 full-forward amortization뿐 아니라 preprocessing, action decode, CPU/GPU transfer, simulator loop까지 추가 최적화가 필요하다.

## 5. 현재 사용 가능한 claim

가장 방어적인 claim:

```text
At 20Hz control, LR-NODE reduces full Seer queries by 66.6% while preserving the K=1 success rate on LIBERO-10.
```

한국어:

```text
20Hz 제어 조건에서 LR-NODE는 LIBERO-10 성공률을 유지하면서 full Seer query를 66.6% 줄였다.
```

high-Hz claim:

```text
At 40Hz control, LR-NODE keeps expensive full Seer queries at 20Hz and preserves/improves task success, but the current implementation does not yet satisfy strict real-time 40Hz policy latency.
```

한국어:

```text
40Hz 제어 조건에서 LR-NODE는 expensive full Seer query를 20Hz로 유지하면서 성공률을 보존 또는 개선했다. 단, 현재 구현의 평균 policy latency는 strict real-time 40Hz budget을 아직 만족하지 못한다.
```

사용하면 안 되는 claim:

```text
distill_node ckpt39 achieved 0% SR, therefore adapter distillation failed.
```

이 문장은 틀렸다. 현재 distill eval은 baseline full checkpoint를 함께 로드하지 않은 구성 오류다.
