# LR-NODE Experiment Brief for ChatGPT

작성일: 2026-06-15 KST

Archive note: 이 문서에서 언급하는 pre-protocol LR-NODE 결과는 2026-06-16에 아래 archive로 이동했다.

```text
/home/mingyujung/private/seer/seer_node3/archived_experiment_results_20260616/pre_protocol_lrnode
```

## 1. 연구/구현 목표

LR-NODE, Latent-Reactive NODE는 기존 Seer의 action head를 교체하는 모듈이 아니다. 목표는 full Seer forward가 만든 action-relevant latent를 캐시하고, skipped environment step에서는 cheap visual-delta encoder와 controlled latent ODE로 latent만 갱신한 뒤 기존 Seer action decoder/head를 그대로 재사용하는 것이다.

핵심 구조:

- Baseline Seer: 매 step full visual encoder/transformer/action head 실행
- LR-NODE training: policy context `C_t`와 한 step shift된 `C_{t+1}`에서 teacher action-latent를 probe하고, `LRNode(z(C_t), delta(C_t,C_{t+1})) -> z(C_{t+1})`가 되도록 latent/action distillation
- LR-NODE eval: full Seer를 K step마다 한 번만 실행하고, 나머지 step은 LR-NODE latent update 후 기존 action head 재사용
- ODE solver: adaptive solver 없음. MVP는 fixed Euler update
- Visual delta: RAFT/CoTracker 없음. tiny image-difference CNN 사용

현재 대표 method:

- Method tag: `lrnode_student_v2_lw05_aw01_g4`
- Ours checkpoint: `ckpt 35`
- Baseline checkpoint: Seer original `ckpt 37`

주의: 위 checkpoint/results는 pre-protocol archived 결과다. 2026-06-17 이후 코드는 `lrnode_teacher_target_mode=shifted_context`를 기본값으로 사용하므로, 새 방법론 성능은 새 protocol run으로 다시 평가해야 한다.

## 2. 이전 sanity 결과

이전에는 아래 두 eval 폴더를 비교했다.

- Baseline:
  `/home/mingyujung/private/seer/seer_main/eval/sd1_libero_10_100pc_original_settings_20260304`
- Ours:
  `/home/mingyujung/private/seer/seer_node3/scratch_eval_lrnode/sd1_scratch_libero_10_converted_seer_lrnode_student_v2_lw05_aw01_g4_K2`

결과:

| 항목 | Baseline Seer | Ours |
|---|---:|---:|
| 유효 checkpoint | 30-37 | 32-39 |
| 평균 성공률 | 80.19% | 81.06% |
| 최고 성공률 | 86.0% at ckpt 37 | 83.0% at ckpt 35 |
| 공통 checkpoint 평균 | 80.17% at ckpt 32-37 | 81.08% at ckpt 32-37 |
| 공통 checkpoint 차이 | - | +0.92%p |

주의:

- 해당 Ours 폴더명에는 `K2`가 들어가지만, 저장된 JSON 기준으로 `lrnode_eval_skip_full_forward=false`, `lrnode_update_calls=0`이었다.
- 따라서 이 결과는 skip-forward efficiency 결과가 아니라, LR-NODE 모듈이 포함된 모델의 full-forward sanity check로 해석해야 한다.

## 3. 이전 efficiency 결과

그 뒤 baseline ckpt37과 ours ckpt35를 사용해 K별 효율 평가를 수행했다.

조건:

- Baseline full-forward: Seer original ckpt37
- Ours full-forward: LR-NODE ckpt35, skip off
- Ours skip-forward: LR-NODE ckpt35, K=2 and K=4
- 평가 suite: LIBERO-10, 10 tasks x 20 episodes = 200 episodes
- 당시 영상 저장은 OFF였다. 결과 디렉터리는 이후 정리하면서 삭제됨

결과 요약:

| Run | SR | Full Seer call reduction | Effective full-query Hz | Policy latency | 해석 |
|---|---:|---:|---:|---:|---|
| Baseline ckpt37 full | 86.0% | 0.0% | 20 Hz | 80.2 ms | 기준 |
| Ours ckpt35 full | 83.0% | 0.0% | 20 Hz | 82.1 ms | ours checkpoint 자체 sanity |
| Ours ckpt35 K=2 | 85.5% | 49.9% | 10 Hz | 49.9 ms | 성공률 거의 유지, 계산량 감소 |
| Ours ckpt35 K=4 | 86.5% | 74.9% | 5 Hz | 34.3 ms | 가장 큰 절감, action jerk 증가 |

추가 계산:

| 항목 | K=2 | K=4 |
|---|---:|---:|
| Success preservation vs baseline | 99.4% | 100.6% |
| Policy latency reduction vs baseline | 37.7% | 57.2% |
| Full Seer call reduction | 49.9% | 74.9% |

중요 해석:

- baseline과 ours는 서로 다른 checkpoint이므로 "LR-NODE가 baseline보다 성능이 높다"라고 주장하면 안 된다.
- 더 정확한 주장은 "LR-NODE ckpt35에서 full Seer query를 절반 또는 1/4로 줄여도 LIBERO-10 성공률이 baseline 수준으로 유지됐고, policy 계산 latency가 크게 줄었다"이다.
- K=4는 성공률은 좋았지만 action jerk와 gripper switch rate가 증가했다.
- 발표/논문 메인 숫자로는 K=2가 가장 방어적이다.

## 4. 이전 task별 관찰

Ours full이 baseline보다 낮아진 주요 task:

| Task | Baseline | Ours full |
|---|---:|---:|
| task 6 | 90% | 70% |
| task 8 | 75% | 45% |
| task 9 | 80% | 60% |

K=2에서 일부 task가 회복됐다.

- task 8: ours full 45% -> K=2 75%
- 전체 SR: ours full 83.0% -> K=2 85.5%

반대로 task 2는 K=2에서 하락했다.

- task 2: ours full 100% -> K=2 85%

## 5. 정리하면서 삭제한 예전 결과

아래 결과 폴더들은 영상이 없거나 ckpt-specific 임시 sweep 결과였기 때문에 삭제했다.

```text
scratch_eval_lrnode/lrnode_efficiency_ckpt35_vs_baseline37_20260612_153637
scratch_eval_lrnode/lrnode_efficiency_ckpt35_vs_baseline37_20260612_153652
scratch_eval_lrnode/lrnode_k_sweep_extra_ckpt35_vs_baseline37_video_20260614_235516
```

현재 비교 실행 스크립트는 하나만 남겼다.

```text
scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh
```

## 6. 현재 진행 중인 실험

현재는 영상 저장을 기본으로 켠 canonical comparison script를 실행 중이다.

실행 명령:

```bash
bash scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh
```

현재 config:

```text
Baseline:
  name: seer_original
  run_name: sd1_libero_10_100pc_original_settings
  ckpt: /home/mingyujung/private/seer/seer_main/checkpoints/sd1_libero_10_100pc_original_settings/37.pth

Ours:
  method: lrnode_student_v2_lw05_aw01_g4
  run_name: sd1_scratch_libero_10_converted_seer_lrnode_student_v2_lw05_aw01_g4
  ckpt: /home/mingyujung/private/seer/seer_node3/scratch_checkpoints_lrnode/sd1_scratch_libero_10_converted_seer_lrnode_student_v2_lw05_aw01_g4/35.pth

K sweep:
  baseline full: K=1
  ours full: K=1
  ours skip: K=2,3,4,5,6,8

Video:
  SAVE_VIDEO=1
  SAVE_VIDEO_SUCC=1
  SAVE_VIDEO_FAIL=1
  SAVE_VIDEO_ALL_RANKS=1
  VIDEO_FPS=20
  VIDEO_STRIDE=1
```

현재 result root:

```text
/home/mingyujung/private/seer/seer_node3/scratch_eval_lrnode/lrnode_compare_lrnode_student_v2_lw05_aw01_g4_ckpt35_vs_seer_original_ckpt37_20260615_001039
```

현재 확인 시점 상태:

- `eval_lrnode_compare.sh` 프로세스 실행 중
- 완료된 run: baseline full K=1, ours full K=1, ours skip K=2, ours skip K=3
- 진행 중인 run: ours skip K=4
- 아직 대기 중인 run: ours skip K=5, K=6, K=8
- 영상 저장: 완료된 run은 각각 200개 mp4 저장 완료
- K=4는 `eval_summary.json` 생성 전이며, 확인 시점 기준 180/200개 영상 저장됨

현재까지 확정된 결과:

| Run | SR | Full Seer call reduction | Effective full-query Hz | Policy latency | Videos |
|---|---:|---:|---:|---:|---:|
| Baseline Seer ckpt37 full K=1 | 86.0% | 0.0% | 20.00 Hz | 79.2 ms | 200 |
| Ours ckpt35 full K=1 | 83.0% | 0.0% | 20.00 Hz | 79.2 ms | 200 |
| Ours ckpt35 skip K=2 | 85.5% | 49.9% | 10.00 Hz | 49.7 ms | 200 |
| Ours ckpt35 skip K=3 | 88.0% | 66.6% | 6.67 Hz | 39.4 ms | 200 |

현재까지의 핵심 관찰:

- K=2는 baseline 대비 `99.4%` success preservation이고, full Seer 호출을 `49.9%` 줄였다.
- K=3는 현재까지 가장 좋은 수치로, baseline보다 +2.0%p 높은 `88.0%` SR이며 full Seer 호출을 `66.6%` 줄였다.
- 다만 K=3는 action smoothness 지표가 더 나빠진다. `action_jerk_l2_p95`가 baseline `0.0707`, K=2 `0.1294`, K=3 `0.1843`으로 증가했다.
- 따라서 현재까지는 K=3가 성능/효율 숫자는 가장 좋지만, 발표에서 안정적인 대표 결과로는 K=2와 K=3를 함께 보여주고 smoothness trade-off를 같이 설명하는 것이 안전하다.

실험이 끝나면 생성될 주요 파일:

```text
experiment_config.env
experiment_summary.csv
*/analysis/eval_summary.json
*/analysis/eval_episode_metrics.csv
*/analysis/eval_latency_profile.json
*/eval_videos/.../success/*.mp4
*/eval_videos/.../fail/*.mp4
```

## 7. 다음 분석에서 봐야 할 것

최종 결과가 나오면 아래를 비교하면 된다.

1. Success rate vs K
2. Full Seer call reduction vs K
3. Policy latency vs K
4. Action smoothness degradation vs K
5. Failure videos by task and K
6. K=2가 가장 안정적인 메인 결과인지, K=4 이상은 aggressive efficiency setting인지 판단

예상되는 발표 방향:

- 메인 결과는 K=2를 우선 사용
- K=4,5,6,8은 efficiency frontier/ablation으로 사용
- latency 숫자는 video-on run보다 이전 video-off run이 더 깨끗하지만, qualitative failure 분석은 현재 video-on run을 사용
