# LR-NODE Extreme Evaluation Plan - 2026-06-24

이 문서는 distill LR-NODE adapter 실험 이후에 돌릴 극단 평가를 정리한다. 기본 checkpoint 조합은 다음이다.

- Baseline full Seer: `BASELINE_CKPT_ID=33`
- Distill LR-NODE adapter: `OURS_CKPT_ID=39`
- GPU: `CUDA_VISIBLE_DEVICES=4,5,6,7`
- Adapter eval 방식: baseline ckpt33을 먼저 로드하고 adapter ckpt39를 overlay한다.

## 전제

기존 `QRED20`과 `HZUP20Q`는 `lrnode_eval_refresh_policy=periodic`이다.

- `periodic`: 기존 K 기반 refresh. `t % K == 0`이면 full Seer, 나머지는 LR-NODE.
- `first_only`: episode 첫 policy step에서만 full Seer, 이후 모든 step은 LR-NODE.
- `fixed_budget`: episode당 full Seer 호출 예산 B개를 episode horizon에 균등하게 배치하고, 나머지는 LR-NODE.

`first_only`와 `fixed_budget`은 K-sweep과 다른 실험이다. K가 아니라 episode-level full-query budget을 보는 실험이다.

## 1. DISTILL-EXTREME-FIRSTONLY

목적: 가장 극단적인 query reduction에서 LR-NODE rollout이 얼마나 버티는지 확인한다.

정의:

- 각 episode 첫 policy step: full Seer forward 1회.
- 이후 episode 종료까지: full Seer refresh 없음.
- 모든 이후 action은 cached latent를 LR-NODE로 업데이트한 뒤 기존 action head로 decode한다.

실행:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT_ID=33 \
OURS_CKPT_ID=39 \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_extreme_firstonly.sh
```

해석:

- 성공률이 유지되면 LR-NODE가 장기 latent rollout을 상당히 잘 한다는 강한 결과다.
- 성공률이 크게 떨어지면 LR-NODE는 장기 open-loop latent rollout에는 약하고, 주기적 full refresh가 필요하다는 뜻이다.
- 핵심 metric은 `success_rate`, `full_query_reduction_ratio`, `effective_full_query_hz`, `action_jerk_l2_p95`, `cache_age_at_failure`, `last_full_forward_step`이다.

## 2. DISTILL-EXTREME-BUDGET-SWEEP

목적: episode당 full Seer refresh가 몇 번 필요하면 성능이 회복되는지 찾는다.

정의:

- `B=1`: first-only와 거의 같은 극단 조건.
- `B=2,4,8`: episode horizon 안에 full Seer refresh를 B회 균등 배치.
- 각 refresh 사이 step은 LR-NODE가 latent를 업데이트한다.

실행:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT_ID=33 \
OURS_CKPT_ID=39 \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_extreme_budget_sweep.sh
```

선택적으로 budget을 바꾸려면:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT_ID=33 \
OURS_CKPT_ID=39 \
FULL_BUDGETS_STR="1 2 4 8 16" \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_extreme_budget_sweep.sh
```

해석:

- `B=1`은 장기 rollout 한계.
- `B=2/4/8`에서 성공률이 회복되면, 방법론의 핵심 주장은 “full VLA 호출을 매 step 하지 않고 sparse refresh만으로 성능을 유지한다”가 된다.
- budget sweep은 K-sweep보다 episode-level query budget을 직접 보여주므로 발표/논문에서 이해시키기 쉽다.

## 3. DISTILL-EXTREME-HZUP-FIRSTONLY

목적: 실제 LIBERO `control_freq`를 올린 상태에서 full Seer를 episode 첫 step에만 쓰는 극단 조건을 본다.

정의:

- `control_freq`: 기본 `20 40 60 80`.
- 각 Hz에서 episode 첫 policy step만 full Seer.
- 이후 모든 high-rate action은 LR-NODE가 만든 latent에서 decode한다.

실행:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT_ID=33 \
OURS_CKPT_ID=39 \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_extreme_hzup_firstonly.sh
```

선택적으로 Hz를 바꾸려면:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT_ID=33 \
OURS_CKPT_ID=39 \
HZS_STR="20 30 40 50 60" \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_extreme_hzup_firstonly.sh
```

해석:

- 이 실험은 매우 공격적인 조건이다.
- 성공률이 낮아도 방법론이 틀렸다는 결론은 아니다. 오히려 “high-Hz에서는 sparse full refresh가 어느 정도 필요하다”는 근거가 된다.
- `policy_latency_over_budget`는 실제 policy 계산 시간이 해당 Hz의 per-step budget을 넘는지 보여준다.

## 4. DISTILL-EXTREME-FIRSTONLY-SHADOW

목적: first-only가 실패할 경우, failure가 latent drift 때문인지 action drift 때문인지 진단한다.

정의:

- 실제 실행 action은 first-only LR-NODE와 동일하다.
- 단, skipped step마다 full Seer를 shadow로 추가 실행해서 `z_full`과 `a_full`을 기록한다.
- 따라서 이 실험의 latency는 deployment latency가 아니다.

실행:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
BASELINE_CKPT_ID=33 \
OURS_CKPT_ID=39 \
bash scripts/LIBERO_LONG/Seer/eval_lrnode_distill_extreme_firstonly_shadow.sh
```

해석:

- `shadow_latent_mse`가 cache age와 함께 급증하면 latent rollout drift가 문제다.
- `shadow_action_l1`이 `shadow_action_hold_l1`보다 낮으면 LR-NODE가 단순 hold보다 낫다.
- `pred_vs_hold_improvement > 0`이면 NODE update가 action repeat/latent hold보다 유리하다.

## 권장 실행 순서

현재 진행 중인 distill `QRED20`, `HZUP20Q`가 끝난 뒤 아래 순서로 실행한다.

1. `eval_lrnode_distill_extreme_firstonly.sh`
2. first-only 성공률이 낮으면 `eval_lrnode_distill_extreme_firstonly_shadow.sh`
3. `eval_lrnode_distill_extreme_budget_sweep.sh`
4. `eval_lrnode_distill_extreme_hzup_firstonly.sh`

보고용 핵심 비교는 다음 두 축으로 정리한다.

- QRED 계열: 같은 20Hz에서 full Seer query를 얼마나 줄여도 되는가.
- HZUP 계열: control Hz를 올렸을 때 full Seer query budget을 얼마나 낮게 유지할 수 있는가.
