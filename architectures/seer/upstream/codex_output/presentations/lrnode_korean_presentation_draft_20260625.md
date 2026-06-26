# LR-NODE Korean Presentation Draft - 2026-06-25

## Figma Slides deck

- Deck option 1: https://www.figma.com/integrations/claim/mRdvSm7ejqBn7pyvyFqZkM
- Deck option 2: https://www.figma.com/integrations/claim/YeJSTkhdBP45GgozgeiHoJ

Click on any of the previews above to browse presentations or edit in Figma.

## Slide-by-slide key message and speaker notes

### 1. Title

**핵심 메시지:** LR-NODE는 full Seer query를 sparse하게 쓰고 skipped step에서는 latent-space adapter로 action-rate를 유지한다.

**Speaker note:** 이 발표의 핵심은 새 action head를 만드는 것이 아니라, 기존 Seer action head가 이해하는 latent space 안에서 더 싸게 반응하는 경로를 붙이는 것이다.

### 2. Motivation: VLA/Seer full forward 병목

**핵심 메시지:** plain Seer policy latency는 20Hz 명목 budget 50 ms를 넘는다.

**확인 수치:** full forward mean 68.827 ms, policy total mean 88.527 ms, env.step mean 279.390 ms. Baseline ckpt33 policy total mean 89.686 ms, p95 94.405 ms.

**Speaker note:** env.step은 simulator/rendering 병목이라 policy 최적화 claim과 분리한다. LR-NODE가 직접 줄이는 것은 full Seer policy query와 redundant history recomputation이다.

### 3. Problem: 매 step full Seer query의 비용

**핵심 메시지:** control Hz가 올라갈수록 per-step budget은 줄지만, Seer full path는 매 step 긴 history와 two-camera input을 반복 재계산한다.

**Speaker note:** 20Hz는 50 ms, 40Hz는 25 ms, 60Hz는 16.667 ms, 80Hz는 12.5 ms다. high-frequency control에서는 full VLA를 매번 부르는 구조가 더 불리해진다.

### 4. Baseline Seer Pipeline

**핵심 메시지:** baseline은 observation history와 language를 full Seer backbone에 넣고 action latent를 만든 뒤 기존 action head로 action을 출력한다.

**Speaker note:** 여기서 action latent가 중요하다. action decoder/head가 직접 읽는 representation이므로 이 latent를 상태 변수처럼 업데이트하면 action head를 그대로 재사용할 수 있다.

### 5. Key Idea: sparse full Seer + latent update

**핵심 메시지:** full Seer는 key step에서만 실행하고, skipped step에서는 cached latent를 현재 visual/proprio 변화량으로 갱신한다.

**Speaker note:** action은 매 control step마다 내보내지만, expensive full Seer query는 K step마다 한 번으로 줄인다. 이게 QRED20의 핵심이다.

### 6. LR-NODE Module

**핵심 메시지:** LR-NODE는 FastVisualDeltaEncoder, ControlledLatentNODE, existing action head의 세 부분으로 동작한다.

**간단 수식:** `z_hat = z_key + g(u, age) * dt * f_theta(z_key, u, age)`, `a_hat = H_action(z_hat)`.

**Speaker note:** FastVisualDeltaEncoder는 perception replacement가 아니라 cheap change encoder다. ControlledLatentNODE는 latent residual dynamics를 만들고, action head는 기존 Seer의 것을 그대로 쓴다.

### 7. Training Protocols

**핵심 메시지:** scratch/joint 결과와 distill adapter 결과를 섞지 않고, 현재 핵심 비교는 baseline ckpt33 + distill adapter ckpt39 overlay다.

**Speaker note:** distill protocol에서는 baseline Seer/action head를 freeze하고 LR-NODE adapter만 학습한다. 따라서 adapter checkpoint만 단독 로드하면 invalid result가 된다.

### 8. Distill/Load-Parity Check

**핵심 메시지:** baseline ckpt33과 adapter-composed K=1 full-forward가 동일하게 SR 83.0%를 내며 parity를 통과했다.

| Row | SR | full query reduction | policy ms | jerk p95 |
|---|---:|---:|---:|---:|
| baseline ckpt33 K=1 | 83.0% | 0.0% | 75.760 | 0.087384 |
| baseline ckpt33 + adapter ckpt39 K=1 | 83.0% | 0.0% | 78.798 | 0.087384 |

**Load log 근거:** base checkpoint `state_dict_keys=400`, `adapter_only=False`; adapter checkpoint `state_dict_keys=30`, `adapter_only=True`.

**Speaker note:** K=1에서는 LR-NODE skip이 없으므로 adapter가 붙어 있어도 full-forward behavior가 baseline과 같아야 한다. 이 parity가 통과했으므로 K=2/3 결과를 해석할 수 있다.

### 9. Evaluation Protocols

**핵심 메시지:** QRED20은 query reduction, HZUP20Q는 high-control-Hz under fixed full-query budget, Extreme-first-only는 long rollout stress test다.

**Speaker note:** QRED20은 20Hz control_freq를 유지한다. HZUP20Q는 실제 LIBERO control_freq를 올리되 full Seer query Hz는 약 20Hz 수준으로 유지한다.

### 10. Metrics

**핵심 메시지:** SR, full query reduction, effective full query Hz, latency, policy budget, jerk를 분리 정의해야 한다.

| Term | Definition |
|---|---|
| policy budget | `T_budget_ms(H)=1000/H` |
| effective full query Hz | skip mode에서 `H/K`, K=1 full은 `H` |
| full query reduction | `1 - N_full / N` |
| full forward latency | full Seer model forward time only |
| LR-NODE latency | skipped step latent update + existing action head decode |
| policy step latency | preprocessing + policy path + action return |
| env.step latency | LIBERO simulator/rendering, policy latency와 분리 |

**Speaker note:** strict real-time claim은 avg policy step latency가 `1000/H`보다 작을 때만 가능하다.

### 11. Distill QRED20 Results

**핵심 메시지:** latest valid distill QRED20에서 K=2/K=3은 SR을 유지하거나 개선하면서 full query와 policy latency를 줄였다.

| Setting | SR | effective full Hz | LR-NODE Hz | full-query reduction | full ms | LR-NODE ms | policy ms | jerk p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| K=1 full | 83.0% | 20.000 | 0.000 | 0.000% | 63.005 | 0.000 | 73.452 | 0.087384 |
| K=2 skip | 84.0% | 10.014 | 9.986 | 49.932% | 62.468 | 6.690 | 44.985 | 0.183275 |
| K=3 skip | 85.5% | 6.683 | 13.317 | 66.584% | 66.403 | 7.236 | 38.133 | 0.229699 |
| K=4 skip | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 | 확인 필요 |

**Speaker note:** K=4는 지정된 latest path에 `eval_summary.json`이 없고 영상도 126개만 있어 완료 결과로 쓰면 안 된다.

### 12. QRED20 Interpretation

**핵심 메시지:** K=3은 full Seer query를 66.584% 줄이면서 SR 85.5%와 policy 38.133 ms를 기록했다.

**Speaker note:** 20Hz budget은 50 ms다. K=1 policy 73.452 ms는 budget 밖이고, K=2/K=3은 budget 안에 들어온다. 다만 jerk p95는 0.087384에서 0.229699로 증가하므로 smoothness tail은 한계로 남는다.

### 13. HZUP20Q Status

**핵심 메시지:** distill-adapter HZUP20Q 결과 디렉토리는 확인되지 않았다. scratch-node reference만 존재한다.

| Reference row | SR | full Hz | LR-NODE Hz | reduction | policy ms | budget ms | jerk p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| scratch 40Hz K=2 | 84.5% | 20.000 | 20.000 | 49.961% | 45.061 | 25.000 | 0.166171 |
| scratch 60Hz K=3 | 78.5% | 20.000 | 40.000 | 66.639% | 36.392 | 16.667 | 0.310004 |
| scratch 80Hz K=4 | 78.0% | 20.000 | 60.000 | 74.978% | 32.855 | 12.500 | 0.616898 |

**Speaker note:** 이 표는 scratch-node reference이지 current distill adapter evidence가 아니다. 또한 policy/budget이 모두 1보다 커서 strict real-time high-Hz claim은 아직 불가하다.

### 14. Qualitative Evidence

**핵심 메시지:** latest distill QRED20은 success/fail videos를 저장했고, K=2/K=3은 200 episode 기준 summary와 영상 수가 맞는다.

| Row | success videos | fail videos | status |
|---|---:|---:|---|
| baseline K=1 | 166 | 34 | complete |
| adapter K=1 | 166 | 34 | complete |
| K=2 | 168 | 32 | complete |
| K=3 | 171 | 29 | complete |
| K=4 | 118 | 8 | incomplete |

**Speaker note:** 다음 정성 분석은 success/failure split, cache age, last full-forward step, task별 failure, jerk spike를 함께 봐야 한다.

### 15. Limitations and Next Experiments

**핵심 메시지:** training overhead, long rollout drift, smoothness degradation, strict real-time budget이 주요 한계다.

**Next experiments:** distill QRED20 K=4 완료, distill HZUP20Q 실행, Extreme-first-only, budget sweep, first-only shadow.

**Speaker note:** first-only가 실패해도 방법론 실패로 바로 해석하면 안 된다. shadow full-forward로 latent drift와 action drift를 분리 진단해야 한다.

### 16. Takeaway

**핵심 메시지:** LR-NODE는 기존 action head를 유지한 latent-space adapter로, full VLA query를 줄이면서 action-rate를 유지/확장하기 위한 실용적 경로다.

**Speaker note:** 현재 가장 안전한 main claim은 distill QRED20 K=3이다. SR 85.5%, full query reduction 66.584%, policy 38.133 ms로, baseline ckpt33 + adapter ckpt39의 valid overlay 결과다.

## Numeric source list

### Methodology and metric definitions

- `codex_output/methodology/lrnode_theory_code_explanation.md`
- `codex_output/methodology/seer_bottleneck_analysis_20260622.md`
- `codex_output/methodology/lrnode_eval_metrics_definition_20260622.md`

### Prior result summaries used for context

- `codex_output/experiment_results/lrnode_result_analysis_20260623.md`
- `codex_output/experiment_results/lrnode_qred_hzup_result_analysis_20260622.md`
- `codex_output/chatgpt_reports/lrnode_report_for_chatgpt_20260622.md`

### Latest distill parity and QRED20 sources

- Load parity root:
  `runs_lrnode_protocol_20260616/eval/lrnode_distill_loadparity_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_ckpt39_vs_seer_scratch_baseline_ckpt33_distill_loadparity_20260623_171900`
- Load parity summary:
  `runs_lrnode_protocol_20260616/eval/lrnode_distill_loadparity_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_ckpt39_vs_seer_scratch_baseline_ckpt33_distill_loadparity_20260623_171900/experiment_summary.csv`
- Load parity log:
  `runs_lrnode_protocol_20260616/eval/lrnode_distill_loadparity_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_ckpt39_vs_seer_scratch_baseline_ckpt33_distill_loadparity_20260623_171900/ours_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_full_K1_20p00hz/lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_ckpt_39.log`
- Latest distill QRED20 root:
  `runs_lrnode_protocol_20260616/eval/lrnode_distill_qred20_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_ckpt39_vs_seer_scratch_baseline_ckpt33_distill_qred20_20260624_135213`
- Latest distill QRED20 config:
  `runs_lrnode_protocol_20260616/eval/lrnode_distill_qred20_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_ckpt39_vs_seer_scratch_baseline_ckpt33_distill_qred20_20260624_135213/experiment_config.env`
- K=1 summary:
  `runs_lrnode_protocol_20260616/eval/lrnode_distill_qred20_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_ckpt39_vs_seer_scratch_baseline_ckpt33_distill_qred20_20260624_135213/ours_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_full_K1_20p00hz/analysis/eval_summary.json`
- K=2 summary:
  `runs_lrnode_protocol_20260616/eval/lrnode_distill_qred20_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_ckpt39_vs_seer_scratch_baseline_ckpt33_distill_qred20_20260624_135213/ours_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_skip_K2_10p00hz/analysis/eval_summary.json`
- K=3 summary:
  `runs_lrnode_protocol_20260616/eval/lrnode_distill_qred20_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_ckpt39_vs_seer_scratch_baseline_ckpt33_distill_qred20_20260624_135213/ours_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_skip_K3_6p67hz/analysis/eval_summary.json`

### HZUP20Q scratch reference source

- `runs_lrnode_protocol_20260616/eval/lrnode_hzup20q_sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053_hzup20q_ckpt36_gpu0123/hz_sweep_summary.md`

### Extreme evaluation plan

- `codex_output/experiment_plans/lrnode_extreme_eval_plan_20260624.md`

## Values requiring confirmation

- Distill QRED20 K=4 in `distill_qred20_20260624_135213`: `analysis/eval_summary.json` was not present and only 126 videos were found. Treat as incomplete.
- Distill-adapter HZUP20Q: no `*distill*hzup*` eval result directory was found under `runs_lrnode_protocol_20260616/eval`. Treat as not yet available.
- Extreme-first-only, budget sweep, HZUP first-only, first-only shadow: plan exists, but no completed eval result directory was found.
- Strict real-time high-Hz claim: not supported by current measured policy latency values.
