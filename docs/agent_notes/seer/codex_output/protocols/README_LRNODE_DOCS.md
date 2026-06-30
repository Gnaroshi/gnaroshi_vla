# LR-NODE 문서 색인

작성일: 2026-06-20 KST

이 문서는 `codex_output/` 아래 LR-NODE 관련 문서의 현재 신뢰도와 사용 순서를 정리한다.
가장 최신 구현/논점은 `lrnode_current_state_handoff.md`를 기준으로 읽는다.

## 0. 최신 한 장 요약

### 현재 구현 handoff

```text
codex_output/lrnode_current_state_handoff.md
```

상태: **최신 / 우선 참고**

포함 내용:

- 현재 Seer + LR-NODE 구현 구조
- `action_latent_full: [B, S, action_pred_steps, D]`
- 현재 `shifted_context` 구현이 왜 학습시간을 2.x까지 늘릴 수 있는지
- `adjacent_sequence`, last-pair adjacent, cache-based shifted-context의 차이
- 현재 script/default 상태
- 다음 구현 우선순위

관련 이미지:

```text
codex_output/figures/lrnode_current_state/lrnode_current_state_overview.png
```

상태: **최신 / raster PNG**

확인:

```text
PNG image data, 1672 x 941, 8-bit/color RGB, non-interlaced
```

### 현재 baseline eval 결과

```text
codex_output/baseline_eval_analysis_20260620.md
```

상태: **최신 / 새 protocol baseline 기준점**

핵심 결과:

- baseline run: `sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040`
- eval root: `runs_lrnode_protocol_20260616/eval/baseline_sweep_sd1_scratch_baseline_seer_scratch_baseline_v1_20260616_141040_20260618_132356`
- best SR: ckpt 33과 ckpt 39가 83.0% 동률
- primary baseline ckpt 권장: ckpt 33
- ckpt당 200 videos 저장 확인
- baseline full-query 감소율: 0%
- LR-NODE disabled 확인

### scratch_node 완료 및 다음 실험

```text
codex_output/scratch_node_completion_and_next_steps_20260620.md
```

상태: **최신 / 실행 판단 문서**

핵심:

- `scratch_node.sh` run `sd1_scratch_node_ts_lrnode_scratch_ts_v1_lw05_aw01_g4_20260619_113053` 완료 확인
- ckpt 26-39 존재 확인
- `eval_node.sh` 실행 가능 여부 확인
- `run_lrnode_post_scratch_all.sh` 추가: scratch_node K=1 eval, best ckpt K sweep, baseline ckpt33 distill을 한 번에 실행
- master port 분리: eval K=1 `12442`, eval K sweep `12443`, distill `12423`, compare `12452`
- 우선 K=1 전체 ckpt 평가 후 selected ckpt만 K sweep 권장
- `distill_node.sh` 기본 baseline ckpt를 33으로 수정

### 현재 running 결과 분석

```text
codex_output/lrnode_running_results_analysis_20260620.md
```

상태: **최신 / 진행 중 결과 분석**

핵심:

- 현재 완료된 scratch_node eval은 ckpt30 K=1 하나
- scratch_node ckpt30 K=1 SR: 80.5%
- baseline ckpt30 SR 78.0% 대비 +2.5%p
- baseline best ckpt33/39 SR 83.0% 대비 -2.5%p
- distill은 baseline ckpt33에서 adapter-only로 정상 학습 중
- distill trainable parameter는 LR-NODE 470,146개뿐이며 Seer/action head는 frozen
- distill checkpoint는 LR-NODE key 30개만 저장되는 adapter-only checkpoint
- distill eval 전 baseline checkpoint와 adapter checkpoint를 함께 로드하는 경로가 필요

## 1. 구현 검증 문서

### Parity 감사

```text
codex_output/lrnode_parity_audit_20260618.md
```

상태: **최신 검증 결과 포함**

포함 내용:

- LR-NODE module 추가가 common Seer/action head 초기화를 깨지 않는지
- detached teacher-student에서 LR-NODE loss가 common gradient/update를 바꾸지 않는지
- shifted-context teacher forward의 RNG/dropout 문제
- `_preserve_torch_rng()` 수정 후 loss/gradient/update parity
- invalid `20260618_143600` scratch_node run 폐기 근거

재현 script:

```text
scripts/debug/check_lrnode_parity.py
```

핵심 결과:

```text
init common diff = 0
base loss diff = 0.0
common grad diff = 0
common AdamW update diff = 0
eval full-forward action/latent diff = 0.0
```

## 2. 방법론/코드 설명 문서

### 코드 수준 이론 설명

```text
codex_output/lrnode_theory_code_explanation.md
```

상태: **유용하지만 일부 섹션은 2026-06-19 이전 논의 기준**

주의:

- gate, detach, action head freeze, LR-NODE loss 구성 설명은 여전히 유효하다.
- scratch/distill/joint 개념 설명도 대부분 유효하다.
- 다만 `shifted_context`를 main target으로 강하게 고정한 부분은 최신 논의와 다르게 읽어야 한다.
- 최신 결론은 `current shifted_context = 정확하지만 느린 2-forward 구현`, `adjacent/last-pair = dataset 변경 없는 one-forward 대안`, `cache-based shifted_context = ideal one-forward target`이다.

### Policy-context distillation

```text
codex_output/lrnode_policy_context_distillation.md
```

상태: **shifted-context teacher 정의 참고용 / 최신 결론과 함께 읽을 것**

주의:

- 이 문서는 `shifted_context` teacher target을 정의하기 위해 작성됐다.
- 현재는 `shifted_context` 자체가 틀린 것이 아니라, 현재 구현이 extra teacher full forward를 사용해서 느리다는 점이 추가로 확인됐다.
- one-forward training 논의는 `lrnode_current_state_handoff.md`가 더 최신이다.

## 3. Protocol/script 정의 문서

### 학습 protocol 정의

```text
codex_output/lrnode_training_protocol_definitions.md
```

상태: **대체로 유효 / target mode 해석만 최신 handoff와 함께 볼 것**

핵심:

- `scratch.sh`: plain Seer scratch baseline
- `scratch_node.sh`: scratch + LR-NODE detached teacher-student
- `distill_node.sh`: frozen baseline checkpoint에서 LR-NODE만 학습
- `scratch_node_joint.sh`: LR-NODE loss가 common Seer/action head에도 영향을 줄 수 있는 coupled ablation

주의:

- 현재 script default는 아직 `shifted_context`다.
- efficiency main 실험으로는 `adjacent_sequence` 또는 last-pair adjacent 변경을 검토해야 한다.

### Protocol script

```text
codex_output/lrnode_protocol_scripts.md
```

상태: **script 목적 참고용**

주의:

- 일부 예시는 `shifted_context`를 main처럼 적고 있다.
- 최신 실험 설계에서는 `shifted_context`를 slow ablation으로 분리할 가능성이 높다.

### 새 run 저장 protocol

```text
codex_output/lrnode_new_run_storage_protocol.md
```

상태: **유효**

새 결과 root:

```text
runs_lrnode_protocol_20260616/
```

### 실험 분리 guide

```text
codex_output/lrnode_experiment_separation_guide.md
```

상태: **유효**

목적:

- 새 실험과 archive/pre-protocol 결과를 섞지 않기 위한 저장 규칙

## 4. Archive / pre-protocol 문서

아래 문서들은 이전 결과를 설명한다.
성능 claim에 직접 쓰면 안 되고, 구현/인프라 확인용 또는 historical context로만 사용한다.

```text
codex_output/lrnode_archive_manifest_20260616.md
codex_output/lrnode_current_experiment_results.md
codex_output/lrnode_experiment_brief_for_chatgpt.md
codex_output/ours_vs_baseline_pipeline_analysis.md
codex_output/etri_lrnode_ppt_summary.md
```

상태: **archive / caution**

이유:

- pre-protocol LR-NODE 결과 기반
- old scratch/eval 결과는 현재 parity/target-mode 논의 이전 결과
- 새 논문 claim에는 새 protocol run 결과가 필요

보관 root:

```text
archived_experiment_results_20260616/pre_protocol_lrnode
```

## 5. 현재 코드 상태 요약

확인일: 2026-06-20

현재 CLI default:

```text
--lrnode_teacher_target_mode shifted_context
```

현재 script default:

```text
scratch_node.sh        -> shifted_context
distill_node.sh        -> shifted_context
scratch_node_joint.sh  -> shifted_context
```

현재 available target modes:

```text
shifted_context
adjacent_sequence
```

현재 미구현이지만 권장되는 next mode:

```text
last-pair adjacent
cache-based shifted_context
```

## 6. 다음 Codex가 반드시 알아야 하는 결론

1. 학습시간 2.x 증가는 LR-NODE tiny module 때문이 아니다.
2. 직접 원인은 현재 `shifted_context`가 teacher target을 만들기 위해 Seer full forward를 한 번 더 실행하기 때문이다.
3. `torch.no_grad()` teacher forward도 compute는 그대로 든다.
4. 현재 parity는 맞춰져 있다. detached LR-NODE loss는 common Seer/action head update를 바꾸지 않는다.
5. dataset/dataloader 변경 없이 가능한 one-forward mode는 `adjacent_sequence`다.
6. eval selected-step과 더 가까운 one-forward proxy는 `last-pair adjacent`다.
7. 이론적으로 가장 정확한 one-forward target은 cache-based shifted-context지만, sample alignment가 필요하다.
8. `scratch_node_joint.sh`는 baseline parity 실험이 아니라 coupled ablation이다.
9. `finetune_node.sh`라는 이름은 deprecated wrapper이고 실제로는 `distill_node.sh`다.

## 7. 권장 읽기 순서

1. `codex_output/lrnode_current_state_handoff.md`
2. `codex_output/baseline_eval_analysis_20260620.md`
3. `codex_output/scratch_node_completion_and_next_steps_20260620.md`
4. `codex_output/lrnode_running_results_analysis_20260620.md`
5. `codex_output/lrnode_parity_audit_20260618.md`
6. `codex_output/lrnode_training_protocol_definitions.md`
7. `codex_output/lrnode_theory_code_explanation.md`
8. 필요한 경우 archive 문서
