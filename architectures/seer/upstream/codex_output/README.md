# Codex Output 문서 구조

작성일: 2026-06-22

이 폴더는 LR-NODE 구현/실험/보고 문서를 성격별로 분리해 저장한다. 앞으로 새 markdown 문서도 아래 기준에 맞춰 넣는다.

## 1. ChatGPT / 발표 보고서

위치:

```text
codex_output/chatgpt_reports/
```

용도:

- ChatGPT 또는 외부 협업자에게 현재 상황을 전달하는 요약 문서
- ETRI/PPT용 정리
- handoff 문서

핵심 파일:

- `lrnode_report_for_chatgpt_20260622.md`: 현재까지 구현/결과/해석을 ChatGPT에 전달하기 위한 최신 종합 보고서
- `lrnode_current_state_handoff.md`: 현재 구현 상태 handoff
- `lrnode_experiment_brief_for_chatgpt.md`: 이전 ChatGPT 공유용 실험 브리프
- `etri_lrnode_ppt_summary.md`: ETRI 발표용 요약

## 2. 방법론 / 이론 / 코드 해설

위치:

```text
codex_output/methodology/
```

용도:

- LR-NODE 원리와 코드 단위 해설
- metric 정의와 수식
- baseline vs ours pipeline 분석

핵심 파일:

- `lrnode_experiment_principle_analysis_20260622.md`: QRED20/HZUP20Q 실험이 성립하는 원리와 근거
- `seer_bottleneck_analysis_20260622.md`: 기존 Seer full-forward/policy/env 병목 분석
- `lrnode_theory_code_explanation.md`: LR-NODE 코드/이론 상세 해설
- `lrnode_eval_metrics_definition_20260622.md`: latency, Hz, query reduction, jerk 등 metric 정의
- `ours_vs_baseline_pipeline_analysis.md`: baseline과 ours pipeline 차이
- `lrnode_policy_context_distillation.md`: policy-context distillation 관련 설명

## 3. 실험 결과 / 분석

위치:

```text
codex_output/experiment_results/
```

용도:

- baseline eval 결과
- LR-NODE QRED/HZUP 결과
- 현재까지 나온 결과 분석
- 다음 실험 제안

핵심 파일:

- `baseline_eval_analysis_20260620.md`: plain Seer baseline eval 분석
- `lrnode_qred_hzup_result_analysis_20260622.md`: QRED20/HZUP20Q 결과 분석
- `lrnode_current_experiment_results.md`: 이전 현재 결과 요약
- `lrnode_results_and_next_experiments_20260621.md`: 결과와 다음 실험
- `scratch_node_completion_and_next_steps_20260620.md`: scratch_node 종료 후 다음 단계

## 4. 실험 프로토콜 / 스크립트 사용법

위치:

```text
codex_output/protocols/
```

용도:

- 학습/eval protocol 정의
- 실험 naming 규칙
- 저장 구조
- sweep script 사용법

핵심 파일:

- `lrnode_training_protocol_definitions.md`: scratch / scratch_node / distill protocol 정의
- `lrnode_eval_experiment_names_20260621.md`: QRED20, HZUP20Q 등 실험명 정의
- `lrnode_scratch_hz_sweep_plan_20260621.md`: Hz/K sweep 계획
- `lrnode_new_run_storage_protocol.md`: 새 run 저장 규칙
- `lrnode_protocol_scripts.md`: protocol script 정리

## 5. 감사 / 검증 / 문제 추적

위치:

```text
codex_output/audits/
```

용도:

- parity 검증
- mismatch 원인 추적
- archive manifest

핵심 파일:

- `lrnode_parity_audit_20260618.md`: parity audit
- `lrnode_k1_ckpt30_mismatch_audit_20260620.md`: K=1 mismatch audit
- `lrnode_archive_manifest_20260616.md`: archived experiment manifest

## 6. Figure / 이미지

위치:

```text
codex_output/figures/
```

용도:

- raster figure
- generated architecture image
- 발표용 이미지

주의:

- vector diagram이 아니라 raster 이미지가 필요하면 여기에 저장한다.
- figure 생성용 script는 필요 시 `codex_output/` 또는 `scripts/figures/`에 둘 수 있다.
