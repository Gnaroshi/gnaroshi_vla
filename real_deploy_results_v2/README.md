# Seer / LatentLoop Real-World Deployment Results (2026-08-15)

## 1. 보존 범위와 provenance

이 디렉터리는 inference computer에서 수행한 최신 두 interactive live-deployment
session만 보존한다.

- 원본 host: `jbr@210.107.197.121:9000`
- 원본 repository: `/home/jbr/gnaroshi_vla`
- branch: `exp/seer-latentloop-real-deploy-20260814`
- commit: `3e3ff3494c55cbab7fa2b7cc02ceefc2579da61d`
- environment: `seer`
- task: `Pick up the red ball and place it in the basketball hoop`
- teacher: real-world Seer checkpoint 37
- LatentLoop adapter: checkpoint 39 paired with teacher 37
- cameras: two RealSense D435I streams, configured at 640x480 and 60 FPS
- execution: live robot commands enabled, synchronous camera path
- requested control frequency: 60 Hz unless a rollout-specific row says 100 Hz

Raw JSON, JSONL, launch logs, camera images, and front/wrist videos are copied without
content changes. The copy was verified against the inference computer with an rsync
checksum dry-run before the old remote runs were deleted.

## 2. 가장 중요한 해석 제한

이 두 session은 고정 조건의 정식 반복 실험이 아니라 GUI에서 설정을 바꾸며
수행한 interactive pilot이다.

1. LatentLoop session은 이름에 `initial_k4`가 있지만 실제 완료 rollout은
   `K={4,8,10,12,16}`과 target `60/100 Hz`가 섞여 있다.
2. Baseline session은 세 rollout 모두 `K=1`이다. GUI의 policy label은
   `full`, `hold_action`, `hold_latent`로 바뀌었지만, K=1이라 모든 step에서
   full Seer가 호출되었다. 실제 hold call 수는 모두 0이다.
3. 각 조건의 표본 수가 1개뿐이며, baseline은 3개, LatentLoop은 7개의
   완료 rollout만 있다. 따라서 아래 success rate는 정식 성능 추정치나
   통계적 우월성 근거가 아니다.
4. `deployment_runtime_rollout_XXX.json`의 최상위 `query_interval`,
   `control_freq`, `rollout_policy`는 rollout 종료 후 GUI에서 다음 설정을
   선택하면 다음 값이 기록될 수 있다. 실제 완료 rollout 설정은
   `deploy_results.json`의 `runtime_settings_by_rollout`과
   `full_forward_calls / policy_steps`를 함께 사용해 판정했다.
5. 두 session 모두 launch process exit code는 0이다. 각 session에 있는
   `deployment_runtime_rollout_000.json`은 policy step이 0인 초기 기록이므로
   아래 집계에서 제외했다.

## 3. 완료 rollout 결과

### 3.1 LatentLoop

| Rollout | Target Hz | K | Success | Steps | Full calls | Fast calls | Query reduction | Policy mean (ms) | Policy p95 (ms) | Achieved Hz | Control period mean / p95 (ms) |
|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 4 | yes | 214 | 54 | 160 | 74.77% | 41.62 | 86.14 | 10.51 | 95.11 / 142.11 |
| 2 | 60 | 8 | yes | 217 | 28 | 189 | 87.10% | 36.75 | 81.54 | 10.73 | 93.20 / 137.09 |
| 3 | 60 | 10 | yes | 219 | 22 | 197 | 89.95% | 34.50 | 73.24 | 10.71 | 93.40 / 144.34 |
| 4 | 60 | 12 | yes | 228 | 19 | 209 | 91.67% | 33.81 | 66.95 | 10.85 | 92.20 / 120.09 |
| 5 | 60 | 16 | yes | 224 | 14 | 210 | 93.75% | 32.85 | 61.51 | 10.83 | 92.31 / 119.11 |
| 6 | 100 | 16 | yes | 229 | 15 | 214 | 93.45% | 33.27 | 65.67 | 10.84 | 92.27 / 115.23 |
| 7 | 100 | 8 | yes | 211 | 27 | 184 | 87.20% | 35.53 | 71.67 | 10.79 | 92.69 / 125.18 |

`Policy mean/p95`는 이미 획득된 observation이 controller `forward()`에 들어온
시점부터 측정한다. 따라서 image preprocessing, context 준비, 선택된 policy path,
temporal action ensemble 및 action postprocessing은 포함하지만 외부 camera capture와
robot command I/O는 포함하지 않는다. 내부 모듈 timer의 전체 pooled mean은 full
Seer 50.27 ms, LatentLoop update 4.50 ms다. 반면 이 per-step policy timer의 pooled
mean은 full step 75.72 ms, fast step 30.13 ms다.

전체 혼합 session의 descriptive aggregate는 1,542 policy steps, 179 full calls,
1,363 fast calls, 88.39% full-query reduction, 35.42 ms mean policy latency,
10.75 achieved Hz이다. 조건이 섞여 있으므로 이 aggregate를 단일 K 결과로
인용하면 안 된다.

### 3.2 Seer baseline

| Rollout | Target Hz | K | GUI policy label | Effective behavior | Success | Steps | Full calls | Policy mean (ms) | Policy p95 (ms) | Achieved Hz | Control period mean / p95 (ms) |
|---:|---:|---:|---|---|:---:|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 1 | full | full Seer every step | yes | 201 | 201 | 74.92 | 101.11 | 7.05 | 141.77 / 170.13 |
| 2 | 60 | 1 | hold_action | full Seer every step | no | 310 | 310 | 71.04 | 89.99 | 7.23 | 138.24 / 166.14 |
| 3 | 60 | 1 | hold_latent | full Seer every step | yes | 242 | 242 | 72.01 | 91.22 | 7.25 | 137.86 / 160.27 |

Baseline pooled descriptive aggregate는 753 policy steps, 753 full calls,
72.39 ms mean policy latency, 47.54 ms mean internal full-forward latency,
7.19 achieved Hz이다. Pilot success 표기는 2/3이다.

## 4. 동일한 60 Hz target에서의 K=4 pilot 비교

LatentLoop rollout 1만 K=4이고, baseline 세 rollout은 모두 효과적으로 동일한
K=1 full-Seer 실행이다. 이 제한된 pilot에서 관찰된 값은 다음과 같다.

| Metric | Seer baseline K=1 (3-rollout pooled) | LatentLoop K=4 (1 rollout) | Descriptive change |
|---|---:|---:|---:|
| Full-query reduction | 0.00% | 74.77% | +74.77 pp |
| Mean end-to-end policy latency | 72.39 ms | 41.62 ms | 42.5% lower |
| Mean internal full / fast module latency | 47.54 ms full | 4.57 ms fast | fast module is 10.4x faster |
| Achieved controller rate | 7.19 Hz | 10.51 Hz | 46.2% higher |
| Mean control period | 139.06 ms | 95.11 ms | 31.6% lower |
| Strict 60 Hz deadline miss rate | 100% | 100% | neither reached 60 Hz |
| Pilot success labels | 2/3 | 1/1 | insufficient for an SR claim |

The 60 Hz target corresponds to a 16.67 ms period. Both paths missed that strict
deadline on every measured interval. LatentLoop improved throughput and reduced full
model calls, but this synchronous live implementation did not execute at 60 Hz.
The achieved rate is computed from consecutive robot-command completion timestamps.
It therefore reflects the complete outer-loop cadence, including work between command
completions such as observation acquisition, preprocessing, policy execution, pacing,
and robot I/O. It is not the reciprocal of model-only latency.

## 5. Raw artifact layout

- `baseline/.../deploy_results.json`: manually recorded outcome and rollout metadata
- `latentloop/.../deploy_results.json`: manually recorded outcome and rollout metadata
- `deployment_runtime_rollout_*.json`: per-rollout latency, call count, rate, and deadline metrics
- `policy_steps_rollout_*.jsonl`: per-policy-step traces
- `SESSION_MANIFEST.json`: session/checkpoint/config provenance
- `launch_logs/...`: exact launch config, command, code snapshot, git state, hashes, console, and exit code
- `rollouts/...`: before-deploy images and front/wrist videos

## 6. Code snapshot hashes used on the inference computer

The exact deployed v2 source was also synchronized back to sd1 but intentionally not
included in the result-only Git commit.

| File | SHA-256 |
|---|---|
| `deploy_ll_gui_v2.sh` | `6b3015f613b3964a6801ca09334f912782d023cdb9ce82fa0246ea8f7de6f968` |
| `controller_v2.py` | `fecc186d7836e9342e5280baa8a58668237da97bcc26dec8db3d2976ac20327b` |
| `deploy_ll_gui_v2.py` | `a25f2b03ab8feaa39600bc4f4ddb2c8eaac262a5c9516c51e47042f4d7d3f747` |
| `runtime_v2.py` | `291f0a0712bdc52bb8b287974d9cb6074ed2814a1d0db3a0bb13cfd798aacf4f` |
| `test_latentloop_real_deploy_v2.py` | `9010f48ea946f1379350cd9269439b566d7649d07d0a9f5128393f3ca1acf3bc` |
