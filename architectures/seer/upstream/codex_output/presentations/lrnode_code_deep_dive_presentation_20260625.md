# LR-NODE Code-Level Deep Dive Presentation - 2026-06-25

## Figma Slides deck

- Code deep-dive deck: https://www.figma.com/integrations/claim/O3h3nqyCPbdtmAkvqVly5C

Click on any of the previews above to browse presentations or edit in Figma.

## Purpose

이 deck는 결과 발표 deck의 companion이다. 목적은 LR-NODE를 코드 기준으로 빠짐없이 이해하는 것이다.

중심 질문:

- LR-NODE module은 어디서 생성되는가?
- Seer action latent는 어디서 추출되는가?
- 기존 action head는 어떻게 재사용되는가?
- distill adapter에서는 어떤 parameter만 학습되는가?
- adapter-only checkpoint는 왜 base ckpt 없이 평가하면 안 되는가?
- eval에서 full branch와 skip branch는 어떻게 갈라지고, 어떤 cache가 갱신되는가?
- metric과 summary JSON은 어디서 계산되는가?

## Slide list

1. **LR-NODE Code-Level Deep Dive**
   - 핵심: Seer action latent에서 LR-NODE adapter, skip-query eval까지 코드 path를 따라간다.

2. **Code Map**
   - 핵심 파일:
     - `models/lrnode_modules.py`
     - `models/seer_model.py`
     - `train.py`
     - `utils/train_utils.py`
     - `eval_libero.py`
     - `utils/eval_utils_libero.py`
     - `utils/arguments_utils.py`
     - `scripts/LIBERO_LONG/Seer/distill_node.sh`
     - `scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh`

3. **LR-NODE Flags**
   - `use_lrnode_latent_update`: LR-NODE module 생성
   - `lrnode_train_latent_distill`: train branch에서 LR-NODE loss 계산
   - `lrnode_train_protocol`: `joint` vs `adapter`
   - `lrnode_eval_skip_full_forward`: eval skip branch 활성화
   - `lrnode_eval_refresh_policy`: `periodic`, `first_only`, `fixed_budget`

4. **SeerAgent Construction**
   - `models/seer_model.py:155`에서 LR-NODE 관련 constructor args가 들어온다.
   - `models/seer_model.py:316`에서 `use_lrnode_latent_update`가 켜져 있으면 LR-NODE module을 붙인다.
   - `models/seer_model.py:329`에서 `FastVisualDeltaEncoder`, `ControlledLatentNODE`를 만든다.
   - `models/seer_model.py:347`에서 RNG state를 보존한다.

5. **Existing Action Head Reuse**
   - `models/seer_model.py:390`의 `decode_action_from_latent()`가 기존 action head다.
   - `models/seer_model.py:408`의 `decode_lrnode_action_from_latent()`도 같은 head를 호출한다.
   - `models/seer_model.py:21`의 `temporarily_freeze_params()`는 LR-NODE branch에서 gradient 업데이트만 막는다.

6. **FastVisualDeltaEncoder**
   - `models/lrnode_modules.py:6`
   - `[key_rgb, cur_rgb, cur_rgb - key_rgb]`를 concat해 9채널 입력을 만든다.
   - `64x64`로 resize 후 3-layer ConvNet, pooling, linear projection을 거친다.
   - multi-camera feature는 mean으로 합친다.
   - proprio는 `[q_key, q_cur, q_delta]` projection으로 더한다.

7. **ControlledLatentNODE**
   - `models/lrnode_modules.py:91`
   - `dt`, `age`를 scalar time embedding으로 넣는다.
   - `dz = dynamics([z, u, dt, age])`
   - `gate = sigmoid(gate([u, age]) + gate_bias)`
   - `z_next = z_prev + gate * dt * dz`

8. **LR-NODE Wrappers in SeerAgent**
   - `models/seer_model.py:414`: `lrnode_encode_delta()`
   - `models/seer_model.py:441`: `lrnode_apply_dynamics()`
   - `models/seer_model.py:449`: `lrnode_predict_next_latent()`
   - wrapper에서 device/dtype 정렬을 먼저 수행한다.

9. **Full Seer Action Latent**
   - `models/seer_model.py:600`에서 transformer output을 만든다.
   - `models/seer_model.py:624` 주석: action latent는 기존 action decoder에 들어가는 action-token transformer output이다.
   - `models/seer_model.py:626`: `action_latent_full` shape는 `[B, S, action_pred_steps, hidden_dim]`.

10. **Training Branch: Teacher and Student Latents**
    - `models/seer_model.py:631`
    - `shifted_context`: external teacher latent를 받는다.
    - `adjacent_sequence`: `action_latent_full[:, :-1] -> action_latent_full[:, 1:]`.
    - `detach_input_latent`, `detach_teacher_latent`가 gradient 경계를 정한다.

11. **Training Branch: Action Distillation Outputs**
    - `models/seer_model.py:782`
    - `lrnode_z_pred_next`는 기존 action head로 decode된다.
    - teacher latent와 hold latent도 같은 action head로 decode된다.
    - `lrnode_teacher_action`, `lrnode_hold_action`은 loss와 diagnostic에 쓰인다.

12. **Training Loss Composition**
    - `utils/train_utils.py:523`
    - latent: `MSE(z_pred_next, z_teacher_next)`
    - action distill: `L1(lrnode_action, teacher_action)`
    - smooth: `MSE(z_pred_next - z_prev, 0)`
    - `utils/train_utils.py:595`에서 base loss와 LR-NODE loss를 합친다.

13. **Adapter Training Protocol**
    - `train.py:41`의 `_is_lrnode_parameter()`는 `lrnode_delta_encoder.*`, `lrnode_dynamics.*`만 LR-NODE parameter로 본다.
    - `train.py:62`에서 adapter protocol이면 전체를 freeze 후 LR-NODE parameter만 trainable로 되돌린다.
    - `train.py:77`에서 `lrnode_assert_only_lrnode_trainable=1`이면 non-LR-NODE trainable parameter가 남을 때 실패한다.

14. **Why Adapter Checkpoint Is Adapter-Only**
    - `utils/train_utils.py:986`의 `get_checkpoint()`는 `requires_grad=False` parameter를 state_dict에서 제거한다.
    - distill adapter는 Seer/action head가 frozen이므로 checkpoint에는 LR-NODE key만 남는다.
    - valid load log 기준 adapter checkpoint는 `state_dict_keys=30`, `adapter_only=True`.

15. **Eval Checkpoint Loading Guard**
    - `eval_libero.py:56`: adapter-only state_dict 감지.
    - `eval_libero.py:188`: adapter-only checkpoint를 base ckpt 없이 resume하면 error.
    - `eval_libero.py:204`: base ckpt를 먼저 load.
    - `eval_libero.py:212`: adapter ckpt를 overlay.
    - `scripts/LIBERO_LONG/Seer/eval_lrnode_compare.sh:254`에서 `LRNODE_EVAL_BASE_CKPT`를 `--finetune_from_pretrained_ckpt`로 넘긴다.

16. **ModelWrapper State and Counters**
    - `utils/eval_utils_libero.py:139`
    - cache:
      - `lrnode_cached_latent`
      - `lrnode_cached_image_primary`
      - `lrnode_cached_image_wrist`
      - `lrnode_cached_state`
      - `lrnode_cached_age`
    - counters:
      - `full_forward_calls`
      - `lrnode_update_calls`
      - `num_policy_steps`

17. **Refresh Policy Decision**
    - `utils/eval_utils_libero.py:283`
    - skip 조건:
      - LR-NODE enabled
      - eval skip enabled
      - cached latent exists
    - `periodic`: `timestep % K != 0`
    - `first_only`: cache가 생기면 계속 skip
    - `fixed_budget`: full refresh 예산/stride 기준

18. **Full Branch: Run Seer and Cache Latent**
    - `utils/eval_utils_libero.py:748`
    - full branch는 `self.model(..., return_action_latent=...)`를 호출한다.
    - `full_forward_calls`를 증가시킨다.
    - `utils/eval_utils_libero.py:781`에서 `action_latent[:, selected_step]`와 current obs/state를 cache한다.

19. **Skip Branch: Update From Cache**
    - `utils/eval_utils_libero.py:327`
    - `age = lrnode_cached_age + 1`
    - cached obs와 current obs로 `u_delta` 계산
    - `z_next = lrnode_apply_dynamics(z_prev, u_delta, dt=1.0, age)`
    - `decode_action_from_latent(z_next)`로 action 생성
    - cache를 `z_next`와 current obs/state로 갱신

20. **Shadow Full-Forward Diagnostics**
    - `utils/eval_utils_libero.py:690`
    - skip action을 실제로 쓰되 full Seer를 shadow로 실행해 비교한다.
    - 기록:
      - `shadow_latent_mse`
      - `shadow_latent_cos`
      - `shadow_action_l1`
      - `shadow_action_l2`
      - `shadow_action_hold_l1`
    - 이 latency는 deployment latency가 아니다.

21. **Metrics and Artifacts Export**
    - `utils/eval_utils_libero.py:993`: distributed stats merge.
    - `utils/eval_utils_libero.py:1160`: `eval_summary.json` 저장.
    - `effective_full_query_hz = control_hz * full_forward_calls / num_env_steps`
    - `full_query_reduction = 1 - full_forward_calls / num_env_steps`
    - artifact:
      - `eval_summary.json`
      - `eval_episode_metrics.csv`
      - `eval_latency_profile.json`

22. **Code-Level Takeaway**
    - LR-NODE는 action head 앞 latent를 상태처럼 업데이트하는 adapter다.
    - 학습은 frozen Seer/action head + LR-NODE-only optimizer + latent/action/smooth losses다.
    - 평가는 full branch cache와 skip branch latent update의 교대다.
    - valid distill eval은 base ckpt first, adapter overlay second가 필수다.

## Code reference checklist

- `models/lrnode_modules.py:6` - `FastVisualDeltaEncoder`
- `models/lrnode_modules.py:91` - `ControlledLatentNODE`
- `models/seer_model.py:21` - `temporarily_freeze_params`
- `models/seer_model.py:329` - LR-NODE module construction
- `models/seer_model.py:390` - existing action head decode
- `models/seer_model.py:414` - delta encode wrapper
- `models/seer_model.py:441` - dynamics wrapper
- `models/seer_model.py:449` - latent prediction wrapper
- `models/seer_model.py:624` - action latent extraction comment
- `models/seer_model.py:631` - LR-NODE training branch
- `models/seer_model.py:782` - LR-NODE predicted action decode
- `train.py:41` - LR-NODE parameter identification
- `train.py:45` - adapter protocol application
- `train.py:192` - freeze/apply protocol before optimizer
- `utils/train_utils.py:523` - LR-NODE loss calculation
- `utils/train_utils.py:595` - total loss composition
- `utils/train_utils.py:986` - adapter-only checkpoint filtering
- `eval_libero.py:56` - adapter-only checkpoint detection
- `eval_libero.py:188` - invalid adapter-only eval guard
- `eval_libero.py:204` - base checkpoint load
- `eval_libero.py:212` - adapter/resume checkpoint load
- `utils/eval_utils_libero.py:139` - `ModelWrapper` LR-NODE state
- `utils/eval_utils_libero.py:283` - skip/full decision
- `utils/eval_utils_libero.py:316` - cache full-forward state
- `utils/eval_utils_libero.py:327` - skip update from cache
- `utils/eval_utils_libero.py:662` - runtime branch selection
- `utils/eval_utils_libero.py:690` - shadow full-forward diagnostics
- `utils/eval_utils_libero.py:993` - distributed metric merge
- `utils/eval_utils_libero.py:1160` - summary artifact save

## Caveats for speaker

- Eval `freeze_status_snapshot.json` can show trainable flags because eval mode is not optimizer construction. For adapter-only training claims, use `train.py` protocol logic, training status snapshots, checkpoint key count, and load logs.
- Adapter-only ckpt must not be evaluated alone. Code now guards this in `eval_libero.py:188`.
- K=4 latest distill QRED20 remains incomplete in the checked result path and should not be used as a confirmed result.
