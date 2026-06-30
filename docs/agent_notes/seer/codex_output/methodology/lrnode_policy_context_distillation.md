# LR-NODE policy-context distillation

이 문서는 2026-06-17 코드 수정 이후 LR-NODE teacher target 정의를 고정한다.

> 2026-06-20 문서 상태: 이 문서는 `shifted_context` teacher target의 원래 의도와 정의를 설명하는 참고 문서다. 이후 논의에서 현재 `shifted_context` 구현은 teacher target을 만들기 위해 Seer full forward를 한 번 더 실행하므로 학습시간이 2.x까지 늘 수 있음이 확인됐다. 최신 구현/실험 판단은 `codex_output/lrnode_current_state_handoff.md`와 `codex_output/README_LRNODE_DOCS.md`를 우선한다.

## 핵심 정리

LR-NODE가 학습해야 하는 것은 같은 sequence tensor 안의 인접 token transition이 아니라, **다음 environment step에서 full policy를 호출했을 때 얻는 action-interface latent**다.

일반 VLA/robot policy에 대해 다음처럼 정의한다.

```text
C_t       : policy가 env step t에서 정상적으로 받는 입력 context
C_{t+1}   : policy가 env step t+1에서 정상적으로 받는 입력 context
Probe(.)  : action head 직전 latent를 추출하는 hook/interface
z_t^T     = Probe(TeacherPolicy(C_t))
z_{t+1}^T = Probe(TeacherPolicy(C_{t+1}))
u_t       = DeltaEncoder(obs_t, obs_{t+1}, q_t, q_{t+1})
z_hat_{t+1} = LRNode(z_t^T, u_t)
```

손실:

```text
L_latent = ||z_hat_{t+1} - stopgrad(z_{t+1}^T)||_2^2
L_action = ||Head(z_hat_{t+1}) - stopgrad(Head(z_{t+1}^T))||_1
L_smooth = ||z_hat_{t+1} - stopgrad(z_t^T)||_2^2

L_LRNode = lambda_z L_latent + lambda_a L_action + lambda_s L_smooth
```

## Seer에서의 구현

Seer는 history window를 입력으로 받으므로 policy context는 window다.

```text
C_t     = images[:, 0:S], states[:, 0:S], text[:, 0:S]
C_{t+1} = images[:, 1:S+1], states[:, 1:S+1], text[:, 1:S+1]
```

기본 probe 위치:

```text
lrnode_context_selected_step = -1
```

즉 steady-state eval에서 cache되는 마지막 context timestep latent를 학습 target으로 삼는다.

코드 위치:

- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:306): target mode 선택
- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:367): shifted context 생성
- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:382): `C_{t+1}` teacher forward
- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:407): `z_t^T`, `z_{t+1}^T` 추출
- [utils/train_utils.py](/home/mingyujung/private/seer/seer_node3/utils/train_utils.py:415): LR-NODE latent update

## 왜 기존 adjacent target보다 낫나

기존 target:

```text
z_prev = action_latent_full[:, :-1]
z_teacher = action_latent_full[:, 1:]
```

이 target은 같은 policy context 안의 token index transition을 학습한다. 하지만 eval skip에서 실제로 근사해야 하는 것은 `C_t`에서 `C_{t+1}`로 policy input 자체가 바뀌었을 때의 teacher latent다.

따라서 현재 기본값은 다음이다.

```text
--lrnode_teacher_target_mode shifted_context
--lrnode_context_selected_step -1
```

기존 방식은 ablation/legacy 비교용으로만 남긴다.

```text
--lrnode_teacher_target_mode adjacent_sequence
```

## Gradient protocol

### teacher-student detached

사용 script:

```text
scratch_node.sh
distill_node.sh
```

핵심:

```text
z_t^T.detach()
z_{t+1}^T.detach()
LR-NODE branch의 action head parameter freeze
```

gradient 경로:

```text
LR-NODE loss -> lrnode_delta_encoder, lrnode_dynamics
LR-NODE loss -/-> Seer backbone
LR-NODE loss -/-> action head parameter
```

단 `scratch_node.sh`에서는 base Seer loss가 동시에 존재하므로 Seer/action head는 base loss로는 학습된다.

### coupled joint

사용 script:

```text
scratch_node_joint.sh
```

핵심:

```text
z_t^T is not detached
LR-NODE branch의 action head는 trainable
z_{t+1}^T target remains detached
```

gradient 경로:

```text
LR-NODE loss -> lrnode modules
LR-NODE loss -> z_t^T -> Seer backbone
LR-NODE action loss -> action head parameter
```

이 설정은 "NODE loss를 원본 Seer 학습에도 같이 고려하는 joint"다. 반대로 `scratch_node.sh`는 "scratch 학습 + LR-NODE detached distillation"이다.

## 현재 제한

1. shifted-context 기본 모드는 one-step context transition만 학습한다.
2. `lrnode_multistep_train=1`은 legacy `adjacent_sequence`에만 구현되어 있다.
3. shifted-context에서 `lrnode_bc_weight > 0`은 막아 두었다. dataset action label을 어느 context/probe step에 맞출지 별도 정의가 필요하기 때문이다.
4. eval skip은 여전히 cached latent에서 Euler update를 반복한다. K가 커질수록 training target과 eval rollout horizon 차이가 생기므로 K sweep과 shadow full-forward logging이 필요하다.
