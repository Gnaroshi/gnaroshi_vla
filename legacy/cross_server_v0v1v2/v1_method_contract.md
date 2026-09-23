# LatentLoop V1 Method Contract

## Purpose and frozen boundary

V1 learns a causal multi-interval latent transition while preserving the existing Seer policy. The Seer visual/language backbone, teacher latent producer, and original action generator remain frozen. V1 adds neither a second action head nor an action-space surrogate. Action consistency is evaluated by feeding predicted latents through `decode_action_diagnostics_from_latent`, the same frozen Seer decoder used by the policy.

## Inputs, targets, and recurrence

For `m in {1,2,3}`, a sample contains:

- the detached full-refresh anchor latent at `tau`;
- ordered Primary, Wrist, and proprio observations from `tau` through `tau+m`;
- only the actions actually returned to the environment at `tau..tau+m-1` after temporal ensembling and gripper thresholding;
- the detached full-Seer teacher latent at `tau+m`.

The direct path encodes all `m` causal adjacent observation transitions once, averages the ordered delta features over the interval, conditions on the ordered executed-action sequence and interval embedding, and predicts `z_(tau+m)` in one transition. The composed path applies the same transition core `m` times, passing its own previous prediction forward and consuming one executed action at each step. Future teacher values are detached targets only and never become live inputs.

Online evaluation deliberately uses only the recurrent composed one-step path. At each skipped environment step it consumes the fresh current Primary/Wrist/proprio observation, the cached previous observation, cached latent, and previous environment-executed action; it decodes exactly one current action with the frozen Seer action generator, then writes the updated latent, observations, and executed action back to cache. The direct path is training/offline-diagnostic only.

## Transition core

The V0 `FastVisualDeltaEncoder` and gated fixed-step latent dynamics are initialized from adapter39. V1 adds the executed-action/interval conditioner. The measured candidate trainable count is 518,658: V0 470,146 plus 48,512 (`1.10318x`). Runtime refuses a candidate above either 600,000 parameters or `1.25 * 470,146 = 587,682.5`.

This remains a fixed Euler-like gated residual update, not an adaptive ODE solver or NCDE. Primary and Wrist use the canonical shared camera encoder; camera features are projected, stacked, and meaned only at the canonical fusion point. The projected proprio feature is added afterward and the sum is normalized.

## Losses and matched ablation

The candidate loss contains direct/composed latent distillation, direct/composed frozen-action-generator consistency, symmetric stop-gradient composition consistency, and canonical smoothness. Raw medians are measured first on the transition-training split and used to freeze loss weights before e20/e40 start.

Every e20/e40 row also trains a true matched no-composition control. Candidate and control start from identical transition parameters and receive the same batches, interval schedule, optimizer type, LR schedule, warmup, weight decay, and number of updates. The only intended difference is `composition_weight = 0` for the control. Both have separate DDP modules and AdamW optimizers. This approximately doubles transition-side training compute in each row, but the control is an ablation and is not counted as candidate inference parameters.

## Budget protocol

e20 and e40 are independent seed-42 runs from the same adapter39 initialization; neither resumes from the other. Each uses four GPUs, per-GPU batch 16, gradient accumulation 8, effective batch 512, LR `1e-3`, weight decay `1e-4`, a run-specific cosine horizon, and 5% warmup. Checkpoint and budget selection use only the episode-disjoint checkpoint-validation split and `validation_total_loss`; online LIBERO SR and final evaluation episodes are forbidden for selection.

The selected V1 must pass offline checks for frozen-module gradients, parameter cap, finite/noncollapsed gripper output, age-wise metrics, hold baselines, and composition benefit against the trained matched control. Only then is the fixed K4 200-episode row permitted. K8/K12 are conditional rows allowed only after `V1_ONLINE_PASS` and cannot modify K4 selection.
