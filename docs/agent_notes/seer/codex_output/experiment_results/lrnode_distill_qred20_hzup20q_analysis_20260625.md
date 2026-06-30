# LR-NODE Distill QRED20 / HZUP20Q Analysis - 2026-06-25

## Checkpoint Protocol

- Baseline full Seer checkpoint: `ckpt33`
- LR-NODE distill adapter checkpoint: `ckpt39`
- Eval loader: load baseline `ckpt33` first, then overlay adapter `ckpt39`
- QRED20 result root:
  `runs_lrnode_protocol_20260616/eval/lrnode_distill_qred20_lrnode_distill_from_scratch_baseline_ckpt33_lronly_v1_lw05_aw01_g4_ckpt39_vs_seer_scratch_baseline_ckpt33_distill_qred20_20260624_135213`
- HZUP20Q result root:
  `runs_lrnode_protocol_20260616/eval/lrnode_distill_hzup20q_distill_hzup20q_20260625_012048`

## QRED20 Final Results

QRED20 is complete. All rows saved `eval_summary.json`, videos, and `experiment_summary.csv`.

| Row | SR | Full Query Hz | LR-NODE Hz | Full Calls | LR-NODE Calls | Full Query Reduction | Policy ms | Full ms | LR-NODE ms | Jerk p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline full K=1 | 83.0 | 20.000 | 0.000 | 66,564 | 0 | 0.0% | 73.159 | 62.870 | 0.000 | 0.087384 |
| Ours full K=1 | 83.0 | 20.000 | 0.000 | 66,564 | 0 | 0.0% | 73.452 | 63.005 | 0.000 | 0.087384 |
| Ours skip K=2 | 84.0 | 10.014 | 9.986 | 32,903 | 32,813 | 49.932% | 44.985 | 62.468 | 6.690 | 0.183275 |
| Ours skip K=3 | 85.5 | 6.683 | 13.317 | 21,365 | 42,571 | 66.584% | 38.133 | 66.403 | 7.236 | 0.229699 |
| Ours skip K=4 | 91.0 | 5.022 | 14.978 | 15,145 | 45,175 | 74.892% | 32.281 | 65.299 | 6.928 | 0.563964 |

## QRED20 Interpretation

The load-parity row is correct: baseline full K=1 and adapter-composed ours full K=1 both produce 83.0% SR and identical jerk p95. This confirms that adapter eval is now using the frozen baseline Seer/action head correctly.

The query-reduction result is strong on SR:

- K=2 reduces full Seer queries by about 50% and slightly improves SR to 84.0%.
- K=3 reduces full Seer queries by about 66.6% and improves SR to 85.5%.
- K=4 reduces full Seer queries by about 74.9% and improves SR to 91.0%.

The efficiency result is also strong:

- Full policy latency is about 73 ms.
- K=4 policy latency is about 32 ms.
- LR-NODE update itself is only about 7 ms.

However, action smoothness worsens as K increases:

- Baseline / full K=1 jerk p95: 0.087
- K=2 jerk p95: 0.183
- K=3 jerk p95: 0.230
- K=4 jerk p95: 0.564

Therefore the current interpretation should be:

> LR-NODE distill can substantially reduce full Seer query rate while preserving or improving LIBERO success rate, but large K changes action dynamics and increases high-percentile jerk. The videos and per-task failures should be inspected before claiming K=4 is unconditionally better.

## HZUP20Q Current Results

HZUP20Q is still running.

Completed official rows so far:

| Row | SR | Control Hz | Eval Max Steps | Full Query Hz | Policy ms | Full ms | Jerk p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20Hz baseline full K=1 | 83.0 | 20 | 600 | 20.0 | 73.242 | 62.903 | 0.087384 |
| 20Hz ours full K=1 | 83.0 | 20 | 600 | 20.0 | 73.632 | 63.134 | 0.087384 |
| 40Hz baseline full K=1 | 83.5 | 40 | 1200 | 40.0 | 71.003 | 60.769 | 0.035067 |

Currently running:

- `40Hz ours full K=1`
- Current video count at inspection time: `179/200`
- No `eval_summary.json` yet for this row, so its official SR should not be interpreted yet.

## HZUP20Q Preliminary Interpretation

The completed 40Hz baseline full row shows that increasing LIBERO `control_freq` to 40 and scaling max steps to 1200 does not break baseline Seer. It gives 83.5% SR, close to the 20Hz 83.0%.

But this row is computationally not real-time:

- 40Hz control budget is 25 ms per control step.
- Baseline full policy latency is about 71 ms.

So the simulator can evaluate 40Hz because it is not enforcing wall-clock real-time policy execution. For a real robot, full Seer at 40Hz would miss the strict 25 ms control budget. This is exactly why HZUP20Q needs LR-NODE skip rows such as 40Hz K=2, 60Hz K=3, and 80Hz K=4.

## Immediate Next Checks

After HZUP20Q finishes, update this analysis with:

- 40Hz ours full K=1
- 40Hz ours skip K=2
- 60Hz baseline/ours full if completed by the script
- 60Hz K=3
- 80Hz baseline/ours full if completed by the script
- 80Hz K=4

Key success criterion:

> At higher control Hz, LR-NODE skip rows should keep effective full Seer query rate near 20Hz or below while maintaining SR close to the corresponding full-query row and reducing policy-step latency.
