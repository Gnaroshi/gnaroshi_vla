# VLA-Cache for SimVLA

This package is a method-faithful architecture adaptation of the released
VLA-Cache implementation, not an author-provided SimVLA port.

## Pinned evidence

- VLA-Cache repository commit:
  `a4909880573868dee2769343d52e793c0341678b`
- Required Transformers fork commit:
  `2302fce58afa3a4f8461625b1394f9e9c8a7f1ea`
- Official pruning layers: `2, 6, 9, 11`
- Official image-patch cosine threshold: `0.996`
- Official entropy-schedule positive-growth factor: `0.55`
- OpenVLA-OFT stable/task-relevant selection: `150/256` and `100/256`
  tokens per view.

The released code supports OpenVLA and OpenVLA-OFT, whose decoder receives a
16x16 visual-token grid. SimVLA's SmolVLM connector exposes a 6x6 grid. The
port therefore preserves the published fractions at the actual compute-token
granularity: 21 stable and 14 task-relevant tokens per view. Patch similarity
is computed after the exact SimVLA resize path but after undoing ImageNet
normalization, so the official RGB cosine threshold is not silently applied in
a different feature space. All other choices above remain unchanged.

## Actual compute path

This is not output interpolation, condition mixing, or FLOPs-only accounting.
On every query the vision encoder still processes the current exterior and
wrist images. At decoder layers 2, 6, 9, and 11, reusable visual queries are
removed from the current hidden sequence. Their old K/V entries remain in a
fixed-position cache while current text and non-reused visual tokens update
their own entries and attend to the complete cache. Consequently Q/K/V
projection, attention-query, output-projection, and MLP work are skipped for
the removed queries.

OpenVLA consumes selected action-token outputs, whereas SimVLA passes the full
122-token condition to its action transformer. Removed visual positions are
therefore reconstructed from their previous final hidden states. This is a
necessary architecture mapping; it is not a learned or weighted interpolation.

## Fair comparison

`vla_cache` shares the exact official parent, real action-transformer overlay,
normalization, two-view preprocessing, deterministic flow noise, fresh H=10
chunk, R=5 execution, and N_G=10 action-generation path used by the real
baseline. Only SmolVLM decoder token computation changes.

For a like-for-like backbone comparison, `condition_loop` uses our learned
condition update at K_C=2 but restores N_G=10. The complete `latentloop`
operating point remains K_C=2,N_G=3 and is compared against the complete
K_C=1,N_G=10 baseline rather than presented as a condition-cache-only result.

The attention-entropy rule requires materialized attention probabilities, so
the reference implementation uses eager attention. `vla_cache_full` executes
the identical eager path without token reuse. Report both:

- `vla_cache_full` versus `vla_cache` isolates the cache's actual speedup;
- the standard SDPA `baseline` and `latentloop`, together with the eager
  `vla_cache`, are measured end to end on the same GPU so that the cache's
  eager-attention cost is not hidden.

Each runtime trace records active tokens per decoder layer, actual skipped
token-layers, selected positions, and whether old K/V was used. The first
query is always a full computation and is unit-tested to equal the native eager
decoder exactly.

## Offline real-data comparison

`architectures/simvla/wrappers/run_real_vla_cache_comparison.sh --all` runs the
five comparison rows on one fixed validation sequence from the converted real
demonstrations. It uses paired flow noise and the same fresh-H=10/execute-R=5
contract for every row. The benchmark reports synchronized CUDA latency, peak
VRAM, action fidelity, and actual token-layer/K/V reuse over 500 queries and
three repeats by default.

This benchmark does not command a robot and does not measure task success.
Task-success claims require a separately paired live evaluation on the
inference computer.
