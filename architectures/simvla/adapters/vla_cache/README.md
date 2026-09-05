# SimVLA VLA-Cache LIBERO adaptation

This adapter evaluates a training-free VLA-Cache adaptation on the frozen
`YuankaiLuo/SimVLA-LIBERO` checkpoint. It preserves SimVLA's action horizon,
execution horizon, and ten-step action-flow solver.

The published VLA-Cache implementation targets OpenVLA-OFT rather than SimVLA.
`official_contract.py` pins the source repositories and maps its published token
selection fractions from a 16x16 image-token grid to SmolVLM's 6x6 connector
grid. `smolvlm_runtime.py` performs actual decoder-token skipping and retains
the corresponding prior K/V entries; it is not a zero-mask proxy.

`vla_cache_full` calls the unchanged native backbone forward; it does not
construct the sparse decoder or change its attention backend. `vla_cache`
uses private eager-attention configuration with shared frozen weights. Both
use `K_C=1`, `N_G=10`, `H=10`, and `R=5`.

Task relevance uses only valid text query positions, excluding padding from
the selector without changing the model's original input/mask. The entropy
schedule includes every actual layer map. Official code has a trailing
metadata entry; this adapter's list does not, so no map is dropped.

Skipped positions retain their prior K/V. Because SimVLA consumes the full
condition sequence, removed output positions retain their previous final
hidden vector. That output reconstruction and the 36-token camera budget
are architecture-specific choices, not features validated by the original
authors on SimVLA. Results must be labelled as a SimVLA adaptation.

The rb2 pipeline consumes the exact three 500-episode LIBERO-Long manifests
already used by the SimVLA paper evaluation. Existing baseline and proposed
method results are referenced rather than rerun.

Run `architectures/simvla/wrappers/run_vla_cache_fidelity_tmux.sh --verify`
on rb2 for CPU tests plus four real policy-query observations. This checks
an independently loaded native reference, cache-off action equality, actual
sparse computation, optimized-versus-reference equality, and H=10/R=5 queue
behavior. Encoder timing is diagnostic, not an end-to-end speedup claim.

Run the same wrapper with `--all` to verify and then evaluate only VLA-Cache:
LIBERO-Long, 10 tasks x 50 trials x 3 seeds. It writes new `oft_runtime_v3`
results and refuses to mix different adapter source versions. Completed
rows are skipped on restart. The outer tmux wrapper returns zero to preserve
the pane; `pipeline.status` and the inner launcher's exit status retain errors.
No success-rate threshold stops the sweep.

The runtime resolves token selection and gather indices before launching the
current vision encoder. Decoder layers use precomputed index gathers instead
of boolean indexing that synchronizes CUDA to resolve dynamic output sizes.
The reference implementation remains available with `optimized=False` for
condition, action, and K/V equivalence checks.

`tools/simvla/profile_vla_cache.py` measures native SDPA, native eager, a
no-reuse adapter control, and sparse reuse on the same frozen model and real
observations. Profiler traces and uninstrumented wall timings are separate.
Policy timings include preprocessing, ten flow steps and five returned
actions, but exclude environment stepping. These bounded measurements are
not replacements for full-suite success rates or paper latency measurements.
