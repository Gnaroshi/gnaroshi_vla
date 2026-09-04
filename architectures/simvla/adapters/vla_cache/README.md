# SimVLA VLA-Cache LIBERO adaptation

This adapter evaluates a training-free VLA-Cache adaptation on the frozen
`YuankaiLuo/SimVLA-LIBERO` checkpoint. It preserves SimVLA's action horizon,
execution horizon, and ten-step action-flow solver.

The published VLA-Cache implementation targets OpenVLA-OFT rather than SimVLA.
`official_contract.py` pins the source repositories and maps its published token
selection fractions from a 16x16 image-token grid to SmolVLM's 6x6 connector
grid. `smolvlm_runtime.py` performs actual decoder-token skipping and retains
the corresponding prior K/V entries; it is not a zero-mask proxy.

`vla_cache_full` runs the same eager-attention decoder backend without token
reuse. It isolates backend effects and is not a replacement for the existing
official SimVLA baseline. `vla_cache` enables the cache. Both use `K_C=1`,
`N_G=10`, `H=10`, and `R=5`.

The rb2 pipeline consumes the exact three 500-episode LIBERO-Long manifests
already used by the SimVLA paper evaluation. Existing baseline and proposed
method results are referenced rather than rerun.
