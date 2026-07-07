# SimVLA DCLD Adapters

This directory contains SimVLA-specific glue for the architecture-neutral
`methods/dcld` implementation.

The first-pass condition hook is:

```python
enc = model.forward_vlm_efficient(image_input, image_mask, input_ids)
condition = enc["vlm_features"]
```

Action decoding is reproduced externally by calling the original SimVLA action
transformer with the cached/precomputed condition latent.

No file in `architectures/simvla/upstream` is modified by this adapter layer.
