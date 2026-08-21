# DCLD Method Layer

This directory contains architecture-neutral DCLD components.

DCLD here means fixed-Euler delta-conditioned latent dynamics:

```text
cached condition latent + observation delta -> updated condition latent
```

Architecture-specific code, such as SimVLA condition extraction and action
decoding, belongs under `architectures/<name>/adapters/dcld/`.

This method layer does not import SimVLA upstream modules.
