# Latent Bridge upstream

The official Latent Bridge repository is kept as an unmodified, ignored nested
clone at `architectures/latent_bridge/upstream`.

- Repository: `https://github.com/1999Lyd/Latent-Bridge.git`
- Pinned commit: `ed556014aa96bae8ed85768194f02360389b9365`
- SimVLA integration: `architectures/simvla/adapters/latent_bridge`

Clone the pinned source with:

```bash
git clone https://github.com/1999Lyd/Latent-Bridge.git architectures/latent_bridge/upstream
git -C architectures/latent_bridge/upstream checkout ed556014aa96bae8ed85768194f02360389b9365
```

The integration verifies the commit and hashes of the official files before
loading `DiTCrossBlock` and `DiTFinalLayer`. Do not patch the nested clone.
This is a SimVLA adaptation of the published feature-bridge algorithm, not an
official Latent Bridge SimVLA implementation.
