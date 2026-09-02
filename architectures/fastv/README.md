# FastV upstream source

The official FastV repository is kept as an unmodified, ignored nested clone.

```bash
git clone https://github.com/pkunlp-icler/FastV.git architectures/fastv/upstream
git -C architectures/fastv/upstream checkout d1659729b5bf1be225e99ee15783deeea80f63b1
```

The SimVLA adapter validates the commit and selected file hashes before an
evaluation starts. Do not patch this clone. All integration code belongs under
`architectures/simvla/adapters/fastv/`.

The official repository does not contain a SimVLA implementation. The local
code is an official-algorithm adaptation whose VLA-specific interface choices
are documented in the adapter README.
