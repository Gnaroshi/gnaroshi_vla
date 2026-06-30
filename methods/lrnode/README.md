# LR-NODE

LR-NODE is currently implemented inside the Seer working copy. This directory
collects method-owned reference code and docs so future refactors can separate
the method from architecture source trees.

- `seer_reference/`: non-invasive reference copies extracted from the current
  Seer source.

Future architecture-neutral LR-NODE code should live directly under this method
directory. Architecture-specific wiring should live under
`architectures/<architecture>/adapters/lrnode/`.
