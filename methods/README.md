# Methods

This directory stores method-level code and documentation independent of a
specific architecture.

Architecture-specific glue belongs under:

```text
architectures/<architecture>/adapters/<method>/
```

Architecture launch wrappers belong under:

```text
architectures/<architecture>/wrappers/
```

Use explicit method names such as `lrnode` instead of a single generic `ours`
directory. The paper/report can still call a selected method "ours", but code
and configs should name the actual idea.
