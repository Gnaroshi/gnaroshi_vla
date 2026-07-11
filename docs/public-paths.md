# Public Path Placeholders

Public documentation and historical experiment notes use variables instead of
personal workstation, server, dataset, and checkpoint roots.

| Variable | Meaning |
| --- | --- |
| `$VLA_WORKSPACE_ROOT` | Local checkout of this repository |
| `$SEER_WORKSPACE_ROOT` | Local known-working Seer source/workspace |
| `$SEER_BASELINE_ROOT` | Local clean or comparison Seer baseline |
| `$LIBERO_ROOT` | Local LIBERO dataset root |
| `$CODEX_GENERATED_IMAGES` | Local generated-image output directory |
| `$SEER_PYTHON` | Python executable for the Seer environment |
| `$HOME` | Current user's home directory, never a committed username |

These names document path relationships; they do not configure experiments.
Executable YAML and launch configuration can still contain machine-local values.
Changing those values requires an explicit configuration migration and smoke
test because historical commands may depend on them.

New public notes must not include `/Users/<name>`, `/home/<name>`, dataset mount
paths, checkpoint storage roots, internal hostnames, or generated-image cache
paths. Use the variables above or a neutral `/path/to/project` example.
