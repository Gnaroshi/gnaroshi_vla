#!/usr/bin/env bash
# Compatibility entry point for the previous Doll baseline.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/deploy_ll.sh" --preset doll_legacy_baseline "$@"
