#!/usr/bin/env bash
# Compatibility entry point; never pairs the old updater with the joint teacher.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/deploy_ll.sh" --preset doll_legacy_ours "$@"
