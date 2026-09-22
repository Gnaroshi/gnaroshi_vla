#!/usr/bin/env bash
# Compatibility entry point; all settings live in deploy_ll.sh.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/deploy_ll.sh" --preset doll_joint_baseline "$@"
