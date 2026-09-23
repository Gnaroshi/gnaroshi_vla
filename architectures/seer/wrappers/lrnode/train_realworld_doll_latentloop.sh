#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_realworld_latentloop_adapter.sh" doll
