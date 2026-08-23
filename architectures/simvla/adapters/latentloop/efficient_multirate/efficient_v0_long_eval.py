"""Run unchanged native V0 Long evaluation under an efficient child lock."""

from __future__ import annotations

import argparse
import sys

from architectures.simvla.adapters.latentloop import native_v0_long_eval
from architectures.simvla.adapters.latentloop.efficient_multirate.lineage_bridge import (
    install_native_evaluator_lineage_bridge,
    load_child_source_lock,
)


def main() -> int:
    bridge_parser = argparse.ArgumentParser(add_help=False)
    bridge_parser.add_argument("--source-lock", required=True)
    bridge, remaining = bridge_parser.parse_known_args()
    child_source = load_child_source_lock(bridge.source_lock)
    install_native_evaluator_lineage_bridge(native_v0_long_eval, child_source)
    sys.argv = [sys.argv[0], *remaining]
    return native_v0_long_eval.main()


if __name__ == "__main__":
    raise SystemExit(main())

