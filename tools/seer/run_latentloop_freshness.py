#!/usr/bin/env python
"""Stable Seer entry point; the caller never executes code from report folders."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from architectures.seer.adapters.latentloop_freshness.campaign import main

if __name__ == '__main__':
    main()
