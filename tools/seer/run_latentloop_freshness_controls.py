#!/usr/bin/env python
"""Stable entry point for the Seer feature/gate control campaign."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from architectures.seer.adapters.latentloop_freshness_controls.campaign import main

if __name__ == '__main__':
    main()
