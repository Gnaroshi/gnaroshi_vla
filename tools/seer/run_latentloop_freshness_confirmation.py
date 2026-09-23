#!/usr/bin/env python
"""Stable project entry point for Seer feature/gate replication."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from architectures.seer.adapters.latentloop_freshness_confirmation.campaign import main

if __name__ == '__main__':
    main()
