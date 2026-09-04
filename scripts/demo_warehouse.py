#!/usr/bin/env python3
"""Run the warehouse SKILL.state demo (Ollama by default, --offline for the scripted policy)."""

from __future__ import annotations

import sys

from skillstate.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["warehouse", *sys.argv[1:]]))
