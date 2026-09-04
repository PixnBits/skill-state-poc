#!/usr/bin/env python3
"""SKILL.state vs history matrix on local Ollama models."""

from __future__ import annotations

import sys

from skillstate.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["bench", *sys.argv[1:]]))
