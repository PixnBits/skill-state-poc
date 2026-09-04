#!/usr/bin/env python3
"""SKILL.state vs honest ReAct-history on the same warehouse seed.

Prints the proof table (flat vs growing prompt tokens) and writes runs/<timestamp>.json.
"""

from __future__ import annotations

import sys

from skillstate.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["compare", *sys.argv[1:]]))
