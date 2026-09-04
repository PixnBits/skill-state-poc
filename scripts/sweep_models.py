#!/usr/bin/env python3
"""Same warehouse + silent-drift protocol across local Ollama models."""

from __future__ import annotations

import sys

from skillstate.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["sweep", *sys.argv[1:]]))
