#!/usr/bin/env python3
"""Compatibility entry point; install MiniCpG before running this script."""
from minicpg.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
