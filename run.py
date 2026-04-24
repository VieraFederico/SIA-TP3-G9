#!/usr/bin/env python3
"""Project entry point → delegates parsing to ``utils`` and the run loop to ``scripts``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.run_experiment import run_experiment
from utils.run_parser import parse_run_config


def main() -> None:
    config_path, run_id = parse_run_config(_ROOT, prog="run.py")
    run_experiment(config_path, run_id=run_id)


if __name__ == "__main__":
    main()
