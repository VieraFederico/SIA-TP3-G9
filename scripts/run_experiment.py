"""Experiment driver: load config, build objects, train, persist.

``run.py`` at the repo root calls :func:`run_experiment`. You can also run::

    python scripts/run_experiment.py -ex 2 -c baseline
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def run_experiment(config_path: Path, run_id: str | None = None) -> None:
    """Main training loop for exercises 0–3 (validation + TP3 exercises).

    Args:
        config_path: Resolved JSON under ``experiments_configs/``.
        run_id: Optional override for the run folder name under ``experiments/``.
    """
    _ = run_id  # wired when persistence is implemented
    _ = config_path
    raise NotImplementedError(
        "TODO: load_config → build_experiment → train → ExperimentWriter.write"
    )


def _cli_main() -> None:
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from utils.run_parser import parse_run_config

    config_path, run_id = parse_run_config(_ROOT, prog="scripts/run_experiment.py")
    run_experiment(config_path, run_id=run_id)


if __name__ == "__main__":
    _cli_main()
