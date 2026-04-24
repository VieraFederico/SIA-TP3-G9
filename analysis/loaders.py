"""Load artefacts from ``experiments/<run_id>/`` without importing training or models.

Data layout matches ``ExperimentWriter`` / ``experiments/*/``:
``config.json``, ``history.json``, ``metrics.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class RunHistory:
    """Deserialised training history (same JSON shape as ``TrainingHistory``).

    Kept in ``analysis/`` so this package does not depend on ``src.training``.
    """

    train_losses: list[float]
    val_losses: list[float]
    epoch_times: list[float]
    extra_metrics: dict[str, list[float]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunHistory:
        return cls(
            train_losses=list(data.get("train_losses", [])),
            val_losses=list(data.get("val_losses", [])),
            epoch_times=list(data.get("epoch_times", [])),
            extra_metrics=dict(data.get("extra_metrics", {})),
        )


def _run_dir(run_id: str, base_dir: Path) -> Path:
    return base_dir / run_id


def load_run(
    run_id: str,
    base_dir: Path | str = Path("experiments"),
) -> dict[str, Any]:
    """Load config, history, and metrics for a run as plain dicts / nested structures.

    Args:
        run_id: Folder name under ``base_dir``.
        base_dir: Root that contains ``<run_id>/``.

    Returns:
        Dict with keys ``run_id``, ``config``, ``history`` (``RunHistory``), ``metrics``.
    """
    root = Path(base_dir)
    d = _run_dir(run_id, root)
    cfg_path = d / "config.json"
    hist_path = d / "history.json"
    met_path = d / "metrics.json"

    config: dict[str, Any] = {}
    if cfg_path.is_file():
        config = json.loads(cfg_path.read_text(encoding="utf-8"))

    history = RunHistory.from_dict({})
    if hist_path.is_file():
        history = RunHistory.from_dict(json.loads(hist_path.read_text(encoding="utf-8")))

    metrics: dict[str, float] = {}
    if met_path.is_file():
        raw = json.loads(met_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            metrics = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}

    return {
        "run_id": run_id,
        "config": config,
        "history": history,
        "metrics": metrics,
    }


def load_history(
    run_id: str,
    base_dir: Path | str = Path("experiments"),
) -> RunHistory:
    """Load ``history.json`` for a run."""
    root = Path(base_dir)
    p = _run_dir(run_id, root) / "history.json"
    if not p.is_file():
        return RunHistory.from_dict({})
    return RunHistory.from_dict(json.loads(p.read_text(encoding="utf-8")))


def load_metrics(
    run_id: str,
    base_dir: Path | str = Path("experiments"),
) -> dict[str, float]:
    """Load ``metrics.json`` for a run."""
    root = Path(base_dir)
    p = _run_dir(run_id, root) / "metrics.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)):
            out[str(k)] = float(v)
    return out


def compare_runs(
    run_ids: list[str],
    base_dir: Path | str = Path("experiments"),
) -> pd.DataFrame:
    """Build a table with one row per run and metric columns from ``metrics.json``.

    Always includes a ``run_id`` column.
    """
    rows: list[dict[str, Any]] = []
    for rid in run_ids:
        m = load_metrics(rid, base_dir=base_dir)
        row: dict[str, Any] = {"run_id": rid}
        row.update(m)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["run_id"])
    return pd.DataFrame(rows)
