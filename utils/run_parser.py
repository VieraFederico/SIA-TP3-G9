"""Argparse and config-path resolution for ``run.py`` (keep CLI logic out of the root script)."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.experiment_cli import RunCliError, describe_cli_help, resolve_config_path


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    epilog = describe_cli_help()
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train a model from experiments_configs (short -ex/-c or explicit --config).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a full experiment JSON. If set, -ex/-c are ignored.",
    )
    parser.add_argument(
        "-ex",
        "--exercise",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        metavar="N",
        help="Exercise folder: 0=validation, 1=fraud, 2=digits, 3=more_digits.",
    )
    parser.add_argument(
        "-c",
        "--case",
        type=str,
        default=None,
        metavar="NAME",
        help="Named preset within that exercise (see --help epilog).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override the auto-generated run ID (timestamp).",
    )
    return parser


def resolve_config_from_args(project_root: Path, args: argparse.Namespace) -> Path:
    """Resolve final config path from parsed CLI args."""
    if args.config is not None:
        p = Path(args.config)
        if not p.is_file():
            raise RunCliError(f"Config file not found: {p}")
        return p.resolve()

    if (args.exercise is None) != (args.case is None):
        raise RunCliError(
            "Use both -ex N and -c NAME together, or pass --config PATH "
            "(see python run.py --help)."
        )
    if args.exercise is not None and args.case is not None:
        return resolve_config_path(project_root, args.exercise, args.case)

    raise RunCliError(
        "Either pass --config PATH, or both -ex N and -c NAME (see python run.py --help)."
    )


def parse_run_config(
    project_root: Path,
    argv: list[str] | None = None,
    *,
    prog: str | None = None,
) -> tuple[Path, str | None]:
    """Parse CLI and return ``(config_path, run_id)``. Calls ``parser.error`` on failure."""
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        config_path = resolve_config_from_args(project_root, args)
    except RunCliError as e:
        parser.error(str(e))
    return config_path, args.run_id
