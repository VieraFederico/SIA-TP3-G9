"""Project-level helpers (CLI). Training utilities live in ``src.utils``."""

from utils.experiment_cli import (
    CONFIG_ROOT,
    RunCliError,
    describe_cli_help,
    list_cases_for_exercise,
    normalize_case_name,
    resolve_config_path,
)
from utils.run_parser import (
    build_parser,
    parse_run_config,
    resolve_config_from_args,
)

__all__ = [
    "CONFIG_ROOT",
    "RunCliError",
    "build_parser",
    "describe_cli_help",
    "list_cases_for_exercise",
    "normalize_case_name",
    "parse_run_config",
    "resolve_config_from_args",
    "resolve_config_path",
]
