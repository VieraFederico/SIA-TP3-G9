"""Map short ``-ex`` / ``-c`` flags to ``experiments_configs/.../*.json`` paths.

Exercise numbers follow the TP numbering (0 = validation scripts, 1–3 = exercises).
"""

from __future__ import annotations

from pathlib import Path

CONFIG_ROOT = "experiments_configs"

# Folder under CONFIG_ROOT for each -ex value
_EXERCISE_DIRS: dict[int, str] = {
    0: "validation",
    1: "ex1_fraud",
    2: "ex2_digits",
    3: "ex3_more_digits",
}

# -c value (normalized) -> JSON filename within that exercise folder
_CASE_FILES: dict[int, dict[str, str]] = {
    0: {
        "and": "and_step.json",
        "step": "and_step.json",
        "linear": "linear_y_eq_x.json",
        "nonlinear": "nonlinear_tanh.json",
        "tanh": "nonlinear_tanh.json",
        "xor": "xor_mlp.json",
    },
    1: {
        "linear": "fraud_linear.json",
        "nonlinear": "fraud_nonlinear.json",
    },
    2: {
        "baseline": "baseline.json",
        "adam": "adam_lr001.json",
        "adam001": "adam_lr001.json",
        "big": "arch_big.json",
        "arch_big": "arch_big.json",
    },
    3: {
        "tuned": "tuned.json",
        "default": "tuned.json",
        "more": "tuned.json",
    },
}


class RunCliError(ValueError):
    """Invalid -ex / -c combination or unknown alias."""


def normalize_case_name(case: str) -> str:
    """Lowercase, strip, turn hyphens into underscores for lookup."""
    s = case.strip().lower().replace("-", "_")
    # collapse double meanings
    if s == "non_linear":
        return "nonlinear"
    return s


# Canonical -c names shown in --help (each must exist in _CASE_FILES)
_CASE_HELP: dict[int, list[str]] = {
    0: ["and", "linear", "nonlinear", "xor"],
    1: ["linear", "nonlinear"],
    2: ["baseline", "adam", "big"],
    3: ["tuned"],
}


def list_cases_for_exercise(exercise: int) -> list[str]:
    """Display names for ``-c`` for a given ``-ex``."""
    if exercise not in _CASE_HELP:
        raise RunCliError(f"Unknown exercise: {exercise!r} (expect 0–3).")
    return list(_CASE_HELP[exercise])


def resolve_config_path(project_root: Path, exercise: int, case: str) -> Path:
    """Return absolute path to JSON for ``experiments_configs/<ex_dir>/<file>``."""
    if exercise not in _EXERCISE_DIRS:
        raise RunCliError(f"Exercise must be 0–3, got {exercise!r}.")
    key = normalize_case_name(case)
    table = _CASE_FILES[exercise]
    if key not in table:
        valid = ", ".join(sorted(set(table.keys())))
        raise RunCliError(
            f"Unknown case {case!r} for -ex {exercise}. Try one of: {valid}"
        )
    rel = Path(CONFIG_ROOT) / _EXERCISE_DIRS[exercise] / table[key]
    path = (project_root / rel).resolve()
    if not path.is_file():
        raise RunCliError(f"Config file missing on disk: {path}")
    return path


def describe_cli_help() -> str:
    """Epilog text for ``python run.py --help``."""
    lines = [
        "Quick runs (alternative to --config):",
        "  python run.py -ex 0 -c xor          # validation XOR MLP",
        "  python run.py -ex 1 -c nonlinear   # exercise 1 fraud (tanh)",
        "  python run.py -ex 2 -c baseline    # exercise 2 digits baseline",
        "  python run.py -ex 3 -c tuned        # exercise 3",
        "",
        "Exercise (-ex) → folder:",
    ]
    for n, d in sorted(_EXERCISE_DIRS.items()):
        lines.append(f"  {n} → {CONFIG_ROOT}/{d}/")
    lines.append("")
    lines.append("Cases (-c) per exercise:")
    for n in range(4):
        cases = list_cases_for_exercise(n)
        lines.append(f"  -ex {n}: {', '.join(cases)}")
    return "\n".join(lines)
