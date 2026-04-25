import argparse
from src.config import ExperimentConfig, load_config
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Construye el ArgumentParser con todos los flags del proyecto."""
    # --ejercicio es obligatorio (1, 2 o 3)
    # --config es obligatorio (path al JSON base)
    # Todos los demás campos de ExperimentConfig son opcionales y sobreescriben el JSON
    raise NotImplementedError("TODO")


def parse_args() -> tuple[int, ExperimentConfig]:
    """Parsea sys.argv. Carga el JSON base y sobreescribe con los flags CLI explícitos.

    Returns:
        (ejercicio: int, cfg: ExperimentConfig)
    """
    raise NotImplementedError("TODO")
