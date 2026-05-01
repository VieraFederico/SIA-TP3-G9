from src.config import ExperimentConfig
from src.data_management.loader import load_csv
from src.data_management.preprocessing import standardize, normalize


def run(cfg: ExperimentConfig) -> None:

    target_columns = cfg.target_column