from src.data_management.dataset import Dataset
import pandas as pd


def load_csv(path: str, target_columns: list[str]) -> Dataset:
    """Lee un CSV y devuelve un Dataset con X y zeta (varias columnas target)."""

    df = pd.read_csv(path)

    X = df.drop(columns=target_columns).values.astype(float)
    zeta = df[target_columns].values.astype(float)

    return Dataset(X=X, zeta=zeta)