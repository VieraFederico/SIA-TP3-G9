import numpy as np
from src.data_management.dataset import Dataset


def train_val_test_split(
    dataset: Dataset,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    """Divide dataset en train/val/test con proporciones dadas."""
    total = train + val + test
    if not np.isclose(total, 1.0):
        raise ValueError(f"train+val+test must be 1.0, got {total}")

    n = len(dataset.X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(n * train)
    n_val = int(n * val)
    # n_test = n - n_train - n_val (rest)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = Dataset(X=dataset.X[train_idx], zeta=dataset.zeta[train_idx])
    val_ds = Dataset(X=dataset.X[val_idx], zeta=dataset.zeta[val_idx])
    test_ds = Dataset(X=dataset.X[test_idx], zeta=dataset.zeta[test_idx])

    return train_ds, val_ds, test_ds

def k_fold_split(dataset: Dataset, k: int, seed: int) -> list[tuple[Dataset, Dataset]]:
    """Genera k particiones (train, val) para validación cruzada."""
    raise NotImplementedError("TODO")
