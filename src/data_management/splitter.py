from src.data_management.dataset import Dataset


def train_val_test_split(
    dataset: Dataset,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    """Divide dataset en train/val/test con proporciones dadas."""
    return dataset.split(train=train, val=val, test=test, seed=seed)

def k_fold_split(dataset: Dataset, k: int, seed: int) -> list[tuple[Dataset, Dataset]]:
    """Genera k particiones (train, val) para validación cruzada."""
    raise NotImplementedError("TODO")
