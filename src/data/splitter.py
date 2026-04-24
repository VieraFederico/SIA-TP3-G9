from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.core.types import Array

from .dataset import Dataset


class Splitter(ABC):
    """Base class for dataset splitting strategies."""

    @abstractmethod
    def split(self, dataset: Dataset) -> list[tuple[Dataset, Dataset]]:
        """Return a list of (train, val/test) Dataset pairs.

        For ``HoldOutSplitter`` this returns a single-element list;
        for ``KFoldSplitter`` it returns ``k`` pairs.
        """
        ...


class HoldOutSplitter(Splitter):
    """Split a dataset into train / validation / test subsets.

    Args:
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for test.  Derived from remaining
            data if ``None``.
        rng: NumPy random generator for reproducible shuffling.
    """

    def __init__(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rng = rng

    def split(self, dataset: Dataset) -> list[tuple[Dataset, Dataset]]:
        """Return [(train_dataset, val_dataset)]; test split is separate.

        Returns:
            Single-element list containing ``(train, val)`` datasets.
        """
        raise NotImplementedError("TODO")

    def split_three_way(
        self, dataset: Dataset
    ) -> tuple[Dataset, Dataset, Dataset]:
        """Return ``(train, val, test)`` datasets.

        Returns:
            Tuple of three Dataset objects.
        """
        raise NotImplementedError("TODO")


class KFoldSplitter(Splitter):
    """K-Fold cross-validation splitter.

    Args:
        k: Number of folds.
        shuffle: Whether to shuffle before splitting.
        rng: NumPy random generator for reproducible shuffling.
    """

    def __init__(
        self,
        k: int,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.k = k
        self.shuffle = shuffle
        self.rng = rng

    def split(self, dataset: Dataset) -> list[tuple[Dataset, Dataset]]:
        """Return k ``(train, val)`` Dataset pairs.

        Returns:
            List of k ``(train_dataset, val_dataset)`` tuples.
        """
        raise NotImplementedError("TODO")
