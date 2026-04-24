from __future__ import annotations

from pathlib import Path

from src.core.types import Array

from .dataset import Dataset


class CSVDatasetLoader:
    """Load a dataset from a CSV file using Pandas.

    Args:
        path: Path to the CSV file.
        target_column: Name of the column to use as ``y``.
        feature_columns: Optional explicit list of feature column names.
            If ``None``, all columns except ``target_column`` are used.
        dtype: NumPy dtype for the resulting arrays.  Defaults to
            ``float64``.
    """

    def __init__(
        self,
        path: Path | str,
        target_column: str,
        feature_columns: list[str] | None = None,
        dtype: str = "float64",
    ) -> None:
        self.path = Path(path)
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.dtype = dtype

    def load(self) -> Dataset:
        """Read the CSV and return a ``Dataset``.

        Returns:
            Dataset with ``X`` (features) and ``y`` (targets) as NumPy arrays.

        Raises:
            FileNotFoundError: If ``self.path`` does not exist.
            KeyError: If ``target_column`` is not found in the CSV.
        """
        raise NotImplementedError("TODO")
