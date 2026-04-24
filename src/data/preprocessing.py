from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.types import Array


class Preprocessor(ABC):
    """Base class for all data preprocessors.

    Follows the fit/transform/fit_transform interface familiar from
    scikit-learn, implemented with NumPy only.
    """

    @abstractmethod
    def fit(self, X: Array) -> Preprocessor:
        """Compute statistics from training data.

        Args:
            X: Training feature matrix, shape ``(n_samples, n_features)``.

        Returns:
            ``self`` for method chaining.
        """
        ...

    @abstractmethod
    def transform(self, X: Array) -> Array:
        """Apply the transformation.

        Args:
            X: Feature matrix to transform.

        Returns:
            Transformed feature matrix, same shape as ``X``.
        """
        ...

    def fit_transform(self, X: Array) -> Array:
        """Fit then transform in one step."""
        return self.fit(X).transform(X)


class StandardScaler(Preprocessor):
    """Standardise features to zero mean and unit variance.

    X_scaled = (X − μ) / σ

    Stores ``mean_`` and ``std_`` after ``fit``.
    """

    def __init__(self) -> None:
        self.mean_: Array | None = None
        self.std_: Array | None = None

    def fit(self, X: Array) -> StandardScaler:
        """Compute per-feature mean and std from X."""
        raise NotImplementedError("TODO")

    def transform(self, X: Array) -> Array:
        """(X − mean_) / std_"""
        raise NotImplementedError("TODO")


class Normalizer(Preprocessor):
    """Scale features to the range [0, 1].

    X_norm = (X − X_min) / (X_max − X_min)
    """

    def __init__(self) -> None:
        self.min_: Array | None = None
        self.max_: Array | None = None

    def fit(self, X: Array) -> Normalizer:
        """Compute per-feature min and max from X."""
        raise NotImplementedError("TODO")

    def transform(self, X: Array) -> Array:
        """(X − min_) / (max_ − min_)"""
        raise NotImplementedError("TODO")


class OneHotEncoder:
    """Encode integer class labels as one-hot vectors.

    Args:
        n_classes: Total number of classes.  If ``None``, inferred from data
            during ``fit``.
    """

    def __init__(self, n_classes: int | None = None) -> None:
        self.n_classes = n_classes
        self.classes_: Array | None = None

    def fit(self, y: Array) -> OneHotEncoder:
        """Discover unique classes from y.

        Args:
            y: Integer label vector, shape ``(n_samples,)``.
        """
        raise NotImplementedError("TODO")

    def transform(self, y: Array) -> Array:
        """Convert integer labels to one-hot matrix.

        Args:
            y: Integer label vector, shape ``(n_samples,)``.

        Returns:
            One-hot matrix, shape ``(n_samples, n_classes)``.
        """
        raise NotImplementedError("TODO")

    def inverse_transform(self, Y: Array) -> Array:
        """Convert one-hot matrix back to integer labels.

        Args:
            Y: One-hot matrix, shape ``(n_samples, n_classes)``.

        Returns:
            Integer label vector, shape ``(n_samples,)``.
        """
        raise NotImplementedError("TODO")
