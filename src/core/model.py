from abc import ABC, abstractmethod

from .types import Array


class Model(ABC):
    """Unified interface for all models (SimplePerceptron, MLP, …).

    ``Trainer`` depends exclusively on this ABC, so any concrete model can
    be swapped in without touching the training loop.
    """

    @abstractmethod
    def forward(self, X: Array) -> Array:
        """Run a forward pass over a batch.

        Args:
            X: Input matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predictions of shape ``(n_samples, n_outputs)``.
        """
        ...

    @abstractmethod
    def backward(self, grad_output: Array) -> None:
        """Run a backward pass given the loss gradient w.r.t. outputs.

        Populates parameter gradients inside each layer.

        Args:
            grad_output: ``∂L/∂output``, shape ``(n_samples, n_outputs)``.
        """
        ...

    @abstractmethod
    def predict(self, X: Array) -> Array:
        """Return predictions without updating internal state.

        Equivalent to ``forward`` but may apply thresholding / argmax for
        classification.  Must not trigger gradient accumulation.

        Args:
            X: Input matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predictions of shape ``(n_samples, n_outputs)``.
        """
        ...

    @abstractmethod
    def get_params(self) -> list[Array]:
        """Return all trainable parameters as a flat list.

        Order must match ``get_grads`` and ``set_params``.
        """
        ...

    @abstractmethod
    def set_params(self, params: list[Array]) -> None:
        """Overwrite all trainable parameters.

        Args:
            params: New parameter arrays in the same order as ``get_params``.
        """
        ...

    @abstractmethod
    def get_grads(self) -> list[Array]:
        """Return gradients from the last backward pass as a flat list.

        Order matches ``get_params``.
        """
        ...

    @abstractmethod
    def zero_grad(self) -> None:
        """Zero all accumulated parameter gradients.

        Must be called before each new forward/backward cycle.
        """
        ...
