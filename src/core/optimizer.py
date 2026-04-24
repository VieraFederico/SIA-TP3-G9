from abc import ABC, abstractmethod

from .types import Array


class Optimizer(ABC):
    """Contract for all parameter-update strategies.

    Each optimizer owns its internal state (e.g. velocity buffers for
    Momentum, first/second moment estimates for Adam).  The state is keyed
    by the index in the ``params`` list so callers do not need to pass
    names.

    The optimizer is intentionally decoupled from the model: it only sees
    flat lists of parameter arrays and their matching gradients.
    """

    @abstractmethod
    def update(self, params: list[Array], grads: list[Array]) -> list[Array]:
        """Apply one update step and return the new parameter arrays.

        Args:
            params: Current parameter tensors (weights, biases, …).
            grads: Matching gradients ``∂L/∂param`` for each parameter.

        Returns:
            Updated parameter tensors in the same order as ``params``.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state (velocities, moment estimates, step counter).

        Call this when reusing the same optimizer object across multiple
        training runs.
        """
        ...
