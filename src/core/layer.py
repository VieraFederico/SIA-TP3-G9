from abc import ABC, abstractmethod

from .types import Array


class Layer(ABC):
    """Contract for a single computational layer inside a model.

    A layer encapsulates *parameters* (weights, biases) and the forward /
    backward logic.  It must cache whatever intermediate values are needed
    for the backward pass during ``forward``.

    Layers are the unit of composition for the MLP: a model is a sequence
    of ``Layer`` objects.
    """

    @abstractmethod
    def forward(self, x: Array) -> Array:
        """Compute the layer output and cache values needed for backprop.

        Args:
            x: Input array of shape ``(n_samples, n_inputs)``.

        Returns:
            Output array of shape ``(n_samples, n_outputs)``.
        """
        ...

    @abstractmethod
    def backward(self, grad_output: Array) -> Array:
        """Backpropagate gradients through the layer.

        Accumulates parameter gradients internally (retrievable via
        ``get_grads``) and returns the gradient to propagate upstream.

        Args:
            grad_output: ``∂L/∂output``, shape ``(n_samples, n_outputs)``.

        Returns:
            ``∂L/∂input``, shape ``(n_samples, n_inputs)``.
        """
        ...

    @abstractmethod
    def get_params(self) -> list[Array]:
        """Return a list of all trainable parameter arrays.

        The order must match the order returned by ``get_grads`` and
        accepted by ``set_params``.
        """
        ...

    @abstractmethod
    def set_params(self, params: list[Array]) -> None:
        """Overwrite trainable parameters in-place.

        Args:
            params: New parameter arrays in the same order as ``get_params``.
        """
        ...

    @abstractmethod
    def get_grads(self) -> list[Array]:
        """Return gradients accumulated during the last ``backward`` call.

        The order matches ``get_params``.
        """
        ...

    def zero_grad(self) -> None:
        """Reset accumulated gradients to zero.

        Called at the start of each forward/backward cycle.  Default
        implementation zeros every array returned by ``get_grads``; override
        if the layer manages gradients differently.
        """
        for g in self.get_grads():
            g[:] = 0.0
