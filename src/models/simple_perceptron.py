from __future__ import annotations

import numpy as np

from src.core.activation import ActivationFunction
from src.core.model import Model
from src.core.types import Array


class SimplePerceptron(Model):
    """Single-layer perceptron with an injectable activation function.

    Supports step (classic Rosenblatt), identity (ADALINE / linear
    regression), and non-linear activations (tanh, logistic).

    Gradient update for differentiable activations follows the delta rule:
        Δw = η · δ · x    where  δ = (y − ŷ) · θ'(h)

    For the step activation the update uses the sign of the error directly
    (no derivative involved).

    Args:
        n_inputs: Dimensionality of the input vector.
        activation: ActivationFunction to use at the output.
        rng: NumPy random generator for weight initialisation.
        initializer: Weight init scheme — ``"xavier"``, ``"he"``, or
            ``"uniform"``.
    """

    def __init__(
        self,
        n_inputs: int,
        activation: ActivationFunction,
        rng: np.random.Generator,
        initializer: str = "xavier",
    ) -> None:
        self.n_inputs = n_inputs
        self.activation = activation
        self.rng = rng
        self.initializer = initializer

        self.W: Array
        self.b: Array
        self._dW: Array
        self._db: Array

        # Cache from last forward pass
        self._x_cache: Array | None = None
        self._h_cache: Array | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise W and b according to ``self.initializer``."""
        raise NotImplementedError("TODO")

    def forward(self, X: Array) -> Array:
        """h = X @ W + b ; output = θ(h)."""
        raise NotImplementedError("TODO")

    def backward(self, grad_output: Array) -> None:
        """Accumulate ∂L/∂W and ∂L/∂b from grad_output."""
        raise NotImplementedError("TODO")

    def predict(self, X: Array) -> Array:
        """Forward pass without side-effects (no cache update)."""
        raise NotImplementedError("TODO")

    def get_params(self) -> list[Array]:
        """Return [W, b]."""
        raise NotImplementedError("TODO")

    def set_params(self, params: list[Array]) -> None:
        """Set [W, b] from params list."""
        raise NotImplementedError("TODO")

    def get_grads(self) -> list[Array]:
        """Return [dW, db]."""
        raise NotImplementedError("TODO")

    def zero_grad(self) -> None:
        """Zero gradient accumulators."""
        raise NotImplementedError("TODO")
