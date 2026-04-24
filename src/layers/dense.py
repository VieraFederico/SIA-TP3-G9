from __future__ import annotations

import numpy as np

from src.core.activation import ActivationFunction
from src.core.layer import Layer
from src.core.types import Array


class DenseLayer(Layer):
    """Fully-connected layer with a learnable weight matrix and bias vector.

    Forward pass:
        h = x @ W + b          # pre-activation, shape (N, n_out)
        a = θ(h)               # post-activation, shape (N, n_out)

    Backward pass (chain rule):
        δ = grad_output ⊙ θ'(h)         # local gradient
        ∂L/∂W = x.T @ δ
        ∂L/∂b = Σ δ  (sum over samples)
        ∂L/∂x = δ @ W.T                 # upstream gradient

    Args:
        n_inputs: Number of input features.
        n_outputs: Number of neurons (output features).
        activation: Activation function θ to apply after the linear step.
        rng: NumPy random generator used for weight initialisation.
        initializer: Weight initialisation scheme — ``"xavier"``, ``"he"``,
            or ``"uniform"``.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        activation: ActivationFunction,
        rng: np.random.Generator,
        initializer: str = "xavier",
    ) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        self.rng = rng
        self.initializer = initializer

        # Parameters — initialised in _init_weights
        self.W: Array
        self.b: Array

        # Gradient accumulators
        self._dW: Array
        self._db: Array

        # Forward-pass cache needed for backward
        self._x_cache: Array | None = None
        self._h_cache: Array | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise W and b according to ``self.initializer``."""
        raise NotImplementedError("TODO")

    def forward(self, x: Array) -> Array:
        """h = x @ W + b ; a = θ(h)."""
        raise NotImplementedError("TODO")

    def backward(self, grad_output: Array) -> Array:
        """δ = grad_output ⊙ θ'(h) ; accumulate ∂L/∂W, ∂L/∂b ; return ∂L/∂x."""
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
