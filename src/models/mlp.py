from __future__ import annotations

import numpy as np

from src.core.activation import ActivationFunction
from src.core.model import Model
from src.core.types import Array
from src.layers.dense import DenseLayer


class MultilayerPerceptron(Model):
    """Feed-forward fully-connected neural network.

    Composed of a sequence of ``DenseLayer`` objects.  Forward pass runs
    them left-to-right; backward pass runs them right-to-left, chaining
    ``∂L/∂input`` of each layer into ``grad_output`` of the previous one.

    Args:
        layer_sizes: List of integers ``[n_inputs, h1, h2, …, n_outputs]``.
            E.g. ``[784, 64, 10]`` → one hidden layer of 64 units.
        hidden_activation: Activation applied to every hidden layer.
        output_activation: Activation applied to the output layer.
        rng: NumPy random generator passed to each ``DenseLayer``.
        initializer: Weight init scheme forwarded to each layer.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        hidden_activation: ActivationFunction,
        output_activation: ActivationFunction,
        rng: np.random.Generator,
        initializer: str = "xavier",
    ) -> None:
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.rng = rng
        self.initializer = initializer
        self.layers: list[DenseLayer] = []
        self._build_layers()

    def _build_layers(self) -> None:
        """Instantiate DenseLayer objects from layer_sizes."""
        raise NotImplementedError("TODO")

    def forward(self, X: Array) -> Array:
        """Pass X through all layers sequentially."""
        raise NotImplementedError("TODO")

    def backward(self, grad_output: Array) -> None:
        """Backpropagate grad_output through all layers in reverse."""
        raise NotImplementedError("TODO")

    def predict(self, X: Array) -> Array:
        """Run forward pass and apply argmax for classification."""
        raise NotImplementedError("TODO")

    def get_params(self) -> list[Array]:
        """Return concatenated parameters from all layers."""
        raise NotImplementedError("TODO")

    def set_params(self, params: list[Array]) -> None:
        """Distribute flat params list across layers."""
        raise NotImplementedError("TODO")

    def get_grads(self) -> list[Array]:
        """Return concatenated gradients from all layers."""
        raise NotImplementedError("TODO")

    def zero_grad(self) -> None:
        """Zero gradients in all layers."""
        raise NotImplementedError("TODO")
