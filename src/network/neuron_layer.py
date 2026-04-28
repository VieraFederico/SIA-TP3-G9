import numpy as np
from src.activation.activation import ActivationFunction, Array


class NeuronLayer:
    """Capa de N neuronas. Versión vectorizada de una fila de Neurons.

    Guarda x, h y V durante el forward para usarlos en backprop.
    """

    def __init__(self, n_inputs: int, n_neurons: int, activation: ActivationFunction) -> None:
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation
        rng = np.random.default_rng()
        self.weights: Array = rng.standard_normal((n_inputs, n_neurons)) * 0.01
        self.bias: Array = np.zeros(n_neurons)
        self._x: Array = np.empty(n_inputs)
        self._h: Array = np.empty(n_neurons)
        self._V: Array = np.empty(n_neurons)
        self.grad_weights: Array = np.zeros((n_inputs, n_neurons))
        self.grad_bias: Array = np.zeros(n_neurons)

    def forward(self, x: Array) -> Array:
        """h = x·W + b,  V = θ(h). Guarda x, h y V para backprop."""
        self._x = x
        self._h = x @ self.weights + self.bias
        self._V = self.activation.compute(self._h)
        return self._V

    def backward(self, delta: Array) -> Array:
        """delta = ∂E/∂V (gradiente de la capa siguiente).

        Calcula ∂E/∂W y ∂E/∂b para esta capa, y devuelve ∂E/∂x
        para que la capa anterior pueda continuar la cadena.
        """
        if self.activation.is_differentiable():
            delta_h = delta * self.activation.derivative(self._h)  # ∂E/∂h = ∂E/∂V · θ'(h)
        else:
            delta_h = delta  # Rosenblatt: θ'(h) = 1 para activaciones no diferenciables
        self.grad_weights = np.outer(self._x, delta_h)          # ∂E/∂W = xᵀ · δh
        self.grad_bias = delta_h                                 # ∂E/∂b = δh
        return self.weights @ delta_h                            # ∂E/∂x → capa anterior

    def get_weights(self) -> tuple[Array, Array]:
        """Retorna (weights, bias)."""
        return (self.weights, self.bias)

    def set_weights(self, weights: Array, bias: Array) -> None:
        """Asigna pesos y bias."""
        self.weights = weights
        self.bias = bias
