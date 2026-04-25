from src.activation.activation import ActivationFunction, Array


class NeuronLayer:
    """Capa de N neuronas. Versión vectorizada de una fila de Neurons.

    Guarda h y V durante el forward para usarlos en backprop.
    """

    def __init__(self, n_inputs: int, n_neurons: int, activation: ActivationFunction) -> None:
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation
        self.weights: Array
        self.bias: Array
        self._h: Array
        self._V: Array

    def forward(self, x: Array) -> Array:
        """h = X·W + b,  V = θ(h). Guarda h y V."""
        raise NotImplementedError("TODO")

    def backward(self, delta: Array) -> Array:
        """Calcula delta_w y delta para la capa anterior."""
        raise NotImplementedError("TODO")

    def get_weights(self) -> tuple[Array, Array]:
        """Retorna (weights, bias)."""
        raise NotImplementedError("TODO")

    def set_weights(self, weights: Array, bias: Array) -> None:
        """Asigna pesos y bias."""
        raise NotImplementedError("TODO")
