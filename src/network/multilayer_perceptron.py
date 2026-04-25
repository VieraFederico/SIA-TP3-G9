from src.network.neuron_layer import NeuronLayer
from src.activation.activation import Array


class MultilayerPerceptron:
    """Lista de NeuronLayer. Implementa feed-forward y backpropagation.

    Un perceptrón simple es un MLP con una sola NeuronLayer de una neurona.
    """

    def __init__(self, layers: list[NeuronLayer]) -> None:
        self.layers = layers

    def forward(self, X: Array) -> Array:
        """Propaga X por todas las capas: x → V¹ → V² → ... → O"""
        raise NotImplementedError("TODO")

    def backward(self, grad_output: Array) -> None:
        """Retropropaga δ por todas las capas usando regla de la cadena."""
        raise NotImplementedError("TODO")

    def get_weights(self) -> list[tuple[Array, Array]]:
        """Retorna lista de (weights, bias) por capa."""
        raise NotImplementedError("TODO")

    def set_weights(self, weights: list[tuple[Array, Array]]) -> None:
        """Asigna pesos a todas las capas."""
        raise NotImplementedError("TODO")
