from src.activation.activation import ActivationFunction, Array


class Neuron:
    """Una neurona. Calcula h = Σ xᵢwᵢ + w₀, luego O = θ(h)."""

    def __init__(self, n_inputs: int, activation: ActivationFunction) -> None:
        self.n_inputs = n_inputs
        self.activation = activation
        self.weights: Array
        self.bias: float
        self._h: float

    def forward(self, x: Array) -> float:
        """Calcula h y O. Guarda h para backprop."""
        raise NotImplementedError("TODO")
