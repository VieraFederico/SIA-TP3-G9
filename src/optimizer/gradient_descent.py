from src.optimizer.optimizer import Optimizer
from src.activation.activation import Array


class GradientDescent(Optimizer):
    """Descenso de gradiente estándar. Δw = -η · ∂E/∂w"""

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def update(
        self,
        params: list[tuple[Array, Array]],
        grads:  list[tuple[Array, Array]],
    ) -> list[tuple[Array, Array]]:
        return [
            (w - self.learning_rate * gw, b - self.learning_rate * gb)
            for (w, b), (gw, gb) in zip(params, grads)
        ]

    def reset(self) -> None:
        pass
