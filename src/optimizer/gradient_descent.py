from src.optimizer.optimizer import Optimizer
from src.activation.activation import Array


class GradientDescent(Optimizer):
    """Descenso de gradiente estándar. Δw = -η · ∂E/∂w"""

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def update(self, weights: list[Array], grads: list[Array]) -> list[Array]:
        """Δw = -η · ∂E/∂w,
        no olvidemos que la gradient (la derivada de E en funcion de w)
        Es una suma E = (1/2N) · Σ (ζ - O)²"""
        updated = []
        for w, g in zip(weights, grads):
            updated.append(w - self.learning_rate * g)
        return updated

    def reset(self) -> None:
        pass #Basicamente un continue de Java, Gradient Descent es medio Dummy, no tiene estado interno
