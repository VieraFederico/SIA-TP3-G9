from abc import ABC, abstractmethod
from src.activation.activation import Array


class Optimizer(ABC):
    """Regla de actualización de pesos. Implementa Δw = f(∂E/∂w)."""

    @abstractmethod
    def update(self, weights: list[Array], grads: list[Array]) -> list[Array]:
        """Aplica la regla Δw y devuelve los pesos actualizados."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Limpia el estado interno (velocidades en Momentum, momentos en Adam)."""
        ...
