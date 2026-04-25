from abc import ABC, abstractmethod
from src.activation.activation import Array


class Metric(ABC):
    """Métrica de evaluación del modelo."""

    @abstractmethod
    def compute(self, zeta: Array, O: Array) -> float:
        """Calcula la métrica comparando salida esperada ζ contra obtenida O."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Nombre de la métrica para logging."""
        ...
