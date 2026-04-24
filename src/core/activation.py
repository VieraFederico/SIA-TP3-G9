from abc import ABC, abstractmethod

from .types import Array


class ActivationFunction(ABC):
    """Contract for all activation functions.

    Subclasses must implement ``forward``, ``derivative``, and may override
    ``is_differentiable``.  Every concrete activation must be stateless so it
    can be shared between layers.
    """

    @abstractmethod
    def forward(self, h: Array) -> Array:
        """Apply the activation function element-wise.

        Args:
            h: Pre-activation values ``h = W·x + b``, shape ``(...,)``.

        Returns:
            Activated values with the same shape as ``h``.
        """
        ...

    @abstractmethod
    def derivative(self, h: Array) -> Array:
        """Compute the element-wise derivative ``θ'(h)``.

        The derivative is always expressed **with respect to the
        pre-activation** ``h``, not the activation output.  This keeps
        backprop uniform regardless of the activation used.

        Args:
            h: Pre-activation values, same shape as in ``forward``.

        Returns:
            ``dθ/dh`` with the same shape as ``h``.
        """
        ...

    def is_differentiable(self) -> bool:
        """Return whether the function is differentiable everywhere.

        ``StepActivation`` overrides this to ``False``; all others default
        to ``True``.
        """
        return True
