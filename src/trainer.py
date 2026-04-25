from src.activation.activation import Array
from src.cost.cost import CostFunction
from src.optimizer.optimizer import Optimizer
from src.metric.metric import Metric
from src.config import ExperimentConfig
from src.network.multilayer_perceptron import MultilayerPerceptron


class Trainer:
    """Loop de entrenamiento genérico. No sabe nada de ejercicios ni de CSVs.

    Recibe un modelo, datos y config. Entrena y devuelve el historial.
    """

    def __init__(
        self,
        cost_fn: CostFunction,
        optimizer: Optimizer,
        metrics: list[Metric],
        cfg: ExperimentConfig,
    ) -> None:
        self.cost_fn = cost_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.cfg = cfg

    def fit(
        self,
        model: MultilayerPerceptron,
        X_train: Array,
        zeta_train: Array,
        X_val: Array,
        zeta_val: Array,
    ) -> dict:
        """Loop de épocas. Soporta modo online / batch / minibatch según cfg.training_mode.

        Returns:
            history: {"train_error": [...], "val_error": [...], "epochs": int}
        """
        raise NotImplementedError("TODO")

    def evaluate(self, model: MultilayerPerceptron, X: Array, zeta: Array) -> dict[str, float]:
        """Evalúa todas las métricas en cfg.metrics sobre (X, zeta)."""
        raise NotImplementedError("TODO")
