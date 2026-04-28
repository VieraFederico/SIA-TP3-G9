import numpy as np
from src.activation.activation import Array
from src.cost.cost import CostFunction
from src.optimizer.optimizer import Optimizer
from src.metric.metric import Metric
from src.config import ExperimentConfig
from src.network.model import Model


class Trainer:
    """Loop de entrenamiento genérico. No sabe nada de ejercicios ni de CSVs."""

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

    # ── INTERFAZ PÚBLICA ─────────────────────────────────────────────────

    def fit(self, model: Model, X_train: Array, zeta_train: Array, X_val: Array, zeta_val: Array) -> dict:
        """Entrena el modelo y devuelve el historial de errores por época.

        X_train / zeta_train — datos con los que el modelo aprende (ajusta pesos).
        X_val   / zeta_val   — datos que el modelo nunca ve durante el aprendizaje.
        """
        train_errors, val_errors = [], []

        for epoch in range(self.cfg.epochs):
            train_errors.append(self._train_epoch(model, X_train, zeta_train))
            val_errors.append(self._evaluate_loss(model, X_val, zeta_val))

            if train_errors[-1] < self.cfg.epsilon:
                break

        return {"train_error": train_errors, "val_error": val_errors, "epochs": epoch + 1}

    def evaluate(self, model: Model, X: Array, zeta: Array) -> dict[str, float]:
        """Evaluación final — solo mide métricas, no toca los pesos."""
        predictions = np.array([model.forward(xi) for xi in X])
        return {metric.name(): metric.compute(zeta, predictions) for metric in self.metrics}

    # ── EPOCH ────────────────────────────────────────────────────────────

    def _train_epoch(self, model: Model, X: Array, zeta: Array) -> float:
        """Una época de entrenamiento online: muestra a muestra."""
        total_loss = 0.0
        for xi, zi in zip(X, zeta):
            total_loss += self._train_step(model, xi, zi)
        return total_loss

    def _evaluate_loss(self, model: Model, X: Array, zeta: Array) -> float:
        """Mide la pérdida total sin tocar los pesos."""
        return sum(self.cost_fn.compute(zi, model.forward(xi)) for xi, zi in zip(X, zeta))

    # ── PASO DE ENTRENAMIENTO ────────────────────────────────────────────

    def _train_step(self, model: Model, xi: Array, zi: Array) -> float:
        """Un paso completo de aprendizaje para una sola muestra.

        forward  → calcula la salida del modelo
        loss     → mide qué tan lejos está de la salida esperada
        backward → propaga el error hacia atrás (calcula gradientes)
        update   → ajusta los pesos según los gradientes
        """
        O    = model.forward(xi)
        loss = self.cost_fn.compute(zi, O)
        grad = self.cost_fn.gradient(zi, O)
        model.backward(grad)
        self._update_weights(model)
        return loss

    def _update_weights(self, model: Model) -> None:
        """Aplica el optimizer y escribe los pesos actualizados en el modelo."""
        updated = self.optimizer.update(model.get_weights(), model.get_grads())
        model.set_weights(updated)
