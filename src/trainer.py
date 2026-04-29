import numpy as np
from src.activation.activation import Array
from src.cost.cost import CostFunction
from src.optimizer.optimizer import Optimizer
from src.metric.metric import Metric
from src.config import ExperimentConfig
from src.network.model import Model


class Trainer:
    """Loop de entrenamiento genérico. No sabe nada de ejercicios ni de CSVs."""

    def __init__(self, cost_fn: CostFunction, optimizer: Optimizer, metrics: list[Metric], cfg: ExperimentConfig) -> None:
        self.cost_fn = cost_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.cfg = cfg

    def fit(self, model: Model, X_train: Array, zeta_train: Array, X_val: Array, zeta_val: Array) -> dict:
        """Entrena el modelo y devuelve el historial de errores por época."""
        if self.cfg.training_mode == "online":
            train_fn = self._train_epoch_online
        elif self.cfg.training_mode == "minibatch":
            train_fn = self._train_epoch_minibatch
        elif self.cfg.training_mode == "batch":
            train_fn = self._train_epoch_batch
        else:
            raise ValueError(f"training_mode desconocido: {self.cfg.training_mode!r}")

        train_errors, val_errors = [], []

        for epoch in range(self.cfg.epochs):
            train_errors.append(train_fn(model, X_train, zeta_train))
            val_errors.append(self._evaluate_loss(model, X_val, zeta_val))

            if train_errors[-1] < self.cfg.epsilon:
                break

        return {"train_error": train_errors, "val_error": val_errors, "epochs": epoch + 1}

    def evaluate(self, model: Model, X: Array, zeta: Array) -> dict[str, float]:
        """Evaluación final — solo mide métricas, no toca los pesos."""
        predictions = np.array([model.forward(xi) for xi in X])
        return {metric.name(): metric.compute(zeta, predictions) for metric in self.metrics}

    def _train_epoch_online(self, model: Model, X: Array, zeta: Array) -> float:
        """Online: update después de cada muestra individual.

        for cada muestra:
            forward  → O    = model.forward(xi)
            backward → grad = cost.gradient(zi, O); model.backward(grad)
            update   → pesos actualizados con el gradiente de ESA muestra
            loss     → E    = cost(zi, O)
        """
        total_loss = 0.0
        for xi, zi in zip(X, zeta):
            O    = model.forward(xi)
            grad = self.cost_fn.gradient(zi, O)
            model.backward(grad)
            model.set_weights(self.optimizer.update(model.get_weights(), model.get_grads()))
            total_loss += self.cost_fn.compute(zi, O)
        return total_loss


    def _train_epoch_minibatch(self, model: Model, X: Array, zeta: Array) -> float:
        """Minibatch: acumula gradientes sobre B muestras, update una vez por batch.

        for cada batch de tamaño B:
            for cada muestra en el batch:
                forward  → O    = model.forward(xi)
                loss     → E   += cost(zi, O)
                backward → grad = cost.gradient(zi, O); model.backward(grad)  # acumula grads
            update   → pesos actualizados con el gradiente PROMEDIO del batch

        Requiere que NeuronLayer soporte acumulación de gradientes (zero_grads + +=).
        """
        raise NotImplementedError("TODO: minibatch — NeuronLayer.backward sobrescribe grad_W en cada llamada; implementar acumulación primero")

    def _train_epoch_batch(self, model: Model, X: Array, zeta: Array) -> float:
        """Batch: acumula gradientes sobre TODOS los datos, update una sola vez por época.

        for cada muestra en el dataset completo:
            forward  → O    = model.forward(xi)
            loss     → E   += cost(zi, O)
            backward → grad = cost.gradient(zi, O); model.backward(grad)  # acumula grads
        update   → pesos actualizados con el gradiente PROMEDIO de todos los datos

        Requiere que NeuronLayer soporte acumulación de gradientes (zero_grads + +=).
        """
        raise NotImplementedError("TODO: batch — NeuronLayer.backward sobrescribe grad_W en cada llamada; implementar acumulación primero")

    def _evaluate_loss(self, model: Model, X: Array, zeta: Array) -> float:
        """Mide la pérdida total sin tocar los pesos."""
        return sum(self.cost_fn.compute(zi, model.forward(xi)) for xi, zi in zip(X, zeta))
