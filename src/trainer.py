import numpy as np
from src.activation.activation import Array
from src.cost.cost import CostFunction
from src.optimizer.optimizer import Optimizer
from src.metric.metric import Metric
from src.config import ExperimentConfig
from src.network.model import Model


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

    def fit( self, model: Model, X_train: Array, zeta_train: Array, X_val: Array, zeta_val: Array) -> dict:
        """Loop de épocas. Soporta modo online / batch / minibatch según cfg.training_mode.

        X_train, zeta_train — datos con los que el modelo APRENDE (ajusta pesos)
        X_val,   zeta_val   — datos que el modelo NUNCA ve durante el aprendizaje, solo los usamos para vigilar si hay overfitting

        Returns:
            history: {"train_error": [...], "val_error": [...], "epochs": int}
        """
        train_errors = []   # E de train por época — debería bajar
        val_errors   = []   # E de val por época — si sube mientras train baja: overfitting

        for epoch in range(self.cfg.epochs):

            # ── ENTRENAMIENTO ─────────────────────────────────────────────
            # Recorre todos los datos muestra por muestra y ajusta los pesos.
            # Equivale exactamente al for xi, target in zip(X, y) del tutorial.
            epoch_error = self._run_online(model, X_train, zeta_train)
            train_errors.append(epoch_error)

            # ── VALIDACIÓN ────────────────────────────────────────────────
            # Solo mide E sobre val — NO llama a backward ni al optimizer.
            # Los pesos no cambian acá. Sirve para detectar overfitting.
            val_error = 0.0
            for xi, zi in zip(X_val, zeta_val):
                O = model.forward(xi)                     # O = θ(h) — solo predice
                val_error += self.cost_fn.compute(zi, O)  # E = (1/2)·(ζ-O)²
            val_errors.append(val_error)

            # ── CONVERGENCIA ──────────────────────────────────────────────
            # Si el error bajó por debajo de epsilon paramos antes de agotar épocas.
            if epoch_error < self.cfg.epsilon:
                break

        return {
            "train_error": train_errors,
            "val_error":   val_errors,
            "epochs":      epoch + 1,
        }

    def _run_online(self, model: Model, X: Array, zeta: Array) -> float:
        """Modo online: ajusta los pesos después de CADA muestra individual.

        Equivale al fit() del tutorial de Juan:
            update = self.lr * (target - y_pred)
            self.weights += update * xi
            self.bias    += update
        Acá esa lógica está repartida en backward() y _apply_optimizer().
        """
        total_error = 0.0

        for xi, zi in zip(X, zeta):
            # PASO 1: forward — calcula h = Σxᵢwᵢ + w₀, luego O = θ(h) (Cuando hagamos el MLP no va a ser exactamenete ese h pero se entiende) (espero)
            O = model.forward(xi)

            # PASO 2: mide E = (1/2)·(ζ-O)² para esta muestra
            total_error += self.cost_fn.compute(zi, O)

            # PASO 3: calcula ∂E/∂O = -(ζ-O) — punto de entrada del gradiente
            grad = self.cost_fn.gradient(zi, O)

            # PASO 4: backward — calcula δ = ∂E/∂O · θ'(h), luego grad_weights = δ·x
            # NO actualiza pesos todavía, solo los guarda en model.grad_weights/grad_bias
            model.backward(grad)

            # PASO 5: optimizer aplica η — w = w - learning_rate · grad_weights
            self._apply_optimizer(model)

        return total_error

    def _apply_optimizer(self, model: Model) -> None:
        """Puente entre el modelo y el optimizer.

        El optimizer solo habla en listas de arrays — no sabe cuántas capas hay.
        El Trainer aplana (W, b) de todas las capas, llama al optimizer, y
        devuelve los pesos actualizados al modelo capa por capa.
        """
        layer_params = model.get_weights()
        layer_grads  = model.get_grads()

        weights = [p for wb in layer_params for p in wb]
        grads   = [g for gb in layer_grads  for g in gb]

        updated = self.optimizer.update(weights, grads)

        it = iter(updated)
        model.set_weights([(next(it), next(it)) for _ in layer_params])

    def evaluate(self, model: Model, X: Array, zeta: Array) -> dict[str, float]:
        """Evaluación final — solo mide, NO toca los pesos."""
        predictions = np.array([model.forward(xi) for xi in X])
        results = {}
        for metric in self.metrics:
            results[metric.name()] = metric.compute(zeta, predictions)
        return results
