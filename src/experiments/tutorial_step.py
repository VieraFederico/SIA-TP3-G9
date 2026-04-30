import numpy as np
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.activation.step import StepActivation
from src.cost.mse import MSECost
from src.optimizer.gradient_descent import GradientDescent
from src.trainer import Trainer
from src.config import ExperimentConfig
from analysis.plots import plot_decision_boundary, plot_error_curve


def run() -> None:
    """Reproduce la compuerta AND del tutorial con la nueva arquitectura.

    Resultado esperado:
        Predicciones: [0, 0, 0, 1]   ← tabla de verdad del AND
    """

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)
    zeta = np.array([0, 0, 0, 1], dtype=float)

    modelo = MultilayerPerceptron([
        NeuronLayer(n_inputs=2, n_neurons=1, activation=StepActivation()),
    ])

    cfg = ExperimentConfig(
        name="tutorial_and",
        seed=42,
        data_path="",
        target_column="",
        preprocessing="normalize",
        split_train=1.0,
        split_val=0.0,
        split_test=0.0,
        activation="step",
        beta=1.0,
        architecture=[2, 1],
        cost_function="mse",
        optimizer="gradient_descent",
        eta=0.1,
        training_mode="online",
        epochs=50,
        epsilon=0.0,
    )

    trainer = Trainer(
        cost_fn=MSECost(),
        optimizer=GradientDescent(learning_rate=cfg.eta),
        metrics=[],
        cfg=cfg,
    )

    history = trainer.fit(
        modelo,
        X_train=X, zeta_train=zeta,
        X_val=X,   zeta_val=zeta,
    )

    layer = modelo.layers[0]
    predictions = [int(modelo.forward(xi).item()) for xi in X]

    print("=== Compuerta AND — verificación de arquitectura ===")
    print(f"Predicciones: {predictions}   (esperado: [0, 0, 0, 1])")
    print(f"Pesos finales: w₁={layer.weights[0,0]:.3f}, w₂={layer.weights[1,0]:.3f}")
    print(f"Bias final:    w₀={layer.bias[0]:.3f}")
    print(f"Épocas:        {history['epochs']}")
    print(f"\n{'✓' if predictions == [0, 0, 0, 1] else '✗'} Arquitectura {'OK' if predictions == [0, 0, 0, 1] else 'con errores — revisar forward/backward/optimizer'}")

    plot_decision_boundary(
        X, zeta, modelo,
        title="Compuerta AND — frontera de decisión",
        output_path="output/tutorial/step/tutorial_and_boundary.png",
    )
    plot_error_curve(history, output_path="output/tutorial/step/tutorial_and_error.png")
    print("Gráficos guardados en output/tutorial/step/")
