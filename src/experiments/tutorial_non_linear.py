import numpy as np
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.activation.tanh import TanhActivation
from src.cost.mse import MSECost
from src.optimizer.gradient_descent import GradientDescent
from src.trainer import Trainer
from src.config import ExperimentConfig
from analysis.plots import plot_regression, plot_error_curve


def run() -> None:
    """Reproduce el perceptrón tanh del tutorial con la nueva arquitectura.

    Resultado esperado:
        La curva predicha se superpone con tanh(x) en [-5, 5]
    """

    X_flat = np.linspace(-5, 5, 50)
    zeta   = np.tanh(X_flat)
    X      = X_flat.reshape(-1, 1)

    modelo = MultilayerPerceptron([
        NeuronLayer(n_inputs=1, n_neurons=1, activation=TanhActivation(beta=0.4)),
    ])

    cfg = ExperimentConfig(
        name="tutorial_non_linear",
        seed=42,
        data_path="",
        target_column="",
        preprocessing="normalize",
        split_train=1.0,
        split_val=0.0,
        split_test=0.0,
        activation="tanh",
        beta=0.4,
        architecture=[1, 1],
        cost_function="mse",
        optimizer="gradient_descent",
        eta=0.15,
        training_mode="online",
        epochs=100,
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
    print("=== Perceptrón Tanh — verificación de arquitectura ===")
    print(f"Peso final:  w={layer.weights[0,0]:.4f}")
    print(f"Bias final:  w₀={layer.bias[0]:.4f}")
    print(f"Épocas:      {history['epochs']}")

    plot_regression(
        X, zeta, modelo,
        title="Perceptrón Tanh — predicción vs tanh(x)",
        output_path="output/tutorial/non_linear/regression.png",
        xlim=(-6, 6),
        ylim=(-1.5, 1.5),
    )
    plot_error_curve(history, output_path="output/tutorial/non_linear/error_curve.png")
    print("Gráficos guardados en output/tutorial/non_linear/")
