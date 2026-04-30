import numpy as np
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.activation.identity import IdentityActivation
from src.cost.mse import MSECost
from src.optimizer.gradient_descent import GradientDescent
from src.trainer import Trainer
from src.config import ExperimentConfig
from analysis.plots import plot_regression, plot_error_curve


def run() -> None:
    """Reproduce el ADALINE lineal del tutorial con la nueva arquitectura.

    Resultado esperado:
        w ≈ 2.0,  bias ≈ 5.0   ← el modelo recupera los coeficientes reales
    """

    X_flat = np.linspace(-5, 5, 50)
    zeta   = 2 * X_flat + 5
    X      = X_flat.reshape(-1, 1)

    modelo = MultilayerPerceptron([
        NeuronLayer(n_inputs=1, n_neurons=1, activation=IdentityActivation()),
    ])

    cfg = ExperimentConfig(
        name="tutorial_linear",
        seed=42,
        data_path="",
        target_column="",
        preprocessing="normalize",
        split_train=1.0,
        split_val=0.0,
        split_test=0.0,
        activation="identity",
        beta=1.0,
        architecture=[1, 1],
        cost_function="mse",
        optimizer="gradient_descent",
        eta=0.01,
        training_mode="online",
        epochs=20,
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
    print("=== ADALINE Lineal — verificación de arquitectura ===")
    print(f"Peso final:  w={layer.weights[0,0]:.4f}   (esperado ≈ 2.0)")
    print(f"Bias final:  w₀={layer.bias[0]:.4f}   (esperado ≈ 5.0)")
    print(f"Épocas:      {history['epochs']}")

    plot_regression(
        X, zeta, modelo,
        title="ADALINE Lineal — predicción vs datos",
        output_path="output/tutorial/linear/regression.png",
        xlim=(-6, 6),
        ylim=(-8, 18),
    )
    plot_error_curve(history, output_path="output/tutorial/linear/error_curve.png")
    print("Gráficos guardados en output/tutorial/linear/")
