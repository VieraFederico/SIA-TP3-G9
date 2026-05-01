import numpy as np

from src.data_management.preprocessing import normalize, normalize_with_params
from src.data_management.loader import load_csv
from src.network.multilayer_perceptron import MultilayerPerceptron
from src.network.neuron_layer import NeuronLayer
from src.activation.identity import IdentityActivation
from src.activation.tanh import TanhActivation
from src.activation.step import StepActivation
from src.cost.mse import MSECost
from src.optimizer.gradient_descent import GradientDescent
from src.trainer import Trainer
from src.config import ExperimentConfig
from analysis.plots import plot_regression, plot_error_curve

def run(cfg: ExperimentConfig) -> None:
    """Ejercicio 1 — Detección de fraude.

    Entrena perceptrón simple lineal vs no lineal. Compara generalización.
    """
    #TODO: LEER ESTO. para el ejercicio 1 viendo el dataset, uno puede observar un par de patrones:
    # 1. Las compras que suelen ser de mayor valor, suelen ser fraudulentas.
    # 2. Las compras que se realizan en poco tiempo de sesion, suelen ser fraudulentas.
    # 3. Las compras con cuentas de pocos dias de uso
    # 4. La diferencia de dias entre las compras suelen ser menor
    # Aca uno se pregunta, si necesita usar las columnas MAS relevantes, o directamente usar todas.
    # Probemos con usar todas a ver como sale *excepto las ultimas, que es lo que quiero estimar*.

    target_columns = cfg.target_column
    excluded_columns = ["flagged_fraud"]

    dataset = load_csv(cfg.data_path, target_column=target_columns,columns_to_ignore=excluded_columns)
    train_dataset, val_dataset, test_dataset = dataset.split(
        train=cfg.split_train,
        val=cfg.split_val,
        test=cfg.split_test,
        seed=cfg.seed,
    )
    if cfg.activation == "identity":
        activation = IdentityActivation()
    elif cfg.activation == "tanh":
        activation = TanhActivation(beta=cfg.beta)
    elif cfg.activation == "step":
        activation = StepActivation()
    else:
        raise ValueError(f"Activation '{cfg.activation}' no soportada en Ejercicio 1.")

    n_inputs = train_dataset.X.shape[1]
    model = MultilayerPerceptron([
        NeuronLayer(n_inputs=n_inputs, n_neurons=1, activation=activation),
    ])

    trainer = Trainer(
        cost_fn=MSECost(),
        optimizer=GradientDescent(learning_rate=cfg.eta),
        metrics=[],
        cfg=cfg,
    )

    Xmin = train_dataset.X.min(axis=0)
    Xmax = train_dataset.X.max(axis=0)
    norm_dataset = normalize_with_params(train_dataset.X, Xmin, Xmax)
    val_dataset.X = normalize_with_params(val_dataset.X, Xmin, Xmax)

    history = trainer.fit(
        model,
        X_train=norm_dataset, zeta_train=train_dataset.zeta,
        X_val=val_dataset.X, zeta_val=val_dataset.zeta,
    )

    layer = model.layers[0]
    print("=== ADALINE Lineal — verificación de arquitectura ===")
    print(f"Peso final:  w={layer.weights[0, 0]:.4f}   (esperado ≈ 2.0)")
    print(f"Bias final:  w₀={layer.bias[0]:.4f}   (esperado ≈ 5.0)")
    print(f"Épocas:      {history['epochs']}")

    if dataset.X.shape[1] == 1:
        plot_regression(
            dataset.X, dataset.zeta, model,
            title="ADALINE Lineal — predicción vs datos",
            output_path="output/experiment/linear/regression.png",
            xlim=(-6, 6),
            ylim=(-8, 18),
        )
    else:
        print("Se omite plot_regression: solo aplica a datasets de 1 feature.")
    plot_error_curve(history, output_path="output/experiment/linear/error_curve.png")
    print("Gráficos guardados en output/experiment/linear/")

    print(f"Error final: {history['train_error'][-1]:.4f}")

